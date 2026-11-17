"""Twenty-person, three-team evolutionary search scored by teamwork frequency."""
from __future__ import annotations

import contextlib
import copy
import html
import itertools
import json
from pathlib import Path
import random
import re
import secrets
import statistics
from urllib.parse import quote

from .protocol import CATEGORIES, TRAITS, PERSONALITY_KEYS, fingerprint, prepare, trait_degrees, write_json


AGENT_ACTION_GUIDANCE = (
    'In this meeting, choose a concrete next action based on the proposal needs, '
    'the information available to you, and your current desires. No participant '
    'has a preset responsibility. State what you say or do, and name the person '
    'or proposal material involved when relevant. Do not assume that teammates '
    'agree or act until you have observed it.'
)

INDIVIDUAL_EVALUATION_CONTEXT = (
    "The team's recommendation will be evaluated as a group result, while each "
    "employee's contribution to the meeting will also affect their individual "
    'performance evaluation.'
)

MEETING_SHARED_MEMORIES = (
    "Three junior employees on a beverage company's product planning team are meeting to discuss a new beer product for adult consumers.",
    'The team has three starting concepts: A, a crisp lower-alcohol beer; B, a beer with a distinctive hop aroma; and C, a beer designed to pair with meals.',
    'The employees will compare the concepts, may combine or revise them or propose another idea, and will decide during the meeting which direction to recommend.',
    INDIVIDUAL_EVALUATION_CONTEXT,
    'The employees decide how to discuss the ideas and organize their work.',
)
MEETING_GOAL = (
    'Compare the three starting beer concepts with the other employees, consider '
    'improvements or other ideas if useful, and decide together during the meeting '
    'which direction to recommend.'
)


def team(members, **origin):
    members = sorted(members)
    if len(members) != 3 or len(set(members)) != 3:
        raise ValueError('チームは異なる3人で構成してください。')
    return {'id': 'team_' + '_'.join(members), 'members': members, 'origin': origin}


def _fixed_duty_memory(memory, role_names):
    return isinstance(memory, str) and any(
        f'{name} is responsible for' in memory for name in role_names)


def _open_meeting_scenario(config):
    scenario = config['scenario']
    scenario['shared_memories'] = list(MEETING_SHARED_MEMORIES)
    scenario['gm_facts_desc'] = 'Shared context for the new beer product idea meeting.'
    defaults = scenario.setdefault('role_defaults', {})
    defaults['goal'] = MEETING_GOAL
    defaults['specific_memories'] = []
    defaults['player_observe'] = (
        '{name} is in the meeting, discussing ideas for a new beer product with two colleagues.'
    )
    defaults['gm_observe'] = '{name} is participating in the new beer product idea meeting.'
    for role in scenario['roles']:
        role['goal'] = MEETING_GOAL
        role['specific_memories'] = []
    scenario['player_observe'] = []
    scenario['gm_observe'] = []
    scenario.pop('scheduled_updates', None)


def prepare_evolution(config, *, pool_seed=None, search_seed=None,
                      simulation_seed=None, generation_updates=10,
                      svo_type='Prosocial',
                      classifier_model_name='gpt-6.1-sol'):
    # Reuse existing ASVO profile/config compatibility checks, but not its trials.
    if not isinstance(generation_updates, int) or generation_updates < 0:
        raise ValueError('generation_updatesは0以上の整数にしてください。')
    if svo_type not in ('Altruistic', 'Prosocial', 'Individualistic', 'Competitive'):
        raise ValueError('svo_typeはASVOの4類型から選んでください。')
    if not isinstance(classifier_model_name, str) or not classifier_model_name.strip():
        raise ValueError('判定用モデル名を指定してください。')
    if pool_seed is None:
        pool_seed = secrets.randbits(32)
    if search_seed is None:
        search_seed = secrets.randbits(32)
    if simulation_seed is None:
        simulation_seed = secrets.randbits(32)
    config = prepare(config)['config']
    config['runtime']['episode_length'] = 3
    _open_meeting_scenario(config)
    config.setdefault('replay', {})['teamwork_classifier_model_name'] = classifier_model_name
    rng = random.Random(pool_seed)
    pool = []
    for index in range(20):
        persona_id = f'P{index+1:02d}'
        # Keep SVO fixed, but give each person stable ASVO personality traits.
        personality = {PERSONALITY_KEYS.get(k, k): rng.randint(3, 9)
                       for k in TRAITS}
        degrees = trait_degrees(personality)
        pool.append({
            'id': persona_id,
            'role': {'name': persona_id, 'gender': 'female',
                     'social_personality': svo_type, 'personality': personality},
            'initial_desires': {d: 5 for d in config['profile']['wanted_desires']},
            'desire_traits': degrees,
        })
    combinations = list(itertools.combinations([p['id'] for p in pool], 3))
    initial = random.Random(search_seed).sample(combinations, 3)
    return {
        'schema_version': 27, 'mode': 'evolution', 'config': config,
        'pool_seed': pool_seed, 'search_seed': search_seed, 'simulation_seed': simulation_seed,
        'personas': pool, 'teams': [team(m, kind='initial_random') for m in initial],
        'population_size': 3, 'team_size': 3, 'generation_updates': generation_updates,
        'planned_simulations': 3 + 2 * generation_updates,
        'possible_teams': len(combinations),
        'targets': list(CATEGORIES), 'selection_rule': 'positive_element_count',
        'teamwork_rubric_version': 7,
        'teamwork_classification_source': 'action',
        'action_parser_version': 2,
        'uniform_svo_type': svo_type,
        'persona_variation': 'personality_only',
        'initial_desire_value': 5,
        'mutation_probability': 0.5,
        'participant_assignment': 'sorted_persona_ids_to_neutral_slots',
        'advice_enabled': False, 'independent_validation': False,
        'note': '欲求はASVO内部の行動生成のみ。選抜・比較指標には使用しない。',
    }


def validate_plan(plan):
    required = {'mode': 'evolution', 'population_size': 3, 'team_size': 3,
                'mutation_probability': 0.5,
                'advice_enabled': False, 'independent_validation': False}
    if any(plan.get(k) != v for k, v in required.items()):
        raise ValueError('20人・3チームの進化計算計画を指定してください。')
    if plan.get('schema_version') != 27:
        raise ValueError('Only evolution plan schema 27 is supported. Create a new plan.')
    if plan.get('selection_rule') != 'positive_element_count' or plan.get('coverage_weight') is not None:
        raise ValueError('The evolution score must count positive teamwork elements.')
    if plan.get('teamwork_rubric_version') != 7:
        raise ValueError('Evolution plans require teamwork rubric 7.')
    expected_personality = {PERSONALITY_KEYS.get(k, k): 5 for k in TRAITS}
    svo_type = plan.get('uniform_svo_type')
    if (svo_type not in ('Altruistic', 'Prosocial', 'Individualistic', 'Competitive')
            or plan.get('persona_variation') != 'personality_only'
            or not isinstance(plan['config'].get('replay', {}).get('teamwork_classifier_model_name'), str)
            or not plan['config']['replay']['teamwork_classifier_model_name'].strip()
            or any(p['role'].get('social_personality') != svo_type
                   or p['role'].get('gender') != 'female'
                   for p in plan['personas'])):
        raise ValueError('SVOを全員同じ類型にしてください。')
    if plan.get('initial_desire_value') != 5 or any(
            p.get('initial_desires') != {d: 5 for d in plan['config']['profile']['wanted_desires']}
            for p in plan['personas']):
        raise ValueError('欲求の初期値は全員・全項目5にしてください。')
    profiles = set()
    for p in plan['personas']:
        personality = p['role'].get('personality')
        if (not isinstance(personality, dict)
                or set(personality) != set(expected_personality)
                or any(type(value) is not int or not 3 <= value <= 9
                       for value in personality.values())):
            raise ValueError('性格特性は8項目・各3〜9の整数にしてください。')
        expected_degrees = trait_degrees(personality)
        if p.get('desire_traits') != expected_degrees:
            raise ValueError('性格値と行動プロファイルの記述が一致しません。')
        profiles.add(tuple(sorted(personality.items())))
    if len(profiles) < 2:
        raise ValueError('候補者ごとに異なる性格特性を生成してください。')
    if plan.get('agent_action_guidance'):
        raise ValueError('標準ASVOの行動生成プロンプトを使用してください。')
    scenario = plan['config']['scenario']
    if (scenario.get('shared_memories') != list(MEETING_SHARED_MEMORIES)
            or scenario.get('role_defaults', {}).get('goal') != MEETING_GOAL
            or plan['config']['runtime'].get('episode_length') != 3
            or 'coverage_weight' in plan
            or any(role.get('goal') != MEETING_GOAL or role.get('specific_memories')
                   for role in scenario['roles'])):
        raise ValueError('会議は商品案の検討と意思決定に絞ってください。')
    if plan.get('action_parser_version') != 2:
        raise ValueError('計画の行動候補解析の版が不正です。')
    if plan.get('teamwork_classification_source') != 'action':
        raise ValueError('新しい計画はactionを判定対象にしてください。')
    role_names = [role['name'] for role in scenario['roles']]
    if (plan.get('participant_assignment') != 'sorted_persona_ids_to_neutral_slots'
            or any(_fixed_duty_memory(memory, role_names)
                   for memory in scenario.get('shared_memories', []))):
        raise ValueError('固定担当のないシナリオを指定してください。')
    if scenario.get('scheduled_updates'):
        raise ValueError('今回の計画では会議途中の情報追加を使用しません。')
    updates = plan.get('generation_updates')
    if not isinstance(updates, int) or updates < 0:
        raise ValueError('generation_updatesは0以上の整数にしてください。')
    if plan.get('planned_simulations') != 3 + 2 * updates:
        raise ValueError('planned_simulationsは3 + 2 * generation_updatesにしてください。')
    ids = [p['id'] for p in plan['personas']]
    if len(ids) != 20 or len(set(ids)) != 20 or plan['targets'] != list(CATEGORIES):
        raise ValueError('候補者または評価指標が不正です。')
    seen = set()
    for candidate in plan['teams']:
        canonical = team(candidate['members'])
        if candidate['id'] != canonical['id'] or not set(candidate['members']) <= set(ids):
            raise ValueError('初期チームが不正です。')
        seen.add(candidate['id'])
    if len(plan['teams']) != 3 or len(seen) != 3:
        raise ValueError('初期チームは異なる3チームが必要です。')


def bind_team(plan, candidate):
    """Assign any three fixed people to job slots and rewrite scenario names."""
    config = copy.deepcopy(plan['config'])
    lookup = {p['id']: p for p in plan['personas']}
    people = [copy.deepcopy(lookup[key]) for key in sorted(candidate['members'])]
    originals = config['scenario']['roles']
    substitutions = {role['name']: p['role']['name'] for role, p in zip(originals, people)}
    pattern = re.compile(r'\b(?:' + '|'.join(re.escape(k) for k in substitutions) + r')\b')

    def replace(value):
        if isinstance(value, str):
            return pattern.sub(lambda match: substitutions[match.group()], value)
        if isinstance(value, list):
            return [replace(v) for v in value]
        if isinstance(value, dict):
            return {k: replace(v) for k, v in value.items()}
        return value

    config['scenario'] = replace(config['scenario'])
    defaults = config['scenario'].get('role_defaults', {})
    for slot, person in enumerate(people):
        role = copy.deepcopy(config['scenario']['roles'][slot])
        role.update(person['role'])
        role['main_character'] = True
        for key in ('goal', 'specific_memories'):
            role.setdefault(key, copy.deepcopy(defaults.get(key, '' if key == 'goal' else [])))
        person['role'] = role
    config['scenario']['roles'] = [p['role'] for p in people]
    return config, people


def teamwork_metrics(document, expected_steps):
    """Strict complete classification measurement, with no access to desire state."""
    frames = document['frames']
    if [f['step'] for f in frames] != list(range(expected_steps + 1)):
        raise ValueError('Replayのステップに欠損・重複があります。')
    names = {a['name'] for a in frames[0]['agents']}
    if len(names) != 3 or len(frames[0]['agents']) != 3:
        raise ValueError('Replayには3人が必要です。')
    counts = dict.fromkeys(CATEGORIES, 0)
    positive_events = 0
    if document.get('teamwork_big_five', {}).get('source') != 'action':
        raise ValueError('actionから判定したReplayが必要です。')
    individual = {name: dict.fromkeys(CATEGORIES, 0) for name in sorted(names)}
    for frame in frames[1:]:
        if len(frame['agents']) != 3 or {a['name'] for a in frame['agents']} != names:
            raise ValueError('Agentイベントに欠損・重複があります。')
        for agent in frame['agents']:
            labels = agent.get('teamwork_big_five') or {}
            vector = labels.get('vector')
            if (not agent.get('action') or labels.get('vector_order') != list(CATEGORIES)
                    or not isinstance(vector, list) or len(vector) != 5
                    or any(type(v) is not int or v not in (0, 1) for v in vector)):
                raise ValueError('チームワーク分類が不完全です。欠損を0点に置換しません。')
            positive_events += int(any(vector))
            for key, value in zip(CATEGORIES, vector):
                counts[key] += value
                individual[agent['name']][key] += value
    events = 3 * expected_steps
    if not events:
        raise ValueError('最低1ステップが必要です。')
    rates = {k: v / events for k, v in counts.items()}
    mean_rate = statistics.mean(rates.values())
    any_event_rate = positive_events / events
    element_coverage = sum(value > 0 for value in counts.values()) / len(CATEGORIES)
    score = sum(counts.values())
    metrics = {'score': score, 'mean_rate': mean_rate,
               'positive_element_count': sum(counts.values()),
               'classification_source': 'action', 'classification_count': events,
               'element_coverage': element_coverage, 'rates': rates, 'counts': counts,
               'classification_coverage': 1.0,
               'individual_rates': {name: {k: v / expected_steps for k, v in values.items()}
                                    for name, values in individual.items()}}
    metrics.update(any_action_rate=any_event_rate,
                   positive_action_count=positive_events, action_count=events)
    return metrics


def rank(population):
    # Earlier evaluations win ties, so the incumbent elite is kept on a tie.
    return sorted(population, key=lambda r: (-r['metrics']['score'], r['evaluation_index']))


def next_generation(population, all_ids, seen, rng, mutation_probability=0.5):
    ranked = rank(population)
    elite = ranked[0]
    children = []
    used = set(seen)
    for first_rank, second_rank in ((1, 2), (1, 3)):
        parent_a, parent_b = ranked[first_rank - 1], ranked[second_rank - 1]
        a, b = set(parent_a['team']['members']), set(parent_b['team']['members'])
        crossovers = [list(c) for c in itertools.combinations(sorted(a | b), 3)
                      if set(c) & (a - b) and set(c) & (b - a)]
        crossover_only = []
        mutated = []
        for cross in crossovers:
            plain = team(cross, kind='crossover',
                         parents=[parent_a['team']['id'], parent_b['team']['id']],
                         parent_ranks=[first_rank, second_rank],
                         crossover=sorted(cross), mutation=None)
            if plain['id'] not in used:
                crossover_only.append(plain)
            for removed in cross:
                for added in sorted(set(all_ids) - (a | b)):
                    members = (set(cross) - {removed}) | {added}
                    if not (members & (a - b) and members & (b - a)):
                        continue
                    child = team(members,
                                 kind='crossover_mutation',
                                 parents=[parent_a['team']['id'], parent_b['team']['id']],
                                 parent_ranks=[first_rank, second_rank],
                                 crossover=sorted(cross), mutation={'removed': removed, 'added': added})
                    if child['id'] not in used:
                        mutated.append(child)
        mutate = rng.random() < mutation_probability
        possibilities = mutated if mutate else crossover_only
        if not possibilities:
            possibilities = crossover_only if mutate else mutated
        if not possibilities:
            raise ValueError('未評価の交叉候補がありません。')
        child = rng.choice(possibilities)
        children.append(child)
        used.add(child['id'])
    return elite, children


def generation_summary(index, population, new_records):
    ranked = rank(population)
    best = ranked[0]
    return {'generation': index, 'population': ranked,
            'best_team': best['team']['id'], 'best_score': best['metrics']['score'],
            'population_mean': statistics.mean(r['metrics']['score'] for r in population),
            'new_mean': statistics.mean(r['metrics']['score'] for r in new_records),
            'new_best': max(r['metrics']['score'] for r in new_records),
            'new_evaluations': len(new_records),
            'best_rates': best['metrics']['rates'],
            'best_mean_rate': best['metrics']['mean_rate'],
            'best_positive_element_count': best['metrics'].get(
                'positive_element_count', sum(best['metrics']['counts'].values())),
            'best_any_event_rate': best['metrics'].get('any_event_rate'),
            'best_any_action_rate': best['metrics'].get('any_action_rate'),
            'best_element_coverage': best['metrics']['element_coverage']}


def evolution_report(generations, plan=None, *, target_generation_updates=None):
    first, last = generations[0], generations[-1]
    updates = len(generations) - 1
    planned_updates = (target_generation_updates if target_generation_updates is not None
                       else plan['generation_updates'] if plan else updates)
    complete = updates == planned_updates
    latest_label = '最終' if complete else '現時点'
    offspring = [record for g in generations[1:] for record in g['population']
                 if record['generation'] == g['generation']]
    mutated_count = sum(record['team']['origin']['mutation'] is not None
                        for record in offspring)
    score_description = ('選抜スコア＝全Agent・全ステップのactionで1と判定された'
                         'チームワーク5要素の合計。3ステップの範囲は0〜45点。'
                         ' 適応性の判定では直前の確定Eventをきっかけの文脈にのみ用いる。')
    lines = ['# チーム編成の進化計算：チームワーク評価', '',
             f'20人から3人を選択。初期3チーム＋{planned_updates}世代×新規2チーム＝{3 + 2 * planned_updates}回を計画。',
             f'進捗: 世代{updates}/{planned_updates}、{3 + 2 * updates}/{3 + 2 * planned_updates}試行。'
             + ('完了。' if complete else '途中経過。'),
             f'突然変異: {mutated_count}/{len(offspring)}子チーム（選択確率50%、候補不足時は方法を切替）。',
             '助言・欲求充足の採点・独立した検証は行いません。', '',
             score_description + ' 達成率ではありません。', '',
             '| 世代 | 新規評価数 | 集団平均スコア | 新規平均スコア | 新規最高スコア | 最高スコア（保持含む） | 最高チームの要素数 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for g in generations:
        lines.append(f'| {g["generation"]} | {g["new_evaluations"]} | '
                     f'{g["population_mean"]:.4f} | {g["new_mean"]:.4f} | '
                     f'{g["new_best"]:.4f} | {g["best_score"]:.4f} | '
                     f'{round(g["best_element_coverage"] * len(CATEGORIES))}/5 |')
    lines += ['', f'初期最高チーム: {first["best_team"]}',
              f'{latest_label}の最高チーム: {last["best_team"]}',
              f'初期最高から{latest_label}までのスコア差: {last["best_score"] - first["best_score"]:+.4f}',
              f'初期最高の陽性要素数: {first["best_positive_element_count"]}、要素カバー率: {first["best_element_coverage"]:.0%}',
              f'{latest_label}の最高チームの陽性要素数: {last["best_positive_element_count"]}、要素カバー率: {last["best_element_coverage"]:.0%}',
              '', f'| 要素 | 初期最高チーム | {latest_label}の最高チーム | 差（ポイント） |',
              '|---|---:|---:|---:|']
    for key in CATEGORIES:
        before, after = first['best_rates'][key], last['best_rates'][key]
        lines.append(f'| {key} | {before:.1%} | {after:.1%} | {(after-before)*100:+.2f} |')
    lines += ['', '## 世代別の編成と系譜', '']
    for g in generations:
        lines += [f'### 世代{g["generation"]}', '',
                  f'| 順位 | メンバー | 選抜スコア（無単位） | 陽性要素数 | 観測要素数 | 由来 |',
                  '|---|---|---:|---:|---:|---|']
        for position, record in enumerate(g['population'], 1):
            candidate = record['team']
            kind = 'elite（再評価なし）' if record['generation'] < g['generation'] else candidate['origin']['kind']
            metrics = record['metrics']
            lines.append(f'| {position} | {", ".join(candidate["members"])} | '
                         f'{metrics["score"]:.4f} | {metrics["positive_element_count"]} | '
                         f'{round(metrics["element_coverage"] * len(CATEGORIES))}/5 | {kind} |')
        lines.append('')
    lines += [score_description,
              'エリート保持により最高値は下がりません。新規チームの平均・最高値も併記しています。',
              'これは探索中の評価値の変化であり、独立試行での再現性・統計的な効果は未確認です。',
              '担当は事前に固定せず、チーム内で決めます。',
              '外部LLMの応答は同じseedでも一致するとは限りません。']
    return '\n'.join(lines) + '\n'


def run_evolution(plan, output, simulate, *, through_generation=None):
    validate_plan(plan)
    if (through_generation is not None
            and (type(through_generation) is not int or through_generation < 0
                 or 3 + 2 * through_generation > plan['possible_teams'])):
        raise ValueError('through_generationは未評価チームを作れる範囲の0以上の整数にしてください。')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    lock = output / 'experiment.lock'
    try:
        descriptor = lock.open('x')
    except FileExistsError as exc:
        raise RuntimeError('同じ出力先で実験が実行中です。') from exc
    with descriptor:
        try:
            return _run(plan, output, simulate, through_generation)
        finally:
            lock.unlink(missing_ok=True)


def summarize_usage(path):
    """Sum token usage returned by all API calls in one trial."""
    totals = {'api_calls': 0, 'input_tokens': 0, 'cached_input_tokens': 0,
              'output_tokens': 0, 'reasoning_output_tokens': 0, 'total_tokens': 0}
    if not path.exists():
        return None
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        totals['api_calls'] += 1
        for key in totals.keys() - {'api_calls'}:
            totals[key] += item.get(key, 0)
    return totals if totals['api_calls'] else None


def write_overview_html(output, plan, records, target_generation_updates, *, complete=False):
    """Write an atomic, self-contained index after each completed trial."""
    output = Path(output)
    planned = 3 + 2 * target_generation_updates
    categories = {
        'team_leadership': 'リーダーシップ',
        'mutual_performance_monitoring': '相互モニタリング',
        'backup_behavior': '支援行動',
        'adaptability': '適応性',
        'team_orientation': 'チーム志向',
    }
    score_note = '陽性判定された5要素の合計。1行動の複数要素や別行動での再出現も数えます。'
    best = max(records, key=lambda row: row['metrics']['score']) if records else None
    best_text = (f'{best["metrics"]["score"]:g}点 · '
                 f'{html.escape(best["team"]["id"])}' if best else '—')
    rows = []
    for index, record in enumerate(records, 1):
        candidate = record['team']
        metrics = record['metrics']
        team_id = candidate['id']
        replay_base = 'trials/' + quote(team_id, safe='') + '/'
        links = (f'<a href="{replay_base}replay.html">Replay</a> '
                 f'<a href="{replay_base}replay.json">JSON</a>')
        counts = ''.join(f'<td>{metrics["counts"][key]}</td>' for key in categories)
        score = f'{metrics["score"]:g}'
        usage = record.get('usage') or {}
        calls = usage.get('api_calls', '—')
        rows.append(
            f'<tr><td>{index}</td><td>{record["generation"]}</td>'
            f'<td><strong>{html.escape(", ".join(candidate["members"]))}</strong></td>'
            f'<td class="score">{score}</td>{counts}<td>{calls}</td><td>{links}</td></tr>'
        )
    status = '完了' if complete else '途中経過'
    page = f'''<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>進化実験の結果一覧</title>
<style>
:root {{ color-scheme: light; font-family: system-ui, sans-serif; color: #17233b; background: #f5f7fb; }}
body {{ max-width: 1200px; margin: 0 auto; padding: 32px 20px 64px; }}
h1 {{ margin-bottom: .25rem; }} .muted {{ color: #596779; }}
.cards {{ display: flex; gap: 12px; flex-wrap: wrap; margin: 24px 0; }}
.card {{ background: white; border: 1px solid #dce3ed; border-radius: 12px; padding: 16px 20px; min-width: 170px; }}
.card span {{ display: block; color: #596779; font-size: .9rem; }} .card strong {{ font-size: 1.35rem; }}
progress {{ width: 100%; height: 14px; accent-color: #167d8d; }}
.table-wrap {{ overflow-x: auto; background: white; border: 1px solid #dce3ed; border-radius: 12px; }}
table {{ width: 100%; border-collapse: collapse; white-space: nowrap; }}
th, td {{ text-align: left; padding: 11px 12px; border-bottom: 1px solid #edf0f4; }}
th {{ background: #edf3f9; font-size: .85rem; }} tr:last-child td {{ border-bottom: 0; }}
tbody tr:hover {{ background: #f6fbfc; }} .score {{ font-weight: 700; font-variant-numeric: tabular-nums; }}
a {{ color: #075e82; }} a + a {{ margin-left: 9px; }}
</style></head><body>
<h1>進化実験の結果一覧</h1>
<p class="muted">{status} · 3人チーム · {plan['config']['runtime']['episode_length']}ステップ · {html.escape(score_note)}</p>
<div class="cards"><div class="card"><span>完了した試行</span><strong>{len(records)} / {planned}</strong></div>
<div class="card"><span>現在の最高スコア</span><strong>{best_text}</strong></div>
<div class="card"><span>到達世代</span><strong>{max((r['generation'] for r in records), default=0)} / {target_generation_updates}</strong></div></div>
<progress value="{len(records)}" max="{planned}"></progress>
<p class="muted">各要素の数値は、その要素が1と判定された回数です。スコアは達成率ではありません。</p>
<div class="table-wrap"><table><thead><tr><th>試行</th><th>世代</th><th>メンバー</th><th>スコア</th>
{''.join(f'<th>{label}</th>' for label in categories.values())}<th>API呼び出し</th><th>記録</th></tr></thead>
<tbody>{''.join(rows) if rows else '<tr><td colspan="11">完了した試行はありません。</td></tr>'}</tbody></table></div>
<p class="muted">このページは試行完了時に更新されます。最新の状態を見るには再読み込みしてください。</p>
</body></html>'''
    temporary = output / 'index.html.tmp'
    temporary.write_text(page, encoding='utf-8')
    temporary.replace(output / 'index.html')


def rebuild_overview_html(plan, output):
    """Rebuild the index from saved results, including interrupted initial trials."""
    validate_plan(plan)
    output = Path(output)
    if not output.is_dir():
        raise ValueError('結果フォルダが見つかりません。')
    saved_plan = output / 'plan.json'
    if saved_plan.exists() and fingerprint(json.loads(saved_plan.read_text())) != fingerprint(plan):
        raise ValueError('保存済み計画と異なります。')
    saved_results = output / 'results.json'
    if saved_results.exists():
        result = json.loads(saved_results.read_text(encoding='utf-8'))
        records = list(result['trials'])
        target = result.get('planned_generation_updates', plan['generation_updates'])
        complete = result.get('complete', False)
    else:
        records = []
        target = plan['generation_updates']
        complete = False
    known = {record['team']['id'] for record in records}
    saved_trials = sorted((output / 'trials').glob('*/trial.json'),
                          key=lambda path: path.stat().st_mtime)
    for path in saved_trials:
        if path.parent.name in known:
            continue
        saved = json.loads(path.read_text(encoding='utf-8'))
        identity = saved['identity']
        replay = json.loads((path.parent / 'replay.json').read_text(encoding='utf-8'))
        if (identity['plan_hash'] != fingerprint(plan)
                or fingerprint(replay) != saved['replay_hash']):
            raise ValueError(f'保存済み試行が計画と一致しません: {path.parent.name}')
        metrics = teamwork_metrics(replay, plan['config']['runtime']['episode_length'])
        usage_path = path.parent / 'usage.json'
        usage = json.loads(usage_path.read_text(encoding='utf-8')) if usage_path.exists() else None
        records.append({'team': identity['team'], 'generation': identity['generation'],
                        'evaluation_index': len(records), 'metrics': metrics,
                        'usage': usage, 'replay': str(path.parent / 'replay.html')})
        known.add(identity['team']['id'])
    target = max(target, max((record['generation'] for record in records), default=0))
    complete = complete and len(records) == 3 + 2 * target
    write_overview_html(output, plan, records, target, complete=complete)
    return len(records)


def _run(plan, output, simulate, through_generation):
    signature = fingerprint(plan)
    plan_path = output / 'plan.json'
    if plan_path.exists() and fingerprint(json.loads(plan_path.read_text())) != signature:
        raise ValueError('保存済み計画と異なります。別の出力先を指定してください。')
    generations_path = output / 'generations.json'
    previous_completed = (json.loads(generations_path.read_text())[-1]['generation']
                          if generations_path.exists() else -1)
    result_path = output / 'results.json'
    previous_target = (json.loads(result_path.read_text()).get('planned_generation_updates',
                       plan['generation_updates']) if result_path.exists()
                       else plan['generation_updates'])
    if through_generation is None:
        through_generation = max(plan['generation_updates'], previous_target,
                                 previous_completed)
    target_generation_updates = max(plan['generation_updates'], previous_target,
                                    previous_completed, through_generation)
    if previous_completed > through_generation:
        raise ValueError(f'世代{previous_completed}まで完了済みです。停止世代をそれ以上にしてください。')
    write_json(plan_path, plan)
    records = []

    def evaluate(candidate, generation):
        identity = {'plan_hash': signature, 'team': candidate,
                    'generation': generation, 'seed': plan['simulation_seed']}
        destination = output / 'trials' / candidate['id']
        saved_path = destination / 'trial.json'
        if saved_path.exists():
            saved = json.loads(saved_path.read_text())
            replay = json.loads((destination / 'replay.json').read_text())
            replay_config = plan['config'].get('replay', {})
            translation_requested = (
                replay_config.get('translate_actions_ja', False)
                or replay_config.get('translate_events_ja', False)
                or replay_config.get('summarize_steps_ja', False)
            )
            has_translated_fields = all(
                not agent.get('action') or agent.get('action_ja')
                for frame in replay.get('frames', [])
                for agent in frame.get('agents', [])
            ) and all(
                not agent.get('event') or agent.get('event_ja')
                for frame in replay.get('frames', [])
                for agent in frame.get('agents', [])
            )
            if saved['identity'] != identity or fingerprint(replay) != saved['replay_hash']:
                raise ValueError('保存済み試行の設定またはReplayが変更されています。')
            if not translation_requested or has_translated_fields:
                metrics = teamwork_metrics(replay, plan['config']['runtime']['episode_length'])
            else:
                saved_path.unlink()
                replay = None
        else:
            replay = None
        if replay is None:
            destination.mkdir(parents=True, exist_ok=True)
            runtime_plan = copy.deepcopy(plan)
            runtime_plan['teams'] = [candidate]
            print(f'世代{generation}: {candidate["id"]} '
                  f'({len(records)+1}/{3 + 2 * target_generation_updates})', flush=True)
            usage_path = destination / 'usage.jsonl'
            models = [getattr(simulate, 'keywords', {}).get(key)
                      for key in ('model', 'classifier_model')]
            models = [model for index, model in enumerate(models)
                      if model is not None and model not in models[:index]
                      and hasattr(model, 'set_usage_log_path')]
            if models:
                usage_path.write_text('', encoding='utf-8')
                for model in models:
                    model.set_usage_log_path(usage_path)
            try:
                with (destination / 'simulation.log').open('w') as log:
                    with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                        replay = simulate(runtime_plan, candidate['id'], plan['simulation_seed'], None, destination)
            finally:
                for model in models:
                    model.set_usage_log_path(None)
            usage = summarize_usage(usage_path)
            if usage is not None:
                write_json(destination / 'usage.json', usage)
            write_json(destination / 'replay.json', replay)
            metrics = teamwork_metrics(replay, plan['config']['runtime']['episode_length'])
            write_json(saved_path, {'identity': identity, 'replay_hash': fingerprint(replay)})
        usage_path = destination / 'usage.json'
        usage = json.loads(usage_path.read_text(encoding='utf-8')) if usage_path.exists() else None
        record = {'team': candidate, 'generation': generation, 'evaluation_index': len(records),
                  'metrics': metrics, 'usage': usage, 'replay': str(destination / 'replay.html')}
        records.append(record)
        write_overview_html(output, plan, records, target_generation_updates)
        return record

    def save_progress(generations):
        if generations[-1]['generation'] < previous_completed:
            return
        completed = generations[-1]['generation'] == target_generation_updates
        measured = [r['usage'] for r in records if r['usage'] is not None]
        usage_total = ({key: sum(u[key] for u in measured) for key in measured[0]}
                       if measured else None)
        delta = generations[-1]['best_score'] - generations[0]['best_score']
        write_json(output / 'generations.json', generations)
        write_json(output / 'results.json', {
            'complete': completed, 'completed_generation': generations[-1]['generation'],
            'planned_generation_updates': target_generation_updates,
            'completed_simulations': len(records),
            'planned_simulations': 3 + 2 * target_generation_updates,
            'trials': records, 'generations': generations,
            'usage': usage_total, 'usage_measured_trials': len(measured),
            'initial_to_current_best_delta': delta,
            'initial_to_final_best_delta': delta if completed else None})
        (output / 'report.md').write_text(
            evolution_report(generations, plan,
                             target_generation_updates=target_generation_updates),
            encoding='utf-8')
        write_overview_html(output, plan, records, target_generation_updates,
                            complete=completed)

    population = [evaluate(candidate, 0) for candidate in plan['teams']]
    generations = [generation_summary(0, population, population)]
    save_progress(generations)
    ids = [p['id'] for p in plan['personas']]
    for index in range(1, through_generation + 1):
        elite, candidates = next_generation(population, ids,
            {r['team']['id'] for r in records}, random.Random(plan['search_seed'] + index),
            plan['mutation_probability'])
        new_records = [evaluate(candidate, index) for candidate in candidates]
        population = [elite, *new_records]
        generations.append(generation_summary(index, population, new_records))
        save_progress(generations)
    if len(records) != 3 + 2 * through_generation:
        raise AssertionError('Simulation budget mismatch')
    return records
