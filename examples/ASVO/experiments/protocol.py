"""Experiment planning and measurement; standard library only, no API calls."""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import statistics

CATEGORIES = (
    'team_leadership', 'mutual_performance_monitoring', 'backup_behavior',
    'adaptability', 'team_orientation',
)
TARGETS = ('mutual_performance_monitoring', 'backup_behavior', 'team_orientation')
ARMS = ('baseline', 'advice', 'reteam', 'reteam_advice')
TRAITS = {
    'competitive': 'sense of superiority', 'ambitious': 'sense_of_achievement',
    'confident': 'confidence', 'cheerful': 'joyfulness', 'relaxed': 'comfort',
    'attention_seeking': 'recognition', 'possessive': 'sense of control',
    'spiritual': 'spiritual satisfaction',
}
PERSONALITY_KEYS = {'confident': 'confidence'}


def trait_degrees(personality):
    """Describe the eight ASVO traits using the same scale as the desire targets."""
    degrees = {}
    for adjective in TRAITS:
        value = personality[PERSONALITY_KEYS.get(adjective, adjective)]
        if value <= 3:
            degree = 'slightly'
        elif value <= 5:
            degree = 'moderately'
        elif value <= 7:
            degree = 'quite'
        else:
            degree = 'extremely'
        degrees[adjective] = degree
    return degrees


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def prepare(config, *, pool_seed=20261031, discovery_seeds=(101,),
            evaluation_seeds=(201, 202)):
    """Freeze six personas (two per job), eight teams and disjoint seeds."""
    if set(discovery_seeds) & set(evaluation_seeds):
        raise ValueError('探索用seedと評価用seedは分離してください。')
    if not discovery_seeds or not evaluation_seeds:
        raise ValueError('探索用・評価用seedは最低1件ずつ必要です。')
    if len(set(discovery_seeds)) != len(discovery_seeds) or len(set(evaluation_seeds)) != len(evaluation_seeds):
        raise ValueError('seedを重複させないでください。')
    config = copy.deepcopy(config)
    config.pop('_config_path', None)
    roles = config['scenario']['roles']
    if len(roles) != 3 or config['runtime']['num_players'] != 3:
        raise ValueError('このPoCは3職務・3人の会議用です。')
    if config['profile'].get('hidden_desires') or config['profile'].get('use_previous'):
        raise ValueError('PoCではhidden_desires/use_previousを無効にしてください。')
    if set(config['profile']['wanted_desires']) != set(TRAITS.values()):
        raise ValueError('PoCは現在の8欲求を対象とします。')
    if config['model'].get('api_key'):
        raise ValueError('キーは設定ファイルでなく環境変数に設定してください。')
    rng = random.Random(pool_seed)
    distribution = config['profile']['initial_value_distribution']
    pool, groups = [], []
    defaults = config['scenario'].get('role_defaults', {})
    for slot, original in enumerate(roles):
        group = []
        for variant in range(2):
            role = copy.deepcopy(original)
            for key in ('goal', 'specific_memories'):
                role.setdefault(key, copy.deepcopy(defaults.get(key, '' if key == 'goal' else [])))
            if variant:
                role['social_personality'] = ('Competitive', 'Prosocial', 'Individualistic')[slot]
                role['personality'] = {k: rng.randint(3, 9) for k in role['personality']}
            persona_id = f'role{slot + 1}_person{variant + 1}'
            initial = {d: max(distribution['min'], min(distribution['max'],
                       round(rng.gauss(distribution['mean'], distribution['stddev']))))
                       for d in config['profile']['wanted_desires']}
            # Desire traits and numerical targets use one consistent profile.
            degrees = trait_degrees(role["personality"])
            pool.append({'id': persona_id, 'slot': slot, 'role': role,
                         'initial_desires': initial, 'desire_traits': degrees,
                         'big_five': {k: rng.randint(1, 10) for k in (
                             'extraversion', 'neuroticism', 'openness',
                             'conscientiousness', 'agreeableness')}})
            group.append(persona_id)
        groups.append(group)
    teams = [{'id': f'team{i:02d}', 'members': list(members)}
             for i, members in enumerate(itertools.product(*groups))]
    return {'schema_version': 1, 'config': config, 'pool_seed': pool_seed,
            'personas': pool, 'teams': teams, 'baseline_team': teams[0]['id'],
            'discovery_seeds': list(discovery_seeds),
            'evaluation_seeds': list(evaluation_seeds), 'targets': list(TARGETS),
            'selection_rule': 'mean_target_rate + mean_gap_reduction / 10',
            'planned_simulations': len(teams) * len(discovery_seeds) + 4 * len(evaluation_seeds),
            'note': '人物名は職務スロットの表示名。人物の同一性はpersona_idで追跡。'}


def _mean(values):
    return statistics.mean(values) if values else None


def measure(document, targets=TARGETS):
    """Count per-event rates, missing labels and initial-to-final deficits."""
    frames = document['frames']
    if not frames:
        raise ValueError('Replay has no frames')
    counts = dict.fromkeys(CATEGORIES, 0)
    expected_events = valid = 0
    for frame in frames:
        if frame['step'] == 0:
            continue
        for agent in frame['agents']:
            expected_events += 1
            classification = agent.get('teamwork_big_five') or {}
            vector = classification.get('vector')
            if (not agent.get('event') or not isinstance(vector, list) or len(vector) != 5
                    or any(type(v) is not int or v not in (0, 1) for v in vector)):
                continue
            valid += 1
            for k, v in zip(CATEGORIES, vector):
                counts[k] += v
    rates = {k: v / valid if valid else None for k, v in counts.items()}
    first = {a['name']: a for a in frames[0]['agents']}
    individual = []
    for agent in frames[-1]['agents']:
        initial, final = [], []
        old = first[agent['name']]['desires']
        for key, state in agent['desires'].items():
            target, value = state.get('target'), state.get('value')
            previous = old.get(key, {})
            if any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in (
                    target, value, previous.get('value'), previous.get('target'))):
                raise ValueError('Missing/non-finite desire measurement')
            if target != previous['target']:
                raise ValueError('欲求の目標値が途中で変更されています。')
            initial.append(max(0, target - previous['value']))
            final.append(max(0, target - value))
        individual.append({'name': agent['name'], 'persona_id': agent.get('persona_id'),
                           'initial_gap': _mean(initial), 'final_gap': _mean(final),
                           'gap_reduction': _mean(initial) - _mean(final),
                           'unmet_desires': sum(v > 0 for v in final)})
    return {'counts': counts, 'rates': rates,
            'target_rate': _mean([rates[k] for k in targets]) if valid else None,
            'valid_events': valid, 'expected_events': expected_events,
            'classification_coverage': valid / expected_events if expected_events else 0,
            'final_state_evaluated': document.get('final_state_evaluated', False),
            'mean_final_gap': _mean([a['final_gap'] for a in individual]),
            'mean_gap_reduction': _mean([a['gap_reduction'] for a in individual]),
            'worst_final_gap': max(a['final_gap'] for a in individual),
            'individuals': individual}


def selection_score(metrics):
    if metrics['classification_coverage'] != 1 or not metrics['final_state_evaluated']:
        raise ValueError('分類欠損または最終状態未評価の試行はチーム選抜に使えません。')
    return metrics['target_rate'] + metrics['mean_gap_reduction'] / 10


def select_team(plan, records):
    scored = []
    for team in plan['teams']:
        trials = [r for r in records if r['team'] == team['id']]
        if sorted(r['seed'] for r in trials) != sorted(plan['discovery_seeds']):
            raise ValueError('探索試行が不足・重複しています。')
        scored.append((statistics.mean(selection_score(r['metrics']) for r in trials), team['id']))
    # Stable tie break; the baseline is retained when it ties for best.
    return sorted(scored, key=lambda pair: (-pair[0], pair[1]))[0][1]


def report(plan, selected, results):
    lines = ['# HumanLinkAI 協働改善PoC：評価結果', '',
             f'選抜チーム: {selected} / 元チーム: {plan["baseline_team"]}',
             '探索用データと評価用データを分離。数値はLLMシミュレーション内の結果です。', '',
             '| 条件 | 試行数 | 対象行動の出現率 | 欲求不足の平均 | 初期からの不足軽減 | 最も不足した人の不足 |',
             '|---|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        rows = [r['metrics'] for r in results if r['arm'] == arm]
        if not rows:
            continue
        means = [statistics.mean(row[k] for row in rows) for k in (
            'target_rate', 'mean_final_gap', 'mean_gap_reduction', 'worst_final_gap')]
        lines.append(f'| {arm} | {len(rows)} | {means[0]:.1%} | {means[1]:.3f} | {means[2]:.3f} | {means[3]:.3f} |')
    lines += ['', '## 同じseedでの介入なしとの差', '',
              '| 条件 | 対象行動の差（ポイント） | 不足軽減量の差 | 行動・不足の両方が改善した試行 |',
              '|---|---:|---:|---:|']
    baseline = {r['seed']: r['metrics'] for r in results if r['arm'] == 'baseline'}
    for arm in ARMS[1:]:
        deltas = [(r['metrics']['target_rate'] - baseline[r['seed']]['target_rate'],
                   r['metrics']['mean_gap_reduction'] - baseline[r['seed']]['mean_gap_reduction'])
                  for r in results if r['arm'] == arm]
        if deltas:
            lines.append(f'| {arm} | {statistics.mean(d[0] for d in deltas)*100:+.2f} | '
                         f'{statistics.mean(d[1] for d in deltas):+.3f} | '
                         f'{sum(a > 0 and b > 0 for a,b in deltas)}/{len(deltas)} |')
    lines += ['', 'チーム再編では人物が変わるため、個人の前後比較とは解釈しません。',
              '行動出現率＝該当イベント数÷分類済みイベント数。5要素の合計は100%に限定されません。',
              '欲求不足＝max(0, 固定目標値−欲求値)の8項目平均（0〜10）。軽減量は初期値−最終値。',
              '個人別・5要素別・seed別の詳細はresults.jsonを参照してください。',
              '少数試行のPoCです。実際の人間の協働改善や、統計的に確立した効果を意味しません。',
              '成果物の品質は自動採点していません。リプレイで提案内容と制約充足を別途確認してください。']
    return '\n'.join(lines) + '\n'
