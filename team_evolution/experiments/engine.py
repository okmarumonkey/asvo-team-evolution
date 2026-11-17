"""ASVO meeting runner independent of the interactive notebook and its API key."""
from __future__ import annotations

import copy
import datetime as dt
import json
import os
from pathlib import Path
import random
import sys

from .protocol import TRAITS


def persona_profile(persona, action_guidance=None):
    profile = '\n'.join('{agent_name} is ' + degree + ' ' + adjective
                        for adjective, degree in persona['desire_traits'].items())
    if action_guidance:
        profile += '\n' + action_guidance
    return profile


def check_runtime(config):
    """Check credentials before constructing a client or loading embeddings."""
    if config['model'].get('disable_language_model'):
        raise ValueError('実験には実際の言語モデルが必要です。')
    if config['model']['api_type'] != 'openai':
        raise ValueError('この実験ランナーの初版はopenaiのみ対応します。')
    variable = config['model'].get('api_key_env') or 'OPENAI_API_KEY'
    if not os.environ.get(variable):
        raise RuntimeError(f'{variable} が未設定です。キーを環境変数に設定してください。')


def load_resources(config):
    import openai
    import sentence_transformers
    from concordia.language_model.base_gpt_model import BaseGPTModel
    check_runtime(config)
    client = openai.OpenAI(api_key=os.environ[config['model'].get('api_key_env') or 'OPENAI_API_KEY'])
    model = BaseGPTModel(model_name=config['model']['model_name'], client=client)
    classifier_name = config.get('replay', {}).get('teamwork_classifier_model_name')
    classifier_model = (BaseGPTModel(model_name=classifier_name, client=client)
                        if classifier_name else model)
    encoder = sentence_transformers.SentenceTransformer(config['embedding']['model_name'])
    return model, classifier_model, lambda text: encoder.encode(text, show_progress_bar=False)


def finish_pending_state(player):
    """Run only ASVO's pre-act evaluation, never choose or execute another action.

    This adapter is deliberately isolated because Concordia does not expose a
    public evaluate-only phase. POST_ACT hooks are skipped (no fictitious action
    in memory), but the legal phase transition and update hooks are preserved.
    """
    from concordia.typing import entity, entity_component
    phase = entity_component.Phase
    with player._control_lock:
        player._set_phase(phase.PRE_ACT)
        try:
            player._parallel_call_('pre_act', entity.DEFAULT_ACTION_SPEC)
        finally:
            player._set_phase(phase.POST_ACT)
            player._set_phase(phase.UPDATE)
            player._parallel_call_('update')
            player._set_phase(phase.READY)


def deliver_scheduled_updates(config, step, players, gm_memory):
    """Give every participant and the game master the same new information."""
    messages = []
    for update in config['scenario'].get('scheduled_updates', []):
        if update['before_step'] != step:
            continue
        message = update['text']
        gm_memory.add(message, tags=['scenario_update'])
        for player in players:
            player.observe(message)
        messages.append(message)
    return messages


def simulate(plan, team_id, seed, advice, output, model, embedder, classifier_model=None):
    """Run the fixed meeting and return a replay with evaluated final desires."""
    import numpy as np
    from concordia.associative_memory import associative_memory, formative_memories, importance_function
    from concordia.clocks import game_clock
    from concordia import components as generic_components
    from concordia.components import game_master as gm_components
    from concordia.environment import game_master
    from examples.ASVO.ASVO_agent.ValueAgent import build_ASVO_agent
    from team_evolution.config_loader import role_expected_values, legacy_scenario
    from team_evolution.replay import (
        ReplayRecorder,
        LLMJapaneseActionTranslator,
        LLMJapaneseReplayFieldTranslator,
        LLMJapaneseResultSummarizer,
        LLMJapaneseSharedMemoriesSummarizer,
        LLMJapaneseStepSummarizer,
        LLMTeamworkClassifier,
        render_replay_html,
    )
    from team_evolution.replay.exporter import _complete_agent_transition
    from examples.ASVO.value_components.init_value_info_social import values_dict

    config = copy.deepcopy(plan['config'])
    if classifier_model is None:
        classifier_model = model
    random.seed(seed)
    np.random.seed(seed)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    team = next(t for t in plan['teams'] if t['id'] == team_id)
    pool = {p['id']: p for p in plan['personas']}
    personas = [pool[key] for key in team['members']]
    is_evolution = plan.get('mode') == 'evolution'
    if is_evolution:
        if advice is not None:
            raise ValueError('進化計算では助言を使用しません。')
        from .evolution import bind_team
        config, personas = bind_team(plan, team)
    config['scenario']['roles'] = [p['role'] for p in personas]
    scenario = legacy_scenario(config)
    clock_config = config['clock']
    clock = game_clock.MultiIntervalClock(
        start=dt.datetime.fromisoformat(clock_config['setup_time']),
        step_sizes=[dt.timedelta(minutes=clock_config.get('major_step_minutes', 20)),
                    dt.timedelta(seconds=clock_config.get('minor_step_seconds', 10))])
    # No LLM-generated background summary: identical world context in all arms.
    background = '\n'.join(scenario['shared_memories'])
    importance = importance_function.AgentImportanceModel(model).importance
    constant_importance = importance_function.ConstantImportanceModel().importance
    players = []
    names = [p['role']['name'] for p in personas]
    for persona in personas:
        role = persona['role']
        memory = associative_memory.AssociativeMemory(embedder, importance, clock=clock.now)
        for text in [*scenario['shared_memories'], role['goal'], *role['specific_memories']]:
            memory.add(text)
        agent_config = formative_memories.AgentConfig(
            name=role['name'], gender=role.get('gender', ''), goal=role['goal'],
            context=background, traits='',
            extras={'main_character': True, 'specific_memories': role['specific_memories']})
        profile = persona_profile(persona, plan.get('agent_action_guidance') if is_evolution else None)
        if advice:
            messages = advice[persona['id']]
            coaching = '\n'.join(item['advice'] for item in messages)
            profile += ('\nPrivate HumanLinkAI recommendations for this meeting. '
                        'Consider these concrete actions while preserving your values and goals:\n'
                        + coaching.replace('{', '{{').replace('}', '}}'))
        player = build_ASVO_agent(
            config=agent_config, context_dict=persona['desire_traits'],
            selected_desire={d: {'adj': next(a for a,v in TRAITS.items() if v == d),
                                  'description': values_dict[d]}
                             for d in config['profile']['wanted_desires']},
            predefined_setting=persona['initial_desires'], model=model, profile=profile,
            memory=memory, background_knowledge=background, clock=clock,
            update_time_interval=None, stored_target_folder=str(output / 'agents'),
            social_personality=role['social_personality'], agent_names=names,
            current_time=f'seed{seed}', expected_value_setting=role_expected_values(role),
            flexible_action_parsing=plan.get('action_parser_version', 1) >= 2)
        players.append(player)
    gm_memory = associative_memory.AssociativeMemory(embedder, constant_importance, clock=clock.now)
    for persona in personas:
        for text in persona['role']['specific_memories']:
            gm_memory.add(text, tags=['initial_player_specific_memory'])
    status = gm_components.player_status.PlayerStatus(clock.now, model, gm_memory, names)
    direct = gm_components.direct_effect.DirectEffect(
        players, model=model, memory=gm_memory, clock_now=clock.now,
        verbose=False, components=[status])
    env = game_master.GameMaster(
        model=model, memory=gm_memory, clock=clock, players=players,
        components=[generic_components.constant.ConstantComponent(background, scenario['gm_facts_desc']),
                    status, direct,
                    gm_components.relevant_events.RelevantEvents(clock.now, model, gm_memory),
                    gm_components.time_display.TimeDisplay(clock)],
        randomise_initiative=True, player_observes_event=False,
        verbose=False, concurrent_externalities=False, seed=seed)
    clock.set(dt.datetime.fromisoformat(clock_config['start_time']))
    for index, player in enumerate(players):
        pronoun = 'she' if personas[index]['role'].get('gender') == 'female' else 'he'
        if index in scenario['player observe']:
            player.observe(scenario['player observe'][index].format(name=player.name, p=pronoun))
        if index in scenario['gm observe']:
            gm_memory.add(scenario['gm observe'][index].format(name=player.name))
    replay_config = config.get('replay', {})
    action_translator = (
        LLMJapaneseActionTranslator(model)
        if replay_config.get('translate_actions_ja', False) else None
    )
    field_translator = (
        LLMJapaneseReplayFieldTranslator(model)
        if replay_config.get('translate_events_ja', False) else None
    )
    step_summarizer = (
        LLMJapaneseStepSummarizer(model)
        if replay_config.get('summarize_steps_ja', False) else None
    )
    initial_context_summarizer = (
        LLMJapaneseSharedMemoriesSummarizer(model)
        if replay_config.get('summarize_shared_memories_ja', False) else None
    )
    result_summarizer = (
        LLMJapaneseResultSummarizer(model)
        if replay_config.get('summarize_result_ja', False) else None
    )
    # The plan fixes whether classification measures chosen actions or GM events.
    recorder = ReplayRecorder(output / 'replay.json', title='HumanLinkAI PoC2',
                              action_translator=action_translator,
                              field_translator=field_translator,
                              summarizer=step_summarizer,
                              teamwork_classifier=LLMTeamworkClassifier(
                                  classifier_model, rubric_version=(
                                      plan.get('teamwork_rubric_version', 1)
                                      if plan.get('mode') == 'evolution' else 1)),
                              teamwork_source=plan.get('teamwork_classification_source', 'event'),
                              result_summarizer=result_summarizer,
                              shared_memories=scenario['shared_memories'],
                              initial_context_summarizer=initial_context_summarizer)
    if config['scenario'].get('scheduled_updates'):
        recorder.document['scenario_updates'] = []
    recorder.capture(step=0, time=clock.now(), players=players)
    for step in range(1, config['runtime']['episode_length'] + 1):
        messages = deliver_scheduled_updates(config, step, players, gm_memory)
        for message in messages:
            recorder.document['scenario_updates'].append(
                {'before_step': step, 'time': clock.now().isoformat(), 'text': message})
        env.step()
        events = {item.get('Active player', {}).get('Name'): item.get('Event statement', '')
                  for item in env.get_history()[-len(players):]}
        recorder.capture(step=step, time=clock.now(), players=players,
                         event='\n'.join(events.values()), events_by_agent=events)
    # Teamwork-only search does not spend calls on an extra desire evaluation.
    pending_players = [] if is_evolution else players
    for player, final in zip(pending_players, recorder.document['frames'][-1]['agents']):
        finish_pending_state(player)
        tracker = player.get_component('ValueTracker')
        values = tracker.get_individual_desire_tracker()
        last_index = max(values, key=int)
        _complete_agent_transition(final, values[last_index],
                                   tracker.get_desire_update_tracker().get(last_index, {}))
    for frame in recorder.document['frames']:
        for agent, persona in zip(frame['agents'], personas):
            if agent['name'] != persona['role']['name']:
                raise ValueError('Unexpected agent ordering')
            agent['persona_id'] = persona['id']
    recorder.document['final_state_evaluated'] = not is_evolution
    document = recorder.finalize()
    render_replay_html(document, output / 'replay.html')
    return document
