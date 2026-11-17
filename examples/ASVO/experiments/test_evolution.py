"""Offline tests of the three-team evolutionary protocol."""
import copy
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import unittest
import httpx
from openai import APIConnectionError
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

from concordia.language_model.base_gpt_model import BaseGPTModel

from .engine import deliver_scheduled_updates, load_resources, persona_profile
from .evolution import (AGENT_ACTION_GUIDANCE, INDIVIDUAL_EVALUATION_CONTEXT,
                        MEETING_GOAL, MEETING_SHARED_MEMORIES,
                        bind_team, next_generation, prepare_evolution, run_evolution,
                        rebuild_overview_html, summarize_usage, teamwork_metrics, validate_plan)
from .protocol import CATEGORIES, fingerprint
from .test_experiments import fixture_config


def replay(score_pattern, expected_steps=2):
    # Deliberately no desire fields: selection must not require them.
    return {'teamwork_big_five': {'source': 'action'},
            'frames': [{'step': step, 'agents': [
        {'name': name, 'event': 'Recorded outcome' if step else None,
         'action': 'Chosen behavior' if step else None,
         'teamwork_big_five': {'vector': list(score_pattern),
                             'vector_order': list(CATEGORIES)} if step else None}
        for name in ('P01', 'P02', 'P03')]} for step in range(expected_steps + 1)]}


class EvolutionTests(unittest.TestCase):
    def setUp(self):
        self.plan = prepare_evolution(fixture_config(), pool_seed=20261031,
                                      search_seed=43, simulation_seed=101)

    def test_fixed_pool_initial_population_and_budget(self):
        self.assertEqual(self.plan, prepare_evolution(fixture_config(),
                         pool_seed=20261031, search_seed=43, simulation_seed=101))
        validate_plan(self.plan)
        self.assertEqual(len(self.plan['personas']), 20)
        self.assertEqual(self.plan['possible_teams'], 1140)
        self.assertEqual(len({t['id'] for t in self.plan['teams']}), 3)
        self.assertEqual(self.plan['generation_updates'], 10)
        self.assertEqual(self.plan['planned_simulations'], 23)
        self.assertEqual(self.plan['selection_rule'], 'positive_element_count')
        self.assertNotIn('coverage_weight', self.plan)
        self.assertEqual(self.plan['config']['runtime']['episode_length'], 3)
        self.assertEqual(self.plan['mutation_probability'], .5)
        self.assertEqual(self.plan['schema_version'], 27)
        self.assertEqual(self.plan['action_parser_version'], 2)
        self.assertNotIn('agent_action_guidance', self.plan)
        self.assertEqual(self.plan['teamwork_rubric_version'], 7)
        self.assertEqual(self.plan['teamwork_classification_source'], 'action')
        self.assertEqual(self.plan['uniform_svo_type'], 'Prosocial')
        self.assertEqual(self.plan['persona_variation'], 'personality_only')
        self.assertEqual(self.plan['initial_desire_value'], 5)
        self.assertEqual(self.plan['config']['replay']['teamwork_classifier_model_name'],
                         'gpt-6.1-sol')
        self.assertEqual({p['role']['social_personality'] for p in self.plan['personas']},
                         {'Prosocial'})
        self.assertGreater(len({tuple(sorted(p['role']['personality'].items()))
                                 for p in self.plan['personas']}), 1)
        self.assertEqual(len({tuple(sorted(p['initial_desires'].items()))
                               for p in self.plan['personas']}), 1)

    def test_unspecified_seeds_change_between_new_plans(self):
        with patch('examples.ASVO.experiments.evolution.secrets.randbits',
                   side_effect=(101, 201, 301, 102, 202, 302)):
            first = prepare_evolution(fixture_config())
            second = prepare_evolution(fixture_config())
        self.assertEqual((first['pool_seed'], first['search_seed'], first['simulation_seed']),
                         (101, 201, 301))
        self.assertEqual((second['pool_seed'], second['search_seed'], second['simulation_seed']),
                         (102, 202, 302))
        self.assertNotEqual(first['teams'], second['teams'])

    def test_new_plan_varies_personality_with_uniform_initial_desires(self):
        plan = prepare_evolution(fixture_config(), svo_type='Competitive')
        validate_plan(plan)
        self.assertEqual({p['role']['social_personality'] for p in plan['personas']},
                         {'Competitive'})
        self.assertEqual({p['role']['gender'] for p in plan['personas']}, {'female'})
        self.assertGreater(len({tuple(sorted(p['role']['personality'].items()))
                                 for p in plan['personas']}), 1)
        self.assertTrue(all(set(p['initial_desires'].values()) == {5}
                            for p in plan['personas']))
        self.assertTrue(all(set(p['desire_traits'].values()) <=
                            {'slightly', 'moderately', 'quite', 'extremely'}
                            for p in plan['personas']))
        invalid = copy.deepcopy(plan)
        invalid['personas'][0]['role']['social_personality'] = 'Prosocial'
        with self.assertRaises(ValueError):
            validate_plan(invalid)
        invalid = copy.deepcopy(plan)
        invalid['personas'][0]['initial_desires']['comfort'] = 4
        with self.assertRaises(ValueError):
            validate_plan(invalid)
        invalid = copy.deepcopy(plan)
        invalid['personas'][0]['desire_traits']['competitive'] = 'invalid degree'
        with self.assertRaises(ValueError):
            validate_plan(invalid)

    def test_previous_fixed_personality_plan_remains_valid(self):
        old = copy.deepcopy(self.plan)
        old['schema_version'] = 25
        old['teamwork_rubric_version'] = 6
        old['persona_variation'] = 'initial_desires_only'
        old.pop('initial_desire_value')
        for persona in old['personas']:
            persona['role']['personality'] = {key: 5 for key in persona['role']['personality']}
            persona['desire_traits'] = {key: 'moderately' for key in persona['desire_traits']}
        validate_plan(old)
        prior = copy.deepcopy(self.plan)
        prior['schema_version'] = 26
        prior['teamwork_rubric_version'] = 6
        validate_plan(prior)

    def test_new_plan_uses_standard_asvo_action_prompt(self):
        persona = self.plan['personas'][0]
        profile = persona_profile(persona)
        self.assertIn('{agent_name} is ', profile)
        self.assertNotIn(AGENT_ACTION_GUIDANCE, profile)
        invalid = copy.deepcopy(self.plan)
        invalid['agent_action_guidance'] = 'different prompt'
        with self.assertRaises(ValueError):
            validate_plan(invalid)
        previous = copy.deepcopy(self.plan)
        previous['schema_version'] = 17
        previous['teamwork_rubric_version'] = 6
        previous['persona_variation'] = 'initial_desires_only'
        previous.pop('initial_desire_value')
        for persona in previous['personas']:
            persona['role']['personality'] = {key: 5 for key in persona['role']['personality']}
            persona['desire_traits'] = {key: 'moderately' for key in persona['desire_traits']}
        previous.pop('action_parser_version')
        previous['selection_rule'] = 'any_action_rate_plus_coverage_bonus'
        previous['coverage_weight'] = 0.1
        previous['agent_action_guidance'] = AGENT_ACTION_GUIDANCE
        validate_plan(previous)
        self.assertIn(AGENT_ACTION_GUIDANCE,
                      persona_profile(persona, previous['agent_action_guidance']))
        self.assertIn(INDIVIDUAL_EVALUATION_CONTEXT,
                      self.plan['config']['scenario']['shared_memories'])

    def test_resources_use_separate_simulation_and_classifier_models(self):
        encoder = SimpleNamespace(encode=lambda text, **kwargs: [len(text)])
        fake_embeddings = SimpleNamespace(SentenceTransformer=lambda name: encoder)
        config = copy.deepcopy(self.plan['config'])
        config['embedding'] = {'model_name': 'test-embedding'}
        with (patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}),
              patch.dict(sys.modules, {'sentence_transformers': fake_embeddings}),
              patch('openai.OpenAI', return_value=object())):
            model, classifier_model, embedder = load_resources(config)
        self.assertEqual(model._model_name, self.plan['config']['model']['model_name'])
        self.assertEqual(classifier_model._model_name, 'gpt-6.1-sol')
        self.assertIsNot(model, classifier_model)
        self.assertEqual(embedder('test'), [4])
        self.assertEqual(self.plan['participant_assignment'], 'sorted_persona_ids_to_neutral_slots')

    def test_open_meeting_scenario_only_changes_new_evolution_plan(self):
        config = json.loads((Path(__file__).parents[1] / 'simulation_config.json').read_text())
        config['runtime']['episode_length'] = 4
        original = fingerprint(config)
        self.assertTrue(any(' is responsible for ' in memory
                            for memory in config['scenario']['shared_memories']))
        plan = prepare_evolution(config)
        validate_plan(plan)
        self.assertEqual(fingerprint(config), original)
        self.assertFalse(any(' is responsible for ' in memory
                             for memory in plan['config']['scenario']['shared_memories']))
        self.assertEqual(plan['config']['scenario']['shared_memories'],
                         list(MEETING_SHARED_MEMORIES))
        self.assertEqual(plan['config']['scenario']['shared_memories'].count(
            INDIVIDUAL_EVALUATION_CONTEXT), 1)
        self.assertEqual(plan['config']['scenario']['role_defaults']['goal'], MEETING_GOAL)
        self.assertEqual(plan['config']['scenario']['role_defaults']['specific_memories'], [])
        self.assertTrue(all(role['goal'] == MEETING_GOAL and role['specific_memories'] == []
                            for role in plan['config']['scenario']['roles']))
        self.assertNotIn('noon', json.dumps(plan['config']['scenario']))
        self.assertNotIn('research', json.dumps(plan['config']['scenario']))
        self.assertNotIn('scheduled_updates', plan['config']['scenario'])
        bound, _ = bind_team(plan, plan['teams'][0])
        self.assertFalse(any(' is responsible for ' in memory
                             for memory in bound['scenario']['shared_memories']))

    def test_saved_fixed_duty_plan_remains_valid_for_resume(self):
        old = copy.deepcopy(self.plan)
        old['schema_version'] = 4
        old.pop('action_parser_version')
        old['selection_rule'] = 'mean_rate_times_element_coverage'
        old['role_assignment'] = 'sorted_persona_ids_to_original_role_slots'
        del old['participant_assignment']
        old['config']['scenario']['shared_memories'].append(
            'P01 is responsible for planning, P02 is responsible for costs.')
        validate_plan(old)

    def test_old_scheduled_update_reaches_everyone_only_at_scheduled_step(self):
        class Memory:
            def __init__(self): self.messages = []
            def add(self, text, tags=None): self.messages.append((text, tags))

        class Player:
            def __init__(self): self.messages = []
            def observe(self, text): self.messages.append(text)

        players = [Player() for _ in range(3)]
        memory = Memory()
        config = copy.deepcopy(self.plan['config'])
        config['scenario']['scheduled_updates'] = [
            {'before_step': 2, 'text': 'Old experiment update'}]
        self.assertEqual(deliver_scheduled_updates(config, 1, players, memory), [])
        self.assertEqual(len(deliver_scheduled_updates(config, 2, players, memory)), 1)
        message = config['scenario']['scheduled_updates'][0]['text']
        self.assertEqual([player.messages for player in players], [[message]] * 3)
        self.assertEqual(memory.messages, [(message, ['scenario_update'])])

    def test_debug_plan_can_use_zero_or_one_generation(self):
        for updates, expected in ((0, 3), (1, 5)):
            plan = prepare_evolution(fixture_config(), generation_updates=updates)
            validate_plan(plan)
            self.assertEqual(plan['planned_simulations'], expected)

    def test_new_plan_rejects_coverage_bonus(self):
        plan = copy.deepcopy(self.plan)
        plan['coverage_weight'] = 0.1
        with self.assertRaises(ValueError):
            validate_plan(plan)

    def test_new_score_counts_every_positive_element(self):
        document = replay([0, 0, 0, 0, 0], expected_steps=3)
        agents = [a for f in document['frames'][1:] for a in f['agents']]
        agents[0]['teamwork_big_five']['vector'] = [1, 1, 0, 0, 0]
        agents[1]['teamwork_big_five']['vector'] = [1, 0, 0, 0, 0]
        metrics = teamwork_metrics(document, 3, 'positive_element_count')
        self.assertEqual(metrics['score'], 3)
        self.assertEqual(metrics['positive_element_count'], 3)
        self.assertEqual(metrics['positive_action_count'], 2)
        self.assertEqual(metrics['action_count'], 9)
        self.assertEqual(metrics['counts']['team_leadership'], 2)
        self.assertEqual(teamwork_metrics(replay([1] * 5, expected_steps=3), 3,
                                          'positive_element_count')['score'], 45)

    def test_api_usage_is_logged_per_trial_and_aggregated(self):
        response = SimpleNamespace(
            model='gpt-5.6-luna',
            usage=SimpleNamespace(
                prompt_tokens=120, completion_tokens=30, total_tokens=150,
                prompt_tokens_details=SimpleNamespace(cached_tokens=20),
                completion_tokens_details=SimpleNamespace(reasoning_tokens=5)),
            choices=[SimpleNamespace(message=SimpleNamespace(content='ok'))],
        )
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kwargs: response)))
        model = BaseGPTModel('gpt-5.6-luna', client)
        plan = prepare_evolution(fixture_config(), generation_updates=0)

        def simulate(plan, team, seed, advice, output, *, model):
            model.sample_text('trial')
            return replay([1, 0, 0, 0, 0], expected_steps=3)

        with tempfile.TemporaryDirectory() as directory:
            records = run_evolution(plan, directory, partial(simulate, model=model))
            self.assertEqual(len(records), 3)
            self.assertTrue(all(r['usage']['api_calls'] == 1 for r in records))
            result = json.loads((Path(directory) / 'results.json').read_text())
            self.assertEqual(result['usage_measured_trials'], 3)
            self.assertEqual(result['usage']['input_tokens'], 360)
            self.assertEqual(result['usage']['cached_input_tokens'], 60)
            self.assertEqual(result['usage']['output_tokens'], 90)
            self.assertEqual(result['usage']['reasoning_output_tokens'], 15)
            self.assertEqual(result['usage']['total_tokens'], 450)
            first = Path(directory) / 'trials' / records[0]['team']['id']
            self.assertEqual(summarize_usage(first / 'usage.jsonl'), records[0]['usage'])

    def test_gpt6_luna_uses_supported_chat_completion_parameters(self):
        requests = []
        response = SimpleNamespace(
            usage=None,
            choices=[SimpleNamespace(message=SimpleNamespace(content='ok'))],
        )
        def create(**kwargs):
            requests.append(kwargs)
            return response
        client = SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=create)))
        model = BaseGPTModel('gpt-6-luna', client)
        self.assertEqual(model.sample_text('test', max_tokens=100), 'ok')
        self.assertEqual(requests[0]['model'], 'gpt-6-luna')
        self.assertEqual(requests[0]['max_completion_tokens'], 100)
        self.assertEqual(requests[0]['reasoning_effort'], 'none')
        self.assertNotIn('temperature', requests[0])
        self.assertNotIn('max_tokens', requests[0])

    def test_gpt61_sol_uses_supported_low_reasoning(self):
        requests = []
        response = SimpleNamespace(
            usage=None,
            choices=[SimpleNamespace(message=SimpleNamespace(content='{}'))],
        )
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kwargs: requests.append(kwargs) or response)))
        model = BaseGPTModel('gpt-6.1-sol', client)
        self.assertEqual(model.sample_text('classify', max_tokens=4000), '{}')
        self.assertEqual(requests[0]['reasoning_effort'], 'low')
        self.assertNotIn('temperature', requests[0])

    def test_transient_api_connection_error_is_retried(self):
        calls = []
        def create(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise APIConnectionError(request=httpx.Request('POST', 'https://api.openai.com/v1/chat/completions'))
            return SimpleNamespace(usage=None, choices=[SimpleNamespace(
                message=SimpleNamespace(content='recovered'))])
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        with patch('concordia.language_model.base_gpt_model.time.sleep') as sleep:
            self.assertEqual(BaseGPTModel('gpt-6-luna', client).sample_text('test'), 'recovered')
        self.assertEqual(len(calls), 2)
        sleep.assert_called_once_with(2)

    def test_separate_classifier_usage_is_included_in_trial_log(self):
        def create(**kwargs):
            return SimpleNamespace(
                model=kwargs['model'],
                usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20,
                                      total_tokens=120, prompt_tokens_details=None,
                                      completion_tokens_details=None),
                choices=[SimpleNamespace(message=SimpleNamespace(content='ok'))],
            )
        client = SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=create)))
        model = BaseGPTModel('gpt-6-luna', client)
        classifier_model = BaseGPTModel('gpt-6.1-sol', client)
        plan = prepare_evolution(fixture_config(), generation_updates=0)

        def simulate(plan, team, seed, advice, output, *, model, classifier_model):
            model.sample_text('simulation')
            classifier_model.sample_text('classification')
            return replay([1, 0, 0, 0, 0], expected_steps=3)

        with tempfile.TemporaryDirectory() as directory:
            records = run_evolution(plan, directory, partial(
                simulate, model=model, classifier_model=classifier_model))
            self.assertTrue(all(record['usage']['api_calls'] == 2 for record in records))
            first = Path(directory) / 'trials' / records[0]['team']['id']
            models = [json.loads(line)['model']
                      for line in (first / 'usage.jsonl').read_text().splitlines()]
            self.assertEqual(models, ['gpt-6-luna', 'gpt-6.1-sol'])

    def test_metrics_use_all_five_and_ignore_desires(self):
        metrics = teamwork_metrics(replay([1, 1, 0, 0, 0]), 2)
        self.assertEqual(metrics['mean_rate'], .4)
        self.assertEqual(metrics['element_coverage'], .4)
        self.assertEqual(metrics['positive_action_count'], 6)
        self.assertEqual(metrics['any_action_rate'], 1)
        self.assertNotIn('any_event_rate', metrics)
        self.assertAlmostEqual(metrics['score'], 1.04)
        self.assertEqual(metrics['rates']['team_leadership'], 1)
        self.assertEqual(metrics['action_count'], 6)

    def test_new_score_requires_action_classification(self):
        document = replay([1, 0, 0, 0, 0])
        document['frames'][1]['agents'][0]['event'] = None
        self.assertEqual(teamwork_metrics(document, 2)['positive_action_count'], 6)
        document['frames'][1]['agents'][0]['action'] = None
        with self.assertRaises(ValueError):
            teamwork_metrics(document, 2)
        document = replay([1, 0, 0, 0, 0])
        document['teamwork_big_five']['source'] = 'event'
        with self.assertRaises(ValueError):
            teamwork_metrics(document, 2)
        old = teamwork_metrics(document, 2, 'any_event_rate_plus_coverage_bonus')
        self.assertEqual(old['event_count'], 6)
        self.assertNotIn('any_action_rate', old)

    def test_any_label_counts_once_and_coverage_bonus_is_additive(self):
        def document_with_labels(assignments):
            document = replay([0] * 5, expected_steps=4)
            agents = [agent for frame in document['frames'][1:] for agent in frame['agents']]
            for event_index, category_indices in assignments.items():
                for category_index in category_indices:
                    agents[event_index]['teamwork_big_five']['vector'][category_index] = 1
            return document

        broad = teamwork_metrics(document_with_labels({0: [0, 1], 1: [2], 2: [3], 3: [4]}), 4)
        narrow = teamwork_metrics(document_with_labels({i: [0] for i in range(5)}), 4)
        self.assertEqual(broad['positive_action_count'], 4)
        self.assertEqual(narrow['positive_action_count'], 5)
        self.assertEqual(broad['element_coverage'], 1)
        self.assertEqual(narrow['element_coverage'], .2)
        self.assertAlmostEqual(broad['score'], 4 / 12 + .10)
        self.assertAlmostEqual(narrow['score'], 5 / 12 + .02)
        self.assertLess(broad['score'], narrow['score'])
        mean_only = teamwork_metrics(document_with_labels({i: [0] for i in range(5)}), 4,
                                     'mean_five_element_rate')
        self.assertEqual(mean_only['score'], narrow['mean_rate'])
        old = teamwork_metrics(document_with_labels({i: [0] for i in range(5)}), 4,
                               'mean_rate_times_element_coverage')
        self.assertAlmostEqual(old['score'], narrow['mean_rate'] * .2)
        schema9 = copy.deepcopy(self.plan)
        schema9['schema_version'] = 9
        schema9.pop('action_parser_version')
        schema9['selection_rule'] = 'mean_rate_plus_coverage_bonus'
        schema9['coverage_weight'] = 0.1
        validate_plan(schema9)
        schema10 = copy.deepcopy(self.plan)
        schema10['schema_version'] = 10
        schema10.pop('action_parser_version')
        schema10['selection_rule'] = 'any_event_rate_plus_coverage_bonus'
        schema10['coverage_weight'] = 0.1
        del schema10['teamwork_rubric_version']
        validate_plan(schema10)
        schema11 = copy.deepcopy(self.plan)
        schema11['schema_version'] = 11
        schema11.pop('action_parser_version')
        schema11['selection_rule'] = 'any_event_rate_plus_coverage_bonus'
        schema11['coverage_weight'] = 0.1
        schema11['teamwork_rubric_version'] = 2
        validate_plan(schema11)
        schema12 = copy.deepcopy(schema11)
        schema12['schema_version'] = 12
        schema12['teamwork_rubric_version'] = 3
        validate_plan(schema12)
        schema13 = copy.deepcopy(self.plan)
        schema13['schema_version'] = 13
        schema13.pop('action_parser_version')
        schema13['selection_rule'] = 'any_action_rate_plus_coverage_bonus'
        schema13['coverage_weight'] = 0.1
        schema13['teamwork_rubric_version'] = 4
        validate_plan(schema13)
        schema14 = copy.deepcopy(self.plan)
        schema14['schema_version'] = 14
        schema14.pop('action_parser_version')
        schema14['selection_rule'] = 'any_action_rate_plus_coverage_bonus'
        schema14['coverage_weight'] = 0.1
        schema14['teamwork_rubric_version'] = 5
        validate_plan(schema14)
        schema15 = copy.deepcopy(self.plan)
        schema15['schema_version'] = 15
        schema15.pop('action_parser_version')
        schema15['selection_rule'] = 'any_action_rate_plus_coverage_bonus'
        schema15['coverage_weight'] = 0.1
        schema15['teamwork_rubric_version'] = 6
        validate_plan(schema15)

    def test_missing_labels_order_or_steps_fail(self):
        for field in ('labels', 'order', 'steps', 'agents'):
            document = replay([0]*5)
            if field == 'labels':
                document['frames'][1]['agents'][0]['teamwork_big_five'] = None
            elif field == 'order':
                document['frames'][1]['agents'][0]['teamwork_big_five']['vector_order'] = []
            elif field == 'steps':
                document['frames'].pop()
            else:
                document['frames'][1]['agents'].pop()
            with self.subTest(field=field), self.assertRaises(ValueError):
                teamwork_metrics(document, 2)

    def test_exact_rank_rule_and_optional_mutation(self):
        population = [{'team': candidate, 'metrics': {'score': 1 - i/4},
                       'evaluation_index': i} for i, candidate in enumerate(self.plan['teams'])]
        seen = {r['team']['id'] for r in population}
        elite, offspring = next_generation(population, [p['id'] for p in self.plan['personas']],
                                            seen, random.Random(1))
        self.assertIs(elite, population[0])
        self.assertEqual(len(offspring), 2)
        ranked = sorted(population, key=lambda row: -row['metrics']['score'])
        for child, (first_rank, second_rank) in zip(offspring, ((1, 2), (1, 3))):
            origin = child['origin']
            self.assertEqual(origin['parent_ranks'], [first_rank, second_rank])
            self.assertEqual(origin['parents'], [ranked[first_rank-1]['team']['id'],
                                                 ranked[second_rank-1]['team']['id']])
            cross = set(origin['crossover'])
            parents = [set(ranked[i-1]['team']['members']) for i in (first_rank, second_rank)]
            self.assertTrue(cross <= parents[0] | parents[1])
            self.assertTrue(cross & (parents[0] - parents[1]))
            self.assertTrue(cross & (parents[1] - parents[0]))
            mutation = origin['mutation']
            if mutation is None:
                self.assertEqual(origin['kind'], 'crossover')
                self.assertEqual(set(child['members']), cross)
            else:
                self.assertEqual(origin['kind'], 'crossover_mutation')
                self.assertIn(mutation['removed'], cross)
                self.assertNotIn(mutation['added'], parents[0] | parents[1])
                self.assertEqual(set(child['members']), cross - {mutation['removed']} | {mutation['added']})
            self.assertTrue(set(child['members']) & (parents[0] - parents[1]))
            self.assertTrue(set(child['members']) & (parents[1] - parents[0]))
        self.assertEqual(len({t['id'] for t in offspring}), 2)
        self.assertFalse({t['id'] for t in offspring} & seen)

    def test_mutation_probability_has_both_paths(self):
        population = [{'team': candidate, 'metrics': {'score': 1 - i/4},
                       'evaluation_index': i} for i, candidate in enumerate(self.plan['teams'])]
        ids = [p['id'] for p in self.plan['personas']]
        seen = {r['team']['id'] for r in population}

        class FixedRandom:
            def __init__(self, draw):
                self.draw = draw

            def random(self):
                return self.draw

            def choice(self, values):
                return values[0]

        for draw, expected in ((0.0, True), (0.5, False)):
            with self.subTest(draw=draw):
                _, children = next_generation(population, ids, seen, FixedRandom(draw))
                self.assertEqual([c['origin']['mutation'] is not None for c in children],
                                 [expected, expected])

    def test_participant_assignment_rewrites_world_without_changing_people(self):
        self.plan['config']['scenario']['shared_memories'] = ['P01 plans, P02 brews, P03 markets.']
        before = fingerprint(self.plan)
        candidate = self.plan['teams'][0]
        config, people = bind_team(self.plan, candidate)
        self.assertEqual([p['id'] for p in people], sorted(candidate['members']))
        self.assertEqual(config['scenario']['shared_memories'], [
            f'{people[0]["id"]} plans, {people[1]["id"]} brews, {people[2]["id"]} markets.'])
        self.assertEqual(fingerprint(self.plan), before)
        for person in people:
            original = next(p for p in self.plan['personas'] if p['id'] == person['id'])
            for key in ('initial_desires', 'desire_traits'):
                self.assertEqual(person[key], original[key])
            self.assertEqual(person['role']['personality'], original['role']['personality'])

    def test_twenty_three_calls_elites_no_advice_and_resume(self):
        calls = []
        def simulate(plan, team, seed, advice, output):
            self.assertIsNone(advice)
            self.assertEqual(seed, self.plan['simulation_seed'])
            if calls:
                overview = (Path(output).parents[1] / 'index.html').read_text()
                self.assertIn(f'{len(calls)} / 23', overview)
                self.assertIn('trials/' + calls[-1] + '/replay.html', overview)
            calls.append(team)
            return replay([1] * (len(calls) % 5) + [0] * (5 - len(calls) % 5),
                          expected_steps=3)
        with tempfile.TemporaryDirectory() as folder:
            records = run_evolution(self.plan, folder, simulate)
            self.assertEqual(len(calls), 23)
            self.assertEqual(len(set(calls)), 23)
            self.assertEqual([sum(r['generation'] == g for r in records) for g in range(11)], [3] + [2] * 10)
            generations = json.loads((Path(folder)/'generations.json').read_text())
            self.assertEqual([len(g['population']) for g in generations], [3] * 11)
            for old, new in zip(generations, generations[1:]):
                self.assertIn(old['best_team'], [r['team']['id'] for r in new['population']])
                self.assertGreaterEqual(new['best_score'], old['best_score'])
            run_evolution(self.plan, folder, simulate)
            self.assertEqual(len(calls), 23)
            text = (Path(folder)/'report.md').read_text()
            self.assertIn('初期3チーム＋10世代×新規2チーム＝23回', text)
            self.assertIn('突然変異:', text)
            self.assertIn('担当は事前に固定せず', text)
            self.assertIn('選抜スコア（無単位）', text)
            self.assertIn('達成率ではありません', text)
            self.assertIn('1と判定された', text)
            self.assertIn('0〜45点', text)
            self.assertIn('独立試行での再現性', text)
            self.assertIn('新規平均', text)
            overview = (Path(folder) / 'index.html').read_text()
            self.assertIn('23 / 23', overview)
            self.assertIn('完了', overview)
            self.assertIn('リーダーシップ', overview)
            self.assertIn('支援行動', overview)
            path = next((Path(folder)/'trials').glob('*/replay.json'))
            path.write_text('{}')
            with self.assertRaises(ValueError):
                run_evolution(self.plan, folder, simulate)

    def test_three_step_ten_generation_plan_completes_twenty_three_trials(self):
        config = fixture_config()
        config['runtime']['episode_length'] = 4
        plan = prepare_evolution(config, generation_updates=10)
        self.assertEqual(plan['config']['runtime']['episode_length'], 3)
        calls = []
        def simulate(plan, team, seed, advice, output):
            calls.append(team)
            return replay([1, 0, 0, 0, 0], expected_steps=3)
        with tempfile.TemporaryDirectory() as folder:
            records = run_evolution(plan, folder, simulate)
            self.assertEqual(len(calls), 23)
            self.assertEqual([r['metrics']['action_count'] for r in records], [9] * 23)
            self.assertEqual([len(g['population']) for g in
                json.loads((Path(folder) / 'generations.json').read_text())], [3] * 11)

    def test_stop_at_generation_three_and_resume_without_repeating_api_trials(self):
        calls = []
        def simulate(plan, team, seed, advice, output):
            calls.append(team)
            return replay([1, 0, 0, 0, 0], expected_steps=3)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            first = run_evolution(self.plan, path, simulate, through_generation=3)
            self.assertEqual(len(first), 9)
            partial = json.loads((path / 'results.json').read_text())
            self.assertFalse(partial['complete'])
            self.assertEqual(partial['completed_generation'], 3)
            self.assertEqual(partial['completed_simulations'], 9)
            self.assertEqual(partial['planned_simulations'], 23)
            self.assertIsNone(partial['initial_to_final_best_delta'])
            self.assertIn('途中経過', (path / 'report.md').read_text())
            self.assertFalse((path / 'experiment.lock').exists())
            run_evolution(self.plan, path, simulate, through_generation=3)
            self.assertEqual(len(calls), 9)
            with self.assertRaises(ValueError):
                run_evolution(self.plan, path, simulate, through_generation=2)
            final = run_evolution(self.plan, path, simulate, through_generation=10)
            self.assertEqual(len(final), 23)
            self.assertEqual(len(calls), 23)
            self.assertEqual(len(set(calls)), 23)
            result = json.loads((path / 'results.json').read_text())
            self.assertTrue(result['complete'])
            self.assertEqual(result['completed_generation'], 10)
            self.assertIsNotNone(result['initial_to_final_best_delta'])
            self.assertIn('完了', (path / 'report.md').read_text())

    def test_extend_completed_ten_generation_run_to_twenty_without_rerunning_trials(self):
        calls = []
        def simulate(plan, team, seed, advice, output):
            calls.append(team)
            return replay([1, 0, 0, 0, 0], expected_steps=3)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            run_evolution(self.plan, path, simulate)
            self.assertEqual(len(calls), 23)
            original_plan = (path / 'plan.json').read_bytes()
            extended = run_evolution(self.plan, path, simulate, through_generation=20)
            self.assertEqual(len(extended), 43)
            self.assertEqual(len(calls), 43)
            self.assertEqual(len(set(calls)), 43)
            self.assertEqual((path / 'plan.json').read_bytes(), original_plan)
            result = json.loads((path / 'results.json').read_text())
            self.assertEqual(result['planned_generation_updates'], 20)
            self.assertEqual(result['planned_simulations'], 43)
            self.assertEqual(result['completed_generation'], 20)
            self.assertTrue(result['complete'])
            self.assertIn('20世代×新規2チーム＝43回', (path / 'report.md').read_text())
            run_evolution(self.plan, path, simulate)
            self.assertEqual(len(calls), 43)

    def test_partial_failure_resume_reuses_successful_trials(self):
        successful = []
        failed = False
        def simulate(plan, team, seed, advice, output):
            nonlocal failed
            if len(successful) == 5 and not failed:
                failed = True
                raise RuntimeError('simulated temporary failure')
            successful.append(team)
            return replay([0]*5, expected_steps=3)
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(RuntimeError):
                run_evolution(self.plan, folder, simulate)
            checkpoint = json.loads((Path(folder) / 'results.json').read_text())
            self.assertEqual(checkpoint['completed_generation'], 1)
            self.assertEqual(checkpoint['completed_simulations'], 5)
            self.assertIn('5 / 23', (Path(folder) / 'index.html').read_text())
            run_evolution(self.plan, folder, simulate)
            self.assertEqual(len(successful), 23)
            self.assertEqual(len(set(successful)), 23)

    def test_overview_rebuilds_after_interruption_before_first_generation(self):
        calls = []
        def simulate(plan, team, seed, advice, output):
            calls.append(team)
            if len(calls) == 2:
                raise RuntimeError('interrupted')
            return replay([1, 0, 0, 0, 0], expected_steps=3)
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(RuntimeError):
                run_evolution(self.plan, folder, simulate)
            root = Path(folder)
            self.assertFalse((root / 'results.json').exists())
            (root / 'index.html').unlink()
            self.assertEqual(rebuild_overview_html(self.plan, root), 1)
            overview = (root / 'index.html').read_text()
            self.assertIn('1 / 23', overview)
            self.assertIn('replay.html', overview)
            self.assertNotIn('2 / 23', overview)

    def test_changed_plan_cannot_reuse_trials(self):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder)/'plan.json').write_text('{}')
            with self.assertRaises(ValueError):
                run_evolution(self.plan, folder, None)
            self.assertFalse((Path(folder)/'experiment.lock').exists())


if __name__ == '__main__':
    unittest.main()
