"""Offline tests: python -m unittest team_evolution.experiments.test_experiments"""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from .engine import finish_pending_state
from .protocol import CATEGORIES, TRAITS, measure, prepare, selection_score
from .workflow import generate_advice, run_experiment


def fixture_config():
    personality = {('confidence' if k == 'confident' else k): 6 for k in TRAITS}
    return {
        'runtime': {'num_players': 3, 'episode_length': 2},
        'model': {'api_type': 'openai', 'model_name': 'test', 'api_key_env': 'OPENAI_API_KEY'},
        'profile': {'wanted_desires': list(TRAITS.values()),
                    'initial_value_distribution': {'mean': 5, 'stddev': 2, 'min': 0, 'max': 10}},
        'scenario': {'roles': [{'name': name, 'social_personality': 'Prosocial',
                               'personality': personality} for name in ('P01', 'P02', 'P03')]},
    }


def fixture_replay(plan, team, improvement=0, advice=False):
    members = next(t['members'] for t in plan['teams'] if t['id'] == team)
    people = {p['id']: p for p in plan['personas']}
    frames = []
    for step in (0, 1, 2):
        agents = []
        for member in members:
            name = people[member]['role']['name']
            agents.append({'name': name, 'persona_id': member,
                           'event': f'{name} checks a colleague work product.' if step else None,
                           'teamwork_big_five': {'vector': [0, 1, 1 if advice else 0, 0, 1],
                                                 'vector_order': list(CATEGORIES)} if step else None,
                           'desires': {k: {'target': 6, 'value': 2 if step == 0 else 3 + improvement}
                                       for k in TRAITS.values()}})
        frames.append({'step': step, 'agents': agents})
    return {'frames': frames, 'final_state_evaluated': True}


class Coach:
    def __init__(self):
        self.calls = 0

    def sample_text(self, prompt, **kwargs):
        self.calls += 1
        data = json.loads(prompt.split('Data:\n')[1])
        return json.dumps({p['id']: [{'advice': 'Ask a colleague to review one concrete assumption.',
                                    'run': 0, 'step': 1, 'name': p['role']['name'],
                                    'quote': p['role']['name'] + ' checks'}]
                           for p in data['personas']})


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.plan = prepare(fixture_config())

    def test_pool_freezes_profiles_and_covers_jobs(self):
        self.assertEqual(self.plan, prepare(fixture_config()))
        self.assertEqual(len(self.plan['personas']), 6)
        self.assertEqual(len(self.plan['teams']), 8)
        self.assertEqual(self.plan['planned_simulations'], 16)
        pool = {p['id']: p for p in self.plan['personas']}
        for team in self.plan['teams']:
            self.assertEqual([pool[x]['slot'] for x in team['members']], [0, 1, 2])

    def test_train_test_separation(self):
        with self.assertRaises(ValueError):
            prepare(fixture_config(), discovery_seeds=[1], evaluation_seeds=[1])

    def test_duplicate_seed_rejected(self):
        with self.assertRaises(ValueError):
            prepare(fixture_config(), discovery_seeds=[1, 1])

    def test_rates_are_per_event_not_composition(self):
        metrics = measure(fixture_replay(self.plan, 'team00'))
        self.assertEqual(metrics['rates']['mutual_performance_monitoring'], 1)
        self.assertEqual(metrics['rates']['team_orientation'], 1)
        self.assertEqual(metrics['mean_gap_reduction'], 1)
        self.assertEqual(metrics['valid_events'], 6)

    def test_missing_classification_not_silent_zero(self):
        replay = fixture_replay(self.plan, 'team00')
        replay['frames'][1]['agents'][0]['teamwork_big_five'] = None
        metrics = measure(replay)
        self.assertEqual(metrics['valid_events'], 5)
        with self.assertRaises(ValueError):
            selection_score(metrics)

    def test_final_state_required(self):
        replay = fixture_replay(self.plan, 'team00')
        replay['final_state_evaluated'] = False
        with self.assertRaises(ValueError):
            selection_score(measure(replay))

    def test_target_changes_rejected(self):
        replay = fixture_replay(self.plan, 'team00')
        replay['frames'][-1]['agents'][0]['desires']['confidence']['target'] = 0
        with self.assertRaises(ValueError):
            measure(replay)

    def test_grounded_coaching(self):
        result = generate_advice(self.plan, 'team00', [fixture_replay(self.plan, 'team00')], Coach())
        self.assertEqual(set(result), set(self.plan['teams'][0]['members']))

    def test_coach_invented_evidence_rejected(self):
        class BadCoach(Coach):
            def sample_text(self, prompt, **kwargs):
                raw = json.loads(super().sample_text(prompt, **kwargs))
                next(iter(raw.values()))[0]['quote'] = 'An event that did not happen'
                return json.dumps(raw)
        with self.assertRaises(ValueError):
            generate_advice(self.plan, 'team00', [fixture_replay(self.plan, 'team00')], BadCoach())

    def test_full_workflow_resume_and_tampering(self):
        calls = []
        def simulate(plan, team, seed, advice, output):
            calls.append((team, seed, bool(advice)))
            return fixture_replay(plan, team, improvement=int(team[-1]) / 10, advice=bool(advice))
        with tempfile.TemporaryDirectory() as folder:
            model = Coach()
            results = run_experiment(self.plan, folder, simulate, model)
            self.assertEqual(len(calls), 16)
            self.assertEqual(model.calls, 2)
            self.assertEqual(len(results), 8)
            self.assertEqual(json.loads((Path(folder)/'selection.json').read_text())['selected_team'], 'team07')
            self.assertTrue((Path(folder)/'report.md').exists())
            run_experiment(self.plan, folder, simulate, model)
            self.assertEqual(len(calls), 16)
            self.assertEqual(model.calls, 2)
            replay_path = next((Path(folder)/'discovery').glob('*/replay.json'))
            replay_path.write_text('{}')
            with self.assertRaises(ValueError):
                run_experiment(self.plan, folder, simulate, model)

    def test_plan_change_blocks_resume(self):
        from .protocol import write_json
        with tempfile.TemporaryDirectory() as folder:
            other = copy.deepcopy(self.plan)
            other['pool_seed'] += 1
            write_json(Path(folder)/'plan.json', other)
            with self.assertRaises(ValueError):
                run_experiment(self.plan, folder, None, None)
            self.assertFalse((Path(folder)/'experiment.lock').exists())

    def test_evaluation_phase_does_not_create_action(self):
        # Use Concordia's real phase machine, not a duplicate of the adapter.
        from concordia.agents.entity_agent import EntityAgent
        from concordia.components.agent.action_spec_ignored import ActionSpecIgnored
        from concordia.typing import entity_component
        class Probe(ActionSpecIgnored):
            def __init__(self):
                super().__init__('probe')
                self.evaluations = 0
                self.actions = []
            def _make_pre_act_value(self):
                self.evaluations += 1
                return 'evaluated'
            def post_act(self, action):
                self.actions.append(action)
                return ''
        class Actor(entity_component.ActingComponent):
            def get_action_attempt(self, contexts, action_spec):
                raise AssertionError('No extra action may be generated')
        probe = Probe()
        agent = EntityAgent('P01', Actor(), context_components={'probe': probe})
        finish_pending_state(agent)
        self.assertEqual(probe.evaluations, 1)
        self.assertEqual(probe.actions, [])
        self.assertEqual(agent.get_phase(), entity_component.Phase.READY)


if __name__ == '__main__':
    unittest.main()
