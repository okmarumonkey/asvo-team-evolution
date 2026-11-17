"""Regression tests for ASVO action candidate parsing."""

from examples.ASVO.ASVO_agent.Value_Act_SVO import MCTSActComponent


def _parser(*, flexible=True):
    component = object.__new__(MCTSActComponent)
    component._num_proposed_actions = 3
    component._flexible_action_parsing = flexible
    return component


def test_unnumbered_concrete_action_remains_a_candidate():
    answer = 'P01 spends 20 minutes discussing the meal-friendly beer with P06 and P19.'
    assert _parser()._preprocess_imagined_action(answer) == [answer]


def test_numbered_activity_variants_are_extracted():
    answer = 'Activity 1: Compare A and B.\nActivity 2) Discuss C.\n3. Revise the recommendation.'
    assert _parser()._preprocess_imagined_action(answer) == [
        'Compare A and B.', 'Discuss C.', 'Revise the recommendation.',
    ]


def test_explicit_no_action_stays_empty():
    assert _parser()._preprocess_imagined_action('No feasible actions were provided.') == []


def test_saved_plans_keep_original_strict_parser():
    answer = 'P01 spends 20 minutes discussing the meal-friendly beer.'
    assert _parser(flexible=False)._preprocess_imagined_action(answer) == []
