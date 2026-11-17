"""Classify recorded agent statements using the Big Five of Teamwork."""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Protocol, Sequence


TEAMWORK_BIG_FIVE_VECTOR_ORDER = (
    "team_leadership",
    "mutual_performance_monitoring",
    "backup_behavior",
    "adaptability",
    "team_orientation",
)

TEAMWORK_BIG_FIVE_METADATA = {
    "framework": "Salas, Sims, and Burke (2005) Big Five of Teamwork",
    "classification_type": "independent_binary_multilabel",
    "vector_order": list(TEAMWORK_BIG_FIVE_VECTOR_ORDER),
}


class TeamworkClassifier(Protocol):
    """Interface for classifying one step's agent statements as a batch."""

    def classify(
        self,
        events_by_agent: Mapping[str, str],
        *,
        step: int,
        scenario_context: Sequence[str] = (),
    ) -> Mapping[str, Mapping[str, Any]]:
        """Return a validated Big Five classification keyed by agent name."""


class LLMTeamworkClassifier:
    """Classify agent statements with one LanguageModel call per replay step."""

    def __init__(self, model: Any, *, max_tokens: int = 4000,
                 rubric_version: int = 3) -> None:
        if rubric_version not in (1, 2, 3, 4, 5, 6, 7):
            raise ValueError('Unsupported teamwork rubric version.')
        self._model = model
        self._max_tokens = max_tokens
        self._rubric_version = rubric_version
        self.uses_previous_event_context = rubric_version >= 7

    def classify(
        self,
        events_by_agent: Mapping[str, str],
        *,
        step: int,
        scenario_context: Sequence[str] = (),
    ) -> Mapping[str, Mapping[str, Any]]:
        if not events_by_agent:
            return {}

        source_field = "agent_action" if self._rubric_version >= 4 else "event_statement"
        source = {
            name: {"focal_agent": name, source_field: statement}
            for name, statement in events_by_agent.items()
        }
        legacy_prompt = f"""Classify the focal agent's recorded behavior in each ASVO Game Master Event statement using Salas, Sims, and Burke's (2005) Big Five of Teamwork.

Evaluate only actions performed by focal_agent. Evidence must describe an action taken by the focal agent. Another person's action, a causal result clause, or a team-level outcome is not evidence about the focal agent. Phrases such as "resulting in", "which enables", "leading to", or "gives the team" must not be used as evidence unless they also explicitly describe the focal agent's qualifying action. Use the scenario context only to understand the team's shared task, prior constraints, changes, and which work products are shared; never treat the context as an action performed by the focal agent. Make five independent binary decisions. Multiple categories and an all-zero result are allowed. Classify observable functional behavior rather than requiring the Event statement to use the category's exact terminology. Apply a conservative threshold: assign 1 only when the statement explicitly identifies a concrete qualifying action and the category-specific target or trigger. Do not infer helpful intent, a teammate's need, a change in plan, or a team benefit from generic verbs or outcomes. When the evidence is ambiguous, assign 0. A single qualifying action may satisfy multiple categories only if each category's distinct positive criterion is explicitly supported.

Categories and positive criteria:
- team_leadership: Use 1 only when the focal agent performs an explicit leadership intervention that changes or establishes at least one teammate's role or task assignment, the team's priorities, the team's decision-making process, expected team behavior or performance standards, or the resolution of a substantive team-level problem or conflict. Formal authority is not required, and leadership may be shared across members, but the intervention and its target must be stated concretely. Routine facilitation—such as running a rehearsal or meeting, managing time or speaking order, checking progress, confirming details, summarizing, closing out, or conducting a retrospective—is 0 by itself, even when described with words such as "leads", "coordinates", or "facilitates". Such facilitation is 1 only when the Event statement also explicitly describes a new role or task allocation, a change in team priorities, a consequential team decision, substantive conflict resolution, or the setting or correction of a team performance standard. Merely reporting information, organizing materials, updating a shared file, planning the focal agent's own work, completing an assigned task, or producing a successful or orderly team outcome is not sufficient.
- mutual_performance_monitoring: Use 1 only when the focal agent intentionally checks a specific teammate's task execution, work product, progress, workload, or errors against a shared requirement, plan, or performance standard. The teammate or the teammate-owned contribution being monitored and what is checked must be concretely identifiable in the Event statement. General quality assurance of an undifferentiated shared output, checking the focal agent's own contribution, vague phrases such as "verifies details", "corrects omissions", or "checks the proposal" without an identifiable teammate-owned target, a check performed by "the team" rather than the focal agent, and routine post-submission rechecking are 0. Comparing one teammate's contribution with another teammate's specifications or figures is 1 only when the focal agent is explicitly checking that teammate-owned contribution for consistency or error; merely integrating or editing content is not sufficient.
- backup_behavior: Use 1 only when the focal agent performs or offers a concrete task-related action to assist an identifiable teammate or their identifiable contribution, takes over or shares that teammate's work, or directly removes a stated blockage to another member's work. A workload crisis is not required, but the teammate or blocked work and the assistance must be explicit. Generic cooperation, sharing information or a file with everyone, finishing one's own work, and an outcome that merely benefits the team are 0.
- adaptability: Use 1 only when the focal agent makes a concrete change to a plan, product concept, strategy, priority, role allocation, resource use, or work approach because of an identifiable changed requirement, new information, feedback, unexpected condition, or discovered discrepancy. The trigger may be stated in the scenario context, but the Event statement must describe the focal agent's responsive change; the new state or successful outcome alone is not enough. Routine drafting, polishing, or adding detail without a stated response to the trigger is 0.
- team_orientation: Use 1 only when the focal agent (a) actively seeks, evaluates, or incorporates a teammate's proposal or expert input into a decision; (b) explicitly puts the team goal above the focal agent's own proposal, credit, convenience, or interest; (c) promotes participatory goal setting or a genuinely joint decision; or (d) integrates competing teammate perspectives into a team proposal. Merely sharing or presenting information, updating a shared document, working beside others, completing an assigned contribution, coordinating a shared deliverable, or using the word "collaborates" is not sufficient. Teammates' confirmation or a resulting state such as team alignment, shared understanding, or a corrected shared proposal is not sufficient.

For each category return an object with integer value (0 or 1) and evidence. For value 1, evidence must be a short exact phrase from the Event statement that describes the focal agent's behavior. For value 0, evidence must be null. Return only one valid JSON object with exactly the same top-level agent keys and exactly the five category keys. Do not return vector or labels; they will be derived by code. Do not use Markdown fences.

Simulation step: {step}
Scenario context:
{json.dumps(list(scenario_context), ensure_ascii=False, indent=2)}
Events:
{json.dumps(source, ensure_ascii=False, indent=2)}
Classification JSON:"""
        concise_prompt = f"""Classify each focal_agent's Event statement using the Big Five of Teamwork. Judge five categories independently as 0 or 1.

Rules: Score only a concrete action explicitly performed by that agent. Context may identify a shared task or a change, but is not evidence of an action. Do not infer intent or credit the agent for another person's action, a group decision, or a team outcome. If a criterion is ambiguous, use 0. Multiple 1s require separate support for each category.

Categories:
- team_leadership: 1 only for an explicit intervention by the agent that changes a teammate's task or role, team priorities, decision process, or performance standard, or resolves a substantive team conflict. Leading a review, facilitating a meeting, summarizing, checking progress, or managing speaking order is 0 by itself. A group agreement is not evidence that this agent led the decision; the agent's own intervention must be stated. The word "led" alone is insufficient.
- mutual_performance_monitoring: 1 only when the agent checks an identifiable teammate's work, progress, workload, or error against an identifiable requirement or plan. Checking one's own work or an undifferentiated shared output is 0.
- backup_behavior: 1 only for concrete task help offered or given to an identifiable teammate or their contribution, taking over their work, or removing a stated blockage to their work. General cooperation, information sharing, and finishing one's own task are 0.
- adaptability: 1 only when the agent concretely changes a plan, objective, priority, role, resource, or work method in response to an identifiable new requirement, information, feedback, or discrepancy. Context may establish the trigger; the Event must state the agent's change. Routine editing is 0.
- team_orientation: 1 only when the agent actively seeks, evaluates, or incorporates a teammate's input into a decision; puts the team goal above personal interest; enables a genuinely joint decision; or integrates competing views. Sharing information, working together, and a resulting team agreement are 0 by themselves.

For each category return {{"value": 0 or 1, "evidence": null or a short exact phrase from that agent's Event}}. A 1 requires a phrase describing the agent's qualifying action, not merely its outcome. Use null for every 0. Return one JSON object with exactly the input agent keys and five category keys. No vector, labels, or Markdown.

Simulation step: {step}
Scenario context:
{json.dumps(list(scenario_context), ensure_ascii=False, indent=2)}
Events:
{json.dumps(source, ensure_ascii=False, indent=2)}
Classification JSON:"""
        strict_prompt = f"""Classify each focal_agent's Event statement using the Big Five of Teamwork. Judge each category independently as 0 or 1.

Use 1 only when the Event explicitly states the focal agent's concrete action, its identifiable target, and behavior that satisfies the category below. Context may establish the shared task or a change, but is not evidence that the agent acted. Do not infer intent, ownership, assistance, or influence from generic words such as "led", "helped", or "collaborated". Participation, team outcomes, and another person's actions are 0. If the evidence is ambiguous or incomplete, use 0. Multiple 1s require full support for each category.

Categories:
- team_leadership: The agent explicitly changes a teammate's task or role, team priority, decision process, or performance standard, or resolves a substantive team conflict. State the intervention and what it changes. Merely chairing a discussion, reviewing work, managing speaking order, or reporting a group decision is 0.
- mutual_performance_monitoring: The agent checks an identifiable teammate-owned task, contribution, progress, workload, or error against a stated requirement or plan. Checking one's own work or reviewing a shared output with colleagues, without identifying whose work is checked, is 0.
- backup_behavior: The agent performs concrete help on an identifiable teammate's task, takes over part of it, removes a stated blockage to it, or offers specific task help to that teammate. Generic offers of help, shared work, and completing one's own task are 0.
- adaptability: In response to an identifiable new requirement, information, feedback, or discrepancy, the agent explicitly changes a plan, objective, priority, role, resource, or work method. The trigger may appear in context; the agent's specific change must appear in the Event. Routine editing or refinement without a stated response is 0.
- team_orientation: The agent seeks a teammate's specific input for a decision, evaluates or uses that input in a decision, explicitly favors the team goal over personal interest, enables a concrete joint decision, or integrates competing views. Generic invitations for feedback, merely hearing views, information sharing, working together, and a resulting agreement are 0.

For each category return {{"value": 0 or 1, "evidence": null or a short exact phrase from that agent's Event}}. For 1, the quoted phrase must show the agent's qualifying action and target, not just an outcome. For 0, use null. Return one JSON object with exactly the input agent keys and all five category keys. No vector, labels, or Markdown.

Simulation step: {step}
Scenario context:
{json.dumps(list(scenario_context), ensure_ascii=False, indent=2)}
Events:
{json.dumps(source, ensure_ascii=False, indent=2)}
Classification JSON:"""
        action_prompt = f"""Classify what each focal_agent does in its Action statement using the Big Five of Teamwork. Judge each category independently as 0 or 1.

Use only the focal agent's explicit behavior. Each positive label needs a concrete action and an identifiable teammate, task, decision, or change as its target. Do not credit the focal agent with another person's behavior or infer a concrete act from generic words such as "helps" or "collaborates". If the action is ambiguous, use 0. Multiple 1s require separate support for each category. Scenario context identifies the shared task but is not evidence of behavior.

Categories:
- team_leadership: The agent assigns or changes teammates' tasks or roles, sets team priorities or a decision process, establishes a performance standard, or mediates a substantive team conflict. Merely chairing a discussion or checking progress is 0.
- mutual_performance_monitoring: The agent checks an identifiable teammate's task, contribution, progress, workload, or error against a stated requirement or plan. Checking a shared output without identifying whose contribution is checked is 0.
- backup_behavior: The agent gives concrete help on an identifiable teammate's task, takes over part of it, removes a stated blockage, or offers specific task help. Generic offers of help or shared work are 0.
- adaptability: The agent makes a specific change to a plan, objective, priority, role, resource, or work method in response to identifiable new information, feedback, a changed requirement, or a discrepancy. Merely polishing wording or adding details is 0.
- team_orientation: The agent seeks a teammate's input for a specific decision, evaluates or incorporates that input, sets up a concrete joint decision process, or integrates competing views. Generic invitations for feedback or information sharing are 0.

For each category return {{"value": 0 or 1, "evidence": null or a short exact phrase from that agent's Action}}. For 1, the phrase must show the agent's action and its target. For 0, use null. Return one JSON object with exactly the input agent keys and all five category keys. No vector, labels, or Markdown.

Simulation step: {step}
Scenario context:
{json.dumps(list(scenario_context), ensure_ascii=False, indent=2)}
Actions:
{json.dumps(source, ensure_ascii=False, indent=2)}
Classification JSON:"""
        focused_action_prompt = action_prompt.replace(
            '- team_orientation: The agent seeks a teammate\'s input for a specific decision, evaluates or incorporates that input, sets up a concrete joint decision process, or integrates competing views. Generic invitations for feedback or information sharing are 0.',
            '- team_orientation: Use 1 only when the agent (a) asks a teammate to compare specified options for an identified decision and states how the input will be used, (b) evaluates or incorporates a specified teammate contribution into a specified choice, (c) sets a concrete shared decision procedure, or (d) reconciles clearly conflicting proposals. Asking others to flag concerns or give final feedback, inviting speaking turns, crediting teammates, or recording agreed edits without described input and choice is 0. Example 0: "asks teammates to flag any concerns". Example 1: "asks each teammate to compare options A and B for a shared decision, then uses their input to choose".'
        )
        clarified_action_prompt = focused_action_prompt.replace(
            '- team_leadership: The agent assigns or changes teammates\' tasks or roles, sets team priorities or a decision process, establishes a performance standard, or mediates a substantive team conflict. Merely chairing a discussion or checking progress is 0.',
            '- team_leadership: The agent assigns or changes teammates\' tasks or roles, sets team priorities or a decision process, establishes a performance standard, or mediates a substantive team conflict. Proposing a specific decision procedure counts when the same Action also describes the agent carrying it out, such as seeking teammates\' input for that decision and comparing alternatives. A proposal alone, chairing a discussion, or checking progress is 0.'
        ).replace(
            '- backup_behavior: The agent gives concrete help on an identifiable teammate\'s task, takes over part of it, removes a stated blockage, or offers specific task help. Generic offers of help or shared work are 0.',
            '- backup_behavior: The agent gives concrete help on an identifiable teammate\'s task, takes over part of it, removes a stated blockage, or offers specific task help. Task-specific feedback on an identifiable teammate\'s contribution counts as help when the Action states what content the feedback addresses. Generic encouragement, unspecified feedback, offers of help to whoever needs it, or shared work without a teammate-owned target are 0.'
        )
        adaptability_action_prompt = clarified_action_prompt.replace(
            'Scenario context identifies the shared task but is not evidence of behavior.',
            'Scenario context identifies the shared task but is not evidence of behavior. '
            'The immediately preceding confirmed Event, when provided, may establish only '
            'the trigger for adaptability. It is not evidence of the focal agent\'s Action '
            'or of any other teamwork category.'
        ).replace(
            '- adaptability: The agent makes a specific change to a plan, objective, priority, role, resource, or work method in response to identifiable new information, feedback, a changed requirement, or a discrepancy. Merely polishing wording or adding details is 0.',
            '- adaptability: Use 1 when the focal agent changes a previously proposed or agreed product concept, plan, priority, task assignment, or work method in response to a specific teammate concern, new information, changed requirement, or discovered problem. The Action must state what the agent changed. The trigger may appear in the Action or the immediately preceding confirmed Event, but their connection must be identifiable; do not infer it from timing alone. Choosing or combining the initial options, routine refinement, and saying "incorporates feedback" without specifying the change are 0. For a positive label, quote the phrase in the Action that describes the change.'
        )
        prompt = (legacy_prompt if self._rubric_version == 1 else
                  concise_prompt if self._rubric_version == 2 else
                  strict_prompt if self._rubric_version == 3 else
                  action_prompt if self._rubric_version == 4 else
                  focused_action_prompt if self._rubric_version == 5 else
                  clarified_action_prompt if self._rubric_version == 6 else
                  adaptability_action_prompt)
        response = self._model.sample_text(
            prompt,
            max_tokens=self._max_tokens,
            terminators=(),
            temperature=0.0,
        )
        parsed = _parse_json_object(response)
        return _validate_classifications(parsed, events_by_agent)


def _parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ValueError(
                "Teamwork classification response did not contain a JSON object."
            )
        value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("Teamwork classification response must be a JSON object.")
    return value


def _validate_classifications(
    raw: Mapping[str, Any], events_by_agent: Mapping[str, str]
) -> dict[str, dict[str, Any]]:
    expected_agents = set(events_by_agent)
    if set(raw) != expected_agents:
        raise ValueError(
            "Teamwork classification agent keys do not match the input: "
            f"expected={sorted(expected_agents)}, actual={sorted(raw)}"
        )

    validated: dict[str, dict[str, Any]] = {}
    expected_categories = set(TEAMWORK_BIG_FIVE_VECTOR_ORDER)
    for agent_name in events_by_agent:
        raw_agent = raw.get(agent_name)
        if not isinstance(raw_agent, Mapping):
            raise ValueError(
                f"Teamwork classification for {agent_name!r} must be an object."
            )
        if set(raw_agent) != expected_categories:
            raise ValueError(
                f"Teamwork category keys for {agent_name!r} do not match: "
                f"expected={sorted(expected_categories)}, "
                f"actual={sorted(raw_agent)}"
            )

        classifications: dict[str, dict[str, Any]] = {}
        vector: list[int] = []
        for category in TEAMWORK_BIG_FIVE_VECTOR_ORDER:
            raw_category = raw_agent[category]
            if not isinstance(raw_category, Mapping):
                raise ValueError(f"{agent_name}.{category} must be an object.")
            value = raw_category.get("value")
            if isinstance(value, bool):
                value = int(value)
            if not isinstance(value, int) or value not in (0, 1):
                raise ValueError(
                    f"{agent_name}.{category}.value must be integer 0 or 1."
                )
            evidence = raw_category.get("evidence")
            if value == 1:
                if not isinstance(evidence, str) or not evidence.strip():
                    raise ValueError(
                        f"{agent_name}.{category}.evidence is required for value 1."
                    )
                evidence = evidence.strip()
            else:
                evidence = None
            classifications[category] = {
                "value": value,
                "evidence": evidence,
            }
            if value == 1 and evidence not in events_by_agent[agent_name]:
                # Evidence is useful for auditing but must not prevent a valid
                # binary classification from being saved when the model
                # slightly paraphrases the Event statement.
                classifications[category]["evidence_is_exact"] = False
            vector.append(value)

        validated[agent_name] = {
            "classifications": classifications,
            "labels": [
                category
                for category, value in zip(
                    TEAMWORK_BIG_FIVE_VECTOR_ORDER, vector, strict=True
                )
                if value == 1
            ],
            "vector_order": list(TEAMWORK_BIG_FIVE_VECTOR_ORDER),
            "vector": vector,
        }
    return validated
