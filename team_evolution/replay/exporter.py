"""Build versioned replay JSON without changing the simulation engine."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .summarizer import (
    DESIRE_LABELS_JA,
    InitialContextSummarizer,
    StepSummarizer,
    TemplateJapaneseSummarizer,
)
from .result_summary import ResultSummarizer, build_result_summary
from .teamwork_classifier import (
    TEAMWORK_BIG_FIVE_METADATA,
    TeamworkClassifier,
)
from .translator import ActionTranslator, ReplayFieldTranslator


SCHEMA_VERSION = 7


def _iso_time(value: Any) -> str:
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()
    return str(value)


def _normalise_desire_key(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum())


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _desire_metadata(keys: Iterable[str]) -> list[dict[str, str]]:
    return [
        {"key": key, "label_ja": DESIRE_LABELS_JA.get(key, key)}
        for key in keys
    ]


def _value_for_step(tracker: Mapping[Any, Any], step: int) -> Mapping[str, Any]:
    for key in (step, str(step)):
        if key in tracker:
            return tracker[key]
    numeric = sorted((int(key), value) for key, value in tracker.items())
    eligible = [value for key, value in numeric if key <= step]
    return eligible[-1] if eligible else {}


def _exact_value_for_step(tracker: Mapping[Any, Any], step: int) -> Mapping[str, Any]:
    """Return an exact step without carrying evidence forward."""
    for key in (step, str(step)):
        if key in tracker:
            value = tracker[key]
            return value if isinstance(value, Mapping) else {}
    return {}


def _action_for_step(actions: Sequence[Mapping[str, Any]], step: int) -> str | None:
    if step == 0 or step > len(actions):
        return None
    action = actions[step - 1].get("action")
    return str(action).strip() if action is not None else None


def _decision_for_step(
    history: Sequence[Mapping[str, Any]], step: int
) -> Mapping[str, Any] | None:
    if step == 0 or step > len(history):
        return None
    decision = history[step - 1]
    return decision if isinstance(decision, Mapping) else None


def _frame_agent(
    name: str,
    social_personality: str | None,
    action: str | None,
    action_ja: str | None,
    values: Mapping[str, Any],
    expected_values: Mapping[str, Any],
    *,
    pending_transition: bool = False,
) -> dict[str, Any]:
    desires: dict[str, dict[str, Any]] = {}
    for raw_key, raw_value in values.items():
        key = _normalise_desire_key(str(raw_key))
        value = float(raw_value)
        target_value = expected_values.get(key)
        target = None if target_value is None else float(target_value)
        desires[key] = {
            "value": value,
            "before": value if pending_transition else None,
            "after": None if pending_transition else value,
            "delta": None,
            "target": target,
            "gap": None if target is None else target - value,
            "reason": None,
            "reasonableness": None,
            "reflection": None,
            "correction": None,
            "update_evidence": None,
        }
    return {
        "name": name,
        "social_personality": social_personality,
        "action": action,
        "action_ja": action_ja,
        "event": None,
        "event_ja": None,
        "teamwork_big_five": None,
        "observation": None,
        "desires": desires,
    }


def _complete_agent_transition(
    agent: dict[str, Any],
    after_values: Mapping[str, Any],
    update_evidence: Mapping[str, Any],
) -> None:
    """Attach the next ASVO value-update cycle to its preceding action."""
    normalised_after = {
        _normalise_desire_key(str(key)): value for key, value in after_values.items()
    }
    normalised_evidence = {
        _normalise_desire_key(str(key)): value
        for key, value in update_evidence.items()
        if isinstance(value, Mapping)
    }
    observations = [
        str(item.get("observation")).strip()
        for item in normalised_evidence.values()
        if item.get("observation")
    ]
    agent["observation"] = observations[0] if observations else None

    for key, desire in agent.get("desires", {}).items():
        if key not in normalised_after:
            continue
        before = desire.get("before", desire.get("value"))
        after = float(normalised_after[key])
        target = desire.get("target")
        evidence = normalised_evidence.get(key)
        desire.update({
            "value": after,
            "before": before,
            "after": after,
            "delta": None if before is None else after - float(before),
            "gap": None if target is None else float(target) - after,
        })
        if evidence:
            desire.update({
                # No accepted-update reason is synthesized. In the original
                # ASVO prompt, only rejected proposals produce reflection text.
                "reason": evidence.get("reason"),
                "reasonableness": evidence.get("reasonableness"),
                "reflection": evidence.get("reflection"),
                "correction": evidence.get("correction"),
                "update_evidence": {
                    "value_before_update": evidence.get("value_before_update"),
                    "value_after_update": evidence.get("value_after_update"),
                    "attempts": evidence.get("attempts", []),
                    "fluctuation": evidence.get("fluctuation"),
                },
            })


def _personality_from_svo(value: Any) -> str | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if score >= 67.47:
        return "Altruistic"
    if score >= 45.0:
        return "Prosocial"
    if score >= 22.53:
        return "Individualistic"
    return "Competitive"


def _saved_social_personality(data: Mapping[str, Any], step: int) -> str | None:
    configured = data.get("configured_social_personality")
    if isinstance(configured, str) and configured.strip():
        return configured.strip()
    tracker = data.get("svo_tracker", {})
    value = tracker.get(str(step), tracker.get(step))
    return _personality_from_svo(value) or data.get("current_social_personality")


def _translate_agents(
    agents: list[dict[str, Any]],
    translator: ActionTranslator | None,
    *,
    step: int,
) -> str | None:
    if translator is None or step == 0:
        return None
    actions = {
        agent["name"]: agent["action"]
        for agent in agents
        if agent.get("action")
    }
    if not actions:
        return None
    try:
        translated = translator.translate(actions, step=step)
    except Exception as exc:  # Translation must never stop the simulation.
        return f"{type(exc).__name__}: {exc}"
    for agent in agents:
        value = translated.get(agent["name"])
        if isinstance(value, str) and value.strip():
            agent["action_ja"] = value.strip()
    return None


def _translate_recorded_fields(
    frame: dict[str, Any], translator: ReplayFieldTranslator | None
) -> str | None:
    """Translate only recorded GM Event statements in one batch."""
    if translator is None:
        return None
    texts: dict[str, str] = {}
    has_agent_events = any(
        agent.get("event") for agent in frame.get("agents", [])
    )
    if frame.get("event") and not has_agent_events:
        texts["step:event"] = str(frame["event"])
    for agent in frame.get("agents", []):
        name = str(agent.get("name", ""))
        if agent.get("event"):
            texts[f"agent:{name}:event"] = str(agent["event"])
    if not texts:
        return None
    try:
        translated = translator.translate(texts, step=int(frame.get("step", 0)))
    except Exception as exc:  # Translation failure must not stop the simulation.
        return f"{type(exc).__name__}: {exc}"
    if frame.get("event"):
        frame["event_ja"] = translated.get("step:event")
    for agent in frame.get("agents", []):
        name = str(agent.get("name", ""))
        if agent.get("event"):
            agent["event_ja"] = translated.get(f"agent:{name}:event")
    return None


def _classify_recorded_events(
    frame: dict[str, Any],
    classifier: TeamworkClassifier | None,
    *,
    scenario_context: Sequence[str] = (),
    source_field: str = "event",
    previous_event: str | None = None,
) -> str | None:
    """Classify each agent's recorded source statement in one batch."""
    if classifier is None:
        return None
    events_by_agent = {
        str(agent.get("name", "")): str(agent[source_field])
        for agent in frame.get("agents", [])
        if agent.get("name") and agent.get(source_field)
    }
    if not events_by_agent:
        return None
    try:
        context = list(scenario_context)
        if previous_event and getattr(classifier, "uses_previous_event_context", False):
            context.append(
                "Immediately preceding confirmed Event (trigger context for "
                "adaptability only): " + previous_event
            )
        classifications = classifier.classify(
            events_by_agent,
            step=int(frame.get("step", 0)),
            scenario_context=context,
        )
    except Exception as exc:  # Classification failure must not stop simulation.
        return f"{type(exc).__name__}: {exc}"
    for agent in frame.get("agents", []):
        result = classifications.get(str(agent.get("name", "")))
        if result is not None:
            agent["teamwork_big_five"] = dict(result)
    return None


class ReplayRecorder:
    """Capture lightweight snapshots after ``env.step()`` and persist each one."""

    def __init__(
        self,
        output_path: str | Path,
        *,
        title: str = "ASVO Simulation",
        summarizer: StepSummarizer | None = None,
        action_translator: ActionTranslator | None = None,
        field_translator: ReplayFieldTranslator | None = None,
        teamwork_classifier: TeamworkClassifier | None = None,
        teamwork_source: str = "event",
        result_summarizer: ResultSummarizer | None = None,
        shared_memories: Sequence[str] = (),
        initial_context_summarizer: InitialContextSummarizer | None = None,
    ) -> None:
        self.output_path = Path(output_path)
        self.summarizer = summarizer or TemplateJapaneseSummarizer()
        self.action_translator = action_translator
        self.field_translator = field_translator
        self.teamwork_classifier = teamwork_classifier
        if teamwork_source not in ("event", "action"):
            raise ValueError("teamwork_source must be 'event' or 'action'.")
        self.teamwork_source = teamwork_source
        self.result_summarizer = result_summarizer
        self.shared_memories = tuple(shared_memories)
        self.initial_context_summarizer = initial_context_summarizer
        self.document: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "simulation": {
                "title": title,
                "captured_at": dt.datetime.now().isoformat(),
                "shared_memories": list(self.shared_memories),
            },
            "desires": [],
            "frames": [],
        }
        if self.teamwork_classifier is not None:
            self.document["teamwork_big_five"] = {
                **TEAMWORK_BIG_FIVE_METADATA, "source": self.teamwork_source}

    @staticmethod
    def _tracker(player: Any) -> Any:
        return player.get_component("ValueTracker")

    def capture(
        self,
        *,
        step: int,
        time: Any,
        players: Sequence[Any],
        event: str | None = None,
        events_by_agent: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        """Capture one frame. Call with step 0 before, then after each env.step()."""
        previous_frame = self.document["frames"][-1] if self.document["frames"] else None
        if previous_frame and int(previous_frame.get("step", 0)) > 0:
            previous_agents = {
                agent["name"]: agent for agent in previous_frame.get("agents", [])
            }
            for player in players:
                previous_agent = previous_agents.get(player.name)
                if previous_agent is None:
                    continue
                tracker = self._tracker(player)
                after_values = _exact_value_for_step(
                    tracker.get_individual_desire_tracker(), step
                )
                get_update_tracker = getattr(
                    tracker, "get_desire_update_tracker", None
                )
                update_evidence = (
                    _exact_value_for_step(get_update_tracker(), step)
                    if callable(get_update_tracker) else {}
                )
                if after_values:
                    _complete_agent_transition(
                        previous_agent, after_values, update_evidence
                    )
            if previous_frame.get("summary_ja") is None:
                previous_frame["summary_ja"] = self.summarizer.summarize(
                    previous_frame
                )

        agents: list[dict[str, Any]] = []
        for player in players:
            tracker = self._tracker(player)
            values = _value_for_step(tracker.get_individual_desire_tracker(), step)
            actions = tracker.get_action_sequence()
            normalised_values = {
                _normalise_desire_key(str(key)): value for key, value in values.items()
            }
            expected_values = {
                _normalise_desire_key(str(key)): value
                for key, value in tracker.get_expected_value_dict().items()
            }
            get_configured_personality = getattr(
                tracker, "get_configured_social_personality", None
            )
            get_personality = getattr(tracker, "get_current_social_personality", None)
            social_personality = (
                get_configured_personality()
                if callable(get_configured_personality)
                else (
                    get_personality()
                    if callable(get_personality)
                    else getattr(player, "extras", {}).get("social_personality")
                )
            )
            frame_agent = _frame_agent(
                player.name,
                social_personality,
                _action_for_step(actions, step),
                None,
                normalised_values,
                expected_values,
                pending_transition=step > 0,
            )
            get_act_component = getattr(player, "get_act_component", None)
            act_component = (
                get_act_component() if callable(get_act_component) else None
            )
            get_decision_history = getattr(
                act_component, "get_decision_history", None
            )
            decision_history = (
                get_decision_history()
                if callable(get_decision_history) else []
            )
            frame_agent["action_decision"] = _decision_for_step(
                decision_history, step
            )
            if events_by_agent and events_by_agent.get(player.name):
                frame_agent["event"] = str(events_by_agent[player.name])
            agents.append(frame_agent)

        translation_error = _translate_agents(
            agents, self.action_translator, step=step
        )
        frame = {
            "step": step,
            "time": _iso_time(time),
            "agents": agents,
            "event": event or None,
            "event_ja": None,
            "summary_ja": None,
        }
        if translation_error:
            frame["translation_error"] = translation_error
        if step == 0:
            frame["summary_ja"] = self.summarizer.summarize(frame)
        if step == 0 and self.shared_memories and self.initial_context_summarizer:
            try:
                frame["summary_ja"] = self.initial_context_summarizer.summarize(
                    self.shared_memories
                )
            except Exception as exc:  # Summary failure must not stop simulation.
                frame["initial_summary_error"] = f"{type(exc).__name__}: {exc}"
        self.document["frames"].append(frame)
        if not self.document["desires"] and agents:
            self.document["desires"] = _desire_metadata(agents[0]["desires"].keys())
        _atomic_write_json(self.output_path, self.document)
        return frame

    def finalize(self) -> dict[str, Any]:
        """Finalize summaries without inventing an unavailable last-step update."""
        for index, frame in enumerate(self.document["frames"]):
            if frame.get("summary_ja") is None:
                frame["summary_ja"] = self.summarizer.summarize(frame)
            translation_error = _translate_recorded_fields(
                frame, self.field_translator
            )
            if translation_error:
                frame["field_translation_error"] = translation_error
            classification_error = _classify_recorded_events(
                frame,
                self.teamwork_classifier,
                scenario_context=self.shared_memories,
                source_field=self.teamwork_source,
                previous_event=(self.document["frames"][index - 1].get("event")
                                if index else None),
            )
            if classification_error:
                frame["teamwork_classification_error"] = classification_error
        try:
            self.document["result_summary"] = build_result_summary(
                self.document["frames"], self.result_summarizer
            )
        except Exception as exc:  # Summary failure must not stop result saving.
            self.document["result_summary"] = build_result_summary(
                self.document["frames"]
            )
            self.document["result_summary_error"] = f"{type(exc).__name__}: {exc}"
        _atomic_write_json(self.output_path, self.document)
        return self.document


def build_replay_from_saved_results(
    result_files: Sequence[str | Path],
    *,
    output_path: str | Path | None = None,
    title: str = "ASVO Simulation",
    summarizer: StepSummarizer | None = None,
    action_translator: ActionTranslator | None = None,
    field_translator: ReplayFieldTranslator | None = None,
    teamwork_classifier: TeamworkClassifier | None = None,
    result_summarizer: ResultSummarizer | None = None,
) -> dict[str, Any]:
    """Convert existing ``*_ASVO_<agent>.json`` files to replay schema v7."""
    if not result_files:
        raise ValueError("At least one agent result JSON is required.")
    summarizer = summarizer or TemplateJapaneseSummarizer()
    sources: list[tuple[str, dict[str, Any]]] = []
    for result_file in result_files:
        path = Path(result_file)
        data = json.loads(path.read_text(encoding="utf-8"))
        name = path.stem.rsplit("_", 1)[-1]
        sources.append((name, data))

    maximum_step = max(
        max(
            len(data.get("action_sequence", [])),
            len(data.get("action_decision_history", [])),
            max((int(key) for key in data.get("individual_desire", {})), default=0),
        )
        for _, data in sources
    )
    frames: list[dict[str, Any]] = []
    for step in range(maximum_step + 1):
        agents: list[dict[str, Any]] = []
        timestamps: list[str] = []
        for name, data in sources:
            actions = data.get("action_sequence", [])
            values = _value_for_step(data.get("individual_desire", {}), step)
            normalised_values = {
                _normalise_desire_key(str(key)): value for key, value in values.items()
            }
            expected_values = {
                _normalise_desire_key(str(key)): value
                for key, value in data.get("expected_values", {}).items()
            }
            action = _action_for_step(actions, step)
            agents.append(
                _frame_agent(
                    name,
                    _saved_social_personality(data, step),
                    action,
                    None,
                    normalised_values,
                    expected_values,
                    pending_transition=step > 0,
                )
            )
            agents[-1]["action_decision"] = _decision_for_step(
                data.get("action_decision_history", []), step
            )
            if step > 0:
                after_values = _exact_value_for_step(
                    data.get("individual_desire", {}), step + 1
                )
                update_evidence = _exact_value_for_step(
                    data.get("desire_update_tracker", {}), step + 1
                )
                if after_values:
                    _complete_agent_transition(
                        agents[-1], after_values, update_evidence
                    )
            if step > 0 and step <= len(actions):
                timestamps.append(str(actions[step - 1].get("timestamp", "")))

        if not timestamps:
            first_actions = sources[0][1].get("action_sequence", [])
            timestamps = [str(first_actions[0].get("timestamp", ""))] if first_actions else [""]
        translation_error = _translate_agents(
            agents, action_translator, step=step
        )
        frame = {
            "step": step,
            "time": timestamps[0],
            "agents": agents,
            "event": None,
            "event_ja": None,
            "summary_ja": None,
        }
        if translation_error:
            frame["translation_error"] = translation_error
        frame["summary_ja"] = summarizer.summarize(frame)
        field_translation_error = _translate_recorded_fields(
            frame, field_translator
        )
        if field_translation_error:
            frame["field_translation_error"] = field_translation_error
        classification_error = _classify_recorded_events(
            frame, teamwork_classifier
        )
        if classification_error:
            frame["teamwork_classification_error"] = classification_error
        frames.append(frame)

    first_keys = frames[0]["agents"][0]["desires"].keys() if frames and frames[0]["agents"] else []
    document = {
        "schema_version": SCHEMA_VERSION,
        "simulation": {"title": title, "source": "saved_agent_results"},
        "desires": _desire_metadata(first_keys),
        "frames": frames,
    }
    if teamwork_classifier is not None:
        document["teamwork_big_five"] = dict(TEAMWORK_BIG_FIVE_METADATA)
    try:
        document["result_summary"] = build_result_summary(
            frames, result_summarizer
        )
    except Exception as exc:
        document["result_summary"] = build_result_summary(frames)
        document["result_summary_error"] = f"{type(exc).__name__}: {exc}"
    if output_path is not None:
        _atomic_write_json(Path(output_path), document)
    return document
