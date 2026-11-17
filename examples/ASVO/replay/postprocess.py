"""Post-process only the final result summary of an existing Replay JSON."""

from __future__ import annotations

import json
from pathlib import Path

from .exporter import (
    SCHEMA_VERSION,
    _atomic_write_json,
    _classify_recorded_events,
)
from .result_summary import ResultSummarizer, build_result_summary
from .teamwork_classifier import (
    TEAMWORK_BIG_FIVE_METADATA,
    TeamworkClassifier,
)
from .viewer import render_replay_html


def _upgrade_schema_version(document: dict) -> None:
    try:
        current_version = int(document.get("schema_version", 0))
    except (TypeError, ValueError):
        current_version = 0
    document["schema_version"] = max(current_version, SCHEMA_VERSION)


def summarize_replay_file(
    replay_json_path: str | Path,
    result_summarizer: ResultSummarizer,
    *,
    html_output_path: str | Path | None = None,
) -> tuple[Path, Path]:
    """Regenerate only ``result_summary`` and its Viewer HTML.

    Existing frames and all other Replay fields are preserved. If the LLM
    summary fails, the function raises before writing either output.
    """
    json_path = Path(replay_json_path)
    document = json.loads(json_path.read_text(encoding="utf-8"))
    frames = document.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("Replay JSON must contain at least one frame.")

    result_summary = build_result_summary(frames, result_summarizer)
    document["result_summary"] = result_summary
    document.pop("result_summary_error", None)
    _upgrade_schema_version(document)

    html_path = (
        Path(html_output_path)
        if html_output_path is not None
        else json_path.with_suffix(".html")
    )
    _atomic_write_json(json_path, document)
    render_replay_html(document, html_path)
    return json_path, html_path


def classify_replay_file(
    replay_json_path: str | Path,
    teamwork_classifier: TeamworkClassifier,
    *,
    html_output_path: str | Path | None = None,
) -> tuple[Path, Path]:
    """Add Big Five classifications to an existing Replay JSON and HTML."""
    json_path = Path(replay_json_path)
    document = json.loads(json_path.read_text(encoding="utf-8"))
    frames = document.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("Replay JSON must contain at least one frame.")

    errors: list[str] = []
    scenario_context = document.get("simulation", {}).get(
        "shared_memories", []
    )
    if not isinstance(scenario_context, list):
        scenario_context = []
    source_field = document.get("teamwork_big_five", {}).get("source", "event")
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            continue
        previous = frames[index - 1] if index else None
        error = _classify_recorded_events(
            frame,
            teamwork_classifier,
            scenario_context=scenario_context,
            source_field=source_field,
            previous_event=previous.get("event") if isinstance(previous, dict) else None,
        )
        if error:
            frame["teamwork_classification_error"] = error
            errors.append(f"step {frame.get('step')}: {error}")
        else:
            frame.pop("teamwork_classification_error", None)
    if errors:
        raise ValueError("; ".join(errors))

    document["teamwork_big_five"] = dict(TEAMWORK_BIG_FIVE_METADATA)
    _upgrade_schema_version(document)
    html_path = (
        Path(html_output_path)
        if html_output_path is not None
        else json_path.with_suffix(".html")
    )
    _atomic_write_json(json_path, document)
    render_replay_html(document, html_path)
    return json_path, html_path
