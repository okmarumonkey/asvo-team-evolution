"""Replaceable post-processing translation for Replay text."""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Protocol


class ActionTranslator(Protocol):
    """Interface for translating one step's actions as a batch."""

    def translate(self, actions: Mapping[str, str], *, step: int) -> Mapping[str, str]:
        """Return Japanese actions keyed by the original agent names."""


class ReplayFieldTranslator(Protocol):
    """Interface for translating recorded Event statements."""

    def translate(self, texts: Mapping[str, str], *, step: int) -> Mapping[str, str]:
        """Return faithful Japanese text with exactly the same keys."""


class LLMJapaneseActionTranslator:
    """Translate all actions in one step with one LanguageModel call."""

    def __init__(self, model: Any, *, max_tokens: int = 1000) -> None:
        self._model = model
        self._max_tokens = max_tokens

    def translate(self, actions: Mapping[str, str], *, step: int) -> Mapping[str, str]:
        if not actions:
            return {}
        if all(_contains_japanese(action) for action in actions.values()):
            return dict(actions)

        source = json.dumps(dict(actions), ensure_ascii=False, indent=2)
        prompt = f"""Translate the following ASVO simulation actions into natural Japanese.
Preserve agent names, facts, intent, duration, and relationships. Do not add or omit information.
Return only one valid JSON object with exactly the same keys. Do not use Markdown fences.
Simulation step: {step}
Actions:
{source}
Japanese JSON:"""
        response = self._model.sample_text(
            prompt,
            max_tokens=self._max_tokens,
            terminators=(),
            temperature=0.0,
        )
        translated = _parse_json_object(response)
        result: dict[str, str] = {}
        for name, original in actions.items():
            value = translated.get(name)
            result[name] = value.strip() if isinstance(value, str) and value.strip() else original
        return result


class LLMJapaneseReplayFieldTranslator:
    """Translate recorded Event statements without reinterpretation."""

    def __init__(self, model: Any, *, max_tokens: int = 4000) -> None:
        self._model = model
        self._max_tokens = max_tokens

    def translate(self, texts: Mapping[str, str], *, step: int) -> Mapping[str, str]:
        if not texts:
            return {}
        if all(_contains_japanese(text) for text in texts.values()):
            return dict(texts)

        source = json.dumps(dict(texts), ensure_ascii=False, indent=2)
        prompt = f"""Translate the following ASVO Game Master Event statements into natural Japanese.
Every value is an Event statement actually recorded by the simulation.
Preserve every fact, agent name, sequence, uncertainty, and causal statement exactly as written.
Do not summarize, interpret, explain, infer, or add information.
Return only one valid JSON object with exactly the same keys. Do not use Markdown fences.
Simulation step: {step}
Recorded text:
{source}
Japanese JSON:"""
        response = self._model.sample_text(
            prompt,
            max_tokens=self._max_tokens,
            terminators=(),
            temperature=0.0,
        )
        translated = _parse_json_object(response)
        result: dict[str, str] = {}
        for key, original in texts.items():
            value = translated.get(key)
            result[key] = (
                value.strip()
                if isinstance(value, str) and value.strip()
                else original
            )
        return result


def _contains_japanese(text: str) -> bool:
    return bool(re.search(r"[\u3040-\u30ff\u3400-\u9fff]", text))


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
            raise ValueError("Translation response did not contain a JSON object.")
        value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("Translation response must be a JSON object.")
    return value
