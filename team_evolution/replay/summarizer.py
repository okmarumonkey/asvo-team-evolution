"""Replaceable Japanese summary generation for replay frames."""

from __future__ import annotations

from collections.abc import Sequence
import json
from typing import Any, Protocol


DESIRE_LABELS_JA = {
    "senseofsuperiority": "優越感",
    "senseofachievement": "達成感",
    "confidence": "自信",
    "joyfulness": "喜び",
    "comfort": "快適さ",
    "recognition": "承認",
    "spiritualsatisfaction": "精神的充足",
    "senseofcontrol": "統制感",
}


class StepSummarizer(Protocol):
    """Interface for template or LLM-backed per-step summaries."""

    def summarize(self, frame: dict[str, Any]) -> str:
        """Return a Japanese summary for one replay frame."""


class InitialContextSummarizer(Protocol):
    """Interface for summarizing shared simulation memories."""

    def summarize(self, shared_memories: Sequence[str]) -> str:
        """Return a concise Japanese scenario summary."""


class LLMJapaneseSharedMemoriesSummarizer:
    """Summarize all shared memories in Japanese with one model call."""

    def __init__(self, model: Any, *, max_tokens: int = 500) -> None:
        self._model = model
        self._max_tokens = max_tokens

    def summarize(self, shared_memories: Sequence[str]) -> str:
        memories = "\n".join(f"- {memory}" for memory in shared_memories)
        prompt = f"""以下はASVOシミュレーション開始時に全員が共有している状況設定です。
事実を追加・削除せず、登場人物、場所、課題、雰囲気が分かる自然な日本語で2〜3文に要約してください。
要約本文だけを出力してください。

共有設定:
{memories}

日本語要約:"""
        result = self._model.sample_text(
            prompt,
            max_tokens=self._max_tokens,
            terminators=(),
            temperature=0.0,
        ).strip()
        if not result:
            raise ValueError("Shared-memory summarizer returned an empty response.")
        return result


class LLMJapaneseStepSummarizer:
    """Summarize actions, events, and desire changes for one replay step."""

    def __init__(
        self,
        model: Any,
        *,
        max_tokens: int = 600,
        fallback: StepSummarizer | None = None,
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._fallback = fallback or TemplateJapaneseSummarizer()

    def summarize(self, frame: dict[str, Any]) -> str:
        step = int(frame.get("step", 0))
        if step == 0:
            return self._fallback.summarize(frame)

        actions = []
        observations = []
        changes: list[dict[str, Any]] = []
        for agent in frame.get("agents", []):
            name = str(agent.get("name", ""))
            action = agent.get("action_ja") or agent.get("action")
            actions.append({"name": name, "action": action})
            observations.append({
                "name": name,
                "observation": agent.get("observation"),
            })
            for key, desire in agent.get("desires", {}).items():
                delta = desire.get("delta")
                if isinstance(delta, (int, float)) and delta:
                    changes.append({
                        "agent": name,
                        "desire": DESIRE_LABELS_JA.get(key, key),
                        "before": desire.get("before"),
                        "after": desire.get("after"),
                        "delta": delta,
                        "reason": desire.get("reason"),
                        "reasonableness": desire.get("reasonableness"),
                    })
        changes.sort(key=lambda item: abs(float(item["delta"])), reverse=True)
        source = {
            "step": step,
            "time": frame.get("time"),
            "actions": actions,
            "observed_results": {
                "event": frame.get("event"),
                "agent_observations_used_by_ASVO": observations,
            },
            "largest_desire_changes": changes[:5],
        }
        prompt = f"""以下のASVOシミュレーションStepを自然な日本語で2〜3文に要約してください。
以下の順序で記述してください。
1. 各人の行動
2. シミュレーションに記録された観測結果
3. 重要な欲求値の変化

制約:
- 行動、観測結果、欲求値変化を区別してください。
- reason欄に明示された内容だけを、欲求値変化の理由として利用できます。
- reasonがnullなら、欲求値変化の理由は一切書かないでください。
- action、event、observation、SVOから因果関係や心理を推測しないでください。
- 「○○したため達成感が下がった」「その結果、欲求が低下した」のような後付け因果は禁止です。
- 「欲求が低下した」ではなく「達成感の値が低下した」のように表現し、実際の心理状態として断定しないでください。
- reasonablenessはASVO内部の妥当性判定であり、理由そのものではありません。
- reflectionは棄却された更新案への反省であり、採用された値変化の理由として使わないでください。
- 入力にない事実を追加せず、数値を過剰に列挙しないでください。
要約本文だけを出力してください。


Stepデータ:
{json.dumps(source, ensure_ascii=False, indent=2)}

日本語要約:"""
        try:
            result = self._model.sample_text(
                prompt,
                max_tokens=self._max_tokens,
                terminators=(),
                temperature=0.0,
            ).strip()
        except Exception:
            return self._fallback.summarize(frame)
        return result or self._fallback.summarize(frame)


class TemplateJapaneseSummarizer:
    """Deterministic summary generator that does not call an LLM."""

    def summarize(self, frame: dict[str, Any]) -> str:
        step = int(frame.get("step", 0))
        agents = frame.get("agents", [])
        if step == 0:
            names = "、".join(str(agent.get("name", "")) for agent in agents)
            return f"{names}の行動前の欲求状態です。"

        action_parts = []
        for agent in agents:
            name = str(agent.get("name", ""))
            action = agent.get("action_ja") or agent.get("action")
            action_parts.append(f"{name}: {action or '行動の記録なし'}")
        changes: list[tuple[float, str]] = []
        for agent in agents:
            name = str(agent.get("name", ""))
            for key, desire in agent.get("desires", {}).items():
                delta = desire.get("delta")
                if isinstance(delta, (int, float)) and delta:
                    label = DESIRE_LABELS_JA.get(key, key)
                    direction = "上昇" if float(delta) > 0 else "低下"
                    changes.append(
                        (abs(float(delta)), f"{name}の{label}の値が{direction}")
                    )

        action_sentence = "行動: " + " / ".join(action_parts) + "。"
        event = frame.get("event")
        result_sentence = (
            f"観測結果: {event}。"
            if event else "観測結果は記録されていません。"
        )
        if changes:
            changes.sort(key=lambda item: item[0], reverse=True)
            highlights = "、".join(text for _, text in changes[:3])
            value_sentence = f"欲求値: 主な変化は{highlights}です。"
        else:
            has_pending = any(
                desire.get("after") is None
                for agent in agents
                for desire in agent.get("desires", {}).values()
            )
            value_sentence = (
                "この行動に対応する欲求値更新は記録されていません。"
                if has_pending else
                "欲求値: 前Stepからの変化はありません。"
            )
        return "".join((action_sentence, result_sentence, value_sentence))
