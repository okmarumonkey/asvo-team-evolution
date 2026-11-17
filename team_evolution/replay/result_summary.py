"""Build the final Replay summary from recorded events and final desire values."""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Protocol, Sequence

from .summarizer import DESIRE_LABELS_JA


class ResultSummarizer(Protocol):
    """Interface for LLM-backed collaboration and behavior summaries."""

    def summarize(
        self,
        events: Sequence[Mapping[str, Any]],
        agent_names: Sequence[str],
    ) -> Mapping[str, Any]:
        """Return Japanese summaries based only on recorded Event statements."""


class LLMJapaneseResultSummarizer:
    """Summarize collaboration and agent behavior from GM events only."""

    def __init__(
        self,
        model: Any,
        *,
        collaboration_max_tokens: int = 1000,
        agent_max_tokens: int = 2000,
    ) -> None:
        self._model = model
        self._collaboration_max_tokens = collaboration_max_tokens
        self._agent_max_tokens = agent_max_tokens

    def summarize(
        self,
        events: Sequence[Mapping[str, Any]],
        agent_names: Sequence[str],
    ) -> Mapping[str, Any]:
        if not events:
            return {"collaboration_ja": None, "agent_behavior_ja": {}}

        collaboration_prompt = f"""以下はASVOシミュレーションでGame Masterが実際に記録したEvent statementです。
これらの記録をもとに、シミュレーション結果を日本語で要約してください。

【全体としてどんな協働になったか】

チームが何の作業をしたかではなく、3人がどのような関係で協働していたかを5文でまとめてください。
以下の順序を必ず守ってください。
1文目: シミュレーション全体で最も目立った協働パターンを述べる。
2文目: 誰が主導・調整・支援・確認などを担ったか、役割関係を述べる。
3文目: 情報共有、依頼、確認、作業の受け渡しがどのように行われたかを述べる。
4文目: 意見の食い違い、偏り、協力不足、再調整が記録されていれば述べる。記録されていなければ、円滑さを支えた具体的な連携を述べる。
5文目: 最終的にその協働によってチームの作業がどの状態に至ったかを述べる。
各Agentの行動をStep順に列挙する文章にはしないでください。
報告書、スライド、ファイルなどの具体的な作業対象は、協働パターンの根拠として必要な場合だけ最小限に触れてください。
文章の中心を「確認した作業の一覧」ではなく、「Agent同士の関わり方」にしてください。
複数のEvent statementから協働の特徴を解釈して構いませんが、入力にない出来事、心理、内面的動機、根拠のない因果関係、評価や改善の助言は追加しないでください。
欲求値、SVO、Social Personalityは根拠に使用しないでください。

Event statements:
{json.dumps(list(events), ensure_ascii=False, indent=2)}

日本語要約本文だけを出力してください。"""
        collaboration = self._sample(
            collaboration_prompt, self._collaboration_max_tokens
        )

        events_by_agent = {
            name: agent_events
            for name in agent_names
            if (agent_events := _events_for_agent(events, name))
        }
        agent_prompt = f"""以下はASVOシミュレーションでGame Masterが実際に記録した、Agent別に分けたEvent statementです。
各Agentに紐付いた記録だけを根拠に、それぞれがシミュレーション全体でどのように振る舞ったかを日本語で要約してください。

【各Agentがどう振る舞ったか】

各Agentについて、作業内容を順番に列挙するのではなく、チームの中でどのような役割を担い、他者に対してどのように振る舞っていたかを5文でまとめてください。
各Agentの5文は以下の順序を必ず守ってください。
1文目: 今回のシミュレーションで最も目立った役割または行動傾向を、「〜する傾向が見られた」「〜の役割を担う場面が多かった」などの表現で述べる。
2文目: 他のAgentへの依頼、支援、確認、共有、調整の仕方を述べる。
3文目: 複数Stepに共通する行動パターンを述べる。
4文目: その傾向を裏付けるEvent statement上の具体例を、時系列の列挙にならないよう必要最小限に示す。
5文目: チーム内での今回の振る舞いを総括する。恒常的な性格としては断定しない。
具体的な作業名の一覧ではなく、行動傾向、役割、他者との関わりを文章の主語・中心にしてください。
入力にない出来事、感情、心理状態、内面的動機、根拠のない因果関係、評価や改善の助言は追加しないでください。
他のAgentのEvent statementを本人の行動として扱わないでください。欲求値、SVO、Social Personalityは根拠に使用しないでください。

Agent別Event statements:
{json.dumps(events_by_agent, ensure_ascii=False, indent=2)}

次のようにAgent名をキー、要約本文を値とする有効なJSONオブジェクトだけを出力してください。
{json.dumps({name: f"{name}の日本語要約" for name in events_by_agent}, ensure_ascii=False, indent=2)}
Markdownやコードフェンスは使用しないでください。"""
        parsed_behaviors = _parse_json_object(
            self._sample(agent_prompt, self._agent_max_tokens)
        )
        behaviors = {
            name: value.strip()
            for name in events_by_agent
            if isinstance((value := parsed_behaviors.get(name)), str)
            and value.strip()
        }

        return {
            "collaboration_ja": collaboration,
            "agent_behavior_ja": behaviors,
        }

    def _sample(self, prompt: str, max_tokens: int) -> str:
        result = self._model.sample_text(
            prompt,
            max_tokens=max_tokens,
            terminators=(),
            temperature=0.0,
        ).strip()
        if not result:
            raise ValueError("Result summarizer returned an empty response.")
        return result


def _events_for_agent(
    events: Sequence[Mapping[str, Any]], agent_name: str
) -> list[dict[str, Any]]:
    """Select only Event statements explicitly associated with one agent."""
    selected: list[dict[str, Any]] = []
    for record in events:
        for item in record.get("events_by_agent", []):
            if item.get("agent") == agent_name and item.get("event"):
                selected.append({
                    "step": record.get("step"),
                    "event": item.get("event"),
                })
    return selected


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
            raise ValueError("Agent summary response did not contain JSON.")
        value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("Agent summary response must be a JSON object.")
    return value


def recorded_events(frames: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return events without duplicating combined step events and agent events."""
    records: list[dict[str, Any]] = []
    for frame in frames:
        if int(frame.get("step", 0)) == 0:
            continue
        agent_events = [
            {"agent": agent.get("name"), "event": agent.get("event")}
            for agent in frame.get("agents", [])
            if agent.get("event")
        ]
        record: dict[str, Any] = {"step": frame.get("step")}
        if agent_events:
            record["events_by_agent"] = agent_events
        elif frame.get("event"):
            record["event"] = frame.get("event")
        else:
            continue
        records.append(record)
    return records


def build_result_summary(
    frames: Sequence[Mapping[str, Any]],
    summarizer: ResultSummarizer | None = None,
) -> dict[str, Any]:
    """Combine optional LLM summaries with deterministic final shortages."""
    final_frame = frames[-1] if frames else {}
    final_agents = final_frame.get("agents", [])
    names = [str(agent.get("name", "")) for agent in final_agents]
    result: dict[str, Any] = {
        "source": "event_statements",
        "final_step": final_frame.get("step"),
        "collaboration_ja": None,
        "agents": [],
    }

    behaviors: Mapping[str, Any] = {}
    events = recorded_events(frames)
    if summarizer is not None and events:
        generated = summarizer.summarize(events, names)
        result["collaboration_ja"] = generated.get("collaboration_ja")
        candidate_behaviors = generated.get("agent_behavior_ja", {})
        if isinstance(candidate_behaviors, Mapping):
            behaviors = candidate_behaviors

    for agent in final_agents:
        name = str(agent.get("name", ""))
        shortages: list[dict[str, Any]] = []
        for key, desire in agent.get("desires", {}).items():
            current = desire.get("value")
            expected = desire.get("target")
            if not isinstance(current, (int, float)) or not isinstance(
                expected, (int, float)
            ):
                continue
            if float(current) < float(expected):
                shortages.append({
                    "key": key,
                    "label_ja": DESIRE_LABELS_JA.get(key, key),
                    "current": current,
                    "expected": expected,
                    "gap": float(expected) - float(current),
                })
        behavior = behaviors.get(name)
        result["agents"].append({
            "name": name,
            "behavior_ja": (
                behavior.strip()
                if isinstance(behavior, str) and behavior.strip()
                else None
            ),
            "insufficient_desires": shortages,
        })
    return result
