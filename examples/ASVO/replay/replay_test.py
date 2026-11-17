from __future__ import annotations

import json
from pathlib import Path

from examples.ASVO.replay import (
    LLMJapaneseActionTranslator,
    LLMJapaneseReplayFieldTranslator,
    LLMJapaneseResultSummarizer,
    LLMJapaneseSharedMemoriesSummarizer,
    LLMJapaneseStepSummarizer,
    LLMTeamworkClassifier,
    TEAMWORK_BIG_FIVE_VECTOR_ORDER,
    ReplayRecorder,
    TemplateJapaneseSummarizer,
    build_result_summary,
    build_replay_from_saved_results,
    render_replay_html,
    summarize_replay_file,
)
from examples.ASVO.replay.exporter import _classify_recorded_events


DESIRES_0 = {
    "senseofsuperiority": 1,
    "senseofachievement": 4,
    "confidence": 3,
    "joyfulness": 6,
    "comfort": 5,
    "recognition": 2,
    "spiritualsatisfaction": 7,
    "senseofcontrol": 8,
}


def _agent_file(tmp_path: Path, name: str, confidence: int) -> Path:
    payload = {
        "action_sequence": [{"timestamp": "2025-10-02 10:00:00", "action": f"{name} acts."}],
        "expected_values": {**DESIRES_0, "confidence": 7},
        "current_social_personality": "Individualistic",
        "configured_social_personality": "Prosocial",
        "svo_tracker": {"0": 56.24, "1": 33.76},
        "individual_desire": {
            "0": DESIRES_0,
            "1": DESIRES_0,
            "2": {**DESIRES_0, "confidence": confidence},
        },
        "desire_update_tracker": {
            "2": {
                "confidence": {
                    "action": f"{name} acts.",
                    "observation": f"{name}'s action was observed.",
                    "value_before_update": 3,
                    "value_after_update": confidence,
                    "reason": None,
                    "reasonableness": True,
                    "reflection": None,
                    "correction": None,
                    "attempts": [{
                        "attempt": 0,
                        "proposed_value": confidence,
                        "reasonableness": True,
                        "reflection": None,
                    }],
                    "fluctuation": {
                        "before": confidence,
                        "after": confidence,
                        "applied": False,
                    },
                }
            }
        },
        "action_decision_history": [{
            "selected_action": f"{name} acts.",
            "selected_candidate_index": 0,
            "selection_reason": None,
            "observation": "The report is open on the desk.",
            "current_desire_states_raw": "Alice's confidence: moderately confident",
            "current_desires": DESIRES_0,
            "current_svo": 56.24,
            "configured_social_personality": "Prosocial",
            "svo_instruction": "Choose actions that promote cooperation.",
            "svo_regulation_prompt": None,
            "candidates": [{
                "action": f"{name} acts.",
                "predicted_outcome": None,
                "predicted_desires": {"confidence": "more confident"},
                "prediction_raw": "confidence: more confident",
                "prediction_format": "qualitative_text",
                "reason": None,
                "utility": None,
                "self_satisfaction": None,
                "svo_alignment": None,
                "resulting_svo": None,
                "selected": True,
            }],
        }],
    }
    path = tmp_path / f"run_ASVO_{name}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_build_replay_calculates_previous_step_delta(tmp_path: Path) -> None:
    files = [_agent_file(tmp_path, "Alice", 5), _agent_file(tmp_path, "Bob", 1)]
    output = tmp_path / "replay.json"
    replay = build_replay_from_saved_results(files, output_path=output)

    assert replay["schema_version"] == 7
    assert len(replay["frames"]) == 3
    assert replay["frames"][0]["agents"][0]["action"] is None
    assert replay["frames"][0]["agents"][0]["social_personality"] == "Prosocial"
    assert replay["frames"][1]["agents"][0]["social_personality"] == "Prosocial"
    assert replay["frames"][1]["agents"][0]["desires"]["confidence"]["delta"] == 2
    assert replay["frames"][1]["agents"][1]["desires"]["confidence"]["delta"] == -2
    assert replay["frames"][1]["agents"][0]["desires"]["confidence"]["target"] == 7
    assert replay["frames"][1]["agents"][0]["desires"]["confidence"]["gap"] == 2
    assert replay["frames"][1]["agents"][0]["observation"] == "Alice's action was observed."
    assert replay["frames"][1]["agents"][0]["desires"]["confidence"]["reason"] is None
    assert replay["frames"][1]["agents"][0]["desires"]["confidence"]["reasonableness"] is True
    decision = replay["frames"][1]["agents"][0]["action_decision"]
    assert decision["selected_action"] == "Alice acts."
    assert decision["selection_reason"] is None
    assert decision["current_desire_states_raw"].startswith("Alice's confidence")
    assert decision["candidates"][0]["predicted_outcome"] is None
    assert decision["candidates"][0]["predicted_desires"] == {
        "confidence": "more confident"
    }
    assert decision["candidates"][0]["utility"] is None
    assert decision["candidates"][0]["svo_alignment"] is None
    assert replay["frames"][2]["agents"][0]["desires"]["confidence"]["after"] is None
    assert "主な変化" in replay["frames"][1]["summary_ja"]
    assert json.loads(output.read_text(encoding="utf-8")) == replay


def test_render_replay_html_embeds_data_and_controls(tmp_path: Path) -> None:
    replay = build_replay_from_saved_results([_agent_file(tmp_path, "Alice", 4)])
    output = render_replay_html(replay, tmp_path / "viewer.html")
    html = output.read_text(encoding="utf-8")

    assert 'id="replay-data"' in html
    assert 'id="timeline"' in html
    assert "Alice acts." in html
    assert "前へ" in html
    assert 'id="language"' in html
    assert 'id="stepSummary"' in html
    assert "【初期状況】" in html
    assert "Number(frame.step)===0&&present(frame.summary_ja)" in html
    assert "【実際に起きたこと（Event statement）】" in html
    assert "【欲求状態】" in html
    assert "【シミュレーション結果サマリー】" in html
    assert "① 全体の協働" in html
    assert "② 各人の振る舞い" in html
    assert "③ チームワークBig Five" in html
    assert "④ 各人で不足している欲求" in html
    assert "teamwork-table" in html
    assert "teamworkOccurrenceRates" in html
    assert "5要素の割合の合計は100%にならない場合があります" in html
    assert "radar-chart" not in html
    assert "renderTeamworkRadar" not in html
    assert "percentagesSummingTo100" not in html
    assert "normalizedRadarValues" not in html
    assert "判定根拠を見る" not in html
    assert "【試みた行動】" not in html
    assert "行動選択の詳細を見る" not in html
    assert "行動後の観測を見る" not in html
    assert "desire-bar" in html
    assert "desire-target" in html
    assert "時間経過による変動" in html
    assert "Reflection・修正・再試行を見る" in html
    assert 'id="summary"' not in html
    assert "理由は記録されていない" not in html
    assert "計算していない" not in html


def test_template_summarizer_handles_initial_frame() -> None:
    summary = TemplateJapaneseSummarizer().summarize(
        {"step": 0, "agents": [{"name": "Alice"}, {"name": "Bob"}]}
    )
    assert summary == "Alice、Bobの行動前の欲求状態です。"


def test_live_recorder_writes_each_captured_step(tmp_path: Path) -> None:
    class FakeTracker:
        def get_action_sequence(self):
            return [
                {"action": "Alice reviews the report."},
                {"action": "Alice checks the slides."},
            ]

        def get_individual_desire_tracker(self):
            return {
                0: DESIRES_0,
                1: DESIRES_0,
                2: {**DESIRES_0, "recognition": 5},
            }

        def get_desire_update_tracker(self):
            return {2: {"recognition": {
                "action": "Alice reviews the report.",
                "observation": "The corrections were recorded.",
                "value_before_update": 2,
                "value_after_update": 5,
                "reason": None,
                "reasonableness": True,
                "reflection": None,
                "correction": None,
                "attempts": [],
                "fluctuation": {"before": 5, "after": 5, "applied": False},
            }}}

        def get_expected_value_dict(self):
            return {**DESIRES_0, "recognition": 8}

        def get_configured_social_personality(self):
            return "Prosocial"

    class FakePlayer:
        name = "Alice"

        def get_component(self, name):
            assert name == "ValueTracker"
            return FakeTracker()

    output = tmp_path / "live_replay.json"
    recorder = ReplayRecorder(output)
    recorder.capture(step=0, time="10:00", players=[FakePlayer()])
    assert len(json.loads(output.read_text(encoding="utf-8"))["frames"]) == 1

    recorder.capture(
        step=1,
        time="10:20",
        players=[FakePlayer()],
        event="Alice completed a review.",
        events_by_agent={"Alice": "Alice completed a review."},
    )
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert len(saved["frames"]) == 2
    assert saved["frames"][1]["event"] == "Alice completed a review."
    assert saved["frames"][1]["agents"][0]["event"] == "Alice completed a review."
    assert saved["frames"][1]["summary_ja"] is None
    assert saved["frames"][1]["agents"][0]["desires"]["recognition"]["delta"] is None

    recorder.capture(
        step=2,
        time="10:40",
        players=[FakePlayer()],
        event="Alice checked the slides.",
        events_by_agent={"Alice": "Alice checked the slides."},
    )
    recorder.finalize()
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["frames"][1]["agents"][0]["desires"]["recognition"]["delta"] == 3
    assert saved["frames"][1]["agents"][0]["desires"]["recognition"]["target"] == 8
    assert saved["frames"][1]["agents"][0]["desires"]["recognition"]["gap"] == 3
    assert saved["frames"][1]["agents"][0]["social_personality"] == "Prosocial"
    assert saved["frames"][1]["agents"][0]["observation"] == "The corrections were recorded."
    assert saved["frames"][1]["agents"][0]["desires"]["recognition"]["reason"] is None
    assert saved["frames"][2]["agents"][0]["desires"]["recognition"]["after"] is None
    shortages = saved["result_summary"]["agents"][0]["insufficient_desires"]
    assert any(item["key"] == "recognition" for item in shortages)


def test_result_summary_uses_events_and_calculates_shortages() -> None:
    class FakeModel:
        calls = []

        def sample_text(self, prompt, **kwargs):
            self.calls.append(prompt)
            assert "内面的動機" in prompt
            assert "Social Personality" in prompt
            if "【全体としてどんな協働になったか】" in prompt:
                assert "1文目: シミュレーション全体で最も目立った協働パターン" in prompt
                assert "Agent同士の関わり方" in prompt
                assert "Alice completed the report review." in prompt
                assert "Bob checked the slides." in prompt
                assert "Charlie organized the files." in prompt
                return "3人は確認と整理を分担しました。"
            if "【各Agentがどう振る舞ったか】" in prompt:
                assert "1文目: 今回のシミュレーションで最も目立った役割" in prompt
                assert "具体的な作業名の一覧ではなく" in prompt
                assert "Alice completed the report review." in prompt
                assert "Bob checked the slides." in prompt
                assert "Charlie organized the files." in prompt
                return json.dumps({
                    "Alice": "報告書を確認する振る舞いが見られました。",
                    "Bob": "スライドを確認する振る舞いが見られました。",
                    "Charlie": "ファイルを整理する振る舞いが見られました。",
                }, ensure_ascii=False)
            raise AssertionError("unexpected prompt")

    frames = [{
        "step": 1,
        "agents": [
            {
                "name": "Alice",
                "event": "Alice completed the report review.",
                "desires": {
                    "confidence": {"value": 3, "target": 5},
                    "comfort": {"value": 6, "target": 6},
                },
            },
            {
                "name": "Bob",
                "event": "Bob checked the slides.",
                "desires": {
                    "confidence": {"value": 7, "target": 5},
                },
            },
            {
                "name": "Charlie",
                "event": "Charlie organized the files.",
                "desires": {
                    "confidence": {"value": 5, "target": 5},
                },
            },
        ],
    }]
    model = FakeModel()
    summary = build_result_summary(
        frames, LLMJapaneseResultSummarizer(model)
    )

    assert summary["source"] == "event_statements"
    assert summary["collaboration_ja"].startswith("3人は")
    assert len(model.calls) == 2
    assert summary["agents"][0]["behavior_ja"].startswith("報告書を確認")
    assert summary["agents"][0]["insufficient_desires"] == [{
        "key": "confidence",
        "label_ja": "自信",
        "current": 3,
        "expected": 5,
        "gap": 2.0,
    }]
    assert summary["agents"][1]["insufficient_desires"] == []
    assert summary["agents"][2]["insufficient_desires"] == []


def test_summarize_replay_file_updates_only_summary_and_html(tmp_path: Path) -> None:
    class FakeSummarizer:
        def summarize(self, events, agent_names):
            return {
                "collaboration_ja": "共同で確認しました。",
                "agent_behavior_ja": {"Alice": "報告書を確認しました。"},
            }

    frames = [{
        "step": 1,
        "time": "10:20",
        "event": "Alice reviewed the report.",
        "agents": [{
            "name": "Alice",
            "event": "Alice reviewed the report.",
            "desires": {"confidence": {"value": 3, "target": 5}},
        }],
    }]
    path = tmp_path / "replay.json"
    path.write_text(json.dumps({
        "schema_version": 5,
        "simulation": {"title": "Keep me"},
        "frames": frames,
        "custom_field": {"preserved": True},
    }), encoding="utf-8")

    json_path, html_path = summarize_replay_file(path, FakeSummarizer())
    updated = json.loads(json_path.read_text(encoding="utf-8"))

    assert updated["frames"] == frames
    assert updated["simulation"] == {"title": "Keep me"}
    assert updated["custom_field"] == {"preserved": True}
    assert updated["result_summary"]["collaboration_ja"] == "共同で確認しました。"
    assert updated["schema_version"] == 7
    assert html_path.exists()
    assert "【シミュレーション結果サマリー】" in html_path.read_text(
        encoding="utf-8"
    )


def test_event_statement_translation_is_exactly_post_processed(tmp_path: Path) -> None:
    class FakeModel:
        calls = 0

        def sample_text(self, prompt, **kwargs):
            self.calls += 1
            assert "Event statements" in prompt
            assert "agent:Alice:event" in prompt
            assert "observation" not in prompt.lower()
            return '{"agent:Alice:event":"Aliceはレビューを完了しました。"}'

    class FakeTracker:
        def get_action_sequence(self):
            return [{"action": "Alice reviews the report."}]

        def get_individual_desire_tracker(self):
            return {0: DESIRES_0, 1: DESIRES_0}

        def get_desire_update_tracker(self):
            return {}

        def get_expected_value_dict(self):
            return DESIRES_0

    class FakePlayer:
        name = "Alice"

        def get_component(self, name):
            return FakeTracker()

    model = FakeModel()
    recorder = ReplayRecorder(
        tmp_path / "replay.json",
        field_translator=LLMJapaneseReplayFieldTranslator(model),
    )
    recorder.capture(step=0, time="10:00", players=[FakePlayer()])
    recorder.capture(
        step=1,
        time="10:20",
        players=[FakePlayer()],
        event="Alice completed a review.",
        events_by_agent={"Alice": "Alice completed a review."},
    )
    replay = recorder.finalize()

    frame = replay["frames"][1]
    assert model.calls == 1
    assert frame["event_ja"] is None
    assert frame["agents"][0]["event"] == "Alice completed a review."
    assert frame["agents"][0]["event_ja"] == "Aliceはレビューを完了しました。"
    assert "observation_ja" not in frame["agents"][0]


def test_event_statement_big_five_classification_is_saved(tmp_path: Path) -> None:
    class FakeModel:
        calls = 0

        def sample_text(self, prompt, **kwargs):
            self.calls += 1
            assert "the focal agent's concrete action, its identifiable target" in prompt
            assert "No vector, labels, or Markdown" in prompt
            assert "The deadline was moved forward." in prompt
            assert "reporting a group decision is 0" in prompt
            assert "managing speaking order" in prompt
            assert "teammate-owned task" in prompt
            assert "Generic offers of help" in prompt
            assert "Routine editing or refinement" in prompt
            assert "Generic invitations for feedback" in prompt
            assert len(prompt) < 5000
            assert kwargs["temperature"] == 0.0
            return json.dumps({
                "Alice": {
                    "team_leadership": {
                        "value": 1,
                        "evidence": "creates a task checklist",
                    },
                    "mutual_performance_monitoring": {
                        "value": 0,
                        "evidence": None,
                    },
                    "backup_behavior": {
                        "value": 1,
                        "evidence": "offers to handle remaining corrections",
                    },
                    "adaptability": {"value": 0, "evidence": None},
                    "team_orientation": {
                        "value": 1,
                        "evidence": "so the team meets the deadline",
                    },
                }
            })

    class FakeTracker:
        def get_action_sequence(self):
            return [{"action": "Alice coordinates the work."}]

        def get_individual_desire_tracker(self):
            return {0: DESIRES_0, 1: DESIRES_0}

        def get_desire_update_tracker(self):
            return {}

        def get_expected_value_dict(self):
            return DESIRES_0

    class FakePlayer:
        name = "Alice"

        def get_component(self, name):
            assert name == "ValueTracker"
            return FakeTracker()

    event = (
        "Alice creates a task checklist, offers to handle remaining "
        "corrections so the team meets the deadline."
    )
    model = FakeModel()
    recorder = ReplayRecorder(
        tmp_path / "replay.json",
        teamwork_classifier=LLMTeamworkClassifier(model),
        shared_memories=["The deadline was moved forward."],
    )
    recorder.capture(step=0, time="10:00", players=[FakePlayer()])
    recorder.capture(
        step=1,
        time="10:20",
        players=[FakePlayer()],
        event=event,
        events_by_agent={"Alice": event},
    )
    replay = recorder.finalize()

    result = replay["frames"][1]["agents"][0]["teamwork_big_five"]
    assert model.calls == 1
    assert replay["teamwork_big_five"]["vector_order"] == [
        "team_leadership",
        "mutual_performance_monitoring",
        "backup_behavior",
        "adaptability",
        "team_orientation",
    ]
    assert result["vector"] == [1, 0, 1, 0, 1]
    assert result["labels"] == [
        "team_leadership",
        "backup_behavior",
        "team_orientation",
    ]
    assert result["classifications"]["team_leadership"] == {
        "value": 1,
        "evidence": "creates a task checklist",
    }


def test_teamwork_rubric_versions_keep_old_experiments_stable() -> None:
    class FakeModel:
        def __init__(self):
            self.prompts = []

        def sample_text(self, prompt, **kwargs):
            self.prompts.append(prompt)
            return json.dumps({"P01": {
                key: {"value": 0, "evidence": None}
                for key in TEAMWORK_BIG_FIVE_VECTOR_ORDER
            }})

    model = FakeModel()
    for version in (1, 2, 3, 4, 5, 6, 7):
        LLMTeamworkClassifier(model, rubric_version=version).classify(
            {"P01": "P01 adjusted the speaking order."}, step=1)
    old, revised, strict, action, focused, clarified, adaptive = model.prompts
    assert "Routine facilitation" in old
    assert "managing speaking order is 0" in revised
    assert "target customer and beer concept" not in revised
    assert len(revised) < len(old)
    assert "reviewing a shared output with colleagues" in strict
    assert "merely hearing views" in strict
    assert "Generic invitations for feedback" not in revised
    assert "what each focal_agent does in its Action statement" in action
    assert '"agent_action"' in action
    assert "later succeeded" not in action
    assert "outcome" not in action.lower()
    assert "result" not in action.lower()
    assert "personal interest" not in action
    assert "Merely polishing wording or adding details is 0" in action
    assert 'Example 0: "asks teammates to flag any concerns"' in focused
    assert 'Example 1: "asks each teammate to compare options A and B' in focused
    assert '"agent_action"' in focused
    assert '"event_statement"' not in focused
    assert 'Example 0:' not in action
    assert "Proposing a specific decision procedure counts when the same Action" in clarified
    assert "Task-specific feedback on an identifiable teammate's contribution counts as help" in clarified
    assert "Proposing a specific decision procedure counts" not in focused
    assert '"agent_action"' in clarified
    assert 'immediately preceding confirmed Event' in adaptive
    assert 'do not infer it from timing alone' in adaptive
    assert 'Choosing or combining the initial options' in adaptive
    assert 'immediately preceding confirmed Event' not in clarified


def test_adaptability_uses_prior_event_only_in_new_rubric() -> None:
    class FakeModel:
        def __init__(self):
            self.prompts = []

        def sample_text(self, prompt, **kwargs):
            self.prompts.append(prompt)
            return json.dumps({"P07": {
                key: {"value": int(key == "adaptability"),
                      "evidence": "reduces the hop aroma" if key == "adaptability" else None}
                for key in TEAMWORK_BIG_FIVE_VECTOR_ORDER
            }})

    model = FakeModel()
    frame = {"step": 2, "agents": [{"name": "P07",
              "action": "P07 reduces the hop aroma in the agreed concept.",
              "event": "P07 talked with the team."}]}
    trigger = "P19 raised a production concern about strong hop aroma."
    new = LLMTeamworkClassifier(model, rubric_version=7)
    assert _classify_recorded_events(
        frame, new, source_field="action", previous_event=trigger) is None
    assert trigger in model.prompts[-1]
    assert frame["agents"][0]["teamwork_big_five"]["classifications"]["adaptability"]["value"] == 1
    old = LLMTeamworkClassifier(model, rubric_version=6)
    assert _classify_recorded_events(
        frame, old, source_field="action", previous_event=trigger) is None
    assert trigger not in model.prompts[-1]


def test_replay_finalize_passes_only_immediately_previous_event(tmp_path: Path) -> None:
    class FakeModel:
        def __init__(self):
            self.prompts = []

        def sample_text(self, prompt, **kwargs):
            self.prompts.append(prompt)
            return json.dumps({"P07": {
                key: {"value": 0, "evidence": None}
                for key in TEAMWORK_BIG_FIVE_VECTOR_ORDER
            }})

    class FakeSummarizer:
        def summarize(self, frame):
            return "summary"

    model = FakeModel()
    recorder = ReplayRecorder(
        tmp_path / "replay.json", summarizer=FakeSummarizer(),
        teamwork_classifier=LLMTeamworkClassifier(model, rubric_version=7),
        teamwork_source="action")
    recorder.document["frames"] = [
        {"step": 0, "agents": [{"name": "P07", "action": None,
                                 "event": None, "desires": {}}],
         "event": None, "summary_ja": "summary"},
        {"step": 1, "agents": [{"name": "P07", "action": "P07 compares proposals.",
                                 "event": "P07 compares proposals.", "desires": {}}],
         "event": "P19 raised a production concern.", "summary_ja": "summary"},
        {"step": 2, "agents": [{"name": "P07", "action": "P07 reduces the hop aroma.",
                                 "event": "P07 revises the plan.", "desires": {}}],
         "event": "P07 revises the plan.", "summary_ja": "summary"},
    ]
    recorder.finalize()
    assert len(model.prompts) == 2
    assert "P19 raised a production concern." not in model.prompts[0]
    assert "P19 raised a production concern." in model.prompts[1]
    assert '"agent_action": "P07 reduces the hop aroma.' in model.prompts[1]


def test_action_classification_reads_action_not_event(tmp_path: Path) -> None:
    class FakeModel:
        def sample_text(self, prompt, **kwargs):
            assert "P01 offers to help P02 with costs" in prompt
            assert "P01 spent the time alone" not in prompt
            return json.dumps({"P01": {
                key: {"value": int(key == "backup_behavior"),
                      "evidence": "offers to help P02 with costs" if key == "backup_behavior" else None}
                for key in TEAMWORK_BIG_FIVE_VECTOR_ORDER
            }})

    classifier = LLMTeamworkClassifier(FakeModel(), rubric_version=6)
    recorder = ReplayRecorder(
        tmp_path / "action_replay.json", teamwork_classifier=classifier,
        teamwork_source="action")
    assert recorder.document["teamwork_big_five"]["source"] == "action"
    frame = {"step": 1, "agents": [{
        "name": "P01", "action": "P01 offers to help P02 with costs.",
        "event": "P01 spent the time alone."}]}
    assert _classify_recorded_events(
        frame, classifier,
        source_field="action") is None
    assert frame["agents"][0]["teamwork_big_five"]["vector"] == [0, 0, 1, 0, 0]


def test_big_five_classification_failure_is_recorded(tmp_path: Path) -> None:
    class BrokenClassifier:
        def classify(self, events_by_agent, *, step, scenario_context=()):
            raise RuntimeError("offline")

    class FakeTracker:
        def get_action_sequence(self):
            return [{"action": "Alice acts."}]

        def get_individual_desire_tracker(self):
            return {0: DESIRES_0, 1: DESIRES_0}

        def get_desire_update_tracker(self):
            return {}

        def get_expected_value_dict(self):
            return DESIRES_0

    class FakePlayer:
        name = "Alice"

        def get_component(self, name):
            return FakeTracker()

    recorder = ReplayRecorder(
        tmp_path / "replay.json", teamwork_classifier=BrokenClassifier()
    )
    recorder.capture(step=0, time="10:00", players=[FakePlayer()])
    recorder.capture(
        step=1,
        time="10:20",
        players=[FakePlayer()],
        events_by_agent={"Alice": "Alice acts."},
    )
    replay = recorder.finalize()

    frame = replay["frames"][1]
    assert frame["teamwork_classification_error"] == "RuntimeError: offline"
    assert frame["agents"][0]["teamwork_big_five"] is None


def test_big_five_paraphrased_evidence_does_not_block_classification() -> None:
    class FakeModel:
        def sample_text(self, prompt, **kwargs):
            return json.dumps({
                "Alice": {
                    "team_leadership": {
                        "value": 1,
                        "evidence": "coordinates the team's checklist",
                    },
                    "mutual_performance_monitoring": {
                        "value": 0,
                        "evidence": None,
                    },
                    "backup_behavior": {"value": 0, "evidence": None},
                    "adaptability": {"value": 0, "evidence": None},
                    "team_orientation": {"value": 0, "evidence": None},
                }
            })

    result = LLMTeamworkClassifier(FakeModel()).classify(
        {"Alice": "Alice creates and shares a task checklist."}, step=1
    )["Alice"]

    assert result["vector"] == [1, 0, 0, 0, 0]
    assert result["classifications"]["team_leadership"] == {
        "value": 1,
        "evidence": "coordinates the team's checklist",
        "evidence_is_exact": False,
    }


def test_llm_translator_batches_agents_and_stores_original(tmp_path: Path) -> None:
    class FakeModel:
        calls = 0

        def sample_text(self, prompt, **kwargs):
            self.calls += 1
            assert '"Alice": "Alice acts."' in prompt
            assert '"Bob": "Bob acts."' in prompt
            return '{"Alice":"Aliceは行動します。","Bob":"Bobは行動します。"}'

    files = [_agent_file(tmp_path, "Alice", 5), _agent_file(tmp_path, "Bob", 1)]
    model = FakeModel()
    replay = build_replay_from_saved_results(
        files, action_translator=LLMJapaneseActionTranslator(model)
    )

    assert model.calls == 1
    alice = replay["frames"][1]["agents"][0]
    assert alice["action"] == "Alice acts."
    assert alice["action_ja"] == "Aliceは行動します。"


def test_translation_failure_does_not_stop_replay(tmp_path: Path) -> None:
    class BrokenTranslator:
        def translate(self, actions, *, step):
            raise RuntimeError("offline")

    replay = build_replay_from_saved_results(
        [_agent_file(tmp_path, "Alice", 5)], action_translator=BrokenTranslator()
    )
    frame = replay["frames"][1]
    assert frame["agents"][0]["action_ja"] is None
    assert frame["translation_error"] == "RuntimeError: offline"


def test_step_zero_uses_japanese_shared_memory_summary(tmp_path: Path) -> None:
    class FakeModel:
        def sample_text(self, prompt, **kwargs):
            assert "週末に3人がオフィスで働いている" in prompt
            return "週末のオフィスで、3人が急な仕事に対応しています。"

    class FakeTracker:
        def get_action_sequence(self):
            return []

        def get_individual_desire_tracker(self):
            return {0: DESIRES_0}

        def get_expected_value_dict(self):
            return DESIRES_0

    class FakePlayer:
        name = "Alice"

        def get_component(self, name):
            return FakeTracker()

    recorder = ReplayRecorder(
        tmp_path / "replay.json",
        shared_memories=["週末に3人がオフィスで働いている。"],
        initial_context_summarizer=LLMJapaneseSharedMemoriesSummarizer(FakeModel()),
    )
    frame = recorder.capture(step=0, time="10:00", players=[FakePlayer()])

    assert frame["summary_ja"] == "週末のオフィスで、3人が急な仕事に対応しています。"
    assert recorder.document["simulation"]["shared_memories"] == [
        "週末に3人がオフィスで働いている。"
    ]


def test_initial_summary_failure_falls_back_without_stopping(tmp_path: Path) -> None:
    class BrokenSummarizer:
        def summarize(self, shared_memories):
            raise RuntimeError("offline")

    class FakeTracker:
        def get_action_sequence(self):
            return []

        def get_individual_desire_tracker(self):
            return {0: DESIRES_0}

        def get_expected_value_dict(self):
            return DESIRES_0

    class FakePlayer:
        name = "Alice"

        def get_component(self, name):
            return FakeTracker()

    recorder = ReplayRecorder(
        tmp_path / "replay.json",
        shared_memories=["context"],
        initial_context_summarizer=BrokenSummarizer(),
    )
    frame = recorder.capture(step=0, time="10:00", players=[FakePlayer()])
    assert frame["summary_ja"] == "Aliceの行動前の欲求状態です。"
    assert frame["initial_summary_error"] == "RuntimeError: offline"


def test_llm_step_summary_uses_translated_actions_and_changes() -> None:
    class FakeModel:
        def sample_text(self, prompt, **kwargs):
            assert "Aliceは報告書を確認しました。" in prompt
            assert "快適さ" in prompt
            assert '"reason": null' in prompt
            assert "reasonがnullなら" in prompt
            return "Aliceが報告書を確認しました。チームが修正点を共有したことが記録されました。Aliceの快適さの値は低下しました。"

    frame = {
        "step": 1,
        "time": "10:20",
        "event": "The team shared corrections.",
        "agents": [{
            "name": "Alice",
            "action": "Alice reviewed the report.",
            "action_ja": "Aliceは報告書を確認しました。",
            "observation": "The corrections were shared.",
            "desires": {"comfort": {
                "before": 5,
                "after": 3,
                "delta": -2,
                "reason": None,
                "reasonableness": True,
            }},
        }],
    }
    summary = LLMJapaneseStepSummarizer(FakeModel()).summarize(frame)
    assert summary.startswith("Aliceが報告書を確認し")


def test_llm_step_summary_falls_back_to_template() -> None:
    class BrokenModel:
        def sample_text(self, prompt, **kwargs):
            raise RuntimeError("offline")

    frame = {
        "step": 1,
        "agents": [{
            "name": "Alice",
            "action": "Alice acts.",
            "action_ja": None,
            "observation": None,
            "desires": {"comfort": {
                "before": 5,
                "after": 3,
                "delta": -2,
                "reason": None,
            }},
        }],
    }
    summary = LLMJapaneseStepSummarizer(BrokenModel()).summarize(frame)
    assert "Alice: Alice acts." in summary
    assert "Aliceの快適さの値が低下" in summary
