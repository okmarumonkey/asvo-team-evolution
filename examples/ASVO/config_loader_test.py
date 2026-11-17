"""Tests for JSON simulation configuration loading."""

import json
import random

import pytest

from examples.ASVO import config_loader


def test_default_config_loads_and_converts_scenario():
  config = config_loader.load_config(config_loader.DEFAULT_CONFIG_PATH)
  scenario = config_loader.legacy_scenario(config)

  assert config["runtime"]["num_players"] == 3
  assert scenario["roles"][0]["name"] == "P01"
  assert len(scenario["player observe"]) == 3
  assert len(scenario["gm observe"]) == 3
  assert scenario["roles"][0]["goal"].startswith("Remain in the conference room")
  assert scenario["roles"][0]["specific_memories"] == [
      "The current task takes place in a company conference room, where the team is planning a new beer product for adult consumers.",
      "The team must agree on one coherent product concept and complete the proposal before the noon deadline.",
      "All three employees remain in the conference room with access to the shared research, cost estimates, brewing notes, and presentation materials.",
  ]
  alice = scenario["roles"][0]
  expected = config_loader.role_expected_values(alice)
  assert expected["sense of control"] == alice["personality"]["possessive"]
  assert expected["sense of superiority"] == alice["personality"]["competitive"]
  assert expected["spiritual satisfaction"] == alice["personality"]["spiritual"]


def test_observations_must_be_empty_or_match_roles(tmp_path):
  config = config_loader.load_config(config_loader.DEFAULT_CONFIG_PATH)
  config.pop("_config_path")
  config["scenario"]["player_observe"] = ["only one"]
  path = tmp_path / "invalid_observations.json"
  path.write_text(json.dumps(config), encoding="utf-8")

  with pytest.raises(ValueError, match="empty or contain one entry per role"):
    config_loader.load_config(path)


def test_role_personality_values_must_be_in_range(tmp_path):
  config = config_loader.load_config(config_loader.DEFAULT_CONFIG_PATH)
  config.pop("_config_path")
  config["scenario"]["roles"][0]["personality"]["confidence"] = 11
  path = tmp_path / "invalid_personality.json"
  path.write_text(json.dumps(config), encoding="utf-8")

  with pytest.raises(ValueError, match="integer from 0 to 10"):
    config_loader.load_config(path)


def test_initial_desire_normal_distribution_is_centered_and_bounded():
  config = config_loader.load_config(config_loader.DEFAULT_CONFIG_PATH)
  distribution = config["profile"]["initial_value_distribution"]
  rng = random.Random(1234)
  samples = [
      config_loader.sample_initial_desire_value(distribution, rng)
      for _ in range(20_000)
  ]

  assert all(isinstance(value, int) and 0 <= value <= 10 for value in samples)
  assert abs(sum(samples) / len(samples) - 5) < 0.1
  assert samples.count(5) > samples.count(0)


def test_invalid_player_count_has_clear_error(tmp_path):
  config = config_loader.load_config(config_loader.DEFAULT_CONFIG_PATH)
  config.pop("_config_path")
  config["runtime"]["num_players"] = 99
  path = tmp_path / "invalid.json"
  path.write_text(json.dumps(config), encoding="utf-8")

  with pytest.raises(ValueError, match="cannot exceed"):
    config_loader.load_config(path)


def test_invalid_json_reports_location(tmp_path):
  path = tmp_path / "broken.json"
  path.write_text('{"runtime": }', encoding="utf-8")

  with pytest.raises(ValueError, match=r"line 1, column"):
    config_loader.load_config(path)
