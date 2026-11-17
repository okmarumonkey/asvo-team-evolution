"""Load and validate ASVO simulation settings from JSON."""

from __future__ import annotations

import copy
import datetime
import json
import os
import random
from pathlib import Path
from typing import Any


DEFAULT_CONFIG_PATH = Path(__file__).with_name("simulation_config.json")
VALID_AGENTS = {"ASVO", "ReAct", "BabyAGI", "LLMob", "D2A", "D2A_without_SVO"}
VALID_SOCIAL_PERSONALITIES = {
    "Altruistic", "Prosocial", "Individualistic", "Competitive"
}
PERSONALITY_TO_DESIRE = {
    "possessive": "sense of control",
    "confidence": "confidence",
    "ambitious": "sense_of_achievement",
    "relaxed": "comfort",
    "attention_seeking": "recognition",
    "competitive": "sense of superiority",
    "cheerful": "joyfulness",
    "spiritual": "spiritual satisfaction",
}


def resolve_config_path(path: str | os.PathLike[str] | None = None) -> Path:
  """Resolve an explicit path, ASVO_CONFIG, or the repository default."""
  selected = path or os.environ.get("ASVO_CONFIG")
  if selected:
    candidate = Path(selected).expanduser()
    if not candidate.is_absolute():
      candidate = Path.cwd() / candidate
    return candidate.resolve()
  return DEFAULT_CONFIG_PATH.resolve()


def _require(mapping: dict[str, Any], key: str, location: str) -> Any:
  if key not in mapping:
    raise ValueError(f"Missing required setting: {location}.{key}")
  return mapping[key]


def _expanded_roles(scenario: dict[str, Any]) -> list[dict[str, Any]]:
  """Apply common role defaults while preserving explicit role overrides."""
  defaults = scenario.get("role_defaults", {})
  roles = copy.deepcopy(scenario.get("roles", []))
  for role in roles:
    role.setdefault("goal", defaults.get("goal", ""))
    role.setdefault(
        "specific_memories", copy.deepcopy(defaults.get("specific_memories", []))
    )
  return roles


def _expanded_observations(
    scenario: dict[str, Any], key: str, role_count: int
) -> list[str]:
  observations = copy.deepcopy(scenario.get(key, []))
  if observations:
    return observations
  common = scenario.get("role_defaults", {}).get(key)
  return [common] * role_count if common else []


def _validate(config: dict[str, Any]) -> None:
  runtime = _require(config, "runtime", "config")
  scenario = _require(config, "scenario", "config")
  model = _require(config, "model", "config")

  if not isinstance(runtime.get("episode_length"), int) or runtime["episode_length"] < 1:
    raise ValueError("runtime.episode_length must be an integer greater than 0")
  if not isinstance(runtime.get("num_players"), int) or runtime["num_players"] < 1:
    raise ValueError("runtime.num_players must be an integer greater than 0")
  unknown_agents = set(runtime.get("tested_agents", [])) - VALID_AGENTS
  if unknown_agents:
    raise ValueError(f"Unknown runtime.tested_agents: {sorted(unknown_agents)}")

  raw_roles = _require(scenario, "roles", "scenario")
  role_defaults = scenario.get("role_defaults", {})
  if not isinstance(role_defaults, dict):
    raise ValueError("scenario.role_defaults must be an object")
  if not isinstance(role_defaults.get("goal", ""), str):
    raise ValueError("scenario.role_defaults.goal must be a string")
  if not isinstance(role_defaults.get("specific_memories", []), list):
    raise ValueError("scenario.role_defaults.specific_memories must be an array")
  roles = _expanded_roles(scenario)
  if runtime["num_players"] > len(roles):
    raise ValueError("runtime.num_players cannot exceed the number of scenario.roles")
  names = [role.get("name") for role in roles]
  if len(names) != len(set(names)):
    raise ValueError("Each scenario role must have a unique name")
  for index, role in enumerate(roles):
    personality = role.get("social_personality")
    if personality not in VALID_SOCIAL_PERSONALITIES:
      raise ValueError(
          f"scenario.roles[{index}].social_personality must be one of "
          f"{sorted(VALID_SOCIAL_PERSONALITIES)}"
      )
    ideal_values = role.get("personality")
    if not isinstance(ideal_values, dict):
      raise ValueError(f"scenario.roles[{index}].personality must be an object")
    missing = set(PERSONALITY_TO_DESIRE) - set(ideal_values)
    unknown = set(ideal_values) - set(PERSONALITY_TO_DESIRE)
    if missing or unknown:
      raise ValueError(
          f"scenario.roles[{index}].personality keys are invalid; "
          f"missing={sorted(missing)}, unknown={sorted(unknown)}"
      )
    for trait, value in ideal_values.items():
      if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 10:
        raise ValueError(
            f"scenario.roles[{index}].personality.{trait} must be an integer from 0 to 10"
        )

  profile = _require(config, "profile", "config")
  distribution = profile.get("initial_value_distribution", {})
  if distribution.get("type", "normal") != "normal":
    raise ValueError("profile.initial_value_distribution.type must be normal")
  for key in ("mean", "stddev", "min", "max"):
    value = distribution.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
      raise ValueError(
          f"profile.initial_value_distribution.{key} must be numeric"
      )
  if distribution["stddev"] <= 0:
    raise ValueError("profile.initial_value_distribution.stddev must be greater than 0")
  if distribution["min"] >= distribution["max"]:
    raise ValueError("profile.initial_value_distribution.min must be less than max")

  observations = _expanded_observations(scenario, "player_observe", len(roles))
  gm_observations = _expanded_observations(scenario, "gm_observe", len(roles))
  for key, values in (
      ("player_observe", observations),
      ("gm_observe", gm_observations),
  ):
    if len(values) not in (0, len(roles)):
      raise ValueError(
          f"scenario.{key} must be empty or contain one entry per role"
      )
  if not model.get("disable_language_model") and not model.get("api_type"):
    raise ValueError("model.api_type is required when the language model is enabled")

  replay = config.get("replay", {})
  if not isinstance(replay.get("translate_actions_ja", False), bool):
    raise ValueError("replay.translate_actions_ja must be true or false")
  if not isinstance(
      replay.get("translate_events_ja", replay.get("translate_actions_ja", False)),
      bool,
  ):
    raise ValueError("replay.translate_events_ja must be true or false")
  if not isinstance(replay.get("summarize_shared_memories_ja", False), bool):
    raise ValueError("replay.summarize_shared_memories_ja must be true or false")
  if not isinstance(replay.get("summarize_steps_ja", False), bool):
    raise ValueError("replay.summarize_steps_ja must be true or false")
  if not isinstance(replay.get("summarize_result_ja", True), bool):
    raise ValueError("replay.summarize_result_ja must be true or false")
  if not isinstance(replay.get("classify_teamwork_big_five", True), bool):
    raise ValueError("replay.classify_teamwork_big_five must be true or false")

  clock = _require(config, "clock", "config")
  try:
    datetime.datetime.fromisoformat(clock["setup_time"])
    datetime.datetime.fromisoformat(clock["start_time"])
  except (KeyError, TypeError, ValueError) as exc:
    raise ValueError("clock.setup_time and start_time must be ISO datetime strings") from exc


def load_config(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
  """Return a validated, independent configuration dictionary."""
  config_path = resolve_config_path(path)
  try:
    config = json.loads(config_path.read_text(encoding="utf-8"))
  except FileNotFoundError as exc:
    raise FileNotFoundError(f"ASVO configuration not found: {config_path}") from exc
  except json.JSONDecodeError as exc:
    raise ValueError(
        f"Invalid JSON in {config_path}, line {exc.lineno}, column {exc.colno}: {exc.msg}"
    ) from exc
  _validate(config)
  config["_config_path"] = str(config_path)
  return copy.deepcopy(config)


def legacy_scenario(config: dict[str, Any]) -> dict[str, Any]:
  """Convert JSON-friendly scenario names to the notebook's legacy shape."""
  scenario = copy.deepcopy(config["scenario"])
  roles = _expanded_roles(scenario)
  player_observe = _expanded_observations(
      scenario, "player_observe", len(roles)
  )
  gm_observe = _expanded_observations(scenario, "gm_observe", len(roles))
  return {
      "shared_memories": scenario["shared_memories"],
      "gm_facts_desc": scenario["gm_facts_desc"],
      # Derived from the role objects so the JSON does not duplicate names.
      "Role Classification": [role["name"] for role in roles],
      "player observe": dict(enumerate(player_observe)),
      "gm observe": dict(enumerate(gm_observe)),
      "roles": roles,
  }


def role_expected_values(role: dict[str, Any]) -> dict[str, int]:
  """Convert a role's personality aliases to ValueTracker desire names."""
  personality = role["personality"]
  return {
      desire: personality[trait]
      for trait, desire in PERSONALITY_TO_DESIRE.items()
  }


def sample_initial_desire_value(
    distribution: dict[str, Any], rng: random.Random | Any = random
) -> int:
  """Sample a rounded normal value and clamp it to the configured range."""
  value = round(rng.gauss(distribution["mean"], distribution["stddev"]))
  return int(max(distribution["min"], min(distribution["max"], value)))
