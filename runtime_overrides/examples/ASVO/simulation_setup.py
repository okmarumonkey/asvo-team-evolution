"""Compatibility module exposing values loaded from simulation_config.json."""

from __future__ import annotations

import datetime
import json
import os
from pathlib import Path

import sentence_transformers

from concordia.language_model import my_utils as utils
from team_evolution.config_loader import (
    load_config,
    role_expected_values,
    sample_initial_desire_value as _sample_initial_desire_value,
)


CONFIG = load_config()
CONFIG_PATH = Path(CONFIG["_config_path"])

# Kept for compatibility with the original notebook.
ROOT = ""

runtime = CONFIG["runtime"]
episode_length = runtime["episode_length"]
NUM_PLAYERS = runtime["num_players"]
tested_agents = runtime["tested_agents"]
random_seed = runtime.get("random_seed")


model_config = CONFIG["model"]
disable_language_model = model_config["disable_language_model"]
api_type = model_config["api_type"]
model_name = model_config["model_name"]
device = model_config.get("device", "cpu")
api_key_env = model_config.get("api_key_env")
api_key = os.environ.get(api_key_env) if api_key_env else None
# if not disable_language_model and api_key_env and not api_key:
#   raise RuntimeError(
#       f"Environment variable {api_key_env!r} is required by {CONFIG_PATH}. "
#       "Set it before starting Jupyter; do not store API keys in JSON."
#   )

embedding_model_name = CONFIG["embedding"]["model_name"]
st_model = sentence_transformers.SentenceTransformer(embedding_model_name)
embedder = lambda text: st_model.encode(text, show_progress_bar=False)

model = utils.language_model_setup(
    api_type=api_type,
    model_name=model_name,
    api_key=api_key,
    device=device if api_type == "pytorch_gemma" else None,
    disable_language_model=disable_language_model,
)

profile_config = CONFIG["profile"]
wanted_desires = profile_config["wanted_desires"]
hidden_desires = profile_config["hidden_desires"]
initial_value_distribution = profile_config["initial_value_distribution"]


def sample_initial_desire_value() -> int:
  return _sample_initial_desire_value(initial_value_distribution)


Use_Previous_profile = profile_config.get("use_previous", False)
previous_profile_file = profile_config.get("previous_profile_file")
previous_profile = None
if Use_Previous_profile:
  if not previous_profile_file:
    raise ValueError("profile.previous_profile_file is required when use_previous is true")
  profile_path = Path(previous_profile_file).expanduser()
  if not profile_path.is_absolute():
    profile_path = CONFIG_PATH.parent / profile_path
  try:
    previous_profile = json.loads(profile_path.read_text(encoding="utf-8"))
  except FileNotFoundError as exc:
    raise ValueError(f"Previous profile file not found: {profile_path}") from exc
  previous_profile_file = str(profile_path.resolve())

output_folder = Path(CONFIG["output"].get("result_folder", "result_folder"))
if not output_folder.is_absolute():
  output_folder = CONFIG_PATH.parent / output_folder
current_folder_path = str(output_folder.resolve())
os.makedirs(current_folder_path, exist_ok=True)

replay_config = CONFIG.get("replay", {})
TRANSLATE_ACTIONS_JA = replay_config.get("translate_actions_ja", False)
TRANSLATE_EVENTS_JA = replay_config.get(
    "translate_events_ja", TRANSLATE_ACTIONS_JA
)
SUMMARIZE_SHARED_MEMORIES_JA = replay_config.get(
    "summarize_shared_memories_ja", False
)
SUMMARIZE_STEPS_JA = replay_config.get("summarize_steps_ja", False)
SUMMARIZE_RESULT_JA = replay_config.get("summarize_result_ja", True)
CLASSIFY_TEAMWORK_BIG_FIVE = replay_config.get(
    "classify_teamwork_big_five", True
)

clock_config = CONFIG["clock"]
SETUP_TIME = datetime.datetime.fromisoformat(clock_config["setup_time"])
START_TIME = datetime.datetime.fromisoformat(clock_config["start_time"])
MAJOR_STEP = datetime.timedelta(minutes=clock_config.get("major_step_minutes", 20))
MINOR_STEP = datetime.timedelta(seconds=clock_config.get("minor_step_seconds", 10))
