"""Compatibility view of the scenario loaded from simulation_config.json."""

from team_evolution.config_loader import legacy_scenario, load_config


ENV_SETTING = legacy_scenario(load_config())
