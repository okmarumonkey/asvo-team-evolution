# ASVO Team Evolution

Search for three-person teams among 20 simulated employees. Each team compares three beer concepts over three meeting steps. A classifier scores five teamwork behaviors in each agent's chosen action. Each positive label earns one point, up to 45 points per meeting.

Each generation keeps the highest-scoring team and tests two new teams formed by crossover, with a 50% chance of replacing a member. Ten generation updates evaluate 23 teams.

## Run

Initialize the pinned upstream checkout and prepare the runtime from the repository root:

```sh
git submodule update --init
conda env create -f environment_ASVO.yml
conda activate ASVO
python scripts/prepare_runtime.py
jupyter lab
```

Open [the experiment notebook](team_evolution/experiments/evolution_experiment.ipynb). Select meeting and classifier models available to your OpenAI account. Creating a plan is offline. Running a meeting requires an API key, model access and an embedding model download on first use. The notebook reads `OPENAI_API_KEY` or prompts privately.

The notebook starts with generation 0 (three meetings). To continue, raise the generation limit and reuse the saved plan and output directory. Each run writes an `index.html` with links to individual replays, a Markdown report and token usage.

## Command line

Set the meeting model in [simulation_config.json](team_evolution/simulation_config.json), set `OPENAI_API_KEY` in your shell, and supply your classifier model:

```sh
python -m team_evolution.experiments prepare --output runs/demo/experiment.json --classifier-model YOUR_CLASSIFIER_MODEL
python -m team_evolution.experiments run --plan runs/demo/experiment.json --output runs/demo/results --through-generation 0
```

For offline checks, run `python -m pytest -q team_evolution`.

## Layout

- `upstream/ASVO/` is the original ASVO Git submodule. The parent repository pins its commit.
- `team_evolution/` contains this project's experiments, configuration, notebooks and replay scoring.
- `runtime_overrides/` contains adapted upstream modules for agent actions, desire tracking, model access and replay support. Paths mirror upstream for comparison.
- `scripts/prepare_runtime.py` copies upstream Python sources and the overrides into the ignored `.asvo-runtime/`. Run it again after updating the submodule or overrides.
- `runs/` holds local results.

## Credits

Based on [ASVO](https://github.com/jingzhe-lin/ASVO) and [Concordia](https://github.com/google-deepmind/concordia). See the original [MIT license](LICENSE), the [Apache 2.0 license](licenses/Apache-2.0.txt) and the notices in individual source files.
