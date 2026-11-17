# ASVO Team Evolution

Search for three-person teams among 20 simulated employees. Each team compares three beer concepts over three meeting steps. A classifier scores five teamwork behaviors in each agent's chosen action. Each positive label earns one point, up to 45 points per meeting.

Each generation keeps the highest-scoring team and tests two new teams formed by crossover, with a 50% chance of replacing a member. Ten generation updates evaluate 23 teams.

## Run

From the repository root, create the Python 3.12 environment:

```sh
conda env create -f environment_ASVO.yml
conda activate ASVO
jupyter lab
```

Open [the experiment notebook](examples/ASVO/experiments/evolution_experiment.ipynb). Select meeting and classifier models available to your OpenAI account. Creating a plan is offline. Running a meeting requires an API key, model access and an embedding model download on first use. The notebook reads `OPENAI_API_KEY` or prompts privately.

The notebook starts with generation 0 (three meetings). To continue, raise the generation limit and reuse the saved plan and output directory. Each run writes an `index.html` with links to individual replays, a Markdown report and token usage.

## Command line

Set the meeting model in [simulation_config.json](examples/ASVO/simulation_config.json), set `OPENAI_API_KEY` in your shell, and supply your classifier model:

```sh
python -m examples.ASVO.experiments prepare --output runs/demo/experiment.json --classifier-model YOUR_CLASSIFIER_MODEL
python -m examples.ASVO.experiments run --plan runs/demo/experiment.json --output runs/demo/results --through-generation 0
```

For offline checks, run `python -m pytest -q examples/ASVO`.

## Credits

Based on [ASVO](https://github.com/jingzhe-lin/ASVO) and [Concordia](https://github.com/google-deepmind/concordia). See the original [MIT license](LICENSE), the [Apache 2.0 license](licenses/Apache-2.0.txt) and the notices in individual source files.
