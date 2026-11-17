"""python -m examples.ASVO.experiments prepare|check|run"""
from __future__ import annotations

import argparse
import functools
import json
from pathlib import Path

from .protocol import prepare, write_json


def main():
    parser = argparse.ArgumentParser(description='ASVO team experiments')
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare', help='Generate an offline experiment plan')
    prep.add_argument('--config', default=str(Path(__file__).resolve().parents[1] / 'simulation_config.json'))
    prep.add_argument('--output', required=True)
    prep.add_argument('--pool-seed', type=int,
                      help='Seed for the candidate pool')
    prep.add_argument('--mode', choices=['evolution', 'legacy'], default='evolution')
    prep.add_argument('--search-seed', type=int,
                      help='Seed for team selection')
    prep.add_argument('--simulation-seed', type=int,
                      help='Seed for the meeting simulation')
    prep.add_argument('--generation-updates', type=int, default=10,
                      help='Generation updates (10 by default, 0 for initial teams only)')
    prep.add_argument('--svo-type', choices=['Altruistic', 'Prosocial', 'Individualistic', 'Competitive'],
                      default='Prosocial', help='Shared social value orientation')
    prep.add_argument('--classifier-model', default='gpt-6.1-sol',
                      help='Model used for teamwork scoring')
    prep.add_argument('--discovery-seeds', type=int, nargs='+', default=[101])
    prep.add_argument('--evaluation-seeds', type=int, nargs='+', default=[201, 202])
    for command in ('check', 'run', 'overview'):
        sub = commands.add_parser(command)
        sub.add_argument('--plan', required=True)
        if command in ('run', 'overview'):
            sub.add_argument('--output', required=True)
        if command == 'run':
            sub.add_argument('--through-generation', type=int,
                             help='Run through this generation, resuming any completed trials')
    args = parser.parse_args()
    try:
        if args.command == 'prepare':
            from examples.ASVO.config_loader import load_config
            destination = Path(args.output)
            if destination.exists():
                raise ValueError('Plan already exists. Choose a new output path.')
            if args.mode == 'evolution':
                from .evolution import prepare_evolution
                plan = prepare_evolution(load_config(args.config), pool_seed=args.pool_seed,
                                         search_seed=args.search_seed,
                                         simulation_seed=args.simulation_seed,
                                         generation_updates=args.generation_updates,
                                         svo_type=args.svo_type,
                                         classifier_model_name=args.classifier_model)
            else:
                plan = prepare(load_config(args.config),
                               pool_seed=args.pool_seed if args.pool_seed is not None else 20261031,
                               discovery_seeds=args.discovery_seeds, evaluation_seeds=args.evaluation_seeds)
            write_json(destination, plan)
            print(f'Plan: {destination}\n{len(plan["personas"])} candidates / '
                  f'{len(plan["teams"])} initial teams / {plan["planned_simulations"]} trials')
            print('Running the plan uses the model API and incurs charges.')
        else:
            plan = json.loads(Path(args.plan).read_text())
            if plan.get('mode') == 'evolution':
                from .evolution import validate_plan
                validate_plan(plan)
            if args.command == 'overview':
                if plan.get('mode') != 'evolution':
                    raise ValueError('HTML overview requires an evolution plan.')
                from .evolution import rebuild_overview_html
                count = rebuild_overview_html(plan, args.output)
                print(f'Overview: {Path(args.output) / "index.html"}（{count} trials）')
                return
            from .engine import check_runtime, load_resources, simulate
            check_runtime(plan['config'])
            if args.command == 'check':
                print('API key is set. Model access requires a live call.')
                return
            model, classifier_model, embedder = load_resources(plan['config'])
            runner = functools.partial(simulate, model=model,
                                       classifier_model=classifier_model, embedder=embedder)
            if plan.get('mode') == 'evolution':
                from .evolution import run_evolution
                run_evolution(plan, args.output, runner,
                              through_generation=args.through_generation)
            else:
                from .workflow import run_experiment
                run_experiment(plan, args.output, runner, model)
            print(f'Report: {Path(args.output) / "report.md"}')
            if plan.get('mode') == 'evolution':
                print(f'Overview: {Path(args.output) / "index.html"}')
    except (ValueError, RuntimeError) as exc:
        parser.exit(2, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
