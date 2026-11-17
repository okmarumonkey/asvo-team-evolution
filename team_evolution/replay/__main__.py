"""Command-line entry point for converting saved ASVO results."""

from __future__ import annotations

import argparse
from pathlib import Path

from .exporter import build_replay_from_saved_results
from .viewer import render_replay_html


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an ASVO Replay Viewer")
    parser.add_argument("result_files", nargs="+", help="Per-agent result JSON files")
    parser.add_argument("--output", required=True, help="Output replay JSON path")
    parser.add_argument("--title", default="ASVO Simulation")
    args = parser.parse_args()

    json_path = Path(args.output)
    document = build_replay_from_saved_results(
        args.result_files, output_path=json_path, title=args.title
    )
    html_path = render_replay_html(document, json_path.with_suffix(".html"))
    print(f"Replay JSON: {json_path}")
    print(f"Replay HTML: {html_path}")


if __name__ == "__main__":
    main()
