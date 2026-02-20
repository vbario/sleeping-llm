"""Sleeping LLM — main entry point.

Usage:
    python -m src.main                    # run with default config
    python -m src.main --config my.yaml   # run with custom config
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import Config
from src.orchestrator import Orchestrator


def main():
    parser = argparse.ArgumentParser(description="Sleeping LLM — a lifelong-learning AI")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yaml (default: project root config.yaml)",
    )
    args = parser.parse_args()

    config = Config(args.config)
    orchestrator = Orchestrator(config)
    orchestrator.run()


if __name__ == "__main__":
    main()
