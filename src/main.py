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
    parser.add_argument(
        "--mark-all-consumed",
        action="store_true",
        help="Mark all existing sessions as consumed (migration helper)",
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="Launch web UI instead of CLI",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for web UI (default: 8000)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset model to base weights (keeps conversation logs)",
    )
    parser.add_argument(
        "--factory-reset",
        action="store_true",
        help="Full reset: clear weights, training data, replay buffer",
    )
    parser.add_argument(
        "--disable-memit",
        action="store_true",
        help="Disable MEMIT injection during wake phase",
    )
    args = parser.parse_args()

    config = Config(args.config)

    if args.reset:
        orchestrator = Orchestrator(config)
        result = orchestrator.reset_weights()
        print(result["message"])
        return

    if args.factory_reset:
        orchestrator = Orchestrator(config)
        result = orchestrator.factory_reset()
        print(result["message"])
        return

    if args.mark_all_consumed:
        from src.memory.session_tracker import SessionTracker
        from src.wake.logger import ConversationLogger
        tracker = SessionTracker(config)
        sessions = ConversationLogger.list_sessions(config)
        if sessions:
            tracker.mark_consumed(sessions, "migration")
            print(f"Marked {len(sessions)} session(s) as consumed.")
        else:
            print("No sessions found.")
        return

    if args.web:
        from src.web.server import create_app
        import uvicorn
        app = create_app(config, disable_memit=args.disable_memit)
        uvicorn.run(app, host="0.0.0.0", port=args.port)
    else:
        orchestrator = Orchestrator(config, disable_memit=args.disable_memit)
        orchestrator.run()


if __name__ == "__main__":
    main()
