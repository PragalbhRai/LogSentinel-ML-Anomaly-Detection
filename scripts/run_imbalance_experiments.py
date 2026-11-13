"""Run two supervised RF experiments without changing any frozen artifacts."""

import argparse
from datetime import datetime, timezone
from pathlib import Path

from logsentinel.artifacts import DEFAULT_ARTIFACT_DIR, PROJECT_ROOT
from logsentinel.imbalance_experiments import DEFAULT_CONFIG, run_experiments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "imbalance_experiments" /
                        datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S_%fZ"))
    args = parser.parse_args()
    print(f"New experiment directory: {args.output}", flush=True)
    run_experiments(args.output, args.artifacts, args.config)


if __name__ == "__main__":
    main()
