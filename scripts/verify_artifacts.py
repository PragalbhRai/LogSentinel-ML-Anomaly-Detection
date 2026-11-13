"""Run with: python scripts/verify_artifacts.py [--output outputs/verification.json]."""

import argparse
import json
from pathlib import Path

from logsentinel.evaluation import verify_frozen_inference
from logsentinel.artifacts import DEFAULT_ARTIFACT_DIR


def main():
    parser = argparse.ArgumentParser(description="Verify frozen Colab inference without retraining.")
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = verify_frozen_inference(args.artifacts)
    text = json.dumps(report, indent=2)
    if args.output:
        destination = args.output.resolve()
        originals = args.artifacts.resolve()
        if destination == originals or originals in destination.parents:
            parser.error("Verification output must be outside the original artifact directory.")
        if destination.exists():
            parser.error("Refusing to overwrite an existing verification report; choose a new output path.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
