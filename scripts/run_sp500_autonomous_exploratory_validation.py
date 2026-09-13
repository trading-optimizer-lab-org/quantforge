"""CLI for one explicitly owner-authorized exploratory validation candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aurora.infra.sp500_autonomous_discovery.validation import (
    run_exploratory_validation,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-results-dir", type=Path, required=True)
    parser.add_argument("--validation-prepared-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--strategy-id", required=True)
    parser.add_argument("--validation-ack", required=True)
    args = parser.parse_args()
    summary = run_exploratory_validation(
        train_results_dir=args.train_results_dir,
        validation_prepared_dir=args.validation_prepared_dir,
        output_dir=args.output_dir,
        strategy_id=args.strategy_id,
        validation_ack=args.validation_ack,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["candidate_status"] == "evaluated" else 3


if __name__ == "__main__":
    raise SystemExit(main())
