"""Refresh autonomous batch inputs inside a reused immutable market snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aurora.infra.sp500_autonomous_discovery.registry import read_batch_registry
from aurora.infra.sp500_autonomous_discovery.workload import (
    refresh_autonomous_prepared_inputs,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-root", required=True, type=Path)
    args = parser.parse_args()
    candidates = refresh_autonomous_prepared_inputs(args.prepared_root)
    registry = read_batch_registry(args.prepared_root)
    if len(registry) != len(candidates):
        raise RuntimeError("REFRESHED_CANDIDATE_REGISTRY_COUNT_MISMATCH")
    print(
        json.dumps(
            {
                "candidate_count": len(candidates),
                "first_strategy_id": candidates[0]["strategy_id"],
                "last_strategy_id": candidates[-1]["strategy_id"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
