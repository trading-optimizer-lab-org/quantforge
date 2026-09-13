#!/usr/bin/env python3
"""Build the compact, train-only input artifact for the massive night run."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from aurora.infra.sp500_autonomous_discovery.contracts import (
    LOCKED_START,
    TRAIN_END,
    VALIDATION_END,
    VALIDATION_START,
)
from aurora.infra.sp500_autonomous_discovery.massive_train import (
    CAMPAIGN_VERSION,
    MINUTES_PER_SHARD,
    SHARDS,
    WAVES,
    WORKERS_PER_SHARD,
    massive_recipe,
)
from aurora.infra.sp500_long_short_daily.data import load_market_snapshot


LEAN_FILES = (
    "market_data_manifest.json",
    "spy_ledger.parquet",
    "causal_series.parquet",
)

PRIOR_FILES = (
    "cumulative_autonomous_train_returns.parquet",
    "autonomous_trial_ledger.parquet",
    "autonomous_batch_summary.json",
    "historical_multiplicity/v1_train_daily_returns.parquet",
    "historical_multiplicity/v2_train_daily_returns.parquet",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-prepared-root", type=Path, required=True)
    parser.add_argument("--prior-results-root", type=Path, required=True)
    parser.add_argument("--prior-output-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.source_prepared_root.resolve()
    output = args.output_dir.resolve()
    prior_source = args.prior_results_root.resolve()
    prior_output = args.prior_output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    prior_output.mkdir(parents=True, exist_ok=True)
    for name in LEAN_FILES:
        path = source / name
        if not path.is_file():
            raise FileNotFoundError(f"MASSIVE_TRAIN_INPUT_MISSING:{name}")
        shutil.copy2(path, output / name)
    for name in PRIOR_FILES:
        path = prior_source / name
        if not path.is_file():
            raise FileNotFoundError(f"MASSIVE_PRIOR_EVIDENCE_MISSING:{name}")
        target = prior_output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    prior_summary = json.loads(
        (prior_output / "autonomous_batch_summary.json").read_text(encoding="utf-8")
    )
    if int(prior_summary.get("global_trial_count", -1)) != 5232:
        raise RuntimeError("MASSIVE_PRIOR_TRIAL_COUNT_MISMATCH")
    if prior_summary.get("validation_used_for_selection") is not False:
        raise RuntimeError("MASSIVE_PRIOR_VALIDATION_BREACH")
    if prior_summary.get("locked_opened") is not False:
        raise RuntimeError("MASSIVE_PRIOR_LOCKED_BREACH")
    prior_ledger = pd.read_parquet(prior_output / "autonomous_trial_ledger.parquet")
    prior_hashes = np.asarray(
        [bytes.fromhex(str(value)) for value in prior_ledger["canonical_hash"]],
        dtype="S32",
    )
    np.save(output / "prior_canonical_hashes.npy", np.unique(prior_hashes))

    data = load_market_snapshot(output)
    if data.split != "train":
        raise RuntimeError("MASSIVE_TRAIN_SPLIT_MISMATCH")
    if data.ledger.index.max().date().isoformat() != TRAIN_END:
        raise RuntimeError("MASSIVE_TRAIN_END_MISMATCH")
    if data.ledger.index.max().date().isoformat() >= LOCKED_START:
        raise RuntimeError("MASSIVE_TRAIN_LOCKED_BREACH")
    for name, series in data.series.items():
        finite = series.dropna()
        if len(finite) and finite.index.max().date().isoformat() > TRAIN_END:
            raise RuntimeError(f"MASSIVE_TRAIN_CAUSAL_SERIES_BREACH:{name}")

    recipe = massive_recipe()
    recipe_path = output / "massive_recipe.json"
    recipe_path.write_text(
        json.dumps(recipe.to_payload(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "campaign_version": CAMPAIGN_VERSION,
        "github_only": True,
        "requires_local_machine": False,
        "train_only": True,
        "train_end": TRAIN_END,
        "validation_start": VALIDATION_START,
        "validation_end": VALIDATION_END,
        "validation_opened": False,
        "locked_start": LOCKED_START,
        "locked_opened": False,
        "waves": WAVES,
        "shards_per_wave": SHARDS,
        "workers_per_shard": WORKERS_PER_SHARD,
        "minutes_per_shard": MINUTES_PER_SHARD,
        "parallel_python_processes": SHARDS * WORKERS_PER_SHARD,
        "effective_rule_combinations": recipe.total_combinations,
        "families": [
            {"family": row.family, "combinations": row.combinations}
            for row in recipe.families
        ],
        "files": {
            name: {
                "bytes": (output / name).stat().st_size,
                "sha256": _sha256(output / name),
            }
            for name in (*LEAN_FILES, "massive_recipe.json", "prior_canonical_hashes.npy")
        },
        "prior_multiplicity": {
            "global_trial_count": 5232,
            "source_batch": 51,
            "files": {
                name: {
                    "bytes": (prior_output / name).stat().st_size,
                    "sha256": _sha256(prior_output / name),
                }
                for name in PRIOR_FILES
            },
        },
    }
    (output / "massive_train_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
