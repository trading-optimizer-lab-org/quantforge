#!/usr/bin/env python3
"""Convert the 5,232 prior trials into mergeable massive-run statistics."""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd

from aurora.infra.sp500_autonomous_discovery.contracts import (
    LOCKED_START,
    MULTIPLICITY_DATE_UNIT,
    TRAIN_END,
)
from aurora.infra.sp500_autonomous_discovery.massive_train import (
    BootstrapAccumulator,
    PboAccumulator,
    normal_pvalue,
)
from aurora.infra.sp500_long_short_daily.data import load_market_snapshot


_MATRIX: pd.DataFrame | None = None
_BENCHMARK: np.ndarray | None = None
_OUTPUT: Path | None = None
_BOOTSTRAP_WEIGHTS: np.ndarray | None = None

def _normalise_multiplicity_dates(values: object) -> pd.DatetimeIndex:
    """Return dates in the one unit used by the massive-run interchange file."""
    dates = pd.DatetimeIndex(pd.to_datetime(values, errors="raise"))
    return dates.as_unit(MULTIPLICITY_DATE_UNIT)


def _write_multiplicity_common_dates(
    path: Path, values: object
) -> pd.DatetimeIndex:
    """Write and immediately verify the common interval as int64 nanoseconds."""
    dates = _normalise_multiplicity_dates(values)
    np.save(path, dates.asi8.astype(np.int64, copy=False))
    raw = np.load(path, allow_pickle=False)
    roundtrip = pd.DatetimeIndex(
        pd.to_datetime(raw, unit=MULTIPLICITY_DATE_UNIT, errors="raise")
    ).as_unit(MULTIPLICITY_DATE_UNIT)
    if not roundtrip.equals(dates):
        raise RuntimeError("PRIOR_MULTIPLICITY_DATE_SERIALIZATION_BREACH")
    return dates


def _load_evaluated_returns(prior: Path, ledger: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    autonomous = pd.read_parquet(prior / "cumulative_autonomous_train_returns.parquet")
    autonomous["date"] = pd.to_datetime(autonomous["date"])
    frames.append(autonomous[["strategy_id", "date", "return"]])
    for campaign, relative_path in (
        ("V1", "historical_multiplicity/v1_train_daily_returns.parquet"),
        ("V2", "historical_multiplicity/v2_train_daily_returns.parquet"),
    ):
        mapping = {
            str(row.source_strategy_id): str(row.strategy_id)
            for row in ledger.loc[
                ledger["campaign"].eq(campaign) & ledger["status"].eq("evaluated")
            ].itertuples(index=False)
        }
        historical = pd.read_parquet(prior / relative_path)
        historical = historical.loc[
            historical["unit_key"].astype(str).isin(mapping)
        ].copy()
        historical["strategy_id"] = historical["unit_key"].astype(str).map(mapping)
        historical["date"] = pd.to_datetime(historical["date"])
        frames.append(historical[["strategy_id", "date", "return"]])
    combined = pd.concat(frames, ignore_index=True).drop_duplicates(
        ["strategy_id", "date"], keep="last"
    )
    expected = set(
        ledger.loc[ledger["status"].eq("evaluated"), "strategy_id"].astype(str)
    )
    observed = set(combined["strategy_id"].astype(str))
    if observed != expected:
        missing = sorted(expected - observed)[:10]
        extra = sorted(observed - expected)[:10]
        raise RuntimeError(
            f"PRIOR_MULTIPLICITY_STREAM_SET_MISMATCH:missing={missing}:extra={extra}"
        )
    return combined


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior-root", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def _worker(worker: int, identifiers: list[str]) -> None:
    assert _MATRIX is not None
    assert _BENCHMARK is not None
    assert _OUTPUT is not None
    assert _BOOTSTRAP_WEIGHTS is not None
    root = _OUTPUT / f"worker-{worker}"
    root.mkdir(parents=True, exist_ok=True)
    bootstrap = BootstrapAccumulator.with_shared_weights(_BOOTSTRAP_WEIGHTS)
    pbo = PboAccumulator.create()
    for offset, identifier in enumerate(identifiers, start=1):
        values = _MATRIX[identifier].to_numpy(dtype=float)
        bootstrap.update(values - _BENCHMARK)
        pbo.update(identifier, values)
        if offset % 250 == 0:
            print(json.dumps({"worker": worker, "prior_completed": offset}), flush=True)
    np.savez_compressed(
        root / "bootstrap_accumulator.npz",
        white_max=bootstrap.white_max,
        spa_max=bootstrap.spa_max,
        observed_max=np.asarray([bootstrap.observed_max]),
        observed_spa_max=np.asarray([bootstrap.observed_spa_max]),
    )
    np.savez_compressed(
        root / "pbo_accumulator.npz",
        histogram=pbo.histogram.astype(np.uint32),
        best_is=pbo.best_is,
        best_oos=pbo.best_oos,
        best_ids=np.asarray([value or "" for value in pbo.best_ids], dtype="U64"),
    )


def main() -> int:
    global _MATRIX, _BENCHMARK, _OUTPUT, _BOOTSTRAP_WEIGHTS
    args = parse_args()
    prior = args.prior_root.resolve()
    prepared = args.prepared_root.resolve()
    _OUTPUT = args.output_dir.resolve()
    _OUTPUT.mkdir(parents=True, exist_ok=True)
    ledger = pd.read_parquet(prior / "autonomous_trial_ledger.parquet")
    if len(ledger) != 5232:
        raise RuntimeError("PRIOR_MULTIPLICITY_LEDGER_COUNT_MISMATCH")
    source_summary = json.loads(
        (prior / "autonomous_batch_summary.json").read_text(encoding="utf-8")
    )
    expected_streams = int(source_summary["evaluated_streams_in_multiplicity"])
    returns = _load_evaluated_returns(prior, ledger)
    if returns["date"].max() > pd.Timestamp(TRAIN_END):
        raise RuntimeError("PRIOR_MULTIPLICITY_TRAIN_BOUNDARY_BREACH")
    if returns["date"].max() >= pd.Timestamp(LOCKED_START):
        raise RuntimeError("PRIOR_MULTIPLICITY_LOCKED_BREACH")
    wide = returns.pivot(index="date", columns="strategy_id", values="return").sort_index()
    if wide.shape[1] != expected_streams:
        raise RuntimeError(
            f"PRIOR_MULTIPLICITY_STREAM_COUNT_MISMATCH:{wide.shape[1]}:{expected_streams}"
        )
    data = load_market_snapshot(prepared)
    benchmark = data.ledger["long_return"].reindex(wide.index)
    if benchmark.isna().any():
        raise RuntimeError("PRIOR_MULTIPLICITY_BENCHMARK_ALIGNMENT_BREACH")
    differential = wide.subtract(benchmark, axis=0)
    pvalue_by_id = {
        str(identifier): normal_pvalue(differential[identifier].dropna().to_numpy(dtype=float))
        for identifier in wide.columns
    }
    _MATRIX = wide.dropna(axis=0, how="any")
    if len(_MATRIX) < 1500 or _MATRIX.isna().any().any():
        raise RuntimeError("PRIOR_MULTIPLICITY_COMMON_INTERVAL_TOO_SHORT")
    _BENCHMARK = benchmark.reindex(_MATRIX.index).to_numpy(dtype=float)
    _BOOTSTRAP_WEIGHTS = BootstrapAccumulator.create(len(_MATRIX)).weights
    common_dates = _write_multiplicity_common_dates(
        prepared / "multiplicity_common_dates.npy", _MATRIX.index
    )
    prepared_manifest_path = prepared / "massive_train_manifest.json"
    prepared_manifest = json.loads(prepared_manifest_path.read_text(encoding="utf-8"))
    prepared_manifest["multiplicity_common_interval"] = {
        "sessions": len(common_dates),
        "start": common_dates.min().date().isoformat(),
        "end": common_dates.max().date().isoformat(),
        "evaluated_prior_streams": expected_streams,
        "dates_file": "multiplicity_common_dates.npy",
        "numpy_datetime_unit": MULTIPLICITY_DATE_UNIT,
    }
    prepared_manifest_path.write_text(
        json.dumps(prepared_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    identifiers = list(_MATRIX.columns.astype(str))
    chunks = [identifiers[offset:: args.workers] for offset in range(args.workers)]
    context = mp.get_context("fork")
    processes = [
        context.Process(target=_worker, args=(worker, chunks[worker]))
        for worker in range(args.workers)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    failures = [process.exitcode for process in processes if process.exitcode != 0]
    if failures:
        raise RuntimeError(f"PRIOR_MULTIPLICITY_WORKER_FAILURES:{failures}")

    hash_by_id = {
        str(row.strategy_id): bytes.fromhex(str(row.canonical_hash))
        for row in ledger.itertuples(index=False)
    }
    canonical = np.asarray(list(hash_by_id.values()), dtype="S32")
    evaluated_hashes = np.asarray([hash_by_id[value] for value in identifiers], dtype="S32")
    pvalues = np.asarray([pvalue_by_id[value] for value in identifiers], dtype=float)
    order = np.argsort(evaluated_hashes, kind="mergesort")
    np.save(_OUTPUT / "canonical_hashes.npy", canonical)
    np.save(_OUTPUT / "evaluated_hashes.npy", evaluated_hashes[order])
    np.save(_OUTPUT / "raw_pvalues.npy", pvalues[order])
    summary = {
        "declared_prior_trials": len(ledger),
        "evaluated_prior_streams": len(identifiers),
        "common_interval_sessions": len(_MATRIX),
        "common_interval_start": _MATRIX.index.min().date().isoformat(),
        "common_interval_end": _MATRIX.index.max().date().isoformat(),
        "train_end": TRAIN_END,
        "validation_opened": False,
        "locked_start": LOCKED_START,
        "locked_opened": False,
    }
    (_OUTPUT / "prior_statistics_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
