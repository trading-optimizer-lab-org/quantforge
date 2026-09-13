#!/usr/bin/env python3
"""Run one four-process, time-bounded massive train-only search shard."""

from __future__ import annotations

import argparse
import heapq
import json
import math
import multiprocessing as mp
import os
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from aurora.infra.sp500_autonomous_discovery.contracts import (
    LOCKED_START,
    MULTIPLICITY_DATE_UNIT,
    PREVIOUS_TRIAL_COUNT,
    TRAIN_END,
)
from aurora.infra.sp500_autonomous_discovery.feature_store import FeatureStore
from aurora.infra.sp500_autonomous_discovery.massive_train import (
    BootstrapAccumulator,
    MassiveRecipe,
    PboAccumulator,
    TOP_ROWS_PER_PROCESS,
    WORKERS_PER_SHARD,
    candidate_for_index,
    candidate_index,
    candidate_metric_row,
)
from aurora.infra.sp500_long_short_daily.data import load_market_snapshot
from aurora.infra.sp500_long_short_daily.signals import (
    CandidateRejected,
    candidate_decisions,
)


EVALUATION_START = pd.Timestamp("1998-01-01")
_DATA: Any = None
_FEATURES: pd.DataFrame | None = None
_RECIPE: MassiveRecipe | None = None
_EVALUATION_DATES: pd.DatetimeIndex | None = None
_BENCHMARK: np.ndarray | None = None
_MULTIPLICITY_DATES: pd.DatetimeIndex | None = None
_MULTIPLICITY_BENCHMARK: np.ndarray | None = None
_OUTPUT_ROOT: Path | None = None
_WAVE = 0
_SHARD = 0
_SEARCH_SECONDS = 0.0
_MAX_CANDIDATES = 0
_PRIOR_HASHES: frozenset[str] = frozenset()
_BOOTSTRAP_WEIGHTS: np.ndarray | None = None

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--recipe-json", type=Path, required=True)
    parser.add_argument("--wave", type=int, required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--minutes", type=int, default=50)
    parser.add_argument("--workers", type=int, default=WORKERS_PER_SHARD)
    parser.add_argument("--max-candidates-per-worker", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _feature_frame(prepared_root: Path, data: Any) -> pd.DataFrame:
    manifest = json.loads(
        (prepared_root / "market_data_manifest.json").read_text(encoding="utf-8")
    )
    return FeatureStore(
        dataset_sha256=str(manifest["snapshot_sha256"]),
        code_sha=os.environ.get("GITHUB_SHA", "LOCAL_STRUCTURAL_TEST"),
        start=str(data.ledger.index.min().date()),
        end=TRAIN_END,
    ).get_or_build("SPY", data.ledger)


def _load_multiplicity_common_dates(
    prepared: Path, evaluation_dates: pd.DatetimeIndex
) -> pd.DatetimeIndex:
    """Load the prepared interval without guessing the NumPy datetime unit."""
    manifest = json.loads(
        (prepared / "massive_train_manifest.json").read_text(encoding="utf-8")
    )
    interval = manifest.get("multiplicity_common_interval", {})
    if interval.get("numpy_datetime_unit") != MULTIPLICITY_DATE_UNIT:
        raise RuntimeError("MASSIVE_MULTIPLICITY_DATETIME_UNIT_BREACH")
    raw = np.load(
        prepared / "multiplicity_common_dates.npy", allow_pickle=False
    )
    if raw.ndim != 1 or raw.dtype.kind not in "iu":
        raise RuntimeError("MASSIVE_MULTIPLICITY_DATE_SHAPE_BREACH")
    dates = pd.DatetimeIndex(
        pd.to_datetime(raw, unit=MULTIPLICITY_DATE_UNIT, errors="raise")
    ).as_unit(MULTIPLICITY_DATE_UNIT)
    train_end = pd.Timestamp(TRAIN_END)
    if (
        len(dates) < 1500
        or not dates.is_monotonic_increasing
        or dates.has_duplicates
        or not dates.isin(evaluation_dates).all()
        or dates.max() > train_end
        or dates.min() > dates.max()
        or interval.get("sessions") != len(dates)
        or interval.get("start") != dates.min().date().isoformat()
        or interval.get("end") != dates.max().date().isoformat()
    ):
        raise RuntimeError("MASSIVE_MULTIPLICITY_COMMON_INTERVAL_BREACH")
    return dates


def _evaluate(candidate: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert _DATA is not None
    assert _FEATURES is not None
    assert _EVALUATION_DATES is not None
    assert _BENCHMARK is not None
    assert _MULTIPLICITY_DATES is not None
    assert _MULTIPLICITY_BENCHMARK is not None
    signal = candidate_decisions(candidate, _DATA, feature_frame=_FEATURES)
    if signal.first_evaluable_date is None:
        raise CandidateRejected("NO_EVALUABLE_SESSION")
    if signal.missing_fraction > 0.02 + 1e-12:
        raise CandidateRejected("DATA_INELIGIBLE:CAUSAL_COVERAGE_LT_98_PERCENT")
    first = pd.Timestamp(signal.first_evaluable_date)
    if first >= EVALUATION_START:
        raise CandidateRejected("DATA_INELIGIBLE:LATE_TRAIN_WARMUP")
    decisions = signal.decisions.reindex(_DATA.ledger.index)
    positions = decisions.shift(1).ffill().fillna(1).astype(np.int8)
    if not positions.isin((-1, 1)).all():
        raise CandidateRejected("TECHNICAL_FAILURE_POSITION")
    values = (
        positions.reindex(_EVALUATION_DATES).to_numpy(dtype=float)
        * _DATA.ledger.loc[_EVALUATION_DATES, "long_return"].to_numpy(dtype=float)
    )
    if len(values) < 2500 or not np.isfinite(values).all():
        raise CandidateRejected("DATA_INELIGIBLE:MINIMUM_TRAIN_OOF_COVERAGE")
    multiplicity_values = (
        positions.reindex(_MULTIPLICITY_DATES).to_numpy(dtype=float)
        * _DATA.ledger.loc[_MULTIPLICITY_DATES, "long_return"].to_numpy(dtype=float)
    )
    differential = multiplicity_values - _MULTIPLICITY_BENCHMARK
    return values, multiplicity_values, differential


def _write_frame(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    pd.DataFrame(list(rows)).to_csv(path, index=False)


def _worker(worker: int) -> None:
    assert _RECIPE is not None
    assert _OUTPUT_ROOT is not None
    assert _EVALUATION_DATES is not None
    assert _BENCHMARK is not None
    root = _OUTPUT_ROOT / f"worker-{worker}"
    root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    deadline = started + _SEARCH_SECONDS
    assert _MULTIPLICITY_DATES is not None
    assert _BOOTSTRAP_WEIGHTS is not None
    bootstrap = BootstrapAccumulator.with_shared_weights(_BOOTSTRAP_WEIGHTS)
    pbo = PboAccumulator.create()
    top_heap: list[tuple[float, str, dict[str, Any]]] = []
    gate_rows: list[dict[str, Any]] = []
    pvalues: list[float] = []
    canonical_hashes: list[bytes] = []
    evaluated_hashes: list[bytes] = []
    seen: set[str] = set()
    rejections: Counter[str] = Counter()
    family_counts: dict[str, Counter[str]] = defaultdict(Counter)
    timing: dict[str, list[float]] = defaultdict(list)
    evaluated = 0
    generated = 0
    deduped = 0
    runtime_errors = 0
    iteration = 0
    while time.perf_counter() < deadline:
        if _MAX_CANDIDATES and generated >= _MAX_CANDIDATES:
            break
        index = candidate_index(_WAVE, _SHARD, worker, iteration)
        iteration += 1
        candidate_started = time.perf_counter()
        try:
            candidate = candidate_for_index(
                index,
                wave=_WAVE,
                recipe=_RECIPE,
            )
            generated += 1
            family = str(candidate["family"])
            digest = str(candidate["canonical_hash"])
            canonical_hashes.append(bytes.fromhex(digest))
            family_counts[family]["generated"] += 1
            if digest in seen or digest in _PRIOR_HASHES:
                deduped += 1
                family_counts[family]["deduped"] += 1
                continue
            seen.add(digest)
            values, multiplicity_values, differential = _evaluate(candidate)
            spa_pvalue = bootstrap.update(differential)
            pbo.update(str(candidate["strategy_id"]), multiplicity_values)
            row = candidate_metric_row(
                candidate,
                _EVALUATION_DATES,
                values,
                _BENCHMARK,
                global_trials=PREVIOUS_TRIAL_COUNT + 1,
                spa_pvalue=spa_pvalue,
            )
            elapsed = time.perf_counter() - candidate_started
            row.update(
                {
                    "wave": _WAVE,
                    "shard": _SHARD,
                    "worker": worker,
                    "candidate_index": index,
                    "seconds_total": elapsed,
                    "dsr_provisional_only": True,
                }
            )
            evaluated += 1
            evaluated_hashes.append(bytes.fromhex(digest))
            family_counts[family]["evaluated"] += 1
            pvalues.append(float(row["raw_pvalue"]))
            timing[family].append(elapsed)
            score = float(row["train_oof_cagr_pct"])
            item = (score, str(row["strategy_id"]), row)
            if len(top_heap) < TOP_ROWS_PER_PROCESS:
                heapq.heappush(top_heap, item)
            elif item[:2] > top_heap[0][:2]:
                heapq.heapreplace(top_heap, item)
            if bool(row["non_global_train_gate"]):
                gate_rows.append(row)
        except CandidateRejected as exc:
            reason = str(exc)
            rejections[reason] += 1
            family_name = str(locals().get("candidate", {}).get("family", "unknown"))
            family_counts[family_name]["rejected"] += 1
        except Exception as exc:  # pragma: no cover - preserved in GitHub artifact
            runtime_errors += 1
            reason = f"{type(exc).__name__}:{exc}"
            rejections[reason] += 1
            with (root / "runtime_errors.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "candidate_index": index,
                            "error": reason,
                            "traceback": traceback.format_exc(),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
        if generated and generated % 250 == 0:
            print(
                json.dumps(
                    {
                        "wave": _WAVE,
                        "shard": _SHARD,
                        "worker": worker,
                        "generated": generated,
                        "evaluated": evaluated,
                        "remaining_seconds": max(0, int(deadline - time.perf_counter())),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    top_rows = [item[2] for item in sorted(top_heap, reverse=True)]
    _write_frame(top_rows, root / "top_candidates.csv")
    _write_frame(gate_rows, root / "non_global_gate_candidates.csv")
    np.save(root / "raw_pvalues.npy", np.asarray(pvalues, dtype=np.float64))
    np.save(
        root / "canonical_hashes.npy",
        np.asarray(canonical_hashes, dtype="S32"),
    )
    np.save(
        root / "evaluated_hashes.npy",
        np.asarray(evaluated_hashes, dtype="S32"),
    )
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
    family_rows = []
    for family in sorted(family_counts):
        values = timing.get(family, [])
        family_rows.append(
            {
                "family": family,
                **family_counts[family],
                "mean_seconds": float(np.mean(values)) if values else None,
                "p95_seconds": float(np.quantile(values, 0.95)) if values else None,
            }
        )
    _write_frame(family_rows, root / "family_runtime.csv")
    summary = {
        "wave": _WAVE,
        "shard": _SHARD,
        "worker": worker,
        "generated": generated,
        "unique_local": len(seen),
        "evaluated": evaluated,
        "rejected": sum(rejections.values()) - runtime_errors,
        "deduped": deduped,
        "runtime_errors": runtime_errors,
        "non_global_gate_candidates": len(gate_rows),
        "elapsed_seconds": time.perf_counter() - started,
        "rejection_reasons": dict(rejections.most_common()),
        "train_end": TRAIN_END,
        "validation_opened": False,
        "locked_start": LOCKED_START,
        "locked_opened": False,
    }
    (root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    global _DATA, _FEATURES, _RECIPE, _EVALUATION_DATES
    global _BENCHMARK, _MULTIPLICITY_DATES, _MULTIPLICITY_BENCHMARK
    global _OUTPUT_ROOT, _WAVE, _SHARD, _SEARCH_SECONDS, _MAX_CANDIDATES
    global _PRIOR_HASHES, _BOOTSTRAP_WEIGHTS
    args = parse_args()
    if args.workers != WORKERS_PER_SHARD:
        raise ValueError(f"MASSIVE_WORKERS_MUST_EQUAL_{WORKERS_PER_SHARD}")
    if not 0 <= args.wave < 7 or not 0 <= args.shard < 360:
        raise ValueError("MASSIVE_WAVE_OR_SHARD_OUT_OF_RANGE")
    prepared = args.prepared_root.resolve()
    _OUTPUT_ROOT = args.output_dir.resolve()
    _OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    _WAVE = args.wave
    _SHARD = args.shard
    _MAX_CANDIDATES = args.max_candidates_per_worker
    _SEARCH_SECONDS = max(1.0, args.minutes * 60.0 - 120.0)
    _RECIPE = MassiveRecipe.from_payload(
        json.loads(args.recipe_json.read_text(encoding="utf-8"))
    )
    _PRIOR_HASHES = frozenset(
        value.tobytes().hex()
        for value in np.load(
            prepared / "prior_canonical_hashes.npy", allow_pickle=False
        ).astype("S32", copy=False)
    )
    _DATA = load_market_snapshot(prepared)
    if _DATA.split != "train" or _DATA.ledger.index.max() > pd.Timestamp(TRAIN_END):
        raise RuntimeError("MASSIVE_SHARD_TRAIN_BOUNDARY_BREACH")
    if _DATA.ledger.index.max() >= pd.Timestamp(LOCKED_START):
        raise RuntimeError("MASSIVE_SHARD_LOCKED_BREACH")
    _FEATURES = _feature_frame(prepared, _DATA)
    mask = (
        (_DATA.ledger.index >= EVALUATION_START)
        & (_DATA.ledger.index <= pd.Timestamp(TRAIN_END))
        & _DATA.ledger["long_return"].notna().to_numpy()
    )
    _EVALUATION_DATES = _DATA.ledger.index[mask]
    _BENCHMARK = _DATA.ledger.loc[_EVALUATION_DATES, "long_return"].to_numpy(dtype=float)
    _MULTIPLICITY_DATES = _load_multiplicity_common_dates(
        prepared, _EVALUATION_DATES
    )
    _MULTIPLICITY_BENCHMARK = _DATA.ledger.loc[
        _MULTIPLICITY_DATES, "long_return"
    ].to_numpy(dtype=float)
    _BOOTSTRAP_WEIGHTS = BootstrapAccumulator.create(
        len(_MULTIPLICITY_DATES)
    ).weights
    context = mp.get_context("fork")
    processes = [context.Process(target=_worker, args=(worker,)) for worker in range(args.workers)]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    failures = [process.exitcode for process in processes if process.exitcode != 0]
    if failures:
        raise RuntimeError(f"MASSIVE_WORKER_FAILURES:{failures}")
    shard_summary = {
        "wave": _WAVE,
        "shard": _SHARD,
        "workers": args.workers,
        "worker_summaries": [
            json.loads(
                (_OUTPUT_ROOT / f"worker-{worker}" / "summary.json").read_text(
                    encoding="utf-8"
                )
            )
            for worker in range(args.workers)
        ],
        "train_end": TRAIN_END,
        "validation_opened": False,
        "locked_start": LOCKED_START,
        "locked_opened": False,
    }
    (_OUTPUT_ROOT / "shard_summary.json").write_text(
        json.dumps(shard_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(shard_summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
