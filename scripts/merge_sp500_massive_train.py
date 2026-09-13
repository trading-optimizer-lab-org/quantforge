#!/usr/bin/env python3
"""Hierarchical merge for the massive train-only SPY search."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from aurora.infra.sp500_autonomous_discovery.contracts import (
    BOOTSTRAP_REPETITIONS,
    LOCKED_START,
    PREVIOUS_TRIAL_COUNT,
    TRAIN_END,
    VALIDATION_END,
    VALIDATION_START,
)
from aurora.infra.sp500_autonomous_discovery.massive_train import (
    CAMPAIGN_VERSION,
    PBO_BINS,
    PBO_MAX_SHARPE,
    PBO_MIN_SHARPE,
    PBO_PARTITIONS,
    dsr_from_moments,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("aggregate", "final"), required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--retain-top", type=int, default=5000)
    parser.add_argument("--prior-root", type=Path)
    return parser.parse_args()


def _files(root: Path, name: str) -> list[Path]:
    return sorted(path for path in root.rglob(name) if path.is_file())


def _frames(paths: Iterable[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        try:
            frame = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        if len(frame):
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _unique_hash_values(
    hash_paths: list[Path],
    value_name: str | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    hashes = []
    values = []
    for path in hash_paths:
        current = np.load(path, allow_pickle=False).astype("S32", copy=False)
        hashes.append(current)
        if value_name is not None:
            value_path = path.with_name(value_name)
            current_values = np.load(value_path, allow_pickle=False).astype(float, copy=False)
            if len(current) != len(current_values):
                raise RuntimeError(f"MASSIVE_HASH_VALUE_LENGTH_MISMATCH:{path}")
            values.append(current_values)
    if not hashes:
        return np.asarray([], dtype="S32"), (
            np.asarray([], dtype=float) if value_name is not None else None
        )
    all_hashes = np.concatenate(hashes)
    if value_name is None:
        return np.unique(all_hashes), None
    all_values = np.concatenate(values)
    order = np.argsort(all_hashes, kind="mergesort")
    ordered_hashes = all_hashes[order]
    ordered_values = all_values[order]
    starts = np.r_[0, np.flatnonzero(ordered_hashes[1:] != ordered_hashes[:-1]) + 1]
    unique_hashes = ordered_hashes[starts]
    minimum_values = np.minimum.reduceat(ordered_values, starts)
    return unique_hashes, minimum_values


def _merge_bootstrap(paths: list[Path], output: Path) -> dict[str, float]:
    white = np.full(BOOTSTRAP_REPETITIONS, -np.inf, dtype=float)
    spa = np.full(BOOTSTRAP_REPETITIONS, -np.inf, dtype=float)
    observed = -math.inf
    observed_spa = -math.inf
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            white = np.maximum(white, data["white_max"])
            spa = np.maximum(spa, data["spa_max"])
            observed = max(observed, float(data["observed_max"][0]))
            observed_spa = max(observed_spa, float(data["observed_spa_max"][0]))
    np.savez_compressed(
        output,
        white_max=white,
        spa_max=spa,
        observed_max=np.asarray([observed]),
        observed_spa_max=np.asarray([observed_spa]),
    )
    return {
        "observed_max": observed,
        "observed_spa_max": observed_spa,
        "white_reality_check_pvalue": float(
            (1 + np.count_nonzero(white >= observed)) / (BOOTSTRAP_REPETITIONS + 1)
        ) if math.isfinite(observed) else 1.0,
        "global_hansen_spa_pvalue": float(
            (1 + np.count_nonzero(spa >= observed_spa))
            / (BOOTSTRAP_REPETITIONS + 1)
        ) if math.isfinite(observed_spa) else 1.0,
    }


def _merge_pbo(paths: list[Path], output: Path) -> dict[str, float]:
    rows = math.comb(PBO_PARTITIONS, PBO_PARTITIONS // 2)
    histogram = np.zeros((rows, PBO_BINS), dtype=np.uint64)
    best_is = np.full(rows, -np.inf, dtype=float)
    best_oos = np.full(rows, -np.inf, dtype=float)
    best_ids = np.full(rows, "", dtype="U64")
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            histogram += data["histogram"].astype(np.uint64)
            winners = data["best_is"] > best_is
            best_is[winners] = data["best_is"][winners]
            best_oos[winners] = data["best_oos"][winners]
            best_ids[winners] = data["best_ids"][winners]
    np.savez_compressed(
        output,
        histogram=histogram,
        best_is=best_is,
        best_oos=best_oos,
        best_ids=best_ids,
    )
    below = 0
    valid = 0
    for row in range(rows):
        total = int(histogram[row].sum())
        if total <= 1 or not math.isfinite(float(best_oos[row])):
            continue
        scaled = (
            (np.clip(best_oos[row], PBO_MIN_SHARPE, PBO_MAX_SHARPE) - PBO_MIN_SHARPE)
            / (PBO_MAX_SHARPE - PBO_MIN_SHARPE)
            * PBO_BINS
        )
        winner_bin = min(max(int(math.floor(scaled)), 0), PBO_BINS - 1)
        rank_lower = int(histogram[row, :winner_bin].sum())
        below += int(rank_lower / total <= 0.5)
        valid += 1
    return {
        "pbo_conservative_upper": float(below / valid) if valid else 1.0,
        "pbo_partitions_evaluated": valid,
    }


def _merge_summaries(root: Path) -> dict[str, Any]:
    summaries = []
    for name in ("summary.json", "aggregate_summary.json"):
        for path in _files(root, name):
            summaries.append(json.loads(path.read_text(encoding="utf-8")))
    rejection_reasons: Counter[str] = Counter()
    totals: Counter[str] = Counter()
    for row in summaries:
        for key in (
            "generated",
            "unique_local",
            "evaluated",
            "rejected",
            "deduped",
            "runtime_errors",
            "non_global_gate_candidates",
        ):
            totals[key] += int(row.get(key, 0))
        rejection_reasons.update(row.get("rejection_reasons", {}))
    return {
        **dict(totals),
        "input_summaries": len(summaries),
        "rejection_reasons": dict(rejection_reasons.most_common()),
    }


def aggregate(root: Path, output: Path, *, retain_top: int) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    top = _frames(_files(root, "top_candidates.csv"))
    gates = _frames(_files(root, "non_global_gate_candidates.csv"))
    for frame, name, limit in (
        (top, "top_candidates.csv", retain_top),
        (gates, "non_global_gate_candidates.csv", None),
    ):
        if len(frame):
            frame = frame.sort_values(
                ["train_oof_cagr_pct", "strategy_id"],
                ascending=[False, True],
                kind="mergesort",
            ).drop_duplicates("canonical_hash", keep="first")
            if limit is not None:
                frame = frame.head(limit)
        frame.to_csv(output / name, index=False)

    family_source = _frames(_files(root, "family_runtime.csv"))
    family_rows = []
    if len(family_source):
        count_columns = [
            column
            for column in ("generated", "evaluated", "rejected", "deduped")
            if column in family_source
        ]
        for family, group in family_source.groupby("family", sort=True):
            evaluated = pd.to_numeric(group.get("evaluated", 0), errors="coerce").fillna(0)
            means = pd.to_numeric(group.get("mean_seconds"), errors="coerce")
            weighted = float((means.fillna(0.0) * evaluated).sum())
            total_evaluated = int(evaluated.sum())
            family_rows.append(
                {
                    "family": family,
                    **{
                        column: int(pd.to_numeric(group[column], errors="coerce").fillna(0).sum())
                        for column in count_columns
                    },
                    "mean_seconds": weighted / total_evaluated if total_evaluated else None,
                    "p95_seconds": float(
                        pd.to_numeric(group.get("p95_seconds"), errors="coerce").max()
                    ),
                }
            )
    pd.DataFrame(family_rows).to_csv(output / "family_runtime.csv", index=False)

    generated_hashes, _ = _unique_hash_values(_files(root, "canonical_hashes.npy"))
    evaluated_hashes, pvalues = _unique_hash_values(
        _files(root, "evaluated_hashes.npy"), "raw_pvalues.npy"
    )
    np.save(output / "canonical_hashes.npy", generated_hashes)
    np.save(output / "evaluated_hashes.npy", evaluated_hashes)
    np.save(output / "raw_pvalues.npy", pvalues)
    bootstrap = _merge_bootstrap(
        _files(root, "bootstrap_accumulator.npz"),
        output / "bootstrap_accumulator.npz",
    )
    pbo = _merge_pbo(
        _files(root, "pbo_accumulator.npz"),
        output / "pbo_accumulator.npz",
    )
    summary = {
        **_merge_summaries(root),
        **bootstrap,
        **pbo,
        "unique_effective_generated": int(len(generated_hashes)),
        "unique_effective_evaluated": int(len(evaluated_hashes)),
        "retained_top_rows": int(len(top)),
        "families_observed": int(len(family_rows)),
        "campaign_version": CAMPAIGN_VERSION,
        "train_end": TRAIN_END,
        "validation_opened": False,
        "locked_start": LOCKED_START,
        "locked_opened": False,
    }
    (output / "aggregate_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _by_qvalues(pvalues: np.ndarray, total_trials: int) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(pvalues, dtype=float)
    order = np.argsort(raw, kind="mergesort")
    ordered = raw[order]
    harmonic = sum(1.0 / rank for rank in range(1, total_trials + 1))
    adjusted = np.minimum(1.0, ordered * total_trials * harmonic / np.arange(1, len(ordered) + 1))
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = adjusted
    return result, order


def _multiplicity_counts(
    prior_ledger: pd.DataFrame,
    all_hashes: np.ndarray,
    evaluated_hashes: np.ndarray,
) -> dict[str, int]:
    prior_unique = int(prior_ledger["canonical_hash"].nunique())
    prior_evaluated = prior_ledger.loc[prior_ledger["status"].eq("evaluated")]
    prior_evaluated_unique = int(prior_evaluated["canonical_hash"].nunique())
    new_unique = int(len(all_hashes)) - prior_unique
    new_evaluated_unique = int(len(evaluated_hashes)) - prior_evaluated_unique
    if new_unique < 0 or new_evaluated_unique < 0:
        raise RuntimeError("MASSIVE_PRIOR_MULTIPLICITY_COUNT_BREACH")
    return {
        "prior_declared_trials": int(len(prior_ledger)),
        "prior_unique_rules": prior_unique,
        "prior_evaluated_streams": int(len(prior_evaluated)),
        "prior_evaluated_unique_rules": prior_evaluated_unique,
        "new_unique_trials": new_unique,
        "new_evaluated_unique_rules": new_evaluated_unique,
        "effective_unique_rules": int(len(all_hashes)),
        "effective_unique_evaluated_rules": int(len(evaluated_hashes)),
        "trials_for_multiplicity": int(len(prior_ledger)) + new_unique,
        "evaluated_streams": int(len(prior_evaluated)) + new_evaluated_unique,
    }


def finalize(root: Path, output: Path, prior_root: Path, *, retain_top: int) -> dict[str, Any]:
    summary = aggregate(root, output, retain_top=retain_top)
    prior_ledger = pd.read_parquet(prior_root / "autonomous_trial_ledger.parquet")
    all_hashes = np.load(output / "canonical_hashes.npy", allow_pickle=False)
    evaluated_hashes = np.load(output / "evaluated_hashes.npy", allow_pickle=False)
    counts = _multiplicity_counts(prior_ledger, all_hashes, evaluated_hashes)
    total_trials = counts["trials_for_multiplicity"]
    all_pvalues = np.load(output / "raw_pvalues.npy", allow_pickle=False)
    qvalues, _ = _by_qvalues(all_pvalues, total_trials)
    qvalue_by_hash = {
        value.tobytes().hex(): float(qvalue)
        for value, qvalue in zip(evaluated_hashes, qvalues, strict=True)
    }
    gates_path = output / "non_global_gate_candidates.csv"
    gates = pd.read_csv(gates_path) if gates_path.stat().st_size else pd.DataFrame()
    if len(gates):
        gates["dsr"] = gates.apply(
            lambda row: dsr_from_moments(
                sessions=int(row["oof_sessions"]),
                mean=float(row["return_mean"]),
                std=float(row["return_std"]),
                skew=float(row["return_skew"]),
                kurtosis=float(row["return_kurtosis"]),
                trials=total_trials,
            ),
            axis=1,
        )
        gates["fdr_qvalue_by"] = gates["canonical_hash"].map(qvalue_by_hash).fillna(1.0)
        gates["wrc_pvalue"] = summary["white_reality_check_pvalue"]
        gates["global_hansen_spa_pvalue"] = summary["global_hansen_spa_pvalue"]
        gates["pbo"] = summary["pbo_conservative_upper"]
        gates["gate_dsr"] = gates["dsr"] >= 0.95
        gates["gate_spa"] = gates["spa_pvalue"] <= 0.10
        gates["gate_wrc"] = gates["wrc_pvalue"] <= 0.10
        gates["gate_hansen_spa"] = gates["global_hansen_spa_pvalue"] <= 0.10
        gates["gate_fdr_by"] = gates["fdr_qvalue_by"] <= 0.10
        gates["gate_pbo"] = gates["pbo"] <= 0.50
        gates["eligible_for_freeze"] = gates[
            ["gate_dsr", "gate_spa", "gate_wrc", "gate_hansen_spa", "gate_fdr_by", "gate_pbo"]
        ].all(axis=1)
        gates = gates.sort_values(
            ["eligible_for_freeze", "train_oof_cagr_pct", "strategy_id"],
            ascending=[False, False, True],
            kind="mergesort",
        )
    gates.to_csv(output / "final_train_candidates.csv", index=False)
    finalists = gates[gates.get("eligible_for_freeze", False)] if len(gates) else gates
    top = pd.read_csv(output / "top_candidates.csv")
    leaderboard = pd.concat([gates, top], ignore_index=True).drop_duplicates(
        "canonical_hash", keep="first"
    ) if len(top) else gates
    leaderboard = leaderboard.sort_values(
        ["train_oof_cagr_pct", "strategy_id"],
        ascending=[False, True],
        kind="mergesort",
    )
    leaderboard.to_csv(output / "leaderboard.csv", index=False)
    annual_rows = []
    for row in leaderboard.head(50_000).itertuples(index=False):
        payload = json.loads(str(row.train_annual_metrics_json))
        for annual in payload:
            annual_rows.append(
                {
                    "strategy_id": row.strategy_id,
                    "family": row.family,
                    **annual,
                }
            )
    pd.DataFrame(annual_rows).to_csv(output / "train_annual_metrics.csv", index=False)
    with (output / "candidate_rules.jsonl").open("w", encoding="utf-8") as handle:
        for value in leaderboard.head(50_000)["candidate_json"]:
            handle.write(json.dumps(json.loads(str(value)), sort_keys=True) + "\n")
    runtime_path = output / "family_runtime.csv"
    runtime = pd.read_csv(runtime_path) if runtime_path.is_file() and runtime_path.stat().st_size else pd.DataFrame()
    leaders = (
        leaderboard.groupby("family", as_index=False)
        .agg(
            retained_candidates=("strategy_id", "size"),
            best_train_oof_cagr_pct=("train_oof_cagr_pct", "max"),
            best_train_oof_sharpe=("train_oof_sharpe", "max"),
        )
        if len(leaderboard)
        else pd.DataFrame()
    )
    family_summary = (
        runtime.merge(leaders, on="family", how="outer")
        if len(runtime) and len(leaders)
        else runtime if len(runtime) else leaders
    )
    family_summary.to_csv(output / "family_summary.csv", index=False)
    final_summary = {
        **summary,
        "github_only": True,
        "requires_local_machine": False,
        "total_prior_trials": counts["prior_declared_trials"],
        "total_prior_unique_rules": counts["prior_unique_rules"],
        "total_prior_evaluated_streams": counts["prior_evaluated_streams"],
        "total_unique_trials": counts["effective_unique_rules"],
        "total_trials_for_multiplicity": total_trials,
        "total_new_unique_trials": counts["new_unique_trials"],
        "total_strategies_loaded": total_trials,
        "total_strategies_evaluated": counts["evaluated_streams"],
        "total_effective_unique_evaluated_rules": counts[
            "effective_unique_evaluated_rules"
        ],
        "total_strategies_rejected_or_unsupported": total_trials
        - counts["evaluated_streams"],
        "total_new_generated": int(summary.get("generated", 0)),
        "total_new_evaluated": int(summary.get("evaluated", 0)),
        "total_new_deduped": int(summary.get("deduped", 0)),
        "total_new_runtime_errors": int(summary.get("runtime_errors", 0)),
        "finalist_count": int(len(finalists)),
        "retained_leaderboard_rows": int(len(leaderboard)),
        "search_waves": 7,
        "parallel_github_jobs": 360,
        "python_processes_per_job": 4,
        "maximum_parallel_python_processes": 1440,
        "best_candidate_id": str(leaderboard.iloc[0]["strategy_id"]) if len(leaderboard) else None,
        "best_train_oof_cagr_pct": float(leaderboard.iloc[0]["train_oof_cagr_pct"]) if len(leaderboard) else None,
        "best_finalist_id": str(finalists.iloc[0]["strategy_id"]) if len(finalists) else None,
        "train_end": TRAIN_END,
        "validation_start": VALIDATION_START,
        "validation_end": VALIDATION_END,
        "validation_opened": False,
        "locked_start": LOCKED_START,
        "locked_opened": False,
        "result_status": "TRAIN_TARGET_FOUND_VALIDATION_NOT_OPENED" if len(finalists) else "TRAIN_SEARCH_CONTINUES",
    }
    (output / "summary.json").write_text(
        json.dumps(final_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return final_summary


def main() -> int:
    args = parse_args()
    root = args.input_root.resolve()
    output = args.output_dir.resolve()
    if args.mode == "aggregate":
        result = aggregate(root, output, retain_top=args.retain_top)
    else:
        if args.prior_root is None:
            raise ValueError("FINAL_MERGE_REQUIRES_PRIOR_ROOT")
        result = finalize(root, output, args.prior_root.resolve(), retain_top=args.retain_top)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
