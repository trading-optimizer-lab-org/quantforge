"""Train OOF metrics and multiplicity gates for autonomous batches."""

from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

from aurora.infra.sp500_long_short_daily.statistics import (
    cscv_pbo,
    deflated_sharpe_probability,
)
from aurora.infra.sp500_long_short_daily.contracts import canonical_json_hash

from .contracts import BLOCK_LENGTH, BOOTSTRAP_REPETITIONS, PREVIOUS_TRIAL_COUNT
from .historical_evidence import (
    HISTORICAL_DIR,
    load_prior_autonomous_returns,
    load_prior_candidate_returns,
)


def _nav_metrics(values: Sequence[float] | np.ndarray) -> dict[str, float | int | None]:
    raw = np.asarray(values, dtype=float)
    raw = raw[np.isfinite(raw)]
    if raw.size == 0:
        return {"sessions": 0, "total_return_pct": None, "cagr_pct": None, "sharpe": None, "sortino": None, "max_drawdown_pct": None, "calmar": None}
    nav = np.cumprod(1.0 + raw)
    final = float(nav[-1])
    years = len(raw) / 252.0
    cagr = final ** (1.0 / years) - 1.0 if final > 0.0 and years > 0.0 else -1.0
    std = float(np.std(raw, ddof=0))
    downside = raw[raw < 0.0]
    downside_std = float(np.std(downside, ddof=0)) if len(downside) else 0.0
    peak = np.maximum.accumulate(nav)
    drawdown = nav / peak - 1.0
    mdd = float(drawdown.min())
    return {
        "sessions": int(len(raw)),
        "total_return_pct": (final - 1.0) * 100.0,
        "cagr_pct": cagr * 100.0,
        "sharpe": float(np.mean(raw) / std * math.sqrt(252.0)) if std > 1e-15 else 0.0,
        "sortino": float(np.mean(raw) / downside_std * math.sqrt(252.0)) if downside_std > 1e-15 else None,
        "max_drawdown_pct": mdd * 100.0,
        "calmar": float(cagr / abs(mdd)) if mdd < -1e-15 else None,
    }


def _annual_rows(dates: Sequence[str], values: Sequence[float] | np.ndarray) -> list[dict[str, Any]]:
    frame = pd.DataFrame({"date": pd.to_datetime(list(dates)), "return": list(values)})
    rows: list[dict[str, Any]] = []
    for year, group in frame.groupby(frame["date"].dt.year, sort=True):
        metrics = _nav_metrics(group["return"].to_numpy(dtype=float))
        rows.append({"year": int(year), **metrics, "positive": bool((metrics["total_return_pct"] or 0.0) > 0.0)})
    return rows


def _normal_pvalue(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if len(values) < 3:
        return 1.0
    std = float(np.std(values, ddof=1))
    if std <= 1e-15:
        return 0.0 if float(np.mean(values)) > 0.0 else 1.0
    return float(1.0 - norm.cdf(float(np.mean(values)) / (std / math.sqrt(len(values)))))


def _block_bootstrap_pvalue(values: np.ndarray, *, seed: int) -> float:
    """Stationary-ish circular block bootstrap with declared repetitions."""

    raw = np.asarray(values, dtype=float)
    raw = raw[np.isfinite(raw)]
    if len(raw) < 100:
        return _normal_pvalue(raw)
    rng = np.random.default_rng(seed)
    blocks = int(math.ceil(len(raw) / BLOCK_LENGTH))
    full_blocks, remainder = divmod(len(raw), BLOCK_LENGTH)
    observed = float(raw.mean())
    centered = raw - observed
    circular = np.arange(len(raw))[:, None]
    full_sums = centered[
        (circular + np.arange(BLOCK_LENGTH)) % len(raw)
    ].sum(axis=1)
    partial_sums = (
        centered[(circular + np.arange(remainder)) % len(raw)].sum(axis=1)
        if remainder
        else None
    )
    exceed = 0
    chunk = 250
    for start in range(0, BOOTSTRAP_REPETITIONS, chunk):
        count = min(chunk, BOOTSTRAP_REPETITIONS - start)
        starts = rng.integers(0, len(raw), size=(count, blocks))
        totals = full_sums[starts[:, :full_blocks]].sum(axis=1)
        if remainder and partial_sums is not None:
            totals += partial_sums[starts[:, full_blocks]]
        means = totals / len(raw)
        exceed += int(np.count_nonzero(means >= observed))
    return float((1 + exceed) / (BOOTSTRAP_REPETITIONS + 1))


def _global_bootstrap_pvalues(matrix: np.ndarray, *, seed: int) -> tuple[float, float]:
    """Return White and studentized Hansen SPA p-values using block sums."""

    values = np.asarray(matrix, dtype=float)
    values = values[np.all(np.isfinite(values), axis=1)]
    if values.ndim != 2 or values.shape[0] < 100 or values.shape[1] == 0:
        return 1.0, 1.0
    observed = values.mean(axis=0)
    observed_max = float(observed.max())
    observed_se = values.std(axis=0, ddof=1) / math.sqrt(len(values))
    observed_t = np.divide(
        observed,
        observed_se,
        out=np.zeros_like(observed),
        where=observed_se > 0.0,
    )
    observed_spa = float(observed_t.max())
    centered = values - observed
    blocks = int(math.ceil(len(values) / BLOCK_LENGTH))
    full_blocks, remainder = divmod(len(values), BLOCK_LENGTH)
    circular = np.arange(len(values))[:, None]
    full_sums = centered[
        (circular + np.arange(BLOCK_LENGTH)) % len(values)
    ].sum(axis=1)
    full_square_sums = np.square(centered)[
        (circular + np.arange(BLOCK_LENGTH)) % len(values)
    ].sum(axis=1)
    partial_sums = (
        centered[(circular + np.arange(remainder)) % len(values)].sum(axis=1)
        if remainder
        else None
    )
    partial_square_sums = (
        np.square(centered)[
            (circular + np.arange(remainder)) % len(values)
        ].sum(axis=1)
        if remainder
        else None
    )
    rng = np.random.default_rng(seed)
    white_exceed = 0
    spa_exceed = 0
    chunk = 25
    for start in range(0, BOOTSTRAP_REPETITIONS, chunk):
        count = min(chunk, BOOTSTRAP_REPETITIONS - start)
        starts = rng.integers(0, len(values), size=(count, blocks))
        totals = full_sums[starts[:, :full_blocks]].sum(axis=1)
        square_totals = full_square_sums[starts[:, :full_blocks]].sum(axis=1)
        if remainder and partial_sums is not None:
            totals += partial_sums[starts[:, full_blocks]]
            assert partial_square_sums is not None
            square_totals += partial_square_sums[starts[:, full_blocks]]
        means = totals / len(values)
        variance = np.maximum(
            (square_totals - np.square(totals) / len(values))
            / max(len(values) - 1, 1),
            0.0,
        )
        standard_error = np.sqrt(variance) / math.sqrt(len(values))
        studentized = np.divide(
            means,
            standard_error,
            out=np.zeros_like(means),
            where=standard_error > 0.0,
        )
        white_exceed += int(
            np.count_nonzero(means.max(axis=1) >= observed_max)
        )
        spa_exceed += int(
            np.count_nonzero(studentized.max(axis=1) >= observed_spa)
        )
    denominator = BOOTSTRAP_REPETITIONS + 1
    return (
        float((1 + white_exceed) / denominator),
        float((1 + spa_exceed) / denominator),
    )


def _global_max_bootstrap_pvalue(matrix: np.ndarray, *, seed: int) -> float:
    """Compatibility wrapper returning the White max-statistic p-value."""

    return _global_bootstrap_pvalues(matrix, seed=seed)[0]


def _benjamini_yekutieli(pvalues: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(((str(k), min(max(float(v), 0.0), 1.0)) for k, v in pvalues.items()), key=lambda item: (item[1], item[0]))
    if not ordered:
        return {}
    m = len(ordered)
    harmonic = sum(1.0 / rank for rank in range(1, m + 1))
    adjusted = [1.0] * m
    running = 1.0
    for index in range(m - 1, -1, -1):
        rank = index + 1
        running = min(running, ordered[index][1] * m * harmonic / rank)
        adjusted[index] = min(max(running, 0.0), 1.0)
    return {ordered[index][0]: adjusted[index] for index in range(m)}


def _share_of_positive_log_profit(annual: Sequence[Mapping[str, Any]]) -> float:
    profits = [math.log1p(max(float(row.get("total_return_pct") or 0.0) / 100.0, -0.999999)) for row in annual]
    positive = [value for value in profits if value > 0.0]
    return float(max(positive) / sum(positive)) if positive and sum(positive) > 0.0 else 1.0


def _candidate_metric(row: Mapping[str, Any], benchmark: np.ndarray, global_trials: int, seed: int) -> tuple[dict[str, Any], np.ndarray]:
    candidate_id = str(row["strategy_id"])
    values = np.asarray(row.get("train_returns") or [], dtype=float)
    dates = list(row.get("train_dates") or [])
    annual = _annual_rows(dates, values)
    metrics = _nav_metrics(values)
    positive_years = sum(bool(item["positive"]) for item in annual)
    covered_years = len(annual)
    diff = values[: len(benchmark)] - benchmark[: len(values)]
    stable_offset = int(hashlib.sha256(candidate_id.encode("utf-8")).hexdigest()[:8], 16)
    pvalue = _block_bootstrap_pvalue(diff, seed=seed + stable_offset % 100_000)
    dsr = deflated_sharpe_probability(values, trials=max(global_trials, 1)) if len(values) else 0.0
    train_oof_cagr = float(metrics["cagr_pct"] or -100.0)
    positive_fraction = positive_years / covered_years if covered_years else 0.0
    median_fold = float(np.median([float(item["cagr_pct"] or -100.0) for item in annual])) if annual else -100.0
    result: dict[str, Any] = {
        "strategy_id": candidate_id,
        "unit_key": str(row["unit_key"]),
        "family": str(row["family"]),
        "canonical_hash": str(row["canonical_hash"]),
        "status": str(row["status"]),
        "train_oof_cagr_pct": train_oof_cagr,
        "train_oof_total_return_pct": metrics["total_return_pct"],
        "train_oof_sharpe": metrics["sharpe"],
        "train_oof_sortino": metrics["sortino"],
        "train_oof_calmar": metrics["calmar"],
        "train_oof_max_drawdown_pct": metrics["max_drawdown_pct"],
        "oof_sessions": int(len(values)),
        "covered_years": covered_years,
        "positive_years": positive_years,
        "positive_fold_fraction": positive_fraction,
        "median_fold_cagr_pct": median_fold,
        "single_year_log_profit_share": _share_of_positive_log_profit(annual),
        "dsr": dsr,
        "spa_pvalue": pvalue,
        "raw_pvalue": pvalue,
        "train_annual_metrics_json": json.dumps(annual, sort_keys=True, separators=(",", ":")),
        "benchmark": "buy_and_hold_spy_total_return",
    }
    return result, diff


def evaluate_batch(
    rows: Sequence[Mapping[str, Any]],
    root: Path,
    *,
    batch_id: int,
    previous_trial_count: int = PREVIOUS_TRIAL_COUNT,
    prepared_root: Path | None = None,
) -> dict[str, Any]:
    """Write auditable train metrics, multiplicity results, and freeze evidence."""

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    candidate_rows = [row for row in rows if row.get("unit_type") == "candidate"]
    candidates = [row for row in candidate_rows if row.get("status") == "evaluated"]
    rejected_candidates = [row for row in candidate_rows if row.get("status") != "evaluated"]
    benchmarks = {str(row["strategy_id"]): row for row in rows if row.get("unit_type") == "benchmark" and row.get("status") == "evaluated"}
    benchmark_row = benchmarks.get("buy_and_hold_spy_total_return") or benchmarks.get("always_long")
    benchmark = np.asarray((benchmark_row or {}).get("train_returns") or [], dtype=float)
    metrics: list[dict[str, Any]] = []
    differential_rows: dict[str, np.ndarray] = {}
    if previous_trial_count < PREVIOUS_TRIAL_COUNT:
        raise ValueError("PREVIOUS_TRIAL_COUNT_MOVED_BACKWARD")
    global_trials = previous_trial_count + len([row for row in rows if row.get("unit_type") == "candidate"])
    for index, row in enumerate(candidates):
        result, diff = _candidate_metric(row, benchmark, global_trials, batch_id * 100_000 + index)
        metrics.append(result)
        differential_rows[str(result["strategy_id"])] = diff
    metric_frame = pd.DataFrame(metrics)
    current_pvalues = {
        str(row["strategy_id"]): float(row["raw_pvalue"]) for row in metrics
    }
    prepared = Path(prepared_root) if prepared_root is not None else None
    ledger_rows: list[dict[str, Any]] = []
    if prepared is not None and (prepared / "trial_ledger.jsonl").is_file():
        ledger_rows = [
            json.loads(line)
            for line in (prepared / "trial_ledger.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]

    current_daily = pd.DataFrame(
        [
            {
                "strategy_id": str(row["strategy_id"]),
                "date": pd.Timestamp(date),
                "return": float(value),
            }
            for row in candidates
            for date, value in zip(
                row.get("train_dates") or [], row.get("train_returns") or [], strict=True
            )
        ],
        columns=["strategy_id", "date", "return"],
    )
    prior_daily = pd.DataFrame(columns=["strategy_id", "date", "return"])
    historical_available = bool(
        prepared is not None
        and (prepared / HISTORICAL_DIR / "historical_evidence_manifest.json").is_file()
    )
    if historical_available and prepared is not None:
        prior_daily = load_prior_candidate_returns(prepared)
    combined_daily = pd.concat([prior_daily, current_daily], ignore_index=True)
    combined_wide = (
        combined_daily.pivot(index="date", columns="strategy_id", values="return")
        if len(combined_daily)
        else pd.DataFrame()
    )
    benchmark_dates = pd.to_datetime(list((benchmark_row or {}).get("train_dates") or []))
    benchmark_series = pd.Series(
        benchmark,
        index=benchmark_dates,
        name="__benchmark__",
        dtype=float,
    )
    common = (
        combined_wide.join(benchmark_series, how="inner").dropna(axis=0, how="any")
        if len(combined_wide) and len(benchmark_series)
        else pd.DataFrame()
    )
    combined_differential = (
        common.drop(columns=["__benchmark__"]).subtract(
            common["__benchmark__"], axis=0
        )
        if len(common)
        else pd.DataFrame()
    )
    evaluated_ledger_ids = {
        str(row["strategy_id"])
        for row in ledger_rows
        if str(row.get("status")) == "evaluated"
        or str(row["strategy_id"]) in current_pvalues
    }
    stream_ids = set(combined_wide.columns.astype(str))
    declared_pvalues: dict[str, float] = {}
    for row in ledger_rows:
        identifier = str(row["strategy_id"])
        is_evaluated = (
            identifier in current_pvalues or str(row.get("status")) == "evaluated"
        )
        if is_evaluated and identifier in combined_wide and len(benchmark_series):
            aligned = pd.concat(
                [combined_wide[identifier], benchmark_series], axis=1, join="inner"
            ).dropna()
            declared_pvalues[identifier] = _normal_pvalue(
                aligned[identifier].to_numpy(dtype=float)
                - aligned["__benchmark__"].to_numpy(dtype=float)
            )
        else:
            declared_pvalues[identifier] = 1.0
    qvalues = _benjamini_yekutieli(declared_pvalues or current_pvalues)
    multiplicity_complete = bool(
        historical_available
        and len(ledger_rows) == global_trials
        and set(range(1, global_trials + 1))
        == {int(row["global_trial_index"]) for row in ledger_rows}
        and evaluated_ledger_ids.issubset(stream_ids)
        and len(declared_pvalues) == global_trials
        and len(common) >= 1500
    )
    wrc_pvalue = 1.0
    hansen_spa_pvalue = 1.0
    global_pbo = 1.0
    if len(metric_frame):
        metric_frame["fdr_raw_pvalue"] = (
            metric_frame["strategy_id"].map(declared_pvalues).fillna(1.0)
        )
        metric_frame["fdr_qvalue_by"] = metric_frame["strategy_id"].map(qvalues).fillna(1.0)
        wrc_pvalue, hansen_spa_pvalue = _global_bootstrap_pvalues(
            combined_differential.to_numpy(dtype=float)
            if multiplicity_complete
            else np.empty((0, 0)),
            seed=batch_id * 100_000 + 77,
        )
        metric_frame["wrc_pvalue"] = wrc_pvalue
        metric_frame["global_hansen_spa_pvalue"] = hansen_spa_pvalue
        metric_frame["pbo"] = 1.0
        matrix = common.drop(columns=["__benchmark__"]) if multiplicity_complete else pd.DataFrame()
        if matrix.shape[1] >= 2 and len(matrix) >= 200:
            try:
                pbo_payload = cscv_pbo(matrix, partitions=10)
                global_pbo = float(pbo_payload.get("pbo")) if pbo_payload.get("pbo") is not None else 1.0
                metric_frame["pbo"] = global_pbo
            except (ValueError, FloatingPointError):
                global_pbo = 1.0
                metric_frame["pbo"] = 1.0
    else:
        metric_frame = pd.DataFrame(columns=["strategy_id", "train_oof_cagr_pct"])
    if len(metric_frame):
        metric_frame["gate_target_cagr"] = metric_frame["train_oof_cagr_pct"] > 20.0
        metric_frame["gate_oof_sessions"] = metric_frame["oof_sessions"] >= 2500
        metric_frame["gate_covered_years"] = metric_frame["covered_years"] >= 10
        metric_frame["gate_positive_folds"] = metric_frame["positive_fold_fraction"] >= 0.60
        metric_frame["gate_median_fold_cagr"] = metric_frame["median_fold_cagr_pct"] > 0.0
        metric_frame["gate_profit_concentration"] = metric_frame["single_year_log_profit_share"] <= 0.50
        metric_frame["gate_dsr"] = metric_frame["dsr"] >= 0.95
        metric_frame["gate_spa"] = metric_frame["spa_pvalue"] <= 0.10
        metric_frame["gate_wrc"] = metric_frame["wrc_pvalue"] <= 0.10
        metric_frame["gate_hansen_spa"] = metric_frame["global_hansen_spa_pvalue"] <= 0.10
        metric_frame["gate_fdr_by"] = metric_frame["fdr_qvalue_by"] <= 0.10
        metric_frame["gate_pbo"] = metric_frame["pbo"] <= 0.50
        gate_columns = [column for column in metric_frame.columns if column.startswith("gate_")]
        metric_frame["eligible_for_freeze"] = metric_frame[gate_columns].all(axis=1)
        metric_frame = metric_frame.sort_values(["eligible_for_freeze", "train_oof_cagr_pct", "strategy_id"], ascending=[False, False, True], kind="mergesort").reset_index(drop=True)
        metric_frame["train_rank"] = np.arange(1, len(metric_frame) + 1)
    metric_frame.to_csv(root / "train_oof_metrics.csv", index=False)
    metric_frame.to_csv(root / "candidate_metrics.csv", index=False)
    annual_rows: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    for row in rows:
        if row.get("status") != "evaluated":
            continue
        identifier = str(row["strategy_id"])
        for date, value, position in zip(row.get("train_dates") or [], row.get("train_returns") or [], row.get("train_positions") or [], strict=True):
            daily_rows.append({"strategy_id": identifier, "unit_key": str(row["unit_key"]), "date": date, "return": value, "position": position, "out_of_fold": True})
        annual_rows.extend({"strategy_id": identifier, "unit_key": str(row["unit_key"]), **item, "out_of_fold": True} for item in json.loads(row.get("annual_metrics_json") or "[]"))
    daily_frame = pd.DataFrame(
        daily_rows,
        columns=["strategy_id", "unit_key", "date", "return", "position", "out_of_fold"],
    )
    daily_frame.to_parquet(root / "train_oof_daily_returns.parquet", index=False)
    pd.DataFrame(annual_rows).to_csv(root / "train_annual_metrics.csv", index=False)
    pd.DataFrame(annual_rows).to_csv(root / "train_fold_metrics.csv", index=False)
    pd.DataFrame([{key: value for key, value in row.items() if key not in {"train_dates", "train_returns", "train_positions", "annual_metrics_json", "performance_by_market_regime_json"}} for row in rows]).to_csv(root / "leaderboard.csv", index=False)
    timing_columns = [
        "strategy_id",
        "unit_key",
        "family",
        "seconds_total",
        "seconds_signal",
        "seconds_simulation",
        "status",
        "rejection_reason",
    ]
    pd.DataFrame(
        [{column: row.get(column) for column in timing_columns} for row in rows],
        columns=timing_columns,
    ).to_csv(root / "timing_diagnostics.csv", index=False)
    pd.DataFrame([{ "strategy_id": row.get("strategy_id"), "status": row.get("status"), "reason": row.get("rejection_reason"), "family": row.get("family") } for row in rows if row.get("status") != "evaluated"]).to_csv(root / "rejections.csv", index=False)
    top = metric_frame.head(30).to_dict("records") if len(metric_frame) else []
    (root / "top_candidates.json").write_text(json.dumps(top, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    pvalues_frame = metric_frame[[column for column in ["strategy_id", "raw_pvalue", "spa_pvalue", "fdr_raw_pvalue", "fdr_qvalue_by", "wrc_pvalue", "pbo", "dsr"] if column in metric_frame]].copy()
    pvalues_frame.to_csv(root / "multiple_testing.csv", index=False)
    multiple = {
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "bootstrap_method": "circular_block_bootstrap",
        "block_length": BLOCK_LENGTH,
        "block_length_justification": "approximately one trading month; sensitivity is recorded by the batch contract",
        "correction": "Benjamini-Yekutieli",
        "global_trial_count": global_trials,
        "white_reality_check": "global circular-block maximum-statistic bootstrap p-value",
        "hansen_spa": "candidate circular-block studentized proxy plus global maximum-statistic p-value",
        "cscv_pbo": "10 chronological partitions when enough common observations exist",
        "white_reality_check_pvalue": wrc_pvalue,
        "hansen_spa_pvalue": hansen_spa_pvalue,
        "pbo": global_pbo,
        "combined_multiplicity_complete": multiplicity_complete,
        "historical_trials_loaded": min(len(ledger_rows), PREVIOUS_TRIAL_COUNT),
        "declared_pvalues": len(declared_pvalues),
        "evaluated_streams": len(stream_ids),
        "common_interval_sessions": int(len(common)),
        "common_interval_start": (
            common.index.min().date().isoformat() if len(common) else None
        ),
        "common_interval_end": (
            common.index.max().date().isoformat() if len(common) else None
        ),
    }
    (root / "multiple_testing.json").write_text(json.dumps(multiple, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    finalists = (
        json.loads(
            metric_frame.loc[metric_frame["eligible_for_freeze"]]
            .head(30)
            .to_json(orient="records")
        )
        if len(metric_frame) and "eligible_for_freeze" in metric_frame
        else []
    )
    freeze = {
        "freeze_status": "eligible" if finalists else "not_eligible",
        "batch_id": batch_id,
        "selection_closed": True,
        "validation_opened": False,
        "locked_opened": False,
        "validation_used_for_selection": False,
        "train_end": "2010-12-31",
        "validation_start": "2011-01-01",
        "validation_end": "2020-12-31",
        "locked_start": "2021-01-01",
        "finalists": finalists,
        "candidate_ids": [str(item["strategy_id"]) for item in finalists],
    }
    freeze["freeze_sha256"] = canonical_json_hash(freeze)
    freeze_payload = json.dumps(freeze, indent=2, sort_keys=True, allow_nan=False) + "\n"
    (root / "train_selection_freeze.json").write_text(freeze_payload, encoding="utf-8")
    (root / "train_freeze_candidate.json").write_text(freeze_payload, encoding="utf-8")
    current_status = {str(row["strategy_id"]): row for row in candidate_rows}
    metric_lookup = {
        str(row["strategy_id"]): row for row in metric_frame.to_dict("records")
    }
    finalized_ledger: list[dict[str, Any]] = []
    for ledger_row in ledger_rows:
        item = dict(ledger_row)
        identifier = str(item["strategy_id"])
        observed = current_status.get(identifier)
        if observed is not None:
            item["status"] = str(observed.get("status"))
            item["rejection_reason"] = str(observed.get("rejection_reason") or "")
        metric = metric_lookup.get(identifier)
        if metric is not None:
            item["fdr_pvalue"] = float(metric["fdr_raw_pvalue"])
            item["fdr_qvalue_by"] = float(metric["fdr_qvalue_by"])
        finalized_ledger.append(item)
    if finalized_ledger:
        ledger_payload = "".join(
            json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n"
            for item in finalized_ledger
        )
        (root / "trial_ledger.jsonl").write_text(ledger_payload, encoding="utf-8")
        pd.DataFrame(finalized_ledger).to_parquet(
            root / "autonomous_trial_ledger.parquet", index=False
        )
        autonomous_status = pd.DataFrame(
            [
                {
                    "strategy_id": item["strategy_id"],
                    "status": item.get("status", "registered"),
                    "rejection_reason": item.get("rejection_reason", ""),
                    "source": f"batch_{item.get('batch_id')}",
                }
                for item in finalized_ledger
                if int(item["global_trial_index"]) > PREVIOUS_TRIAL_COUNT
            ]
        )
        autonomous_status.to_csv(root / "cumulative_autonomous_status.csv", index=False)
    prior_autonomous = (
        load_prior_autonomous_returns(prepared)
        if historical_available and prepared is not None
        else pd.DataFrame(columns=["strategy_id", "date", "return"])
    )
    cumulative_autonomous = pd.concat(
        [prior_autonomous, current_daily], ignore_index=True
    ).drop_duplicates(["strategy_id", "date"], keep="last")
    cumulative_autonomous.to_parquet(
        root / "cumulative_autonomous_train_returns.parquet", index=False
    )

    result_status = (
        "TRAIN_TARGET_FOUND_VALIDATION_NOT_OPENED"
        if finalists
        else "TRAIN_SEARCH_CONTINUES"
        if multiplicity_complete
        else "COMBINED_MULTIPLICITY_INCOMPLETE"
    )
    summary = {
        "schema_version": "1",
        "batch_id": batch_id,
        "result_status": result_status,
        "total_strategies_loaded": len(candidate_rows),
        "total_strategies_evaluated": len(candidates),
        "total_strategies_rejected": len(rejected_candidates),
        "candidate_count": len(candidates),
        "eligible_finalists": len(finalists),
        "best_candidate_id": str(metric_frame.iloc[0]["strategy_id"]) if len(metric_frame) else None,
        "best_train_oof_cagr_pct": float(metric_frame.iloc[0]["train_oof_cagr_pct"]) if len(metric_frame) else None,
        "train_end": "2010-12-31",
        "validation_start": "2011-01-01",
        "validation_end": "2020-12-31",
        "locked_start": "2021-01-01",
        "locked_opened": False,
        "validation_used_for_selection": False,
        "global_trial_count": global_trials,
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "combined_multiplicity_complete": multiplicity_complete,
        "trial_ledger_rows": len(finalized_ledger),
        "evaluated_streams_in_multiplicity": len(stream_ids),
        "finalists": [str(item["strategy_id"]) for item in finalists],
    }
    (root / "autonomous_batch_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return summary
