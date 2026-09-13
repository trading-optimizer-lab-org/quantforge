"""Single, fail-closed validation pass for a frozen train finalist."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from aurora.infra.github_performance.workloads.common import primary_metric_record
from aurora.infra.sp500_long_short_daily.data import load_market_snapshot
from aurora.infra.sp500_long_short_daily.ledger import apply_positions
from aurora.infra.sp500_long_short_daily.signals import (
    CandidateRejected,
    benchmark_decisions,
    candidate_decisions,
)
from aurora.infra.sp500_long_short_daily.workload import (
    BENCHMARK_IDS,
    _annual_rows,
    _extended_metrics,
    _market_regime_states,
)

from aurora.infra.sp500_long_short_daily.contracts import canonical_json_hash

from .contracts import (
    LOCKED_START,
    VALIDATION_END,
    VALIDATION_START,
    VALIDATION_WARMUP_START,
)
from .feature_store import FeatureStore
from .registry import base_package, read_batch_registry


VALIDATION_ACK = "OPEN_VALIDATION_2011_2020_ONCE_AUTONOMOUS"
EXPLORATORY_VALIDATION_ACK = (
    "OPEN_EXPLORATORY_VALIDATION_2011_2020_OWNER_AUTHORIZED"
)


class ValidationGateError(RuntimeError):
    """Raised before validation output exists when the freeze is invalid."""


def _verify_freeze(path: Path, *, require_finalized: bool = False) -> Mapping[str, Any]:
    freeze = json.loads(path.read_text(encoding="utf-8"))
    claimed = str(freeze.get("freeze_sha256", ""))
    content = dict(freeze)
    content.pop("freeze_sha256", None)
    if not claimed or canonical_json_hash(content) != claimed:
        raise ValidationGateError("TRAIN_FREEZE_HASH_MISMATCH")
    if freeze.get("selection_closed") is not True:
        raise ValidationGateError("TRAIN_SELECTION_NOT_CLOSED")
    if freeze.get("validation_opened") is not False or freeze.get("locked_opened") is not False:
        raise ValidationGateError("TRAIN_FREEZE_BOUNDARY_INVALID")
    if freeze.get("train_end") != "2010-12-31" or freeze.get("locked_start") != LOCKED_START:
        raise ValidationGateError("TRAIN_FREEZE_DATES_INVALID")
    if require_finalized:
        if freeze.get("freeze_status") != "eligible" or not freeze.get("finalists"):
            raise ValidationGateError("TRAIN_FREEZE_NOT_ELIGIBLE")
        required_fields = (
            "campaign_version",
            "repository",
            "commit_sha",
            "dataset_hashes",
            "candidate_registry_hash",
            "trial_ledger_hash",
            "total_accumulated_trials",
            "exact_rules",
            "multiple_testing_results",
        )
        if any(not freeze.get(field) for field in required_fields):
            raise ValidationGateError("TRAIN_FREEZE_PROVENANCE_INCOMPLETE")
        if not all(str(value) for value in freeze["dataset_hashes"].values()):
            raise ValidationGateError("TRAIN_FREEZE_DATASET_HASH_MISSING")
        current_sha = os.environ.get("GITHUB_SHA", "")
        frozen_sha = str(freeze.get("commit_sha", ""))
        if current_sha and frozen_sha not in {current_sha, "LOCAL_TEST_ONLY"}:
            raise ValidationGateError("TRAIN_FREEZE_CODE_SHA_MISMATCH")
        for finalist in freeze["finalists"]:
            if finalist.get("eligible_for_freeze") is not True:
                raise ValidationGateError("TRAIN_FREEZE_FINALIST_GATE_MISSING")
    return freeze


def _evaluate(
    data: Any,
    candidate: Mapping[str, Any],
    lookup: Mapping[str, Mapping[str, Any]],
    feature_frame: pd.DataFrame | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    candidate_id = str(candidate["strategy_id"])
    row: dict[str, Any] = {
        "strategy_id": candidate_id,
        "family": str(candidate.get("family", "")),
        "canonical_hash": str(candidate["canonical_hash"]),
        "status": "evaluated",
        "validation_opened": True,
        "validation_used_for_selection": False,
        "locked_opened": False,
    }
    try:
        signal = candidate_decisions(
            candidate,
            data,
            candidate_lookup=lookup,
            feature_frame=feature_frame,
        )
        if signal.first_evaluable_date is None or signal.missing_fraction > 0.02 + 1e-12:
            raise CandidateRejected("DATA_INELIGIBLE:VALIDATION_CAUSAL_COVERAGE")
        applied = apply_positions(data.ledger, signal.decisions)
        selected = applied.loc[
            (applied.index >= pd.Timestamp(VALIDATION_START))
            & (applied.index <= pd.Timestamp(VALIDATION_END))
        ]
        returns = selected["strategy_return"].dropna().astype(float)
        positions = selected.loc[returns.index, "position"].astype(np.int8)
        if len(returns) < 2000 or set(returns.index.year) != set(range(2011, 2021)):
            raise CandidateRejected("DATA_INELIGIBLE:INCOMPLETE_VALIDATION_COVERAGE")
        if not positions.isin((-1, 1)).all():
            raise ValidationGateError("TECHNICAL_FAILURE_POSITION_CONTRACT")
        metrics = primary_metric_record(
            candidate_id,
            "validation",
            returns.to_numpy(dtype=float),
            periods_per_year=252,
        )
        row.update({f"validation_{key}": value for key, value in metrics.reported.items()})
        row.update(
            {
                key.replace("train_", "validation_", 1): value
                for key, value in _extended_metrics(
                    returns,
                    positions,
                    _market_regime_states(data, returns.index),
                ).items()
            }
        )
        daily = [
            {
                "strategy_id": candidate_id,
                "date": date.isoformat(),
                "return": float(value),
                "position": int(position),
            }
            for date, value, position in zip(
                returns.index, returns.to_numpy(), positions.to_numpy(), strict=True
            )
        ]
        annual = [
            {"strategy_id": candidate_id, **item}
            for item in _annual_rows(returns)
        ]
        return row, daily, annual
    except CandidateRejected as exc:
        row["status"] = "rejected"
        row["rejection_reason"] = str(exc)
        return row, [], []


def _candidate_from_registry(
    registry: Sequence[Mapping[str, Any]], strategy_id: str
) -> Mapping[str, Any]:
    matches = [row for row in registry if str(row.get("strategy_id")) == strategy_id]
    if len(matches) != 1:
        raise ValidationGateError("EXPLORATORY_CANDIDATE_NOT_UNIQUE_IN_REGISTRY")
    return matches[0]


def exploratory_validation_package(
    train_results_dir: Path, strategy_id: str
) -> Any:
    """Return the base package scoped to the exact exploratory candidate."""

    registry = read_batch_registry(Path(train_results_dir))
    candidate = _candidate_from_registry(registry, strategy_id)
    return replace(base_package(), candidates=(candidate,))


def run_exploratory_validation(
    *,
    train_results_dir: Path,
    validation_prepared_dir: Path,
    output_dir: Path,
    strategy_id: str,
    validation_ack: str,
) -> Mapping[str, Any]:
    """Evaluate one owner-authorized train candidate without claiming confirmation.

    This deliberately does not weaken or call ``run_validation_once``. The output is
    labelled exploratory because the candidate did not pass the frozen train gates.
    Locked data remain inaccessible.
    """
    if validation_ack != EXPLORATORY_VALIDATION_ACK:
        raise ValidationGateError("EXPLORATORY_VALIDATION_ACK_MISMATCH")
    train_results_dir = Path(train_results_dir).resolve()
    validation_prepared_dir = Path(validation_prepared_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    freeze = _verify_freeze(
        train_results_dir / "train_selection_freeze.json",
        require_finalized=False,
    )
    registry = read_batch_registry(train_results_dir)
    candidate = _candidate_from_registry(registry, strategy_id)
    data = load_market_snapshot(validation_prepared_dir)
    if (
        data.ledger.index.min() < pd.Timestamp(VALIDATION_WARMUP_START)
        or data.ledger.index.max() > pd.Timestamp(VALIDATION_END)
    ):
        raise ValidationGateError("VALIDATION_BOUNDARY_BREACH")

    manifest = json.loads(
        (validation_prepared_dir / "market_data_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    feature_store = FeatureStore(
        dataset_sha256=str(manifest["snapshot_sha256"]),
        code_sha=os.environ.get("GITHUB_SHA", "LOCAL_TEST_ONLY"),
        start=VALIDATION_WARMUP_START,
        end=VALIDATION_END,
    )
    feature_frame = feature_store.get_or_build("SPY", data.ledger)
    (output_dir / "feature_store_manifest.json").write_text(
        json.dumps(feature_store.manifest(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lookup = base_package().candidate_by_id()
    lookup.update({str(row["strategy_id"]): row for row in registry})
    row, daily, annual = _evaluate(
        data,
        candidate,
        lookup,
        feature_frame,
    )
    row["confirmatory_result"] = False
    row["owner_authorized_exploratory"] = True

    rows = [row]
    for benchmark_id in BENCHMARK_IDS:
        signal = benchmark_decisions(benchmark_id, data)
        applied = apply_positions(data.ledger, signal.decisions)
        returns = applied["strategy_return"].dropna().astype(float)
        metric = primary_metric_record(
            f"BENCHMARK::{benchmark_id}",
            "validation",
            returns.to_numpy(),
            periods_per_year=252,
        )
        rows.append(
            {
                "strategy_id": f"BENCHMARK::{benchmark_id}",
                "unit_type": "benchmark",
                "status": "evaluated",
                **{f"validation_{key}": value for key, value in metric.reported.items()},
            }
        )

    pd.DataFrame(rows).to_csv(output_dir / "validation_metrics.csv", index=False)
    pd.DataFrame(annual).to_csv(
        output_dir / "validation_annual_metrics.csv", index=False
    )
    if daily:
        pq.write_table(
            pa.Table.from_pylist(daily),
            output_dir / "validation_daily_returns.parquet",
        )
    else:
        pd.DataFrame(
            columns=["strategy_id", "date", "return", "position"]
        ).to_parquet(output_dir / "validation_daily_returns.parquet", index=False)

    summary = {
        "schema_version": "1",
        "campaign_id": "sp500-autonomous-discovery",
        "result_status": "EXPLORATORY_VALIDATION_RESULT",
        "strategy_id": strategy_id,
        "candidate_status": row["status"],
        "candidate_rejection_reason": row.get("rejection_reason"),
        "confirmatory_result": False,
        "owner_authorized_exploratory": True,
        "official_validation_reservation_created": False,
        "validation_start": VALIDATION_START,
        "validation_end": VALIDATION_END,
        "validation_opened": True,
        "validation_used_for_selection": False,
        "locked_start": LOCKED_START,
        "locked_opened": False,
        "train_freeze_sha256": freeze["freeze_sha256"],
        "code_sha": os.environ.get("GITHUB_SHA", "LOCAL_TEST_ONLY"),
    }
    (output_dir / "validation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "exploratory_candidate.json").write_text(
        json.dumps(candidate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    shutil.copy2(
        train_results_dir / "train_selection_freeze.json",
        output_dir / "train_selection_freeze.json",
    )
    (output_dir / "RESULT_STATUS.md").write_text(
        "# EXPLORATORY_VALIDATION_RESULT\n\n"
        "Owner-authorized exploratory validation. This is not a confirmatory result "
        "because the candidate did not pass every frozen train gate. Locked remained "
        "closed.\n",
        encoding="utf-8",
    )
    return summary


def run_validation_once(
    *,
    train_results_dir: Path,
    train_prepared_dir: Path,
    validation_prepared_dir: Path,
    output_dir: Path,
    validation_ack: str,
) -> Mapping[str, Any]:
    if validation_ack != VALIDATION_ACK:
        raise ValidationGateError("VALIDATION_ACK_MISMATCH")
    train_results_dir = Path(train_results_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    freeze = _verify_freeze(
        train_results_dir / "train_selection_freeze.json",
        require_finalized=True,
    )
    finalists = list(freeze.get("finalists", []))
    if not finalists:
        raise ValidationGateError("NO_FROZEN_FINALISTS_VALIDATION_MUST_NOT_OPEN")
    registry = read_batch_registry(train_results_dir)
    frozen_ids = {str(item["strategy_id"]) for item in finalists}
    candidates = {str(row["strategy_id"]): row for row in registry}
    if frozen_ids - candidates.keys():
        raise ValidationGateError("FROZEN_CANDIDATE_NOT_IN_REGISTRY")
    for item in finalists:
        candidate = candidates[str(item["strategy_id"])]
        if candidate["canonical_hash"] != item.get("canonical_hash"):
            raise ValidationGateError("FROZEN_CANDIDATE_HASH_MISMATCH")
    train = load_market_snapshot(Path(train_prepared_dir))
    validation = load_market_snapshot(Path(validation_prepared_dir))
    if train.ledger.index.max() > pd.Timestamp("2010-12-31"):
        raise ValidationGateError("TRAIN_BOUNDARY_BREACH")
    if validation.ledger.index.min() < pd.Timestamp(VALIDATION_WARMUP_START) or validation.ledger.index.max() > pd.Timestamp(VALIDATION_END):
        raise ValidationGateError("VALIDATION_BOUNDARY_BREACH")
    data = validation
    manifest = json.loads(
        (Path(validation_prepared_dir) / "market_data_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    feature_store = FeatureStore(
        dataset_sha256=str(manifest["snapshot_sha256"]),
        code_sha=os.environ.get("GITHUB_SHA", "LOCAL_TEST_ONLY"),
        start=VALIDATION_WARMUP_START,
        end=VALIDATION_END,
    )
    feature_frame = feature_store.get_or_build("SPY", data.ledger)
    (output_dir / "feature_store_manifest.json").write_text(
        json.dumps(feature_store.manifest(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lookup = base_package().candidate_by_id()
    lookup.update(candidates)
    rows: list[dict[str, Any]] = []
    daily: list[dict[str, Any]] = []
    annual: list[dict[str, Any]] = []
    for item in finalists:
        row, row_daily, row_annual = _evaluate(
            data,
            candidates[str(item["strategy_id"])],
            lookup,
            feature_frame,
        )
        rows.append(row)
        daily.extend(row_daily)
        annual.extend(row_annual)
    for benchmark_id in BENCHMARK_IDS:
        signal = benchmark_decisions(benchmark_id, data)
        applied = apply_positions(data.ledger, signal.decisions)
        returns = applied["strategy_return"].dropna().astype(float)
        metric = primary_metric_record(f"BENCHMARK::{benchmark_id}", "validation", returns.to_numpy(), periods_per_year=252)
        rows.append({"strategy_id": f"BENCHMARK::{benchmark_id}", "unit_type": "benchmark", "status": "evaluated", **{f"validation_{k}": v for k, v in metric.reported.items()}})
    metrics = pd.DataFrame(rows)
    annual_frame = pd.DataFrame(annual)
    metrics.to_csv(output_dir / "validation_metrics.csv", index=False)
    annual_frame.to_csv(output_dir / "validation_annual_metrics.csv", index=False)
    if daily:
        pq.write_table(pa.Table.from_pylist(daily), output_dir / "validation_daily_returns.parquet")
    else:
        pd.DataFrame(columns=["strategy_id", "date", "return", "position"]).to_parquet(output_dir / "validation_daily_returns.parquet", index=False)
    summary = {
        "schema_version": "1",
        "campaign_id": "sp500-autonomous-discovery",
        "result_status": "POSITIVE_VALIDATED_RESULT" if all(row["status"] == "evaluated" for row in rows[: len(finalists)]) else "TECHNICAL_FAILURE",
        "frozen_finalists": len(finalists),
        "evaluated_finalists": sum(row["status"] == "evaluated" for row in rows[: len(finalists)]),
        "validation_start": VALIDATION_START,
        "validation_end": VALIDATION_END,
        "validation_opened": True,
        "validation_used_for_selection": False,
        "locked_start": LOCKED_START,
        "locked_opened": False,
        "train_freeze_sha256": freeze["freeze_sha256"],
        "code_sha": os.environ.get("GITHUB_SHA", "LOCAL_TEST_ONLY"),
    }
    (output_dir / "validation_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    shutil.copy2(train_results_dir / "train_selection_freeze.json", output_dir / "train_selection_freeze.json")
    (output_dir / "RESULT_STATUS.md").write_text(f"# {summary['result_status']}\n\nValidation opened once; locked remained closed.\n", encoding="utf-8")
    return summary
