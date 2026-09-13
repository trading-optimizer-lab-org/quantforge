"""Aurora workload for autonomous train-only SPY discovery batches."""

from __future__ import annotations

import json
import os
import shutil
import time
import csv
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa

from aurora.infra.github_performance.contracts import (
    PreparedInputs,
    RunSpec,
    SmokeResult,
    canonical_sha256,
)
from aurora.infra.github_performance.metric_verifier import MetricInputRecord
from aurora.infra.github_performance.workloads.common import primary_metric_record
from aurora.infra.sp500_long_short_daily.contracts import CampaignPackage
from aurora.infra.sp500_long_short_daily.contracts import canonical_json_hash
from aurora.infra.sp500_long_short_daily.data import (
    PreparedMarketData,
    load_market_snapshot,
    prepare_market_snapshot,
)
from aurora.infra.sp500_long_short_daily.ledger import apply_positions
from aurora.infra.sp500_long_short_daily.signals import (
    CandidateRejected,
    benchmark_decisions,
    candidate_decisions,
)
from aurora.infra.sp500_long_short_daily.workload import (
    METRIC_FIELDS,
    Sp500LongShortTrainWorkload,
    _result_schema,
    _extended_metrics,
    _market_regime_states,
)

from .contracts import LOCKED_START, TRAIN_END
from .dedupe import build_dedupe_map
from .feature_store import FeatureStore
from .historical_evidence import (
    install_historical_evidence,
    install_prior_autonomous_evidence,
    prepared_evidence_files,
)
from .registry import (
    base_package,
    generate_candidates,
    read_batch_registry,
    get_previous_trial_count,
    repo_root,
    write_batch_registry,
)
from .statistics import evaluate_batch
from .scheduling import assign_by_cost, cost_score


BENCHMARK_IDS = (
    "buy_and_hold_spy_total_return",
    "always_long",
    "always_short",
    "symmetric_sma_200",
    "symmetric_momentum_12m",
)


def _batch_id() -> int:
    value = os.environ.get("AURORA_AUTONOMOUS_BATCH_ID", "0").strip()
    if not value.isdigit():
        raise ValueError("AURORA_AUTONOMOUS_BATCH_ID_MUST_BE_INTEGER")
    return int(value)


def _candidate_count(default: int) -> int:
    value = os.environ.get("AURORA_AUTONOMOUS_CANDIDATE_COUNT", str(default)).strip()
    if not value.isdigit() or int(value) < 1:
        raise ValueError("AURORA_AUTONOMOUS_CANDIDATE_COUNT_MUST_BE_POSITIVE")
    return int(value)


def freeze_selection_reason(finalists: Sequence[Mapping[str, Any]]) -> str:
    return (
        "all frozen train gates passed"
        if finalists
        else "no candidate passed all frozen train gates"
    )


def freeze_rejection_reasons(
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    return {
        str(row.get("strategy_id")): str(row.get("rejection_reason") or "")
        for row in candidates
        if row.get("status") != "evaluated"
        and str(row.get("rejection_reason") or "").strip()
    }


def _light_package(candidates: Sequence[Mapping[str, Any]]) -> CampaignPackage:
    package = base_package()
    return CampaignPackage(
        root=package.root,
        zip_path=package.zip_path,
        spec=package.spec,
        candidates=tuple(candidates),
        features=package.features,
        datasets=package.datasets,
        research=package.research,
    )


def refresh_autonomous_prepared_inputs(root: Path) -> tuple[dict[str, Any], ...]:
    """Refresh batch-specific inputs while preserving immutable market data."""

    root = Path(root).resolve()
    batch_id = _batch_id()
    count = _candidate_count(AutonomousDiscoveryWorkload.default_candidate_count)
    prior_ledger_value = os.environ.get(
        "AURORA_PRIOR_TRIAL_LEDGER_PATH", ""
    ).strip()
    pilot_source = os.environ.get(
        "AURORA_AUTONOMOUS_PILOT_EVIDENCE_ROOT", ""
    ).strip()
    install_prior_autonomous_evidence(
        root,
        pilot_result_root=Path(pilot_source) if pilot_source else None,
        prior_result_root=(
            Path(prior_ledger_value).parent if prior_ledger_value else None
        ),
    )
    candidates = generate_candidates(batch_id, count=count)
    write_batch_registry(
        root,
        batch_id=batch_id,
        candidates=candidates,
        previous_trial_count=get_previous_trial_count(),
    )
    with (root / "dedupe_map.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        dedupe_rows = build_dedupe_map(candidates)
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "strategy_id",
                "canonical_hash",
                "canonical_strategy_id",
                "deduped",
            ],
        )
        writer.writeheader()
        writer.writerows(dedupe_rows)
    priced = [
        {
            **candidate,
            "cost_score": cost_score(
                family_timeout_rate=float(candidate.get("complexity_score", 1))
                / 10.0,
                concept_timeout_rate=float(candidate.get("complexity_score", 1))
                / 20.0,
            ),
        }
        for candidate in candidates
    ]
    with (root / "job_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        manifest_rows = assign_by_cost(priced, max(1, min(360, len(priced))))
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "job_id",
                "strategy_id",
                "canonical_hash",
                "cost_score",
                "estimated_cost_bucket",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)
    return candidates


class AutonomousDiscoveryWorkload(Sp500LongShortTrainWorkload):
    """Evaluate one pre-registered batch without accessing validation or locked data."""

    workload_name = "sp500_autonomous_discovery_train"
    workload_family = "sp500_autonomous_discovery"
    dataset_name = "bounded_spy_autonomous_train_market"
    result_filename = "sp500_autonomous_train_results.parquet"
    phase_name = "train"
    data_start = "1993-01-22"
    data_end = TRAIN_END
    evaluation_start = "1998-01-01"
    minimum_rows = 2500
    minimum_years = 10
    default_candidate_count = 96
    result_schema = _result_schema().append(pa.field("seconds_total", pa.float64()))
    result_schema = result_schema.append(pa.field("seconds_signal", pa.float64()))
    result_schema = result_schema.append(pa.field("seconds_simulation", pa.float64()))

    def __init__(self) -> None:
        super().__init__()
        self._feature_root: Path | None = None
        self._feature_frame: pd.DataFrame | None = None
        self._feature_store_manifest: Mapping[str, object] | None = None

    def describe_contract(self) -> Mapping[str, Any]:
        payload = dict(super().describe_contract())
        payload["name"] = self.workload_name
        payload["scientific_contract"] = {
            **dict(payload["scientific_contract"]),
            "oof_method": "chronological_calendar_year_outer_folds_static_rules",
            "global_trial_count_includes_previous_campaigns": 312,
            "validation_opened": False,
            "locked_opened": False,
        }
        return payload

    def _registry_path(self, root: Path) -> Path:
        return Path(root).resolve() / "candidate_registry.jsonl"

    def _prepare_dataset(self, root: Path) -> tuple[tuple[str, ...], str]:
        historical_source = os.environ.get(
            "AURORA_HISTORICAL_MULTIPLICITY_SOURCE", ""
        ).strip()
        if historical_source:
            install_historical_evidence(Path(historical_source), root)
        candidates = refresh_autonomous_prepared_inputs(root)
        package = _light_package(candidates)
        manifest = prepare_market_snapshot(
            root,
            package,
            start=self.data_start,
            end=self.data_end,
            split="train",
        )
        # These files are immutable provenance inputs, not performance results.
        registry_root = repo_root() / "campaigns" / "sp500_long_short_daily" / "research_input"
        for source_name, target_name in (
            ("research_library.csv", "research_registry_source.csv"),
            ("feature_catalog.csv", "feature_registry_source.csv"),
            ("data_source_inventory.csv", "dataset_registry_source.csv"),
        ):
            target = Path(root) / target_name
            target.write_bytes((registry_root / source_name).read_bytes())
        return (
            "market_data_manifest.json",
            "raw_manifest.jsonl",
            "spy_ledger.parquet",
            "causal_series.parquet",
            "candidate_registry.jsonl",
            "candidate_registry_manifest.json",
            "dedupe_map.csv",
            "job_manifest.csv",
            "research_registry.csv",
            "feature_registry.csv",
            "dataset_registry.csv",
            "research_registry_source.csv",
            "feature_registry_source.csv",
            "dataset_registry_source.csv",
            "trial_ledger.jsonl",
            "autonomous_trial_ledger.parquet",
            *prepared_evidence_files(root),
        ), str(manifest["snapshot_sha256"])

    def _load_dataset(self, root: Path) -> PreparedMarketData:
        root = Path(root).resolve()
        data = load_market_snapshot(root)
        if data.split != "train" or data.ledger.index.max() >= pd.Timestamp(LOCKED_START):
            raise RuntimeError("TRAIN_DATA_BOUNDARY_MISMATCH")
        if self._feature_root != root or self._feature_frame is None:
            snapshot_manifest = json.loads(
                (root / "market_data_manifest.json").read_text(encoding="utf-8")
            )
            store = FeatureStore(
                dataset_sha256=str(snapshot_manifest["snapshot_sha256"]),
                code_sha=os.environ.get("GITHUB_SHA", "LOCAL_TEST_ONLY"),
                start=self.data_start,
                end=self.data_end,
            )
            self._feature_frame = store.get_or_build("SPY", data.ledger)
            self._feature_store_manifest = store.manifest()
            self._feature_root = root
            (root / "feature_store_manifest.json").write_text(
                json.dumps(self._feature_store_manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        return data

    def _candidate_rows(self) -> tuple[dict[str, Any], ...]:
        return tuple(read_batch_registry(self._prepared_root()))

    def _candidate_lookup(self) -> dict[str, Mapping[str, Any]]:
        package = base_package()
        return {**package.candidate_by_id(), **{row["strategy_id"]: row for row in self._candidate_rows()}}

    def _unit_definitions(self) -> Sequence[tuple[str, Mapping[str, Any], float]]:
        definitions: list[tuple[str, Mapping[str, Any], float]] = []
        candidates = self._candidate_rows()
        for candidate in candidates:
            estimate = 0.5 + float(candidate.get("complexity_score", 1)) * 0.25
            definitions.append((str(candidate["strategy_id"]), dict(candidate), estimate))
        for benchmark in BENCHMARK_IDS:
            definitions.append((
                f"BENCHMARK::{benchmark}",
                {"benchmark_id": benchmark, "unit_type": "benchmark"},
                0.2,
            ))
        return tuple(definitions)

    def _evaluate(
        self,
        data: PreparedMarketData,
        unit_key: str,
        parameters: Mapping[str, Any],
        attempt_id: str,
    ) -> tuple[dict[str, Any], tuple[MetricInputRecord, ...]]:
        started = time.perf_counter()
        benchmark = parameters.get("unit_type") == "benchmark"
        row = self._base_row(unit_key, parameters, attempt_id)
        row.update(
            {
                "seconds_total": None,
                "seconds_signal": None,
                "seconds_simulation": None,
            }
        )
        records: tuple[MetricInputRecord, ...] = ()
        try:
            signal_started = time.perf_counter()
            signal = (
                benchmark_decisions(str(parameters["benchmark_id"]), data)
                if benchmark
                else candidate_decisions(
                    parameters,
                    data,
                    candidate_lookup=self._candidate_lookup(),
                    feature_frame=self._feature_frame,
                )
            )
            row["seconds_signal"] = time.perf_counter() - signal_started
            if signal.first_evaluable_date is None:
                raise CandidateRejected("NO_EVALUABLE_SESSION")
            if signal.missing_fraction > 0.02 + 1e-12:
                raise CandidateRejected("DATA_INELIGIBLE:CAUSAL_COVERAGE_LT_98_PERCENT")
            simulation_started = time.perf_counter()
            applied = apply_positions(data.ledger, signal.decisions)
            first_decision = pd.Timestamp(signal.first_evaluable_date)
            later = data.ledger.index[data.ledger.index > first_decision]
            evaluation = applied.loc[later]
            evaluation = evaluation.loc[
                (evaluation.index >= pd.Timestamp(self.evaluation_start))
                & (evaluation.index <= pd.Timestamp(self.data_end))
            ]
            returns = evaluation["strategy_return"].dropna().astype(float)
            positions = evaluation.loc[returns.index, "position"].astype(np.int8)
            if len(returns) < self.minimum_rows or len(set(returns.index.year)) < self.minimum_years:
                raise CandidateRejected("DATA_INELIGIBLE:MINIMUM_TRAIN_OOF_COVERAGE")
            if not positions.isin((-1, 1)).all():
                raise CandidateRejected("TECHNICAL_FAILURE_POSITION")
            record = primary_metric_record(unit_key, "train_oof", returns.to_numpy(dtype=float), periods_per_year=252)
            row.update({f"train_{key}": record.reported.get(key) for key in METRIC_FIELDS})
            regimes = _market_regime_states(data, returns.index)
            row.update(_extended_metrics(returns, positions, regimes))
            row.update({
                "first_evaluable_date": signal.first_evaluable_date,
                "missing_fraction": float(signal.missing_fraction),
                "train_dates": [date.date().isoformat() for date in returns.index],
                "train_returns": [float(value) for value in returns],
                "train_positions": [int(value) for value in positions],
                "out_of_fold": True,
                "oof_method": "calendar_year_outer_fold_static_rule",
            })
            records = (record,)
        except CandidateRejected as exc:
            row["status"] = "rejected"
            row["rejection_reason"] = str(exc)
            records = ()
        else:
            row["seconds_simulation"] = time.perf_counter() - simulation_started
        row["seconds_total"] = time.perf_counter() - started
        row["unit_output_sha256"] = canonical_sha256(
            {
                key: value
                for key, value in row.items()
                if key not in {"source_attempt_id", "seconds_total", "seconds_signal", "seconds_simulation"}
            }
        )
        return row, records

    def _write_reduction(self, rows: Sequence[Mapping[str, Any]], root: Path) -> None:
        expected = {key for key, _, _ in self._unit_definitions()}
        observed = {str(row["unit_key"]) for row in rows}
        if expected != observed:
            raise RuntimeError(f"INCOMPLETE_BATCH_COVERAGE:expected={len(expected)}:observed={len(observed)}")
        root = Path(root)
        if self._feature_frame is None:
            # Merge runs also carry the immutable prepared train snapshot. Build
            # the same keyed manifest there so the final artifact proves which
            # cache identity was available to the workers.
            self._load_dataset(self._prepared_root())
        summary = evaluate_batch(
            rows,
            root,
            batch_id=_batch_id(),
            previous_trial_count=get_previous_trial_count(),
            prepared_root=self._prepared_root(),
        )
        (root / "candidate_registry.jsonl").write_text(
            self._registry_path(self._prepared_root()).read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        summary = dict(summary)
        summary.update(
            {
                "feature_store_file": "feature_store_manifest.json",
                "feature_store_enabled": self._feature_frame is not None,
                "trial_ledger_file": "trial_ledger.jsonl",
                "trial_ledger_rows": sum(
                    1
                    for line in (self._prepared_root() / "trial_ledger.jsonl")
                    .read_text(encoding="utf-8")
                    .splitlines()
                    if line.strip()
                ),
            }
        )
        (root / "autonomous_batch_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        if self._feature_store_manifest is not None:
            (root / "feature_store_manifest.json").write_text(
                json.dumps(self._feature_store_manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        prepared_root = self._prepared_root()
        for filename in (
            "market_data_manifest.json",
            "candidate_registry_manifest.json",
            "dedupe_map.csv",
            "job_manifest.csv",
            "research_registry.csv",
            "feature_registry.csv",
            "dataset_registry.csv",
            "research_registry_source.csv",
            "feature_registry_source.csv",
            "dataset_registry_source.csv",
            "trial_ledger.jsonl",
            "autonomous_trial_ledger.parquet",
        ):
            source = prepared_root / filename
            if source.is_file() and not (root / filename).is_file():
                shutil.copy2(source, root / filename)
        historical_source = prepared_root / "historical_multiplicity"
        if historical_source.is_dir():
            historical_target = root / "historical_multiplicity"
            historical_target.mkdir(parents=True, exist_ok=True)
            for source in historical_source.iterdir():
                if source.is_file() and not (historical_target / source.name).is_file():
                    shutil.copy2(source, historical_target / source.name)
        self._finalize_train_freeze(root, summary)
        (root / "autonomous_batch_identity.json").write_text(
            json.dumps({
                "batch_id": _batch_id(),
                "locked_opened": False,
                "validation_used_for_selection": False,
                "train_end": TRAIN_END,
                "validation_start": "2011-01-01",
                "validation_end": "2020-12-31",
                "summary": summary,
            }, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _finalize_train_freeze(self, root: Path, summary: Mapping[str, Any]) -> None:
        """Bind the selection freeze to the exact batch inputs and hashes."""

        root = Path(root)
        freeze_path = root / "train_selection_freeze.json"
        if not freeze_path.is_file():
            raise RuntimeError("TRAIN_SELECTION_FREEZE_MISSING")
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        candidates = {row["strategy_id"]: row for row in self._candidate_rows()}

        def sha256_file(path: Path) -> str:
            import hashlib

            return hashlib.sha256(path.read_bytes()).hexdigest()

        def read_json(path: Path) -> Mapping[str, Any]:
            return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}

        commit_sha = os.environ.get("GITHUB_SHA", "LOCAL_TEST_ONLY")
        freeze.update(
            {
                "campaign_version": "sp500-autonomous-discovery-v1",
                "repository": os.environ.get(
                    "GITHUB_REPOSITORY", "trading-optimizer-lab-org/aurora"
                ),
                "branch": os.environ.get("GITHUB_REF_NAME", "LOCAL_TEST_ONLY"),
                "commit_sha": commit_sha,
                "environment_hash": os.environ.get("AURORA_ENVIRONMENT_SHA256", ""),
                "code_hashes": {"commit_sha": commit_sha},
                "dataset_hashes": {
                    "market_snapshot": str(
                        read_json(root / "market_data_manifest.json").get(
                            "snapshot_sha256", ""
                        )
                    )
                },
                "feature_registry_hash": (
                    sha256_file(root / "feature_registry.csv")
                    if (root / "feature_registry.csv").is_file()
                    else ""
                ),
                "candidate_registry_hash": sha256_file(root / "candidate_registry.jsonl"),
                "trial_ledger_hash": sha256_file(root / "autonomous_trial_ledger.parquet"),
                "total_prior_trials": int(summary["global_trial_count"])
                - int(summary["total_strategies_loaded"]),
                "total_new_trials": int(summary["total_strategies_loaded"]),
                "total_accumulated_trials": int(summary["global_trial_count"]),
                "ranking_complete": True,
                "Pareto_front": list(freeze.get("finalists", [])),
                "eligible_candidates": list(freeze.get("finalists", [])),
                "exact_rules": {
                    str(item["strategy_id"]): candidates[str(item["strategy_id"])]
                    for item in freeze.get("finalists", [])
                    if str(item["strategy_id"]) in candidates
                },
                "train_oof_metrics": list(freeze.get("finalists", [])),
                "outer_fold_metrics_file": "train_fold_metrics.csv",
                "multiple_testing_results": read_json(root / "multiple_testing.json"),
                "selection_reason": freeze_selection_reason(
                    freeze.get("finalists", [])
                ),
                "rejection_reasons": freeze_rejection_reasons(
                    self._candidate_rows()
                ),
                "authorization_token_required": "OPEN_VALIDATION_2011_2020_ONCE_AUTONOMOUS",
                "locked_state": "closed",
                "timestamp": commit_sha,
            }
        )
        freeze.pop("freeze_sha256", None)
        freeze["freeze_sha256"] = canonical_json_hash(freeze)
        payload = json.dumps(freeze, indent=2, sort_keys=True, allow_nan=False) + "\n"
        freeze_path.write_text(payload, encoding="utf-8")
        # Compatibility alias for older readers; validation uses the explicit
        # immutable selection-freeze filename above.
        (root / "train_freeze_candidate.json").write_text(payload, encoding="utf-8")

    def smoke(self, spec: RunSpec, prepared: PreparedInputs) -> SmokeResult:
        result = super().smoke(spec, prepared)
        if result.passed and result.output_sha256 is not None:
            return result
        return result


class AutonomousDiscoverySmokeWorkload(AutonomousDiscoveryWorkload):
    workload_name = "sp500_autonomous_discovery_smoke"
    workload_family = "sp500_autonomous_discovery_smoke"
    dataset_name = "bounded_spy_autonomous_smoke_market"
    result_filename = "sp500_autonomous_smoke_results.parquet"
    data_start = "2005-10-01"
    data_end = "2009-09-30"
    evaluation_start = "2006-10-01"
    minimum_rows = 200
    minimum_years = 1
    default_candidate_count = 8


class AutonomousDiscoveryPilotWorkload(AutonomousDiscoveryWorkload):
    workload_name = "sp500_autonomous_discovery_pilot"
    workload_family = "sp500_autonomous_discovery_pilot"
    dataset_name = "bounded_spy_autonomous_pilot_market"
    result_filename = "sp500_autonomous_pilot_results.parquet"
    default_candidate_count = 24


SMOKE_WORKLOAD = AutonomousDiscoverySmokeWorkload()
PILOT_WORKLOAD = AutonomousDiscoveryPilotWorkload()
TRAIN_WORKLOAD = AutonomousDiscoveryWorkload()
