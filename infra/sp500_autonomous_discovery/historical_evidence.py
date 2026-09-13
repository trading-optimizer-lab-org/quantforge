"""Verified cumulative V1/V2 and autonomous train evidence.

The autonomous campaign must not treat earlier trials as a count-only penalty.
This module installs the immutable historical artefacts into the prepared input
bundle so every later batch can reuse the exact ledgers and return streams.
"""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd


HISTORICAL_DIR = "historical_multiplicity"
EXPECTED_V1_RESULTS_SHA256 = (
    "164ce2d50909c5224e5260fa185516e9ecee368d948201852f35a72fa0780775"
)
EXPECTED_V2_HASHES = {
    "candidate_and_benchmark_metrics.csv": (
        "0d96a9d8c673f62619eba33aa6d761b09bb29e432fb8b825e081b5a7326c34d2"
    ),
    "combined_multiple_testing_results.json": (
        "127f4f93da3b9908abe6baee7a69548d4400c6a57035628a0863a22853abaf88"
    ),
    "cumulative_trial_ledger.csv": (
        "eeabc900ca6c233a6e7f17d76040606b820fdb826be2a4df012aef3ff1dacf71"
    ),
    "v1_ingestion_audit.json": (
        "7e6012b8b52acbad83dc2aa28b494b5d9b1f85e7f091bf397100856c0044b345"
    ),
    "v2_train_daily_returns.parquet": (
        "cc26f7e1d0053647513c7fa4a2a552ad8c57c1b67a52a70ec8b61ec34c52984d"
    ),
}
EXPECTED_HISTORICAL_TRIALS = 312
EXPECTED_V1_CANDIDATES = 168
EXPECTED_V2_CANDIDATES = 144
TRAIN_END = pd.Timestamp("2010-12-31")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _find_one(root: Path, name: str) -> Path:
    matches = tuple(Path(root).rglob(name))
    if len(matches) != 1:
        raise RuntimeError(f"HISTORICAL_EVIDENCE_FILE_COUNT:{name}:{len(matches)}")
    return matches[0]


def _candidate_lookup(frame: pd.DataFrame) -> dict[str, Mapping[str, Any]]:
    candidates = frame.loc[frame["unit_type"].eq("candidate")]
    lookup: dict[str, Mapping[str, Any]] = {}
    for row in candidates.to_dict("records"):
        lookup[str(row["strategy_id"])] = row
        lookup[str(row["unit_key"])] = row
    return lookup


def build_historical_trial_ledger(
    cumulative: pd.DataFrame,
    v1_metrics: pd.DataFrame,
    v2_metrics: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Convert the canonical 312-row V1+V2 ledger to autonomous schema."""

    if len(cumulative) != EXPECTED_HISTORICAL_TRIALS:
        raise RuntimeError("COMBINED_MULTIPLICITY_INCOMPLETE:HISTORICAL_LEDGER_COUNT")
    if cumulative["campaign"].value_counts().to_dict() != {
        "V1": EXPECTED_V1_CANDIDATES,
        "V2": EXPECTED_V2_CANDIDATES,
    }:
        raise RuntimeError("COMBINED_MULTIPLICITY_INCOMPLETE:HISTORICAL_CAMPAIGN_COUNTS")
    lookups = {"V1": _candidate_lookup(v1_metrics), "V2": _candidate_lookup(v2_metrics)}
    rows: list[dict[str, Any]] = []
    for index, source in enumerate(cumulative.to_dict("records"), start=1):
        campaign = str(source["campaign"])
        source_id = str(source["strategy_id"])
        metric = lookups[campaign].get(source_id)
        if metric is None:
            raise RuntimeError(f"HISTORICAL_STRATEGY_NOT_FOUND:{campaign}:{source_id}")
        status = str(source["status"])
        if status != str(metric["status"]):
            raise RuntimeError(f"HISTORICAL_STATUS_MISMATCH:{campaign}:{source_id}")
        pvalue = float(source["fdr_pvalue"])
        rows.append(
            {
                "batch_id": campaign,
                "campaign": campaign,
                "canonical_hash": str(metric["canonical_hash"]),
                "fdr_pvalue": pvalue,
                "global_trial_index": index,
                "pre_registered_before_performance": True,
                "source_strategy_id": source_id,
                "status": status,
                "strategy_id": f"{campaign}::{source_id}",
            }
        )
    if len({row["canonical_hash"] for row in rows}) != len(rows):
        raise RuntimeError("HISTORICAL_CANONICAL_HASH_COLLISION")
    return rows


def install_historical_evidence(source_root: Path, prepared_root: Path) -> tuple[str, ...]:
    """Verify and copy the canonical V1+V2 evidence into prepared inputs."""

    source_root = Path(source_root)
    target = Path(prepared_root) / HISTORICAL_DIR
    target.mkdir(parents=True, exist_ok=True)
    source_files = {name: _find_one(source_root, name) for name in EXPECTED_V2_HASHES}
    for name, expected in EXPECTED_V2_HASHES.items():
        if _sha256(source_files[name]) != expected:
            raise RuntimeError(f"HISTORICAL_EVIDENCE_HASH_MISMATCH:{name}")
    v1_zip = _find_one(source_root, "prior_v1_results.zip")
    if _sha256(v1_zip) != EXPECTED_V1_RESULTS_SHA256:
        raise RuntimeError("COMBINED_MULTIPLICITY_INCOMPLETE:V1_RESULTS_HASH")

    with zipfile.ZipFile(v1_zip) as archive:
        v1_metrics_bytes = archive.read("candidate_and_benchmark_metrics.csv")
        v1_daily_bytes = archive.read("train_daily_returns.parquet")
    v1_metrics = pd.read_csv(io.BytesIO(v1_metrics_bytes))
    v2_metrics = pd.read_csv(source_files["candidate_and_benchmark_metrics.csv"])
    cumulative = pd.read_csv(source_files["cumulative_trial_ledger.csv"])
    ledger = build_historical_trial_ledger(cumulative, v1_metrics, v2_metrics)

    (target / "v1_candidate_metrics.csv").write_bytes(v1_metrics_bytes)
    (target / "v1_train_daily_returns.parquet").write_bytes(v1_daily_bytes)
    shutil.copy2(
        source_files["candidate_and_benchmark_metrics.csv"],
        target / "v2_candidate_metrics.csv",
    )
    shutil.copy2(
        source_files["v2_train_daily_returns.parquet"],
        target / "v2_train_daily_returns.parquet",
    )
    for name in (
        "combined_multiple_testing_results.json",
        "cumulative_trial_ledger.csv",
        "v1_ingestion_audit.json",
    ):
        shutil.copy2(source_files[name], target / name)
    ledger_payload = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in ledger
    )
    (target / "historical_trial_ledger.jsonl").write_text(
        ledger_payload, encoding="utf-8"
    )

    for name in ("v1_train_daily_returns.parquet", "v2_train_daily_returns.parquet"):
        frame = pd.read_parquet(target / name, columns=["date"])
        if pd.to_datetime(frame["date"]).max() > TRAIN_END:
            raise RuntimeError(f"HISTORICAL_EVIDENCE_AFTER_TRAIN:{name}")
    manifest = {
        "schema_version": "1",
        "source_run_id": 31007105419,
        "source_artifact": "sp500-ls-v2-train-results",
        "historical_trials": len(ledger),
        "v1_candidates": EXPECTED_V1_CANDIDATES,
        "v2_candidates": EXPECTED_V2_CANDIDATES,
        "train_end": TRAIN_END.date().isoformat(),
        "validation_opened": False,
        "locked_opened": False,
        "source_hashes": {
            **EXPECTED_V2_HASHES,
            "prior_v1_results.zip": EXPECTED_V1_RESULTS_SHA256,
        },
        "ledger_sha256": hashlib.sha256(ledger_payload.encode("utf-8")).hexdigest(),
        "complete": True,
    }
    (target / "historical_evidence_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return tuple(
        str(path.relative_to(prepared_root)).replace("\\", "/")
        for path in sorted(target.iterdir())
        if path.is_file()
    )


def _candidate_returns_from_result(root: Path) -> pd.DataFrame:
    cumulative = tuple(Path(root).rglob("cumulative_autonomous_train_returns.parquet"))
    source = cumulative[0] if len(cumulative) == 1 else _find_one(root, "train_oof_daily_returns.parquet")
    frame = pd.read_parquet(source)
    identifier = "strategy_id" if "strategy_id" in frame else "unit_key"
    unit_identity = "unit_key" if "unit_key" in frame else identifier
    frame = frame.loc[
        ~frame[unit_identity].astype(str).str.startswith("BENCHMARK::")
    ].copy()
    frame["strategy_id"] = frame[identifier].astype(str)
    frame["date"] = pd.to_datetime(frame["date"])
    if frame["date"].max() > TRAIN_END:
        raise RuntimeError("PRIOR_AUTONOMOUS_EVIDENCE_AFTER_TRAIN")
    return frame[["strategy_id", "date", "return"]]


def _status_from_result(root: Path, returns: pd.DataFrame, source_label: str) -> pd.DataFrame:
    cumulative = tuple(Path(root).rglob("cumulative_autonomous_status.csv"))
    if len(cumulative) == 1:
        status = pd.read_csv(cumulative[0], keep_default_na=False)
        return status[["strategy_id", "status", "rejection_reason", "source"]]
    rows = [
        {
            "strategy_id": identifier,
            "status": "evaluated",
            "rejection_reason": "",
            "source": source_label,
        }
        for identifier in sorted(returns["strategy_id"].unique())
    ]
    rejection_paths = tuple(Path(root).rglob("rejections.csv"))
    if len(rejection_paths) == 1:
        try:
            rejected = pd.read_csv(rejection_paths[0], keep_default_na=False)
        except pd.errors.EmptyDataError:
            rejected = pd.DataFrame()
        for row in rejected.to_dict("records"):
            identifier = str(row.get("strategy_id", ""))
            if identifier and not identifier.startswith("BENCHMARK::"):
                rows.append(
                    {
                        "strategy_id": identifier,
                        "status": str(row.get("status") or "rejected"),
                        "rejection_reason": str(row.get("reason") or ""),
                        "source": source_label,
                    }
                )
    return pd.DataFrame(rows).drop_duplicates("strategy_id", keep="last")


def install_prior_autonomous_evidence(
    prepared_root: Path,
    *,
    pilot_result_root: Path | None,
    prior_result_root: Path | None,
) -> tuple[str, ...]:
    """Carry all prior autonomous return streams and statuses into this batch."""

    sources: list[tuple[str, Path]] = []
    if pilot_result_root is not None and Path(pilot_result_root).is_dir():
        sources.append(("pilot_run_31036879593", Path(pilot_result_root)))
    if prior_result_root is not None and Path(prior_result_root).is_dir():
        sources.append(("prior_autonomous_result", Path(prior_result_root)))
    if not sources:
        return ()
    return_frames: list[pd.DataFrame] = []
    status_frames: list[pd.DataFrame] = []
    for label, source in sources:
        returns = _candidate_returns_from_result(source)
        return_frames.append(returns)
        status_frames.append(_status_from_result(source, returns, label))
    combined = pd.concat(return_frames, ignore_index=True)
    combined = combined.drop_duplicates(["strategy_id", "date"], keep="last")
    status = pd.concat(status_frames, ignore_index=True).drop_duplicates(
        "strategy_id", keep="last"
    )
    if set(combined["strategy_id"]) - set(status["strategy_id"]):
        raise RuntimeError("PRIOR_AUTONOMOUS_STATUS_INCOMPLETE")
    target = Path(prepared_root) / HISTORICAL_DIR
    target.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(target / "prior_autonomous_train_returns.parquet", index=False)
    status.to_csv(target / "prior_autonomous_status.csv", index=False)
    manifest = {
        "source_runs": [label for label, _ in sources],
        "candidate_streams": int(combined["strategy_id"].nunique()),
        "status_rows": int(len(status)),
        "evaluated": int(status["status"].eq("evaluated").sum()),
        "rejected": int((~status["status"].eq("evaluated")).sum()),
        "train_end": TRAIN_END.date().isoformat(),
        "locked_opened": False,
    }
    (target / "prior_autonomous_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return tuple(
        str(path.relative_to(prepared_root)).replace("\\", "/")
        for path in sorted(target.iterdir())
        if path.is_file()
    )


def load_prior_candidate_returns(prepared_root: Path) -> pd.DataFrame:
    """Load V1, V2, and prior autonomous evaluated return streams."""

    root = Path(prepared_root) / HISTORICAL_DIR
    frames: list[pd.DataFrame] = []
    for campaign, metrics_name, daily_name in (
        ("V1", "v1_candidate_metrics.csv", "v1_train_daily_returns.parquet"),
        ("V2", "v2_candidate_metrics.csv", "v2_train_daily_returns.parquet"),
    ):
        metrics = pd.read_csv(root / metrics_name)
        evaluated = metrics.loc[
            metrics["unit_type"].eq("candidate") & metrics["status"].eq("evaluated"),
            ["unit_key"],
        ]
        keys = set(evaluated["unit_key"].astype(str))
        daily = pd.read_parquet(root / daily_name)
        daily = daily.loc[daily["unit_key"].astype(str).isin(keys)].copy()
        daily["strategy_id"] = campaign + "::" + daily["unit_key"].astype(str)
        daily["date"] = pd.to_datetime(daily["date"])
        frames.append(daily[["strategy_id", "date", "return"]])
    autonomous = root / "prior_autonomous_train_returns.parquet"
    if autonomous.is_file():
        frame = pd.read_parquet(autonomous)
        frame["date"] = pd.to_datetime(frame["date"])
        frames.append(frame[["strategy_id", "date", "return"]])
    combined = pd.concat(frames, ignore_index=True)
    if combined["date"].max() > TRAIN_END:
        raise RuntimeError("COMBINED_EVIDENCE_AFTER_TRAIN")
    return combined.drop_duplicates(["strategy_id", "date"], keep="last")


def load_prior_autonomous_returns(prepared_root: Path) -> pd.DataFrame:
    path = (
        Path(prepared_root)
        / HISTORICAL_DIR
        / "prior_autonomous_train_returns.parquet"
    )
    if not path.is_file():
        return pd.DataFrame(columns=["strategy_id", "date", "return"])
    frame = pd.read_parquet(path)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame[["strategy_id", "date", "return"]]


def load_historical_trial_ledger(prepared_root: Path) -> list[dict[str, Any]]:
    path = Path(prepared_root) / HISTORICAL_DIR / "historical_trial_ledger.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def load_prior_autonomous_status(prepared_root: Path) -> dict[str, Mapping[str, Any]]:
    path = Path(prepared_root) / HISTORICAL_DIR / "prior_autonomous_status.csv"
    if not path.is_file():
        return {}
    return {
        str(row["strategy_id"]): row
        for row in pd.read_csv(path, keep_default_na=False).to_dict("records")
    }


def prepared_evidence_files(prepared_root: Path) -> tuple[str, ...]:
    root = Path(prepared_root) / HISTORICAL_DIR
    if not root.is_dir():
        return ()
    return tuple(
        str(path.relative_to(prepared_root)).replace("\\", "/")
        for path in sorted(root.iterdir())
        if path.is_file()
    )
