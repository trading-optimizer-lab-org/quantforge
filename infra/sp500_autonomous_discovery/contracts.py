"""Frozen contracts for the autonomous SPY discovery campaign."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from typing import Any, Mapping

TRAIN_START = "1993-01-22"
TRAIN_END = "2010-12-31"
VALIDATION_START = "2011-01-01"
VALIDATION_END = "2020-12-31"
VALIDATION_WARMUP_START = "2010-01-01"
LOCKED_START = "2021-01-01"
MULTIPLICITY_DATE_UNIT = "ns"
VALIDATION_ACK = "OPEN_VALIDATION_2011_2020_ONCE_AUTONOMOUS"
PREVIOUS_TRIAL_COUNT = 312
BOOTSTRAP_REPETITIONS = 5000
BLOCK_LENGTH = 20

FINAL_STATES = frozenset(
    {
        "TRAIN_TARGET_FOUND_VALIDATION_PASSED",
        "TRAIN_TARGET_FOUND_VALIDATION_FAILED",
        "TRAIN_TARGET_FOUND_VALIDATION_NOT_OPENED",
        "TRAIN_SEARCH_CONTINUES",
        "COMBINED_MULTIPLICITY_INCOMPLETE",
        "TECHNICAL_FAILURE",
    }
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(child) for child in value]
    if isinstance(value, (date,)):
        return value.isoformat()
    return value


_NON_EFFECTIVE_FIELDS = frozenset(
    {
        "strategy_id",
        "canonical_hash",
        "job_id",
        "shard_id",
        "slot_in_shard",
        "research_source_ids",
        "notes",
        "priority_score",
        "runnable_status",
        "selection_role",
        "variant_label",
        "family_name",
        "evidence_track",
        "dataset_classifications",
        "feature_hash",
        "known_failure_modes",
        "overfitting_risk",
    }
)


def effective_rule_payload(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Return only fields that can alter the generated position series."""

    return {
        str(key): _jsonable(value)
        for key, value in sorted(candidate.items())
        if str(key) not in _NON_EFFECTIVE_FIELDS
    }


def canonical_rule_hash(candidate: Mapping[str, Any]) -> str:
    payload = json.dumps(
        effective_rule_payload(candidate),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def assert_contract(candidate: Mapping[str, Any]) -> None:
    if candidate.get("instrument") != "SPY":
        raise ValueError("INVALID_INSTRUMENT")
    if list(candidate.get("position_values", ())) != [-1, 1]:
        raise ValueError("INVALID_POSITION_CONTRACT")
    if float(candidate.get("absolute_exposure", 0.0)) != 1.0:
        raise ValueError("INVALID_EXPOSURE")
    for field in (
        "cash_allowed",
        "partial_exposure_allowed",
        "leverage_allowed",
        "volatility_scaling_allowed",
        "pyramiding_allowed",
        "multiple_assets_in_portfolio",
    ):
        if candidate.get(field) is not False:
            raise ValueError(f"INVALID_OPERATING_CONTRACT:{field}")
    for field in (
        "commission_bps",
        "slippage_bps",
        "borrow_cost_bps",
        "financing_bps",
        "switching_cost_bps",
        "market_impact_bps",
    ):
        if float(candidate.get(field, 0.0)) != 0.0:
            raise ValueError(f"NON_ZERO_COST:{field}")
    if candidate.get("locked_boundary") != ">=2021-01-01 unopened":
        raise ValueError("LOCKED_CONTRACT_MISMATCH")
    if candidate.get("canonical_hash") != canonical_rule_hash(candidate):
        raise ValueError("CANONICAL_HASH_MISMATCH")


def boundary_payload() -> dict[str, Any]:
    return {
        "train_start": TRAIN_START,
        "train_end": TRAIN_END,
        "validation_start": VALIDATION_START,
        "validation_end": VALIDATION_END,
        "locked_start": LOCKED_START,
        "locked_opened": False,
        "validation_used_for_selection": False,
    }
