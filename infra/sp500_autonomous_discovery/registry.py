"""Deterministic candidate and provenance registries for autonomous batches."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from aurora.infra.sp500_long_short_daily.contracts import CampaignPackage
from aurora.infra.sp500_long_short_daily.signals import IMPLEMENTED_FAMILIES

from .contracts import (
    LOCKED_START,
    PREVIOUS_TRIAL_COUNT,
    TRAIN_END,
    VALIDATION_END,
    VALIDATION_START,
    assert_contract,
    canonical_rule_hash,
)
from .historical_evidence import (
    HISTORICAL_DIR,
    load_historical_trial_ledger,
    load_prior_autonomous_status,
)


def repo_root() -> Path:
    for candidate in (Path.cwd(), Path(__file__).resolve().parents[2]):
        if (candidate / "campaigns" / "sp500_long_short_daily" / "research_input").is_dir():
            return candidate.resolve()
    raise RuntimeError("AURORA_REPO_ROOT_NOT_FOUND")


def base_package() -> CampaignPackage:
    root = repo_root() / "campaigns" / "sp500_long_short_daily"
    return CampaignPackage.load(
        root / "research_input",
        root / "input_package" / "SP500_LONG_SHORT_DIARIO_RESEARCH_AURORA_FINAL.zip",
    )


def _seed(batch_id: int) -> int:
    digest = hashlib.sha256(f"sp500-autonomous:{batch_id}".encode()).hexdigest()
    return int(digest[:16], 16)


def get_previous_trial_count() -> int:
    value = os.environ.get("AURORA_AUTONOMOUS_PREVIOUS_TRIAL_COUNT", str(PREVIOUS_TRIAL_COUNT))
    if not value.isdigit() or int(value) < PREVIOUS_TRIAL_COUNT:
        raise ValueError("INVALID_PREVIOUS_TRIAL_COUNT")
    return int(value)


def _numeric_mutation(value: Any, rng: random.Random) -> Any:
    if (
        isinstance(value, list)
        and value
        and all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value)
    ):
        scale = rng.choice((0.75, 0.9, 1.0, 1.1, 1.25))
        mutated = [
            max(1, int(round(item * scale)))
            if isinstance(item, int)
            else round(float(item) * scale, 8)
            for item in value
        ]
        return mutated
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return value
    if isinstance(value, int):
        scale = rng.choice((0.75, 0.9, 1.0, 1.1, 1.25))
        return max(1, int(round(value * scale)))
    scale = rng.choice((0.75, 0.9, 1.0, 1.1, 1.25))
    return round(float(value) * scale, 8)


def _mutate(
    template: Mapping[str, Any], batch_id: int, index: int, rng: random.Random
) -> dict[str, Any]:
    candidate = json.loads(json.dumps(template))
    candidate.update(
        {
            "instrument": "SPY",
            "cash_allowed": False,
            "partial_exposure_allowed": False,
            "leverage_allowed": False,
            "volatility_scaling_allowed": False,
            "pyramiding_allowed": False,
            "multiple_assets_in_portfolio": False,
        }
    )
    candidate["strategy_id"] = f"AUTO-B{batch_id:04d}-{index:04d}"
    candidate["variant_label"] = f"autonomous_batch_{batch_id}_{index}"
    candidate["evidence_track"] = "pre_2011_evidence"
    candidate["selection_role"] = "autonomous_pre_registered_candidate"
    parameters = dict(candidate.get("parameters", {}))
    for key in sorted(parameters):
        if rng.random() < 0.85:
            parameters[key] = _numeric_mutation(parameters[key], rng)
    candidate["parameters"] = parameters
    candidate["priority_score"] = max(1, 100 - index)
    candidate["canonical_hash"] = canonical_rule_hash(candidate)
    assert_contract(candidate)
    return candidate


def _targeted_reversal_candidates(
    package: CampaignPackage,
    batch_id: int,
    count: int,
) -> tuple[dict[str, Any], ...]:
    """Build a deterministic train-only grid after the broad search plateaued."""

    usable = [
        row
        for row in package.candidates
        if str(row.get("family")) in IMPLEMENTED_FAMILIES
        and set(row.get("required_datasets", ())).issubset({"DS001", "DS002"})
    ]
    if not usable:
        raise RuntimeError("NO_USABLE_CAUSAL_TEMPLATES")
    base = next(
        (row for row in usable if str(row.get("family")) == "short_horizon_reversal"),
        usable[0],
    )
    cycle = max(0, batch_id - 3)
    threshold_shift = 0.1 * cycle
    definitions: list[tuple[str, dict[str, Any], str, str, str]] = []

    for rsi_window, lower, upper, trend_window in (
        (2, 10, 90, 20),
        (2, 10, 90, 50),
        (2, 10, 90, 100),
        (2, 10, 90, 200),
        (2, 20, 80, 20),
        (2, 20, 80, 50),
        (2, 20, 80, 100),
        (2, 20, 80, 200),
        (3, 20, 80, 20),
        (3, 20, 80, 50),
        (3, 20, 80, 100),
        (3, 20, 80, 200),
    ):
        definitions.append(
            (
                "rsi_trend_blend",
                {
                    "rsi_window": rsi_window,
                    "lower": lower,
                    "upper": upper,
                    "trend_window": trend_window,
                },
                "use RSI reversal at extremes; otherwise use causal price trend",
                "score_t > 0",
                "score_t < 0",
            )
        )
    for window, lower, upper in (
        (2, 5, 95),
        (2, 10, 90),
        (2, 15, 85),
        (2, 20, 80),
        (3, 10, 90),
        (3, 15, 85),
        (3, 20, 80),
        (3, 25, 75),
        (5, 15, 85),
        (5, 20, 80),
        (5, 25, 75),
        (5, 30, 70),
    ):
        definitions.append(
            (
                "rsi_reversal",
                {"window": window, "lower": lower, "upper": upper},
                f"Wilder_RSI_{window} through close t",
                f"RSI_t <= {lower}",
                f"RSI_t >= {upper}",
            )
        )
    for lower_fraction, upper_fraction in (
        (0.05, 0.95),
        (0.10, 0.90),
        (0.15, 0.85),
        (0.20, 0.80),
        (0.25, 0.75),
        (0.30, 0.70),
        (0.35, 0.65),
        (0.40, 0.60),
        (0.10, 0.80),
        (0.20, 0.90),
        (0.15, 0.75),
        (0.25, 0.85),
    ):
        definitions.append(
            (
                "internal_bar_strength_reversal",
                {"lower": lower_fraction, "upper": upper_fraction},
                "IBS_t = (TR_CLOSE_t - LOW_t) / (HIGH_t - LOW_t)",
                f"IBS_t <= {lower_fraction}",
                f"IBS_t >= {upper_fraction}",
            )
        )
    for lookback, threshold in (
        (1, 0.25),
        (1, 0.50),
        (1, 0.75),
        (1, 1.00),
        (2, 0.50),
        (2, 1.00),
        (2, 1.50),
        (2, 2.00),
        (3, 0.75),
        (3, 1.50),
        (5, 1.00),
        (5, 2.00),
    ):
        adjusted = round(threshold + threshold_shift, 4)
        definitions.append(
            (
                "return_threshold_reversal",
                {"lookback": lookback, "threshold_pct": adjusted},
                f"lag_return_t = TR_CLOSE_t / TR_CLOSE[t-{lookback}] - 1",
                f"lag_return_t <= -{adjusted}%",
                f"lag_return_t >= {adjusted}%",
            )
        )
    for streak in range(2, 14):
        definitions.append(
            (
                "streak_reversal",
                {"streak": streak},
                "streak_t = signed count of consecutive close-to-close moves through t",
                f"streak_t <= -{streak}",
                f"streak_t >= {streak}",
            )
        )
    for reversal_window, trend_window, threshold in (
        (1, 20, 0.5),
        (1, 50, 0.5),
        (1, 100, 0.5),
        (1, 200, 0.5),
        (2, 20, 1.0),
        (2, 50, 1.0),
        (2, 100, 1.0),
        (2, 200, 1.0),
        (3, 20, 1.5),
        (3, 50, 1.5),
        (3, 100, 1.5),
        (3, 200, 1.5),
    ):
        adjusted = round(threshold + threshold_shift, 4)
        definitions.append(
            (
                "reversal_trend_blend",
                {
                    "reversal_window": reversal_window,
                    "trend_window": trend_window,
                    "reversal_threshold_pct": adjusted,
                },
                "use short-return reversal after an extreme move; otherwise use causal trend",
                "effective score_t > 0",
                "effective score_t < 0",
            )
        )
    for horizons in (
        [1, 2],
        [1, 3],
        [1, 5],
        [2, 3],
        [2, 5],
        [3, 5],
        [1, 2, 3],
        [1, 2, 5],
        [1, 3, 5],
        [2, 3, 5],
        [1, 2, 3, 5],
        [2, 3, 5, 10],
    ):
        definitions.append(
            (
                "multi_horizon_reversal",
                {"horizons": horizons},
                f"score_t = -mean(return_h through t for h in {horizons})",
                "score_t > 0",
                "score_t < 0",
            )
        )
    for threshold in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 1.0, 1.25, 1.5, 2.0):
        adjusted = round(threshold + threshold_shift, 4)
        definitions.append(
            (
                "intraday_return_reversal",
                {"threshold_pct": adjusted},
                "intraday_return_t = TR_CLOSE_t / TR_OPEN_t - 1",
                f"intraday_return_t <= -{adjusted}%",
                f"intraday_return_t >= {adjusted}%",
            )
        )

    candidates: list[dict[str, Any]] = []
    hashes: set[str] = set()
    for index, (family, parameters, formula, long_rule, short_rule) in enumerate(definitions):
        if len(candidates) >= count:
            break
        candidate = json.loads(json.dumps(base))
        candidate.update(
            {
                "instrument": "SPY",
                "cash_allowed": False,
                "partial_exposure_allowed": False,
                "leverage_allowed": False,
                "volatility_scaling_allowed": False,
                "pyramiding_allowed": False,
                "multiple_assets_in_portfolio": False,
                "strategy_id": f"AUTO-B{batch_id:04d}-{index:04d}",
                "variant_label": f"autonomous_targeted_batch_{batch_id}_{index}",
                "family": family,
                "family_name": family.replace("_", " ").title(),
                "parameters": parameters,
                "required_datasets": ["DS001", "DS002"],
                "feature_formulas": [formula],
                "long_rule": long_rule,
                "short_rule": short_rule,
                "features": [f"AUTO_TARGETED_{family.upper()}"],
                "warmup_rule": "No signal before every causal input required by the rule is defined.",
                "known_failure_modes": "Train-only hypothesis; must pass every frozen robustness gate.",
                "economic_sign_rationale": "Tests whether documented short-horizon price pressure mean-reverts at the next open.",
                "priority_score": max(1, 100 - index),
                "evidence_track": "pre_2011_evidence",
                "selection_role": "autonomous_pre_registered_candidate",
            }
        )
        candidate["canonical_hash"] = canonical_rule_hash(candidate)
        if candidate["canonical_hash"] in hashes:
            continue
        assert_contract(candidate)
        hashes.add(str(candidate["canonical_hash"]))
        candidates.append(candidate)
    if len(candidates) != count:
        raise RuntimeError(f"TARGETED_CANDIDATE_COUNT_MISMATCH:{len(candidates)}:{count}")
    return tuple(candidates)


def _neighborhood_reversal_candidates(
    package: CampaignPackage,
    batch_id: int,
    count: int,
) -> tuple[dict[str, Any], ...]:
    """Search the two strongest batch-3 families without repeating its grid."""

    usable = [
        row
        for row in package.candidates
        if str(row.get("family")) in IMPLEMENTED_FAMILIES
        and set(row.get("required_datasets", ())).issubset({"DS001", "DS002"})
    ]
    if not usable:
        raise RuntimeError("NO_USABLE_CAUSAL_TEMPLATES")
    base = next(
        (row for row in usable if str(row.get("family")) == "short_horizon_reversal"),
        usable[0],
    )
    prior_reversal = {
        (1, 20, 0.5),
        (1, 50, 0.5),
        (1, 100, 0.5),
        (1, 200, 0.5),
        (2, 20, 1.0),
        (2, 50, 1.0),
        (2, 100, 1.0),
        (2, 200, 1.0),
        (3, 20, 1.5),
        (3, 50, 1.5),
        (3, 100, 1.5),
        (3, 200, 1.5),
    }
    prior_rsi = {
        (2, 10, 90, 20),
        (2, 10, 90, 50),
        (2, 10, 90, 100),
        (2, 10, 90, 200),
        (2, 20, 80, 20),
        (2, 20, 80, 50),
        (2, 20, 80, 100),
        (2, 20, 80, 200),
        (3, 20, 80, 20),
        (3, 20, 80, 50),
        (3, 20, 80, 100),
        (3, 20, 80, 200),
    }
    generation = batch_id - 4
    trend_windows = [30, 40, 60, 80, 100, 126, 150, 180, 200, 225]
    reversal_thresholds = [
        round(value + generation * 0.05, 4)
        for value in (0.35, 0.5, 0.65, 0.75, 0.9, 1.0, 1.1, 1.25, 1.4, 1.6)
    ]
    reversal_grid = [
        (reversal_window, trend_window, threshold)
        for trend_window in trend_windows
        for threshold in reversal_thresholds
        for reversal_window in (1, 2, 3, 4, 5)
        if (reversal_window, trend_window, threshold) not in prior_reversal
    ]
    rsi_grid = [
        (window, lower, 100 - lower, trend_window)
        for trend_window in trend_windows
        for lower in (5, 10, 15, 20, 25, 30)
        for window in (2, 3, 4, 5, 7)
        if (window, lower, 100 - lower, trend_window) not in prior_rsi
    ]

    # Interleave the whole parameter space so the first 48 are not clustered
    # around one lookback or one trend horizon.
    def spread(rows: list[tuple[Any, ...]], wanted: int) -> list[tuple[Any, ...]]:
        return [rows[(index * len(rows)) // wanted] for index in range(wanted)]

    definitions: list[tuple[str, dict[str, Any], str, str, str]] = []
    for reversal_window, trend_window, threshold in spread(reversal_grid, count // 2):
        definitions.append(
            (
                "reversal_trend_blend",
                {
                    "reversal_window": reversal_window,
                    "trend_window": trend_window,
                    "reversal_threshold_pct": threshold,
                },
                "use short-return reversal after an extreme move; otherwise use causal trend",
                "effective score_t > 0",
                "effective score_t < 0",
            )
        )
    for window, lower, upper, trend_window in spread(rsi_grid, count - len(definitions)):
        definitions.append(
            (
                "rsi_trend_blend",
                {
                    "rsi_window": window,
                    "lower": lower,
                    "upper": upper,
                    "trend_window": trend_window,
                },
                "use RSI reversal at extremes; otherwise use causal price trend",
                "score_t > 0",
                "score_t < 0",
            )
        )

    candidates: list[dict[str, Any]] = []
    for index, (family, parameters, formula, long_rule, short_rule) in enumerate(definitions):
        candidate = json.loads(json.dumps(base))
        candidate.update(
            {
                "instrument": "SPY",
                "cash_allowed": False,
                "partial_exposure_allowed": False,
                "leverage_allowed": False,
                "volatility_scaling_allowed": False,
                "pyramiding_allowed": False,
                "multiple_assets_in_portfolio": False,
                "strategy_id": f"AUTO-B{batch_id:04d}-{index:04d}",
                "variant_label": f"autonomous_neighborhood_batch_{batch_id}_{index}",
                "family": family,
                "family_name": family.replace("_", " ").title(),
                "parameters": parameters,
                "required_datasets": ["DS001", "DS002"],
                "feature_formulas": [formula],
                "long_rule": long_rule,
                "short_rule": short_rule,
                "features": [f"AUTO_NEIGHBORHOOD_{family.upper()}"],
                "warmup_rule": "No signal before every causal input required by the rule is defined.",
                "known_failure_modes": "Train-only neighborhood hypothesis; must pass every frozen robustness gate.",
                "economic_sign_rationale": "Refines the strongest pre-2011 reversal and trend interactions without validation access.",
                "priority_score": max(1, 100 - index),
                "evidence_track": "pre_2011_evidence",
                "selection_role": "autonomous_pre_registered_candidate",
            }
        )
        candidate["canonical_hash"] = canonical_rule_hash(candidate)
        assert_contract(candidate)
        candidates.append(candidate)
    if len(candidates) != count or len({row["canonical_hash"] for row in candidates}) != count:
        raise RuntimeError("NEIGHBORHOOD_CANDIDATE_COUNT_OR_HASH_MISMATCH")
    return tuple(candidates)


def _combined_reversal_candidates(
    package: CampaignPackage,
    batch_id: int,
    count: int,
) -> tuple[dict[str, Any], ...]:
    """Combine the two strongest train rules and refine the stable RSI region."""

    usable = [
        row
        for row in package.candidates
        if str(row.get("family")) in IMPLEMENTED_FAMILIES
        and set(row.get("required_datasets", ())).issubset({"DS001", "DS002"})
    ]
    if not usable:
        raise RuntimeError("NO_USABLE_CAUSAL_TEMPLATES")
    base = next(
        (row for row in usable if str(row.get("family")) == "short_horizon_reversal"),
        usable[0],
    )
    generation = batch_id - 5
    definitions: list[tuple[str, dict[str, Any], str, str, str]] = []
    vote_grid = [
        {
            "rsi_window": rsi_window,
            "lower": lower,
            "upper": 100 - lower,
            "rsi_trend_window": rsi_trend,
            "reversal_window": reversal_window,
            "reversal_threshold_pct": round(threshold + generation * 0.05, 4),
            "reversal_trend_window": reversal_trend,
            "rsi_weight": rsi_weight,
            "reversal_weight": reversal_weight,
        }
        for rsi_window in (4, 5, 6, 7)
        for lower in (20, 22, 25, 28, 30)
        for rsi_trend in (180, 200, 225, 252)
        for reversal_window in (3, 4, 5, 6)
        for threshold in (0.8, 1.0, 1.1, 1.25, 1.4)
        for reversal_trend in (40, 50, 60, 75, 100)
        for rsi_weight, reversal_weight in ((1, 1), (2, 1), (1, 2))
    ]
    fine_rsi_grid = [
        {
            "rsi_window": window,
            "lower": lower,
            "upper": 100 - lower,
            "trend_window": trend_window,
        }
        for trend_window in (205, 210, 215, 220, 230, 235, 240, 245, 250, 252, 260, 270)
        for lower in (18, 20, 22, 24, 25, 26, 28, 30, 32)
        for window in (3, 4, 5, 6, 7, 8)
    ]

    def spread(rows: list[dict[str, Any]], wanted: int) -> list[dict[str, Any]]:
        return [rows[(index * len(rows)) // wanted] for index in range(wanted)]

    vote_count = count if generation >= 1 else count // 2
    for parameters in spread(vote_grid, vote_count):
        definitions.append(
            (
                "dual_reversal_trend_vote",
                parameters,
                "weighted vote of causal RSI-trend and return-reversal-trend components",
                "weighted score_t > 0",
                "weighted score_t < 0",
            )
        )
    for parameters in spread(fine_rsi_grid, count - len(definitions)):
        definitions.append(
            (
                "rsi_trend_blend",
                parameters,
                "use RSI reversal at extremes; otherwise use causal price trend",
                "score_t > 0",
                "score_t < 0",
            )
        )

    candidates: list[dict[str, Any]] = []
    for index, (family, parameters, formula, long_rule, short_rule) in enumerate(definitions):
        candidate = json.loads(json.dumps(base))
        candidate.update(
            {
                "instrument": "SPY",
                "cash_allowed": False,
                "partial_exposure_allowed": False,
                "leverage_allowed": False,
                "volatility_scaling_allowed": False,
                "pyramiding_allowed": False,
                "multiple_assets_in_portfolio": False,
                "strategy_id": f"AUTO-B{batch_id:04d}-{index:04d}",
                "variant_label": f"autonomous_combined_batch_{batch_id}_{index}",
                "family": family,
                "family_name": family.replace("_", " ").title(),
                "parameters": parameters,
                "required_datasets": ["DS001", "DS002"],
                "feature_formulas": [formula],
                "long_rule": long_rule,
                "short_rule": short_rule,
                "features": [f"AUTO_COMBINED_{family.upper()}"],
                "warmup_rule": "No signal before every causal input required by the rule is defined.",
                "known_failure_modes": "Train-only combined hypothesis; must pass every frozen robustness gate.",
                "economic_sign_rationale": "Combines independently pre-registered reversal and trend interactions without validation access.",
                "priority_score": max(1, 100 - index),
                "evidence_track": "pre_2011_evidence",
                "selection_role": "autonomous_pre_registered_candidate",
            }
        )
        candidate["canonical_hash"] = canonical_rule_hash(candidate)
        assert_contract(candidate)
        candidates.append(candidate)
    if len(candidates) != count or len({row["canonical_hash"] for row in candidates}) != count:
        raise RuntimeError("COMBINED_CANDIDATE_COUNT_OR_HASH_MISMATCH")
    return tuple(candidates)


def _trend_guarded_reversal_candidates(
    package: CampaignPackage,
    batch_id: int,
    count: int,
) -> tuple[dict[str, Any], ...]:
    """Add a causal trend vote to the strongest train-only reversal blend."""

    usable = [
        row
        for row in package.candidates
        if str(row.get("family")) in IMPLEMENTED_FAMILIES
        and set(row.get("required_datasets", ())).issubset({"DS001", "DS002"})
    ]
    if not usable:
        raise RuntimeError("NO_USABLE_CAUSAL_TEMPLATES")
    base = next(
        (row for row in usable if str(row.get("family")) == "short_horizon_reversal"),
        usable[0],
    )
    generation = batch_id - 7
    grid = [
        {
            "rsi_window": rsi_window,
            "lower": lower,
            "upper": 100 - lower,
            "rsi_trend_window": rsi_trend_window,
            "reversal_window": reversal_window,
            "reversal_threshold_pct": round(threshold + generation * 0.025, 4),
            "reversal_trend_window": reversal_trend_window,
            "rsi_weight": rsi_weight,
            "reversal_weight": reversal_weight,
            "guard_trend_window": guard_trend_window,
            "guard_weight": guard_weight,
        }
        for rsi_window in (3, 4, 5, 6)
        for lower in (24, 26, 28, 30)
        for rsi_trend_window in (160, 180, 200, 225)
        for reversal_window in (3, 4, 5, 6)
        for threshold in (0.7, 0.85, 1.0, 1.15)
        for reversal_trend_window in (30, 40, 50, 60)
        for rsi_weight, reversal_weight in ((1, 1), (2, 1), (1, 2))
        for guard_trend_window in (50, 75, 100, 150, 200)
        for guard_weight in (0.5, 1.0, 1.5, 2.0)
    ]
    parameters_list = [grid[(index * len(grid)) // count] for index in range(count)]
    candidates: list[dict[str, Any]] = []
    for index, parameters in enumerate(parameters_list):
        candidate = json.loads(json.dumps(base))
        candidate.update(
            {
                "instrument": "SPY",
                "cash_allowed": False,
                "partial_exposure_allowed": False,
                "leverage_allowed": False,
                "volatility_scaling_allowed": False,
                "pyramiding_allowed": False,
                "multiple_assets_in_portfolio": False,
                "strategy_id": f"AUTO-B{batch_id:04d}-{index:04d}",
                "variant_label": f"autonomous_trend_guarded_batch_{batch_id}_{index}",
                "family": "trend_guarded_dual_reversal",
                "family_name": "Trend Guarded Dual Reversal",
                "parameters": parameters,
                "required_datasets": ["DS001", "DS002"],
                "feature_formulas": [
                    "weighted vote of causal RSI reversal, return reversal, and price trend"
                ],
                "long_rule": "weighted causal score_t > 0",
                "short_rule": "weighted causal score_t < 0",
                "features": ["AUTO_TREND_GUARDED_DUAL_REVERSAL"],
                "warmup_rule": "No signal before every causal input required by the rule is defined.",
                "known_failure_modes": "Train-only recovery guard; must pass every frozen robustness gate.",
                "economic_sign_rationale": "A causal trend vote restrains reversal shorts during persistent recoveries.",
                "priority_score": max(1, 100 - index),
                "evidence_track": "pre_2011_evidence",
                "selection_role": "autonomous_pre_registered_candidate",
            }
        )
        candidate["canonical_hash"] = canonical_rule_hash(candidate)
        assert_contract(candidate)
        candidates.append(candidate)
    if len(candidates) != count or len({row["canonical_hash"] for row in candidates}) != count:
        raise RuntimeError("TREND_GUARDED_CANDIDATE_COUNT_OR_HASH_MISMATCH")
    return tuple(candidates)


def _volatility_regime_candidates(
    package: CampaignPackage,
    batch_id: int,
    count: int,
) -> tuple[dict[str, Any], ...]:
    """Use the strongest reversal blend normally and causal trend in high volatility."""

    usable = [
        row
        for row in package.candidates
        if str(row.get("family")) in IMPLEMENTED_FAMILIES
        and set(row.get("required_datasets", ())).issubset({"DS001", "DS002"})
    ]
    if not usable:
        raise RuntimeError("NO_USABLE_CAUSAL_TEMPLATES")
    base = next(
        (row for row in usable if str(row.get("family")) == "short_horizon_reversal"),
        usable[0],
    )
    generation = batch_id - 8
    grid = [
        {
            "rsi_window": rsi_window,
            "lower": lower,
            "upper": 100 - lower,
            "rsi_trend_window": rsi_trend_window,
            "reversal_window": reversal_window,
            "reversal_threshold_pct": round(threshold + generation * 0.025, 4),
            "reversal_trend_window": reversal_trend_window,
            "rsi_weight": rsi_weight,
            "reversal_weight": reversal_weight,
            "volatility_window": volatility_window,
            "high_volatility_pct": high_volatility_pct,
            "regime_trend_window": regime_trend_window,
        }
        for rsi_window in (3, 4, 5, 6)
        for lower in (24, 26, 28, 30)
        for rsi_trend_window in (160, 180, 200, 225)
        for reversal_window in (3, 4, 5, 6)
        for threshold in (0.7, 0.85, 1.0, 1.15)
        for reversal_trend_window in (30, 40, 50, 60)
        for rsi_weight, reversal_weight in ((1, 1), (2, 1), (1, 2))
        for volatility_window in (10, 15, 20, 30)
        for high_volatility_pct in (15.0, 20.0, 25.0, 30.0, 35.0)
        for regime_trend_window in (20, 40, 60, 100, 150, 200)
    ]
    parameters_list = [grid[(index * len(grid)) // count] for index in range(count)]
    candidates: list[dict[str, Any]] = []
    for index, parameters in enumerate(parameters_list):
        candidate = json.loads(json.dumps(base))
        candidate.update(
            {
                "instrument": "SPY",
                "cash_allowed": False,
                "partial_exposure_allowed": False,
                "leverage_allowed": False,
                "volatility_scaling_allowed": False,
                "pyramiding_allowed": False,
                "multiple_assets_in_portfolio": False,
                "strategy_id": f"AUTO-B{batch_id:04d}-{index:04d}",
                "variant_label": f"autonomous_volatility_regime_batch_{batch_id}_{index}",
                "family": "volatility_regime_reversal",
                "family_name": "Volatility Regime Reversal",
                "parameters": parameters,
                "required_datasets": ["DS001", "DS002"],
                "feature_formulas": [
                    "causal reversal blend in normal volatility; causal trend in high volatility"
                ],
                "long_rule": "active regime score_t > 0",
                "short_rule": "active regime score_t < 0",
                "features": ["AUTO_VOLATILITY_REGIME_REVERSAL"],
                "warmup_rule": "No signal before every causal input required by either regime is defined.",
                "known_failure_modes": "Train-only volatility regime hypothesis; must pass every frozen robustness gate.",
                "economic_sign_rationale": "Causal trend replaces fragile reversal exposure during turbulent markets.",
                "priority_score": max(1, 100 - index),
                "evidence_track": "pre_2011_evidence",
                "selection_role": "autonomous_pre_registered_candidate",
            }
        )
        candidate["canonical_hash"] = canonical_rule_hash(candidate)
        assert_contract(candidate)
        candidates.append(candidate)
    if len(candidates) != count or len({row["canonical_hash"] for row in candidates}) != count:
        raise RuntimeError("VOLATILITY_REGIME_CANDIDATE_COUNT_OR_HASH_MISMATCH")
    return tuple(candidates)


def _overnight_tug_reversal_candidates(
    package: CampaignPackage,
    batch_id: int,
    count: int,
) -> tuple[dict[str, Any], ...]:
    """Combine the strongest train reversal blend with the independent V2 tug signal."""

    usable = [
        row
        for row in package.candidates
        if str(row.get("family")) in IMPLEMENTED_FAMILIES
        and set(row.get("required_datasets", ())).issubset({"DS001", "DS002"})
    ]
    if not usable:
        raise RuntimeError("NO_USABLE_CAUSAL_TEMPLATES")
    base = next(
        (row for row in usable if str(row.get("family")) == "short_horizon_reversal"),
        usable[0],
    )
    generation = batch_id - 9
    grid = [
        {
            "rsi_window": rsi_window,
            "lower": lower,
            "upper": 100 - lower,
            "rsi_trend_window": rsi_trend_window,
            "reversal_window": reversal_window,
            "reversal_threshold_pct": round(threshold + generation * 0.025, 4),
            "reversal_trend_window": reversal_trend_window,
            "rsi_weight": rsi_weight,
            "reversal_weight": reversal_weight,
            "tug_lookback": tug_lookback,
            "tug_weight": tug_weight,
        }
        for rsi_window in (3, 4, 5, 6)
        for lower in (24, 26, 28, 30)
        for rsi_trend_window in (160, 180, 200, 225)
        for reversal_window in (3, 4, 5, 6)
        for threshold in (0.7, 0.85, 1.0, 1.15)
        for reversal_trend_window in (30, 40, 50, 60)
        for rsi_weight, reversal_weight in ((1, 1), (2, 1), (1, 2))
        for tug_lookback in (1, 2, 3, 5)
        for tug_weight in (0.5, 1.0, 1.5, 2.0, 3.0)
    ]
    parameters_list = [grid[(index * len(grid)) // count] for index in range(count)]
    candidates: list[dict[str, Any]] = []
    for index, parameters in enumerate(parameters_list):
        candidate = json.loads(json.dumps(base))
        candidate.update(
            {
                "instrument": "SPY",
                "cash_allowed": False,
                "partial_exposure_allowed": False,
                "leverage_allowed": False,
                "volatility_scaling_allowed": False,
                "pyramiding_allowed": False,
                "multiple_assets_in_portfolio": False,
                "strategy_id": f"AUTO-B{batch_id:04d}-{index:04d}",
                "variant_label": f"autonomous_overnight_tug_batch_{batch_id}_{index}",
                "family": "overnight_tug_reversal_vote",
                "family_name": "Overnight Tug Reversal Vote",
                "parameters": parameters,
                "required_datasets": ["DS001", "DS002"],
                "feature_formulas": [
                    "weighted vote of causal RSI reversal, return reversal, and overnight-intraday tug"
                ],
                "long_rule": "weighted causal score_t > 0",
                "short_rule": "weighted causal score_t < 0",
                "features": ["AUTO_OVERNIGHT_TUG_REVERSAL_VOTE"],
                "warmup_rule": "No signal before every causal input required by the rule is defined.",
                "known_failure_modes": "Train-only cross-mechanism hypothesis; must pass every frozen robustness gate.",
                "economic_sign_rationale": "Independent overnight and intraday information augments causal reversal signals.",
                "priority_score": max(1, 100 - index),
                "evidence_track": "pre_2011_evidence",
                "selection_role": "autonomous_pre_registered_candidate",
            }
        )
        candidate["canonical_hash"] = canonical_rule_hash(candidate)
        assert_contract(candidate)
        candidates.append(candidate)
    if len(candidates) != count or len({row["canonical_hash"] for row in candidates}) != count:
        raise RuntimeError("OVERNIGHT_TUG_CANDIDATE_COUNT_OR_HASH_MISMATCH")
    return tuple(candidates)


def _strong_trend_override_candidates(
    package: CampaignPackage,
    batch_id: int,
    count: int,
) -> tuple[dict[str, Any], ...]:
    """Override the strongest reversal blend only during exceptional causal trends."""

    usable = [
        row
        for row in package.candidates
        if str(row.get("family")) in IMPLEMENTED_FAMILIES
        and set(row.get("required_datasets", ())).issubset({"DS001", "DS002"})
    ]
    if not usable:
        raise RuntimeError("NO_USABLE_CAUSAL_TEMPLATES")
    base = next(
        (row for row in usable if str(row.get("family")) == "short_horizon_reversal"),
        usable[0],
    )
    generation = batch_id - 10
    grid = [
        {
            "rsi_window": rsi_window,
            "lower": lower,
            "upper": 100 - lower,
            "rsi_trend_window": rsi_trend_window,
            "reversal_window": reversal_window,
            "reversal_threshold_pct": round(threshold + generation * 0.025, 4),
            "reversal_trend_window": reversal_trend_window,
            "override_window": override_window,
            "override_threshold_pct": override_threshold_pct,
            "override_mode": override_mode,
        }
        for rsi_window in (4, 5)
        for lower in (26, 28, 30)
        for rsi_trend_window in (160, 180, 200)
        for reversal_window in (4, 5, 6)
        for threshold in (0.75, 0.85, 0.95)
        for reversal_trend_window in (30, 40, 50)
        for override_window in (20, 40, 60, 90, 120, 180, 200, 252)
        for override_threshold_pct in (3.0, 5.0, 8.0, 10.0, 15.0, 20.0, 25.0)
        for override_mode in ("long_only", "symmetric")
    ]
    parameters_list = [grid[(index * len(grid)) // count] for index in range(count)]
    candidates: list[dict[str, Any]] = []
    for index, parameters in enumerate(parameters_list):
        candidate = json.loads(json.dumps(base))
        candidate.update(
            {
                "instrument": "SPY",
                "cash_allowed": False,
                "partial_exposure_allowed": False,
                "leverage_allowed": False,
                "volatility_scaling_allowed": False,
                "pyramiding_allowed": False,
                "multiple_assets_in_portfolio": False,
                "strategy_id": f"AUTO-B{batch_id:04d}-{index:04d}",
                "variant_label": f"autonomous_strong_trend_override_batch_{batch_id}_{index}",
                "family": "strong_trend_override_reversal",
                "family_name": "Strong Trend Override Reversal",
                "parameters": parameters,
                "required_datasets": ["DS001", "DS002"],
                "feature_formulas": [
                    "causal dual reversal score with a frozen exceptional-trend override"
                ],
                "long_rule": "dual score_t > 0 or exceptional positive trend",
                "short_rule": "dual score_t < 0, or exceptional negative trend in symmetric mode",
                "features": ["AUTO_STRONG_TREND_OVERRIDE_REVERSAL"],
                "warmup_rule": "No signal before every causal input required by the rule is defined.",
                "known_failure_modes": "Train-only recovery override; must pass every frozen robustness gate.",
                "economic_sign_rationale": "Exceptional persistent moves override fragile counter-trend entries.",
                "priority_score": max(1, 100 - index),
                "evidence_track": "pre_2011_evidence",
                "selection_role": "autonomous_pre_registered_candidate",
            }
        )
        candidate["canonical_hash"] = canonical_rule_hash(candidate)
        assert_contract(candidate)
        candidates.append(candidate)
    if len(candidates) != count or len({row["canonical_hash"] for row in candidates}) != count:
        raise RuntimeError("STRONG_TREND_OVERRIDE_CANDIDATE_COUNT_OR_HASH_MISMATCH")
    return tuple(candidates)


def _stability_refined_dual_reversal_candidates(
    package: CampaignPackage,
    batch_id: int,
    count: int,
) -> tuple[dict[str, Any], ...]:
    """Search nearest untested rules around the strongest autonomous train rule."""

    usable = [
        row
        for row in package.candidates
        if str(row.get("family")) in IMPLEMENTED_FAMILIES
        and set(row.get("required_datasets", ())).issubset({"DS001", "DS002"})
    ]
    if not usable:
        raise RuntimeError("NO_USABLE_CAUSAL_TEMPLATES")
    base = next(
        (row for row in usable if str(row.get("family")) == "short_horizon_reversal"),
        usable[0],
    )
    generation = batch_id - 13
    anchor = {
        "rsi_window": 4,
        "lower": 28,
        "rsi_trend_window": 180,
        "reversal_window": 5,
        "reversal_threshold_pct": 0.85,
        "reversal_trend_window": 40,
        "rsi_weight": 1,
        "reversal_weight": 1,
    }
    known_parameters = {
        json.dumps(row["parameters"], sort_keys=True, separators=(",", ":"))
        for prior_batch in (5, 6)
        for row in _combined_reversal_candidates(package, prior_batch, 96)
        if str(row["family"]) == "dual_reversal_trend_vote"
    }
    grid = [
        {
            "rsi_window": rsi_window,
            "lower": lower,
            "upper": 100 - lower,
            "rsi_trend_window": rsi_trend_window,
            "reversal_window": reversal_window,
            "reversal_threshold_pct": reversal_threshold_pct,
            "reversal_trend_window": reversal_trend_window,
            "rsi_weight": rsi_weight,
            "reversal_weight": reversal_weight,
        }
        for rsi_window in (3, 4, 5, 6)
        for lower in (24, 26, 28, 30, 32)
        for rsi_trend_window in (140, 160, 180, 200, 220)
        for reversal_window in (3, 4, 5, 6, 7)
        for reversal_threshold_pct in (
            0.65,
            0.70,
            0.75,
            0.80,
            0.825,
            0.85,
            0.875,
            0.90,
            0.95,
            1.00,
            1.05,
        )
        for reversal_trend_window in (20, 30, 40, 50, 60, 80)
        for rsi_weight, reversal_weight in ((1, 1), (2, 1), (1, 2))
    ]

    def distance(parameters: Mapping[str, Any]) -> tuple[float, str]:
        score = (
            abs(int(parameters["rsi_window"]) - anchor["rsi_window"]) / 1.0
            + abs(int(parameters["lower"]) - anchor["lower"]) / 2.0
            + abs(int(parameters["rsi_trend_window"]) - anchor["rsi_trend_window"]) / 20.0
            + abs(int(parameters["reversal_window"]) - anchor["reversal_window"]) / 1.0
            + abs(float(parameters["reversal_threshold_pct"]) - anchor["reversal_threshold_pct"])
            / 0.025
            + abs(int(parameters["reversal_trend_window"]) - anchor["reversal_trend_window"]) / 10.0
            + 4.0
            * (
                abs(int(parameters["rsi_weight"]) - anchor["rsi_weight"])
                + abs(int(parameters["reversal_weight"]) - anchor["reversal_weight"])
            )
        )
        return score, json.dumps(parameters, sort_keys=True, separators=(",", ":"))

    untested = [
        parameters
        for parameters in sorted(grid, key=distance)
        if json.dumps(parameters, sort_keys=True, separators=(",", ":")) not in known_parameters
    ]
    start = generation * count
    parameters_list = untested[start : start + count]
    if len(parameters_list) != count:
        raise RuntimeError("STABILITY_REFINEMENT_GRID_EXHAUSTED")
    candidates: list[dict[str, Any]] = []
    for index, parameters in enumerate(parameters_list):
        candidate = json.loads(json.dumps(base))
        candidate.update(
            {
                "instrument": "SPY",
                "cash_allowed": False,
                "partial_exposure_allowed": False,
                "leverage_allowed": False,
                "volatility_scaling_allowed": False,
                "pyramiding_allowed": False,
                "multiple_assets_in_portfolio": False,
                "strategy_id": f"AUTO-B{batch_id:04d}-{index:04d}",
                "variant_label": (f"autonomous_stability_refinement_batch_{batch_id}_{index}"),
                "family": "dual_reversal_trend_vote",
                "family_name": "Stability Refined Dual Reversal Trend Vote",
                "parameters": parameters,
                "required_datasets": ["DS001", "DS002"],
                "feature_formulas": [
                    "nearest untested weighted causal RSI-trend and return-reversal-trend vote"
                ],
                "long_rule": "weighted score_t > 0",
                "short_rule": "weighted score_t < 0",
                "features": ["AUTO_STABILITY_REFINED_DUAL_REVERSAL_TREND_VOTE"],
                "warmup_rule": "No signal before every causal input required by the rule is defined.",
                "known_failure_modes": "Train-only stability refinement; must pass every frozen robustness gate.",
                "economic_sign_rationale": "Refines the strongest all-positive-year train rule without changing its causal mechanism.",
                "priority_score": max(1, 100 - index),
                "evidence_track": "pre_2011_evidence",
                "selection_role": "autonomous_pre_registered_candidate",
            }
        )
        candidate["canonical_hash"] = canonical_rule_hash(candidate)
        assert_contract(candidate)
        candidates.append(candidate)
    if len(candidates) != count or len({row["canonical_hash"] for row in candidates}) != count:
        raise RuntimeError("STABILITY_REFINEMENT_COUNT_OR_HASH_MISMATCH")
    return tuple(candidates)


def _asymmetric_override_candidates(
    package: CampaignPackage,
    batch_id: int,
    count: int,
) -> tuple[dict[str, Any], ...]:
    """Refine the stable reversal rule with independent bull and bear overrides."""

    if count != 96:
        raise ValueError("ASYMMETRIC_OVERRIDE_REQUIRES_96_CANDIDATES")
    usable = [
        row
        for row in package.candidates
        if str(row.get("family")) in IMPLEMENTED_FAMILIES
        and set(row.get("required_datasets", ())).issubset({"DS001", "DS002"})
    ]
    if not usable:
        raise RuntimeError("NO_USABLE_CAUSAL_TEMPLATES")
    base = next(
        (row for row in usable if str(row.get("family")) == "short_horizon_reversal"),
        usable[0],
    )
    generation = batch_id - 16
    core_variants = (
        {
            "rsi_window": 4,
            "lower": 30,
            "upper": 70,
            "rsi_trend_window": 180,
            "reversal_window": 5,
            "reversal_threshold_pct": 0.875,
            "reversal_trend_window": 40,
        },
        {
            "rsi_window": 4,
            "lower": 29,
            "upper": 71,
            "rsi_trend_window": 180,
            "reversal_window": 5,
            "reversal_threshold_pct": 0.875,
            "reversal_trend_window": 40,
        },
        {
            "rsi_window": 4,
            "lower": 31,
            "upper": 69,
            "rsi_trend_window": 180,
            "reversal_window": 5,
            "reversal_threshold_pct": 0.875,
            "reversal_trend_window": 40,
        },
        {
            "rsi_window": 4,
            "lower": 30,
            "upper": 70,
            "rsi_trend_window": 160,
            "reversal_window": 5,
            "reversal_threshold_pct": 0.85,
            "reversal_trend_window": 40,
        },
    )
    if batch_id >= 18:
        local_generation = batch_id - 18
        positive_overrides = (
            (90, 2.0),
            (120, 2.5),
            (120, 3.0),
            (150, 3.0),
        )
        negative_overrides = (
            (90, 2.0),
            (90, 3.0),
            (120, 2.0),
            (120, 3.0),
            (120, 4.0),
            (150, 3.0),
        )
        positive_step = local_generation * 0.125
        negative_step = local_generation * 0.25
    else:
        positive_overrides = ((90, 2.0), (120, 2.0), (120, 3.0), (150, 3.0))
        negative_overrides = (
            (40, 3.0),
            (60, 3.0),
            (60, 5.0),
            (90, 5.0),
            (120, 5.0),
            (120, 8.0),
        )
        positive_step = generation * 0.25
        negative_step = generation * 0.5
    parameters_list = [
        {
            **core,
            "positive_override_window": positive_window,
            "positive_override_threshold_pct": round(positive_threshold + positive_step, 4),
            "negative_override_window": negative_window,
            "negative_override_threshold_pct": round(negative_threshold + negative_step, 4),
        }
        for core in core_variants
        for positive_window, positive_threshold in positive_overrides
        for negative_window, negative_threshold in negative_overrides
    ]
    candidates: list[dict[str, Any]] = []
    for index, parameters in enumerate(parameters_list):
        candidate = json.loads(json.dumps(base))
        candidate.update(
            {
                "instrument": "SPY",
                "cash_allowed": False,
                "partial_exposure_allowed": False,
                "leverage_allowed": False,
                "volatility_scaling_allowed": False,
                "pyramiding_allowed": False,
                "multiple_assets_in_portfolio": False,
                "strategy_id": f"AUTO-B{batch_id:04d}-{index:04d}",
                "variant_label": (f"autonomous_asymmetric_override_batch_{batch_id}_{index}"),
                "family": "asymmetric_trend_override_reversal",
                "family_name": "Asymmetric Trend Override Reversal",
                "parameters": parameters,
                "required_datasets": ["DS001", "DS002"],
                "feature_formulas": [
                    "causal dual reversal score with independent positive and negative trend overrides"
                ],
                "long_rule": "dual score_t > 0 or exceptional positive trend",
                "short_rule": "dual score_t < 0 or exceptional negative trend",
                "features": ["AUTO_ASYMMETRIC_TREND_OVERRIDE_REVERSAL"],
                "warmup_rule": "No signal before every causal input required by the rule is defined.",
                "known_failure_modes": "Train-only asymmetric refinement; must pass every frozen robustness gate.",
                "economic_sign_rationale": "Different causal horizons capture recoveries and persistent bear trends without cash or leverage.",
                "priority_score": max(1, 100 - index),
                "evidence_track": "pre_2011_evidence",
                "selection_role": "autonomous_pre_registered_candidate",
            }
        )
        candidate["canonical_hash"] = canonical_rule_hash(candidate)
        assert_contract(candidate)
        candidates.append(candidate)
    if len(candidates) != count or len({row["canonical_hash"] for row in candidates}) != count:
        raise RuntimeError("ASYMMETRIC_OVERRIDE_COUNT_OR_HASH_MISMATCH")
    return tuple(candidates)


def _drawdown_recovery_override_candidates(
    package: CampaignPackage,
    batch_id: int,
    count: int,
) -> tuple[dict[str, Any], ...]:
    """Test causal recovery overrides around the strongest train rules."""

    if count != 96:
        raise ValueError("DRAWDOWN_RECOVERY_OVERRIDE_REQUIRES_96_CANDIDATES")
    usable = [
        row
        for row in package.candidates
        if str(row.get("family")) in IMPLEMENTED_FAMILIES
        and set(row.get("required_datasets", ())).issubset({"DS001", "DS002"})
    ]
    if not usable:
        raise RuntimeError("NO_USABLE_CAUSAL_TEMPLATES")
    base = next(
        (row for row in usable if str(row.get("family")) == "short_horizon_reversal"),
        usable[0],
    )
    generation = batch_id - 19
    core_variants = (
        {
            "rsi_window": 4,
            "lower": 28,
            "upper": 72,
            "rsi_trend_window": 220,
            "rsi_weight": 1,
            "reversal_window": 5,
            "reversal_threshold_pct": 0.825,
            "reversal_trend_window": 40,
            "reversal_weight": 1,
        },
        {
            "rsi_window": 4,
            "lower": 26,
            "upper": 74,
            "rsi_trend_window": 180,
            "rsi_weight": 1,
            "reversal_window": 5,
            "reversal_threshold_pct": 0.875,
            "reversal_trend_window": 40,
            "reversal_weight": 1,
        },
    )
    parameters_list = [
        {
            **core,
            "drawdown_lookback": 252,
            "drawdown_trigger_pct": drawdown_trigger_pct,
            "recovery_memory_window": recovery_memory_window,
            "recovery_window": recovery_window,
            "recovery_threshold_pct": round(
                recovery_threshold_pct + generation * 0.25,
                4,
            ),
        }
        for core in core_variants
        for drawdown_trigger_pct in (12.5, 17.5, 22.5, 27.5)
        for recovery_memory_window in (63, 126, 189)
        for recovery_window in (20, 63)
        for recovery_threshold_pct in (2.0, 5.0)
    ]
    candidates: list[dict[str, Any]] = []
    for index, parameters in enumerate(parameters_list):
        candidate = json.loads(json.dumps(base))
        candidate.update(
            {
                "instrument": "SPY",
                "cash_allowed": False,
                "partial_exposure_allowed": False,
                "leverage_allowed": False,
                "volatility_scaling_allowed": False,
                "pyramiding_allowed": False,
                "multiple_assets_in_portfolio": False,
                "strategy_id": f"AUTO-B{batch_id:04d}-{index:04d}",
                "variant_label": (f"autonomous_drawdown_recovery_batch_{batch_id}_{index}"),
                "family": "drawdown_recovery_override_reversal",
                "family_name": "Drawdown Recovery Override Reversal",
                "parameters": parameters,
                "required_datasets": ["DS001", "DS002"],
                "feature_formulas": [
                    "causal dual reversal score forced long only after a prior deep drawdown and confirmed recovery"
                ],
                "long_rule": (
                    "dual score_t > 0 or prior deep drawdown followed by causal recovery"
                ),
                "short_rule": "dual score_t < 0 outside a confirmed recovery",
                "features": ["AUTO_DRAWDOWN_RECOVERY_OVERRIDE_REVERSAL"],
                "warmup_rule": (
                    "No signal before every causal input required by the rule is defined."
                ),
                "known_failure_modes": (
                    "Train-only recovery refinement; must pass every frozen robustness gate."
                ),
                "economic_sign_rationale": (
                    "Post-drawdown recoveries can persist after the reversal base would turn short."
                ),
                "priority_score": max(1, 100 - index),
                "evidence_track": "pre_2011_evidence",
                "selection_role": "autonomous_pre_registered_candidate",
            }
        )
        candidate["canonical_hash"] = canonical_rule_hash(candidate)
        assert_contract(candidate)
        candidates.append(candidate)
    if len(candidates) != count or len({row["canonical_hash"] for row in candidates}) != count:
        raise RuntimeError("DRAWDOWN_RECOVERY_COUNT_OR_HASH_MISMATCH")
    return tuple(candidates)


def _quiet_bull_recovery_override_candidates(
    package: CampaignPackage,
    batch_id: int,
    count: int,
) -> tuple[dict[str, Any], ...]:
    """Combine recovery protection with a causal quiet-bull override."""

    if count != 96:
        raise ValueError("QUIET_BULL_RECOVERY_OVERRIDE_REQUIRES_96_CANDIDATES")
    usable = [
        row
        for row in package.candidates
        if str(row.get("family")) in IMPLEMENTED_FAMILIES
        and set(row.get("required_datasets", ())).issubset({"DS001", "DS002"})
    ]
    if not usable:
        raise RuntimeError("NO_USABLE_CAUSAL_TEMPLATES")
    base = next(
        (row for row in usable if str(row.get("family")) == "short_horizon_reversal"),
        usable[0],
    )
    generation = batch_id - 21
    core_variants = (
        {
            "rsi_window": 4,
            "lower": 28,
            "upper": 72,
            "rsi_trend_window": 220,
            "rsi_weight": 1,
            "reversal_window": 5,
            "reversal_threshold_pct": 0.825,
            "reversal_trend_window": 40,
            "reversal_weight": 1,
            "drawdown_lookback": 252,
            "drawdown_trigger_pct": 22.5,
            "recovery_memory_window": 63,
            "recovery_window": 20,
            "recovery_threshold_pct": 2.25,
        },
        {
            "rsi_window": 4,
            "lower": 28,
            "upper": 72,
            "rsi_trend_window": 220,
            "rsi_weight": 1,
            "reversal_window": 5,
            "reversal_threshold_pct": 0.825,
            "reversal_trend_window": 40,
            "reversal_weight": 1,
            "drawdown_lookback": 252,
            "drawdown_trigger_pct": 22.5,
            "recovery_memory_window": 189,
            "recovery_window": 20,
            "recovery_threshold_pct": 2.25,
        },
    )
    parameters_list = [
        {
            **core,
            "bull_ma_window": bull_ma_window,
            "bull_slope_window": bull_slope_window,
            "bull_min_return_pct": round(
                bull_min_return_pct + generation * 0.25,
                4,
            ),
            "bull_max_volatility_pct": bull_max_volatility_pct,
        }
        for core in core_variants
        for bull_ma_window in (100, 150, 200)
        for bull_slope_window in (20, 63)
        for bull_min_return_pct in (0.0, 5.0)
        for bull_max_volatility_pct in (15.0, 20.0, 25.0, 30.0)
    ]
    candidates: list[dict[str, Any]] = []
    for index, parameters in enumerate(parameters_list):
        candidate = json.loads(json.dumps(base))
        candidate.update(
            {
                "instrument": "SPY",
                "cash_allowed": False,
                "partial_exposure_allowed": False,
                "leverage_allowed": False,
                "volatility_scaling_allowed": False,
                "pyramiding_allowed": False,
                "multiple_assets_in_portfolio": False,
                "strategy_id": f"AUTO-B{batch_id:04d}-{index:04d}",
                "variant_label": (f"autonomous_quiet_bull_recovery_batch_{batch_id}_{index}"),
                "family": "quiet_bull_recovery_override_reversal",
                "family_name": "Quiet Bull Recovery Override Reversal",
                "parameters": parameters,
                "required_datasets": ["DS001", "DS002"],
                "feature_formulas": [
                    "causal dual reversal score forced long during confirmed recovery or a quiet rising market"
                ],
                "long_rule": ("dual score_t > 0, confirmed recovery, or quiet rising market"),
                "short_rule": "dual score_t < 0 outside both long overrides",
                "features": ["AUTO_QUIET_BULL_RECOVERY_OVERRIDE_REVERSAL"],
                "warmup_rule": (
                    "No signal before every causal input required by the rule is defined."
                ),
                "known_failure_modes": (
                    "Train-only bull-regime refinement; must pass every frozen robustness gate."
                ),
                "economic_sign_rationale": (
                    "Quiet rising markets reward persistent long exposure while the reversal base protects adverse regimes."
                ),
                "priority_score": max(1, 100 - index),
                "evidence_track": "pre_2011_evidence",
                "selection_role": "autonomous_pre_registered_candidate",
            }
        )
        candidate["canonical_hash"] = canonical_rule_hash(candidate)
        assert_contract(candidate)
        candidates.append(candidate)
    if len(candidates) != count or len({row["canonical_hash"] for row in candidates}) != count:
        raise RuntimeError("QUIET_BULL_RECOVERY_COUNT_OR_HASH_MISMATCH")
    return tuple(candidates)


def _recovery_trend_breakout_majority_candidates(
    package: CampaignPackage,
    batch_id: int,
    count: int,
) -> tuple[dict[str, Any], ...]:
    """Diversify the strongest recovery rule with trend and breakout votes."""

    if count != 96:
        raise ValueError("RECOVERY_TREND_BREAKOUT_REQUIRES_96_CANDIDATES")
    usable = [
        row
        for row in package.candidates
        if str(row.get("family")) in IMPLEMENTED_FAMILIES
        and set(row.get("required_datasets", ())).issubset({"DS001", "DS002"})
    ]
    if not usable:
        raise RuntimeError("NO_USABLE_CAUSAL_TEMPLATES")
    base = next(
        (row for row in usable if str(row.get("family")) == "short_horizon_reversal"),
        usable[0],
    )
    generation = batch_id - 22
    core_variants = (
        {
            "rsi_window": 4,
            "lower": 28,
            "upper": 72,
            "rsi_trend_window": 220,
            "rsi_weight": 1,
            "reversal_window": 5,
            "reversal_threshold_pct": 0.825,
            "reversal_trend_window": 40,
            "reversal_weight": 1,
            "drawdown_lookback": 252,
            "drawdown_trigger_pct": 22.5,
            "recovery_window": 20,
            "recovery_threshold_pct": round(2.25 + generation * 0.25, 4),
        },
        {
            "rsi_window": 4,
            "lower": 26,
            "upper": 74,
            "rsi_trend_window": 180,
            "rsi_weight": 1,
            "reversal_window": 5,
            "reversal_threshold_pct": 0.875,
            "reversal_trend_window": 40,
            "reversal_weight": 1,
            "drawdown_lookback": 252,
            "drawdown_trigger_pct": 22.5,
            "recovery_window": 20,
            "recovery_threshold_pct": round(2.25 + generation * 0.25, 4),
        },
    )
    trend_horizon_sets = (
        [20, 63, 126],
        [20, 63, 126, 252],
        [63, 126, 252],
        [20, 126, 252],
        [10, 20, 63, 126],
        [20, 63, 126, 189, 252],
    )
    parameters_list = [
        {
            **core,
            "recovery_memory_window": recovery_memory_window,
            "trend_horizons": trend_horizons,
            "breakout_window": breakout_window,
        }
        for core in core_variants
        for recovery_memory_window in (63, 189)
        for trend_horizons in trend_horizon_sets
        for breakout_window in (20, 50, 100, 252)
    ]
    candidates: list[dict[str, Any]] = []
    for index, parameters in enumerate(parameters_list):
        candidate = json.loads(json.dumps(base))
        candidate.update(
            {
                "instrument": "SPY",
                "cash_allowed": False,
                "partial_exposure_allowed": False,
                "leverage_allowed": False,
                "volatility_scaling_allowed": False,
                "pyramiding_allowed": False,
                "multiple_assets_in_portfolio": False,
                "strategy_id": f"AUTO-B{batch_id:04d}-{index:04d}",
                "variant_label": (f"autonomous_recovery_trend_breakout_batch_{batch_id}_{index}"),
                "family": "recovery_trend_breakout_majority",
                "family_name": "Recovery Trend Breakout Majority",
                "parameters": parameters,
                "required_datasets": ["DS001", "DS002"],
                "feature_formulas": [
                    "majority vote of causal recovery reversal, multi-horizon trend, and prior-range breakout"
                ],
                "long_rule": "at least two of three causal component states are long",
                "short_rule": "at least two of three causal component states are short",
                "features": ["AUTO_RECOVERY_TREND_BREAKOUT_MAJORITY"],
                "warmup_rule": (
                    "No signal before every causal input required by all three votes is defined."
                ),
                "known_failure_modes": (
                    "Train-only diversified ensemble; must pass every frozen robustness gate."
                ),
                "economic_sign_rationale": (
                    "Independent reversal, trend, and breakout evidence reduces dependence on one market mechanism."
                ),
                "priority_score": max(1, 100 - index),
                "evidence_track": "pre_2011_evidence",
                "selection_role": "autonomous_pre_registered_candidate",
            }
        )
        candidate["canonical_hash"] = canonical_rule_hash(candidate)
        assert_contract(candidate)
        candidates.append(candidate)
    if len(candidates) != count or len({row["canonical_hash"] for row in candidates}) != count:
        raise RuntimeError("RECOVERY_TREND_BREAKOUT_COUNT_OR_HASH_MISMATCH")
    return tuple(candidates)


def _high_vol_crash_recovery_candidates(
    package: CampaignPackage,
    batch_id: int,
    count: int,
) -> tuple[dict[str, Any], ...]:
    """Protect the strongest recovery rule during confirmed high-vol crashes."""

    if count != 96:
        raise ValueError("HIGH_VOL_CRASH_RECOVERY_REQUIRES_96_CANDIDATES")
    usable = [
        row
        for row in package.candidates
        if str(row.get("family")) in IMPLEMENTED_FAMILIES
        and set(row.get("required_datasets", ())).issubset({"DS001", "DS002"})
    ]
    if not usable:
        raise RuntimeError("NO_USABLE_CAUSAL_TEMPLATES")
    base = next(
        (row for row in usable if str(row.get("family")) == "short_horizon_reversal"),
        usable[0],
    )
    generation = batch_id - 23
    core_variants = (
        {
            "rsi_window": 4,
            "lower": 28,
            "upper": 72,
            "rsi_trend_window": 220,
            "rsi_weight": 1,
            "reversal_window": 5,
            "reversal_threshold_pct": 0.825,
            "reversal_trend_window": 40,
            "reversal_weight": 1,
            "drawdown_lookback": 252,
            "drawdown_trigger_pct": 22.5,
            "recovery_memory_window": 189,
            "recovery_window": 20,
            "recovery_threshold_pct": 2.25,
        },
        {
            "rsi_window": 4,
            "lower": 28,
            "upper": 72,
            "rsi_trend_window": 220,
            "rsi_weight": 1,
            "reversal_window": 5,
            "reversal_threshold_pct": 0.825,
            "reversal_trend_window": 40,
            "reversal_weight": 1,
            "drawdown_lookback": 252,
            "drawdown_trigger_pct": 22.5,
            "recovery_memory_window": 63,
            "recovery_window": 20,
            "recovery_threshold_pct": 2.25,
        },
    )
    parameters_list = [
        {
            **core,
            "volatility_window": volatility_window,
            "high_volatility_pct": high_volatility_pct,
            "crash_window": crash_window,
            "crash_threshold_pct": round(
                crash_threshold_pct + generation,
                4,
            ),
        }
        for core in core_variants
        for volatility_window in (10, 20, 40)
        for high_volatility_pct in (20.0, 25.0, 30.0, 35.0)
        for crash_window in (10, 20)
        for crash_threshold_pct in (5.0, 10.0)
    ]
    candidates: list[dict[str, Any]] = []
    for index, parameters in enumerate(parameters_list):
        candidate = json.loads(json.dumps(base))
        candidate.update(
            {
                "instrument": "SPY",
                "cash_allowed": False,
                "partial_exposure_allowed": False,
                "leverage_allowed": False,
                "volatility_scaling_allowed": False,
                "pyramiding_allowed": False,
                "multiple_assets_in_portfolio": False,
                "strategy_id": f"AUTO-B{batch_id:04d}-{index:04d}",
                "variant_label": (f"autonomous_high_vol_crash_recovery_batch_{batch_id}_{index}"),
                "family": "high_vol_crash_recovery_reversal",
                "family_name": "High Vol Crash Recovery Reversal",
                "parameters": parameters,
                "required_datasets": ["DS001", "DS002"],
                "feature_formulas": [
                    "causal recovery reversal forced short in a confirmed high-volatility decline and long after confirmed recovery"
                ],
                "long_rule": "dual score_t > 0 or confirmed post-drawdown recovery",
                "short_rule": (
                    "dual score_t < 0 or high realized volatility with a sufficiently negative trailing return"
                ),
                "features": ["AUTO_HIGH_VOL_CRASH_RECOVERY_REVERSAL"],
                "warmup_rule": (
                    "No signal before every causal input required by the rule is defined."
                ),
                "known_failure_modes": (
                    "Train-only crash-state refinement; must pass every frozen robustness gate."
                ),
                "economic_sign_rationale": (
                    "Persistent high-volatility declines favor short continuation until a causal recovery is confirmed."
                ),
                "priority_score": max(1, 100 - index),
                "evidence_track": "pre_2011_evidence",
                "selection_role": "autonomous_pre_registered_candidate",
            }
        )
        candidate["canonical_hash"] = canonical_rule_hash(candidate)
        assert_contract(candidate)
        candidates.append(candidate)
    if len(candidates) != count or len({row["canonical_hash"] for row in candidates}) != count:
        raise RuntimeError("HIGH_VOL_CRASH_RECOVERY_COUNT_OR_HASH_MISMATCH")
    return tuple(candidates)


def _adaptive_recovery_edge_candidates(
    package: CampaignPackage,
    batch_id: int,
    count: int,
) -> tuple[dict[str, Any], ...]:
    """Adapt the strongest recovery rule to its own recent causal edge."""

    if count != 96:
        raise ValueError("ADAPTIVE_RECOVERY_EDGE_REQUIRES_96_CANDIDATES")
    usable = [
        row
        for row in package.candidates
        if str(row.get("family")) in IMPLEMENTED_FAMILIES
        and set(row.get("required_datasets", ())).issubset({"DS001", "DS002"})
    ]
    if not usable:
        raise RuntimeError("NO_USABLE_CAUSAL_TEMPLATES")
    base = next(
        (row for row in usable if str(row.get("family")) == "short_horizon_reversal"),
        usable[0],
    )
    generation = batch_id - 24
    core_variants = (
        {
            "rsi_window": 4,
            "lower": 28,
            "upper": 72,
            "rsi_trend_window": 220,
            "reversal_window": 5,
            "reversal_threshold_pct": 0.825,
            "reversal_trend_window": 40,
            "drawdown_lookback": 252,
            "drawdown_trigger_pct": 22.5,
            "recovery_memory_window": 189,
            "recovery_window": 20,
            "recovery_threshold_pct": 2.25,
        },
        {
            "rsi_window": 4,
            "lower": 28,
            "upper": 72,
            "rsi_trend_window": 220,
            "reversal_window": 5,
            "reversal_threshold_pct": 0.825,
            "reversal_trend_window": 40,
            "drawdown_lookback": 252,
            "drawdown_trigger_pct": 22.5,
            "recovery_memory_window": 63,
            "recovery_window": 20,
            "recovery_threshold_pct": 2.25,
        },
    )
    parameters_list = [
        {
            **core,
            "edge_window": edge_window,
            "edge_threshold_pct": round(
                edge_threshold_pct + generation * 0.5,
                4,
            ),
        }
        for core in core_variants
        for edge_window in (10, 20, 40, 63, 126, 252)
        for edge_threshold_pct in (-10.0, -5.0, -2.5, 0.0, 2.5, 5.0, 10.0, 15.0)
    ]
    candidates: list[dict[str, Any]] = []
    for index, parameters in enumerate(parameters_list):
        candidate = json.loads(json.dumps(base))
        candidate.update(
            {
                "instrument": "SPY",
                "cash_allowed": False,
                "partial_exposure_allowed": False,
                "leverage_allowed": False,
                "volatility_scaling_allowed": False,
                "pyramiding_allowed": False,
                "multiple_assets_in_portfolio": False,
                "strategy_id": f"AUTO-B{batch_id:04d}-{index:04d}",
                "variant_label": (f"autonomous_adaptive_recovery_edge_batch_{batch_id}_{index}"),
                "family": "adaptive_recovery_edge_switch",
                "family_name": "Adaptive Recovery Edge Switch",
                "parameters": parameters,
                "required_datasets": ["DS001", "DS002"],
                "feature_formulas": [
                    "causal recovery reversal inverted only when its own fully observed trailing return is below threshold"
                ],
                "long_rule": (
                    "recovery reversal long, or its inverse when trailing realized edge is weak"
                ),
                "short_rule": (
                    "recovery reversal short, or its inverse when trailing realized edge is weak"
                ),
                "features": ["AUTO_ADAPTIVE_RECOVERY_EDGE_SWITCH"],
                "warmup_rule": (
                    "No signal before the recovery base and lagged edge window are both defined."
                ),
                "known_failure_modes": (
                    "Train-only adaptive meta-rule; must pass every frozen robustness gate."
                ),
                "economic_sign_rationale": (
                    "A signal can alternate between persistent and anti-persistent regimes; only settled returns may switch its orientation."
                ),
                "priority_score": max(1, 100 - index),
                "evidence_track": "pre_2011_evidence",
                "selection_role": "autonomous_pre_registered_candidate",
            }
        )
        candidate["canonical_hash"] = canonical_rule_hash(candidate)
        assert_contract(candidate)
        candidates.append(candidate)
    if len(candidates) != count or len({row["canonical_hash"] for row in candidates}) != count:
        raise RuntimeError("ADAPTIVE_RECOVERY_EDGE_COUNT_OR_HASH_MISMATCH")
    return tuple(candidates)


def _recovery_overnight_tug_candidates(
    package: CampaignPackage,
    batch_id: int,
    count: int,
) -> tuple[dict[str, Any], ...]:
    """Combine the strongest recovery rule with causal overnight tug."""

    if count != 96:
        raise ValueError("RECOVERY_OVERNIGHT_TUG_REQUIRES_96_CANDIDATES")
    usable = [
        row
        for row in package.candidates
        if str(row.get("family")) in IMPLEMENTED_FAMILIES
        and set(row.get("required_datasets", ())).issubset({"DS001", "DS002"})
    ]
    if not usable:
        raise RuntimeError("NO_USABLE_CAUSAL_TEMPLATES")
    base = next(
        (row for row in usable if str(row.get("family")) == "short_horizon_reversal"),
        usable[0],
    )
    generation = batch_id - 25
    core_variants = (
        {
            "rsi_window": 4,
            "lower": 28,
            "upper": 72,
            "rsi_trend_window": 220,
            "reversal_window": 5,
            "reversal_threshold_pct": 0.825,
            "reversal_trend_window": 40,
            "drawdown_lookback": 252,
            "drawdown_trigger_pct": 22.5,
            "recovery_memory_window": 189,
            "recovery_window": 20,
            "recovery_threshold_pct": 2.25,
        },
        {
            "rsi_window": 4,
            "lower": 28,
            "upper": 72,
            "rsi_trend_window": 220,
            "reversal_window": 5,
            "reversal_threshold_pct": 0.825,
            "reversal_trend_window": 40,
            "drawdown_lookback": 252,
            "drawdown_trigger_pct": 22.5,
            "recovery_memory_window": 63,
            "recovery_window": 20,
            "recovery_threshold_pct": 2.25,
        },
    )
    parameters_list = [
        {
            **core,
            "tug_lookback": tug_lookback,
            "tug_weight": round(tug_weight + generation * 0.05, 4),
        }
        for core in core_variants
        for tug_lookback in (1, 2, 3, 5, 10, 20)
        for tug_weight in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)
    ]
    candidates: list[dict[str, Any]] = []
    for index, parameters in enumerate(parameters_list):
        candidate = json.loads(json.dumps(base))
        candidate.update(
            {
                "instrument": "SPY",
                "cash_allowed": False,
                "partial_exposure_allowed": False,
                "leverage_allowed": False,
                "volatility_scaling_allowed": False,
                "pyramiding_allowed": False,
                "multiple_assets_in_portfolio": False,
                "strategy_id": f"AUTO-B{batch_id:04d}-{index:04d}",
                "variant_label": (f"autonomous_recovery_overnight_tug_batch_{batch_id}_{index}"),
                "family": "recovery_overnight_tug_vote",
                "family_name": "Recovery Overnight Tug Vote",
                "parameters": parameters,
                "required_datasets": ["DS001", "DS002"],
                "feature_formulas": [
                    "causal recovery reversal plus the settled overnight-minus-intraday tug"
                ],
                "long_rule": (
                    "weighted recovery-reversal and overnight tug score is positive, or recovery is confirmed"
                ),
                "short_rule": (
                    "weighted recovery-reversal and overnight tug score is negative outside recovery"
                ),
                "features": ["AUTO_RECOVERY_OVERNIGHT_TUG_VOTE"],
                "warmup_rule": (
                    "No signal before recovery, RSI, reversal, and overnight tug inputs are defined."
                ),
                "known_failure_modes": (
                    "Train-only mechanism combination; must pass every frozen robustness gate."
                ),
                "economic_sign_rationale": (
                    "Overnight and regular-session price pressure can diverge, while post-drawdown recovery protects rebound regimes."
                ),
                "priority_score": max(1, 100 - index),
                "evidence_track": "pre_2011_evidence",
                "selection_role": "autonomous_pre_registered_candidate",
            }
        )
        candidate["canonical_hash"] = canonical_rule_hash(candidate)
        assert_contract(candidate)
        candidates.append(candidate)
    if len(candidates) != count or len({row["canonical_hash"] for row in candidates}) != count:
        raise RuntimeError("RECOVERY_OVERNIGHT_TUG_COUNT_OR_HASH_MISMATCH")
    return tuple(candidates)


def _recovery_turn_month_candidates(
    package: CampaignPackage,
    batch_id: int,
    count: int,
) -> tuple[dict[str, Any], ...]:
    """Combine the strongest recovery rule with causal turn-of-month support."""

    if count != 96:
        raise ValueError("RECOVERY_TURN_MONTH_REQUIRES_96_CANDIDATES")
    usable = [
        row
        for row in package.candidates
        if str(row.get("family")) in IMPLEMENTED_FAMILIES
        and set(row.get("required_datasets", ())).issubset({"DS001", "DS002"})
    ]
    if not usable:
        raise RuntimeError("NO_USABLE_CAUSAL_TEMPLATES")
    base = next(
        (row for row in usable if str(row.get("family")) == "short_horizon_reversal"),
        usable[0],
    )
    generation = batch_id - 26
    core_variants = (
        {
            "rsi_window": 4,
            "lower": 28,
            "upper": 72,
            "rsi_trend_window": 220,
            "reversal_window": 5,
            "reversal_threshold_pct": 0.825,
            "reversal_trend_window": 40,
            "drawdown_lookback": 252,
            "drawdown_trigger_pct": 22.5,
            "recovery_memory_window": recovery_memory_window,
            "recovery_window": 20,
            "recovery_threshold_pct": 2.25,
        }
        for recovery_memory_window in (63, 189)
    )
    parameters_list = [
        {
            **core,
            "first_sessions": first_sessions,
            "last_sessions": last_sessions,
            "calendar_weight": round(calendar_weight + generation * 0.05, 4),
        }
        for core in core_variants
        for first_sessions in (1, 2, 3, 4)
        for last_sessions in (1, 2, 3, 4)
        for calendar_weight in (0.5, 1.5, 3.0)
    ]
    candidates: list[dict[str, Any]] = []
    for index, parameters in enumerate(parameters_list):
        candidate = json.loads(json.dumps(base))
        candidate.update(
            {
                "instrument": "SPY",
                "cash_allowed": False,
                "partial_exposure_allowed": False,
                "leverage_allowed": False,
                "volatility_scaling_allowed": False,
                "pyramiding_allowed": False,
                "multiple_assets_in_portfolio": False,
                "strategy_id": f"AUTO-B{batch_id:04d}-{index:04d}",
                "variant_label": (f"autonomous_recovery_turn_month_batch_{batch_id}_{index}"),
                "family": "recovery_turn_month_vote",
                "family_name": "Recovery Turn-Month Vote",
                "parameters": parameters,
                "required_datasets": ["DS001", "DS002"],
                "research_source_ids": sorted(
                    {
                        *base.get("research_source_ids", ()),
                        "SRC0048",
                        "SRC0175",
                        "SRC0176",
                    }
                ),
                "feature_formulas": [
                    "causal recovery reversal plus a known next-session turn-of-month vote"
                ],
                "long_rule": (
                    "weighted recovery-reversal score is positive, the next session is near a month boundary, or recovery is confirmed"
                ),
                "short_rule": (
                    "weighted recovery-reversal score is negative outside recovery and month-boundary support"
                ),
                "features": ["AUTO_RECOVERY_TURN_MONTH_VOTE"],
                "warmup_rule": (
                    "No signal before recovery, RSI, reversal, and next-session calendar inputs are defined."
                ),
                "known_failure_modes": (
                    "Published calendar effects can decay; the train-only combination must pass every frozen robustness gate."
                ),
                "economic_sign_rationale": (
                    "Month-boundary institutional cash flows can support equities while post-drawdown recovery protects rebound regimes."
                ),
                "priority_score": max(1, 100 - index),
                "evidence_track": "pre_2011_evidence",
                "selection_role": "autonomous_pre_registered_candidate",
            }
        )
        candidate["canonical_hash"] = canonical_rule_hash(candidate)
        assert_contract(candidate)
        candidates.append(candidate)
    if len(candidates) != count or len({row["canonical_hash"] for row in candidates}) != count:
        raise RuntimeError("RECOVERY_TURN_MONTH_COUNT_OR_HASH_MISMATCH")
    return tuple(candidates)


def _recovery_internal_bar_strength_candidates(
    package: CampaignPackage,
    batch_id: int,
    count: int,
) -> tuple[dict[str, Any], ...]:
    """Combine recovery reversal with causal internal-bar-strength reversal."""

    if count != 96:
        raise ValueError("RECOVERY_INTERNAL_BAR_STRENGTH_REQUIRES_96_CANDIDATES")
    usable = [
        row
        for row in package.candidates
        if str(row.get("family")) in IMPLEMENTED_FAMILIES
        and set(row.get("required_datasets", ())).issubset({"DS001", "DS002"})
    ]
    if not usable:
        raise RuntimeError("NO_USABLE_CAUSAL_TEMPLATES")
    base = next(
        (row for row in usable if str(row.get("family")) == "short_horizon_reversal"),
        usable[0],
    )
    generation = batch_id - 27
    core_variants = (
        {
            "rsi_window": 4,
            "lower": 28,
            "upper": 72,
            "rsi_trend_window": 220,
            "rsi_weight": 1.0,
            "reversal_window": 5,
            "reversal_threshold_pct": 0.825,
            "reversal_trend_window": 40,
            "reversal_weight": 1.0,
            "drawdown_lookback": 252,
            "drawdown_trigger_pct": 22.5,
            "recovery_memory_window": recovery_memory_window,
            "recovery_window": 20,
            "recovery_threshold_pct": 2.25,
        }
        for recovery_memory_window in (63, 189)
    )
    parameters_list = [
        {
            **core,
            "ibs_smooth_window": smooth_window,
            "ibs_lower": lower,
            "ibs_upper": round(1.0 - lower, 4),
            "ibs_weight": round(weight + generation * 0.05, 4),
        }
        for core in core_variants
        for smooth_window in (1, 2, 3, 5)
        for lower in (0.1, 0.2, 0.3, 0.4)
        for weight in (0.5, 1.5, 3.0)
    ]
    candidates: list[dict[str, Any]] = []
    for index, parameters in enumerate(parameters_list):
        candidate = json.loads(json.dumps(base))
        candidate.update(
            {
                "instrument": "SPY",
                "cash_allowed": False,
                "partial_exposure_allowed": False,
                "leverage_allowed": False,
                "volatility_scaling_allowed": False,
                "pyramiding_allowed": False,
                "multiple_assets_in_portfolio": False,
                "strategy_id": f"AUTO-B{batch_id:04d}-{index:04d}",
                "variant_label": (
                    f"autonomous_recovery_internal_bar_strength_batch_{batch_id}_{index}"
                ),
                "family": "recovery_internal_bar_strength_vote",
                "family_name": "Recovery Internal-Bar-Strength Vote",
                "parameters": parameters,
                "required_datasets": ["DS001", "DS002"],
                "feature_formulas": [
                    "causal recovery reversal plus smoothed internal-bar-strength reversal"
                ],
                "long_rule": (
                    "weighted recovery-reversal and low internal-bar-strength score is positive, or recovery is confirmed"
                ),
                "short_rule": (
                    "weighted recovery-reversal and high internal-bar-strength score is negative outside recovery"
                ),
                "features": ["AUTO_RECOVERY_INTERNAL_BAR_STRENGTH_VOTE"],
                "warmup_rule": (
                    "No signal before recovery, RSI, reversal, and internal-bar-strength inputs are defined."
                ),
                "known_failure_modes": (
                    "Intraday range reversal can decay or duplicate close-return reversal; every signal is deduplicated and must pass all frozen gates."
                ),
                "economic_sign_rationale": (
                    "Closing near an intraday range extreme can reflect short-lived price pressure that mean-reverts at the next open."
                ),
                "priority_score": max(1, 100 - index),
                "evidence_track": "pre_2011_evidence",
                "selection_role": "autonomous_pre_registered_candidate",
            }
        )
        candidate["canonical_hash"] = canonical_rule_hash(candidate)
        assert_contract(candidate)
        candidates.append(candidate)
    if len(candidates) != count or len({row["canonical_hash"] for row in candidates}) != count:
        raise RuntimeError("RECOVERY_INTERNAL_BAR_STRENGTH_COUNT_OR_HASH_MISMATCH")
    return tuple(candidates)


def _recovery_multi_horizon_candidates(
    package: CampaignPackage,
    batch_id: int,
    count: int,
) -> tuple[dict[str, Any], ...]:
    """Combine recovery reversal with thresholded multi-horizon reversal."""

    if count != 96:
        raise ValueError("RECOVERY_MULTI_HORIZON_REQUIRES_96_CANDIDATES")
    usable = [
        row
        for row in package.candidates
        if str(row.get("family")) in IMPLEMENTED_FAMILIES
        and set(row.get("required_datasets", ())).issubset({"DS001", "DS002"})
    ]
    if not usable:
        raise RuntimeError("NO_USABLE_CAUSAL_TEMPLATES")
    base = next(
        (row for row in usable if str(row.get("family")) == "short_horizon_reversal"),
        usable[0],
    )
    generation = batch_id - 28
    core_variants = (
        {
            "rsi_window": 4,
            "lower": 28,
            "upper": 72,
            "rsi_trend_window": 220,
            "rsi_weight": 1.0,
            "reversal_window": 5,
            "reversal_threshold_pct": 0.825,
            "reversal_trend_window": 40,
            "reversal_weight": 1.0,
            "drawdown_lookback": 252,
            "drawdown_trigger_pct": 22.5,
            "recovery_memory_window": recovery_memory_window,
            "recovery_window": 20,
            "recovery_threshold_pct": 2.25,
        }
        for recovery_memory_window in (63, 189)
    )
    parameters_list = [
        {
            **core,
            "multi_horizons": list(horizons),
            "multi_threshold_pct": round(threshold + generation * 0.025, 4),
            "multi_reversal_weight": weight,
        }
        for core in core_variants
        for horizons in ((1, 2), (1, 3, 5), (2, 3, 5), (1, 2, 3, 5))
        for threshold in (0.0, 0.25, 0.5, 1.0)
        for weight in (0.5, 1.5, 3.0)
    ]
    candidates: list[dict[str, Any]] = []
    for index, parameters in enumerate(parameters_list):
        candidate = json.loads(json.dumps(base))
        candidate.update(
            {
                "instrument": "SPY",
                "cash_allowed": False,
                "partial_exposure_allowed": False,
                "leverage_allowed": False,
                "volatility_scaling_allowed": False,
                "pyramiding_allowed": False,
                "multiple_assets_in_portfolio": False,
                "strategy_id": f"AUTO-B{batch_id:04d}-{index:04d}",
                "variant_label": (f"autonomous_recovery_multi_horizon_batch_{batch_id}_{index}"),
                "family": "recovery_multi_horizon_reversal_vote",
                "family_name": "Recovery Multi-Horizon Reversal Vote",
                "parameters": parameters,
                "required_datasets": ["DS001", "DS002"],
                "feature_formulas": [
                    "causal recovery reversal plus thresholded mean reversal across declared price horizons"
                ],
                "long_rule": (
                    "weighted recovery-reversal and negative multi-horizon return score is positive, or recovery is confirmed"
                ),
                "short_rule": (
                    "weighted recovery-reversal and positive multi-horizon return score is negative outside recovery"
                ),
                "features": ["AUTO_RECOVERY_MULTI_HORIZON_REVERSAL_VOTE"],
                "warmup_rule": (
                    "No signal before recovery, RSI, and every declared return horizon are defined."
                ),
                "known_failure_modes": (
                    "Correlated horizons can duplicate the base reversal signal; canonical and signal-level dedupe remain mandatory."
                ),
                "economic_sign_rationale": (
                    "Price pressure measured across several short horizons can identify temporary overreaction more robustly than one horizon alone."
                ),
                "priority_score": max(1, 100 - index),
                "evidence_track": "pre_2011_evidence",
                "selection_role": "autonomous_pre_registered_candidate",
            }
        )
        candidate["canonical_hash"] = canonical_rule_hash(candidate)
        assert_contract(candidate)
        candidates.append(candidate)
    if len(candidates) != count or len({row["canonical_hash"] for row in candidates}) != count:
        raise RuntimeError("RECOVERY_MULTI_HORIZON_COUNT_OR_HASH_MISMATCH")
    return tuple(candidates)


def _recovery_volume_gated_candidates(
    package: CampaignPackage,
    batch_id: int,
    count: int,
) -> tuple[dict[str, Any], ...]:
    """Combine recovery reversal with a causal high-volume reversal vote."""

    if count != 96:
        raise ValueError("RECOVERY_VOLUME_GATED_REQUIRES_96_CANDIDATES")
    usable = [
        row
        for row in package.candidates
        if str(row.get("family")) in IMPLEMENTED_FAMILIES
        and set(row.get("required_datasets", ())).issubset({"DS001", "DS002"})
    ]
    if not usable:
        raise RuntimeError("NO_USABLE_CAUSAL_TEMPLATES")
    base = next(
        (row for row in usable if str(row.get("family")) == "short_horizon_reversal"),
        usable[0],
    )
    generation = batch_id - 29
    core_variants = (
        {
            "rsi_window": 4,
            "lower": 28,
            "upper": 72,
            "rsi_trend_window": 220,
            "rsi_weight": 1.0,
            "reversal_window": 5,
            "reversal_threshold_pct": 0.825,
            "reversal_trend_window": 40,
            "reversal_weight": 1.0,
            "drawdown_lookback": 252,
            "drawdown_trigger_pct": 22.5,
            "recovery_memory_window": recovery_memory_window,
            "recovery_window": 20,
            "recovery_threshold_pct": 2.25,
        }
        for recovery_memory_window in (63, 189)
    )
    parameters_list = [
        {
            **core,
            "volume_return_lookback": return_lookback,
            "volume_window": volume_window,
            "volume_z_threshold": round(z_threshold + generation * 0.025, 4),
            "volume_reversal_weight": weight,
        }
        for core in core_variants
        for return_lookback in (1, 2, 5, 10)
        for volume_window in (20, 63)
        for z_threshold in (0.75, 1.5)
        for weight in (0.5, 1.5, 3.0)
    ]
    candidates: list[dict[str, Any]] = []
    volume_sources = {"SRC0047", "SRC0050", "SRC0133"}
    for index, parameters in enumerate(parameters_list):
        candidate = json.loads(json.dumps(base))
        candidate.update(
            {
                "instrument": "SPY",
                "cash_allowed": False,
                "partial_exposure_allowed": False,
                "leverage_allowed": False,
                "volatility_scaling_allowed": False,
                "pyramiding_allowed": False,
                "multiple_assets_in_portfolio": False,
                "strategy_id": f"AUTO-B{batch_id:04d}-{index:04d}",
                "variant_label": (f"autonomous_recovery_volume_gated_batch_{batch_id}_{index}"),
                "family": "recovery_volume_gated_reversal_vote",
                "family_name": "Recovery Volume-Gated Reversal Vote",
                "parameters": parameters,
                "required_datasets": ["DS001", "DS002"],
                "research_source_ids": sorted(
                    {*base.get("research_source_ids", ()), *volume_sources}
                ),
                "feature_formulas": [
                    "causal recovery reversal plus return reversal gated by rolling abnormal volume"
                ],
                "long_rule": (
                    "weighted recovery-reversal and high-volume negative-return score is positive, or recovery is confirmed"
                ),
                "short_rule": (
                    "weighted recovery-reversal and high-volume positive-return score is negative outside recovery"
                ),
                "features": ["AUTO_RECOVERY_VOLUME_GATED_REVERSAL_VOTE"],
                "warmup_rule": (
                    "No signal before recovery, RSI, return, and rolling volume inputs are defined."
                ),
                "known_failure_modes": (
                    "Volume definitions can change and abnormal-volume reversal may overlap with price reversal; dedupe and all frozen gates remain mandatory."
                ),
                "economic_sign_rationale": (
                    "Unusually heavy trading can mark temporary price pressure whose next-open direction mean-reverts."
                ),
                "priority_score": max(1, 100 - index),
                "evidence_track": "pre_2011_evidence",
                "selection_role": "autonomous_pre_registered_candidate",
            }
        )
        candidate["canonical_hash"] = canonical_rule_hash(candidate)
        assert_contract(candidate)
        candidates.append(candidate)
    if len(candidates) != count or len({row["canonical_hash"] for row in candidates}) != count:
        raise RuntimeError("RECOVERY_VOLUME_GATED_COUNT_OR_HASH_MISMATCH")
    return tuple(candidates)


def _recovery_calendar_volume_candidates(
    package: CampaignPackage,
    batch_id: int,
    count: int,
) -> tuple[dict[str, Any], ...]:
    """Combine the strongest train-only calendar and volume mechanisms."""

    if count != 96:
        raise ValueError("RECOVERY_CALENDAR_VOLUME_REQUIRES_96_CANDIDATES")
    usable = [
        row
        for row in package.candidates
        if str(row.get("family")) in IMPLEMENTED_FAMILIES
        and set(row.get("required_datasets", ())).issubset({"DS001", "DS002"})
    ]
    if not usable:
        raise RuntimeError("NO_USABLE_CAUSAL_TEMPLATES")
    base = next(
        (row for row in usable if str(row.get("family")) == "short_horizon_reversal"),
        usable[0],
    )
    generation = batch_id - 30
    core_variants = (
        {
            "rsi_window": 4,
            "lower": 28,
            "upper": 72,
            "rsi_trend_window": 220,
            "rsi_weight": 1.0,
            "reversal_window": 5,
            "reversal_threshold_pct": 0.825,
            "reversal_trend_window": 40,
            "reversal_weight": 1.0,
            "drawdown_lookback": 252,
            "drawdown_trigger_pct": 22.5,
            "recovery_memory_window": recovery_memory_window,
            "recovery_window": 20,
            "recovery_threshold_pct": 2.25,
        }
        for recovery_memory_window in (63, 189)
    )
    parameters_list = [
        {
            **core,
            "first_sessions": first_sessions,
            "last_sessions": last_sessions,
            "calendar_weight": calendar_weight,
            "volume_return_lookback": return_lookback,
            "volume_window": 20,
            "volume_z_threshold": round(z_threshold + generation * 0.025, 4),
            "volume_reversal_weight": volume_weight,
        }
        for core in core_variants
        for first_sessions, last_sessions in ((2, 1), (3, 2))
        for calendar_weight in (0.5, 1.5)
        for return_lookback in (1, 5)
        for z_threshold in (0.75, 1.5)
        for volume_weight in (0.5, 1.5, 3.0)
    ]
    candidates: list[dict[str, Any]] = []
    source_ids = {
        "SRC0047",
        "SRC0048",
        "SRC0050",
        "SRC0133",
        "SRC0175",
        "SRC0176",
    }
    for index, parameters in enumerate(parameters_list):
        candidate = json.loads(json.dumps(base))
        candidate.update(
            {
                "instrument": "SPY",
                "cash_allowed": False,
                "partial_exposure_allowed": False,
                "leverage_allowed": False,
                "volatility_scaling_allowed": False,
                "pyramiding_allowed": False,
                "multiple_assets_in_portfolio": False,
                "strategy_id": f"AUTO-B{batch_id:04d}-{index:04d}",
                "variant_label": (f"autonomous_recovery_calendar_volume_batch_{batch_id}_{index}"),
                "family": "recovery_calendar_volume_reversal_vote",
                "family_name": "Recovery Calendar-Volume Reversal Vote",
                "parameters": parameters,
                "required_datasets": ["DS001", "DS002"],
                "research_source_ids": sorted({*base.get("research_source_ids", ()), *source_ids}),
                "feature_formulas": [
                    "causal recovery reversal plus next-session month-boundary and abnormal-volume reversal votes"
                ],
                "long_rule": (
                    "weighted recovery-reversal, next-session month-boundary, and high-volume negative-return score is positive, or recovery is confirmed"
                ),
                "short_rule": (
                    "the combined score is negative outside recovery and month-boundary support"
                ),
                "features": ["AUTO_RECOVERY_CALENDAR_VOLUME_REVERSAL_VOTE"],
                "warmup_rule": (
                    "No signal before recovery, RSI, return, rolling volume, and next-session calendar inputs are defined."
                ),
                "known_failure_modes": (
                    "Calendar and volume effects can overlap with the base reversal; canonical and signal-level dedupe plus every frozen gate remain mandatory."
                ),
                "economic_sign_rationale": (
                    "Institutional month-boundary flows and high-volume price pressure are distinct causal votes that may stabilize the recovery-reversal edge."
                ),
                "priority_score": max(1, 100 - index),
                "evidence_track": "pre_2011_evidence",
                "selection_role": "autonomous_pre_registered_candidate",
            }
        )
        candidate["canonical_hash"] = canonical_rule_hash(candidate)
        assert_contract(candidate)
        candidates.append(candidate)
    if len(candidates) != count or len({row["canonical_hash"] for row in candidates}) != count:
        raise RuntimeError("RECOVERY_CALENDAR_VOLUME_COUNT_OR_HASH_MISMATCH")
    return tuple(candidates)


def _recovery_calendar_volume_vxo_candidates(
    package: CampaignPackage,
    batch_id: int,
    count: int,
) -> tuple[dict[str, Any], ...]:
    """Test causal VXO stress votes around the strongest train-only rule."""

    if count != 96:
        raise ValueError("RECOVERY_CALENDAR_VOLUME_VXO_REQUIRES_96_CANDIDATES")
    usable = [
        row
        for row in package.candidates
        if str(row.get("family")) in IMPLEMENTED_FAMILIES
        and set(row.get("required_datasets", ())).issubset({"DS001", "DS002"})
    ]
    if not usable:
        raise RuntimeError("NO_USABLE_CAUSAL_TEMPLATES")
    base = next(
        (row for row in usable if str(row.get("family")) == "short_horizon_reversal"),
        usable[0],
    )
    core = {
        "rsi_window": 4,
        "lower": 28,
        "upper": 72,
        "rsi_trend_window": 220,
        "rsi_weight": 1.0,
        "reversal_window": 5,
        "reversal_threshold_pct": 0.825,
        "reversal_trend_window": 40,
        "reversal_weight": 1.0,
        "drawdown_lookback": 252,
        "drawdown_trigger_pct": 22.5,
        "recovery_memory_window": 63,
        "recovery_window": 20,
        "recovery_threshold_pct": 2.25,
        "volume_return_lookback": 1,
        "volume_window": 20,
        "volume_z_threshold": 1.5,
        "volume_reversal_weight": 0.5,
        "vxo_z_window": 20,
    }
    if batch_id == 32:
        parameters_list = [
            {
                **core,
                "first_sessions": first_sessions,
                "last_sessions": last_sessions,
                "calendar_weight": calendar_weight,
                "vxo_change_lookback": vxo_lookback,
                "vxo_z_threshold": vxo_threshold,
                "vxo_weight": vxo_weight,
                "vxo_mode": vxo_mode,
            }
            for first_sessions, last_sessions in ((2, 1), (3, 2))
            for calendar_weight in (0.5, 1.5)
            for vxo_lookback in (1, 5)
            for vxo_threshold in (0.75, 1.5)
            for vxo_weight in (0.5, 1.5, 3.0)
            for vxo_mode in ("reversal", "continuation")
        ]
    elif batch_id == 33:
        # Batch 32 established two train-only neighborhoods: an immediate VXO
        # continuation vote and a weaker five-session reversal vote. Explore
        # their duration and activation boundary without repeating an observed
        # rule.
        branches = (
            ("continuation", 1, (2.0, 3.0, 4.0)),
            ("reversal", 5, (0.25, 0.75, 1.25)),
        )
        parameters_list = [
            {
                **core,
                "first_sessions": 2,
                "last_sessions": 1,
                "calendar_weight": 1.5,
                "vxo_change_lookback": vxo_lookback,
                "vxo_z_window": vxo_window,
                "vxo_z_threshold": vxo_threshold,
                "vxo_weight": vxo_weight,
                "vxo_mode": vxo_mode,
            }
            for vxo_mode, vxo_lookback, vxo_weights in branches
            for vxo_window in (10, 15, 30, 40)
            for vxo_threshold in (1.2, 1.4, 1.6, 1.8)
            for vxo_weight in vxo_weights
        ]
    elif batch_id == 34:
        # Batch 33 improved the train-only Sharpe from 1.40 to 1.53. Its four
        # leaders were immediate VXO continuation rules around 10 or 40 days.
        # Refine only those two neighborhoods with new activation boundaries.
        parameters_list = [
            {
                **core,
                "first_sessions": 2,
                "last_sessions": 1,
                "calendar_weight": 1.5,
                "vxo_change_lookback": 1,
                "vxo_z_window": vxo_window,
                "vxo_z_threshold": vxo_threshold,
                "vxo_weight": vxo_weight,
                "vxo_mode": "continuation",
            }
            for vxo_window in (7, 10, 12, 35, 40, 45)
            for vxo_threshold in (1.3, 1.45, 1.55, 1.7)
            for vxo_weight in (2.5, 3.0, 3.5, 4.0)
        ]
    elif batch_id == 35:
        # Batch 34 confirmed the ten-session continuation region and passed
        # every candidate-level gate. Keep that VXO rule fixed while varying
        # only the recovery memory and volume confirmation around its optimum.
        parameters_list = [
            {
                **core,
                "first_sessions": 2,
                "last_sessions": 1,
                "calendar_weight": 1.5,
                "recovery_memory_window": recovery_memory_window,
                "recovery_threshold_pct": recovery_threshold,
                "volume_z_threshold": volume_z_threshold,
                "volume_reversal_weight": volume_weight,
                "vxo_change_lookback": 1,
                "vxo_z_window": 10,
                "vxo_z_threshold": 1.55,
                "vxo_weight": 3.0,
                "vxo_mode": "continuation",
            }
            for recovery_memory_window in (42, 63, 84, 126)
            for recovery_threshold in (1.5, 2.0, 2.5, 3.0)
            for volume_z_threshold in (1.0, 1.5, 2.0)
            for volume_weight in (0.25, 0.75)
        ]
    elif batch_id == 36:
        # Batch 35 showed that symmetric VXO weights and volume refinements do
        # not improve the global tests. Separate positive and negative VXO
        # shocks while retaining the same causal one-session input lag.
        parameters_list = [
            {
                **core,
                "first_sessions": 2,
                "last_sessions": 1,
                "calendar_weight": 1.5,
                "recovery_memory_window": recovery_memory_window,
                "recovery_threshold_pct": recovery_threshold,
                "volume_z_threshold": 1.5,
                "volume_reversal_weight": 0.25,
                "vxo_change_lookback": 1,
                "vxo_z_window": 10,
                "vxo_z_threshold": vxo_threshold,
                "vxo_weight": 3.0,
                "vxo_positive_weight": positive_weight,
                "vxo_negative_weight": negative_weight,
                "vxo_mode": "continuation",
            }
            for recovery_memory_window in (42, 63, 84)
            for recovery_threshold in (1.25, 1.5)
            for vxo_threshold in (1.4, 1.55)
            for positive_weight in (2.5, 3.5, 4.5, 5.5)
            for negative_weight in (0.0, 1.0)
        ]
    elif batch_id == 37:
        # Batch 36 rejected asymmetric VXO shocks. Add an independent slow
        # trend vote to improve ordinary bull years without removing the crash
        # recovery and VXO protections that drive the leading train rule.
        parameters_list = [
            {
                **core,
                "first_sessions": 2,
                "last_sessions": 1,
                "calendar_weight": calendar_weight,
                "rsi_weight": rsi_weight,
                "recovery_memory_window": 63,
                "recovery_threshold_pct": 1.5,
                "volume_z_threshold": 1.5,
                "volume_reversal_weight": 0.25,
                "vxo_change_lookback": 1,
                "vxo_z_window": 10,
                "vxo_z_threshold": 1.55,
                "vxo_weight": 3.0,
                "vxo_mode": "continuation",
                "slow_trend_window": slow_trend_window,
                "slow_trend_weight": slow_trend_weight,
            }
            for slow_trend_window in (63, 126, 189, 252)
            for slow_trend_weight in (0.5, 1.0, 1.5, 2.0)
            for rsi_weight in (0.5, 1.0, 1.5)
            for calendar_weight in (1.0, 2.0)
        ]
    elif batch_id == 38:
        # The global differential statistic peaks between the symmetric
        # three-vote VXO rule and the one-vote asymmetric rule. Interpolate
        # that unobserved boundary while keeping the remaining core fixed.
        parameters_list = [
            {
                **core,
                "first_sessions": 2,
                "last_sessions": 1,
                "calendar_weight": 1.5,
                "recovery_memory_window": 63,
                "recovery_threshold_pct": recovery_threshold,
                "volume_z_threshold": 1.5,
                "volume_reversal_weight": 0.25,
                "vxo_change_lookback": 1,
                "vxo_z_window": vxo_window,
                "vxo_z_threshold": vxo_threshold,
                "vxo_weight": 3.0,
                "vxo_positive_weight": positive_weight,
                "vxo_negative_weight": negative_weight,
                "vxo_mode": "continuation",
            }
            for recovery_threshold in (1.25, 1.5)
            for vxo_window in (8, 10)
            for vxo_threshold in (1.4, 1.55)
            for positive_weight in (2.5, 3.5)
            for negative_weight in (1.25, 1.5, 1.75, 2.0, 2.25, 2.5)
        ]
    elif batch_id == 39:
        # Batch 38 produced the strongest common-interval differential with a
        # 1.25% recovery threshold and intermediate VXO weights. Add only a
        # small slow-trend vote and rebalance the two base votes to improve the
        # ordinary bull years that still trail the benchmark.
        parameters_list = [
            {
                **core,
                "first_sessions": 2,
                "last_sessions": 1,
                "calendar_weight": calendar_weight,
                "rsi_weight": rsi_weight,
                "reversal_weight": reversal_weight,
                "recovery_memory_window": 63,
                "recovery_threshold_pct": 1.25,
                "volume_z_threshold": 1.5,
                "volume_reversal_weight": 0.25,
                "vxo_change_lookback": 1,
                "vxo_z_window": 10,
                "vxo_z_threshold": 1.4,
                "vxo_weight": 3.0,
                "vxo_positive_weight": 2.5,
                "vxo_negative_weight": 2.0,
                "vxo_mode": "continuation",
                "slow_trend_window": slow_trend_window,
                "slow_trend_weight": slow_trend_weight,
            }
            for slow_trend_window in (63, 126, 252)
            for slow_trend_weight in (0.1, 0.25, 0.5, 0.75)
            for rsi_weight in (0.5, 1.0)
            for reversal_weight in (0.5, 1.0)
            for calendar_weight in (1.0, 1.5)
        ]
    elif batch_id == 40:
        # The slow-trend rebalance in batch 39 did not improve the global
        # differential tests. Combine the strongest intermediate VXO rule
        # with the already-supported causal overnight-minus-intraday tug,
        # which can distinguish ordinary bull sessions from stress reversals.
        parameters_list = [
            {
                **core,
                "first_sessions": 2,
                "last_sessions": 1,
                "calendar_weight": calendar_weight,
                "rsi_weight": rsi_weight,
                "reversal_weight": 1.0,
                "recovery_memory_window": 63,
                "recovery_threshold_pct": 1.25,
                "volume_z_threshold": 1.5,
                "volume_reversal_weight": 0.25,
                "vxo_change_lookback": 1,
                "vxo_z_window": 10,
                "vxo_z_threshold": 1.4,
                "vxo_weight": 3.0,
                "vxo_positive_weight": 2.5,
                "vxo_negative_weight": 2.0,
                "vxo_mode": "continuation",
                "tug_lookback": tug_lookback,
                "tug_weight": tug_weight,
            }
            for tug_lookback in (1, 2, 3, 5)
            for tug_weight in (0.25, 0.5, 1.0, 1.5)
            for rsi_weight in (0.5, 1.0, 1.5)
            for calendar_weight in (1.0, 1.5)
        ]
    elif batch_id == 41:
        # Batch 40 materially improved the global SPA statistic around a
        # two-session tug with weight 0.5. Refine the adjacent, previously
        # unobserved weights and vote boundaries without repeating batch 40.
        vxo_thresholds = (1.35, 1.45)
        parameters_list = [
            {
                **core,
                "first_sessions": 2,
                "last_sessions": 1,
                "calendar_weight": calendar_weight,
                "rsi_weight": rsi_weight,
                "reversal_weight": 1.0,
                "recovery_memory_window": 63,
                "recovery_threshold_pct": 1.25,
                "volume_z_threshold": 1.5,
                "volume_reversal_weight": 0.25,
                "vxo_change_lookback": 1,
                "vxo_z_window": 10,
                "vxo_z_threshold": vxo_threshold,
                "vxo_weight": 3.0,
                "vxo_positive_weight": 2.5,
                "vxo_negative_weight": 2.0,
                "vxo_mode": "continuation",
                "tug_lookback": 2,
                "tug_weight": tug_weight,
            }
            for tug_weight in (0.3, 0.4, 0.6, 0.7)
            for rsi_weight in (1.25, 1.5, 1.75)
            for calendar_weight in (0.75, 1.0, 1.25, 1.5)
            for vxo_threshold in vxo_thresholds
        ]
    elif batch_id <= 44:
        # Batch 41 moved away from the batch-40 optimum. Keep its two-session
        # tug fixed and vary only recovery activation, falling-VXO weight, and
        # the two secondary votes that can repair ordinary bull years.
        generation = batch_id - 42
        recovery_thresholds = tuple(
            round(value + generation * 0.025, 4)
            for value in (1.1, 1.2, 1.3)
        )
        parameters_list = [
            {
                **core,
                "first_sessions": 2,
                "last_sessions": 1,
                "calendar_weight": calendar_weight,
                "rsi_weight": 1.5,
                "reversal_weight": 1.0,
                "recovery_memory_window": 63,
                "recovery_threshold_pct": recovery_threshold,
                "volume_z_threshold": 1.5,
                "volume_reversal_weight": volume_weight,
                "vxo_change_lookback": 1,
                "vxo_z_window": 10,
                "vxo_z_threshold": 1.4,
                "vxo_weight": 3.0,
                "vxo_positive_weight": 2.5,
                "vxo_negative_weight": negative_weight,
                "vxo_mode": "continuation",
                "tug_lookback": 2,
                "tug_weight": 0.5,
            }
            for recovery_threshold in recovery_thresholds
            for negative_weight in (1.5, 1.75, 2.25, 2.5)
            for calendar_weight in (0.5, 0.75, 1.0, 1.25)
            for volume_weight in (0.0, 0.25)
        ]
    elif batch_id <= 46:
        # Batches 42-44 converged to duplicate decision streams. Stop a
        # falling-VXO continuation vote from forcing a short while a causal
        # price trend remains above a pre-registered boundary. Rising-VXO and
        # crash-recovery behavior remain unchanged.
        generation = batch_id - 45
        gate_thresholds = tuple(
            round(value + generation * 0.25, 4)
            for value in (-2.0, 0.0, 2.0)
        )
        parameters_list = [
            {
                **core,
                "first_sessions": 2,
                "last_sessions": 1,
                "calendar_weight": 0.5,
                "rsi_weight": 1.5,
                "reversal_weight": 1.0,
                "recovery_memory_window": 63,
                "recovery_threshold_pct": recovery_threshold,
                "volume_z_threshold": 1.5,
                "volume_reversal_weight": 0.25,
                "vxo_change_lookback": 1,
                "vxo_z_window": 10,
                "vxo_z_threshold": 1.4,
                "vxo_weight": 3.0,
                "vxo_positive_weight": 2.5,
                "vxo_negative_weight": negative_weight,
                "vxo_negative_gate_window": gate_window,
                "vxo_negative_gate_threshold_pct": gate_threshold,
                "vxo_mode": "continuation",
                "tug_lookback": 2,
                "tug_weight": 0.5,
            }
            for gate_window in (20, 40, 63, 126)
            for gate_threshold in gate_thresholds
            for negative_weight in (1.5, 2.0, 2.5, 3.0)
            for recovery_threshold in (1.2, 1.25)
        ]
    elif batch_id == 47:
        # The negative-VXO gate passed global SPA but did not restore DSR or
        # WRC. Require every final short state to agree with a causal price
        # trend; otherwise remain fully invested long rather than enter cash.
        parameters_list = [
            {
                **core,
                "first_sessions": 2,
                "last_sessions": 1,
                "calendar_weight": 0.5,
                "rsi_weight": 1.5,
                "reversal_weight": 1.0,
                "recovery_memory_window": 63,
                "recovery_threshold_pct": recovery_threshold,
                "volume_z_threshold": 1.5,
                "volume_reversal_weight": 0.25,
                "vxo_change_lookback": 1,
                "vxo_z_window": 10,
                "vxo_z_threshold": 1.4,
                "vxo_weight": 3.0,
                "vxo_positive_weight": 2.5,
                "vxo_negative_weight": 2.5,
                "vxo_negative_gate_window": 20,
                "vxo_negative_gate_threshold_pct": vxo_gate_threshold,
                "vxo_mode": "continuation",
                "tug_lookback": 2,
                "tug_weight": tug_weight,
                "short_gate_window": short_gate_window,
                "short_gate_threshold_pct": short_threshold,
            }
            for short_gate_window in (20, 40, 63, 126)
            for short_threshold in (-5.0, 0.0, 5.0)
            for recovery_threshold in (1.1, 1.2)
            for vxo_gate_threshold in (0.0, 2.0)
            for tug_weight in (0.5, 1.0)
        ]
    elif batch_id == 48:
        # Batch 45 produced the strongest common-period differential with a
        # 40-session negative-VXO trend gate. Search its unobserved local
        # neighbourhood without changing the underlying economic rule.
        parameters_list = [
            {
                **core,
                "first_sessions": 2,
                "last_sessions": 1,
                "calendar_weight": 0.5,
                "rsi_weight": 1.5,
                "reversal_weight": 1.0,
                "recovery_memory_window": 63,
                "recovery_threshold_pct": recovery_threshold,
                "volume_z_threshold": 1.5,
                "volume_reversal_weight": 0.25,
                "vxo_change_lookback": 1,
                "vxo_z_window": 10,
                "vxo_z_threshold": 1.4,
                "vxo_weight": 3.0,
                "vxo_positive_weight": 2.5,
                "vxo_negative_weight": negative_weight,
                "vxo_negative_gate_window": gate_window,
                "vxo_negative_gate_threshold_pct": gate_threshold,
                "vxo_mode": "continuation",
                "tug_lookback": 2,
                "tug_weight": tug_weight,
            }
            for gate_window in (30, 35, 45, 50)
            for gate_threshold in (1.0, 2.0, 3.0)
            for negative_weight in (2.25, 2.5)
            for recovery_threshold in (1.15, 1.2)
            for tug_weight in (0.25, 0.5)
        ]
    elif batch_id == 49:
        # Ordinary bull years still suffer from short states. Keep the strongest
        # VXO rule but permit a final short only after a causal drawdown from a
        # recent rolling high. The alternative state remains fully invested long.
        parameters_list = [
            {
                **core,
                "first_sessions": 2,
                "last_sessions": 1,
                "calendar_weight": 0.5,
                "rsi_weight": 1.5,
                "reversal_weight": 1.0,
                "recovery_memory_window": 63,
                "recovery_threshold_pct": 1.2,
                "volume_z_threshold": 1.5,
                "volume_reversal_weight": 0.25,
                "vxo_change_lookback": 1,
                "vxo_z_window": 10,
                "vxo_z_threshold": 1.4,
                "vxo_weight": 3.0,
                "vxo_positive_weight": 2.5,
                "vxo_negative_weight": 2.5,
                "vxo_negative_gate_window": vxo_gate_window,
                "vxo_negative_gate_threshold_pct": vxo_gate_threshold,
                "vxo_mode": "continuation",
                "tug_lookback": 2,
                "tug_weight": 0.5,
                "short_drawdown_gate_window": drawdown_window,
                "short_drawdown_gate_threshold_pct": drawdown_threshold,
            }
            for drawdown_window in (63, 126, 252)
            for drawdown_threshold in (0.0, 2.5, 5.0, 7.5)
            for vxo_gate_window in (35, 40, 45, 50)
            for vxo_gate_threshold in (1.0, 2.0)
        ]
    elif batch_id == 50:
        # Batch 49 improved every global statistic at a 252-session peak and a
        # 2.5% drawdown boundary. Refine the previously unobserved neighbourhood
        # around that boundary while retaining the same causal rule.
        parameters_list = [
            {
                **core,
                "first_sessions": 2,
                "last_sessions": 1,
                "calendar_weight": 0.5,
                "rsi_weight": 1.5,
                "reversal_weight": 1.0,
                "recovery_memory_window": 63,
                "recovery_threshold_pct": 1.2,
                "volume_z_threshold": 1.5,
                "volume_reversal_weight": 0.25,
                "vxo_change_lookback": 1,
                "vxo_z_window": 10,
                "vxo_z_threshold": 1.4,
                "vxo_weight": 3.0,
                "vxo_positive_weight": 2.5,
                "vxo_negative_weight": 2.5,
                "vxo_negative_gate_window": vxo_gate_window,
                "vxo_negative_gate_threshold_pct": vxo_gate_threshold,
                "vxo_mode": "continuation",
                "tug_lookback": 2,
                "tug_weight": 0.5,
                "short_drawdown_gate_window": drawdown_window,
                "short_drawdown_gate_threshold_pct": drawdown_threshold,
            }
            for drawdown_window in (189, 220, 252, 300)
            for drawdown_threshold in (1.5, 2.0, 3.0)
            for vxo_gate_window in (30, 35, 40, 45)
            for vxo_gate_threshold in (0.5, 1.0)
        ]
    else:
        # Batch 50 placed the maximum common-period differential at the longest
        # drawdown window and deepest tested boundary. Extend only that edge.
        generation = batch_id - 51
        drawdown_windows = tuple(
            value + generation * 10
            for value in (320, 360, 420, 500)
        )
        drawdown_thresholds = tuple(
            round(value + generation * 0.1, 4)
            for value in (3.0, 3.5, 4.0, 4.5)
        )
        parameters_list = [
            {
                **core,
                "first_sessions": 2,
                "last_sessions": 1,
                "calendar_weight": 0.5,
                "rsi_weight": 1.5,
                "reversal_weight": 1.0,
                "recovery_memory_window": 63,
                "recovery_threshold_pct": 1.2,
                "volume_z_threshold": 1.5,
                "volume_reversal_weight": 0.25,
                "vxo_change_lookback": 1,
                "vxo_z_window": 10,
                "vxo_z_threshold": 1.4,
                "vxo_weight": 3.0,
                "vxo_positive_weight": 2.5,
                "vxo_negative_weight": 2.5,
                "vxo_negative_gate_window": vxo_gate_window,
                "vxo_negative_gate_threshold_pct": vxo_gate_threshold,
                "vxo_mode": "continuation",
                "tug_lookback": 2,
                "tug_weight": 0.5,
                "short_drawdown_gate_window": drawdown_window,
                "short_drawdown_gate_threshold_pct": drawdown_threshold,
            }
            for drawdown_window in drawdown_windows
            for drawdown_threshold in drawdown_thresholds
            for vxo_gate_window in (35, 40, 45)
            for vxo_gate_threshold in (0.0, 0.5)
        ]
    candidates: list[dict[str, Any]] = []
    source_ids = {
        "SRC0047",
        "SRC0048",
        "SRC0050",
        "SRC0056",
        "SRC0057",
        "SRC0058",
        "SRC0059",
        "SRC0133",
        "SRC0175",
        "SRC0176",
    }
    for index, parameters in enumerate(parameters_list):
        candidate = json.loads(json.dumps(base))
        candidate.update(
            {
                "instrument": "SPY",
                "cash_allowed": False,
                "partial_exposure_allowed": False,
                "leverage_allowed": False,
                "volatility_scaling_allowed": False,
                "pyramiding_allowed": False,
                "multiple_assets_in_portfolio": False,
                "strategy_id": f"AUTO-B{batch_id:04d}-{index:04d}",
                "variant_label": (
                    f"autonomous_recovery_calendar_volume_vxo_batch_{batch_id}_{index}"
                ),
                "family": "recovery_calendar_volume_vxo_vote",
                "family_name": "Recovery Calendar-Volume VXO Vote",
                "parameters": parameters,
                "required_datasets": ["DS001", "DS002", "DS005"],
                "dataset_classifications": [
                    "usable_after_repair",
                    "usable_after_repair",
                    "usable_after_repair",
                ],
                "research_source_ids": sorted({*base.get("research_source_ids", ()), *source_ids}),
                "feature_formulas": [
                    "causal recovery/calendar/volume score plus extreme VXO-change and optional overnight-tug votes"
                ],
                "long_rule": (
                    "combined recovery, month-boundary, volume-pressure, and VXO score is positive"
                ),
                "short_rule": (
                    "combined recovery, month-boundary, volume-pressure, and VXO score is negative"
                ),
                "features": ["AUTO_RECOVERY_CALENDAR_VOLUME_VXO_VOTE"],
                "warmup_rule": (
                    "No signal before every price, volume, calendar, and causally released VXO input is defined."
                ),
                "known_failure_modes": (
                    "VXO is an S&P 100 implied-volatility series and its equity-return sign can reverse by horizon; both pre-registered signs count in multiplicity."
                ),
                "economic_sign_rationale": (
                    "Extreme implied-volatility changes can represent either immediate risk continuation or panic reversal; both signs are tested symmetrically."
                ),
                "priority_score": max(1, 100 - index),
                "evidence_track": "pre_2011_evidence",
                "selection_role": "autonomous_pre_registered_candidate",
            }
        )
        candidate["canonical_hash"] = canonical_rule_hash(candidate)
        assert_contract(candidate)
        candidates.append(candidate)
    if len(candidates) != count or len({row["canonical_hash"] for row in candidates}) != count:
        raise RuntimeError("RECOVERY_CALENDAR_VOLUME_VXO_COUNT_OR_HASH_MISMATCH")
    return tuple(candidates)


def generate_candidates(batch_id: int, *, count: int = 96) -> tuple[dict[str, Any], ...]:
    """Generate a reproducible, pre-registered batch from causal templates."""

    if batch_id < 0 or count < 1:
        raise ValueError("INVALID_BATCH_ARGUMENT")
    package = base_package()
    if batch_id >= 32:
        return _recovery_calendar_volume_vxo_candidates(package, batch_id, count)
    if batch_id >= 30:
        return _recovery_calendar_volume_candidates(package, batch_id, count)
    if batch_id >= 29:
        return _recovery_volume_gated_candidates(package, batch_id, count)
    if batch_id >= 28:
        return _recovery_multi_horizon_candidates(package, batch_id, count)
    if batch_id >= 27:
        return _recovery_internal_bar_strength_candidates(package, batch_id, count)
    if batch_id >= 26:
        return _recovery_turn_month_candidates(package, batch_id, count)
    if batch_id >= 25:
        return _recovery_overnight_tug_candidates(package, batch_id, count)
    if batch_id >= 24:
        return _adaptive_recovery_edge_candidates(package, batch_id, count)
    if batch_id >= 23:
        return _high_vol_crash_recovery_candidates(package, batch_id, count)
    if batch_id >= 22:
        return _recovery_trend_breakout_majority_candidates(package, batch_id, count)
    if batch_id >= 21:
        return _quiet_bull_recovery_override_candidates(package, batch_id, count)
    if batch_id >= 19:
        return _drawdown_recovery_override_candidates(package, batch_id, count)
    if batch_id >= 16:
        return _asymmetric_override_candidates(package, batch_id, count)
    if batch_id >= 13:
        return _stability_refined_dual_reversal_candidates(package, batch_id, count)
    if batch_id >= 10:
        return _strong_trend_override_candidates(package, batch_id, count)
    if batch_id >= 9:
        return _overnight_tug_reversal_candidates(package, batch_id, count)
    if batch_id >= 8:
        return _volatility_regime_candidates(package, batch_id, count)
    if batch_id >= 7:
        return _trend_guarded_reversal_candidates(package, batch_id, count)
    if batch_id >= 5:
        return _combined_reversal_candidates(package, batch_id, count)
    if batch_id >= 4:
        return _neighborhood_reversal_candidates(package, batch_id, count)
    if batch_id >= 3:
        return _targeted_reversal_candidates(package, batch_id, count)
    templates = [
        row
        for row in package.candidates
        if str(row.get("family")) in IMPLEMENTED_FAMILIES
        and set(row.get("required_datasets", ())).issubset({"DS001", "DS002"})
    ]
    if not templates:
        raise RuntimeError("NO_USABLE_CAUSAL_TEMPLATES")
    rng = random.Random(_seed(batch_id))
    candidates: list[dict[str, Any]] = []
    hashes: set[str] = set()
    for index in range(count):
        template_offset = index + batch_id * 7
        for attempt in range(max(100, len(templates) * 20)):
            template = templates[(template_offset + attempt) % len(templates)]
            candidate = _mutate(template, batch_id, index + attempt * count, rng)
            digest = str(candidate["canonical_hash"])
            if digest not in hashes:
                candidate["strategy_id"] = f"AUTO-B{batch_id:04d}-{index:04d}"
                candidate["canonical_hash"] = canonical_rule_hash(candidate)
                assert_contract(candidate)
                candidates.append(candidate)
                hashes.add(digest)
                break
        else:
            raise RuntimeError("CANDIDATE_HASH_COLLISION_LIMIT")
    return tuple(candidates)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(dict(row), sort_keys=True, separators=(",", ":")) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def write_batch_registry(
    root: Path,
    *,
    batch_id: int,
    candidates: tuple[Mapping[str, Any], ...],
    previous_trial_count: int | None = None,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    write_jsonl(root / "candidate_registry.jsonl", candidates)
    package = base_package()
    research_rows = list(package.research)
    feature_rows = list(package.features)
    dataset_rows = list(package.datasets)
    with (root / "research_registry.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=sorted({key for row in research_rows for key in row})
        )
        writer.writeheader()
        writer.writerows(research_rows)
    with (root / "feature_registry.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=sorted({key for row in feature_rows for key in row})
        )
        writer.writeheader()
        writer.writerows(feature_rows)
    with (root / "dataset_registry.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=sorted({key for row in dataset_rows for key in row})
        )
        writer.writeheader()
        writer.writerows(dataset_rows)
    previous_trial_count = (
        previous_trial_count if previous_trial_count is not None else get_previous_trial_count()
    )
    new_ledger_rows = [
        {
            "batch_id": batch_id,
            "canonical_hash": str(candidate["canonical_hash"]),
            "global_trial_index": previous_trial_count + index + 1,
            "pre_registered_before_performance": True,
            "status": "registered",
            "strategy_id": str(candidate["strategy_id"]),
        }
        for index, candidate in enumerate(candidates)
    ]
    prior_ledger_value = os.environ.get("AURORA_PRIOR_TRIAL_LEDGER_PATH", "").strip()
    prior_ledger_path = Path(prior_ledger_value) if prior_ledger_value else None
    prior_ledger_rows = (
        read_jsonl(prior_ledger_path)
        if prior_ledger_path is not None and prior_ledger_path.is_file()
        else []
    )
    historical_path = Path(root) / HISTORICAL_DIR / "historical_trial_ledger.jsonl"
    historical_rows = load_historical_trial_ledger(root) if historical_path.is_file() else []
    combined_prior: dict[int, dict[str, Any]] = {}
    for row in (*historical_rows, *prior_ledger_rows):
        index = int(row.get("global_trial_index", 0))
        if index < 1 or index > previous_trial_count:
            continue
        existing = combined_prior.get(index)
        if existing is not None and str(existing.get("strategy_id")) != str(row.get("strategy_id")):
            raise ValueError(f"PRIOR_TRIAL_LEDGER_INDEX_COLLISION:{index}")
        combined_prior[index] = dict(row)
    status_lookup = load_prior_autonomous_status(root)
    for row in combined_prior.values():
        status = status_lookup.get(str(row.get("strategy_id")))
        if status is not None:
            row["status"] = str(status["status"])
            row["rejection_reason"] = str(status.get("rejection_reason") or "")
    complete_prior = [combined_prior[index] for index in sorted(combined_prior)]
    if previous_trial_count > PREVIOUS_TRIAL_COUNT and not prior_ledger_rows:
        raise ValueError("PRIOR_TRIAL_LEDGER_REQUIRED")
    if historical_rows and [int(row["global_trial_index"]) for row in complete_prior] != list(
        range(1, previous_trial_count + 1)
    ):
        raise ValueError("PRIOR_TRIAL_LEDGER_COUNT_MISMATCH")
    if not historical_rows and prior_ledger_rows:
        last_index = int(prior_ledger_rows[-1].get("global_trial_index", 0))
        if last_index != previous_trial_count:
            raise ValueError("PRIOR_TRIAL_LEDGER_COUNT_MISMATCH")
        complete_prior = list(prior_ledger_rows)
    ledger_rows = [*complete_prior, *new_ledger_rows]
    for row in ledger_rows:
        row["batch_id"] = str(row.get("batch_id", ""))
    ledger_payload = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in ledger_rows
    )
    (root / "trial_ledger.jsonl").write_text(ledger_payload, encoding="utf-8")
    pd.DataFrame(ledger_rows).to_parquet(root / "autonomous_trial_ledger.parquet", index=False)
    manifest = {
        "schema_version": "1",
        "batch_id": batch_id,
        "candidate_count": len(candidates),
        "previous_trial_count": previous_trial_count,
        "global_trial_count_after_batch": previous_trial_count + len(candidates),
        "pre_registered_before_performance": True,
        "canonical_hashes_unique": len({row["canonical_hash"] for row in candidates})
        == len(candidates),
        "train_end": TRAIN_END,
        "validation_start": VALIDATION_START,
        "validation_end": VALIDATION_END,
        "locked_start": LOCKED_START,
        "locked_opened": False,
        "validation_used_for_selection": False,
        "trial_ledger_file": "trial_ledger.jsonl",
        "trial_ledger_rows": len(ledger_rows),
        "new_trial_ledger_rows": len(new_ledger_rows),
        "prior_trial_ledger_rows": len(complete_prior),
        "historical_trial_ledger_rows": len(historical_rows),
        "trial_ledger_sha256": hashlib.sha256(ledger_payload.encode("utf-8")).hexdigest(),
        "trial_indices": [row["global_trial_index"] for row in new_ledger_rows],
    }
    (root / "candidate_registry_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def read_batch_registry(root: Path) -> tuple[dict[str, Any], ...]:
    rows = tuple(read_jsonl(Path(root) / "candidate_registry.jsonl"))
    if not rows:
        raise RuntimeError("EMPTY_CANDIDATE_REGISTRY")
    for row in rows:
        assert_contract(row)
    if len({row["strategy_id"] for row in rows}) != len(rows):
        raise RuntimeError("DUPLICATE_CANDIDATE_IDS")
    if len({row["canonical_hash"] for row in rows}) != len(rows):
        raise RuntimeError("DUPLICATE_CANDIDATE_HASHES")
    return rows
