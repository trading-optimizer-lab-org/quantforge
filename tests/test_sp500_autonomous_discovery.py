from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tomllib

import numpy as np
import pandas as pd
import pytest
import yaml

from aurora.infra.sp500_autonomous_discovery import registry
from aurora.infra.sp500_autonomous_discovery import statistics as autonomous_statistics
from aurora.infra.sp500_autonomous_discovery.contracts import canonical_rule_hash
from aurora.infra.sp500_autonomous_discovery.dedupe import build_dedupe_map
from aurora.infra.sp500_autonomous_discovery.feature_store import FeatureStore
from aurora.infra.sp500_autonomous_discovery.historical_evidence import (
    build_historical_trial_ledger,
)
from aurora.infra.sp500_autonomous_discovery.scheduling import assign_by_cost
from aurora.infra.sp500_autonomous_discovery.statistics import evaluate_batch
from aurora.infra.sp500_autonomous_discovery.workload import (
    freeze_rejection_reasons,
    freeze_selection_reason,
    refresh_autonomous_prepared_inputs,
)
from aurora.infra.sp500_autonomous_discovery.validation import (
    EXPLORATORY_VALIDATION_ACK,
    ValidationGateError,
    _candidate_from_registry,
    _verify_freeze,
    exploratory_validation_package,
)
from aurora.infra.sp500_long_short_daily.data import PreparedMarketData
from aurora.infra.sp500_long_short_daily.signals import candidate_decisions


def test_autonomous_discovery_runtime_is_packaged_in_wheel() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    setuptools = pyproject["tool"]["setuptools"]
    package = "aurora.infra.sp500_autonomous_discovery"
    assert package in setuptools["packages"]
    assert setuptools["package-dir"][package] == "infra/sp500_autonomous_discovery"


def _template() -> dict[str, object]:
    candidate = {
        "strategy_id": "template",
        "instrument": "SPY",
        "family": "price_trend_sma",
        "variant_label": "template",
        "evidence_track": "pre_2011_evidence",
        "position_values": [-1, 1],
        "absolute_exposure": 1.0,
        "cash_allowed": False,
        "partial_exposure_allowed": False,
        "leverage_allowed": False,
        "volatility_scaling_allowed": False,
        "pyramiding_allowed": False,
        "multiple_assets_in_portfolio": False,
        "locked_boundary": ">=2021-01-01 unopened",
        "commission_bps": 0.0,
        "slippage_bps": 0.0,
        "borrow_cost_bps": 0.0,
        "financing_bps": 0.0,
        "switching_cost_bps": 0.0,
        "market_impact_bps": 0.0,
        "required_datasets": ["DS001"],
        "parameters": {"window": 20, "threshold": 0.1},
        "rules": {"entry": "close > sma(window)", "exit": "reverse"},
        "complexity_score": 1,
        "priority_score": 10,
    }
    candidate["canonical_hash"] = canonical_rule_hash(candidate)
    return candidate


def test_canonical_hash_ignores_identity_but_changes_effective_rule() -> None:
    first = _template()
    second = dict(first, strategy_id="other", notes="new note", research_source_ids=["new"])
    assert canonical_rule_hash(first) == canonical_rule_hash(second)
    changed = dict(first, parameters={"window": 21, "threshold": 0.1})
    assert canonical_rule_hash(first) != canonical_rule_hash(changed)


def test_candidate_generation_is_reproducible_and_contract_bound(monkeypatch) -> None:
    fake_package = SimpleNamespace(
        candidates=(_template(),),
        research=(),
        features=(),
        datasets=(),
    )
    monkeypatch.setattr(registry, "base_package", lambda: fake_package)
    first = registry.generate_candidates(2, count=8)
    second = registry.generate_candidates(2, count=8)
    assert [row["strategy_id"] for row in first] == [row["strategy_id"] for row in second]
    assert [row["canonical_hash"] for row in first] == [row["canonical_hash"] for row in second]
    assert {row["position_values"][0] for row in first} == {-1}
    assert all(row["position_values"] == [-1, 1] for row in first)
    assert all(row["locked_boundary"] == ">=2021-01-01 unopened" for row in first)


def test_real_candidate_grid_supports_full_96_candidate_batch() -> None:
    candidates = registry.generate_candidates(1, count=96)
    assert len(candidates) == 96
    assert len({row["canonical_hash"] for row in candidates}) == 96
    assert all(row["locked_boundary"] == ">=2021-01-01 unopened" for row in candidates)


def test_targeted_batch_three_is_distinct_causal_and_full() -> None:
    candidates = registry.generate_candidates(3, count=96)
    assert len(candidates) == 96
    assert len({row["canonical_hash"] for row in candidates}) == 96
    assert {
        "rsi_reversal",
        "internal_bar_strength_reversal",
        "return_threshold_reversal",
        "streak_reversal",
        "reversal_trend_blend",
        "rsi_trend_blend",
        "multi_horizon_reversal",
        "intraday_return_reversal",
    }.issubset({row["family"] for row in candidates})
    assert all(row["required_datasets"] == ["DS001", "DS002"] for row in candidates)
    assert all(row["locked_boundary"] == ">=2021-01-01 unopened" for row in candidates)


def test_batch_four_neighborhood_is_new_balanced_and_full() -> None:
    batch_three = registry.generate_candidates(3, count=96)
    batch_four = registry.generate_candidates(4, count=96)
    assert len(batch_four) == 96
    assert len({row["canonical_hash"] for row in batch_four}) == 96
    assert {row["family"] for row in batch_four} == {
        "reversal_trend_blend",
        "rsi_trend_blend",
    }
    assert sum(row["family"] == "reversal_trend_blend" for row in batch_four) == 48
    assert sum(row["family"] == "rsi_trend_blend" for row in batch_four) == 48
    batch_three_rules = {
        (row["family"], json.dumps(row["parameters"], sort_keys=True)) for row in batch_three
    }
    assert not batch_three_rules.intersection(
        (row["family"], json.dumps(row["parameters"], sort_keys=True)) for row in batch_four
    )


def test_batch_five_combines_and_refines_without_repeating_batch_four() -> None:
    batch_four = registry.generate_candidates(4, count=96)
    batch_five = registry.generate_candidates(5, count=96)
    assert len(batch_five) == 96
    assert len({row["canonical_hash"] for row in batch_five}) == 96
    assert sum(row["family"] == "dual_reversal_trend_vote" for row in batch_five) == 48
    assert sum(row["family"] == "rsi_trend_blend" for row in batch_five) == 48
    batch_four_rules = {
        (row["family"], json.dumps(row["parameters"], sort_keys=True)) for row in batch_four
    }
    assert not batch_four_rules.intersection(
        (row["family"], json.dumps(row["parameters"], sort_keys=True)) for row in batch_five
    )


def test_batch_six_uses_only_new_combined_rules() -> None:
    batch_five = registry.generate_candidates(5, count=96)
    batch_six = registry.generate_candidates(6, count=96)
    assert len(batch_six) == 96
    assert {row["family"] for row in batch_six} == {"dual_reversal_trend_vote"}
    batch_five_rules = {
        (row["family"], json.dumps(row["parameters"], sort_keys=True)) for row in batch_five
    }
    assert not batch_five_rules.intersection(
        (row["family"], json.dumps(row["parameters"], sort_keys=True)) for row in batch_six
    )


def test_batch_seven_adds_unique_causal_trend_guards() -> None:
    batch_six = registry.generate_candidates(6, count=96)
    batch_seven = registry.generate_candidates(7, count=96)
    assert len(batch_seven) == 96
    assert {row["family"] for row in batch_seven} == {"trend_guarded_dual_reversal"}
    assert all(row["position_values"] == [-1, 1] for row in batch_seven)
    assert all(row["cash_allowed"] is False for row in batch_seven)
    assert not {row["canonical_hash"] for row in batch_six}.intersection(
        row["canonical_hash"] for row in batch_seven
    )


def test_trend_guarded_dual_reversal_is_causal_and_fully_covered() -> None:
    index = pd.date_range("2000-01-03", periods=300, freq="B")
    close = pd.Series(np.linspace(100.0, 200.0, len(index)), index=index)
    ledger = pd.DataFrame(
        {
            "tr_close": close,
            "tr_open": close,
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "volume": 1000.0,
            "long_return": close.pct_change().fillna(0.0),
            "short_return": -close.pct_change().fillna(0.0),
        },
        index=index,
    )
    candidate = registry.generate_candidates(7, count=1)[0]
    result = candidate_decisions(
        candidate,
        PreparedMarketData(
            ledger=ledger,
            series={},
            available_dataset_ids=frozenset({"DS001", "DS002"}),
            rejected_datasets={},
            receipts=(),
            split="train",
        ),
    )
    expected_warmup = max(
        int(candidate["parameters"]["rsi_trend_window"]),
        int(candidate["parameters"]["reversal_trend_window"]),
        int(candidate["parameters"]["guard_trend_window"]),
    )
    assert result.first_evaluable_date == index[expected_warmup].date().isoformat()
    assert result.missing_fraction == 0.0


def test_batch_eight_adds_unique_volatility_regime_rules() -> None:
    batch_seven = registry.generate_candidates(7, count=96)
    batch_eight = registry.generate_candidates(8, count=96)
    assert len(batch_eight) == 96
    assert {row["family"] for row in batch_eight} == {"volatility_regime_reversal"}
    assert all(row["position_values"] == [-1, 1] for row in batch_eight)
    assert all(row["cash_allowed"] is False for row in batch_eight)
    assert not {row["canonical_hash"] for row in batch_seven}.intersection(
        row["canonical_hash"] for row in batch_eight
    )


def test_volatility_regime_rule_is_causal_and_fully_covered() -> None:
    index = pd.date_range("2000-01-03", periods=300, freq="B")
    close = pd.Series(np.linspace(100.0, 200.0, len(index)), index=index)
    ledger = pd.DataFrame(
        {
            "tr_close": close,
            "tr_open": close,
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "volume": 1000.0,
            "long_return": close.pct_change().fillna(0.0),
            "short_return": -close.pct_change().fillna(0.0),
        },
        index=index,
    )
    candidate = registry.generate_candidates(8, count=1)[0]
    result = candidate_decisions(
        candidate,
        PreparedMarketData(
            ledger=ledger,
            series={},
            available_dataset_ids=frozenset({"DS001", "DS002"}),
            rejected_datasets={},
            receipts=(),
            split="train",
        ),
    )
    expected_warmup = max(
        int(candidate["parameters"]["rsi_trend_window"]),
        int(candidate["parameters"]["reversal_trend_window"]),
        int(candidate["parameters"]["volatility_window"]),
        int(candidate["parameters"]["regime_trend_window"]),
    )
    assert result.first_evaluable_date == index[expected_warmup].date().isoformat()
    assert result.missing_fraction == 0.0


def test_batch_nine_adds_unique_overnight_tug_rules() -> None:
    batch_eight = registry.generate_candidates(8, count=96)
    batch_nine = registry.generate_candidates(9, count=96)
    assert len(batch_nine) == 96
    assert {row["family"] for row in batch_nine} == {"overnight_tug_reversal_vote"}
    assert all(row["position_values"] == [-1, 1] for row in batch_nine)
    assert all(row["cash_allowed"] is False for row in batch_nine)
    assert not {row["canonical_hash"] for row in batch_eight}.intersection(
        row["canonical_hash"] for row in batch_nine
    )


def test_overnight_tug_rule_is_causal_and_fully_covered() -> None:
    index = pd.date_range("2000-01-03", periods=300, freq="B")
    close = pd.Series(np.linspace(100.0, 200.0, len(index)), index=index)
    adjusted_open = close.shift(1).fillna(close.iloc[0]) * 1.001
    ledger = pd.DataFrame(
        {
            "tr_close": close,
            "tr_open": adjusted_open,
            "open": adjusted_open,
            "high": pd.concat([close, adjusted_open], axis=1).max(axis=1) + 1.0,
            "low": pd.concat([close, adjusted_open], axis=1).min(axis=1) - 1.0,
            "volume": 1000.0,
            "long_return": close.pct_change().fillna(0.0),
            "short_return": -close.pct_change().fillna(0.0),
        },
        index=index,
    )
    candidate = registry.generate_candidates(9, count=1)[0]
    result = candidate_decisions(
        candidate,
        PreparedMarketData(
            ledger=ledger,
            series={},
            available_dataset_ids=frozenset({"DS001", "DS002"}),
            rejected_datasets={},
            receipts=(),
            split="train",
        ),
    )
    expected_warmup = max(
        int(candidate["parameters"]["rsi_trend_window"]),
        int(candidate["parameters"]["reversal_trend_window"]),
        int(candidate["parameters"]["tug_lookback"]),
    )
    assert result.first_evaluable_date == index[expected_warmup].date().isoformat()
    assert result.missing_fraction == 0.0


def test_batch_ten_adds_unique_strong_trend_overrides() -> None:
    batch_nine = registry.generate_candidates(9, count=96)
    batch_ten = registry.generate_candidates(10, count=96)
    assert len(batch_ten) == 96
    assert {row["family"] for row in batch_ten} == {"strong_trend_override_reversal"}
    assert all(row["position_values"] == [-1, 1] for row in batch_ten)
    assert all(row["cash_allowed"] is False for row in batch_ten)
    assert not {row["canonical_hash"] for row in batch_nine}.intersection(
        row["canonical_hash"] for row in batch_ten
    )


def test_batch_thirteen_refines_best_dual_reversal_without_repeats() -> None:
    prior = (
        *registry.generate_candidates(5, count=96),
        *registry.generate_candidates(6, count=96),
    )
    batch_thirteen = registry.generate_candidates(13, count=96)
    batch_fourteen = registry.generate_candidates(14, count=96)
    assert len(batch_thirteen) == 96
    assert len(batch_fourteen) == 96
    assert {row["family"] for row in batch_thirteen} == {"dual_reversal_trend_vote"}
    prior_hashes = {row["canonical_hash"] for row in prior}
    assert not prior_hashes.intersection(row["canonical_hash"] for row in batch_thirteen)
    assert not {row["canonical_hash"] for row in batch_thirteen}.intersection(
        row["canonical_hash"] for row in batch_fourteen
    )
    assert all(row["position_values"] == [-1, 1] for row in batch_thirteen)
    assert all(row["cash_allowed"] is False for row in batch_thirteen)


def test_batch_sixteen_adds_unique_asymmetric_trend_overrides() -> None:
    batch_sixteen = registry.generate_candidates(16, count=96)
    batch_seventeen = registry.generate_candidates(17, count=96)
    batch_eighteen = registry.generate_candidates(18, count=96)
    assert len(batch_sixteen) == 96
    assert len(batch_seventeen) == 96
    assert len(batch_eighteen) == 96
    assert {row["family"] for row in batch_sixteen} == {"asymmetric_trend_override_reversal"}
    assert not {row["canonical_hash"] for row in batch_sixteen}.intersection(
        row["canonical_hash"] for row in batch_seventeen
    )
    assert not {row["canonical_hash"] for row in batch_sixteen}.intersection(
        row["canonical_hash"] for row in batch_eighteen
    )
    assert not {row["canonical_hash"] for row in batch_seventeen}.intersection(
        row["canonical_hash"] for row in batch_eighteen
    )
    assert all(row["position_values"] == [-1, 1] for row in batch_sixteen)
    assert all(row["cash_allowed"] is False for row in batch_sixteen)
    assert all(row["leverage_allowed"] is False for row in batch_sixteen)


def test_batch_eighteen_searches_the_stable_override_neighborhood() -> None:
    candidates = registry.generate_candidates(18, count=96)
    parameter_rows = [row["parameters"] for row in candidates]
    assert any(
        row["positive_override_window"] == 120
        and row["positive_override_threshold_pct"] == 3.0
        and row["negative_override_window"] == 120
        and row["negative_override_threshold_pct"] == 3.0
        for row in parameter_rows
    )
    assert {row["negative_override_window"] for row in parameter_rows} == {90, 120, 150}


def test_asymmetric_trend_override_is_causal_and_fully_covered() -> None:
    index = pd.date_range("2000-01-03", periods=320, freq="B")
    close = pd.Series(np.linspace(100.0, 220.0, len(index)), index=index)
    ledger = pd.DataFrame(
        {
            "tr_close": close,
            "tr_open": close,
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "volume": 1000.0,
            "long_return": close.pct_change().fillna(0.0),
            "short_return": -close.pct_change().fillna(0.0),
        },
        index=index,
    )
    candidate = registry.generate_candidates(16, count=96)[0]
    data = PreparedMarketData(
        ledger=ledger,
        series={},
        available_dataset_ids=frozenset({"DS001", "DS002"}),
        rejected_datasets={},
        receipts=(),
        split="train",
    )
    result = candidate_decisions(candidate, data)
    expected_warmup = max(
        int(candidate["parameters"]["rsi_trend_window"]),
        int(candidate["parameters"]["reversal_trend_window"]),
        int(candidate["parameters"]["positive_override_window"]),
        int(candidate["parameters"]["negative_override_window"]),
    )
    assert result.first_evaluable_date == index[expected_warmup].date().isoformat()
    assert result.missing_fraction == 0.0
    assert set(result.decisions.unique()) <= {-1, 1}

    changed = ledger.copy()
    changed.loc[index[-1], "tr_close"] *= 1.25
    changed.loc[index[-1], "high"] = changed.loc[index[-1], "tr_close"] + 1.0
    future_changed = candidate_decisions(
        candidate,
        PreparedMarketData(
            ledger=changed,
            series={},
            available_dataset_ids=frozenset({"DS001", "DS002"}),
            rejected_datasets={},
            receipts=(),
            split="train",
        ),
    )
    pd.testing.assert_series_equal(result.decisions.iloc[:-1], future_changed.decisions.iloc[:-1])


def test_batch_nineteen_adds_unique_drawdown_recovery_overrides() -> None:
    batch_eighteen = registry.generate_candidates(18, count=96)
    batch_nineteen = registry.generate_candidates(19, count=96)
    batch_twenty = registry.generate_candidates(20, count=96)

    assert len(batch_nineteen) == 96
    assert len({row["canonical_hash"] for row in batch_nineteen}) == 96
    assert {row["family"] for row in batch_nineteen} == {"drawdown_recovery_override_reversal"}
    assert not {row["canonical_hash"] for row in batch_eighteen}.intersection(
        row["canonical_hash"] for row in batch_nineteen
    )
    assert not {row["canonical_hash"] for row in batch_nineteen}.intersection(
        row["canonical_hash"] for row in batch_twenty
    )
    assert all(row["position_values"] == [-1, 1] for row in batch_nineteen)
    assert all(row["cash_allowed"] is False for row in batch_nineteen)
    assert all(row["leverage_allowed"] is False for row in batch_nineteen)


def test_drawdown_recovery_override_is_causal_and_requires_prior_drawdown() -> None:
    index = pd.date_range("2000-01-03", periods=520, freq="B")
    first = np.linspace(100.0, 150.0, 260)
    selloff = np.linspace(150.0, 105.0, 80)
    recovery = np.linspace(105.0, 165.0, 180)
    close = pd.Series(np.concatenate((first, selloff, recovery)), index=index)
    ledger = pd.DataFrame(
        {
            "tr_close": close,
            "tr_open": close,
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "volume": 1000.0,
            "long_return": close.pct_change().fillna(0.0),
            "short_return": -close.pct_change().fillna(0.0),
        },
        index=index,
    )
    candidate = registry.generate_candidates(19, count=96)[0]
    data = PreparedMarketData(
        ledger=ledger,
        series={},
        available_dataset_ids=frozenset({"DS001", "DS002"}),
        rejected_datasets={},
        receipts=(),
        split="train",
    )
    result = candidate_decisions(candidate, data)
    assert set(result.decisions.unique()) <= {-1, 1}
    assert result.missing_fraction == 0.0

    parameters = candidate["parameters"]
    peak = close.rolling(
        int(parameters["drawdown_lookback"]),
        min_periods=int(parameters["drawdown_lookback"]),
    ).max()
    drawdown = close / peak - 1.0
    recent_deep_drawdown = drawdown.rolling(
        int(parameters["recovery_memory_window"]),
        min_periods=1,
    ).min() <= -(float(parameters["drawdown_trigger_pct"]) / 100.0)
    recovery_return = close / close.shift(int(parameters["recovery_window"])) - 1.0
    override = recent_deep_drawdown & (
        recovery_return > float(parameters["recovery_threshold_pct"]) / 100.0
    )
    aligned_override = override.loc[result.decisions.index]
    assert aligned_override.any()
    assert (result.decisions.loc[aligned_override] == 1).all()

    changed = ledger.copy()
    changed.loc[index[-1], "tr_close"] *= 1.25
    changed.loc[index[-1], "high"] = changed.loc[index[-1], "tr_close"] + 1.0
    future_changed = candidate_decisions(
        candidate,
        PreparedMarketData(
            ledger=changed,
            series={},
            available_dataset_ids=frozenset({"DS001", "DS002"}),
            rejected_datasets={},
            receipts=(),
            split="train",
        ),
    )
    pd.testing.assert_series_equal(result.decisions.iloc[:-1], future_changed.decisions.iloc[:-1])


def test_batch_twenty_one_adds_unique_quiet_bull_recovery_overrides() -> None:
    batch_twenty = registry.generate_candidates(20, count=96)
    batch_twenty_one = registry.generate_candidates(21, count=96)
    batch_twenty_two = registry.generate_candidates(22, count=96)

    assert len(batch_twenty_one) == 96
    assert len({row["canonical_hash"] for row in batch_twenty_one}) == 96
    assert {row["family"] for row in batch_twenty_one} == {"quiet_bull_recovery_override_reversal"}
    assert not {row["canonical_hash"] for row in batch_twenty}.intersection(
        row["canonical_hash"] for row in batch_twenty_one
    )
    assert not {row["canonical_hash"] for row in batch_twenty_one}.intersection(
        row["canonical_hash"] for row in batch_twenty_two
    )
    assert all(row["position_values"] == [-1, 1] for row in batch_twenty_one)
    assert all(row["cash_allowed"] is False for row in batch_twenty_one)
    assert all(row["leverage_allowed"] is False for row in batch_twenty_one)


def test_quiet_bull_recovery_override_is_causal_and_fully_invested() -> None:
    index = pd.date_range("2000-01-03", periods=620, freq="B")
    close = pd.Series(np.linspace(100.0, 220.0, len(index)), index=index)
    ledger = pd.DataFrame(
        {
            "tr_close": close,
            "tr_open": close,
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "volume": 1000.0,
            "long_return": close.pct_change().fillna(0.0),
            "short_return": -close.pct_change().fillna(0.0),
        },
        index=index,
    )
    candidate = registry.generate_candidates(21, count=96)[0]
    data = PreparedMarketData(
        ledger=ledger,
        series={},
        available_dataset_ids=frozenset({"DS001", "DS002"}),
        rejected_datasets={},
        receipts=(),
        split="train",
    )
    result = candidate_decisions(candidate, data)
    assert set(result.decisions.unique()) <= {-1, 1}
    assert result.missing_fraction == 0.0
    assert (result.decisions.iloc[-100:] == 1).all()

    changed = ledger.copy()
    changed.loc[index[-1], "tr_close"] *= 0.75
    changed.loc[index[-1], "low"] = changed.loc[index[-1], "tr_close"] - 1.0
    future_changed = candidate_decisions(
        candidate,
        PreparedMarketData(
            ledger=changed,
            series={},
            available_dataset_ids=frozenset({"DS001", "DS002"}),
            rejected_datasets={},
            receipts=(),
            split="train",
        ),
    )
    pd.testing.assert_series_equal(result.decisions.iloc[:-1], future_changed.decisions.iloc[:-1])


def test_batch_twenty_two_adds_unique_recovery_trend_breakout_majorities() -> None:
    batch_twenty_one = registry.generate_candidates(21, count=96)
    batch_twenty_two = registry.generate_candidates(22, count=96)
    batch_twenty_three = registry.generate_candidates(23, count=96)

    assert len(batch_twenty_two) == 96
    assert len({row["canonical_hash"] for row in batch_twenty_two}) == 96
    assert {row["family"] for row in batch_twenty_two} == {"recovery_trend_breakout_majority"}
    assert not {row["canonical_hash"] for row in batch_twenty_one}.intersection(
        row["canonical_hash"] for row in batch_twenty_two
    )
    assert not {row["canonical_hash"] for row in batch_twenty_two}.intersection(
        row["canonical_hash"] for row in batch_twenty_three
    )
    assert all(row["position_values"] == [-1, 1] for row in batch_twenty_two)
    assert all(row["cash_allowed"] is False for row in batch_twenty_two)
    assert all(row["leverage_allowed"] is False for row in batch_twenty_two)


def test_recovery_trend_breakout_majority_is_causal_and_fully_invested() -> None:
    index = pd.date_range("2000-01-03", periods=700, freq="B")
    cycle = np.sin(np.arange(len(index)) / 13.0) * 6.0
    close = pd.Series(100.0 + np.arange(len(index)) * 0.08 + cycle, index=index)
    ledger = pd.DataFrame(
        {
            "tr_close": close,
            "tr_open": close,
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "volume": 1000.0,
            "long_return": close.pct_change().fillna(0.0),
            "short_return": -close.pct_change().fillna(0.0),
        },
        index=index,
    )
    candidate = registry.generate_candidates(22, count=96)[0]
    data = PreparedMarketData(
        ledger=ledger,
        series={},
        available_dataset_ids=frozenset({"DS001", "DS002"}),
        rejected_datasets={},
        receipts=(),
        split="train",
    )
    result = candidate_decisions(candidate, data)
    assert set(result.decisions.unique()) <= {-1, 1}
    assert result.missing_fraction == 0.0
    assert result.first_evaluable_date is not None

    changed = ledger.copy()
    changed.loc[index[-1], "tr_close"] *= 0.70
    changed.loc[index[-1], "low"] = changed.loc[index[-1], "tr_close"] - 1.0
    future_changed = candidate_decisions(
        candidate,
        PreparedMarketData(
            ledger=changed,
            series={},
            available_dataset_ids=frozenset({"DS001", "DS002"}),
            rejected_datasets={},
            receipts=(),
            split="train",
        ),
    )
    pd.testing.assert_series_equal(result.decisions.iloc[:-1], future_changed.decisions.iloc[:-1])


def test_batch_twenty_three_adds_unique_high_vol_crash_recovery_rules() -> None:
    batch_twenty_two = registry.generate_candidates(22, count=96)
    batch_twenty_three = registry.generate_candidates(23, count=96)
    batch_twenty_four = registry.generate_candidates(24, count=96)

    assert len(batch_twenty_three) == 96
    assert len({row["canonical_hash"] for row in batch_twenty_three}) == 96
    assert {row["family"] for row in batch_twenty_three} == {"high_vol_crash_recovery_reversal"}
    assert not {row["canonical_hash"] for row in batch_twenty_two}.intersection(
        row["canonical_hash"] for row in batch_twenty_three
    )
    assert not {row["canonical_hash"] for row in batch_twenty_three}.intersection(
        row["canonical_hash"] for row in batch_twenty_four
    )
    assert all(row["position_values"] == [-1, 1] for row in batch_twenty_three)
    assert all(row["cash_allowed"] is False for row in batch_twenty_three)
    assert all(row["leverage_allowed"] is False for row in batch_twenty_three)


def test_high_vol_crash_recovery_rule_is_causal_and_fully_invested() -> None:
    index = pd.date_range("2000-01-03", periods=720, freq="B")
    rising = np.linspace(100.0, 170.0, 400)
    crash = np.linspace(170.0, 90.0, 100) * (1.0 + 0.04 * np.sin(np.arange(100) * 2.1))
    recovery = np.linspace(90.0, 180.0, 220)
    close = pd.Series(np.concatenate((rising, crash, recovery)), index=index)
    ledger = pd.DataFrame(
        {
            "tr_close": close,
            "tr_open": close,
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "volume": 1000.0,
            "long_return": close.pct_change().fillna(0.0),
            "short_return": -close.pct_change().fillna(0.0),
        },
        index=index,
    )
    candidate = registry.generate_candidates(23, count=96)[0]
    data = PreparedMarketData(
        ledger=ledger,
        series={},
        available_dataset_ids=frozenset({"DS001", "DS002"}),
        rejected_datasets={},
        receipts=(),
        split="train",
    )
    result = candidate_decisions(candidate, data)
    assert set(result.decisions.unique()) <= {-1, 1}
    assert result.missing_fraction == 0.0
    assert (result.decisions.iloc[430:480] == -1).any()
    assert (result.decisions.iloc[-100:] == 1).any()

    changed = ledger.copy()
    changed.loc[index[-1], "tr_close"] *= 0.70
    changed.loc[index[-1], "low"] = changed.loc[index[-1], "tr_close"] - 1.0
    future_changed = candidate_decisions(
        candidate,
        PreparedMarketData(
            ledger=changed,
            series={},
            available_dataset_ids=frozenset({"DS001", "DS002"}),
            rejected_datasets={},
            receipts=(),
            split="train",
        ),
    )
    pd.testing.assert_series_equal(result.decisions.iloc[:-1], future_changed.decisions.iloc[:-1])


def test_batch_twenty_four_adds_unique_adaptive_recovery_edge_rules() -> None:
    batch_twenty_three = registry.generate_candidates(23, count=96)
    batch_twenty_four = registry.generate_candidates(24, count=96)
    batch_twenty_five = registry.generate_candidates(25, count=96)

    assert len(batch_twenty_four) == 96
    assert len({row["canonical_hash"] for row in batch_twenty_four}) == 96
    assert {row["family"] for row in batch_twenty_four} == {"adaptive_recovery_edge_switch"}
    assert not {row["canonical_hash"] for row in batch_twenty_three}.intersection(
        row["canonical_hash"] for row in batch_twenty_four
    )
    assert not {row["canonical_hash"] for row in batch_twenty_four}.intersection(
        row["canonical_hash"] for row in batch_twenty_five
    )
    assert all(row["position_values"] == [-1, 1] for row in batch_twenty_four)
    assert all(row["cash_allowed"] is False for row in batch_twenty_four)
    assert all(row["leverage_allowed"] is False for row in batch_twenty_four)


def test_adaptive_recovery_edge_rule_is_causal_and_fully_invested() -> None:
    index = pd.date_range("2000-01-03", periods=900, freq="B")
    trend = np.linspace(100.0, 180.0, 300)
    chop = 180.0 + 12.0 * np.sin(np.arange(300) * 0.55)
    decline = np.linspace(180.0, 95.0, 150)
    recovery = np.linspace(95.0, 170.0, 150)
    close = pd.Series(np.concatenate((trend, chop, decline, recovery)), index=index)
    ledger = pd.DataFrame(
        {
            "tr_close": close,
            "tr_open": close,
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "volume": 1000.0,
            "long_return": close.shift(-1).div(close).sub(1.0),
            "short_return": -close.shift(-1).div(close).sub(1.0),
        },
        index=index,
    )
    candidate = registry.generate_candidates(24, count=96)[0]
    data = PreparedMarketData(
        ledger=ledger,
        series={},
        available_dataset_ids=frozenset({"DS001", "DS002"}),
        rejected_datasets={},
        receipts=(),
        split="train",
    )
    result = candidate_decisions(candidate, data)
    assert set(result.decisions.unique()) <= {-1, 1}
    assert result.missing_fraction == 0.0
    assert result.first_evaluable_date is not None

    changed = ledger.copy()
    changed.loc[index[-1], "tr_close"] *= 0.70
    changed.loc[index[-1], "low"] = changed.loc[index[-1], "tr_close"] - 1.0
    future_changed = candidate_decisions(
        candidate,
        PreparedMarketData(
            ledger=changed,
            series={},
            available_dataset_ids=frozenset({"DS001", "DS002"}),
            rejected_datasets={},
            receipts=(),
            split="train",
        ),
    )
    pd.testing.assert_series_equal(result.decisions.iloc[:-1], future_changed.decisions.iloc[:-1])


def test_batch_twenty_five_adds_unique_recovery_overnight_tug_rules() -> None:
    batch_twenty_four = registry.generate_candidates(24, count=96)
    batch_twenty_five = registry.generate_candidates(25, count=96)
    batch_twenty_six = registry.generate_candidates(26, count=96)

    assert len(batch_twenty_five) == 96
    assert len({row["canonical_hash"] for row in batch_twenty_five}) == 96
    assert {row["family"] for row in batch_twenty_five} == {"recovery_overnight_tug_vote"}
    assert not {row["canonical_hash"] for row in batch_twenty_four}.intersection(
        row["canonical_hash"] for row in batch_twenty_five
    )
    assert not {row["canonical_hash"] for row in batch_twenty_five}.intersection(
        row["canonical_hash"] for row in batch_twenty_six
    )
    assert all(row["position_values"] == [-1, 1] for row in batch_twenty_five)
    assert all(row["cash_allowed"] is False for row in batch_twenty_five)
    assert all(row["leverage_allowed"] is False for row in batch_twenty_five)


def test_batch_twenty_six_adds_unique_recovery_turn_month_rules() -> None:
    batch_twenty_five = registry.generate_candidates(25, count=96)
    batch_twenty_six = registry.generate_candidates(26, count=96)
    batch_twenty_seven = registry.generate_candidates(27, count=96)

    assert len(batch_twenty_six) == 96
    assert len({row["canonical_hash"] for row in batch_twenty_six}) == 96
    assert {row["family"] for row in batch_twenty_six} == {"recovery_turn_month_vote"}
    assert not {row["canonical_hash"] for row in batch_twenty_five}.intersection(
        row["canonical_hash"] for row in batch_twenty_six
    )
    assert not {row["canonical_hash"] for row in batch_twenty_six}.intersection(
        row["canonical_hash"] for row in batch_twenty_seven
    )
    assert all(row["position_values"] == [-1, 1] for row in batch_twenty_six)
    assert all(row["cash_allowed"] is False for row in batch_twenty_six)
    assert all(row["leverage_allowed"] is False for row in batch_twenty_six)
    assert all(
        {"SRC0048", "SRC0175", "SRC0176"}.issubset(row["research_source_ids"])
        for row in batch_twenty_six
    )


def test_recovery_turn_month_rule_is_causal_and_fully_invested() -> None:
    index = pd.date_range("2000-01-03", periods=800, freq="B")
    close = pd.Series(
        120.0 + np.linspace(0.0, 45.0, len(index)) + 8.0 * np.sin(np.arange(len(index)) * 0.17),
        index=index,
    )
    open_price = close.shift(1).fillna(close.iloc[0])
    ledger = pd.DataFrame(
        {
            "tr_close": close,
            "tr_open": open_price,
            "open": open_price,
            "high": np.maximum(open_price, close) + 1.0,
            "low": np.minimum(open_price, close) - 1.0,
            "volume": 1000.0,
            "long_return": open_price.shift(-1).div(open_price).sub(1.0),
            "short_return": -open_price.shift(-1).div(open_price).sub(1.0),
        },
        index=index,
    )
    candidate = registry.generate_candidates(26, count=96)[0]
    data = PreparedMarketData(
        ledger=ledger,
        series={},
        available_dataset_ids=frozenset({"DS001", "DS002"}),
        rejected_datasets={},
        receipts=(),
        split="train",
    )
    result = candidate_decisions(candidate, data)
    assert set(result.decisions.unique()) <= {-1, 1}
    assert result.missing_fraction == 0.0
    assert result.first_evaluable_date is not None

    changed = ledger.copy()
    changed.loc[index[-1], "tr_close"] *= 0.70
    future_changed = candidate_decisions(
        candidate,
        PreparedMarketData(
            ledger=changed,
            series={},
            available_dataset_ids=frozenset({"DS001", "DS002"}),
            rejected_datasets={},
            receipts=(),
            split="train",
        ),
    )
    pd.testing.assert_series_equal(result.decisions.iloc[:-1], future_changed.decisions.iloc[:-1])


def test_batch_twenty_seven_adds_unique_recovery_ibs_rules() -> None:
    batch_twenty_six = registry.generate_candidates(26, count=96)
    batch_twenty_seven = registry.generate_candidates(27, count=96)
    batch_twenty_eight = registry.generate_candidates(28, count=96)

    assert len(batch_twenty_seven) == 96
    assert len({row["canonical_hash"] for row in batch_twenty_seven}) == 96
    assert {row["family"] for row in batch_twenty_seven} == {"recovery_internal_bar_strength_vote"}
    assert not {row["canonical_hash"] for row in batch_twenty_six}.intersection(
        row["canonical_hash"] for row in batch_twenty_seven
    )
    assert not {row["canonical_hash"] for row in batch_twenty_seven}.intersection(
        row["canonical_hash"] for row in batch_twenty_eight
    )
    assert all(row["position_values"] == [-1, 1] for row in batch_twenty_seven)
    assert all(row["cash_allowed"] is False for row in batch_twenty_seven)
    assert all(row["leverage_allowed"] is False for row in batch_twenty_seven)


def test_recovery_internal_bar_strength_rule_is_causal_and_fully_invested() -> None:
    index = pd.date_range("2000-01-03", periods=800, freq="B")
    close = pd.Series(
        120.0 + np.linspace(0.0, 45.0, len(index)) + 8.0 * np.sin(np.arange(len(index)) * 0.17),
        index=index,
    )
    open_price = close.shift(1).fillna(close.iloc[0])
    high = np.maximum(open_price, close) + 1.0
    low = np.minimum(open_price, close) - 1.0
    ledger = pd.DataFrame(
        {
            "tr_close": close,
            "tr_open": open_price,
            "open": open_price,
            "high": high,
            "low": low,
            "volume": 1000.0,
            "long_return": open_price.shift(-1).div(open_price).sub(1.0),
            "short_return": -open_price.shift(-1).div(open_price).sub(1.0),
        },
        index=index,
    )
    candidate = registry.generate_candidates(27, count=96)[0]
    data = PreparedMarketData(
        ledger=ledger,
        series={},
        available_dataset_ids=frozenset({"DS001", "DS002"}),
        rejected_datasets={},
        receipts=(),
        split="train",
    )
    result = candidate_decisions(candidate, data)
    assert set(result.decisions.unique()) <= {-1, 1}
    assert result.missing_fraction == 0.0
    assert result.first_evaluable_date is not None

    changed = ledger.copy()
    changed.loc[index[-1], "tr_close"] *= 0.70
    changed.loc[index[-1], "high"] *= 1.20
    future_changed = candidate_decisions(
        candidate,
        PreparedMarketData(
            ledger=changed,
            series={},
            available_dataset_ids=frozenset({"DS001", "DS002"}),
            rejected_datasets={},
            receipts=(),
            split="train",
        ),
    )
    pd.testing.assert_series_equal(result.decisions.iloc[:-1], future_changed.decisions.iloc[:-1])


def test_batch_twenty_eight_adds_unique_recovery_multi_horizon_rules() -> None:
    batch_twenty_seven = registry.generate_candidates(27, count=96)
    batch_twenty_eight = registry.generate_candidates(28, count=96)
    batch_twenty_nine = registry.generate_candidates(29, count=96)

    assert len(batch_twenty_eight) == 96
    assert len({row["canonical_hash"] for row in batch_twenty_eight}) == 96
    assert {row["family"] for row in batch_twenty_eight} == {"recovery_multi_horizon_reversal_vote"}
    assert not {row["canonical_hash"] for row in batch_twenty_seven}.intersection(
        row["canonical_hash"] for row in batch_twenty_eight
    )
    assert not {row["canonical_hash"] for row in batch_twenty_eight}.intersection(
        row["canonical_hash"] for row in batch_twenty_nine
    )
    assert all(row["position_values"] == [-1, 1] for row in batch_twenty_eight)
    assert all(row["cash_allowed"] is False for row in batch_twenty_eight)
    assert all(row["leverage_allowed"] is False for row in batch_twenty_eight)


def test_recovery_multi_horizon_rule_is_causal_and_fully_invested() -> None:
    index = pd.date_range("2000-01-03", periods=800, freq="B")
    close = pd.Series(
        120.0 + np.linspace(0.0, 45.0, len(index)) + 8.0 * np.sin(np.arange(len(index)) * 0.17),
        index=index,
    )
    open_price = close.shift(1).fillna(close.iloc[0])
    ledger = pd.DataFrame(
        {
            "tr_close": close,
            "tr_open": open_price,
            "open": open_price,
            "high": np.maximum(open_price, close) + 1.0,
            "low": np.minimum(open_price, close) - 1.0,
            "volume": 1000.0,
            "long_return": open_price.shift(-1).div(open_price).sub(1.0),
            "short_return": -open_price.shift(-1).div(open_price).sub(1.0),
        },
        index=index,
    )
    candidate = registry.generate_candidates(28, count=96)[0]
    data = PreparedMarketData(
        ledger=ledger,
        series={},
        available_dataset_ids=frozenset({"DS001", "DS002"}),
        rejected_datasets={},
        receipts=(),
        split="train",
    )
    result = candidate_decisions(candidate, data)
    assert set(result.decisions.unique()) <= {-1, 1}
    assert result.missing_fraction == 0.0
    assert result.first_evaluable_date is not None

    changed = ledger.copy()
    changed.loc[index[-1], "tr_close"] *= 0.70
    future_changed = candidate_decisions(
        candidate,
        PreparedMarketData(
            ledger=changed,
            series={},
            available_dataset_ids=frozenset({"DS001", "DS002"}),
            rejected_datasets={},
            receipts=(),
            split="train",
        ),
    )
    pd.testing.assert_series_equal(result.decisions.iloc[:-1], future_changed.decisions.iloc[:-1])


def test_batch_twenty_nine_adds_unique_recovery_volume_rules() -> None:
    batch_twenty_eight = registry.generate_candidates(28, count=96)
    batch_twenty_nine = registry.generate_candidates(29, count=96)
    batch_thirty = registry.generate_candidates(30, count=96)

    assert len(batch_twenty_nine) == 96
    assert len({row["canonical_hash"] for row in batch_twenty_nine}) == 96
    assert {row["family"] for row in batch_twenty_nine} == {"recovery_volume_gated_reversal_vote"}
    assert not {row["canonical_hash"] for row in batch_twenty_eight}.intersection(
        row["canonical_hash"] for row in batch_twenty_nine
    )
    assert not {row["canonical_hash"] for row in batch_twenty_nine}.intersection(
        row["canonical_hash"] for row in batch_thirty
    )
    assert all(row["position_values"] == [-1, 1] for row in batch_twenty_nine)
    assert all(row["cash_allowed"] is False for row in batch_twenty_nine)
    assert all(row["leverage_allowed"] is False for row in batch_twenty_nine)
    assert all(
        {"SRC0047", "SRC0050", "SRC0133"}.issubset(row["research_source_ids"])
        for row in batch_twenty_nine
    )


def test_recovery_volume_gated_rule_is_causal_and_fully_invested() -> None:
    index = pd.date_range("2000-01-03", periods=800, freq="B")
    close = pd.Series(
        120.0 + np.linspace(0.0, 45.0, len(index)) + 8.0 * np.sin(np.arange(len(index)) * 0.17),
        index=index,
    )
    open_price = close.shift(1).fillna(close.iloc[0])
    volume = pd.Series(
        1000.0 + 500.0 * (1.0 + np.sin(np.arange(len(index)) * 0.13)),
        index=index,
    )
    ledger = pd.DataFrame(
        {
            "tr_close": close,
            "tr_open": open_price,
            "open": open_price,
            "high": np.maximum(open_price, close) + 1.0,
            "low": np.minimum(open_price, close) - 1.0,
            "volume": volume,
            "long_return": open_price.shift(-1).div(open_price).sub(1.0),
            "short_return": -open_price.shift(-1).div(open_price).sub(1.0),
        },
        index=index,
    )
    candidate = registry.generate_candidates(29, count=96)[0]
    data = PreparedMarketData(
        ledger=ledger,
        series={},
        available_dataset_ids=frozenset({"DS001", "DS002"}),
        rejected_datasets={},
        receipts=(),
        split="train",
    )
    result = candidate_decisions(candidate, data)
    assert set(result.decisions.unique()) <= {-1, 1}
    assert result.missing_fraction == 0.0
    assert result.first_evaluable_date is not None

    changed = ledger.copy()
    changed.loc[index[-1], "tr_close"] *= 0.70
    changed.loc[index[-1], "volume"] *= 10.0
    future_changed = candidate_decisions(
        candidate,
        PreparedMarketData(
            ledger=changed,
            series={},
            available_dataset_ids=frozenset({"DS001", "DS002"}),
            rejected_datasets={},
            receipts=(),
            split="train",
        ),
    )
    pd.testing.assert_series_equal(result.decisions.iloc[:-1], future_changed.decisions.iloc[:-1])


def test_batch_thirty_adds_unique_recovery_calendar_volume_rules() -> None:
    batch_twenty_nine = registry.generate_candidates(29, count=96)
    batch_thirty = registry.generate_candidates(30, count=96)
    batch_thirty_one = registry.generate_candidates(31, count=96)

    assert len(batch_thirty) == 96
    assert len({row["canonical_hash"] for row in batch_thirty}) == 96
    assert {row["family"] for row in batch_thirty} == {"recovery_calendar_volume_reversal_vote"}
    assert not {row["canonical_hash"] for row in batch_twenty_nine}.intersection(
        row["canonical_hash"] for row in batch_thirty
    )
    assert not {row["canonical_hash"] for row in batch_thirty}.intersection(
        row["canonical_hash"] for row in batch_thirty_one
    )
    assert all(row["position_values"] == [-1, 1] for row in batch_thirty)
    assert all(row["cash_allowed"] is False for row in batch_thirty)
    assert all(row["leverage_allowed"] is False for row in batch_thirty)
    assert all(
        {"SRC0047", "SRC0048", "SRC0050", "SRC0133", "SRC0175", "SRC0176"}.issubset(
            row["research_source_ids"]
        )
        for row in batch_thirty
    )


def test_recovery_calendar_volume_rule_is_causal_and_fully_invested() -> None:
    index = pd.date_range("2000-01-03", periods=800, freq="B")
    close = pd.Series(
        120.0 + np.linspace(0.0, 45.0, len(index)) + 8.0 * np.sin(np.arange(len(index)) * 0.17),
        index=index,
    )
    open_price = close.shift(1).fillna(close.iloc[0])
    volume = pd.Series(
        1000.0 + 500.0 * (1.0 + np.sin(np.arange(len(index)) * 0.13)),
        index=index,
    )
    ledger = pd.DataFrame(
        {
            "tr_close": close,
            "tr_open": open_price,
            "open": open_price,
            "high": np.maximum(open_price, close) + 1.0,
            "low": np.minimum(open_price, close) - 1.0,
            "volume": volume,
            "long_return": open_price.shift(-1).div(open_price).sub(1.0),
            "short_return": -open_price.shift(-1).div(open_price).sub(1.0),
        },
        index=index,
    )
    candidate = registry.generate_candidates(30, count=96)[0]
    data = PreparedMarketData(
        ledger=ledger,
        series={},
        available_dataset_ids=frozenset({"DS001", "DS002"}),
        rejected_datasets={},
        receipts=(),
        split="train",
    )
    result = candidate_decisions(candidate, data)
    assert set(result.decisions.unique()) <= {-1, 1}
    assert result.missing_fraction == 0.0
    assert result.first_evaluable_date is not None

    changed = ledger.copy()
    changed.loc[index[-1], "tr_close"] *= 0.70
    changed.loc[index[-1], "volume"] *= 10.0
    future_changed = candidate_decisions(
        candidate,
        PreparedMarketData(
            ledger=changed,
            series={},
            available_dataset_ids=frozenset({"DS001", "DS002"}),
            rejected_datasets={},
            receipts=(),
            split="train",
        ),
    )
    pd.testing.assert_series_equal(result.decisions.iloc[:-1], future_changed.decisions.iloc[:-1])


def test_batch_thirty_two_adds_unique_causal_vxo_rules() -> None:
    batch_thirty_one = registry.generate_candidates(31, count=96)
    batch_thirty_two = registry.generate_candidates(32, count=96)
    batch_thirty_three = registry.generate_candidates(33, count=96)

    assert len(batch_thirty_two) == 96
    assert len({row["canonical_hash"] for row in batch_thirty_two}) == 96
    assert {row["family"] for row in batch_thirty_two} == {"recovery_calendar_volume_vxo_vote"}
    assert all(row["required_datasets"] == ["DS001", "DS002", "DS005"] for row in batch_thirty_two)
    assert {row["parameters"]["vxo_mode"] for row in batch_thirty_two} == {
        "reversal",
        "continuation",
    }
    assert not {row["canonical_hash"] for row in batch_thirty_one}.intersection(
        row["canonical_hash"] for row in batch_thirty_two
    )
    assert not {row["canonical_hash"] for row in batch_thirty_two}.intersection(
        row["canonical_hash"] for row in batch_thirty_three
    )
    assert all(row["position_values"] == [-1, 1] for row in batch_thirty_two)
    assert all(row["cash_allowed"] is False for row in batch_thirty_two)
    assert all(row["leverage_allowed"] is False for row in batch_thirty_two)


def test_batch_thirty_three_focuses_distinct_train_only_vxo_neighborhoods() -> None:
    batch_thirty_two = registry.generate_candidates(32, count=96)
    batch_thirty_three = registry.generate_candidates(33, count=96)
    batch_thirty_four = registry.generate_candidates(34, count=96)

    assert len(batch_thirty_three) == 96
    assert len({row["canonical_hash"] for row in batch_thirty_three}) == 96
    assert not {row["canonical_hash"] for row in batch_thirty_two}.intersection(
        row["canonical_hash"] for row in batch_thirty_three
    )
    assert not {row["canonical_hash"] for row in batch_thirty_three}.intersection(
        row["canonical_hash"] for row in batch_thirty_four
    )
    assert {
        (
            row["parameters"]["vxo_mode"],
            row["parameters"]["vxo_change_lookback"],
        )
        for row in batch_thirty_three
    } == {("continuation", 1), ("reversal", 5)}
    assert {row["parameters"]["vxo_z_window"] for row in batch_thirty_three} == {10, 15, 30, 40}
    assert {row["parameters"]["vxo_z_threshold"] for row in batch_thirty_three} == {
        1.2,
        1.4,
        1.6,
        1.8,
    }
    assert all(row["parameters"]["first_sessions"] == 2 for row in batch_thirty_three)
    assert all(row["parameters"]["last_sessions"] == 1 for row in batch_thirty_three)
    assert all(row["parameters"]["calendar_weight"] == 1.5 for row in batch_thirty_three)
    assert all(row["locked_boundary"] == ">=2021-01-01 unopened" for row in batch_thirty_three)
    assert all(row["position_values"] == [-1, 1] for row in batch_thirty_three)
    assert all(row["cash_allowed"] is False for row in batch_thirty_three)
    assert all(row["leverage_allowed"] is False for row in batch_thirty_three)


def test_batch_thirty_four_refines_only_leading_vxo_continuation_regions() -> None:
    batch_thirty_three = registry.generate_candidates(33, count=96)
    batch_thirty_four = registry.generate_candidates(34, count=96)
    batch_thirty_five = registry.generate_candidates(35, count=96)

    assert len(batch_thirty_four) == 96
    assert len({row["canonical_hash"] for row in batch_thirty_four}) == 96
    assert not {row["canonical_hash"] for row in batch_thirty_three}.intersection(
        row["canonical_hash"] for row in batch_thirty_four
    )
    assert not {row["canonical_hash"] for row in batch_thirty_four}.intersection(
        row["canonical_hash"] for row in batch_thirty_five
    )
    assert {row["parameters"]["vxo_z_window"] for row in batch_thirty_four} == {
        7,
        10,
        12,
        35,
        40,
        45,
    }
    assert {row["parameters"]["vxo_z_threshold"] for row in batch_thirty_four} == {
        1.3,
        1.45,
        1.55,
        1.7,
    }
    assert {row["parameters"]["vxo_weight"] for row in batch_thirty_four} == {2.5, 3.0, 3.5, 4.0}
    assert all(row["parameters"]["vxo_change_lookback"] == 1 for row in batch_thirty_four)
    assert all(row["parameters"]["vxo_mode"] == "continuation" for row in batch_thirty_four)
    assert all(row["locked_boundary"] == ">=2021-01-01 unopened" for row in batch_thirty_four)
    assert all(row["position_values"] == [-1, 1] for row in batch_thirty_four)
    assert all(row["cash_allowed"] is False for row in batch_thirty_four)
    assert all(row["leverage_allowed"] is False for row in batch_thirty_four)


def test_batch_thirty_five_refines_recovery_and_volume_around_vxo_leader() -> None:
    batch_thirty_four = registry.generate_candidates(34, count=96)
    batch_thirty_five = registry.generate_candidates(35, count=96)
    batch_thirty_six = registry.generate_candidates(36, count=96)

    assert len(batch_thirty_five) == 96
    assert len({row["canonical_hash"] for row in batch_thirty_five}) == 96
    assert not {row["canonical_hash"] for row in batch_thirty_four}.intersection(
        row["canonical_hash"] for row in batch_thirty_five
    )
    assert not {row["canonical_hash"] for row in batch_thirty_five}.intersection(
        row["canonical_hash"] for row in batch_thirty_six
    )
    assert {row["parameters"]["recovery_memory_window"] for row in batch_thirty_five} == {
        42,
        63,
        84,
        126,
    }
    assert {row["parameters"]["recovery_threshold_pct"] for row in batch_thirty_five} == {
        1.5,
        2.0,
        2.5,
        3.0,
    }
    assert {row["parameters"]["volume_z_threshold"] for row in batch_thirty_five} == {
        1.0,
        1.5,
        2.0,
    }
    assert {row["parameters"]["volume_reversal_weight"] for row in batch_thirty_five} == {
        0.25,
        0.75,
    }
    assert all(row["parameters"]["vxo_change_lookback"] == 1 for row in batch_thirty_five)
    assert all(row["parameters"]["vxo_z_window"] == 10 for row in batch_thirty_five)
    assert all(row["parameters"]["vxo_z_threshold"] == 1.55 for row in batch_thirty_five)
    assert all(row["parameters"]["vxo_weight"] == 3.0 for row in batch_thirty_five)
    assert all(row["parameters"]["vxo_mode"] == "continuation" for row in batch_thirty_five)
    assert all(row["locked_boundary"] == ">=2021-01-01 unopened" for row in batch_thirty_five)
    assert all(row["position_values"] == [-1, 1] for row in batch_thirty_five)
    assert all(row["cash_allowed"] is False for row in batch_thirty_five)
    assert all(row["leverage_allowed"] is False for row in batch_thirty_five)


def test_batch_thirty_six_uses_distinct_asymmetric_vxo_shock_weights() -> None:
    batch_thirty_five = registry.generate_candidates(35, count=96)
    batch_thirty_six = registry.generate_candidates(36, count=96)
    batch_thirty_seven = registry.generate_candidates(37, count=96)

    assert len(batch_thirty_six) == 96
    assert len({row["canonical_hash"] for row in batch_thirty_six}) == 96
    assert not {row["canonical_hash"] for row in batch_thirty_five}.intersection(
        row["canonical_hash"] for row in batch_thirty_six
    )
    assert not {row["canonical_hash"] for row in batch_thirty_six}.intersection(
        row["canonical_hash"] for row in batch_thirty_seven
    )
    assert {row["parameters"]["recovery_memory_window"] for row in batch_thirty_six} == {
        42,
        63,
        84,
    }
    assert {row["parameters"]["recovery_threshold_pct"] for row in batch_thirty_six} == {
        1.25,
        1.5,
    }
    assert {row["parameters"]["vxo_z_threshold"] for row in batch_thirty_six} == {
        1.4,
        1.55,
    }
    assert {row["parameters"]["vxo_positive_weight"] for row in batch_thirty_six} == {
        2.5,
        3.5,
        4.5,
        5.5,
    }
    assert {row["parameters"]["vxo_negative_weight"] for row in batch_thirty_six} == {
        0.0,
        1.0,
    }
    assert all(row["parameters"]["vxo_change_lookback"] == 1 for row in batch_thirty_six)
    assert all(row["parameters"]["vxo_z_window"] == 10 for row in batch_thirty_six)
    assert all(row["parameters"]["vxo_mode"] == "continuation" for row in batch_thirty_six)
    assert all(row["locked_boundary"] == ">=2021-01-01 unopened" for row in batch_thirty_six)
    assert all(row["position_values"] == [-1, 1] for row in batch_thirty_six)
    assert all(row["cash_allowed"] is False for row in batch_thirty_six)
    assert all(row["leverage_allowed"] is False for row in batch_thirty_six)


def test_batch_thirty_seven_adds_distinct_slow_trend_vote() -> None:
    batch_thirty_six = registry.generate_candidates(36, count=96)
    batch_thirty_seven = registry.generate_candidates(37, count=96)
    batch_thirty_eight = registry.generate_candidates(38, count=96)

    assert len(batch_thirty_seven) == 96
    assert len({row["canonical_hash"] for row in batch_thirty_seven}) == 96
    assert not {row["canonical_hash"] for row in batch_thirty_six}.intersection(
        row["canonical_hash"] for row in batch_thirty_seven
    )
    assert not {row["canonical_hash"] for row in batch_thirty_seven}.intersection(
        row["canonical_hash"] for row in batch_thirty_eight
    )
    assert {row["parameters"]["slow_trend_window"] for row in batch_thirty_seven} == {
        63,
        126,
        189,
        252,
    }
    assert {row["parameters"]["slow_trend_weight"] for row in batch_thirty_seven} == {
        0.5,
        1.0,
        1.5,
        2.0,
    }
    assert {row["parameters"]["rsi_weight"] for row in batch_thirty_seven} == {
        0.5,
        1.0,
        1.5,
    }
    assert {row["parameters"]["calendar_weight"] for row in batch_thirty_seven} == {
        1.0,
        2.0,
    }
    assert all(row["parameters"]["vxo_z_window"] == 10 for row in batch_thirty_seven)
    assert all(row["parameters"]["vxo_z_threshold"] == 1.55 for row in batch_thirty_seven)
    assert all(row["parameters"]["vxo_weight"] == 3.0 for row in batch_thirty_seven)
    assert all(row["locked_boundary"] == ">=2021-01-01 unopened" for row in batch_thirty_seven)
    assert all(row["position_values"] == [-1, 1] for row in batch_thirty_seven)
    assert all(row["cash_allowed"] is False for row in batch_thirty_seven)
    assert all(row["leverage_allowed"] is False for row in batch_thirty_seven)


def test_batch_thirty_eight_interpolates_unseen_vxo_shock_weights() -> None:
    batch_thirty_seven = registry.generate_candidates(37, count=96)
    batch_thirty_eight = registry.generate_candidates(38, count=96)
    batch_thirty_nine = registry.generate_candidates(39, count=96)

    assert len(batch_thirty_eight) == 96
    assert len({row["canonical_hash"] for row in batch_thirty_eight}) == 96
    assert not {row["canonical_hash"] for row in batch_thirty_seven}.intersection(
        row["canonical_hash"] for row in batch_thirty_eight
    )
    assert not {row["canonical_hash"] for row in batch_thirty_eight}.intersection(
        row["canonical_hash"] for row in batch_thirty_nine
    )
    assert {row["parameters"]["recovery_threshold_pct"] for row in batch_thirty_eight} == {
        1.25,
        1.5,
    }
    assert {row["parameters"]["vxo_z_window"] for row in batch_thirty_eight} == {8, 10}
    assert {row["parameters"]["vxo_z_threshold"] for row in batch_thirty_eight} == {
        1.4,
        1.55,
    }
    assert {row["parameters"]["vxo_positive_weight"] for row in batch_thirty_eight} == {
        2.5,
        3.5,
    }
    assert {row["parameters"]["vxo_negative_weight"] for row in batch_thirty_eight} == {
        1.25,
        1.5,
        1.75,
        2.0,
        2.25,
        2.5,
    }
    assert all(row["parameters"]["recovery_memory_window"] == 63 for row in batch_thirty_eight)
    assert all(row["parameters"]["vxo_change_lookback"] == 1 for row in batch_thirty_eight)
    assert all(row["parameters"]["vxo_mode"] == "continuation" for row in batch_thirty_eight)
    assert all(row["locked_boundary"] == ">=2021-01-01 unopened" for row in batch_thirty_eight)
    assert all(row["position_values"] == [-1, 1] for row in batch_thirty_eight)
    assert all(row["cash_allowed"] is False for row in batch_thirty_eight)
    assert all(row["leverage_allowed"] is False for row in batch_thirty_eight)


def test_batch_thirty_nine_rebalances_best_intermediate_vxo_rule() -> None:
    batch_thirty_eight = registry.generate_candidates(38, count=96)
    batch_thirty_nine = registry.generate_candidates(39, count=96)
    batch_forty = registry.generate_candidates(40, count=96)

    assert len(batch_thirty_nine) == 96
    assert len({row["canonical_hash"] for row in batch_thirty_nine}) == 96
    assert not {row["canonical_hash"] for row in batch_thirty_eight}.intersection(
        row["canonical_hash"] for row in batch_thirty_nine
    )
    assert not {row["canonical_hash"] for row in batch_thirty_nine}.intersection(
        row["canonical_hash"] for row in batch_forty
    )
    assert {row["parameters"]["slow_trend_window"] for row in batch_thirty_nine} == {
        63,
        126,
        252,
    }
    assert {row["parameters"]["slow_trend_weight"] for row in batch_thirty_nine} == {
        0.1,
        0.25,
        0.5,
        0.75,
    }
    assert {row["parameters"]["rsi_weight"] for row in batch_thirty_nine} == {0.5, 1.0}
    assert {row["parameters"]["reversal_weight"] for row in batch_thirty_nine} == {
        0.5,
        1.0,
    }
    assert {row["parameters"]["calendar_weight"] for row in batch_thirty_nine} == {
        1.0,
        1.5,
    }
    assert all(row["parameters"]["recovery_threshold_pct"] == 1.25 for row in batch_thirty_nine)
    assert all(row["parameters"]["vxo_z_window"] == 10 for row in batch_thirty_nine)
    assert all(row["parameters"]["vxo_z_threshold"] == 1.4 for row in batch_thirty_nine)
    assert all(row["parameters"]["vxo_positive_weight"] == 2.5 for row in batch_thirty_nine)
    assert all(row["parameters"]["vxo_negative_weight"] == 2.0 for row in batch_thirty_nine)
    assert all(row["locked_boundary"] == ">=2021-01-01 unopened" for row in batch_thirty_nine)
    assert all(row["position_values"] == [-1, 1] for row in batch_thirty_nine)
    assert all(row["cash_allowed"] is False for row in batch_thirty_nine)
    assert all(row["leverage_allowed"] is False for row in batch_thirty_nine)


def test_batch_forty_adds_distinct_causal_overnight_tug_vote() -> None:
    batch_thirty_nine = registry.generate_candidates(39, count=96)
    batch_forty = registry.generate_candidates(40, count=96)
    batch_forty_one = registry.generate_candidates(41, count=96)

    assert len(batch_forty) == 96
    assert len({row["canonical_hash"] for row in batch_forty}) == 96
    assert not {row["canonical_hash"] for row in batch_thirty_nine}.intersection(
        row["canonical_hash"] for row in batch_forty
    )
    assert not {row["canonical_hash"] for row in batch_forty}.intersection(
        row["canonical_hash"] for row in batch_forty_one
    )
    assert {row["parameters"]["tug_lookback"] for row in batch_forty} == {
        1,
        2,
        3,
        5,
    }
    assert {row["parameters"]["tug_weight"] for row in batch_forty} == {
        0.25,
        0.5,
        1.0,
        1.5,
    }
    assert {row["parameters"]["rsi_weight"] for row in batch_forty} == {
        0.5,
        1.0,
        1.5,
    }
    assert {row["parameters"]["calendar_weight"] for row in batch_forty} == {
        1.0,
        1.5,
    }
    assert all(row["parameters"]["reversal_weight"] == 1.0 for row in batch_forty)
    assert all(row["parameters"]["recovery_threshold_pct"] == 1.25 for row in batch_forty)
    assert all(row["parameters"]["vxo_z_window"] == 10 for row in batch_forty)
    assert all(row["parameters"]["vxo_z_threshold"] == 1.4 for row in batch_forty)
    assert all(row["parameters"]["vxo_positive_weight"] == 2.5 for row in batch_forty)
    assert all(row["parameters"]["vxo_negative_weight"] == 2.0 for row in batch_forty)
    assert all(row["locked_boundary"] == ">=2021-01-01 unopened" for row in batch_forty)
    assert all(row["position_values"] == [-1, 1] for row in batch_forty)
    assert all(row["cash_allowed"] is False for row in batch_forty)
    assert all(row["leverage_allowed"] is False for row in batch_forty)


def test_batch_forty_one_refines_unseen_overnight_tug_boundary() -> None:
    batch_forty = registry.generate_candidates(40, count=96)
    batch_forty_one = registry.generate_candidates(41, count=96)
    batch_forty_two = registry.generate_candidates(42, count=96)

    assert len(batch_forty_one) == 96
    assert len({row["canonical_hash"] for row in batch_forty_one}) == 96
    assert not {row["canonical_hash"] for row in batch_forty}.intersection(
        row["canonical_hash"] for row in batch_forty_one
    )
    assert not {row["canonical_hash"] for row in batch_forty_one}.intersection(
        row["canonical_hash"] for row in batch_forty_two
    )
    assert {row["parameters"]["tug_lookback"] for row in batch_forty_one} == {2}
    assert {row["parameters"]["tug_weight"] for row in batch_forty_one} == {
        0.3,
        0.4,
        0.6,
        0.7,
    }
    assert {row["parameters"]["rsi_weight"] for row in batch_forty_one} == {
        1.25,
        1.5,
        1.75,
    }
    assert {row["parameters"]["calendar_weight"] for row in batch_forty_one} == {
        0.75,
        1.0,
        1.25,
        1.5,
    }
    assert {row["parameters"]["vxo_z_threshold"] for row in batch_forty_one} == {
        1.35,
        1.45,
    }
    assert all(row["locked_boundary"] == ">=2021-01-01 unopened" for row in batch_forty_one)
    assert all(row["position_values"] == [-1, 1] for row in batch_forty_one)
    assert all(row["cash_allowed"] is False for row in batch_forty_one)
    assert all(row["leverage_allowed"] is False for row in batch_forty_one)


def test_batch_forty_two_rebalances_recovery_and_negative_vxo_vote() -> None:
    batch_forty_one = registry.generate_candidates(41, count=96)
    batch_forty_two = registry.generate_candidates(42, count=96)
    batch_forty_three = registry.generate_candidates(43, count=96)

    assert len(batch_forty_two) == 96
    assert len({row["canonical_hash"] for row in batch_forty_two}) == 96
    assert not {row["canonical_hash"] for row in batch_forty_one}.intersection(
        row["canonical_hash"] for row in batch_forty_two
    )
    assert not {row["canonical_hash"] for row in batch_forty_two}.intersection(
        row["canonical_hash"] for row in batch_forty_three
    )
    assert {row["parameters"]["recovery_threshold_pct"] for row in batch_forty_two} == {
        1.1,
        1.2,
        1.3,
    }
    assert {row["parameters"]["vxo_negative_weight"] for row in batch_forty_two} == {
        1.5,
        1.75,
        2.25,
        2.5,
    }
    assert {row["parameters"]["calendar_weight"] for row in batch_forty_two} == {
        0.5,
        0.75,
        1.0,
        1.25,
    }
    assert {row["parameters"]["volume_reversal_weight"] for row in batch_forty_two} == {
        0.0,
        0.25,
    }
    assert all(row["parameters"]["tug_lookback"] == 2 for row in batch_forty_two)
    assert all(row["parameters"]["tug_weight"] == 0.5 for row in batch_forty_two)
    assert all(row["parameters"]["vxo_positive_weight"] == 2.5 for row in batch_forty_two)
    assert all(row["locked_boundary"] == ">=2021-01-01 unopened" for row in batch_forty_two)
    assert all(row["position_values"] == [-1, 1] for row in batch_forty_two)
    assert all(row["cash_allowed"] is False for row in batch_forty_two)
    assert all(row["leverage_allowed"] is False for row in batch_forty_two)


def test_batch_forty_five_gates_negative_vxo_vote_with_causal_trend() -> None:
    batch_forty_four = registry.generate_candidates(44, count=96)
    batch_forty_five = registry.generate_candidates(45, count=96)
    batch_forty_six = registry.generate_candidates(46, count=96)

    assert len(batch_forty_five) == 96
    assert len({row["canonical_hash"] for row in batch_forty_five}) == 96
    assert not {row["canonical_hash"] for row in batch_forty_four}.intersection(
        row["canonical_hash"] for row in batch_forty_five
    )
    assert not {row["canonical_hash"] for row in batch_forty_five}.intersection(
        row["canonical_hash"] for row in batch_forty_six
    )
    assert {row["parameters"]["vxo_negative_gate_window"] for row in batch_forty_five} == {
        20,
        40,
        63,
        126,
    }
    assert {
        row["parameters"]["vxo_negative_gate_threshold_pct"]
        for row in batch_forty_five
    } == {-2.0, 0.0, 2.0}
    assert {row["parameters"]["vxo_negative_weight"] for row in batch_forty_five} == {
        1.5,
        2.0,
        2.5,
        3.0,
    }
    assert {row["parameters"]["recovery_threshold_pct"] for row in batch_forty_five} == {
        1.2,
        1.25,
    }
    assert all(row["locked_boundary"] == ">=2021-01-01 unopened" for row in batch_forty_five)
    assert all(row["position_values"] == [-1, 1] for row in batch_forty_five)
    assert all(row["cash_allowed"] is False for row in batch_forty_five)
    assert all(row["leverage_allowed"] is False for row in batch_forty_five)


def test_batch_forty_seven_requires_causal_trend_for_every_short() -> None:
    batch_forty_six = registry.generate_candidates(46, count=96)
    batch_forty_seven = registry.generate_candidates(47, count=96)
    batch_forty_eight = registry.generate_candidates(48, count=96)

    assert len(batch_forty_seven) == 96
    assert len({row["canonical_hash"] for row in batch_forty_seven}) == 96
    assert not {row["canonical_hash"] for row in batch_forty_six}.intersection(
        row["canonical_hash"] for row in batch_forty_seven
    )
    assert not {row["canonical_hash"] for row in batch_forty_seven}.intersection(
        row["canonical_hash"] for row in batch_forty_eight
    )
    assert {row["parameters"]["short_gate_window"] for row in batch_forty_seven} == {
        20,
        40,
        63,
        126,
    }
    assert {row["parameters"]["short_gate_threshold_pct"] for row in batch_forty_seven} == {
        -5.0,
        0.0,
        5.0,
    }
    assert {row["parameters"]["recovery_threshold_pct"] for row in batch_forty_seven} == {
        1.1,
        1.2,
    }
    assert {row["parameters"]["vxo_negative_gate_threshold_pct"] for row in batch_forty_seven} == {
        0.0,
        2.0,
    }
    assert {row["parameters"]["tug_weight"] for row in batch_forty_seven} == {
        0.5,
        1.0,
    }
    assert all(row["locked_boundary"] == ">=2021-01-01 unopened" for row in batch_forty_seven)
    assert all(row["position_values"] == [-1, 1] for row in batch_forty_seven)
    assert all(row["cash_allowed"] is False for row in batch_forty_seven)
    assert all(row["leverage_allowed"] is False for row in batch_forty_seven)


def test_batch_forty_eight_refines_negative_vxo_gate_neighbourhood() -> None:
    batch_forty_seven = registry.generate_candidates(47, count=96)
    batch_forty_eight = registry.generate_candidates(48, count=96)
    batch_forty_nine = registry.generate_candidates(49, count=96)

    assert len(batch_forty_eight) == 96
    assert len({row["canonical_hash"] for row in batch_forty_eight}) == 96
    assert not {row["canonical_hash"] for row in batch_forty_seven}.intersection(
        row["canonical_hash"] for row in batch_forty_eight
    )
    assert not {row["canonical_hash"] for row in batch_forty_eight}.intersection(
        row["canonical_hash"] for row in batch_forty_nine
    )
    assert {
        row["parameters"]["vxo_negative_gate_window"]
        for row in batch_forty_eight
    } == {30, 35, 45, 50}
    assert {
        row["parameters"]["vxo_negative_gate_threshold_pct"]
        for row in batch_forty_eight
    } == {1.0, 2.0, 3.0}
    assert {
        row["parameters"]["vxo_negative_weight"]
        for row in batch_forty_eight
    } == {2.25, 2.5}
    assert {
        row["parameters"]["recovery_threshold_pct"]
        for row in batch_forty_eight
    } == {1.15, 1.2}
    assert {
        row["parameters"]["tug_weight"]
        for row in batch_forty_eight
    } == {0.25, 0.5}
    assert all(
        "short_gate_window" not in row["parameters"]
        for row in batch_forty_eight
    )
    assert all(row["locked_boundary"] == ">=2021-01-01 unopened" for row in batch_forty_eight)
    assert all(row["position_values"] == [-1, 1] for row in batch_forty_eight)
    assert all(row["cash_allowed"] is False for row in batch_forty_eight)
    assert all(row["leverage_allowed"] is False for row in batch_forty_eight)


def test_batch_forty_nine_requires_causal_drawdown_for_every_short() -> None:
    batch_forty_eight = registry.generate_candidates(48, count=96)
    batch_forty_nine = registry.generate_candidates(49, count=96)
    batch_fifty = registry.generate_candidates(50, count=96)

    assert len(batch_forty_nine) == 96
    assert len({row["canonical_hash"] for row in batch_forty_nine}) == 96
    assert not {row["canonical_hash"] for row in batch_forty_eight}.intersection(
        row["canonical_hash"] for row in batch_forty_nine
    )
    assert not {row["canonical_hash"] for row in batch_forty_nine}.intersection(
        row["canonical_hash"] for row in batch_fifty
    )
    assert {
        row["parameters"]["short_drawdown_gate_window"]
        for row in batch_forty_nine
    } == {63, 126, 252}
    assert {
        row["parameters"]["short_drawdown_gate_threshold_pct"]
        for row in batch_forty_nine
    } == {0.0, 2.5, 5.0, 7.5}
    assert {
        row["parameters"]["vxo_negative_gate_window"]
        for row in batch_forty_nine
    } == {35, 40, 45, 50}
    assert {
        row["parameters"]["vxo_negative_gate_threshold_pct"]
        for row in batch_forty_nine
    } == {1.0, 2.0}
    assert all(row["locked_boundary"] == ">=2021-01-01 unopened" for row in batch_forty_nine)
    assert all(row["position_values"] == [-1, 1] for row in batch_forty_nine)
    assert all(row["cash_allowed"] is False for row in batch_forty_nine)
    assert all(row["leverage_allowed"] is False for row in batch_forty_nine)


def test_batch_fifty_refines_causal_drawdown_neighbourhood() -> None:
    batch_forty_nine = registry.generate_candidates(49, count=96)
    batch_fifty = registry.generate_candidates(50, count=96)
    batch_fifty_one = registry.generate_candidates(51, count=96)

    assert len(batch_fifty) == 96
    assert len({row["canonical_hash"] for row in batch_fifty}) == 96
    assert not {row["canonical_hash"] for row in batch_forty_nine}.intersection(
        row["canonical_hash"] for row in batch_fifty
    )
    assert not {row["canonical_hash"] for row in batch_fifty}.intersection(
        row["canonical_hash"] for row in batch_fifty_one
    )
    assert {
        row["parameters"]["short_drawdown_gate_window"]
        for row in batch_fifty
    } == {189, 220, 252, 300}
    assert {
        row["parameters"]["short_drawdown_gate_threshold_pct"]
        for row in batch_fifty
    } == {1.5, 2.0, 3.0}
    assert {
        row["parameters"]["vxo_negative_gate_window"]
        for row in batch_fifty
    } == {30, 35, 40, 45}
    assert {
        row["parameters"]["vxo_negative_gate_threshold_pct"]
        for row in batch_fifty
    } == {0.5, 1.0}
    assert all(row["locked_boundary"] == ">=2021-01-01 unopened" for row in batch_fifty)
    assert all(row["position_values"] == [-1, 1] for row in batch_fifty)
    assert all(row["cash_allowed"] is False for row in batch_fifty)
    assert all(row["leverage_allowed"] is False for row in batch_fifty)


def test_batch_fifty_one_extends_causal_drawdown_boundary() -> None:
    batch_fifty = registry.generate_candidates(50, count=96)
    batch_fifty_one = registry.generate_candidates(51, count=96)
    batch_fifty_two = registry.generate_candidates(52, count=96)

    assert len(batch_fifty_one) == 96
    assert len({row["canonical_hash"] for row in batch_fifty_one}) == 96
    assert not {row["canonical_hash"] for row in batch_fifty}.intersection(
        row["canonical_hash"] for row in batch_fifty_one
    )
    assert not {row["canonical_hash"] for row in batch_fifty_one}.intersection(
        row["canonical_hash"] for row in batch_fifty_two
    )
    assert {
        row["parameters"]["short_drawdown_gate_window"]
        for row in batch_fifty_one
    } == {320, 360, 420, 500}
    assert {
        row["parameters"]["short_drawdown_gate_threshold_pct"]
        for row in batch_fifty_one
    } == {3.0, 3.5, 4.0, 4.5}
    assert {
        row["parameters"]["vxo_negative_gate_window"]
        for row in batch_fifty_one
    } == {35, 40, 45}
    assert {
        row["parameters"]["vxo_negative_gate_threshold_pct"]
        for row in batch_fifty_one
    } == {0.0, 0.5}
    assert all(row["locked_boundary"] == ">=2021-01-01 unopened" for row in batch_fifty_one)
    assert all(row["position_values"] == [-1, 1] for row in batch_fifty_one)
    assert all(row["cash_allowed"] is False for row in batch_fifty_one)
    assert all(row["leverage_allowed"] is False for row in batch_fifty_one)


def test_recovery_calendar_volume_vxo_rule_is_causal_and_fully_invested() -> None:
    index = pd.date_range("1997-01-02", periods=900, freq="B")
    close = pd.Series(
        120.0 + np.linspace(0.0, 45.0, len(index)) + 8.0 * np.sin(np.arange(len(index)) * 0.17),
        index=index,
    )
    open_price = close.shift(1).fillna(close.iloc[0])
    volume = pd.Series(
        1000.0 + 500.0 * (1.0 + np.sin(np.arange(len(index)) * 0.13)),
        index=index,
    )
    vxo = pd.Series(
        20.0 + 4.0 * np.sin(np.arange(len(index)) * 0.11),
        index=index,
    )
    ledger = pd.DataFrame(
        {
            "tr_close": close,
            "tr_open": open_price,
            "open": open_price,
            "high": np.maximum(open_price, close) + 1.0,
            "low": np.minimum(open_price, close) - 1.0,
            "volume": volume,
            "long_return": open_price.shift(-1).div(open_price).sub(1.0),
            "short_return": -open_price.shift(-1).div(open_price).sub(1.0),
        },
        index=index,
    )
    candidate = registry.generate_candidates(32, count=96)[0]
    data = PreparedMarketData(
        ledger=ledger,
        series={"VXO": vxo},
        available_dataset_ids=frozenset({"DS001", "DS002", "DS005"}),
        rejected_datasets={},
        receipts=(),
        split="train",
    )
    result = candidate_decisions(candidate, data)
    assert set(result.decisions.unique()) <= {-1, 1}
    assert result.missing_fraction == 0.0
    assert result.first_evaluable_date is not None

    future_vxo = vxo.copy()
    future_vxo.iloc[-1] *= 4.0
    future_changed = candidate_decisions(
        candidate,
        PreparedMarketData(
            ledger=ledger,
            series={"VXO": future_vxo},
            available_dataset_ids=frozenset({"DS001", "DS002", "DS005"}),
            rejected_datasets={},
            receipts=(),
            split="train",
        ),
    )
    pd.testing.assert_series_equal(result.decisions.iloc[:-1], future_changed.decisions.iloc[:-1])

    symmetric_candidate = json.loads(json.dumps(candidate))
    symmetric_candidate["parameters"]["vxo_positive_weight"] = symmetric_candidate["parameters"][
        "vxo_weight"
    ]
    symmetric_candidate["parameters"]["vxo_negative_weight"] = symmetric_candidate["parameters"][
        "vxo_weight"
    ]
    symmetric_result = candidate_decisions(symmetric_candidate, data)
    pd.testing.assert_series_equal(result.decisions, symmetric_result.decisions)

    asymmetric_candidate = registry.generate_candidates(36, count=96)[0]
    asymmetric_result = candidate_decisions(asymmetric_candidate, data)
    assert set(asymmetric_result.decisions.unique()) <= {-1, 1}
    assert asymmetric_result.missing_fraction == 0.0
    assert asymmetric_result.first_evaluable_date is not None

    slow_trend_candidate = registry.generate_candidates(37, count=96)[0]
    slow_trend_result = candidate_decisions(slow_trend_candidate, data)
    assert set(slow_trend_result.decisions.unique()) <= {-1, 1}
    assert slow_trend_result.missing_fraction == 0.0
    assert slow_trend_result.first_evaluable_date is not None

    tug_candidate = registry.generate_candidates(40, count=96)[0]
    tug_result = candidate_decisions(tug_candidate, data)
    assert set(tug_result.decisions.unique()) <= {-1, 1}
    assert tug_result.missing_fraction == 0.0
    assert tug_result.first_evaluable_date is not None

    future_ledger = ledger.copy()
    future_ledger.loc[index[-1], "tr_open"] *= 1.3
    future_ledger.loc[index[-1], "tr_close"] *= 0.7
    future_ledger.loc[index[-1], "high"] = max(
        future_ledger.loc[index[-1], "tr_open"],
        future_ledger.loc[index[-1], "tr_close"],
    ) + 1.0
    future_ledger.loc[index[-1], "low"] = min(
        future_ledger.loc[index[-1], "tr_open"],
        future_ledger.loc[index[-1], "tr_close"],
    ) - 1.0
    future_tug_result = candidate_decisions(
        tug_candidate,
        PreparedMarketData(
            ledger=future_ledger,
            series={"VXO": vxo},
            available_dataset_ids=frozenset({"DS001", "DS002", "DS005"}),
            rejected_datasets={},
            receipts=(),
            split="train",
        ),
    )
    pd.testing.assert_series_equal(
        tug_result.decisions.iloc[:-1], future_tug_result.decisions.iloc[:-1]
    )

    gated_candidate = registry.generate_candidates(45, count=96)[0]
    gated_result = candidate_decisions(gated_candidate, data)
    assert set(gated_result.decisions.unique()) <= {-1, 1}
    assert gated_result.missing_fraction == 0.0
    assert gated_result.first_evaluable_date is not None
    future_gated_result = candidate_decisions(
        gated_candidate,
        PreparedMarketData(
            ledger=future_ledger,
            series={"VXO": vxo},
            available_dataset_ids=frozenset({"DS001", "DS002", "DS005"}),
            rejected_datasets={},
            receipts=(),
            split="train",
        ),
    )
    pd.testing.assert_series_equal(
        gated_result.decisions.iloc[:-1], future_gated_result.decisions.iloc[:-1]
    )

    short_gated_candidate = registry.generate_candidates(47, count=96)[0]
    short_gated_result = candidate_decisions(short_gated_candidate, data)
    assert set(short_gated_result.decisions.unique()) <= {-1, 1}
    assert short_gated_result.missing_fraction == 0.0
    assert short_gated_result.first_evaluable_date is not None
    future_short_gated_result = candidate_decisions(
        short_gated_candidate,
        PreparedMarketData(
            ledger=future_ledger,
            series={"VXO": vxo},
            available_dataset_ids=frozenset({"DS001", "DS002", "DS005"}),
            rejected_datasets={},
            receipts=(),
            split="train",
        ),
    )
    pd.testing.assert_series_equal(
        short_gated_result.decisions.iloc[:-1],
        future_short_gated_result.decisions.iloc[:-1],
    )

    drawdown_gated_candidate = registry.generate_candidates(49, count=96)[0]
    drawdown_gated_result = candidate_decisions(drawdown_gated_candidate, data)
    assert set(drawdown_gated_result.decisions.unique()) <= {-1, 1}
    assert drawdown_gated_result.missing_fraction == 0.0
    assert drawdown_gated_result.first_evaluable_date is not None
    future_drawdown_gated_result = candidate_decisions(
        drawdown_gated_candidate,
        PreparedMarketData(
            ledger=future_ledger,
            series={"VXO": vxo},
            available_dataset_ids=frozenset({"DS001", "DS002", "DS005"}),
            rejected_datasets={},
            receipts=(),
            split="train",
        ),
    )
    pd.testing.assert_series_equal(
        drawdown_gated_result.decisions.iloc[:-1],
        future_drawdown_gated_result.decisions.iloc[:-1],
    )


def test_recovery_overnight_tug_rule_is_causal_and_fully_invested() -> None:
    index = pd.date_range("2000-01-03", periods=800, freq="B")
    close = pd.Series(
        120.0 + np.linspace(0.0, 45.0, len(index)) + 8.0 * np.sin(np.arange(len(index)) * 0.17),
        index=index,
    )
    open_price = close.shift(1).fillna(close.iloc[0]) * (
        1.0 + 0.008 * np.sin(np.arange(len(index)) * 0.31)
    )
    ledger = pd.DataFrame(
        {
            "tr_close": close,
            "tr_open": open_price,
            "open": open_price,
            "high": np.maximum(open_price, close) + 1.0,
            "low": np.minimum(open_price, close) - 1.0,
            "volume": 1000.0,
            "long_return": open_price.shift(-1).div(open_price).sub(1.0),
            "short_return": -open_price.shift(-1).div(open_price).sub(1.0),
        },
        index=index,
    )
    candidate = registry.generate_candidates(25, count=96)[0]
    data = PreparedMarketData(
        ledger=ledger,
        series={},
        available_dataset_ids=frozenset({"DS001", "DS002"}),
        rejected_datasets={},
        receipts=(),
        split="train",
    )
    result = candidate_decisions(candidate, data)
    assert set(result.decisions.unique()) <= {-1, 1}
    assert result.missing_fraction == 0.0
    assert result.first_evaluable_date is not None

    changed = ledger.copy()
    changed.loc[index[-1], "tr_close"] *= 0.70
    changed.loc[index[-1], "tr_open"] *= 1.20
    future_changed = candidate_decisions(
        candidate,
        PreparedMarketData(
            ledger=changed,
            series={},
            available_dataset_ids=frozenset({"DS001", "DS002"}),
            rejected_datasets={},
            receipts=(),
            split="train",
        ),
    )
    pd.testing.assert_series_equal(result.decisions.iloc[:-1], future_changed.decisions.iloc[:-1])


def test_strong_trend_override_is_causal_and_fully_covered() -> None:
    index = pd.date_range("2000-01-03", periods=300, freq="B")
    close = pd.Series(np.linspace(100.0, 200.0, len(index)), index=index)
    ledger = pd.DataFrame(
        {
            "tr_close": close,
            "tr_open": close,
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "volume": 1000.0,
            "long_return": close.pct_change().fillna(0.0),
            "short_return": -close.pct_change().fillna(0.0),
        },
        index=index,
    )
    candidate = registry.generate_candidates(10, count=1)[0]
    result = candidate_decisions(
        candidate,
        PreparedMarketData(
            ledger=ledger,
            series={},
            available_dataset_ids=frozenset({"DS001", "DS002"}),
            rejected_datasets={},
            receipts=(),
            split="train",
        ),
    )
    expected_warmup = max(
        int(candidate["parameters"]["rsi_trend_window"]),
        int(candidate["parameters"]["reversal_trend_window"]),
        int(candidate["parameters"]["override_window"]),
    )
    assert result.first_evaluable_date == index[expected_warmup].date().isoformat()
    assert result.missing_fraction == 0.0


def test_complete_rsi_definition_preserves_coverage_during_one_way_market() -> None:
    index = pd.date_range("2000-01-03", periods=300, freq="B")
    close = pd.Series(np.linspace(100.0, 200.0, len(index)), index=index)
    ledger = pd.DataFrame(
        {
            "tr_close": close,
            "tr_open": close,
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "volume": 1000.0,
            "long_return": close.pct_change().fillna(0.0),
            "short_return": -close.pct_change().fillna(0.0),
        },
        index=index,
    )
    candidate = _template() | {
        "family": "dual_reversal_trend_vote",
        "required_datasets": ["DS001", "DS002"],
        "parameters": {
            "rsi_window": 5,
            "lower": 25,
            "upper": 75,
            "rsi_trend_window": 225,
            "reversal_window": 5,
            "reversal_threshold_pct": 1.1,
            "reversal_trend_window": 60,
            "rsi_weight": 1,
            "reversal_weight": 1,
        },
    }
    result = candidate_decisions(
        candidate,
        PreparedMarketData(
            ledger=ledger,
            series={},
            available_dataset_ids=frozenset({"DS001", "DS002"}),
            rejected_datasets={},
            receipts=(),
            split="train",
        ),
    )
    assert result.first_evaluable_date == index[225].date().isoformat()
    assert result.missing_fraction == 0.0


def test_trial_ledger_is_cumulative_and_pre_registered(tmp_path, monkeypatch) -> None:
    candidates = tuple(
        _template()
        | {
            "strategy_id": f"candidate-{index}",
            "canonical_hash": canonical_rule_hash(
                _template() | {"strategy_id": f"candidate-{index}"}
            ),
        }
        for index in range(3)
    )
    monkeypatch.setattr(
        registry,
        "base_package",
        lambda: SimpleNamespace(candidates=(), research=(), features=(), datasets=()),
    )
    registry.write_batch_registry(
        tmp_path,
        batch_id=4,
        candidates=candidates,
        previous_trial_count=312,
    )
    rows = registry.read_jsonl(tmp_path / "trial_ledger.jsonl")
    assert [row["global_trial_index"] for row in rows] == [313, 314, 315]
    assert all(row["pre_registered_before_performance"] is True for row in rows)
    manifest = json.loads(
        (tmp_path / "candidate_registry_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["global_trial_count_after_batch"] == 315
    assert manifest["trial_ledger_rows"] == 3


def test_trial_ledger_appends_to_prior_batch(tmp_path, monkeypatch) -> None:
    prior = [
        {
            "batch_id": 0,
            "canonical_hash": f"hash-{index}",
            "global_trial_index": index,
            "pre_registered_before_performance": True,
            "status": "registered",
            "strategy_id": f"prior-{index}",
        }
        for index in range(1, 4)
    ]
    prior_path = tmp_path / "prior" / "trial_ledger.jsonl"
    registry.write_jsonl(prior_path, prior)
    monkeypatch.setenv("AURORA_PRIOR_TRIAL_LEDGER_PATH", str(prior_path))
    candidate = _template() | {
        "strategy_id": "candidate-4",
        "canonical_hash": canonical_rule_hash(_template() | {"strategy_id": "candidate-4"}),
    }
    monkeypatch.setattr(
        registry,
        "base_package",
        lambda: SimpleNamespace(candidates=(), research=(), features=(), datasets=()),
    )
    registry.write_batch_registry(
        tmp_path / "current",
        batch_id=1,
        candidates=(candidate,),
        previous_trial_count=3,
    )
    rows = registry.read_jsonl(tmp_path / "current" / "trial_ledger.jsonl")
    assert [row["global_trial_index"] for row in rows] == [1, 2, 3, 4]
    assert (tmp_path / "current" / "autonomous_trial_ledger.parquet").is_file()


def test_reused_market_data_refreshes_batch_registry_without_mutating_snapshot(
    tmp_path, monkeypatch
) -> None:
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    snapshot = prepared / "spy_ledger.parquet"
    snapshot.write_bytes(b"immutable-market-snapshot")
    monkeypatch.setenv("AURORA_AUTONOMOUS_BATCH_ID", "11")
    monkeypatch.setenv("AURORA_AUTONOMOUS_CANDIDATE_COUNT", "2")
    monkeypatch.setenv("AURORA_AUTONOMOUS_PREVIOUS_TRIAL_COUNT", "312")
    monkeypatch.delenv("AURORA_PRIOR_TRIAL_LEDGER_PATH", raising=False)

    candidates = refresh_autonomous_prepared_inputs(prepared)

    assert [row["strategy_id"] for row in candidates] == [
        "AUTO-B0011-0000",
        "AUTO-B0011-0001",
    ]
    assert snapshot.read_bytes() == b"immutable-market-snapshot"
    ledger = registry.read_jsonl(prepared / "trial_ledger.jsonl")
    assert [row["global_trial_index"] for row in ledger] == [313, 314]
    assert pd.read_csv(prepared / "job_manifest.csv")["strategy_id"].tolist() == [
        "AUTO-B0011-0000",
        "AUTO-B0011-0001",
    ]


def test_trial_ledger_prepends_verified_312_historical_rows(tmp_path, monkeypatch) -> None:
    historical = [
        {
            "batch_id": "V1" if index <= 168 else "V2",
            "canonical_hash": f"historical-{index}",
            "global_trial_index": index,
            "pre_registered_before_performance": True,
            "status": "evaluated",
            "strategy_id": f"historical-{index}",
        }
        for index in range(1, 313)
    ]
    historical_path = (
        tmp_path / "current" / "historical_multiplicity" / "historical_trial_ledger.jsonl"
    )
    registry.write_jsonl(historical_path, historical)
    prior = [
        {
            "batch_id": "0",
            "canonical_hash": "pilot-313",
            "global_trial_index": 313,
            "pre_registered_before_performance": True,
            "status": "evaluated",
            "strategy_id": "pilot-313",
        }
    ]
    prior_path = tmp_path / "prior" / "trial_ledger.jsonl"
    registry.write_jsonl(prior_path, prior)
    monkeypatch.setenv("AURORA_PRIOR_TRIAL_LEDGER_PATH", str(prior_path))
    candidate = _template() | {
        "strategy_id": "candidate-314",
        "canonical_hash": canonical_rule_hash(_template() | {"strategy_id": "candidate-314"}),
    }
    monkeypatch.setattr(
        registry,
        "base_package",
        lambda: SimpleNamespace(candidates=(), research=(), features=(), datasets=()),
    )
    registry.write_batch_registry(
        tmp_path / "current",
        batch_id=1,
        candidates=(candidate,),
        previous_trial_count=313,
    )
    rows = registry.read_jsonl(tmp_path / "current" / "trial_ledger.jsonl")
    assert len(rows) == 314
    assert [row["global_trial_index"] for row in rows] == list(range(1, 315))


def test_historical_ledger_preserves_all_312_canonical_trials() -> None:
    v1_ids = [f"V1-{index:03d}" for index in range(168)]
    v2_ids = [f"V2-{index:03d}" for index in range(144)]
    cumulative = pd.DataFrame(
        [
            {
                "campaign": campaign,
                "strategy_id": identifier,
                "status": "evaluated",
                "fdr_pvalue": 0.5,
            }
            for campaign, identifiers in (("V1", v1_ids), ("V2", v2_ids))
            for identifier in identifiers
        ]
    )

    def metrics(ids: list[str]) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "unit_type": "candidate",
                    "unit_key": identifier,
                    "strategy_id": identifier,
                    "canonical_hash": f"hash-{identifier}",
                    "status": "evaluated",
                }
                for identifier in ids
            ]
        )

    rows = build_historical_trial_ledger(cumulative, metrics(v1_ids), metrics(v2_ids))
    assert len(rows) == 312
    assert [row["global_trial_index"] for row in rows] == list(range(1, 313))
    assert len({row["canonical_hash"] for row in rows}) == 312


def test_trial_ledger_requires_prior_source_after_first_batch(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("AURORA_PRIOR_TRIAL_LEDGER_PATH", raising=False)
    monkeypatch.setattr(
        registry,
        "base_package",
        lambda: SimpleNamespace(candidates=(), research=(), features=(), datasets=()),
    )
    with pytest.raises(ValueError, match="PRIOR_TRIAL_LEDGER_REQUIRED"):
        registry.write_batch_registry(
            tmp_path,
            batch_id=2,
            candidates=(_template(),),
            previous_trial_count=313,
        )


def _metric_row(
    strategy_id: str, values: np.ndarray, *, family: str = "price_trend_sma"
) -> dict[str, object]:
    dates = pd.date_range("2000-01-03", periods=len(values), freq="B")
    return {
        "unit_key": strategy_id,
        "unit_type": "candidate",
        "strategy_id": strategy_id,
        "family": family,
        "canonical_hash": strategy_id,
        "status": "evaluated",
        "train_dates": [item.isoformat() for item in dates],
        "train_returns": values.tolist(),
        "train_positions": [1 if index % 2 else -1 for index in range(len(values))],
        "annual_metrics_json": json.dumps(
            [
                {
                    "year": 2000,
                    "sessions": len(values),
                    "return_pct": 5.0,
                    "cagr_pct": 5.0,
                    "sharpe": 1.0,
                    "positive": True,
                }
            ]
        ),
    }


def test_evaluate_batch_writes_auditable_rows(tmp_path) -> None:
    candidate = _metric_row("AUTO-1", np.full(30, 0.001))
    benchmark = dict(_metric_row("buy_and_hold_spy_total_return", np.full(30, 0.0005)))
    benchmark["unit_type"] = "benchmark"
    result = evaluate_batch([candidate, benchmark], tmp_path, batch_id=0)
    assert result["total_strategies_evaluated"] == 1
    assert result["total_strategies_rejected"] == 0
    leaderboard = pd.read_csv(tmp_path / "leaderboard.csv")
    assert set(leaderboard["strategy_id"]) == {"AUTO-1", "buy_and_hold_spy_total_return"}
    summary = json.loads((tmp_path / "autonomous_batch_summary.json").read_text(encoding="utf-8"))
    assert summary["locked_opened"] is False
    assert summary["validation_used_for_selection"] is False
    freeze = _verify_freeze(tmp_path / "train_selection_freeze.json")
    assert freeze["locked_opened"] is False
    assert (
        json.loads((tmp_path / "train_freeze_candidate.json").read_text(encoding="utf-8")) == freeze
    )
    with pytest.raises(ValidationGateError, match="TRAIN_FREEZE_NOT_ELIGIBLE"):
        _verify_freeze(tmp_path / "train_selection_freeze.json", require_finalized=True)


def test_exploratory_candidate_selection_is_exact_and_fail_closed() -> None:
    first = _template()
    second = dict(_template(), strategy_id="other")
    assert _candidate_from_registry([first, second], "other")["strategy_id"] == "other"
    with pytest.raises(
        ValidationGateError,
        match="EXPLORATORY_CANDIDATE_NOT_UNIQUE_IN_REGISTRY",
    ):
        _candidate_from_registry([first, second], "missing")
    assert EXPLORATORY_VALIDATION_ACK == ("OPEN_EXPLORATORY_VALIDATION_2011_2020_OWNER_AUTHORIZED")


def test_exploratory_validation_package_uses_exact_candidate_requirements(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = dict(_template(), strategy_id="vxo-candidate", required_datasets=["DS001", "DS005"])
    candidate["canonical_hash"] = canonical_rule_hash(candidate)
    (tmp_path / "candidate_registry.jsonl").write_text(
        json.dumps(candidate) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        "aurora.infra.sp500_autonomous_discovery.validation.base_package",
        lambda: registry.base_package(),
    )
    package = exploratory_validation_package(tmp_path, "vxo-candidate")
    assert package.candidates == (candidate,)
    assert package.required_dataset_ids() == ("DS001", "DS005")


def test_validation_workflow_requests_only_the_frozen_validation_period() -> None:
    text = Path(".github/workflows/sp500-autonomous-discovery.yml").read_text(encoding="utf-8")
    assert text.count('start="2011-01-01"') >= 2
    assert 'split="validation"' in text
    assert 'start="2010-01-01"' not in text

    distributions = pd.read_csv(
        "campaigns/sp500_long_short_daily/official_inputs/"
        "state_street_spy_distributions_2011_2020.csv"
    )
    dates = pd.to_datetime(distributions["ex_date"])
    assert len(distributions) == 40
    assert dates.min() >= pd.Timestamp("2011-01-01")
    assert dates.max() <= pd.Timestamp("2020-12-31")


def test_exploratory_validation_can_reuse_verified_stooq_windows() -> None:
    workflow = yaml.safe_load(
        Path(".github/workflows/sp500-autonomous-discovery.yml").read_text(encoding="utf-8")
    )
    inputs = workflow[True]["workflow_dispatch"]["inputs"]
    assert inputs["validation_stooq_run_id"]["default"] == ""

    steps = workflow["jobs"]["exploratory_validation"]["steps"]
    download = next(
        step for step in steps if step["name"] == "Download reusable validation Stooq windows"
    )
    merge = next(
        step for step in steps if step["name"] == "Merge reusable validation Stooq windows"
    )
    prepare = next(step for step in steps if step["name"] == "Prepare bounded validation data")
    upload = next(
        step for step in steps if step["name"] == "Upload exploratory validation result"
    )
    assert "sp500-ls-stooq-window-${VALIDATION_STOOQ_RUN_ID}-*" in download["run"]
    assert "--expected-windows 82" in merge["run"]
    assert "--requested-start 2011-01-01" in merge["run"]
    assert "--requested-end 2020-12-31" in merge["run"]
    assert "SP500_STOOQ_HISTORY_CSV" in prepare["run"]
    assert "SP500_STOOQ_HISTORY_MANIFEST" in prepare["run"]
    assert upload["if"] == "always()"


def test_block_sum_bootstrap_matches_original_sampling(monkeypatch) -> None:
    repetitions = 100
    monkeypatch.setattr(autonomous_statistics, "BOOTSTRAP_REPETITIONS", repetitions)
    values = np.linspace(-0.02, 0.03, 137)
    matrix = np.column_stack([values, values[::-1] * 0.7, np.sin(np.arange(137)) / 100])

    def reference_global(raw: np.ndarray, seed: int) -> float:
        observed = raw.mean(axis=0)
        centered = raw - observed
        blocks = int(np.ceil(len(raw) / autonomous_statistics.BLOCK_LENGTH))
        rng = np.random.default_rng(seed)
        exceed = 0
        for _ in range(repetitions):
            starts = rng.integers(0, len(raw), size=blocks)
            sampled = np.empty((blocks * autonomous_statistics.BLOCK_LENGTH, raw.shape[1]))
            for offset in range(autonomous_statistics.BLOCK_LENGTH):
                sampled[offset :: autonomous_statistics.BLOCK_LENGTH] = centered[
                    (starts + offset) % len(raw)
                ]
            means = sampled[: len(raw)].mean(axis=0)
            exceed += int(means.max() >= observed.max())
        return (1 + exceed) / (repetitions + 1)

    expected = reference_global(matrix, 77)
    observed = autonomous_statistics._global_max_bootstrap_pvalue(matrix, seed=77)
    assert observed == expected


def test_freeze_reason_and_rejections_are_semantically_consistent() -> None:
    assert freeze_selection_reason([]) == "no candidate passed all frozen train gates"
    assert freeze_selection_reason([{"strategy_id": "winner"}]) == ("all frozen train gates passed")
    assert freeze_rejection_reasons(
        [
            {"strategy_id": "ok", "status": "evaluated", "rejection_reason": None},
            {"strategy_id": "bad", "status": "rejected", "rejection_reason": "DATA_MISSING"},
            {"strategy_id": "empty", "status": "rejected", "rejection_reason": None},
        ]
    ) == {"bad": "DATA_MISSING"}


def test_dedupe_and_cost_assignment_are_traceable() -> None:
    first = _template()
    second = dict(first, strategy_id="same-effective-rule", notes="different identity")
    dedupe = build_dedupe_map([first, second])
    assert len(dedupe) == 2
    assert sum(bool(row["deduped"]) for row in dedupe) == 1
    assignments = assign_by_cost(
        [
            {"strategy_id": "slow", "canonical_hash": "s", "cost_score": 5.0},
            {"strategy_id": "fast", "canonical_hash": "f", "cost_score": 0.5},
        ],
        2,
    )
    assert {row["strategy_id"] for row in assignments} == {"slow", "fast"}
    assert {row["estimated_cost_bucket"] for row in assignments} == {"slow", "fast"}


def test_feature_store_is_causal_and_keyed() -> None:
    index = pd.date_range("2020-01-01", periods=260, freq="B")
    frame = pd.DataFrame(
        {
            "tr_close": np.linspace(100, 130, len(index)),
            "high": np.linspace(101, 131, len(index)),
            "low": np.linspace(99, 129, len(index)),
            "volume": np.full(len(index), 1000.0),
        },
        index=index,
    )
    store = FeatureStore(
        dataset_sha256="data", code_sha="code", start="2020-01-01", end="2020-12-31"
    )
    features = store.get_or_build("SPY", frame)
    assert features.index.equals(index)
    assert features.loc[index[0], "return_20d"] != features.loc[index[-1], "return_20d"]
    assert store.key("SPY").value == store.key("SPY").value
    assert store.key("SPY").value != store.key("OTHER").value
    assert "SPY" in store.manifest()["symbols"]


def test_cached_price_signal_matches_uncached_signal() -> None:
    index = pd.date_range("2000-01-01", periods=260, freq="B")
    close = pd.Series(np.linspace(100.0, 130.0, len(index)), index=index)
    ledger = pd.DataFrame(
        {
            "tr_close": close,
            "tr_open": close,
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "volume": 1000.0,
            "long_return": close.pct_change().fillna(0.0),
            "short_return": -close.pct_change().fillna(0.0),
        },
        index=index,
    )
    data = PreparedMarketData(
        ledger=ledger,
        series={},
        available_dataset_ids=frozenset({"DS001"}),
        rejected_datasets={},
        receipts=(),
        split="train",
    )
    store = FeatureStore(
        dataset_sha256="data",
        code_sha="code",
        start="2000-01-01",
        end="2000-12-31",
    )
    features = store.get_or_build("SPY", ledger)
    cases = (
        ("price_trend_sma", {"lookback": 20}),
        ("time_series_momentum", {"lookback": 20}),
        ("short_horizon_reversal", {"lookback": 20}),
        ("trend_ensemble", {"horizons": [20, 63]}),
        ("dual_ma_cross", {"fast": 10, "slow": 20}),
    )
    for family, parameters in cases:
        candidate = _template() | {
            "family": family,
            "required_datasets": ["DS001"],
            "parameters": parameters,
        }
        uncached = candidate_decisions(candidate, data).decisions
        cached = candidate_decisions(candidate, data, feature_frame=features).decisions
        pd.testing.assert_series_equal(uncached, cached)


@pytest.mark.parametrize(
    ("family", "parameters"),
    (
        ("rsi_reversal", {"window": 2, "lower": 10, "upper": 90}),
        ("internal_bar_strength_reversal", {"lower": 0.2, "upper": 0.8}),
        ("return_threshold_reversal", {"lookback": 2, "threshold_pct": 1.0}),
        ("streak_reversal", {"streak": 3}),
        (
            "reversal_trend_blend",
            {"reversal_window": 2, "trend_window": 20, "reversal_threshold_pct": 1.0},
        ),
        (
            "rsi_trend_blend",
            {"rsi_window": 2, "lower": 10, "upper": 90, "trend_window": 20},
        ),
        (
            "dual_reversal_trend_vote",
            {
                "rsi_window": 5,
                "lower": 25,
                "upper": 75,
                "rsi_trend_window": 225,
                "reversal_window": 5,
                "reversal_threshold_pct": 1.1,
                "reversal_trend_window": 60,
                "rsi_weight": 1,
                "reversal_weight": 1,
            },
        ),
        ("multi_horizon_reversal", {"horizons": [1, 2, 5]}),
        ("intraday_return_reversal", {"threshold_pct": 0.5}),
    ),
)
def test_targeted_reversal_signals_are_causal_and_always_invested(
    family: str,
    parameters: dict[str, object],
) -> None:
    index = pd.date_range("2000-01-03", periods=80, freq="B")
    close = pd.Series(
        100.0 + np.sin(np.arange(len(index)) / 2.0) * 4.0 + np.arange(len(index)) * 0.05,
        index=index,
    )
    ledger = pd.DataFrame(
        {
            "tr_close": close,
            "tr_open": close.shift(1).fillna(close.iloc[0]) * 1.001,
            "open": close.shift(1).fillna(close.iloc[0]) * 1.001,
            "high": close + 1.5,
            "low": close - 1.5,
            "volume": 1000.0,
            "long_return": close.pct_change().fillna(0.0),
            "short_return": -close.pct_change().fillna(0.0),
        },
        index=index,
    )
    data = PreparedMarketData(
        ledger=ledger,
        series={},
        available_dataset_ids=frozenset({"DS001", "DS002"}),
        rejected_datasets={},
        receipts=(),
        split="train",
    )
    candidate = _template() | {
        "family": family,
        "required_datasets": ["DS001", "DS002"],
        "parameters": parameters,
    }
    original = candidate_decisions(candidate, data).decisions
    changed = ledger.copy()
    changed.loc[index[-1], "tr_close"] *= 1.25
    changed.loc[index[-1], "high"] = changed.loc[index[-1], "tr_close"] + 1.5
    future_changed = candidate_decisions(
        candidate,
        PreparedMarketData(
            ledger=changed,
            series={},
            available_dataset_ids=frozenset({"DS001", "DS002"}),
            rejected_datasets={},
            receipts=(),
            split="train",
        ),
    ).decisions
    assert set(original.unique()) <= {-1, 1}
    pd.testing.assert_series_equal(original.iloc[:-1], future_changed.iloc[:-1])


def test_workflow_is_github_only_and_bounded() -> None:
    path = ".github/workflows/sp500-autonomous-discovery.yml"
    text = open(path, encoding="utf-8").read()
    yaml.safe_load(text)
    assert "C:\\" not in text
    assert "self-hosted" not in text
    assert "ubuntu-24.04" in text
    assert "2010-12-31" in text
    assert "2020-12-31" in text
    assert "2021-01-01" in text
    assert "OPEN_VALIDATION_2011_2020_ONCE_AUTONOMOUS" in text
    assert "sp500-autonomous-validation-once" in text
    assert "cancel-in-progress: false" in text
    reusable_text = open(".github/workflows/_aurora-future-run-v3.yml", encoding="utf-8").read()
    assert "autonomous_prior_ledger_artifact_name" in reusable_text
    assert 'or "sp500_autonomous_discovery" in workload' in reusable_text
    reusable = yaml.safe_load(reusable_text)
    assert "refreshed-prepared" in reusable["env"]["AURORA_PREPARED_ARTIFACT_NAME"]
    assert reusable["env"]["AURORA_PREPARED_ARTIFACT_RUN_ID"] == (
        "${{ inputs.prepared_artifact_name != '' && contains(inputs.workload, "
        "'sp500_autonomous_discovery') && github.run_id || "
        "inputs.prepared_artifact_run_id || github.run_id }}"
    )
    prepare_data_steps = reusable["jobs"]["prepare_data"]["steps"]
    source_download = next(
        item
        for item in prepare_data_steps
        if item.get("name") == "Download shared immutable inputs"
    )
    assert source_download["with"]["name"] == (
        "${{ inputs.prepared_artifact_name || env.AURORA_PREPARED_ARTIFACT_NAME }}"
    )
    assert source_download["with"]["run-id"] == (
        "${{ inputs.prepared_artifact_run_id || github.run_id }}"
    )
    refresh_index = next(
        index
        for index, item in enumerate(prepare_data_steps)
        if item.get("name") == "Refresh autonomous batch inputs on reused market data"
    )
    refreshed_upload_index = next(
        index
        for index, item in enumerate(prepare_data_steps)
        if item.get("name") == "Upload refreshed autonomous prepared inputs"
    )
    assert refresh_index < refreshed_upload_index
    refreshed_upload = prepare_data_steps[refreshed_upload_index]
    assert refreshed_upload["with"]["name"] == ("${{ env.AURORA_PREPARED_ARTIFACT_NAME }}")
    for job in reusable["jobs"].values():
        for step in job.get("steps", []):
            if step.get("name") == "Download prepared inputs":
                assert step["with"]["run-id"] == ("${{ env.AURORA_PREPARED_ARTIFACT_RUN_ID }}")
        if "uses" in job and "prepared-artifact-run-id" in job.get("with", {}):
            assert job["with"]["prepared-artifact-run-id"] == (
                "${{ env.AURORA_PREPARED_ARTIFACT_RUN_ID }}"
            )
    for job_name, step_name in (
        ("smoke", "Run bounded smoke"),
        ("pilot", "Resolve exact profile or measure fresh pilot"),
        ("plan", "Build adaptive balanced plan"),
    ):
        step = next(
            item for item in reusable["jobs"][job_name]["steps"] if item.get("name") == step_name
        )
        assert step["env"]["AURORA_PREPARED_ROOT"] == "${{ runner.temp }}/prepared"
    sp500_prepare_steps = reusable["jobs"]["sp500_prepare_data"]["steps"]
    prior_ledger_index = next(
        index
        for index, item in enumerate(sp500_prepare_steps)
        if item.get("name") == "Download prior autonomous trial ledger"
    )
    prepare_index = next(
        index
        for index, item in enumerate(sp500_prepare_steps)
        if item.get("name") == "Prepare immutable SPY data"
    )
    assert prior_ledger_index < prepare_index
    prior_ledger_step = sp500_prepare_steps[prior_ledger_index]
    assert prior_ledger_step["with"]["run-id"] == ("${{ inputs.autonomous_prior_ledger_run_id }}")
    historical_step = next(
        item
        for item in sp500_prepare_steps
        if item.get("name") == "Download canonical V1 and V2 multiplicity evidence"
    )
    assert historical_step["with"]["run-id"] == "31007105419"
    assert historical_step["with"]["name"] == "sp500-ls-v2-train-results"
    pilot_step = next(
        item
        for item in sp500_prepare_steps
        if item.get("name") == "Download canonical autonomous pilot evidence"
    )
    assert pilot_step["with"]["run-id"] == "31036879593"
    prepare_step = next(
        item for item in sp500_prepare_steps if item.get("name") == "Prepare immutable SPY data"
    )
    assert "AURORA_HISTORICAL_MULTIPLICITY_SOURCE" in prepare_step["env"]
    for phase in (
        "preflight",
        "research",
        "data_build",
        "pilot",
        "search_batch",
        "merge_batch",
        "statistical_gate",
        "freeze",
        "validation_once",
        "verify",
    ):
        assert f"- {phase}" in text
