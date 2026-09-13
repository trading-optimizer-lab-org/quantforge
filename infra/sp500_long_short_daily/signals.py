"""Exact causal rule dispatcher for the frozen SPY candidate pack."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from aurora.infra.sp500_long_short_daily.data import PreparedMarketData


class CandidateRejected(RuntimeError):
    """A disclosed candidate cannot be evaluated under the frozen contract."""


IMPLEMENTED_FAMILIES = frozenset(
    {
        "calendar_seasonality",
        "credit_spread_regime",
        "dual_ma_cross",
        "dual_reversal_trend_vote",
        "trend_guarded_dual_reversal",
        "volatility_regime_reversal",
        "overnight_tug_reversal_vote",
        "strong_trend_override_reversal",
        "asymmetric_trend_override_reversal",
        "drawdown_recovery_override_reversal",
        "quiet_bull_recovery_override_reversal",
        "recovery_trend_breakout_majority",
        "high_vol_crash_recovery_reversal",
        "adaptive_recovery_edge_switch",
        "recovery_overnight_tug_vote",
        "recovery_turn_month_vote",
        "recovery_internal_bar_strength_vote",
        "recovery_multi_horizon_reversal_vote",
        "recovery_volume_gated_reversal_vote",
        "recovery_calendar_volume_reversal_vote",
        "recovery_calendar_volume_vxo_vote",
        "financial_conditions_regime",
        "monetary_inflation_regime",
        "overnight_futures_proxy",
        "price_breakout",
        "price_trend_sma",
        "internal_bar_strength_reversal",
        "intraday_return_reversal",
        "multi_horizon_reversal",
        "realized_volatility_state",
        "return_threshold_reversal",
        "reversal_trend_blend",
        "rsi_reversal",
        "rsi_trend_blend",
        "short_horizon_reversal",
        "simple_rule_ensemble",
        "streak_reversal",
        "time_series_momentum",
        "trend_ensemble",
        "variance_risk_premium_proxy",
        "vix_extreme_reversal",
        "vix_level_change",
        "vix_term_structure",
        "volatility_conditioned_trend",
        "volume_conditioned_reversal",
        "yield_curve_regime",
    }
)

EXPLICIT_FAMILY_REJECTIONS = {
    "breadth_thrust_proxy": "DATA_INELIGIBLE:PROXY_ONLY_DS071",
    "breadth_trend_proxy": "DATA_INELIGIBLE:PROXY_ONLY_DS071",
    "correlation_dispersion_proxy": (
        "DATA_ADAPTER_REQUIRED:DS056_CAUSAL_SECTOR_TOTAL_RETURN_PANEL"
    ),
    "cross_asset_risk_off": (
        "INCOMPLETE_FROZEN_RULE_SPEC:CROSS_ASSET_COMPONENT_SIGNS"
    ),
    "markov_regime_filtered": (
        "INCOMPLETE_FROZEN_MODEL_SPEC:MARKOV_MODEL_RESTART_AND_CONVERGENCE_GRID"
    ),
    "regularized_logit": (
        "INCOMPLETE_FROZEN_MODEL_SPEC:MISSING_DECLARED_HYPERPARAMETER_GRID"
    ),
    "sentiment_positioning": (
        "INCOMPLETE_FROZEN_RULE_SPEC:CAUSAL_STANDARDIZATION_WINDOW_AND_STALENESS"
    ),
    "valuation_equity_premium": (
        "INCOMPLETE_FROZEN_MODEL_SPEC:VALUATION_RECURSIVE_ESTIMATION_CONSTRAINTS"
    ),
}


@dataclass(frozen=True)
class SignalResult:
    decisions: pd.Series
    first_evaluable_date: str | None
    missing_fraction: float


def _state_from_events(
    index: pd.DatetimeIndex,
    long_event: pd.Series | np.ndarray,
    short_event: pd.Series | np.ndarray,
    *,
    eligible: pd.Series | np.ndarray | None = None,
) -> pd.Series:
    long_mask = pd.Series(long_event, index=index).fillna(False).astype(bool)
    short_mask = pd.Series(short_event, index=index).fillna(False).astype(bool)
    if (long_mask & short_mask).any():
        raise CandidateRejected("CONTRADICTORY_LONG_SHORT_EVENT")
    events = pd.Series(np.nan, index=index, dtype=float)
    events.loc[long_mask] = 1.0
    events.loc[short_mask] = -1.0
    result = events.ffill().fillna(1.0).astype(np.int8)
    if eligible is None:
        eligible_mask = long_mask | short_mask
    else:
        eligible_mask = pd.Series(eligible, index=index).fillna(False).astype(bool)
    first_valid = eligible_mask[eligible_mask].index.min() if eligible_mask.any() else None
    result.attrs["first_valid"] = first_valid
    result.attrs["eligible_count"] = int(eligible_mask.sum())
    return result


def _state_from_score(score: pd.Series, *, long_on_zero: bool = False) -> pd.Series:
    valid = score.notna()
    long_event = valid & (score >= 0 if long_on_zero else score > 0)
    short_event = valid & (score < 0)
    return _state_from_events(
        pd.DatetimeIndex(score.index),
        long_event,
        short_event,
        eligible=valid,
    )


def _rolling_zscore(values: pd.Series, window: int) -> pd.Series:
    mean = values.rolling(window, min_periods=window).mean()
    std = values.rolling(window, min_periods=window).std(ddof=1)
    return (values - mean) / std.replace(0.0, np.nan)


def _expanding_zscore(values: pd.Series, minimum: int = 252) -> pd.Series:
    mean = values.expanding(min_periods=minimum).mean()
    std = values.expanding(min_periods=minimum).std(ddof=1)
    return (values - mean) / std.replace(0.0, np.nan)


def _rolling_percentile(values: pd.Series, window: int) -> pd.Series:
    def rank_last(raw: np.ndarray) -> float:
        if np.isnan(raw).any():
            return np.nan
        return float(np.count_nonzero(raw <= raw[-1]) / len(raw))

    return values.rolling(window, min_periods=window).apply(rank_last, raw=True)


def _calendar_lag(
    values: pd.Series,
    *,
    months: int = 0,
    weeks: int = 0,
) -> pd.Series:
    source = values.dropna().sort_index(kind="mergesort")
    if source.empty:
        return pd.Series(np.nan, index=values.index, dtype=float)
    targets = pd.DatetimeIndex(values.index) - pd.DateOffset(months=months, weeks=weeks)
    lagged = source.reindex(targets, method="ffill")
    lagged.index = values.index
    return lagged


def _series(data: PreparedMarketData, name: str) -> pd.Series:
    try:
        values = data.series[name]
    except KeyError as exc:
        raise CandidateRejected(f"MISSING_CAUSAL_SERIES:{name}") from exc
    return values.reindex(data.ledger.index)


def _required_data_gate(candidate: Mapping[str, Any], data: PreparedMarketData) -> None:
    missing = sorted(set(candidate["required_datasets"]) - set(data.available_dataset_ids))
    if missing:
        reasons = [f"{dataset}:{data.rejected_datasets.get(dataset, 'UNAVAILABLE')}" for dataset in missing]
        raise CandidateRejected("DATA_GATE_REJECTED:" + "|".join(reasons))


def _price_score(
    candidate: Mapping[str, Any],
    data: PreparedMarketData,
    *,
    feature_frame: pd.DataFrame | None = None,
) -> pd.Series:
    family = str(candidate["family"])
    parameters = candidate["parameters"]
    ledger = data.ledger
    close = ledger["tr_close"].astype(float)
    log_return = np.log(close / close.shift(1))

    def cached(name: str) -> pd.Series | None:
        if feature_frame is None or name not in feature_frame.columns:
            return None
        return feature_frame[name].reindex(close.index)

    if family == "price_trend_sma":
        lookback = int(parameters["lookback"])
        cached_sma = cached(f"sma_{lookback}")
        return close - cached_sma if cached_sma is not None else close - close.rolling(lookback, min_periods=lookback).mean()
    if family == "time_series_momentum":
        lookback = int(parameters["lookback"])
        cached_return = cached(f"return_{lookback}d")
        return cached_return if cached_return is not None else close / close.shift(lookback) - 1.0
    if family == "short_horizon_reversal":
        lookback = int(parameters["lookback"])
        cached_return = cached(f"return_{lookback}d")
        return -cached_return if cached_return is not None else -(close / close.shift(lookback) - 1.0)
    if family == "multi_horizon_reversal":
        horizons = [int(value) for value in parameters["horizons"]]
        components = [close / close.shift(window) - 1.0 for window in horizons]
        return -pd.concat(components, axis=1).mean(axis=1, skipna=False)
    if family == "reversal_trend_blend":
        reversal_window = int(parameters["reversal_window"])
        trend_window = int(parameters["trend_window"])
        threshold = float(parameters["reversal_threshold_pct"]) / 100.0
        reversal_return = close / close.shift(reversal_window) - 1.0
        trend_return = close / close.shift(trend_window) - 1.0
        return (-reversal_return).where(reversal_return.abs() >= threshold, trend_return)
    if family == "rsi_trend_blend":
        window = int(parameters["rsi_window"])
        trend_window = int(parameters["trend_window"])
        delta = close.diff()
        gain = delta.clip(lower=0.0).ewm(
            alpha=1.0 / window,
            adjust=False,
            min_periods=window,
        ).mean()
        loss = (-delta.clip(upper=0.0)).ewm(
            alpha=1.0 / window,
            adjust=False,
            min_periods=window,
        ).mean()
        rsi = 100.0 - 100.0 / (1.0 + gain / loss.replace(0.0, np.nan))
        trend = close / close.shift(trend_window) - 1.0
        score = trend.copy()
        score.loc[rsi <= float(parameters["lower"])] = 1.0
        score.loc[rsi >= float(parameters["upper"])] = -1.0
        return score.where(rsi.notna() & trend.notna())
    if family == "dual_reversal_trend_vote":
        rsi_window = int(parameters["rsi_window"])
        delta = close.diff()
        gain = delta.clip(lower=0.0).ewm(
            alpha=1.0 / rsi_window,
            adjust=False,
            min_periods=rsi_window,
        ).mean()
        loss = (-delta.clip(upper=0.0)).ewm(
            alpha=1.0 / rsi_window,
            adjust=False,
            min_periods=rsi_window,
        ).mean()
        relative_strength = gain / loss.replace(0.0, np.nan)
        rsi = 100.0 - 100.0 / (1.0 + relative_strength)
        rsi = rsi.mask((loss == 0.0) & (gain > 0.0), 100.0)
        rsi = rsi.mask((gain == 0.0) & (loss > 0.0), 0.0)
        rsi = rsi.mask((gain == 0.0) & (loss == 0.0), 50.0)
        rsi_trend_return = (
            close / close.shift(int(parameters["rsi_trend_window"])) - 1.0
        )
        rsi_component = np.sign(rsi_trend_return)
        rsi_component = rsi_component.where(rsi > float(parameters["lower"]), 1.0)
        rsi_component = rsi_component.where(rsi < float(parameters["upper"]), -1.0)

        reversal_return = close / close.shift(int(parameters["reversal_window"])) - 1.0
        reversal_trend_return = (
            close / close.shift(int(parameters["reversal_trend_window"])) - 1.0
        )
        reversal_component = np.sign(reversal_trend_return)
        threshold = float(parameters["reversal_threshold_pct"]) / 100.0
        reversal_component = reversal_component.where(
            reversal_return.abs() < threshold,
            -np.sign(reversal_return),
        )
        score = (
            float(parameters["rsi_weight"]) * rsi_component
            + float(parameters["reversal_weight"]) * reversal_component
        )
        return score.where(
            rsi.notna()
            & rsi_trend_return.notna()
            & reversal_return.notna()
            & reversal_trend_return.notna()
        )
    if family == "trend_guarded_dual_reversal":
        rsi_window = int(parameters["rsi_window"])
        delta = close.diff()
        gain = delta.clip(lower=0.0).ewm(
            alpha=1.0 / rsi_window,
            adjust=False,
            min_periods=rsi_window,
        ).mean()
        loss = (-delta.clip(upper=0.0)).ewm(
            alpha=1.0 / rsi_window,
            adjust=False,
            min_periods=rsi_window,
        ).mean()
        relative_strength = gain / loss.replace(0.0, np.nan)
        rsi = 100.0 - 100.0 / (1.0 + relative_strength)
        rsi = rsi.mask((loss == 0.0) & (gain > 0.0), 100.0)
        rsi = rsi.mask((gain == 0.0) & (loss > 0.0), 0.0)
        rsi = rsi.mask((gain == 0.0) & (loss == 0.0), 50.0)

        rsi_trend_return = (
            close / close.shift(int(parameters["rsi_trend_window"])) - 1.0
        )
        rsi_component = np.sign(rsi_trend_return)
        rsi_component = rsi_component.where(rsi > float(parameters["lower"]), 1.0)
        rsi_component = rsi_component.where(rsi < float(parameters["upper"]), -1.0)

        reversal_return = close / close.shift(int(parameters["reversal_window"])) - 1.0
        reversal_trend_return = (
            close / close.shift(int(parameters["reversal_trend_window"])) - 1.0
        )
        reversal_component = np.sign(reversal_trend_return)
        threshold = float(parameters["reversal_threshold_pct"]) / 100.0
        reversal_component = reversal_component.where(
            reversal_return.abs() < threshold,
            -np.sign(reversal_return),
        )

        guard_return = close / close.shift(int(parameters["guard_trend_window"])) - 1.0
        score = (
            float(parameters["rsi_weight"]) * rsi_component
            + float(parameters["reversal_weight"]) * reversal_component
            + float(parameters["guard_weight"]) * np.sign(guard_return)
        )
        return score.where(
            rsi.notna()
            & rsi_trend_return.notna()
            & reversal_return.notna()
            & reversal_trend_return.notna()
            & guard_return.notna()
        )
    if family == "volatility_regime_reversal":
        rsi_window = int(parameters["rsi_window"])
        delta = close.diff()
        gain = delta.clip(lower=0.0).ewm(
            alpha=1.0 / rsi_window,
            adjust=False,
            min_periods=rsi_window,
        ).mean()
        loss = (-delta.clip(upper=0.0)).ewm(
            alpha=1.0 / rsi_window,
            adjust=False,
            min_periods=rsi_window,
        ).mean()
        relative_strength = gain / loss.replace(0.0, np.nan)
        rsi = 100.0 - 100.0 / (1.0 + relative_strength)
        rsi = rsi.mask((loss == 0.0) & (gain > 0.0), 100.0)
        rsi = rsi.mask((gain == 0.0) & (loss > 0.0), 0.0)
        rsi = rsi.mask((gain == 0.0) & (loss == 0.0), 50.0)

        rsi_trend_return = (
            close / close.shift(int(parameters["rsi_trend_window"])) - 1.0
        )
        rsi_component = np.sign(rsi_trend_return)
        rsi_component = rsi_component.where(rsi > float(parameters["lower"]), 1.0)
        rsi_component = rsi_component.where(rsi < float(parameters["upper"]), -1.0)

        reversal_return = close / close.shift(int(parameters["reversal_window"])) - 1.0
        reversal_trend_return = (
            close / close.shift(int(parameters["reversal_trend_window"])) - 1.0
        )
        reversal_component = np.sign(reversal_trend_return)
        threshold = float(parameters["reversal_threshold_pct"]) / 100.0
        reversal_component = reversal_component.where(
            reversal_return.abs() < threshold,
            -np.sign(reversal_return),
        )
        normal_score = (
            float(parameters["rsi_weight"]) * rsi_component
            + float(parameters["reversal_weight"]) * reversal_component
        )

        volatility_window = int(parameters["volatility_window"])
        realized_volatility = log_return.rolling(
            volatility_window,
            min_periods=volatility_window,
        ).std(ddof=1) * np.sqrt(252.0)
        regime_trend_return = (
            close / close.shift(int(parameters["regime_trend_window"])) - 1.0
        )
        high_volatility = realized_volatility >= (
            float(parameters["high_volatility_pct"]) / 100.0
        )
        score = normal_score.where(~high_volatility, np.sign(regime_trend_return))
        return score.where(
            rsi.notna()
            & rsi_trend_return.notna()
            & reversal_return.notna()
            & reversal_trend_return.notna()
            & realized_volatility.notna()
            & regime_trend_return.notna()
        )
    if family == "overnight_tug_reversal_vote":
        rsi_window = int(parameters["rsi_window"])
        delta = close.diff()
        gain = delta.clip(lower=0.0).ewm(
            alpha=1.0 / rsi_window,
            adjust=False,
            min_periods=rsi_window,
        ).mean()
        loss = (-delta.clip(upper=0.0)).ewm(
            alpha=1.0 / rsi_window,
            adjust=False,
            min_periods=rsi_window,
        ).mean()
        relative_strength = gain / loss.replace(0.0, np.nan)
        rsi = 100.0 - 100.0 / (1.0 + relative_strength)
        rsi = rsi.mask((loss == 0.0) & (gain > 0.0), 100.0)
        rsi = rsi.mask((gain == 0.0) & (loss > 0.0), 0.0)
        rsi = rsi.mask((gain == 0.0) & (loss == 0.0), 50.0)

        rsi_trend_return = (
            close / close.shift(int(parameters["rsi_trend_window"])) - 1.0
        )
        rsi_component = np.sign(rsi_trend_return)
        rsi_component = rsi_component.where(rsi > float(parameters["lower"]), 1.0)
        rsi_component = rsi_component.where(rsi < float(parameters["upper"]), -1.0)

        reversal_return = close / close.shift(int(parameters["reversal_window"])) - 1.0
        reversal_trend_return = (
            close / close.shift(int(parameters["reversal_trend_window"])) - 1.0
        )
        reversal_component = np.sign(reversal_trend_return)
        threshold = float(parameters["reversal_threshold_pct"]) / 100.0
        reversal_component = reversal_component.where(
            reversal_return.abs() < threshold,
            -np.sign(reversal_return),
        )

        adjusted_open = ledger["tr_open"].astype(float)
        overnight = np.log(adjusted_open / close.shift(1))
        intraday = np.log(close / adjusted_open)
        tug = (overnight - intraday).rolling(
            int(parameters["tug_lookback"]),
            min_periods=int(parameters["tug_lookback"]),
        ).sum()
        score = (
            float(parameters["rsi_weight"]) * rsi_component
            + float(parameters["reversal_weight"]) * reversal_component
            + float(parameters["tug_weight"]) * np.sign(tug)
        )
        return score.where(
            rsi.notna()
            & rsi_trend_return.notna()
            & reversal_return.notna()
            & reversal_trend_return.notna()
            & tug.notna()
        )
    if family == "strong_trend_override_reversal":
        rsi_window = int(parameters["rsi_window"])
        delta = close.diff()
        gain = delta.clip(lower=0.0).ewm(
            alpha=1.0 / rsi_window,
            adjust=False,
            min_periods=rsi_window,
        ).mean()
        loss = (-delta.clip(upper=0.0)).ewm(
            alpha=1.0 / rsi_window,
            adjust=False,
            min_periods=rsi_window,
        ).mean()
        relative_strength = gain / loss.replace(0.0, np.nan)
        rsi = 100.0 - 100.0 / (1.0 + relative_strength)
        rsi = rsi.mask((loss == 0.0) & (gain > 0.0), 100.0)
        rsi = rsi.mask((gain == 0.0) & (loss > 0.0), 0.0)
        rsi = rsi.mask((gain == 0.0) & (loss == 0.0), 50.0)

        rsi_trend_return = (
            close / close.shift(int(parameters["rsi_trend_window"])) - 1.0
        )
        rsi_component = np.sign(rsi_trend_return)
        rsi_component = rsi_component.where(rsi > float(parameters["lower"]), 1.0)
        rsi_component = rsi_component.where(rsi < float(parameters["upper"]), -1.0)

        reversal_return = close / close.shift(int(parameters["reversal_window"])) - 1.0
        reversal_trend_return = (
            close / close.shift(int(parameters["reversal_trend_window"])) - 1.0
        )
        reversal_component = np.sign(reversal_trend_return)
        threshold = float(parameters["reversal_threshold_pct"]) / 100.0
        reversal_component = reversal_component.where(
            reversal_return.abs() < threshold,
            -np.sign(reversal_return),
        )
        score = rsi_component + reversal_component

        override_return = close / close.shift(int(parameters["override_window"])) - 1.0
        override_threshold = float(parameters["override_threshold_pct"]) / 100.0
        score = score.where(override_return <= override_threshold, 1.0)
        if str(parameters["override_mode"]) == "symmetric":
            score = score.where(override_return >= -override_threshold, -1.0)
        return score.where(
            rsi.notna()
            & rsi_trend_return.notna()
            & reversal_return.notna()
            & reversal_trend_return.notna()
            & override_return.notna()
        )
    if family == "asymmetric_trend_override_reversal":
        rsi_window = int(parameters["rsi_window"])
        delta = close.diff()
        gain = delta.clip(lower=0.0).ewm(
            alpha=1.0 / rsi_window,
            adjust=False,
            min_periods=rsi_window,
        ).mean()
        loss = (-delta.clip(upper=0.0)).ewm(
            alpha=1.0 / rsi_window,
            adjust=False,
            min_periods=rsi_window,
        ).mean()
        relative_strength = gain / loss.replace(0.0, np.nan)
        rsi = 100.0 - 100.0 / (1.0 + relative_strength)
        rsi = rsi.mask((loss == 0.0) & (gain > 0.0), 100.0)
        rsi = rsi.mask((gain == 0.0) & (loss > 0.0), 0.0)
        rsi = rsi.mask((gain == 0.0) & (loss == 0.0), 50.0)

        rsi_trend_return = (
            close / close.shift(int(parameters["rsi_trend_window"])) - 1.0
        )
        rsi_component = np.sign(rsi_trend_return)
        rsi_component = rsi_component.where(
            rsi > float(parameters["lower"]), 1.0
        )
        rsi_component = rsi_component.where(
            rsi < float(parameters["upper"]), -1.0
        )

        reversal_return = (
            close / close.shift(int(parameters["reversal_window"])) - 1.0
        )
        reversal_trend_return = (
            close / close.shift(int(parameters["reversal_trend_window"])) - 1.0
        )
        reversal_component = np.sign(reversal_trend_return)
        reversal_threshold = (
            float(parameters["reversal_threshold_pct"]) / 100.0
        )
        reversal_component = reversal_component.where(
            reversal_return.abs() < reversal_threshold,
            -np.sign(reversal_return),
        )
        score = rsi_component + reversal_component

        positive_override_return = (
            close / close.shift(int(parameters["positive_override_window"]))
            - 1.0
        )
        negative_override_return = (
            close / close.shift(int(parameters["negative_override_window"]))
            - 1.0
        )
        positive_override = positive_override_return > (
            float(parameters["positive_override_threshold_pct"]) / 100.0
        )
        negative_override = negative_override_return < -(
            float(parameters["negative_override_threshold_pct"]) / 100.0
        )
        score = score.where(~negative_override, -1.0)
        # Positive trend is the deterministic tie-break if both overrides fire.
        score = score.where(~positive_override, 1.0)
        return score.where(
            rsi.notna()
            & rsi_trend_return.notna()
            & reversal_return.notna()
            & reversal_trend_return.notna()
            & positive_override_return.notna()
            & negative_override_return.notna()
        )
    if family == "drawdown_recovery_override_reversal":
        rsi_window = int(parameters["rsi_window"])
        delta = close.diff()
        gain = delta.clip(lower=0.0).ewm(
            alpha=1.0 / rsi_window,
            adjust=False,
            min_periods=rsi_window,
        ).mean()
        loss = (-delta.clip(upper=0.0)).ewm(
            alpha=1.0 / rsi_window,
            adjust=False,
            min_periods=rsi_window,
        ).mean()
        relative_strength = gain / loss.replace(0.0, np.nan)
        rsi = 100.0 - 100.0 / (1.0 + relative_strength)
        rsi = rsi.mask((loss == 0.0) & (gain > 0.0), 100.0)
        rsi = rsi.mask((gain == 0.0) & (loss > 0.0), 0.0)
        rsi = rsi.mask((gain == 0.0) & (loss == 0.0), 50.0)

        rsi_trend_return = (
            close / close.shift(int(parameters["rsi_trend_window"])) - 1.0
        )
        rsi_component = np.sign(rsi_trend_return)
        rsi_component = rsi_component.where(
            rsi > float(parameters["lower"]), 1.0
        )
        rsi_component = rsi_component.where(
            rsi < float(parameters["upper"]), -1.0
        )

        reversal_return = (
            close / close.shift(int(parameters["reversal_window"])) - 1.0
        )
        reversal_trend_return = (
            close / close.shift(int(parameters["reversal_trend_window"])) - 1.0
        )
        reversal_component = np.sign(reversal_trend_return)
        reversal_threshold = (
            float(parameters["reversal_threshold_pct"]) / 100.0
        )
        reversal_component = reversal_component.where(
            reversal_return.abs() < reversal_threshold,
            -np.sign(reversal_return),
        )
        score = (
            float(parameters["rsi_weight"]) * rsi_component
            + float(parameters["reversal_weight"]) * reversal_component
        )

        drawdown_lookback = int(parameters["drawdown_lookback"])
        rolling_peak = close.rolling(
            drawdown_lookback,
            min_periods=drawdown_lookback,
        ).max()
        drawdown = close / rolling_peak - 1.0
        recent_deep_drawdown = drawdown.rolling(
            int(parameters["recovery_memory_window"]),
            min_periods=1,
        ).min() <= -(float(parameters["drawdown_trigger_pct"]) / 100.0)
        recovery_return = (
            close / close.shift(int(parameters["recovery_window"])) - 1.0
        )
        recovery_override = recent_deep_drawdown & (
            recovery_return
            > float(parameters["recovery_threshold_pct"]) / 100.0
        )
        score = score.where(~recovery_override, 1.0)
        return score.where(
            rsi.notna()
            & rsi_trend_return.notna()
            & reversal_return.notna()
            & reversal_trend_return.notna()
            & rolling_peak.notna()
            & recovery_return.notna()
        )
    if family == "quiet_bull_recovery_override_reversal":
        rsi_window = int(parameters["rsi_window"])
        delta = close.diff()
        gain = delta.clip(lower=0.0).ewm(
            alpha=1.0 / rsi_window,
            adjust=False,
            min_periods=rsi_window,
        ).mean()
        loss = (-delta.clip(upper=0.0)).ewm(
            alpha=1.0 / rsi_window,
            adjust=False,
            min_periods=rsi_window,
        ).mean()
        relative_strength = gain / loss.replace(0.0, np.nan)
        rsi = 100.0 - 100.0 / (1.0 + relative_strength)
        rsi = rsi.mask((loss == 0.0) & (gain > 0.0), 100.0)
        rsi = rsi.mask((gain == 0.0) & (loss > 0.0), 0.0)
        rsi = rsi.mask((gain == 0.0) & (loss == 0.0), 50.0)
        rsi_trend_return = (
            close / close.shift(int(parameters["rsi_trend_window"])) - 1.0
        )
        rsi_component = np.sign(rsi_trend_return)
        rsi_component = rsi_component.where(
            rsi > float(parameters["lower"]), 1.0
        )
        rsi_component = rsi_component.where(
            rsi < float(parameters["upper"]), -1.0
        )

        reversal_return = (
            close / close.shift(int(parameters["reversal_window"])) - 1.0
        )
        reversal_trend_return = (
            close / close.shift(int(parameters["reversal_trend_window"])) - 1.0
        )
        reversal_component = np.sign(reversal_trend_return)
        reversal_threshold = (
            float(parameters["reversal_threshold_pct"]) / 100.0
        )
        reversal_component = reversal_component.where(
            reversal_return.abs() < reversal_threshold,
            -np.sign(reversal_return),
        )
        score = rsi_component + reversal_component

        drawdown_lookback = int(parameters["drawdown_lookback"])
        rolling_peak = close.rolling(
            drawdown_lookback,
            min_periods=drawdown_lookback,
        ).max()
        drawdown = close / rolling_peak - 1.0
        recent_deep_drawdown = drawdown.rolling(
            int(parameters["recovery_memory_window"]),
            min_periods=1,
        ).min() <= -(float(parameters["drawdown_trigger_pct"]) / 100.0)
        recovery_return = (
            close / close.shift(int(parameters["recovery_window"])) - 1.0
        )
        recovery_override = recent_deep_drawdown & (
            recovery_return
            > float(parameters["recovery_threshold_pct"]) / 100.0
        )

        bull_ma_window = int(parameters["bull_ma_window"])
        bull_average = close.rolling(
            bull_ma_window,
            min_periods=bull_ma_window,
        ).mean()
        bull_slope = (
            bull_average
            / bull_average.shift(int(parameters["bull_slope_window"]))
            - 1.0
        )
        bull_return = close / close.shift(bull_ma_window) - 1.0
        realized_volatility = log_return.rolling(20, min_periods=20).std(
            ddof=1
        ) * np.sqrt(252.0)
        quiet_bull_override = (
            (close > bull_average)
            & (bull_slope > 0.0)
            & (
                bull_return
                > float(parameters["bull_min_return_pct"]) / 100.0
            )
            & (
                realized_volatility
                < float(parameters["bull_max_volatility_pct"]) / 100.0
            )
        )
        score = score.where(~(recovery_override | quiet_bull_override), 1.0)
        return score.where(
            rsi.notna()
            & rsi_trend_return.notna()
            & reversal_return.notna()
            & reversal_trend_return.notna()
            & rolling_peak.notna()
            & recovery_return.notna()
            & bull_average.notna()
            & bull_slope.notna()
            & bull_return.notna()
            & realized_volatility.notna()
        )
    if family == "recovery_trend_breakout_majority":
        rsi_window = int(parameters["rsi_window"])
        delta = close.diff()
        gain = delta.clip(lower=0.0).ewm(
            alpha=1.0 / rsi_window,
            adjust=False,
            min_periods=rsi_window,
        ).mean()
        loss = (-delta.clip(upper=0.0)).ewm(
            alpha=1.0 / rsi_window,
            adjust=False,
            min_periods=rsi_window,
        ).mean()
        relative_strength = gain / loss.replace(0.0, np.nan)
        rsi = 100.0 - 100.0 / (1.0 + relative_strength)
        rsi = rsi.mask((loss == 0.0) & (gain > 0.0), 100.0)
        rsi = rsi.mask((gain == 0.0) & (loss > 0.0), 0.0)
        rsi = rsi.mask((gain == 0.0) & (loss == 0.0), 50.0)

        rsi_trend_return = (
            close / close.shift(int(parameters["rsi_trend_window"])) - 1.0
        )
        rsi_component = np.sign(rsi_trend_return)
        rsi_component = rsi_component.where(
            rsi > float(parameters["lower"]), 1.0
        )
        rsi_component = rsi_component.where(
            rsi < float(parameters["upper"]), -1.0
        )
        reversal_return = (
            close / close.shift(int(parameters["reversal_window"])) - 1.0
        )
        reversal_trend_return = (
            close / close.shift(int(parameters["reversal_trend_window"])) - 1.0
        )
        reversal_component = np.sign(reversal_trend_return)
        reversal_threshold = (
            float(parameters["reversal_threshold_pct"]) / 100.0
        )
        reversal_component = reversal_component.where(
            reversal_return.abs() < reversal_threshold,
            -np.sign(reversal_return),
        )
        recovery_score = rsi_component + reversal_component

        drawdown_lookback = int(parameters["drawdown_lookback"])
        rolling_peak = close.rolling(
            drawdown_lookback,
            min_periods=drawdown_lookback,
        ).max()
        drawdown = close / rolling_peak - 1.0
        recent_deep_drawdown = drawdown.rolling(
            int(parameters["recovery_memory_window"]),
            min_periods=1,
        ).min() <= -(float(parameters["drawdown_trigger_pct"]) / 100.0)
        recovery_return = (
            close / close.shift(int(parameters["recovery_window"])) - 1.0
        )
        recovery_override = recent_deep_drawdown & (
            recovery_return
            > float(parameters["recovery_threshold_pct"]) / 100.0
        )
        recovery_score = recovery_score.where(~recovery_override, 1.0)
        recovery_valid = (
            rsi.notna()
            & rsi_trend_return.notna()
            & reversal_return.notna()
            & reversal_trend_return.notna()
            & rolling_peak.notna()
            & recovery_return.notna()
        )
        recovery_vote = _state_from_score(recovery_score.where(recovery_valid))

        trend_components = [
            np.sign(close / close.shift(int(horizon)) - 1.0)
            for horizon in parameters["trend_horizons"]
        ]
        trend_score = pd.concat(trend_components, axis=1).sum(
            axis=1,
            min_count=len(trend_components),
        )
        trend_vote = _state_from_score(trend_score)

        breakout_window = int(parameters["breakout_window"])
        prior_high = close.shift(1).rolling(
            breakout_window,
            min_periods=breakout_window,
        ).max()
        prior_low = close.shift(1).rolling(
            breakout_window,
            min_periods=breakout_window,
        ).min()
        breakout_valid = prior_high.notna() & prior_low.notna()
        breakout_vote = _state_from_events(
            pd.DatetimeIndex(close.index),
            breakout_valid & (close > prior_high),
            breakout_valid & (close < prior_low),
            eligible=breakout_valid,
        )

        valid = recovery_valid & trend_score.notna() & breakout_valid
        majority_score = recovery_vote + trend_vote + breakout_vote
        if (majority_score.loc[valid] == 0).any():
            raise CandidateRejected("EVEN_ENSEMBLE_VOTE")
        return majority_score.astype(float).where(valid)
    if family == "high_vol_crash_recovery_reversal":
        rsi_window = int(parameters["rsi_window"])
        delta = close.diff()
        gain = delta.clip(lower=0.0).ewm(
            alpha=1.0 / rsi_window,
            adjust=False,
            min_periods=rsi_window,
        ).mean()
        loss = (-delta.clip(upper=0.0)).ewm(
            alpha=1.0 / rsi_window,
            adjust=False,
            min_periods=rsi_window,
        ).mean()
        relative_strength = gain / loss.replace(0.0, np.nan)
        rsi = 100.0 - 100.0 / (1.0 + relative_strength)
        rsi = rsi.mask((loss == 0.0) & (gain > 0.0), 100.0)
        rsi = rsi.mask((gain == 0.0) & (loss > 0.0), 0.0)
        rsi = rsi.mask((gain == 0.0) & (loss == 0.0), 50.0)

        rsi_trend_return = (
            close / close.shift(int(parameters["rsi_trend_window"])) - 1.0
        )
        rsi_component = np.sign(rsi_trend_return)
        rsi_component = rsi_component.where(
            rsi > float(parameters["lower"]), 1.0
        )
        rsi_component = rsi_component.where(
            rsi < float(parameters["upper"]), -1.0
        )
        reversal_return = (
            close / close.shift(int(parameters["reversal_window"])) - 1.0
        )
        reversal_trend_return = (
            close / close.shift(int(parameters["reversal_trend_window"])) - 1.0
        )
        reversal_component = np.sign(reversal_trend_return)
        reversal_threshold = (
            float(parameters["reversal_threshold_pct"]) / 100.0
        )
        reversal_component = reversal_component.where(
            reversal_return.abs() < reversal_threshold,
            -np.sign(reversal_return),
        )
        score = rsi_component + reversal_component

        volatility_window = int(parameters["volatility_window"])
        realized_volatility = log_return.rolling(
            volatility_window,
            min_periods=volatility_window,
        ).std(ddof=1) * np.sqrt(252.0)
        crash_return = (
            close / close.shift(int(parameters["crash_window"])) - 1.0
        )
        crash_override = (
            realized_volatility
            >= float(parameters["high_volatility_pct"]) / 100.0
        ) & (
            crash_return
            <= -(float(parameters["crash_threshold_pct"]) / 100.0)
        )
        score = score.where(~crash_override, -1.0)

        drawdown_lookback = int(parameters["drawdown_lookback"])
        rolling_peak = close.rolling(
            drawdown_lookback,
            min_periods=drawdown_lookback,
        ).max()
        drawdown = close / rolling_peak - 1.0
        recent_deep_drawdown = drawdown.rolling(
            int(parameters["recovery_memory_window"]),
            min_periods=1,
        ).min() <= -(float(parameters["drawdown_trigger_pct"]) / 100.0)
        recovery_return = (
            close / close.shift(int(parameters["recovery_window"])) - 1.0
        )
        recovery_override = recent_deep_drawdown & (
            recovery_return
            > float(parameters["recovery_threshold_pct"]) / 100.0
        )
        # A confirmed recovery is the deterministic final tie-break.
        score = score.where(~recovery_override, 1.0)
        return score.where(
            rsi.notna()
            & rsi_trend_return.notna()
            & reversal_return.notna()
            & reversal_trend_return.notna()
            & realized_volatility.notna()
            & crash_return.notna()
            & rolling_peak.notna()
            & recovery_return.notna()
        )
    if family == "adaptive_recovery_edge_switch":
        rsi_window = int(parameters["rsi_window"])
        delta = close.diff()
        gain = delta.clip(lower=0.0).ewm(
            alpha=1.0 / rsi_window,
            adjust=False,
            min_periods=rsi_window,
        ).mean()
        loss = (-delta.clip(upper=0.0)).ewm(
            alpha=1.0 / rsi_window,
            adjust=False,
            min_periods=rsi_window,
        ).mean()
        relative_strength = gain / loss.replace(0.0, np.nan)
        rsi = 100.0 - 100.0 / (1.0 + relative_strength)
        rsi = rsi.mask((loss == 0.0) & (gain > 0.0), 100.0)
        rsi = rsi.mask((gain == 0.0) & (loss > 0.0), 0.0)
        rsi = rsi.mask((gain == 0.0) & (loss == 0.0), 50.0)

        rsi_trend_return = (
            close / close.shift(int(parameters["rsi_trend_window"])) - 1.0
        )
        rsi_component = np.sign(rsi_trend_return)
        rsi_component = rsi_component.where(
            rsi > float(parameters["lower"]), 1.0
        )
        rsi_component = rsi_component.where(
            rsi < float(parameters["upper"]), -1.0
        )
        reversal_return = (
            close / close.shift(int(parameters["reversal_window"])) - 1.0
        )
        reversal_trend_return = (
            close / close.shift(int(parameters["reversal_trend_window"])) - 1.0
        )
        reversal_component = np.sign(reversal_trend_return)
        reversal_threshold = (
            float(parameters["reversal_threshold_pct"]) / 100.0
        )
        reversal_component = reversal_component.where(
            reversal_return.abs() < reversal_threshold,
            -np.sign(reversal_return),
        )
        base_score = rsi_component + reversal_component

        drawdown_lookback = int(parameters["drawdown_lookback"])
        rolling_peak = close.rolling(
            drawdown_lookback,
            min_periods=drawdown_lookback,
        ).max()
        drawdown = close / rolling_peak - 1.0
        recent_deep_drawdown = drawdown.rolling(
            int(parameters["recovery_memory_window"]),
            min_periods=1,
        ).min() <= -(float(parameters["drawdown_trigger_pct"]) / 100.0)
        recovery_return = (
            close / close.shift(int(parameters["recovery_window"])) - 1.0
        )
        recovery_override = recent_deep_drawdown & (
            recovery_return
            > float(parameters["recovery_threshold_pct"]) / 100.0
        )
        base_score = base_score.where(~recovery_override, 1.0)
        base_valid = (
            rsi.notna()
            & rsi_trend_return.notna()
            & reversal_return.notna()
            & reversal_trend_return.notna()
            & rolling_peak.notna()
            & recovery_return.notna()
        )
        base_decision = _state_from_score(base_score.where(base_valid))

        # long_return on row t contains open(t)->open(t+1), so it is only
        # observable from row t+1. The extra shift keeps the meta-rule causal.
        base_position = base_decision.shift(1).ffill().fillna(1.0)
        base_realized_return = (
            base_position.astype(float) * ledger["long_return"].astype(float)
        ).shift(1)
        edge_window = int(parameters["edge_window"])
        rolling_edge = np.expm1(
            np.log1p(base_realized_return.clip(lower=-0.999999)).rolling(
                edge_window,
                min_periods=edge_window,
            ).sum()
        )
        invert = rolling_edge < (
            float(parameters["edge_threshold_pct"]) / 100.0
        )
        adaptive_decision = base_decision.where(~invert, -base_decision)
        return adaptive_decision.astype(float).where(base_valid & rolling_edge.notna())
    if family == "recovery_overnight_tug_vote":
        rsi_window = int(parameters["rsi_window"])
        delta = close.diff()
        gain = delta.clip(lower=0.0).ewm(
            alpha=1.0 / rsi_window,
            adjust=False,
            min_periods=rsi_window,
        ).mean()
        loss = (-delta.clip(upper=0.0)).ewm(
            alpha=1.0 / rsi_window,
            adjust=False,
            min_periods=rsi_window,
        ).mean()
        relative_strength = gain / loss.replace(0.0, np.nan)
        rsi = 100.0 - 100.0 / (1.0 + relative_strength)
        rsi = rsi.mask((loss == 0.0) & (gain > 0.0), 100.0)
        rsi = rsi.mask((gain == 0.0) & (loss > 0.0), 0.0)
        rsi = rsi.mask((gain == 0.0) & (loss == 0.0), 50.0)

        rsi_trend_return = (
            close / close.shift(int(parameters["rsi_trend_window"])) - 1.0
        )
        rsi_component = np.sign(rsi_trend_return)
        rsi_component = rsi_component.where(
            rsi > float(parameters["lower"]), 1.0
        )
        rsi_component = rsi_component.where(
            rsi < float(parameters["upper"]), -1.0
        )
        reversal_return = (
            close / close.shift(int(parameters["reversal_window"])) - 1.0
        )
        reversal_trend_return = (
            close / close.shift(int(parameters["reversal_trend_window"])) - 1.0
        )
        reversal_component = np.sign(reversal_trend_return)
        reversal_threshold = (
            float(parameters["reversal_threshold_pct"]) / 100.0
        )
        reversal_component = reversal_component.where(
            reversal_return.abs() < reversal_threshold,
            -np.sign(reversal_return),
        )

        adjusted_open = ledger["tr_open"].astype(float)
        overnight = np.log(adjusted_open / close.shift(1))
        intraday = np.log(close / adjusted_open)
        tug = (overnight - intraday).rolling(
            int(parameters["tug_lookback"]),
            min_periods=int(parameters["tug_lookback"]),
        ).sum()
        score = (
            rsi_component
            + reversal_component
            + float(parameters["tug_weight"]) * np.sign(tug)
        )

        drawdown_lookback = int(parameters["drawdown_lookback"])
        rolling_peak = close.rolling(
            drawdown_lookback,
            min_periods=drawdown_lookback,
        ).max()
        drawdown = close / rolling_peak - 1.0
        recent_deep_drawdown = drawdown.rolling(
            int(parameters["recovery_memory_window"]),
            min_periods=1,
        ).min() <= -(float(parameters["drawdown_trigger_pct"]) / 100.0)
        recovery_return = (
            close / close.shift(int(parameters["recovery_window"])) - 1.0
        )
        recovery_override = recent_deep_drawdown & (
            recovery_return
            > float(parameters["recovery_threshold_pct"]) / 100.0
        )
        score = score.where(~recovery_override, 1.0)
        return score.where(
            rsi.notna()
            & rsi_trend_return.notna()
            & reversal_return.notna()
            & reversal_trend_return.notna()
            & tug.notna()
            & rolling_peak.notna()
            & recovery_return.notna()
        )
    if family == "recovery_turn_month_vote":
        rsi_window = int(parameters["rsi_window"])
        delta = close.diff()
        gain = delta.clip(lower=0.0).ewm(
            alpha=1.0 / rsi_window,
            adjust=False,
            min_periods=rsi_window,
        ).mean()
        loss = (-delta.clip(upper=0.0)).ewm(
            alpha=1.0 / rsi_window,
            adjust=False,
            min_periods=rsi_window,
        ).mean()
        relative_strength = gain / loss.replace(0.0, np.nan)
        rsi = 100.0 - 100.0 / (1.0 + relative_strength)
        rsi = rsi.mask((loss == 0.0) & (gain > 0.0), 100.0)
        rsi = rsi.mask((gain == 0.0) & (loss > 0.0), 0.0)
        rsi = rsi.mask((gain == 0.0) & (loss == 0.0), 50.0)

        rsi_trend_return = (
            close / close.shift(int(parameters["rsi_trend_window"])) - 1.0
        )
        rsi_component = np.sign(rsi_trend_return)
        rsi_component = rsi_component.where(
            rsi > float(parameters["lower"]), 1.0
        )
        rsi_component = rsi_component.where(
            rsi < float(parameters["upper"]), -1.0
        )
        reversal_return = (
            close / close.shift(int(parameters["reversal_window"])) - 1.0
        )
        reversal_trend_return = (
            close / close.shift(int(parameters["reversal_trend_window"])) - 1.0
        )
        reversal_component = np.sign(reversal_trend_return)
        reversal_threshold = (
            float(parameters["reversal_threshold_pct"]) / 100.0
        )
        reversal_component = reversal_component.where(
            reversal_return.abs() < reversal_threshold,
            -np.sign(reversal_return),
        )

        session_frame = pd.DataFrame(index=ledger.index)
        session_frame["month"] = ledger.index.to_period("M")
        session_frame["from_start"] = (
            session_frame.groupby("month").cumcount() + 1
        )
        session_frame["from_end"] = (
            session_frame.groupby("month").cumcount(ascending=False) + 1
        )
        next_position = session_frame[["from_start", "from_end"]].shift(-1)
        turn_month = (
            next_position["from_start"] <= int(parameters["first_sessions"])
        ) | (next_position["from_end"] <= int(parameters["last_sessions"]))
        score = (
            rsi_component
            + reversal_component
            + float(parameters["calendar_weight"]) * turn_month.astype(float)
        )

        drawdown_lookback = int(parameters["drawdown_lookback"])
        rolling_peak = close.rolling(
            drawdown_lookback,
            min_periods=drawdown_lookback,
        ).max()
        drawdown = close / rolling_peak - 1.0
        recent_deep_drawdown = drawdown.rolling(
            int(parameters["recovery_memory_window"]),
            min_periods=1,
        ).min() <= -(float(parameters["drawdown_trigger_pct"]) / 100.0)
        recovery_return = (
            close / close.shift(int(parameters["recovery_window"])) - 1.0
        )
        recovery_override = recent_deep_drawdown & (
            recovery_return
            > float(parameters["recovery_threshold_pct"]) / 100.0
        )
        score = score.where(~recovery_override, 1.0)
        return score.where(
            rsi.notna()
            & rsi_trend_return.notna()
            & reversal_return.notna()
            & reversal_trend_return.notna()
            & rolling_peak.notna()
            & recovery_return.notna()
        )
    if family == "recovery_internal_bar_strength_vote":
        base_score = _price_score(
            {
                "family": "drawdown_recovery_override_reversal",
                "parameters": parameters,
            },
            data,
            feature_frame=feature_frame,
        )
        high = ledger["high"].astype(float)
        low = ledger["low"].astype(float)
        ibs = (close - low) / (high - low).replace(0.0, np.nan)
        smooth_window = int(parameters["ibs_smooth_window"])
        smoothed_ibs = ibs.rolling(
            smooth_window,
            min_periods=smooth_window,
        ).mean()
        ibs_component = pd.Series(0.0, index=ledger.index)
        ibs_component = ibs_component.where(
            smoothed_ibs > float(parameters["ibs_lower"]), 1.0
        )
        ibs_component = ibs_component.where(
            smoothed_ibs < float(parameters["ibs_upper"]), -1.0
        )
        score = base_score + float(parameters["ibs_weight"]) * ibs_component

        drawdown_lookback = int(parameters["drawdown_lookback"])
        rolling_peak = close.rolling(
            drawdown_lookback,
            min_periods=drawdown_lookback,
        ).max()
        drawdown = close / rolling_peak - 1.0
        recent_deep_drawdown = drawdown.rolling(
            int(parameters["recovery_memory_window"]),
            min_periods=1,
        ).min() <= -(float(parameters["drawdown_trigger_pct"]) / 100.0)
        recovery_return = (
            close / close.shift(int(parameters["recovery_window"])) - 1.0
        )
        recovery_override = recent_deep_drawdown & (
            recovery_return
            > float(parameters["recovery_threshold_pct"]) / 100.0
        )
        score = score.where(~recovery_override, 1.0)
        return score.where(base_score.notna() & smoothed_ibs.notna())
    if family == "recovery_multi_horizon_reversal_vote":
        base_score = _price_score(
            {
                "family": "drawdown_recovery_override_reversal",
                "parameters": parameters,
            },
            data,
            feature_frame=feature_frame,
        )
        horizon_returns = [
            close / close.shift(int(horizon)) - 1.0
            for horizon in parameters["multi_horizons"]
        ]
        mean_horizon_return = pd.concat(horizon_returns, axis=1).mean(
            axis=1,
            skipna=False,
        )
        threshold = float(parameters["multi_threshold_pct"]) / 100.0
        multi_component = pd.Series(0.0, index=ledger.index)
        multi_component = multi_component.where(
            mean_horizon_return > -threshold, 1.0
        )
        multi_component = multi_component.where(
            mean_horizon_return < threshold, -1.0
        )
        score = (
            base_score
            + float(parameters["multi_reversal_weight"]) * multi_component
        )

        drawdown_lookback = int(parameters["drawdown_lookback"])
        rolling_peak = close.rolling(
            drawdown_lookback,
            min_periods=drawdown_lookback,
        ).max()
        drawdown = close / rolling_peak - 1.0
        recent_deep_drawdown = drawdown.rolling(
            int(parameters["recovery_memory_window"]),
            min_periods=1,
        ).min() <= -(float(parameters["drawdown_trigger_pct"]) / 100.0)
        recovery_return = (
            close / close.shift(int(parameters["recovery_window"])) - 1.0
        )
        recovery_override = recent_deep_drawdown & (
            recovery_return
            > float(parameters["recovery_threshold_pct"]) / 100.0
        )
        score = score.where(~recovery_override, 1.0)
        return score.where(base_score.notna() & mean_horizon_return.notna())
    if family == "recovery_volume_gated_reversal_vote":
        base_score = _price_score(
            {
                "family": "drawdown_recovery_override_reversal",
                "parameters": parameters,
            },
            data,
            feature_frame=feature_frame,
        )
        volume = ledger["volume"].astype(float)
        volume_z = _rolling_zscore(
            np.log1p(volume),
            int(parameters["volume_window"]),
        )
        lag_return = (
            close / close.shift(int(parameters["volume_return_lookback"])) - 1.0
        )
        volume_component = pd.Series(0.0, index=ledger.index)
        high_volume = volume_z >= float(parameters["volume_z_threshold"])
        volume_component = volume_component.where(
            ~(high_volume & (lag_return < 0.0)), 1.0
        )
        volume_component = volume_component.where(
            ~(high_volume & (lag_return > 0.0)), -1.0
        )
        score = (
            base_score
            + float(parameters["volume_reversal_weight"]) * volume_component
        )

        drawdown_lookback = int(parameters["drawdown_lookback"])
        rolling_peak = close.rolling(
            drawdown_lookback,
            min_periods=drawdown_lookback,
        ).max()
        drawdown = close / rolling_peak - 1.0
        recent_deep_drawdown = drawdown.rolling(
            int(parameters["recovery_memory_window"]),
            min_periods=1,
        ).min() <= -(float(parameters["drawdown_trigger_pct"]) / 100.0)
        recovery_return = (
            close / close.shift(int(parameters["recovery_window"])) - 1.0
        )
        recovery_override = recent_deep_drawdown & (
            recovery_return
            > float(parameters["recovery_threshold_pct"]) / 100.0
        )
        score = score.where(~recovery_override, 1.0)
        return score.where(
            base_score.notna() & volume_z.notna() & lag_return.notna()
        )
    if family == "recovery_calendar_volume_reversal_vote":
        base_score = _price_score(
            {
                "family": "drawdown_recovery_override_reversal",
                "parameters": parameters,
            },
            data,
            feature_frame=feature_frame,
        )
        volume = ledger["volume"].astype(float)
        volume_z = _rolling_zscore(
            np.log1p(volume),
            int(parameters["volume_window"]),
        )
        lag_return = (
            close / close.shift(int(parameters["volume_return_lookback"])) - 1.0
        )
        volume_component = pd.Series(0.0, index=ledger.index)
        high_volume = volume_z >= float(parameters["volume_z_threshold"])
        volume_component = volume_component.where(
            ~(high_volume & (lag_return < 0.0)), 1.0
        )
        volume_component = volume_component.where(
            ~(high_volume & (lag_return > 0.0)), -1.0
        )

        session_frame = pd.DataFrame(index=ledger.index)
        session_frame["month"] = ledger.index.to_period("M")
        session_frame["from_start"] = (
            session_frame.groupby("month").cumcount() + 1
        )
        session_frame["from_end"] = (
            session_frame.groupby("month").cumcount(ascending=False) + 1
        )
        next_position = session_frame[["from_start", "from_end"]].shift(-1)
        turn_month = (
            next_position["from_start"] <= int(parameters["first_sessions"])
        ) | (next_position["from_end"] <= int(parameters["last_sessions"]))
        score = (
            base_score
            + float(parameters["calendar_weight"]) * turn_month.astype(float)
            + float(parameters["volume_reversal_weight"]) * volume_component
        )

        drawdown_lookback = int(parameters["drawdown_lookback"])
        rolling_peak = close.rolling(
            drawdown_lookback,
            min_periods=drawdown_lookback,
        ).max()
        drawdown = close / rolling_peak - 1.0
        recent_deep_drawdown = drawdown.rolling(
            int(parameters["recovery_memory_window"]),
            min_periods=1,
        ).min() <= -(float(parameters["drawdown_trigger_pct"]) / 100.0)
        recovery_return = (
            close / close.shift(int(parameters["recovery_window"])) - 1.0
        )
        recovery_override = recent_deep_drawdown & (
            recovery_return
            > float(parameters["recovery_threshold_pct"]) / 100.0
        )
        score = score.where(~recovery_override, 1.0)
        return score.where(
            base_score.notna() & volume_z.notna() & lag_return.notna()
        )
    if family == "recovery_calendar_volume_vxo_vote":
        base_score = _price_score(
            {
                "family": "recovery_calendar_volume_reversal_vote",
                "parameters": parameters,
            },
            data,
            feature_frame=feature_frame,
        )
        vxo = _series(data, "VXO").astype(float)
        vxo_change = np.log(
            vxo / vxo.shift(int(parameters["vxo_change_lookback"]))
        )
        vxo_z = _rolling_zscore(
            vxo_change,
            int(parameters["vxo_z_window"]),
        )
        active = vxo_z.abs() >= float(parameters["vxo_z_threshold"])
        direction = np.sign(vxo_change)
        if str(parameters["vxo_mode"]) == "reversal":
            direction = -direction
        vxo_component = direction.where(active, 0.0)
        vxo_gate_valid = pd.Series(True, index=ledger.index)
        if "vxo_negative_gate_window" in parameters:
            gate_return = (
                close / close.shift(int(parameters["vxo_negative_gate_window"])) - 1.0
            )
            gate_threshold = (
                float(parameters["vxo_negative_gate_threshold_pct"]) / 100.0
            )
            vxo_component = vxo_component.where(
                (vxo_component >= 0.0) | (gate_return <= gate_threshold),
                0.0,
            )
            vxo_gate_valid = gate_return.notna()
        if "vxo_positive_weight" in parameters or "vxo_negative_weight" in parameters:
            positive_weight = float(
                parameters.get("vxo_positive_weight", parameters["vxo_weight"])
            )
            negative_weight = float(
                parameters.get("vxo_negative_weight", parameters["vxo_weight"])
            )
            weighted_vxo_component = (
                vxo_component.clip(lower=0.0) * positive_weight
                + vxo_component.clip(upper=0.0) * negative_weight
            )
            score = base_score + weighted_vxo_component
        else:
            score = base_score + float(parameters["vxo_weight"]) * vxo_component

        slow_trend_valid = pd.Series(True, index=ledger.index)
        if "slow_trend_weight" in parameters:
            slow_trend_return = (
                close / close.shift(int(parameters["slow_trend_window"])) - 1.0
            )
            score = score + float(parameters["slow_trend_weight"]) * np.sign(
                slow_trend_return
            )
            slow_trend_valid = slow_trend_return.notna()

        tug_valid = pd.Series(True, index=ledger.index)
        if "tug_weight" in parameters:
            adjusted_open = ledger["tr_open"].astype(float)
            overnight = np.log(adjusted_open / close.shift(1))
            intraday = np.log(close / adjusted_open)
            tug = (overnight - intraday).rolling(
                int(parameters["tug_lookback"]),
                min_periods=int(parameters["tug_lookback"]),
            ).sum()
            score = score + float(parameters["tug_weight"]) * np.sign(tug)
            tug_valid = tug.notna()

        rolling_peak = close.rolling(
            int(parameters["drawdown_lookback"]),
            min_periods=int(parameters["drawdown_lookback"]),
        ).max()
        drawdown = close / rolling_peak - 1.0
        recent_deep_drawdown = drawdown.rolling(
            int(parameters["recovery_memory_window"]),
            min_periods=1,
        ).min() <= -(float(parameters["drawdown_trigger_pct"]) / 100.0)
        recovery_return = (
            close / close.shift(int(parameters["recovery_window"])) - 1.0
        )
        recovery_override = recent_deep_drawdown & (
            recovery_return
            > float(parameters["recovery_threshold_pct"]) / 100.0
        )
        score = score.where(~recovery_override, 1.0)

        short_gate_valid = pd.Series(True, index=ledger.index)
        if "short_gate_window" in parameters:
            short_gate_return = (
                close / close.shift(int(parameters["short_gate_window"])) - 1.0
            )
            short_gate_threshold = (
                float(parameters["short_gate_threshold_pct"]) / 100.0
            )
            score = score.where(
                (score >= 0.0) | (short_gate_return <= short_gate_threshold),
                1.0,
            )
            short_gate_valid = short_gate_return.notna()

        short_drawdown_gate_valid = pd.Series(True, index=ledger.index)
        if "short_drawdown_gate_window" in parameters:
            short_drawdown_peak = close.rolling(
                int(parameters["short_drawdown_gate_window"]),
                min_periods=int(parameters["short_drawdown_gate_window"]),
            ).max()
            short_drawdown = close / short_drawdown_peak - 1.0
            short_drawdown_threshold = (
                -float(parameters["short_drawdown_gate_threshold_pct"]) / 100.0
            )
            score = score.where(
                (score >= 0.0) | (short_drawdown <= short_drawdown_threshold),
                1.0,
            )
            short_drawdown_gate_valid = short_drawdown.notna()
        return score.where(
            base_score.notna()
            & vxo_z.notna()
            & vxo_gate_valid
            & slow_trend_valid
            & tug_valid
            & short_gate_valid
            & short_drawdown_gate_valid
        )
    if family == "trend_ensemble":
        components = []
        for horizon in parameters["horizons"]:
            lookback = int(horizon)
            cached_return = cached(f"return_{lookback}d")
            components.append(
                np.sign(cached_return)
                if cached_return is not None
                else np.sign(close / close.shift(lookback) - 1.0)
            )
        return pd.concat(components, axis=1).sum(axis=1, min_count=len(components))
    if family == "dual_ma_cross":
        fast = int(parameters["fast"])
        slow = int(parameters["slow"])
        cached_fast = cached(f"sma_{fast}")
        cached_slow = cached(f"sma_{slow}")
        if cached_fast is not None and cached_slow is not None:
            return cached_fast - cached_slow
        return close.rolling(fast, min_periods=fast).mean() - close.rolling(slow, min_periods=slow).mean()
    if family == "realized_volatility_state":
        window = int(parameters["rv"])
        threshold = float(parameters["z"])
        rv = np.sqrt(252.0) * log_return.rolling(window, min_periods=window).std(ddof=1)
        high = _expanding_zscore(rv) > threshold
        direction = np.sign(log_return)
        if parameters["mode"] == "reversal":
            direction = -direction
        return direction.where(high, 0.0).where(rv.notna())
    if family == "overnight_futures_proxy":
        overnight = ledger["tr_open"] / ledger["tr_close"].shift(1) - 1.0
        mode = str(parameters["mode"])
        if mode == "spy_gap_continuation":
            return overnight
        if mode == "spy_gap_reversal":
            return -overnight
        if mode == "spy_gap_5_session_continuation":
            return (1.0 + overnight).rolling(5, min_periods=5).apply(
                np.prod,
                raw=True,
            ) - 1.0
        if mode == "spy_gap_z20_continuation":
            return _rolling_zscore(overnight, 20)
        if mode == "spy_gap_z20_reversal":
            return -_rolling_zscore(overnight, 20)
        if mode == "spy_gap_vix_change_filter":
            vix = _series(data, "VIX")
            vix_change = vix / vix.shift(1) - 1.0
            return np.sign(overnight) * np.where(vix_change <= 0, 1.0, -1.0)
    if family == "volatility_conditioned_trend":
        trend_window = int(parameters["trend"])
        rv_window = int(parameters["rv"])
        threshold = float(parameters["z"])
        trend = np.sign(close / close.shift(trend_window) - 1.0)
        rv = np.sqrt(252.0) * log_return.rolling(rv_window, min_periods=rv_window).std(ddof=1)
        high = _expanding_zscore(rv) > threshold
        if parameters["mode"] == "reversal":
            return trend * np.where(high, -1.0, 1.0)
        # The frozen continuation formula deliberately multiplies by one in
        # both states. Preserve that exact, economically redundant rule.
        return trend
    raise CandidateRejected(f"NOT_A_PRICE_SCORE_FAMILY:{family}")


def _vix_score(candidate: Mapping[str, Any], data: PreparedMarketData) -> pd.Series:
    family = str(candidate["family"])
    parameters = candidate["parameters"]
    vix = _series(data, "VIX")
    close = data.ledger["tr_close"]
    if family == "vix_term_structure":
        ratio_name = str(parameters["ratio"])
        if ratio_name != "VIX/VIX3M":
            raise CandidateRejected("VX_FUTURES_ROLL_SERIES_UNAVAILABLE")
        raw = vix / _series(data, "VIX3M")
        ratio = raw.rolling(int(parameters["smooth"]), min_periods=int(parameters["smooth"])).mean()
        return float(parameters["threshold"]) - ratio
    if family == "variance_risk_premium_proxy":
        window = int(parameters["rv"])
        log_return = np.log(close / close.shift(1))
        realized_variance = 252.0 * log_return.pow(2).rolling(window, min_periods=window).mean()
        score = (vix / 100.0).pow(2) - realized_variance
        return score if parameters["positive_sign"] == "long" else -score
    if family == "vix_level_change":
        if "lookback" in parameters:
            return -np.log(vix / vix.shift(int(parameters["lookback"])))
        if "zlookback" in parameters:
            zscore = _rolling_zscore(np.log(vix), int(parameters["zlookback"]))
            active = zscore.abs() > float(parameters["threshold"])
            return (-zscore).where(active, 0.0).where(zscore.notna())
        return vix.rolling(int(parameters["ma"]), min_periods=int(parameters["ma"])).mean() - vix
    raise CandidateRejected(f"NOT_A_VIX_SCORE_FAMILY:{family}")


def _macro_score(candidate: Mapping[str, Any], data: PreparedMarketData) -> pd.Series:
    family = str(candidate["family"])
    p = candidate["parameters"]
    if family == "yield_curve_regime":
        spread = str(p.get("spread", ""))
        if spread == "10y-3m":
            values = _series(data, "T10Y3M") if "T10Y3M" in data.series else _series(data, "DGS10") - _series(data, "DGS3MO")
        elif spread == "10y-2y":
            values = _series(data, "T10Y2Y") if "T10Y2Y" in data.series else _series(data, "DGS10") - _series(data, "DGS2")
        elif spread == "both":
            left = _series(data, "T10Y3M")
            right = _series(data, "T10Y2Y")
            return np.sign(left) + np.sign(right)
        else:
            raise CandidateRejected("UNKNOWN_YIELD_CURVE_RULE")
        if "change" in p:
            return values - values.shift(int(p["change"]))
        return values - float(p["threshold"])
    if family == "credit_spread_regime":
        series = str(p["series"])
        lookback = int(p["lookback"])
        if series == "HY_IG_BAA":
            parts = [_series(data, name) for name in ("HY_OAS", "IG_OAS", "BAA10Y")]
            return sum(-(part - part.shift(lookback)) for part in parts)
        return -(_series(data, series) - _series(data, series).shift(lookback))
    if family == "financial_conditions_regime":
        series = str(p["series"])
        if series == "NFCI_ANFCI_OFR":
            return -sum(
                np.sign(_series(data, name))
                for name in ("NFCI", "ANFCI", "OFR_FSI")
            )
        values = _series(data, series)
        if p["mode"] == "change":
            lookback = int(p["lookback"])
            if series in {"NFCI", "ANFCI"}:
                return -(values - _calendar_lag(values, weeks=lookback))
            return -(values - values.shift(lookback))
        return -values
    if family == "monetary_inflation_regime":
        if p.get("rate") == "fedfunds":
            inflation_name = "CPI" if p["inflation"] == "CPI" else "PCE"
            inflation = _series(data, inflation_name)
            inflation_yoy = (inflation / _calendar_lag(inflation, months=12) - 1.0) * 100.0
            return -(_series(data, "DFF") - inflation_yoy)
        if p.get("series") == "T10YIE":
            values = _series(data, "T10YIE")
            return values - values.shift(int(p["lookback"]))
        if p.get("series") == "WALCL":
            values = _series(data, "WALCL")
            return np.log(values / _calendar_lag(values, weeks=int(p["lookback"])))
        if p.get("series") == "M2/CPI":
            real_m2 = _series(data, "M2") / _series(data, "CPI")
            return real_m2 / _calendar_lag(real_m2, months=12) - 1.0
        if p.get("series") == "real_rate_liquidity_inflation":
            raise CandidateRejected(
                "INCOMPLETE_FROZEN_RULE_SPEC:INFLATION_ACCELERATION_HORIZON"
            )
        raise CandidateRejected("UNKNOWN_MONETARY_INFLATION_RULE")
    raise CandidateRejected(f"NOT_A_MACRO_SCORE_FAMILY:{family}")


def _calendar_decisions(candidate: Mapping[str, Any], data: PreparedMarketData) -> pd.Series:
    index = data.ledger.index
    close = data.ledger["tr_close"]
    trend = close - close.rolling(200, min_periods=200).mean()
    rule = str(candidate["parameters"]["rule"])
    next_dates = pd.Series(index, index=index).shift(-1)
    override_long = pd.Series(False, index=index)
    override_short = pd.Series(False, index=index)
    if rule.startswith("last1_first2") or rule.startswith("last4_first3"):
        first_count, last_count = ((2, 1) if rule.startswith("last1") else (3, 4))
        session_frame = pd.DataFrame({"date": index}, index=index)
        session_frame["month"] = index.to_period("M")
        session_frame["from_start"] = session_frame.groupby("month").cumcount() + 1
        session_frame["from_end"] = session_frame.groupby("month").cumcount(ascending=False) + 1
        next_position = session_frame[["from_start", "from_end"]].shift(-1)
        override_long = (next_position["from_start"] <= first_count) | (next_position["from_end"] <= last_count)
    elif rule.startswith("monday_short"):
        override_short = next_dates.dt.dayofweek == 0
    elif rule.startswith("friday_long"):
        override_long = next_dates.dt.dayofweek == 4
    elif rule.startswith("preholiday_long"):
        following_dates = next_dates.shift(-1)
        skipped_weekdays = pd.Series(False, index=index)
        valid = next_dates.notna() & following_dates.notna()
        skipped_weekdays.loc[valid] = [
            np.busday_count(
                (left + pd.Timedelta(days=1)).date(),
                right.date(),
            )
            > 0
            for left, right in zip(
                next_dates.loc[valid],
                following_dates.loc[valid],
                strict=True,
            )
        ]
        override_long = skipped_weekdays
    elif rule.startswith("nov_apr_long"):
        override_long = next_dates.dt.month.isin([11, 12, 1, 2, 3, 4])
    else:
        raise CandidateRejected("UNKNOWN_CALENDAR_RULE")
    long_event = override_long | (~override_short & ~override_long & (trend > 0))
    short_event = override_short | (~override_short & ~override_long & (trend < 0))
    return _state_from_events(index, long_event, short_event, eligible=trend.notna())


def _price_breakout_decisions(candidate: Mapping[str, Any], data: PreparedMarketData) -> pd.Series:
    close = data.ledger["tr_close"]
    lookback = int(candidate["parameters"]["lookback"])
    prior_high = close.shift(1).rolling(lookback, min_periods=lookback).max()
    prior_low = close.shift(1).rolling(lookback, min_periods=lookback).min()
    return _state_from_events(
        data.ledger.index,
        close > prior_high,
        close < prior_low,
        eligible=prior_high.notna() & prior_low.notna(),
    )


def _volume_reversal_decisions(candidate: Mapping[str, Any], data: PreparedMarketData) -> pd.Series:
    p = candidate["parameters"]
    close = data.ledger["tr_close"]
    lag_return = close / close.shift(int(p["ret"])) - 1.0
    volume_z = _rolling_zscore(np.log1p(data.ledger["volume"].astype(float)), int(p["vol"]))
    active = volume_z >= float(p["z"])
    return _state_from_events(
        data.ledger.index,
        active & (lag_return < 0),
        active & (lag_return > 0),
        eligible=volume_z.notna() & lag_return.notna(),
    )


def _targeted_reversal_decisions(
    candidate: Mapping[str, Any],
    data: PreparedMarketData,
) -> pd.Series:
    """Causal event rules for the targeted post-batch-2 train search."""

    family = str(candidate["family"])
    p = candidate["parameters"]
    ledger = data.ledger
    close = ledger["tr_close"].astype(float)
    if family == "internal_bar_strength_reversal":
        high = ledger["high"].astype(float)
        low = ledger["low"].astype(float)
        ibs = (close - low) / (high - low).replace(0.0, np.nan)
        lower = float(p["lower"])
        upper = float(p["upper"])
        return _state_from_events(
            ledger.index,
            ibs <= lower,
            ibs >= upper,
            eligible=ibs.notna(),
        )
    if family == "intraday_return_reversal":
        threshold = float(p["threshold_pct"]) / 100.0
        intraday_return = close / ledger["tr_open"].astype(float) - 1.0
        return _state_from_events(
            ledger.index,
            intraday_return <= -threshold,
            intraday_return >= threshold,
            eligible=intraday_return.notna(),
        )
    if family == "return_threshold_reversal":
        lookback = int(p["lookback"])
        threshold = float(p["threshold_pct"]) / 100.0
        lag_return = close / close.shift(lookback) - 1.0
        return _state_from_events(
            ledger.index,
            lag_return <= -threshold,
            lag_return >= threshold,
            eligible=lag_return.notna(),
        )
    if family == "rsi_reversal":
        window = int(p["window"])
        delta = close.diff()
        gain = delta.clip(lower=0.0).ewm(
            alpha=1.0 / window,
            adjust=False,
            min_periods=window,
        ).mean()
        loss = (-delta.clip(upper=0.0)).ewm(
            alpha=1.0 / window,
            adjust=False,
            min_periods=window,
        ).mean()
        relative_strength = gain / loss.replace(0.0, np.nan)
        rsi = 100.0 - 100.0 / (1.0 + relative_strength)
        lower = float(p["lower"])
        upper = float(p["upper"])
        return _state_from_events(
            ledger.index,
            rsi <= lower,
            rsi >= upper,
            eligible=rsi.notna(),
        )
    if family == "streak_reversal":
        required = int(p["streak"])
        direction = np.sign(close.diff()).fillna(0.0)
        groups = direction.ne(direction.shift()).cumsum()
        streak = direction * direction.groupby(groups).cumcount().add(1)
        return _state_from_events(
            ledger.index,
            streak <= -required,
            streak >= required,
            eligible=direction.ne(0.0),
        )
    raise CandidateRejected(f"NOT_A_TARGETED_REVERSAL_FAMILY:{family}")


def _vix_extreme_decisions(candidate: Mapping[str, Any], data: PreparedMarketData) -> pd.Series:
    p = candidate["parameters"]
    vix = _series(data, "VIX")
    if "high" in p:
        transformed = _rolling_percentile(vix, int(p["window"]))
        return _state_from_events(
            data.ledger.index,
            transformed >= float(p["high"]),
            transformed <= float(p["low"]),
            eligible=transformed.notna(),
        )
    transformed = _rolling_zscore(np.log(vix), int(p["window"]))
    return _state_from_events(
        data.ledger.index,
        transformed >= float(p["z"]),
        transformed <= -float(p["z"]),
        eligible=transformed.notna(),
    )


_COMPONENT_IDS_PATTERN = re.compile(r"^component_strategy_ids\s*=\s*(\[.*\])$")


def _frozen_component_ids(candidate: Mapping[str, Any]) -> tuple[str, ...]:
    """Read the exact component IDs embedded in the frozen feature formula."""

    matches = []
    for formula in candidate.get("feature_formulas", ()):
        match = _COMPONENT_IDS_PATTERN.fullmatch(str(formula).strip())
        if match:
            matches.append(match.group(1))
    if len(matches) != 1:
        raise CandidateRejected("INCOMPLETE_FROZEN_ENSEMBLE_COMPONENT_IDS")
    try:
        parsed = ast.literal_eval(matches[0])
    except (SyntaxError, ValueError) as exc:
        raise CandidateRejected("INVALID_FROZEN_ENSEMBLE_COMPONENT_IDS") from exc
    if not isinstance(parsed, list) or not parsed or not all(
        isinstance(value, str) and re.fullmatch(r"STRAT\d{4}", value)
        for value in parsed
    ):
        raise CandidateRejected("INVALID_FROZEN_ENSEMBLE_COMPONENT_IDS")
    return tuple(parsed)


def _ensemble_decisions(
    candidate: Mapping[str, Any],
    data: PreparedMarketData,
    candidate_lookup: Mapping[str, Mapping[str, Any]] | None,
    evaluation_stack: Sequence[str],
    feature_frame: pd.DataFrame | None,
) -> pd.Series:
    if candidate_lookup is None:
        raise CandidateRejected("ENSEMBLE_REQUIRES_FROZEN_CANDIDATE_REGISTRY")
    component_ids = _frozen_component_ids(candidate)
    strategy_id = str(candidate["strategy_id"])
    if strategy_id in evaluation_stack:
        raise CandidateRejected("CYCLIC_ENSEMBLE_COMPONENT_GRAPH")
    components: list[pd.Series] = []
    next_stack = (*evaluation_stack, strategy_id)
    for component_id in component_ids:
        try:
            component = candidate_lookup[component_id]
        except KeyError as exc:
            raise CandidateRejected(
                f"UNKNOWN_FROZEN_ENSEMBLE_COMPONENT:{component_id}"
            ) from exc
        components.append(
            candidate_decisions(
                component,
                data,
                candidate_lookup=candidate_lookup,
                evaluation_stack=next_stack,
                feature_frame=feature_frame,
            ).decisions
        )
    score = pd.concat(components, axis=1).sum(axis=1, min_count=len(components))
    starts = [component.attrs.get("first_valid") for component in components]
    if any(value is None for value in starts):
        score[:] = np.nan
    else:
        score.loc[score.index < max(pd.Timestamp(value) for value in starts)] = np.nan
    return _state_from_score(score)


def candidate_decisions(
    candidate: Mapping[str, Any],
    data: PreparedMarketData,
    *,
    candidate_lookup: Mapping[str, Mapping[str, Any]] | None = None,
    evaluation_stack: Sequence[str] = (),
    feature_frame: pd.DataFrame | None = None,
) -> SignalResult:
    _required_data_gate(candidate, data)
    family = str(candidate["family"])
    if family == "price_breakout":
        decisions = _price_breakout_decisions(candidate, data)
    elif family == "volume_conditioned_reversal":
        decisions = _volume_reversal_decisions(candidate, data)
    elif family in {
        "internal_bar_strength_reversal",
        "intraday_return_reversal",
        "return_threshold_reversal",
        "rsi_reversal",
        "streak_reversal",
    }:
        decisions = _targeted_reversal_decisions(candidate, data)
    elif family == "vix_extreme_reversal":
        decisions = _vix_extreme_decisions(candidate, data)
    elif family == "calendar_seasonality":
        decisions = _calendar_decisions(candidate, data)
    elif family == "simple_rule_ensemble":
        decisions = _ensemble_decisions(
            candidate,
            data,
            candidate_lookup,
            evaluation_stack,
            feature_frame,
        )
    elif family in {
        "price_trend_sma",
        "time_series_momentum",
        "short_horizon_reversal",
        "multi_horizon_reversal",
        "reversal_trend_blend",
        "rsi_trend_blend",
        "trend_ensemble",
        "dual_ma_cross",
        "dual_reversal_trend_vote",
        "trend_guarded_dual_reversal",
        "volatility_regime_reversal",
            "overnight_tug_reversal_vote",
            "strong_trend_override_reversal",
            "asymmetric_trend_override_reversal",
            "drawdown_recovery_override_reversal",
            "quiet_bull_recovery_override_reversal",
            "recovery_trend_breakout_majority",
            "high_vol_crash_recovery_reversal",
            "adaptive_recovery_edge_switch",
            "recovery_overnight_tug_vote",
            "recovery_turn_month_vote",
            "recovery_internal_bar_strength_vote",
            "recovery_multi_horizon_reversal_vote",
            "recovery_volume_gated_reversal_vote",
            "recovery_calendar_volume_reversal_vote",
            "recovery_calendar_volume_vxo_vote",
            "realized_volatility_state",
        "overnight_futures_proxy",
        "volatility_conditioned_trend",
    }:
        decisions = _state_from_score(_price_score(candidate, data, feature_frame=feature_frame))
    elif family in {"vix_term_structure", "variance_risk_premium_proxy", "vix_level_change"}:
        decisions = _state_from_score(_vix_score(candidate, data))
    elif family in {
        "yield_curve_regime",
        "credit_spread_regime",
        "financial_conditions_regime",
        "monetary_inflation_regime",
    }:
        decisions = _state_from_score(_macro_score(candidate, data))
    elif family in EXPLICIT_FAMILY_REJECTIONS:
        raise CandidateRejected(EXPLICIT_FAMILY_REJECTIONS[family])
    else:
        raise CandidateRejected(f"FAMILY_REQUIRES_UNIMPLEMENTED_CAUSAL_ADAPTER:{family}")
    if not decisions.isin((-1, 1)).all():
        raise CandidateRejected("INVALID_POSITION_OUTPUT")
    first_value = decisions.attrs.get("first_valid")
    first = pd.Timestamp(first_value).date().isoformat() if first_value is not None else None
    eligible_count = int(decisions.attrs.get("eligible_count", len(decisions)))
    if first_value is None:
        missing_fraction = 1.0
    else:
        first_offset = int(data.ledger.index.searchsorted(pd.Timestamp(first_value)))
        expected_after_start = len(decisions) - first_offset
        missing_fraction = (
            1.0 - eligible_count / expected_after_start
            if expected_after_start > 0
            else 1.0
        )
    return SignalResult(
        decisions=decisions,
        first_evaluable_date=first,
        missing_fraction=missing_fraction,
    )


def benchmark_decisions(benchmark_id: str, data: PreparedMarketData) -> SignalResult:
    index = data.ledger.index
    close = data.ledger["tr_close"]
    if benchmark_id in {"buy_and_hold_spy_total_return", "always_long"}:
        decisions = pd.Series(1, index=index, dtype=np.int8)
        decisions.attrs["first_valid"] = index[0] if len(index) else None
        decisions.attrs["eligible_count"] = len(index)
    elif benchmark_id == "always_short":
        decisions = pd.Series(-1, index=index, dtype=np.int8)
        decisions.attrs["first_valid"] = index[0] if len(index) else None
        decisions.attrs["eligible_count"] = len(index)
    elif benchmark_id == "symmetric_sma_200":
        decisions = _state_from_score(close - close.rolling(200, min_periods=200).mean())
    elif benchmark_id == "symmetric_momentum_12m":
        decisions = _state_from_score(close / close.shift(252) - 1.0)
    else:
        raise CandidateRejected(f"UNKNOWN_BENCHMARK:{benchmark_id}")
    first_value = decisions.attrs.get("first_valid")
    return SignalResult(
        decisions=decisions,
        first_evaluable_date=(pd.Timestamp(first_value).date().isoformat() if first_value is not None else None),
        missing_fraction=0.0,
    )
