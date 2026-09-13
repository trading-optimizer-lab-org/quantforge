"""Time-bounded, train-only massive SPY search primitives.

The module deliberately keeps candidate identity as a compact deterministic
recipe. GitHub workers can reconstruct any tested rule from its integer index
without downloading a central multi-gigabyte candidate registry.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from bisect import bisect_right
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

from aurora.infra.sp500_long_short_daily.statistics import (
    deflated_sharpe_probability,
)
from aurora.infra.sp500_long_short_daily.signals import IMPLEMENTED_FAMILIES

from .contracts import (
    BLOCK_LENGTH,
    BOOTSTRAP_REPETITIONS,
    LOCKED_START,
    TRAIN_END,
    assert_contract,
    canonical_rule_hash,
)
from .registry import base_package, generate_candidates


CAMPAIGN_VERSION = "sp500-massive-train-night-v1"
WAVES = 7
SHARDS = 360
WORKERS_PER_SHARD = 4
MINUTES_PER_SHARD = 50
TOP_ROWS_PER_PROCESS = 100
MAX_ITERATIONS_PER_WORKER = 100_000
PBO_PARTITIONS = 10
PBO_BINS = 512
PBO_MIN_SHARPE = -1.0
PBO_MAX_SHARPE = 1.0
GLOBAL_BOOTSTRAP_SEED = 2_026_080_7
MASSIVE_BATCH_ID = 1000

UNAVAILABLE_MASSIVE_FAMILIES = frozenset(
    {
        "credit_spread_regime",
        "financial_conditions_regime",
        "monetary_inflation_regime",
        "variance_risk_premium_proxy",
        "vix_level_change",
        "vix_term_structure",
        "yield_curve_regime",
    }
)

PAIR_KEYS = (
    ("lower", "upper"),
    ("ibs_lower", "ibs_upper"),
    ("fast", "slow"),
)


def _stable_u64(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _json_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _immutable_json(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_immutable_json(child) for child in value)
    if isinstance(value, dict):
        return {str(key): _immutable_json(child) for key, child in value.items()}
    return value


def _sort_domain(values: Iterable[Any]) -> tuple[Any, ...]:
    unique = {_json_key(value): value for value in values}
    return tuple(unique[key] for key in sorted(unique))


def _expanded_numeric_domain(values: Iterable[Any]) -> tuple[Any, ...]:
    original = _sort_domain(values)
    if len(original) < 2 or any(isinstance(value, bool) for value in original):
        return original
    if not all(isinstance(value, (int, float)) for value in original):
        return original
    low = float(min(original))
    high = float(max(original))
    if not math.isfinite(low) or not math.isfinite(high) or low == high:
        return original
    integral = all(float(value).is_integer() for value in original)
    if integral:
        span = int(round(high - low))
        if span <= 64:
            expanded: Iterable[Any] = range(int(round(low)), int(round(high)) + 1)
        else:
            expanded = np.rint(np.linspace(low, high, 65)).astype(int).tolist()
    else:
        expanded = [round(float(value), 8) for value in np.linspace(low, high, 33)]
    return _sort_domain((*original, *expanded))


def _effective_seed_candidates() -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    package = base_package()
    for source in package.candidates:
        family = str(source.get("family"))
        required = set(source.get("required_datasets", ()))
        if family not in IMPLEMENTED_FAMILIES or family in UNAVAILABLE_MASSIVE_FAMILIES:
            continue
        if not required.issubset({"DS001", "DS002", "DS005"}):
            continue
        digest = str(source.get("canonical_hash") or canonical_rule_hash(source))
        if digest not in seen:
            rows.append(json.loads(json.dumps(source)))
            seen.add(digest)
    for batch_id in range(52):
        for source in generate_candidates(batch_id, count=96):
            family = str(source.get("family"))
            if family not in IMPLEMENTED_FAMILIES or family in UNAVAILABLE_MASSIVE_FAMILIES:
                continue
            digest = str(source["canonical_hash"])
            if digest not in seen:
                rows.append(json.loads(json.dumps(source)))
                seen.add(digest)
    return tuple(rows)


@dataclass(frozen=True)
class FamilyRecipe:
    family: str
    template: Mapping[str, Any]
    keys: tuple[str, ...]
    domains: tuple[tuple[Any, ...], ...]
    combinations: int
    permutation_stride: int
    permutation_offset: int

    def parameters_at(self, ordinal: int) -> dict[str, Any]:
        if ordinal < 0 or ordinal >= self.combinations:
            raise IndexError("FAMILY_RECIPE_ORDINAL_OUT_OF_RANGE")
        values: dict[str, Any] = {}
        remaining = (
            ordinal * self.permutation_stride + self.permutation_offset
        ) % self.combinations
        for key, domain in zip(reversed(self.keys), reversed(self.domains), strict=True):
            remaining, offset = divmod(remaining, len(domain))
            value = domain[offset]
            if key.startswith("__pair__"):
                left, right = key.removeprefix("__pair__").split("__", maxsplit=1)
                values[left], values[right] = value
            else:
                values[key] = value
        return {key: values[key] for key in sorted(values)}


@dataclass(frozen=True)
class MassiveRecipe:
    families: tuple[FamilyRecipe, ...]
    total_combinations: int
    round_starts: tuple[int, ...]
    round_candidate_boundaries: tuple[int, ...]

    @classmethod
    def build(cls, families: Sequence[FamilyRecipe]) -> "MassiveRecipe":
        rows = tuple(families)
        if not rows:
            raise ValueError("EMPTY_MASSIVE_RECIPE_FAMILIES")
        total = sum(row.combinations for row in rows)
        round_starts: list[int] = []
        round_boundaries: list[int] = []
        previous = 0
        cumulative = 0
        for stop in sorted({row.combinations for row in rows}):
            active = sum(row.combinations > previous for row in rows)
            round_starts.append(previous)
            cumulative += (stop - previous) * active
            round_boundaries.append(cumulative)
            previous = stop
        if cumulative != total:
            raise RuntimeError("MASSIVE_RECIPE_ROUND_BOUNDARY_MISMATCH")
        return cls(
            families=rows,
            total_combinations=total,
            round_starts=tuple(round_starts),
            round_candidate_boundaries=tuple(round_boundaries),
        )

    def family_and_ordinal(self, index: int) -> tuple[FamilyRecipe, int]:
        """Round-robin the still-unseen combinations of every family.

        This is a bijection over the union of all family grids. It prevents the
        very large VXO family from starving every smaller causal family while
        also avoiding duplicate effective rules.
        """

        if index < 0 or index >= self.total_combinations:
            raise ValueError("MASSIVE_CANDIDATE_INDEX_OUT_OF_RANGE")

        segment = bisect_right(self.round_candidate_boundaries, index)
        prior_candidates = (
            self.round_candidate_boundaries[segment - 1] if segment else 0
        )
        round_start = self.round_starts[segment]
        active = [
            row for row in self.families if row.combinations > round_start
        ]
        local = index - prior_candidates
        ordinal = round_start + local // len(active)
        offset = local % len(active)
        active = [row for row in self.families if row.combinations > ordinal]
        if offset >= len(active):
            raise RuntimeError("MASSIVE_RECIPE_BALANCING_BREACH")
        return active[offset], ordinal

    def to_payload(self) -> dict[str, Any]:
        return {
            "campaign_version": CAMPAIGN_VERSION,
            "total_combinations": self.total_combinations,
            "round_starts": list(self.round_starts),
            "round_candidate_boundaries": list(self.round_candidate_boundaries),
            "families": [
                {
                    "family": row.family,
                    "template": row.template,
                    "keys": list(row.keys),
                    "domains": [list(domain) for domain in row.domains],
                    "combinations": row.combinations,
                    "permutation_stride": row.permutation_stride,
                    "permutation_offset": row.permutation_offset,
                }
                for row in self.families
            ],
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "MassiveRecipe":
        if payload.get("campaign_version") != CAMPAIGN_VERSION:
            raise ValueError("MASSIVE_RECIPE_VERSION_MISMATCH")
        families = tuple(
            FamilyRecipe(
                family=str(row["family"]),
                template=dict(row["template"]),
                keys=tuple(str(value) for value in row["keys"]),
                domains=tuple(
                    tuple(_immutable_json(value) for value in domain)
                    for domain in row["domains"]
                ),
                combinations=int(row["combinations"]),
                permutation_stride=int(row["permutation_stride"]),
                permutation_offset=int(row["permutation_offset"]),
            )
            for row in payload["families"]
        )
        recipe = cls(
            families=families,
            total_combinations=int(payload["total_combinations"]),
            round_starts=tuple(int(value) for value in payload["round_starts"]),
            round_candidate_boundaries=tuple(
                int(value) for value in payload["round_candidate_boundaries"]
            ),
        )
        if sum(row.combinations for row in families) != recipe.total_combinations:
            raise ValueError("MASSIVE_RECIPE_TOTAL_MISMATCH")
        return recipe


def _family_recipes() -> tuple[FamilyRecipe, ...]:
    seeds = _effective_seed_candidates()
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in seeds:
        grouped.setdefault(str(row["family"]), []).append(row)
    recipes: list[FamilyRecipe] = []
    for family, rows in sorted(grouped.items()):
        parameter_rows = [dict(row.get("parameters", {})) for row in rows]
        parameter_keys = sorted({key for row in parameter_rows for key in row})
        domains_by_key = {
            key: _expanded_numeric_domain(
                row[key] for row in parameter_rows if key in row
            )
            for key in parameter_keys
        }
        consumed: set[str] = set()
        packed: list[tuple[str, tuple[Any, ...]]] = []
        for left, right in PAIR_KEYS:
            if left not in domains_by_key or right not in domains_by_key:
                continue
            pairs = []
            for row in parameter_rows:
                if left not in row or right not in row:
                    continue
                if float(row[left]) < float(row[right]):
                    pairs.append((row[left], row[right]))
            domain = _sort_domain(pairs)
            if domain:
                packed.append((f"__pair__{left}__{right}", domain))
                consumed.update((left, right))
        for key in parameter_keys:
            if key not in consumed:
                packed.append((key, domains_by_key[key]))
        if not packed:
            packed.append(("__constant__", (None,)))
        keys = tuple(key for key, _ in packed)
        domains = tuple(domain for _, domain in packed)
        combinations = math.prod(len(domain) for domain in domains)
        template = min(rows, key=lambda row: (int(row.get("complexity_score", 99)), str(row["strategy_id"])))
        recipes.append(
            FamilyRecipe(
                family=family,
                template=template,
                keys=keys,
                domains=domains,
                combinations=combinations,
                permutation_stride=_coprime_stride(combinations, family),
                permutation_offset=_stable_u64(CAMPAIGN_VERSION, family, "offset")
                % combinations,
            )
        )
    return tuple(recipes)


def _coprime_stride(combinations: int, family: str) -> int:
    if combinations <= 1:
        return 1
    stride = 1 + _stable_u64(CAMPAIGN_VERSION, family, "stride") % (combinations - 1)
    while math.gcd(stride, combinations) != 1:
        stride = 1 + stride % (combinations - 1)
    return stride


@lru_cache(maxsize=1)
def massive_recipe() -> MassiveRecipe:
    families = _family_recipes()
    if not families:
        raise RuntimeError("EMPTY_MASSIVE_RECIPE")
    running = sum(family.combinations for family in families)
    if running < 100_000_000:
        raise RuntimeError(f"MASSIVE_RECIPE_TOO_SMALL:{running}")
    return MassiveRecipe.build(families)


def _candidate_from_parameters(
    recipe: FamilyRecipe,
    parameters: Mapping[str, Any],
    *,
    candidate_index: int,
    wave: int,
    mode: str,
) -> dict[str, Any]:
    candidate = json.loads(json.dumps(recipe.template))
    if "__constant__" in parameters:
        parameters = {}
    candidate.update(
        {
            "strategy_id": f"MTRAIN-{mode[0].upper()}-W{wave:02d}-{candidate_index:012d}",
            "variant_label": f"{CAMPAIGN_VERSION}:{mode}:wave={wave}:index={candidate_index}",
            "family": recipe.family,
            "family_name": recipe.family.replace("_", " ").title(),
            "parameters": dict(parameters),
            "instrument": "SPY",
            "cash_allowed": False,
            "partial_exposure_allowed": False,
            "leverage_allowed": False,
            "volatility_scaling_allowed": False,
            "pyramiding_allowed": False,
            "multiple_assets_in_portfolio": False,
            "selection_role": "massive_train_only_pre_registered_recipe",
            "evidence_track": "pre_2011_evidence",
            "locked_boundary": ">=2021-01-01 unopened",
            "train_boundary": TRAIN_END,
            "validation_boundary": "2011-01-01..2020-12-31 unopened",
        }
    )
    candidate["canonical_hash"] = canonical_rule_hash(candidate)
    assert_contract(candidate)
    return candidate


def broad_candidate(
    candidate_index: int,
    *,
    wave: int,
    recipe: MassiveRecipe | None = None,
) -> dict[str, Any]:
    recipe = recipe or massive_recipe()
    family, ordinal = recipe.family_and_ordinal(candidate_index)
    return _candidate_from_parameters(
        family,
        family.parameters_at(ordinal),
        candidate_index=candidate_index,
        wave=wave,
        mode="broad",
    )


def candidate_for_index(
    candidate_index: int,
    *,
    wave: int,
    recipe: MassiveRecipe | None = None,
) -> dict[str, Any]:
    if not 0 <= wave < WAVES:
        raise ValueError("MASSIVE_WAVE_OUT_OF_RANGE")
    return broad_candidate(candidate_index, wave=wave, recipe=recipe)


def candidate_index(wave: int, shard: int, worker: int, iteration: int) -> int:
    if not 0 <= shard < SHARDS or not 0 <= worker < WORKERS_PER_SHARD:
        raise ValueError("MASSIVE_ASSIGNMENT_OUT_OF_RANGE")
    if iteration < 0:
        raise ValueError("NEGATIVE_MASSIVE_ITERATION")
    if iteration >= MAX_ITERATIONS_PER_WORKER:
        raise ValueError("MASSIVE_ITERATION_CAP_EXCEEDED")
    wave_capacity = SHARDS * WORKERS_PER_SHARD * MAX_ITERATIONS_PER_WORKER
    assignment = shard * WORKERS_PER_SHARD + worker
    return wave * wave_capacity + iteration * SHARDS * WORKERS_PER_SHARD + assignment


def nav_metrics(values: np.ndarray) -> dict[str, float | int]:
    raw = np.asarray(values, dtype=float)
    raw = raw[np.isfinite(raw)]
    if not len(raw):
        return {
            "sessions": 0,
            "total_return_pct": -100.0,
            "cagr_pct": -100.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "max_drawdown_pct": -100.0,
            "calmar": 0.0,
        }
    nav = np.cumprod(1.0 + raw)
    final = float(nav[-1])
    years = len(raw) / 252.0
    cagr = final ** (1.0 / years) - 1.0 if final > 0.0 else -1.0
    std = float(np.std(raw, ddof=0))
    downside = raw[raw < 0.0]
    downside_std = float(np.std(downside, ddof=0)) if len(downside) else 0.0
    drawdown = nav / np.maximum.accumulate(nav) - 1.0
    mdd = float(drawdown.min())
    return {
        "sessions": int(len(raw)),
        "total_return_pct": (final - 1.0) * 100.0,
        "cagr_pct": cagr * 100.0,
        "sharpe": float(np.mean(raw) / std * math.sqrt(252.0)) if std > 1e-15 else 0.0,
        "sortino": (
            float(np.mean(raw) / downside_std * math.sqrt(252.0))
            if downside_std > 1e-15
            else 0.0
        ),
        "max_drawdown_pct": mdd * 100.0,
        "calmar": float(cagr / abs(mdd)) if mdd < -1e-15 else 0.0,
    }


def annual_metrics(dates: pd.DatetimeIndex, values: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    years = dates.year
    for year in sorted(set(int(value) for value in years)):
        metrics = nav_metrics(values[years == year])
        rows.append(
            {
                "year": year,
                **metrics,
                "positive": bool(float(metrics["total_return_pct"]) > 0.0),
            }
        )
    return rows


def non_global_train_gate(metrics: Mapping[str, Any], annual: Sequence[Mapping[str, Any]]) -> bool:
    positive = sum(bool(row["positive"]) for row in annual)
    positive_share = positive / len(annual) if annual else 0.0
    positive_logs = [
        math.log1p(max(float(row["total_return_pct"]) / 100.0, -0.999999))
        for row in annual
        if float(row["total_return_pct"]) > 0.0
    ]
    concentration = max(positive_logs) / sum(positive_logs) if positive_logs else 1.0
    return bool(
        float(metrics["cagr_pct"]) > 20.0
        and int(metrics["sessions"]) >= 2500
        and len(annual) >= 10
        and positive_share >= 0.60
        and float(np.median([float(row["cagr_pct"]) for row in annual])) > 0.0
        and concentration <= 0.50
    )


def normal_pvalue(values: np.ndarray) -> float:
    raw = np.asarray(values, dtype=float)
    raw = raw[np.isfinite(raw)]
    if len(raw) < 3:
        return 1.0
    std = float(np.std(raw, ddof=1))
    if std <= 1e-15:
        return 0.0 if float(raw.mean()) > 0.0 else 1.0
    return float(1.0 - norm.cdf(float(raw.mean()) / (std / math.sqrt(len(raw)))))


def shared_bootstrap_starts(length: int) -> np.ndarray:
    if length < 100:
        raise ValueError("MASSIVE_BOOTSTRAP_SERIES_TOO_SHORT")
    blocks = int(math.ceil(length / BLOCK_LENGTH))
    rng = np.random.default_rng(GLOBAL_BOOTSTRAP_SEED)
    return rng.integers(
        0,
        length,
        size=(BOOTSTRAP_REPETITIONS, blocks),
        dtype=np.int32,
    )


@dataclass
class BootstrapAccumulator:
    length: int
    weights: np.ndarray
    white_max: np.ndarray
    spa_max: np.ndarray
    observed_max: float = -math.inf
    observed_spa_max: float = -math.inf

    @classmethod
    def create(cls, length: int) -> "BootstrapAccumulator":
        starts = shared_bootstrap_starts(length)
        weights = np.zeros((BOOTSTRAP_REPETITIONS, length), dtype=np.float32)
        rows = np.arange(BOOTSTRAP_REPETITIONS)
        full_blocks, remainder = divmod(length, BLOCK_LENGTH)
        for block in range(full_blocks):
            origins = starts[:, block]
            for offset in range(BLOCK_LENGTH):
                weights[rows, (origins + offset) % length] += 1.0
        if remainder:
            origins = starts[:, full_blocks]
            for offset in range(remainder):
                weights[rows, (origins + offset) % length] += 1.0
        weights /= np.float32(length)
        return cls(
            length=length,
            weights=weights,
            white_max=np.full(BOOTSTRAP_REPETITIONS, -np.inf, dtype=np.float64),
            spa_max=np.full(BOOTSTRAP_REPETITIONS, -np.inf, dtype=np.float64),
        )

    @classmethod
    def with_shared_weights(cls, weights: np.ndarray) -> "BootstrapAccumulator":
        shared = np.asarray(weights, dtype=np.float32)
        if shared.ndim != 2 or shared.shape[0] != BOOTSTRAP_REPETITIONS:
            raise ValueError("MASSIVE_BOOTSTRAP_SHARED_WEIGHTS_SHAPE_MISMATCH")
        return cls(
            length=int(shared.shape[1]),
            weights=shared,
            white_max=np.full(BOOTSTRAP_REPETITIONS, -np.inf, dtype=np.float64),
            spa_max=np.full(BOOTSTRAP_REPETITIONS, -np.inf, dtype=np.float64),
        )

    def update(self, differential: np.ndarray) -> float:
        raw = np.asarray(differential, dtype=float)
        if len(raw) != self.length:
            raise ValueError("MASSIVE_BOOTSTRAP_LENGTH_MISMATCH")
        observed = float(raw.mean())
        observed_std = float(raw.std(ddof=1))
        observed_t = observed / (observed_std / math.sqrt(len(raw))) if observed_std > 1e-15 else 0.0
        self.observed_max = max(self.observed_max, observed)
        self.observed_spa_max = max(self.observed_spa_max, observed_t)
        centered = np.asarray(raw - observed, dtype=np.float32)
        means = self.weights @ centered
        second_moments = self.weights @ np.square(centered)
        variance = np.maximum(
            (second_moments - np.square(means)) * len(raw) / max(len(raw) - 1, 1),
            0.0,
        )
        standard_error = np.sqrt(variance) / math.sqrt(len(raw))
        studentized = np.divide(
            means,
            standard_error,
            out=np.zeros_like(means),
            where=standard_error > 0.0,
        )
        self.white_max = np.maximum(self.white_max, means)
        self.spa_max = np.maximum(self.spa_max, studentized)
        exceed = int(np.count_nonzero(means >= observed))
        return float((1 + exceed) / (BOOTSTRAP_REPETITIONS + 1))


@lru_cache(maxsize=1)
def pbo_combinations() -> np.ndarray:
    rows = []
    for selected in itertools.combinations(range(PBO_PARTITIONS), PBO_PARTITIONS // 2):
        row = np.zeros(PBO_PARTITIONS, dtype=np.int8)
        row[list(selected)] = 1
        rows.append(row)
    return np.asarray(rows, dtype=np.int8)


@dataclass
class PboAccumulator:
    histogram: np.ndarray
    best_is: np.ndarray
    best_oos: np.ndarray
    best_ids: list[str | None]

    @classmethod
    def create(cls) -> "PboAccumulator":
        combinations = math.comb(PBO_PARTITIONS, PBO_PARTITIONS // 2)
        return cls(
            histogram=np.zeros((combinations, PBO_BINS), dtype=np.int64),
            best_is=np.full(combinations, -np.inf, dtype=np.float64),
            best_oos=np.full(combinations, -np.inf, dtype=np.float64),
            best_ids=[None] * combinations,
        )

    def update(self, strategy_id: str, values: np.ndarray) -> None:
        blocks = np.array_split(np.arange(len(values)), PBO_PARTITIONS)
        sums = np.asarray([values[index].sum() for index in blocks], dtype=float)
        square_sums = np.asarray([np.square(values[index]).sum() for index in blocks], dtype=float)
        counts = np.asarray([len(index) for index in blocks], dtype=float)
        mask = pbo_combinations().astype(bool)

        def sharpe(selected: np.ndarray) -> np.ndarray:
            selected_float = selected.astype(float)
            total = selected_float @ sums
            square = selected_float @ square_sums
            count = selected_float @ counts
            mean = total / count
            variance = np.maximum(square / count - np.square(mean), 0.0)
            return np.divide(mean, np.sqrt(variance), out=np.zeros_like(mean), where=variance > 1e-15)

        in_sample = sharpe(mask)
        out_sample = sharpe(~mask)
        bins = np.floor(
            (np.clip(out_sample, PBO_MIN_SHARPE, PBO_MAX_SHARPE) - PBO_MIN_SHARPE)
            / (PBO_MAX_SHARPE - PBO_MIN_SHARPE)
            * PBO_BINS
        ).astype(int)
        bins = np.clip(bins, 0, PBO_BINS - 1)
        np.add.at(self.histogram, (np.arange(len(bins)), bins), 1)
        winners = in_sample > self.best_is
        for offset in np.flatnonzero(winners):
            self.best_is[offset] = in_sample[offset]
            self.best_oos[offset] = out_sample[offset]
            self.best_ids[int(offset)] = strategy_id


def dsr_from_moments(
    *,
    sessions: int,
    mean: float,
    std: float,
    skew: float,
    kurtosis: float,
    trials: int,
) -> float:
    if sessions < 3 or std <= 1e-15:
        return 0.0
    daily_sharpe = mean / std
    annual_sharpe = daily_sharpe * math.sqrt(252.0)
    expected_max = norm.ppf((trials - 0.375) / (trials + 0.25))
    benchmark = expected_max / math.sqrt(sessions - 1) * math.sqrt(252.0)
    denominator = math.sqrt(
        max(
            (1.0 - skew * daily_sharpe + ((kurtosis - 1.0) / 4.0) * daily_sharpe**2)
            / (sessions - 1),
            1e-15,
        )
    ) * math.sqrt(252.0)
    return float(norm.cdf((annual_sharpe - benchmark) / denominator))


def candidate_metric_row(
    candidate: Mapping[str, Any],
    dates: pd.DatetimeIndex,
    returns: np.ndarray,
    benchmark: np.ndarray,
    *,
    global_trials: int,
    spa_pvalue: float,
) -> dict[str, Any]:
    metrics = nav_metrics(returns)
    annual = annual_metrics(dates, returns)
    raw = np.asarray(returns, dtype=float)
    std = float(raw.std(ddof=1)) if len(raw) > 1 else 0.0
    skew = float(pd.Series(raw).skew()) if len(raw) > 2 else 0.0
    kurtosis = float(pd.Series(raw).kurtosis() + 3.0) if len(raw) > 3 else 3.0
    return {
        "strategy_id": candidate["strategy_id"],
        "canonical_hash": candidate["canonical_hash"],
        "family": candidate["family"],
        "parameters_json": json.dumps(candidate["parameters"], sort_keys=True, separators=(",", ":")),
        "candidate_json": json.dumps(candidate, sort_keys=True, separators=(",", ":")),
        "train_oof_cagr_pct": metrics["cagr_pct"],
        "train_oof_total_return_pct": metrics["total_return_pct"],
        "train_oof_sharpe": metrics["sharpe"],
        "train_oof_sortino": metrics["sortino"],
        "train_oof_calmar": metrics["calmar"],
        "train_oof_max_drawdown_pct": metrics["max_drawdown_pct"],
        "oof_sessions": metrics["sessions"],
        "covered_years": len(annual),
        "positive_years": sum(bool(row["positive"]) for row in annual),
        "median_fold_cagr_pct": float(np.median([float(row["cagr_pct"]) for row in annual])),
        "spa_pvalue": spa_pvalue,
        "raw_pvalue": normal_pvalue(raw - benchmark),
        "dsr": deflated_sharpe_probability(raw, trials=global_trials),
        "return_mean": float(raw.mean()),
        "return_std": std,
        "return_skew": skew,
        "return_kurtosis": kurtosis,
        "train_annual_metrics_json": json.dumps(annual, sort_keys=True, separators=(",", ":")),
        "non_global_train_gate": non_global_train_gate(metrics, annual),
        "train_end": TRAIN_END,
        "locked_start": LOCKED_START,
        "locked_opened": False,
        "validation_used_for_selection": False,
    }


__all__ = [
    "BootstrapAccumulator",
    "CAMPAIGN_VERSION",
    "MASSIVE_BATCH_ID",
    "MINUTES_PER_SHARD",
    "MAX_ITERATIONS_PER_WORKER",
    "MassiveRecipe",
    "PboAccumulator",
    "SHARDS",
    "WAVES",
    "WORKERS_PER_SHARD",
    "broad_candidate",
    "candidate_for_index",
    "candidate_index",
    "candidate_metric_row",
    "massive_recipe",
]
