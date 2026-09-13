from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from aurora.infra.sp500_autonomous_discovery.massive_train import (
    BootstrapAccumulator,
    MassiveRecipe,
    PboAccumulator,
    PBO_BINS,
    PBO_PARTITIONS,
    SHARDS,
    WAVES,
    WORKERS_PER_SHARD,
    broad_candidate,
    candidate_for_index,
    candidate_index,
    massive_recipe,
    shared_bootstrap_starts,
)
from aurora.infra.sp500_autonomous_discovery.contracts import BLOCK_LENGTH
from aurora.infra.sp500_long_short_daily.signals import IMPLEMENTED_FAMILIES
from aurora.scripts.merge_sp500_massive_train import (
    _multiplicity_counts,
    _unique_hash_values,
    aggregate,
)
from aurora.scripts.prepare_sp500_massive_prior_statistics import (
    _load_evaluated_returns,
    _write_multiplicity_common_dates,
)
from aurora.scripts.run_sp500_massive_train_shard import (
    _load_multiplicity_common_dates,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def recipe() -> MassiveRecipe:
    return massive_recipe()


def test_massive_recipe_is_large_balanced_supported_and_json_stable(recipe: MassiveRecipe) -> None:
    assert recipe.total_combinations > 100_000_000
    assert len(recipe.families) >= 30
    assert {row.family for row in recipe.families}.issubset(IMPLEMENTED_FAMILIES)
    restored = MassiveRecipe.from_payload(json.loads(json.dumps(recipe.to_payload())))
    assert restored.total_combinations == recipe.total_combinations
    assert [row.family for row in restored.families] == [row.family for row in recipe.families]
    first_round = [restored.family_and_ordinal(index)[0].family for index in range(len(restored.families))]
    assert first_round == [row.family for row in restored.families]


def test_first_five_thousand_broad_rules_are_unique_and_contract_bound(recipe: MassiveRecipe) -> None:
    rows = [broad_candidate(index, wave=0, recipe=recipe) for index in range(5000)]
    assert len({row["canonical_hash"] for row in rows}) == len(rows)
    assert all(row["instrument"] == "SPY" for row in rows)
    assert all(row["position_values"] == [-1, 1] for row in rows)
    assert all(row["cash_allowed"] is False for row in rows)
    assert all(row["leverage_allowed"] is False for row in rows)
    assert all(row["train_boundary"] == "2010-12-31" for row in rows)
    assert all(row["validation_boundary"] == "2011-01-01..2020-12-31 unopened" for row in rows)
    assert all(row["locked_boundary"] == ">=2021-01-01 unopened" for row in rows)


def test_wave_shard_worker_streams_never_overlap() -> None:
    indices = {
        candidate_index(wave, shard, worker, iteration)
        for wave in range(WAVES)
        for shard in range(SHARDS)
        for worker in range(WORKERS_PER_SHARD)
        for iteration in (0, 1, 99_999)
    }
    assert len(indices) == WAVES * SHARDS * WORKERS_PER_SHARD * 3


def test_all_seven_waves_use_one_deterministic_unique_stream(recipe: MassiveRecipe) -> None:
    starts = [candidate_index(wave, 0, 0, 0) for wave in range(WAVES)]
    rows = [candidate_for_index(start, wave=wave, recipe=recipe) for wave, start in enumerate(starts)]
    assert len({row["canonical_hash"] for row in rows}) == len(rows)
    repeated = candidate_for_index(starts[-1], wave=WAVES - 1, recipe=recipe)
    assert repeated == rows[-1]


def test_bootstrap_rejects_wrong_length_and_pbo_accepts_train_vector() -> None:
    accumulator = BootstrapAccumulator.create(120)
    values = np.linspace(-0.01, 0.02, 120)
    pvalue = accumulator.update(values)
    assert 0.0 < pvalue <= 1.0
    with pytest.raises(ValueError, match="MASSIVE_BOOTSTRAP_LENGTH_MISMATCH"):
        accumulator.update(values[:-1])
    pbo = PboAccumulator.create()
    pbo.update("candidate", values)
    assert pbo.histogram.shape == (
        252,
        PBO_BINS,
    )
    assert int(pbo.histogram.sum()) == 252


def test_dense_bootstrap_cache_preserves_the_declared_circular_samples() -> None:
    length = 123
    raw = np.linspace(-0.02, 0.03, length)
    accumulator = BootstrapAccumulator.create(length)
    starts = shared_bootstrap_starts(length)
    expected = []
    blocks = int(np.ceil(length / BLOCK_LENGTH))
    for repetition in range(20):
        sampled = []
        for block in range(blocks):
            sampled.extend(
                raw[(starts[repetition, block] + np.arange(BLOCK_LENGTH)) % length]
            )
        expected.append(float(np.mean(sampled[:length])))
    observed = accumulator.weights[:20] @ raw.astype(np.float32)
    np.testing.assert_allclose(observed, expected, atol=1e-8, rtol=1e-6)
    np.testing.assert_allclose(accumulator.weights.sum(axis=1), 1.0, atol=1e-6)
    shared = BootstrapAccumulator.with_shared_weights(accumulator.weights)
    assert np.shares_memory(shared.weights, accumulator.weights)
    assert shared.update(raw) == pytest.approx(accumulator.update(raw))


def test_prior_multiplicity_loader_includes_v1_v2_and_autonomous(tmp_path: Path) -> None:
    prior = tmp_path / "prior"
    historical = prior / "historical_multiplicity"
    historical.mkdir(parents=True)
    dates = pd.date_range("2000-01-03", periods=3, freq="B")
    pd.DataFrame(
        {
            "strategy_id": ["AUTO-1"] * 3,
            "date": dates,
            "return": [0.01, -0.01, 0.02],
        }
    ).to_parquet(prior / "cumulative_autonomous_train_returns.parquet", index=False)
    for campaign, unit, filename, count in (
        ("V1", "STRAT0001", "v1_train_daily_returns.parquet", 3),
        ("V2", "V2STRAT0001", "v2_train_daily_returns.parquet", 2),
    ):
        pd.DataFrame(
            {
                "unit_key": [unit] * count,
                "date": dates[:count],
                "return": np.linspace(0.001, 0.003, count),
                "position": np.ones(count),
            }
        ).to_parquet(historical / filename, index=False)
    ledger = pd.DataFrame(
        [
            {
                "campaign": np.nan,
                "status": "evaluated",
                "strategy_id": "AUTO-1",
                "source_strategy_id": "AUTO-1",
            },
            {
                "campaign": "V1",
                "status": "evaluated",
                "strategy_id": "V1::STRAT0001",
                "source_strategy_id": "STRAT0001",
            },
            {
                "campaign": "V2",
                "status": "evaluated",
                "strategy_id": "V2::V2STRAT0001",
                "source_strategy_id": "V2STRAT0001",
            },
        ]
    )
    loaded = _load_evaluated_returns(prior, ledger)
    assert set(loaded["strategy_id"]) == {
        "AUTO-1",
        "V1::STRAT0001",
        "V2::V2STRAT0001",
    }
    wide = loaded.pivot(index="date", columns="strategy_id", values="return")
    assert len(wide.dropna(axis=0, how="any")) == 2


def test_multiplicity_common_dates_roundtrip_uses_explicit_nanoseconds(
    tmp_path: Path,
) -> None:
    dates = pd.bdate_range("2004-05-03", "2010-12-30")
    path = tmp_path / "multiplicity_common_dates.npy"
    written = _write_multiplicity_common_dates(path, dates)
    raw = np.load(path, allow_pickle=False)
    assert raw.dtype == np.int64
    assert raw[0] == pd.Timestamp(dates[0]).value
    assert raw[-1] == pd.Timestamp(dates[-1]).value
    assert written.dtype == "datetime64[ns]"


def test_multiplicity_common_dates_loader_rejects_missing_or_wrong_unit(
    tmp_path: Path,
) -> None:
    dates = pd.date_range("2004-05-03", periods=1500, freq="B")
    manifest = {
        "multiplicity_common_interval": {
            "sessions": len(dates),
            "start": dates[0].date().isoformat(),
            "end": dates[-1].date().isoformat(),
            "numpy_datetime_unit": "ns",
        }
    }
    (tmp_path / "massive_train_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    np.save(
        tmp_path / "multiplicity_common_dates.npy",
        dates.as_unit("ns").asi8,
    )
    loaded = _load_multiplicity_common_dates(tmp_path, dates)
    assert loaded.equals(dates.as_unit("ns"))

    np.save(
        tmp_path / "multiplicity_common_dates.npy",
        dates.as_unit("us").asi8,
    )
    with pytest.raises(RuntimeError, match="MASSIVE_MULTIPLICITY_COMMON_INTERVAL_BREACH"):
        _load_multiplicity_common_dates(tmp_path, dates)

    manifest["multiplicity_common_interval"].pop("numpy_datetime_unit")
    (tmp_path / "massive_train_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="MASSIVE_MULTIPLICITY_DATETIME_UNIT_BREACH"):
        _load_multiplicity_common_dates(tmp_path, dates)


def _fake_worker(root: Path, name: str, candidate: str, digest: bytes, cagr: float) -> None:
    target = root / name
    target.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "strategy_id": candidate,
                "canonical_hash": digest.hex(),
                "train_oof_cagr_pct": cagr,
            }
        ]
    ).to_csv(target / "top_candidates.csv", index=False)
    pd.DataFrame(columns=["strategy_id", "canonical_hash", "train_oof_cagr_pct"]).to_csv(
        target / "non_global_gate_candidates.csv", index=False
    )
    np.save(target / "canonical_hashes.npy", np.asarray([digest], dtype="S32"))
    np.save(target / "evaluated_hashes.npy", np.asarray([digest], dtype="S32"))
    np.save(target / "raw_pvalues.npy", np.asarray([0.05]))
    np.savez_compressed(
        target / "bootstrap_accumulator.npz",
        white_max=np.zeros(5000),
        spa_max=np.zeros(5000),
        observed_max=np.asarray([0.01]),
        observed_spa_max=np.asarray([1.0]),
    )
    np.savez_compressed(
        target / "pbo_accumulator.npz",
        histogram=np.ones((252, PBO_BINS), dtype=np.uint32),
        best_is=np.ones(252),
        best_oos=np.ones(252),
        best_ids=np.asarray([candidate] * 252, dtype="U64"),
    )
    (target / "summary.json").write_text(
        json.dumps({"generated": 1, "evaluated": 1, "locked_opened": False}),
        encoding="utf-8",
    )


def test_hierarchical_aggregate_preserves_counts_and_rows(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _fake_worker(source, "a", "A", b"a" * 32, 21.0)
    _fake_worker(source, "b", "B", b"b" * 32, 22.0)
    output = tmp_path / "output"
    summary = aggregate(source, output, retain_top=10)
    assert summary["generated"] == 2
    assert summary["evaluated"] == 2
    assert summary["unique_effective_generated"] == 2
    assert len(pd.read_csv(output / "top_candidates.csv")) == 2
    assert len(np.load(output / "raw_pvalues.npy")) == 2


def test_declared_prior_duplicates_still_count_for_multiplicity() -> None:
    ledger = pd.DataFrame(
        {
            "canonical_hash": ["61" * 32, "61" * 32, "62" * 32, "63" * 32],
            "status": ["evaluated", "evaluated", "evaluated", "rejected"],
        }
    )
    all_hashes = np.asarray([b"a" * 32, b"b" * 32, b"c" * 32, b"d" * 32], dtype="S32")
    evaluated_hashes = np.asarray([b"a" * 32, b"b" * 32, b"d" * 32], dtype="S32")
    counts = _multiplicity_counts(ledger, all_hashes, evaluated_hashes)
    assert counts["prior_declared_trials"] == 4
    assert counts["prior_unique_rules"] == 3
    assert counts["trials_for_multiplicity"] == 5
    assert counts["evaluated_streams"] == 4


def test_pvalues_remain_attached_to_their_canonical_hash(tmp_path: Path) -> None:
    hashes_path = tmp_path / "evaluated_hashes.npy"
    np.save(hashes_path, np.asarray([b"b" * 32, b"a" * 32], dtype="S32"))
    np.save(tmp_path / "raw_pvalues.npy", np.asarray([0.20, 0.10]))
    hashes, pvalues = _unique_hash_values([hashes_path], "raw_pvalues.npy")
    assert hashes.tolist() == [b"a" * 32, b"b" * 32]
    assert pvalues is not None
    np.testing.assert_allclose(pvalues, [0.10, 0.20])


def test_massive_workflow_is_github_only_360_parallel_and_seven_waves() -> None:
    caller_path = ROOT / ".github" / "workflows" / "sp500-autonomous-discovery.yml"
    wave_path = ROOT / ".github" / "workflows" / "_sp500-massive-train-wave.yml"
    caller_text = caller_path.read_text(encoding="utf-8")
    wave_text = wave_path.read_text(encoding="utf-8")
    caller = yaml.safe_load(caller_text)
    wave = yaml.safe_load(wave_text)
    caller_jobs = caller["jobs"]
    assert caller_jobs["massive_wave_0"]["needs"] == "massive_prepare"
    assert "needs.massive_prepare.result == 'success'" in caller_jobs["massive_wave_0"]["if"]
    jobs = wave["jobs"]
    assert len(jobs["shard_a"]["strategy"]["matrix"]["shard"]) == 256
    assert len(jobs["shard_b"]["strategy"]["matrix"]["shard"]) == 104
    assert jobs["shard_a"]["strategy"]["max-parallel"] == 256
    assert jobs["shard_b"]["strategy"]["max-parallel"] == 104
    assert jobs["recover_missing_a"]["strategy"]["max-parallel"] == 256
    assert jobs["recover_missing_b"]["strategy"]["max-parallel"] == 104
    assert jobs["merge_block"]["needs"] == [
        "recovery_plan",
        "recover_missing_a",
        "recover_missing_b",
    ]
    assert all(jobs[name]["runs-on"] == "ubuntu-24.04" for name in jobs)
    assert caller_text.count("minutes_per_shard: 50") == 7
    assert all(f"massive_wave_{wave_id}:" in caller_text for wave_id in range(7))
    assert "inputs.batch_id != 1000" in caller_text
    assert "inputs.batch_id == 1000" in caller_text
    assert "self-hosted" not in caller_text + wave_text
    assert "C:\\" not in caller_text + wave_text
    assert "--mode final" in caller_text
    assert "refinement" not in caller_text + wave_text
    assert "validation_opened\"] is False" in caller_text
    assert "locked_opened\"] is False" in caller_text
