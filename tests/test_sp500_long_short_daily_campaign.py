from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import requests
import yaml

from aurora.infra.sp500_long_short_daily.contracts import (
    EXPECTED_CANDIDATES,
    EXPECTED_FAMILIES,
    EXPECTED_FEATURES,
    CampaignPackage,
    LockedBoundaryError,
    canonical_json_hash,
    candidate_canonical_payload,
)
from aurora.infra.sp500_long_short_daily.ledger import (
    apply_positions,
    build_total_return_ledger,
)
from aurora.infra.sp500_long_short_daily.data import (
    DataGateError,
    DownloadReceipt,
    PreparedMarketData,
    _align_initial_releases,
    _adjudicate_stooq_open_prices,
    _download_stooq_html_history,
    _load_stooq_history_page,
    _parse_stooq_html_history,
    _request_stooq_history_page,
    _parse_yahoo_chart,
    _parse_kibot_daily_history,
    _reconcile_spy_sources,
    _repo_campaign_root,
    _solve_stooq_browser_verification,
    download_stooq_history,
    download_alfred_initial_series,
    load_cboe_vxo_history,
    load_sec_distribution_totals,
    load_state_street_distributions,
    reconcile_official_distribution_audit,
    reconcile_sponsor_distributions,
    write_fixture_snapshot,
    load_market_snapshot,
)
from aurora.infra.sp500_long_short_daily.signals import (
    CandidateRejected,
    EXPLICIT_FAMILY_REJECTIONS,
    IMPLEMENTED_FAMILIES,
    _frozen_component_ids,
    benchmark_decisions,
    candidate_decisions,
)
from aurora.infra.sp500_long_short_daily.workload import (
    PILOT_WORKLOAD,
    SMOKE_WORKLOAD,
    TRAIN_WORKLOAD,
)
from aurora.infra.sp500_long_short_daily.statistics import (
    _benjamini_hochberg,
    _stationary_bootstrap_indices,
    effective_independent_trials,
    reality_check_and_spa,
)
from aurora.infra.sp500_long_short_daily.validation import (
    DIAGNOSTIC_VALIDATION_ACK,
    VALIDATION_ACK,
    ValidationGateError,
    build_diagnostic_train_freeze,
    combine_phase_snapshots,
    run_validation_once,
    verify_train_freeze,
)
from aurora.infra.github_performance.contracts import RunSpec
from aurora.infra.github_performance.preflight import validate_run_spec
from scripts.merge_sp500_stooq_windows import merge_windows
from scripts.download_sp500_stooq_window import download_window


REPO_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ROOT = REPO_ROOT / "campaigns" / "sp500_long_short_daily"


def _campaign() -> CampaignPackage:
    return CampaignPackage.load(
        CAMPAIGN_ROOT / "research_input",
        CAMPAIGN_ROOT / "input_package" / "SP500_LONG_SHORT_DIARIO_RESEARCH_AURORA_FINAL.zip",
    )


def test_campaign_root_prefers_explicit_github_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = tmp_path / "campaigns" / "sp500_long_short_daily"
    (expected / "official_inputs").mkdir(parents=True)
    monkeypatch.setenv("GITHUB_WORKSPACE", str(tmp_path))
    assert _repo_campaign_root() == expected.resolve()


def _prices() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-12-23", "2020-12-24", "2020-12-28", "2020-12-29"]),
            "open": [100.0, 101.0, 50.0, 51.0],
            "high": [102.0, 102.0, 51.0, 52.0],
            "low": [99.0, 100.0, 49.0, 50.0],
            "close": [101.0, 101.5, 50.5, 51.5],
            "volume": [10, 11, 22, 23],
        }
    )


def test_exact_package_contract_and_counts() -> None:
    package = _campaign()
    assert len(package.candidates) == EXPECTED_CANDIDATES
    assert len(package.features) == EXPECTED_FEATURES
    assert len({row["family"] for row in package.candidates}) == EXPECTED_FAMILIES
    assert len(package.spec["benchmarks"]) == 5
    assert all(row["position_values"] == [-1, 1] for row in package.candidates)


def test_candidate_hashes_are_exact_and_unique() -> None:
    package = _campaign()
    observed = []
    for candidate in package.candidates:
        expected = canonical_json_hash(candidate_canonical_payload(candidate))
        assert candidate["canonical_hash"] == expected
        observed.append(expected)
    assert len(observed) == len(set(observed)) == 168


def test_all_positions_and_costs_are_frozen() -> None:
    package = _campaign()
    costs = {
        "commission_bps",
        "slippage_bps",
        "borrow_cost_bps",
        "financing_bps",
        "switching_cost_bps",
        "market_impact_bps",
    }
    for candidate in package.candidates:
        assert candidate["position_values"] == [-1, 1]
        assert candidate["absolute_exposure"] == 1.0
        assert all(candidate[key] == 0 for key in costs)


def test_dividend_and_split_open_to_open_hand_calculation() -> None:
    distributions = pd.DataFrame({"date": [pd.Timestamp("2020-12-24")], "distribution": [1.0]})
    splits = pd.DataFrame({"date": [pd.Timestamp("2020-12-28")], "split_ratio": [2.0]})
    ledger, audit = build_total_return_ledger(_prices(), distributions, splits)
    assert audit.distribution_count == 1
    assert audit.split_count == 1
    assert audit.long_short_max_abs_error == 0.0
    expected_first = ((101.0 / 2.0) + (1.0 / 2.0) - (100.0 / 2.0)) / (100.0 / 2.0)
    assert ledger.loc["2020-12-23", "long_return"] == pytest.approx(expected_first)
    assert np.allclose(
        ledger["short_return"].dropna(),
        -ledger["long_return"].dropna(),
        rtol=0,
        atol=0,
    )
    assert ledger["tr_open"].iloc[0] == 1.0
    assert ledger["tr_open"].iloc[1] == pytest.approx(1.0 + expected_first)


def test_adjusted_close_is_never_used_as_an_open_price() -> None:
    prices = _prices().assign(adj_close=[1.0, 2.0, 3.0, 4.0])
    plain, _ = build_total_return_ledger(_prices())
    adjusted_column_present, _ = build_total_return_ledger(prices)
    pd.testing.assert_series_equal(plain["long_return"], adjusted_column_present["long_return"])


def test_yahoo_chart_parser_preserves_raw_prices_events_and_bounds() -> None:
    timestamps = [
        int(pd.Timestamp("2009-03-19", tz="UTC").timestamp()),
        int(pd.Timestamp("2009-03-20", tz="UTC").timestamp()),
    ]
    payload = json.dumps(
        {
            "chart": {
                "error": None,
                "result": [
                    {
                        "meta": {
                            "regularMarketTime": int(
                                pd.Timestamp("2026-08-04", tz="UTC").timestamp()
                            ),
                            "regularMarketPrice": 999.0,
                        },
                        "timestamp": timestamps,
                        "indicators": {
                            "quote": [
                                {
                                    "open": [80.0, 81.0],
                                    "high": [82.0, 83.0],
                                    "low": [79.0, 80.0],
                                    "close": [81.0, 82.0],
                                    "volume": [1000, 1200],
                                }
                            ],
                            "adjclose": [{"adjclose": [75.0, 76.0]}],
                        },
                        "events": {
                            "dividends": {
                                str(timestamps[1]): {
                                    "date": timestamps[1],
                                    "amount": 0.56172,
                                }
                            },
                            "splits": {
                                str(timestamps[0]): {
                                    "date": timestamps[0],
                                    "numerator": 2.0,
                                    "denominator": 1.0,
                                }
                            },
                        },
                    }
                ],
            }
        }
    ).encode()

    prices, dividends, splits, bounded = _parse_yahoo_chart(
        payload,
        start=pd.Timestamp("2009-03-19"),
        end=pd.Timestamp("2009-03-20"),
    )

    assert prices["close"].tolist() == [81.0, 82.0]
    assert prices["adj_close"].tolist() == [75.0, 76.0]
    assert dividends.to_dict(orient="records") == [
        {"date": pd.Timestamp("2009-03-20"), "distribution": 0.56172}
    ]
    assert splits.to_dict(orient="records") == [
        {"date": pd.Timestamp("2009-03-19"), "split_ratio": 2.0}
    ]
    bounded_document = json.loads(bounded)
    assert "meta" not in bounded_document
    assert bounded_document["prices"][-1]["date"] == "2009-03-20"
    assert "2026" not in bounded.decode()


def test_yahoo_chart_parser_rejects_observations_outside_requested_period() -> None:
    timestamp = int(pd.Timestamp("2011-01-03", tz="UTC").timestamp())
    payload = json.dumps(
        {
            "chart": {
                "error": None,
                "result": [
                    {
                        "timestamp": [timestamp],
                        "indicators": {
                            "quote": [
                                {
                                    "open": [100.0],
                                    "high": [101.0],
                                    "low": [99.0],
                                    "close": [100.5],
                                    "volume": [1000],
                                }
                            ],
                            "adjclose": [{"adjclose": [100.5]}],
                        },
                    }
                ],
            }
        }
    ).encode()
    with pytest.raises(DataGateError, match="UNBOUNDED_SOURCE_RESPONSE:yahoo_history"):
        _parse_yahoo_chart(
            payload,
            start=pd.Timestamp("2010-01-01"),
            end=pd.Timestamp("2010-12-31"),
        )


def test_stooq_javascript_verification_is_solved_without_browser() -> None:
    challenge = "unit-test-challenge"
    payload = (
        '<script>(async()=>{const c="'
        + challenge
        + '",d=2,t="0".repeat(d);'
        + 'crypto.subtle.digest("SHA-256");'
        + 'await fetch("/__verify")})();</script>'
    ).encode()

    class Response:
        def raise_for_status(self) -> None:
            return None

    class Session:
        posted: dict[str, str] | None = None

        def post(self, url: str, *, data: dict[str, str], timeout: int) -> Response:
            assert url == "https://stooq.com/__verify"
            assert timeout == 60
            self.posted = data
            return Response()

    session = Session()
    assert _solve_stooq_browser_verification(session, payload)
    assert session.posted is not None
    solved = hashlib.sha256(f"{challenge}{session.posted['n']}".encode()).hexdigest()
    assert solved.startswith("00")


def _stooq_html(rows: list[tuple[str, ...]], *, pages: int) -> bytes:
    body = "".join(
        "<tr>" + "".join(f"<td>{value}</td>" for value in row) + "</tr>"
        for row in rows
    )
    return (
        "<html><body><table>"
        "<tr><th>No.</th><th>Date</th><th>Open</th><th>High</th><th>Low</th>"
        "<th>Close</th><th>Change</th><th>Move</th><th>Volume</th></tr>"
        f"{body}</table><a href='q/d/?s=spy.us&amp;i=d&amp;l={pages}'>&gt;&gt;</a>"
        "</body></html>"
    ).encode()


def test_stooq_html_history_parser_reads_ohlcv_and_page_count() -> None:
    payload = _stooq_html(
        [
            ("2", "31 Dec 2010", "96.8014", "97.0694", "96.7251", "97.0026", "+0.03%", "+0.0291", "78,672,053"),
            ("1", "30 Dec 2010", "97.0404", "97.2702", "96.8303", "96.9735", "-0.13%", "-0.1251", "99,285,113"),
        ],
        pages=3,
    )
    frame, pages = _parse_stooq_html_history(payload)
    assert pages == 3
    assert frame["Date"].tolist() == [pd.Timestamp("2010-12-31"), pd.Timestamp("2010-12-30")]
    assert frame["Open"].tolist() == pytest.approx([96.8014, 97.0404])
    assert frame["Volume"].tolist() == [78_672_053, 99_285_113]


def test_stooq_html_history_parser_reports_daily_hit_limit() -> None:
    payload = b"<html><body>Exceeded the daily site hits limit</body></html>"
    with pytest.raises(DataGateError, match="STOOQ_DAILY_HITS_LIMIT"):
        _parse_stooq_html_history(payload)


def test_stooq_github_transport_uses_headless_browser(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload = _stooq_html(
        [("1", "31 Dec 2010", "96.8", "97.1", "96.7", "97.0", "+0.03%", "+0.03", "78,672,053")],
        pages=1,
    )
    commands: list[list[str]] = []

    monkeypatch.setenv("GITHUB_ACTIONS", "false")
    monkeypatch.setattr(
        "aurora.infra.sp500_long_short_daily.data.shutil.which",
        lambda executable: "/usr/bin/google-chrome" if executable == "google-chrome" else None,
    )

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        commands.append(command)
        assert kwargs["capture_output"] is True
        return subprocess.CompletedProcess(command, 0, stdout=payload, stderr=b"")

    monkeypatch.setattr(
        "aurora.infra.sp500_long_short_daily.data.subprocess.run",
        fake_run,
    )
    result = _request_stooq_history_page(
        requests.Session(),
        {"s": "spy.us", "d1": "20041001", "d2": "20070930", "i": "d"},
        browser_profile=tmp_path,
    )
    assert result == payload
    assert len(commands) == 1
    assert "--headless=new" in commands[0]
    assert commands[0][-1].startswith("https://stooq.com/q/d/?")


def test_stooq_github_transport_preserves_form_session_via_cdp(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload = _stooq_html(
        [("1", "16 Jul 2009", "92.1", "93.1", "91.8", "92.9", "+1.1%", "+1.0", "80,000,000")],
        pages=1,
    )
    seen: dict[str, object] = {}
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setattr(
        "aurora.infra.sp500_long_short_daily.data.shutil.which",
        lambda executable: "/usr/bin/google-chrome" if executable == "google-chrome" else None,
    )

    def fake_cdp(
        browser: str,
        history_url: str,
        browser_profile: Path,
        *,
        symbol: str,
    ) -> bytes:
        seen.update(
            browser=browser,
            history_url=history_url,
            browser_profile=browser_profile,
            symbol=symbol,
        )
        return payload

    monkeypatch.setattr(
        "aurora.infra.sp500_long_short_daily.data._request_stooq_history_page_via_cdp",
        fake_cdp,
    )
    result = _request_stooq_history_page(
        requests.Session(),
        {"s": "spy.us", "f": "20090701", "t": "20090731", "o": "1111111"},
        browser_profile=tmp_path,
    )
    assert result == payload
    assert seen["browser"] == "/usr/bin/google-chrome"
    assert seen["browser_profile"] == tmp_path
    assert seen["symbol"] == "spy.us"
    assert "f=20090701" in str(seen["history_url"])
    assert "o=1111111" in str(seen["history_url"])


def test_stooq_page_loader_retries_transient_verification_screen(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    valid_payload = _stooq_html(
        [("1", "31 Dec 2010", "96.8", "97.1", "96.7", "97.0", "+0.03%", "+0.03", "78,672,053")],
        pages=1,
    )
    payloads = iter([b"<html>verification required</html>", valid_payload])
    calls = 0

    def fake_request(*args: object, **kwargs: object) -> bytes:
        nonlocal calls
        del args, kwargs
        calls += 1
        return next(payloads)

    monkeypatch.setattr(
        "aurora.infra.sp500_long_short_daily.data._request_stooq_history_page",
        fake_request,
    )
    monkeypatch.setattr("aurora.infra.sp500_long_short_daily.data.time.sleep", lambda _: None)
    payload, frame, page_count = _load_stooq_history_page(
        requests.Session(),
        {"s": "spy.us", "i": "d", "l": 26},
        browser_profile=tmp_path,
    )
    assert calls == 2
    assert payload == valid_payload
    assert page_count == 1
    assert frame["Date"].tolist() == [pd.Timestamp("2010-12-31")]


def test_stooq_page_loader_uses_bounded_transient_retry_budget(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls = 0

    def fake_request(*args: object, **kwargs: object) -> bytes:
        nonlocal calls
        del args, kwargs
        calls += 1
        return b"<html>verification required</html>"

    monkeypatch.setattr(
        "aurora.infra.sp500_long_short_daily.data._request_stooq_history_page",
        fake_request,
    )
    monkeypatch.setattr("aurora.infra.sp500_long_short_daily.data.time.sleep", lambda _: None)
    with pytest.raises(DataGateError, match="STOOQ_HTML_HISTORY_ROWS_NOT_FOUND"):
        _load_stooq_history_page(
            requests.Session(),
            {"s": "spy.us", "i": "d"},
            browser_profile=tmp_path,
        )
    assert calls == 3


def test_stooq_download_uses_bounded_public_html(tmp_path: Path) -> None:
    pages = {
        1: _stooq_html(
            [("3", "31 Dec 2010", "96.8", "97.1", "96.7", "97.0", "+0.03%", "+0.03", "78,672,053")],
            pages=2,
        ),
        2: _stooq_html(
            [
                ("2", "30 Dec 2010", "97.0", "97.3", "96.8", "96.9", "-0.13%", "-0.13", "99,285,113"),
                ("1", "29 Dec 2010", "97.1", "97.4", "97.0", "97.1", "+0.05%", "+0.05", "75,179,103"),
            ],
            pages=1,
        ),
    }

    class Response:
        def __init__(self, content: bytes) -> None:
            self.content = content

        def raise_for_status(self) -> None:
            return None

    class Session:
        def __init__(self) -> None:
            self.headers: dict[str, str] = {}
            self.seen_params: list[dict[str, object]] = []

        def get(self, url: str, *, params=None, timeout: int) -> Response:
            del url, timeout
            self.seen_params.append(dict(params or {}))
            return Response(pages[int((params or {}).get("l", 1))])

    session = Session()
    frame, receipt = download_stooq_history(
        "spy.us",
        "2010-12-29",
        "2010-12-31",
        split="train",
        session=session,
        raw_dir=tmp_path,
    )
    assert frame["date"].tolist() == list(pd.date_range("2010-12-29", "2010-12-31"))
    assert receipt.status == "downloaded_bounded_html_public_history_raw_unadjusted"
    assert receipt.minimum_date == "2010-12-29"
    assert receipt.maximum_date == "2010-12-31"
    assert receipt.reason is not None and "window_count=1;page_count=2" in receipt.reason
    for params in session.seen_params:
        assert params["c"] == "0"
        assert params["o"] == "1111111"
        assert all(params[f"o_{suffix}"] == "1" for suffix in "sdpnomx")
        assert "i" not in params
        assert "f" in params and "t" in params
        assert "d1" not in params and "d2" not in params
    assert (tmp_path / "stooq_spy_us_history.csv").is_file()


def test_stooq_download_chunks_long_history_below_public_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dates = pd.bdate_range(end="2008-12-31", periods=1000)

    class Response:
        def __init__(self, content: bytes) -> None:
            self.content = content

        def raise_for_status(self) -> None:
            return None

    class Session:
        def __init__(self) -> None:
            self.headers: dict[str, str] = {}
            self.window_sizes: list[int] = []

        def get(self, url: str, *, params=None, timeout: int) -> Response:
            del url, timeout
            values = params or {}
            start = pd.Timestamp(str(values.get("d1") or values.get("f")))
            end = pd.Timestamp(str(values.get("d2") or values.get("t")))
            bounded = dates[(dates >= start) & (dates <= end)][::-1]
            if "l" not in values:
                self.window_sizes.append(len(bounded))
            page = int(values.get("l", 1))
            page_count = max(1, int(np.ceil(len(bounded) / 40)))
            selected = bounded[(page - 1) * 40 : page * 40]
            rows: list[tuple[str, ...]] = [
                (
                    str((page - 1) * 40 + slot + 1),
                    pd.Timestamp(date).strftime("%d %b %Y"),
                    "100.0",
                    "101.0",
                    "99.0",
                    "100.5",
                    "+0.5%",
                    "+0.5",
                    "1000000",
                )
                for slot, date in enumerate(selected)
            ]
            return Response(_stooq_html(rows, pages=page_count))

    monkeypatch.setattr("aurora.infra.sp500_long_short_daily.data.time.sleep", lambda _: None)
    session = Session()
    frame, receipt = download_stooq_history(
        "spy.us",
        dates.min().date().isoformat(),
        dates.max().date().isoformat(),
        split="train",
        session=session,
        raw_dir=tmp_path,
    )
    assert len(frame) == 1000
    assert frame["date"].min() == dates.min()
    assert frame["date"].max() == dates.max()
    assert len(session.window_sizes) == 2
    assert max(session.window_sizes) < 1000
    assert receipt.reason is not None and "window_count=2" in receipt.reason
    assert receipt.reason is not None and "page_count=26" in receipt.reason


def test_stooq_github_download_uses_verified_http_session_per_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profiles: list[Path | None] = []
    sleeps: list[float] = []

    def fake_load(
        client: requests.Session,
        params: dict[str, object],
        *,
        browser_profile: Path | None,
        attempts: int = 3,
    ) -> tuple[bytes, pd.DataFrame, int]:
        del client, attempts
        profiles.append(browser_profile)
        date = pd.Timestamp(str(params["f"]))
        frame = pd.DataFrame(
            {
                "Date": [date],
                "Open": [100.0],
                "High": [101.0],
                "Low": [99.0],
                "Close": [100.5],
                "Volume": [1_000_000],
            }
        )
        return str(date.date()).encode(), frame, 1

    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setattr(
        "aurora.infra.sp500_long_short_daily.data.time.sleep",
        sleeps.append,
    )
    monkeypatch.setattr(
        "aurora.infra.sp500_long_short_daily.data._load_stooq_history_page",
        fake_load,
    )
    frame, _, _, page_count, window_count = _download_stooq_html_history(
        requests.Session(),
        "spy.us",
        pd.Timestamp("2005-10-01"),
        pd.Timestamp("2009-09-30"),
    )
    assert len(frame) == 2
    assert page_count == 2
    assert window_count == 2
    assert len(profiles) == 2
    assert profiles == [None, None]
    assert sleeps == []


def test_stooq_prebuilt_sharded_input_is_hash_bound_and_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    csv_path = tmp_path / "stooq_spy_us_history.csv"
    csv_path.write_text(
        "date,open,high,low,close,volume\n"
        "2009-01-02,90,91,89,90.5,1000000\n"
        "2009-01-05,91,92,90,91.5,1100000\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "stooq_sharded_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "window_count": 2,
                "merged_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
                "locked_opened": False,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("SP500_STOOQ_HISTORY_CSV", str(csv_path))
    monkeypatch.setenv("SP500_STOOQ_HISTORY_MANIFEST", str(manifest_path))
    frame, receipt = download_stooq_history(
        "spy.us",
        "2009-01-01",
        "2009-01-06",
        split="train",
        raw_dir=tmp_path / "raw",
    )
    assert len(frame) == 2
    assert frame["date"].max() == pd.Timestamp("2009-01-05")
    assert receipt.status == "loaded_github_sharded_html_history_raw_unadjusted"
    assert receipt.reason is not None and "window_count=2" in receipt.reason

    manifest_path.write_text(
        json.dumps({"merged_sha256": "0" * 64}),
        encoding="utf-8",
    )
    with pytest.raises(DataGateError, match="STOOQ_PREBUILT_HASH_MISMATCH"):
        download_stooq_history(
            "spy.us",
            "2009-01-01",
            "2009-01-06",
            split="train",
        )


def test_stooq_window_merge_preserves_rows_hashes_and_locked_boundary(tmp_path: Path) -> None:
    input_root = tmp_path / "windows"
    for window_id, row in (
        ("000", "2009-01-02,90,91,89,90.5,1000000\n"),
        ("001", "2009-01-05,91,92,90,91.5,1100000\n"),
    ):
        root = input_root / f"window-{window_id}"
        root.mkdir(parents=True)
        (root / "stooq_spy_us_history.csv").write_text(
            "date,open,high,low,close,volume\n" + row,
            encoding="utf-8",
        )
        (root / "stooq_window_receipt.json").write_text(
            json.dumps(
                {
                    "window_id": window_id,
                    "requested_start": "2009-01-01",
                    "requested_end": "2009-01-06",
                    "effective_source": (
                        "yahoo_chart_raw_unadjusted_fallback"
                        if window_id == "001"
                        else "stooq_public_html_raw_unadjusted"
                    ),
                    "receipt": {
                        "dataset_id": "DS002",
                        "status": (
                            "downloaded_documented_free_fallback_yahoo_raw_unadjusted"
                            if window_id == "001"
                            else "downloaded_bounded_html_public_history_raw_unadjusted"
                        ),
                    },
                }
            ),
            encoding="utf-8",
        )
    output = tmp_path / "merged"
    manifest = merge_windows(
        input_root,
        output,
        expected_windows=2,
        requested_start="2009-01-01",
        requested_end="2009-01-06",
    )
    merged = pd.read_csv(output / "stooq_spy_us_history.csv")
    assert len(merged) == 2
    assert manifest["rows"] == 2
    assert manifest["window_count"] == 2
    assert manifest["locked_opened"] is False
    assert manifest["source"] == "mixed_stooq_and_documented_yahoo_fallback_raw_unadjusted"
    assert manifest["source_counts"] == {
        "yahoo_chart_raw_unadjusted_fallback": 1,
        "stooq_public_html_raw_unadjusted": 1,
    }
    assert manifest["merged_sha256"] == hashlib.sha256(
        (output / "stooq_spy_us_history.csv").read_bytes()
    ).hexdigest()


def test_yahoo_window_merge_preserves_adjusted_close_and_corporate_events(
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "windows"
    windows = (
        (
            "000",
            "2009-03-19,80,81,79,80,79.43828,1000000\n"
            "2009-03-20,79.5,80.5,78.5,79.5,79.5,1100000\n",
            "2009-03-20,0.56172\n",
        ),
        (
            "001",
            "2009-03-23,80.5,81.5,79.5,80.5,80.5,1200000\n",
            "",
        ),
    )
    for window_id, rows, distributions in windows:
        root = input_root / f"window-{window_id}"
        root.mkdir(parents=True)
        (root / "stooq_spy_us_history.csv").write_text(
            "date,open,high,low,close,adj_close,volume\n" + rows,
            encoding="utf-8",
        )
        (root / "spy_distributions.csv").write_text(
            "date,distribution\n" + distributions,
            encoding="utf-8",
        )
        (root / "spy_splits.csv").write_text(
            "date,split_ratio\n",
            encoding="utf-8",
        )
        (root / "stooq_window_receipt.json").write_text(
            json.dumps(
                {
                    "schema_version": "3",
                    "window_id": window_id,
                    "requested_start": "2009-03-19",
                    "requested_end": "2009-03-23",
                    "effective_source": "yahoo_chart_adjusted_close_with_events",
                    "receipt": {
                        "dataset_id": "DS002",
                        "status": "downloaded_documented_free_yahoo_adjusted_close_with_events",
                    },
                }
            ),
            encoding="utf-8",
        )

    output = tmp_path / "merged"
    manifest = merge_windows(
        input_root,
        output,
        expected_windows=2,
        requested_start="2009-03-19",
        requested_end="2009-03-23",
    )

    merged = pd.read_csv(output / "stooq_spy_us_history.csv")
    events = pd.read_csv(output / "spy_distributions.csv")
    assert "adj_close" in merged
    assert len(merged) == 3
    assert events.to_dict(orient="records") == [
        {"date": "2009-03-20", "distribution": 0.56172}
    ]
    assert manifest["adjusted_close_complete"] is True
    assert manifest["distribution_event_count"] == 1
    assert manifest["split_event_count"] == 0
    corporate_action_hashes = manifest["corporate_action_files_sha256"]
    assert isinstance(corporate_action_hashes, dict)
    assert corporate_action_hashes["distributions"] == hashlib.sha256(
        (output / "spy_distributions.csv").read_bytes()
    ).hexdigest()


@pytest.mark.parametrize(
    "provider_error",
    ["STOOQ_DAILY_HITS_LIMIT", "STOOQ_HTML_HISTORY_ROWS_NOT_FOUND"],
)
def test_stooq_window_uses_documented_yahoo_fallback_for_provider_unavailability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    provider_error: str,
) -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2009-01-02", "2009-01-05"]),
            "open": [90.0, 91.0],
            "high": [91.0, 92.0],
            "low": [89.0, 90.0],
            "close": [90.5, 91.5],
            "adj_close": [90.5, 91.5],
            "volume": [1_000_000, 1_100_000],
        }
    )
    dividends = pd.DataFrame(
        {"date": pd.to_datetime(["2009-01-05"]), "distribution": [0.5]}
    )
    splits = pd.DataFrame(columns=["date", "split_ratio"])
    fallback_receipt = DownloadReceipt(
        dataset_id="YAHOO_SPY_BOUNDED_CHART",
        url_template="https://query1.finance.yahoo.com/v8/finance/chart/SPY",
        sha256="a" * 64,
        byte_count=123,
        minimum_date="2009-01-02",
        maximum_date="2009-01-05",
        status="downloaded_bounded_chart_json_current_metadata_discarded",
    )

    def fail_stooq(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise DataGateError(provider_error)

    def fake_yahoo(
        *args: object,
        **kwargs: object,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, tuple[DownloadReceipt, ...]]:
        del args
        assert kwargs["include_events"] is True
        return frame, dividends, splits, (fallback_receipt,)

    monkeypatch.setattr(
        "scripts.download_sp500_stooq_window.download_stooq_history",
        fail_stooq,
    )
    monkeypatch.setattr(
        "scripts.download_sp500_stooq_window.download_yahoo_history",
        fake_yahoo,
    )
    metadata = download_window(
        start="2009-01-01",
        end="2009-01-06",
        split="train",
        window_id="000",
        output_dir=tmp_path,
    )
    assert metadata["effective_source"] == "yahoo_chart_adjusted_close_with_events"
    receipt = metadata["receipt"]
    assert isinstance(receipt, dict)
    assert receipt["dataset_id"] == "DS002"
    assert receipt["status"] == (
        "downloaded_documented_free_yahoo_adjusted_close_with_events"
    )
    assert f"fallback_for={provider_error}" in str(receipt["reason"])
    written = pd.read_csv(tmp_path / "stooq_spy_us_history.csv")
    assert len(written) == 2
    assert written["date"].tolist() == ["2009-01-02", "2009-01-05"]
    assert written["adj_close"].tolist() == [90.5, 91.5]
    assert pd.read_csv(tmp_path / "spy_distributions.csv").to_dict(
        orient="records"
    ) == [{"date": "2009-01-05", "distribution": 0.5}]
    assert pd.read_csv(tmp_path / "spy_splits.csv").empty


def test_stooq_window_does_not_hide_unexpected_provider_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_stooq(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise DataGateError("STOOQ_HTML_ROW_SCHEMA_MISMATCH")

    monkeypatch.setattr(
        "scripts.download_sp500_stooq_window.download_stooq_history",
        fail_stooq,
    )
    with pytest.raises(DataGateError, match="STOOQ_HTML_ROW_SCHEMA_MISMATCH"):
        download_window(
            start="2009-01-01",
            end="2009-01-06",
            split="train",
            window_id="000",
            output_dir=tmp_path,
        )


def test_stooq_window_can_freeze_confirmed_provider_outage_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2009-01-02"]),
            "open": [90.0],
            "high": [91.0],
            "low": [89.0],
            "close": [90.5],
            "adj_close": [90.5],
            "volume": [1_000_000],
        }
    )
    receipt = DownloadReceipt(
        dataset_id="YAHOO_SPY_BOUNDED_CHART",
        url_template="https://query1.finance.yahoo.com/v8/finance/chart/SPY",
        sha256="b" * 64,
        byte_count=64,
        minimum_date="2009-01-02",
        maximum_date="2009-01-02",
        status="downloaded_bounded_chart_json_current_metadata_discarded",
    )

    def forbidden_stooq(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("Stooq must not be retried in frozen outage mode")

    monkeypatch.setattr(
        "scripts.download_sp500_stooq_window.download_stooq_history",
        forbidden_stooq,
    )
    monkeypatch.setattr(
        "scripts.download_sp500_stooq_window.download_yahoo_history",
        lambda *args, **kwargs: (
            frame,
            pd.DataFrame(columns=["date"]),
            pd.DataFrame(columns=["date"]),
            (receipt,),
        ),
    )
    metadata = download_window(
        start="2009-01-01",
        end="2009-01-06",
        split="train",
        window_id="000",
        output_dir=tmp_path,
        source_mode="yahoo-fallback",
    )
    assert metadata["effective_source"] == "yahoo_chart_adjusted_close_with_events"
    assert "STOOQ_PROVIDER_OUTAGE_CONFIRMED" in str(metadata["receipt"])


def test_next_session_open_crosses_nyse_holiday_without_calendar_day_fill() -> None:
    ledger, _ = build_total_return_ledger(_prices())
    assert ledger.index[1] == pd.Timestamp("2020-12-24")
    assert ledger.index[2] == pd.Timestamp("2020-12-28")
    expected = (50.0 - 101.0) / 101.0
    assert ledger.loc["2020-12-24", "long_return"] == pytest.approx(expected)


def test_official_sponsor_snapshot_and_yahoo_events_must_match(tmp_path: Path) -> None:
    path = tmp_path / "state-street.csv"
    path.write_text(
        "ex_date,distribution\n2009-03-20,0.56172\n2010-03-19,0.48073\n",
        encoding="utf-8",
    )
    sponsor, receipt = load_state_street_distributions(
        path, "2009-01-01", "2010-12-31", split="train"
    )
    assert receipt.dataset_id == "DS001"
    assert receipt.status == "loaded_official_frozen_snapshot"
    audit = reconcile_sponsor_distributions(sponsor, sponsor.copy())
    assert audit["event_count"] == 2
    rounded = sponsor.copy()
    rounded["distribution"] = rounded["distribution"].round(3)
    rounded_audit = reconcile_sponsor_distributions(sponsor, rounded)
    assert rounded_audit["maximum_absolute_amount_difference"] == pytest.approx(0.00028)
    mismatched = sponsor.copy()
    mismatched.loc[0, "distribution"] += 0.01
    with pytest.raises(DataGateError, match="SPONSOR_DISTRIBUTION_AMOUNT_MISMATCH"):
        reconcile_sponsor_distributions(sponsor, mismatched)


def test_layered_official_distribution_audit_covers_every_operational_event(
    tmp_path: Path,
) -> None:
    exact_path = tmp_path / "exact.csv"
    exact_path.write_text(
        "ex_date,distribution\n2006-06-16,0.55\n2006-09-15,0.58\n",
        encoding="utf-8",
    )
    totals_path = tmp_path / "totals.csv"
    totals_path.write_text(
        "period_start,period_end,distribution_total\n2005-10-01,2006-09-30,1.13\n",
        encoding="utf-8",
    )
    exact, _ = load_state_street_distributions(
        exact_path, "2005-10-01", "2010-12-31", split="train"
    )
    totals, _ = load_sec_distribution_totals(totals_path, "2005-10-01", "2010-12-31", split="train")
    audit = reconcile_official_distribution_audit(exact, totals, exact.copy())
    assert audit["operational_event_count"] == 2
    assert audit["uncovered_event_count"] == 0

    uncovered = pd.concat(
        [
            exact,
            pd.DataFrame({"date": pd.to_datetime(["2005-09-16"]), "distribution": [0.10]}),
        ],
        ignore_index=True,
    )
    with pytest.raises(
        DataGateError,
        match="OPERATIONAL_DISTRIBUTION_EVENT_WITHOUT_OFFICIAL_COVERAGE",
    ):
        reconcile_official_distribution_audit(exact, totals, uncovered)


def test_sec_totals_can_cover_window_before_exact_event_export() -> None:
    exact = pd.DataFrame(columns=["date", "distribution"])
    totals = pd.DataFrame(
        {
            "period_start": [pd.Timestamp("1994-01-01")],
            "period_end": [pd.Timestamp("1994-12-31")],
            "distribution_total": [1.23],
        }
    )
    operational = pd.DataFrame(
        {
            "date": pd.to_datetime(["1994-03-18", "1994-06-17", "1994-09-16", "1994-12-16"]),
            "distribution": [0.27, 0.30, 0.31, 0.35],
        }
    )
    audit = reconcile_official_distribution_audit(exact, totals, operational)
    assert audit["exact_event_audit"]["event_count"] == 0
    assert audit["fiscal_period_audit"][0]["event_count"] == 4
    assert audit["uncovered_event_count"] == 0


def test_layered_official_distribution_audit_rejects_fiscal_total_mismatch(
    tmp_path: Path,
) -> None:
    exact_path = tmp_path / "exact.csv"
    exact_path.write_text(
        "ex_date,distribution\n2006-06-16,0.55\n2006-09-15,0.58\n",
        encoding="utf-8",
    )
    totals_path = tmp_path / "totals.csv"
    totals_path.write_text(
        "period_start,period_end,distribution_total\n2005-10-01,2006-09-30,9.99\n",
        encoding="utf-8",
    )
    exact, _ = load_state_street_distributions(
        exact_path, "2005-10-01", "2010-12-31", split="train"
    )
    totals, _ = load_sec_distribution_totals(totals_path, "2005-10-01", "2010-12-31", split="train")
    with pytest.raises(DataGateError, match="SEC_DISTRIBUTION_FISCAL_TOTAL_MISMATCH"):
        reconcile_official_distribution_audit(exact, totals, exact.copy())


def test_sec_1996_rounded_total_uses_only_its_explicit_bounded_tolerance() -> None:
    exact = pd.DataFrame(columns=["date", "distribution"])
    operational = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["1996-03-15", "1996-06-21", "1996-09-20", "1996-12-20"]
            ),
            "distribution": [0.285, 0.351, 0.352, 0.367],
        }
    )
    strict_totals = pd.DataFrame(
        {
            "period_start": [pd.Timestamp("1996-01-01")],
            "period_end": [pd.Timestamp("1996-12-31")],
            "distribution_total": [1.40],
        }
    )
    with pytest.raises(DataGateError, match="SEC_DISTRIBUTION_FISCAL_TOTAL_MISMATCH"):
        reconcile_official_distribution_audit(exact, strict_totals, operational)

    documented_totals = strict_totals.assign(event_sum_tolerance=0.050001)
    audit = reconcile_official_distribution_audit(exact, documented_totals, operational)
    period = audit["fiscal_period_audit"][0]
    assert period["observed_total"] == pytest.approx(1.355)
    assert period["audited_total"] == pytest.approx(1.40)
    assert period["event_sum_tolerance"] == pytest.approx(0.050001)


def test_sec_fiscal_event_sum_tolerance_cannot_exceed_documented_maximum() -> None:
    exact = pd.DataFrame(columns=["date", "distribution"])
    totals = pd.DataFrame(
        {
            "period_start": [pd.Timestamp("1996-01-01")],
            "period_end": [pd.Timestamp("1996-12-31")],
            "distribution_total": [1.40],
            "event_sum_tolerance": [0.050002],
        }
    )
    operational = pd.DataFrame(
        {
            "date": pd.to_datetime(["1996-12-20"]),
            "distribution": [1.355],
        }
    )
    with pytest.raises(
        DataGateError,
        match="SEC_DISTRIBUTION_TOTALS_INVALID_EVENT_SUM_TOLERANCE",
    ):
        reconcile_official_distribution_audit(exact, totals, operational)


def test_sponsor_snapshot_cannot_cross_train_or_locked_boundary(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text(
        "ex_date,distribution\n2010-12-17,0.65\n2021-03-19,1.38\n",
        encoding="utf-8",
    )
    with pytest.raises(
        (DataGateError, LockedBoundaryError),
        match="STATE_STREET_SNAPSHOT_EXCEEDS_PHASE_BOUNDARY|LOCKED_BREACH",
    ):
        load_state_street_distributions(path, "1993-01-22", "2010-12-31", split="train")


def test_spy_reconciliation_requires_99_5_percent_and_explains_outliers() -> None:
    dates = pd.bdate_range("2000-01-03", periods=1001)
    close = 100.0 * np.cumprod(np.full(len(dates), 1.0001))
    yahoo = pd.DataFrame(
        {
            "date": dates,
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 1_000_000,
        }
    )
    stooq = yahoo.copy()
    distributions = pd.DataFrame(columns=["date", "distribution"])
    splits = pd.DataFrame(columns=["date", "split_ratio"])
    report = _reconcile_spy_sources(yahoo, stooq, distributions, splits, minimum_overlap=1000)
    assert report["within_5_bps_fraction"] == 1.0
    broken = stooq.copy()
    broken.loc[500, ["open", "high", "low", "close"]] *= 1.01
    with pytest.raises(DataGateError, match="SPY_RECONCILIATION_99_5_PERCENT_GATE_FAILED"):
        _reconcile_spy_sources(yahoo, broken, distributions, splits, minimum_overlap=1000)


def test_kibot_parser_is_bounded_and_rejects_locked_rows() -> None:
    payload = (
        b"01/03/2006,125.10,127.00,124.39,126.76,1000000\n"
        b"01/04/2006,126.83,127.49,126.70,127.26,1100000\n"
    )
    frame = _parse_kibot_daily_history(
        payload,
        start=pd.Timestamp("2006-01-03"),
        end=pd.Timestamp("2006-01-04"),
    )
    assert frame["date"].dt.strftime("%Y-%m-%d").tolist() == ["2006-01-03", "2006-01-04"]
    with pytest.raises((DataGateError, LockedBoundaryError), match="UNBOUNDED|LOCKED_BREACH"):
        _parse_kibot_daily_history(
            b"01/03/2021,375.31,378.60,375.01,376.13,68766800\n",
            start=pd.Timestamp("2006-01-03"),
            end=pd.Timestamp("2020-12-31"),
        )


def test_three_source_open_adjudication_changes_only_supported_values() -> None:
    dates = pd.bdate_range("2006-01-03", periods=5)
    base = pd.DataFrame(
        {
            "date": dates,
            "open": [100.0, 100.0, 100.0, 100.0, 100.0],
            "high": 102.0,
            "low": 98.0,
            "close": 101.0,
            "volume": 1_000_000,
        }
    )
    yahoo = base.copy()
    stooq = base.copy()
    kibot = base.copy()
    stooq.loc[:, "high"] = 103.0
    stooq.loc[0, "open"] = 99.8  # Kibot supports Yahoo only: use Yahoo.
    stooq.loc[1, "open"] = 99.91
    kibot.loc[1, "open"] = 99.955  # Within 5 bps of both: use the bridge.
    yahoo.loc[2, "open"] = 100.2  # Kibot supports Stooq only: keep Stooq.
    stooq.loc[3, "open"] = 99.8
    kibot.loc[3, "open"] = 100.1  # No pair agrees: retain and report unresolved.
    stooq.loc[4, "open"] = 99.9
    kibot.loc[4, "open"] = 100.1  # No pair agrees, but bounded median is safe.
    stooq.loc[0, "close"] = 100.7  # Kibot supports Yahoo only: use Yahoo.
    stooq.loc[1, "close"] = 100.91
    kibot.loc[1, "close"] = 100.955  # Within 5 bps of both: use the bridge.
    yahoo.loc[2, "close"] = 101.2  # Kibot supports Stooq only: keep Stooq.
    stooq.loc[3, "close"] = 100.7
    kibot.loc[3, "close"] = 101.1  # No pair agrees: retain and report unresolved.
    stooq.loc[4, "close"] = 100.9
    kibot.loc[4, "close"] = 101.1  # No pair agrees, but bounded median is safe.

    canonical, audit = _adjudicate_stooq_open_prices(yahoo, stooq, kibot)

    assert canonical.loc[0, "open"] == pytest.approx(100.0)
    assert canonical.loc[1, "open"] == pytest.approx(99.955)
    assert canonical.loc[2, "open"] == pytest.approx(100.0)
    assert canonical.loc[3, "open"] == pytest.approx(99.8)
    assert canonical.loc[4, "open"] == pytest.approx(100.0)
    assert canonical.loc[0, "close"] == pytest.approx(101.0)
    assert canonical.loc[1, "close"] == pytest.approx(100.955)
    assert canonical.loc[2, "close"] == pytest.approx(101.0)
    assert canonical.loc[3, "close"] == pytest.approx(100.7)
    assert canonical.loc[4, "close"] == pytest.approx(101.0)
    assert canonical["high"].eq(103.0).all()
    assert audit["yahoo_supported_repair_count"] == 1
    assert audit["kibot_bridge_repair_count"] == 1
    assert audit["retained_stooq_count"] == 1
    assert audit["unresolved_level_count"] == 1
    assert audit["changed_close_count"] == 3
    assert audit["fields"]["close"]["retained_stooq_count"] == 1
    assert audit["fields"]["open"]["three_source_median_repair_count"] == 1
    assert audit["fields"]["close"]["three_source_median_repair_count"] == 1
    assert audit["unresolved_close_level_count"] == 1


def test_three_source_open_adjudication_keeps_ohlc_ranges_valid() -> None:
    date = pd.DatetimeIndex(["2006-01-03"])
    yahoo = pd.DataFrame(
        {
            "date": date,
            "open": [100.0],
            "high": [100.2],
            "low": [99.8],
            "close": [100.1],
            "volume": [1_000_000],
        }
    )
    stooq = yahoo.copy()
    stooq.loc[0, ["open", "high", "close"]] = [99.8, 99.9, 99.7]
    kibot = yahoo.copy()

    canonical, audit = _adjudicate_stooq_open_prices(yahoo, stooq, kibot)

    assert canonical.loc[0, "open"] == pytest.approx(100.0)
    assert canonical.loc[0, "close"] == pytest.approx(100.1)
    assert canonical.loc[0, "high"] == pytest.approx(100.1)
    assert canonical.loc[0, "low"] == pytest.approx(99.8)
    assert audit["expanded_high_count"] == 1
    assert audit["expanded_low_count"] == 0


def test_three_source_adjudication_resolves_tick_boundary_with_volume_consensus() -> None:
    date = pd.DatetimeIndex(["2008-09-12"])
    yahoo = pd.DataFrame(
        {
            "date": date,
            "open": [124.29],
            "high": [126.19],
            "low": [123.83],
            "close": [126.09],
            "volume": [297_851_200],
        }
    )
    stooq = pd.DataFrame(
        {
            "date": date,
            "open": [124.29894244803],
            "high": [126.16028287525],
            "low": [123.83986106141],
            "close": [126.02211778261],
            "volume": [297_851_196],
        }
    )
    kibot = pd.DataFrame(
        {
            "date": date,
            "open": [124.29],
            "high": [125.98],
            "low": [123.83],
            "close": [125.75],
            "volume": [288_993_178],
        }
    )

    canonical, audit = _adjudicate_stooq_open_prices(yahoo, stooq, kibot)

    assert canonical.loc[0, "close"] == pytest.approx(126.09)
    close_audit = audit["fields"]["close"]
    assert close_audit["primary_volume_supported_repair_count"] == 1
    assert close_audit["primary_volume_supported_repair_dates"] == ["2008-09-12"]
    assert close_audit["unresolved_level_count"] == 0


def test_three_source_adjudication_does_not_hide_wide_price_disagreement() -> None:
    date = pd.DatetimeIndex(["2008-09-12"])
    yahoo = pd.DataFrame(
        {
            "date": date,
            "open": [100.0],
            "high": [102.0],
            "low": [98.0],
            "close": [101.0],
            "volume": [1_000_000],
        }
    )
    stooq = yahoo.copy()
    stooq.loc[0, "close"] = 100.7
    kibot = yahoo.copy()
    kibot.loc[0, "close"] = 101.4
    kibot.loc[0, "volume"] = 900_000

    canonical, audit = _adjudicate_stooq_open_prices(yahoo, stooq, kibot)

    assert canonical.loc[0, "close"] == pytest.approx(100.7)
    close_audit = audit["fields"]["close"]
    assert close_audit["primary_volume_supported_repair_count"] == 0
    assert close_audit["unresolved_level_count"] == 1


def test_three_source_close_adjudication_preserves_the_frozen_return_gate() -> None:
    dates = pd.bdate_range("2000-01-03", periods=1001)
    close = 100.0 * np.cumprod(np.full(len(dates), 1.0001))
    yahoo = pd.DataFrame(
        {
            "date": dates,
            "open": close,
            "high": close * 1.02,
            "low": close * 0.98,
            "close": close,
            "volume": 1_000_000,
        }
    )
    stooq = yahoo.copy()
    kibot = yahoo.copy()
    stooq.loc[500, "close"] *= 0.998

    canonical, audit = _adjudicate_stooq_open_prices(yahoo, stooq, kibot)
    report = _reconcile_spy_sources(
        yahoo,
        canonical,
        pd.DataFrame(columns=["date", "distribution"]),
        pd.DataFrame(columns=["date", "split_ratio"]),
        minimum_overlap=1000,
    )

    assert audit["changed_close_count"] == 1
    assert canonical.loc[500, "close"] == pytest.approx(yahoo.loc[500, "close"])
    assert report["close_return_unreconciled_outlier_count"] == 0


def test_three_source_open_adjudication_requires_complete_overlap() -> None:
    dates = pd.bdate_range("2006-01-03", periods=3)
    frame = pd.DataFrame(
        {
            "date": dates,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1_000_000,
        }
    )
    with pytest.raises(DataGateError, match="KIBOT_ADJUDICATOR_INCOMPLETE"):
        _adjudicate_stooq_open_prices(frame, frame, frame.iloc[:-1])


def test_three_source_open_adjudication_preserves_the_frozen_return_gate() -> None:
    dates = pd.bdate_range("2000-01-03", periods=1001)
    daily = 0.0001 + 0.002 * np.sin(np.arange(len(dates), dtype=float) / 17.0)
    open_prices = 100.0 * np.cumprod(1.0 + daily)
    yahoo = pd.DataFrame(
        {
            "date": dates,
            "open": open_prices,
            "high": open_prices * 1.02,
            "low": open_prices * 0.98,
            "close": open_prices * 1.001,
            "volume": 1_000_000,
        }
    )
    stooq = yahoo.copy()
    kibot = yahoo.copy()
    stooq.loc[500, "open"] *= 0.998
    stooq.loc[600, "open"] *= 0.9991
    kibot.loc[600, "open"] *= 0.99955

    canonical, audit = _adjudicate_stooq_open_prices(yahoo, stooq, kibot)
    report = _reconcile_spy_sources(
        yahoo,
        canonical,
        pd.DataFrame(columns=["date", "distribution"]),
        pd.DataFrame(columns=["date", "split_ratio"]),
        minimum_overlap=1000,
    )

    assert audit["changed_open_count"] == 2
    assert report["within_5_bps_fraction"] == 1.0
    assert report["unreconciled_outlier_count"] == 0


def test_spy_reconciliation_compares_raw_series_without_using_adjusted_close() -> None:
    dates = pd.bdate_range("2000-01-03", periods=1001)
    adjusted_close = 100.0 * np.cumprod(np.full(len(dates), 1.0001))
    raw_close = adjusted_close.copy()
    raw_close[500:] *= 0.99
    yahoo = pd.DataFrame(
        {
            "date": dates,
            "open": raw_close,
            "high": raw_close * 1.01,
            "low": raw_close * 0.99,
            "close": raw_close,
            "adj_close": adjusted_close,
            "volume": 1_000_000,
        }
    )
    stooq = yahoo.drop(columns="adj_close").copy()
    distributions = pd.DataFrame(
        {"date": [dates[500]], "distribution": [1.0]}
    )
    splits = pd.DataFrame(columns=["date", "split_ratio"])
    report = _reconcile_spy_sources(
        yahoo,
        stooq,
        distributions,
        splits,
        minimum_overlap=1000,
    )
    assert report["within_5_bps_fraction"] == 1.0
    assert report["comparison_basis"] == "open_to_open_total_return"
    assert report["canonical_price_source"] == "stooq_raw_ohlcv"
    assert yahoo.loc[500, "close"] == pytest.approx(raw_close[500])


def test_spy_reconciliation_audits_isolated_yahoo_close_error_without_using_it() -> None:
    dates = pd.bdate_range("2000-01-03", periods=1001)
    close = 100.0 * np.cumprod(np.full(len(dates), 1.0001))
    stooq = pd.DataFrame(
        {
            "date": dates,
            "open": close,
            "high": close * 1.02,
            "low": close * 0.98,
            "close": close,
            "volume": 1_000_000,
        }
    )
    yahoo = stooq.copy()
    yahoo.loc[500, "close"] *= 0.99
    report = _reconcile_spy_sources(
        yahoo,
        stooq,
        pd.DataFrame(columns=["date", "distribution"]),
        pd.DataFrame(columns=["date", "split_ratio"]),
        minimum_overlap=1000,
    )
    assert report["within_5_bps_fraction"] == 1.0
    assert report["close_return_outlier_count"] == 2
    assert report["close_return_unreconciled_outlier_count"] == 0
    assert report["close_only_vendor_discrepancy_dates"] == [
        dates[500].date().isoformat()
    ]


def test_spy_close_return_reconciliation_does_not_depend_on_other_ohlc_fields() -> None:
    dates = pd.bdate_range("2000-01-03", periods=1001)
    close = 100.0 * np.cumprod(np.full(len(dates), 1.0001))
    yahoo = pd.DataFrame(
        {
            "date": dates,
            "open": close,
            "high": close * 1.02,
            "low": close * 0.98,
            "close": close,
            "volume": 1_000_000,
        }
    )
    stooq = yahoo.copy()
    stooq.loc[499, "close"] *= 1.00045
    stooq.loc[500, "close"] *= 0.99955
    stooq.loc[499:500, "high"] *= 1.01
    stooq.loc[499:500, "low"] *= 0.99

    report = _reconcile_spy_sources(
        yahoo,
        stooq,
        pd.DataFrame(columns=["date", "distribution"]),
        pd.DataFrame(columns=["date", "split_ratio"]),
        minimum_overlap=1000,
        close_consensus_dates=[dates[499], dates[500]],
    )

    assert report["close_return_outlier_count"] >= 1
    assert report["close_return_unreconciled_outlier_count"] == 0


def test_spy_reconciliation_reconciles_isolated_yahoo_open_error() -> None:
    dates = pd.bdate_range("2000-01-03", periods=1001)
    close = 100.0 * np.cumprod(np.full(len(dates), 1.0001))
    stooq = pd.DataFrame(
        {
            "date": dates,
            "open": close,
            "high": close * 1.02,
            "low": close * 0.98,
            "close": close * 1.001,
            "volume": 1_000_000,
        }
    )
    yahoo = stooq.copy()
    yahoo.loc[500, "open"] *= 1.01
    report = _reconcile_spy_sources(
        yahoo,
        stooq,
        pd.DataFrame(columns=["date", "distribution"]),
        pd.DataFrame(columns=["date", "split_ratio"]),
        minimum_overlap=1000,
    )
    assert report["raw_within_5_bps_fraction"] < 1.0
    assert report["within_5_bps_fraction"] == 1.0
    assert report["reconciled_outlier_count"] == 2
    assert report["unreconciled_outlier_count"] == 0
    assert report["isolated_yahoo_open_discrepancy_dates"] == [
        dates[500].date().isoformat()
    ]


def test_spy_execution_reconciliation_does_not_mix_high_low_vendor_noise() -> None:
    dates = pd.bdate_range("2000-01-03", periods=1001)
    daily = 0.0001 + 0.002 * np.sin(np.arange(len(dates), dtype=float) / 17.0)
    close = 100.0 * np.cumprod(1.0 + daily)
    stooq = pd.DataFrame(
        {
            "date": dates,
            "open": close,
            "high": close * 1.02,
            "low": close * 0.98,
            "close": close,
            "volume": 1_000_000,
        }
    )
    yahoo = stooq.copy()
    yahoo.loc[500, "open"] *= 1.000049
    yahoo.loc[500, "high"] *= 0.99
    report = _reconcile_spy_sources(
        yahoo,
        stooq,
        pd.DataFrame(columns=["date", "distribution"]),
        pd.DataFrame(columns=["date", "split_ratio"]),
        minimum_overlap=1000,
    )
    assert report["within_5_bps_fraction"] == 1.0
    assert report["field_level_difference_diagnostics"]["high"]["over_5_bps_count"] == 1


class _FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self.content = payload

    def raise_for_status(self) -> None:
        return None


class _FakeSession:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.params = None

    def get(self, url, *, params=None, timeout=None):
        del url, timeout
        self.params = params
        return _FakeResponse(self.payload)


def test_alfred_uses_initial_release_dates_and_never_latest_revised_values() -> None:
    payload = json.dumps(
        {
            "observations": [
                {
                    "date": "2000-01-01",
                    "realtime_start": "2000-02-15",
                    "realtime_end": "2000-02-15",
                    "value": "100.0",
                },
                {
                    "date": "2000-02-01",
                    "realtime_start": "2000-03-15",
                    "realtime_end": "2000-03-15",
                    "value": "101.0",
                },
            ]
        }
    ).encode()
    session = _FakeSession(payload)
    releases, receipt = download_alfred_initial_series(
        "CPIAUCSL",
        "DS033",
        "2000-01-01",
        "2000-12-31",
        split="train",
        session=session,
        api_key="test-key",
    )
    assert session.params is not None
    assert session.params["output_type"] == 4
    assert receipt.status == "downloaded_initial_releases_only"
    sessions = pd.DatetimeIndex(["2000-02-15", "2000-02-16", "2000-03-15", "2000-03-16"])
    aligned = _align_initial_releases(releases, sessions)
    assert pd.isna(aligned.loc["2000-02-15"])
    assert aligned.loc["2000-02-16"] == 100.0
    assert aligned.loc["2000-03-16"] == 101.0


def test_alfred_rejects_release_after_phase_end_without_persisting_it(
    tmp_path: Path,
) -> None:
    payload = json.dumps(
        {
            "observations": [
                {
                    "date": "2010-12-01",
                    "realtime_start": "2011-01-15",
                    "realtime_end": "2011-01-15",
                    "value": "100.0",
                }
            ]
        }
    ).encode()
    with pytest.raises(DataGateError, match="POST_PHASE_RELEASE_IN_RESPONSE"):
        download_alfred_initial_series(
            "CPIAUCSL",
            "DS033",
            "2010-01-01",
            "2010-12-31",
            split="train",
            session=_FakeSession(payload),
            api_key="test-key",
            raw_dir=tmp_path,
        )
    assert list(tmp_path.iterdir()) == []


def test_cftc_publication_lag_uses_release_date_not_observation_date() -> None:
    sessions = pd.bdate_range("2008-01-08", "2008-01-14")
    releases = pd.DataFrame(
        {
            "observation_date": [pd.Timestamp("2008-01-08")],
            "release_date": [pd.Timestamp("2008-01-11")],
            "value": [42.0],
        }
    )
    aligned = _align_initial_releases(releases, sessions)
    assert aligned.loc["2008-01-08":"2008-01-11"].isna().all()
    assert aligned.loc[pd.Timestamp("2008-01-14")] == 42.0


def test_cboe_first_dissemination_blocks_retrospective_backfill() -> None:
    sessions = pd.DatetimeIndex(
        [
            pd.Timestamp("2003-09-19"),
            pd.Timestamp("2003-09-22"),
            pd.Timestamp("2003-09-23"),
        ]
    )
    releases = pd.DataFrame(
        {
            "release_date": [pd.Timestamp("2003-01-02")],
            "value": [20.0],
        }
    )
    aligned = _align_initial_releases(
        releases,
        sessions,
        first_dissemination=pd.Timestamp("2003-09-22"),
    )
    assert pd.isna(aligned.loc[pd.Timestamp("2003-09-19")])
    assert aligned.loc[pd.Timestamp("2003-09-22")] == 20.0


def test_vxo_first_dissemination_blocks_pre_launch_backfill() -> None:
    sessions = pd.DatetimeIndex(
        [
            pd.Timestamp("1993-01-18"),
            pd.Timestamp("1993-01-19"),
            pd.Timestamp("1993-01-20"),
        ]
    )
    releases = pd.DataFrame(
        {
            "release_date": [pd.Timestamp("1986-01-02")],
            "value": [20.0],
        }
    )
    aligned = _align_initial_releases(
        releases,
        sessions,
        first_dissemination=pd.Timestamp("1993-01-19"),
    )
    assert pd.isna(aligned.loc[pd.Timestamp("1993-01-18")])
    assert aligned.loc[pd.Timestamp("1993-01-19")] == 20.0


def test_cboe_vxo_train_capture_is_phase_bounded_and_causal() -> None:
    path = (
        _repo_campaign_root()
        / "official_inputs"
        / "cboe_vxo_daily_1993_2010.csv"
    )
    series, receipt = load_cboe_vxo_history(
        path,
        "1993-01-22",
        "2010-12-31",
        split="train",
    )
    assert series.index.min() == pd.Timestamp("1993-01-29")
    assert series.index.max() == pd.Timestamp("2010-12-31")
    assert len(series) == 4515
    assert receipt.dataset_id == "DS005"
    assert receipt.status == "loaded_phase_bounded_official_cboe_history"


def test_cboe_vxo_validation_capture_contains_no_locked_rows() -> None:
    path = (
        _repo_campaign_root()
        / "official_inputs"
        / "cboe_vxo_daily_2011_2020.csv"
    )
    series, _ = load_cboe_vxo_history(
        path,
        "2011-01-01",
        "2020-12-31",
        split="validation",
    )
    assert series.index.min() == pd.Timestamp("2011-01-03")
    assert series.index.max() == pd.Timestamp("2020-12-31")
    assert len(series) == 2514
    assert (series.index < pd.Timestamp("2021-01-01")).all()


def test_cboe_vxo_loader_rejects_rows_beyond_requested_phase(tmp_path: Path) -> None:
    path = tmp_path / "vxo.csv"
    path.write_text(
        "DATE,OPEN,HIGH,LOW,CLOSE\n"
        "12/31/2010,20,20,20,20\n"
        "01/03/2011,21,21,21,21\n",
        encoding="utf-8",
    )
    with pytest.raises(DataGateError, match="UNBOUNDED_SOURCE_RESPONSE"):
        load_cboe_vxo_history(
            path,
            "1993-01-22",
            "2010-12-31",
            split="train",
        )


def test_cboe_series_after_train_is_rejected_instead_of_backfilled() -> None:
    candidate = _campaign().candidate_by_id()["STRAT0019"]
    base = _long_fixture()
    data = PreparedMarketData(
        ledger=base.ledger,
        series={"VIX": pd.Series(20.0, index=base.ledger.index)},
        available_dataset_ids=frozenset({"DS001", "DS002", "DS004"}),
        rejected_datasets={
            "DS009": "FIRST_DISSEMINATION_AFTER_TRAIN_END:VIX3M_2013_09_30"
        },
        receipts=(),
        split="train",
    )
    with pytest.raises(
        CandidateRejected,
        match="FIRST_DISSEMINATION_AFTER_TRAIN_END:VIX3M_2013_09_30",
    ):
        candidate_decisions(candidate, data)


def test_breadth_current_membership_proxy_is_rejected_fail_closed() -> None:
    candidate = next(
        row for row in _campaign().candidates if row["family"] == "breadth_trend_proxy"
    )
    base = _long_fixture()
    data = PreparedMarketData(
        ledger=base.ledger,
        series={},
        available_dataset_ids=base.available_dataset_ids,
        rejected_datasets={"DS071": "PROXY_ONLY_NOT_EXECUTION_GRADE"},
        receipts=(),
        split="train",
    )
    with pytest.raises(CandidateRejected, match="PROXY_ONLY_NOT_EXECUTION_GRADE"):
        candidate_decisions(candidate, data)


def test_markov_family_cannot_fall_back_to_smoothed_probabilities() -> None:
    candidate = next(
        row for row in _campaign().candidates if row["family"] == "markov_regime_filtered"
    )
    with pytest.raises(
        CandidateRejected,
        match="MARKOV_MODEL_RESTART_AND_CONVERGENCE_GRID",
    ):
        candidate_decisions(candidate, _long_fixture())
    signals_source = (
        REPO_ROOT / "infra" / "sp500_long_short_daily" / "signals.py"
    ).read_text(encoding="utf-8")
    assert "smoothed_prob" not in signals_source


def test_close_decision_executes_at_next_open_and_ties_persist() -> None:
    ledger, _ = build_total_return_ledger(_prices())
    decisions = pd.Series([1.0, -1.0, np.nan, 1.0], index=ledger.index, dtype=float)
    result = apply_positions(ledger, decisions)
    assert result["position"].tolist() == [1, 1, -1, -1]


def test_locked_firewall_fails_without_value_disclosure() -> None:
    prices = _prices()
    prices.loc[len(prices)] = [
        pd.Timestamp("2021-01-04"),
        52.0,
        53.0,
        51.0,
        52.5,
        24,
    ]
    with pytest.raises(LockedBoundaryError, match="TECHNICAL_FAILURE_LOCKED_BREACH"):
        build_total_return_ledger(prices)


def _long_fixture() -> PreparedMarketData:
    dates = pd.bdate_range("1993-01-22", "2010-12-31")
    phase = np.linspace(0.0, 20.0 * np.pi, len(dates))
    close = 100.0 * np.exp(np.cumsum(0.0002 + 0.002 * np.sin(phase)))
    prices = pd.DataFrame(
        {
            "date": dates,
            "open": close * 0.999,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.linspace(1_000_000, 2_000_000, len(dates)),
        }
    )
    ledger, _ = build_total_return_ledger(prices)
    return PreparedMarketData(
        ledger=ledger,
        series={},
        available_dataset_ids=frozenset({"DS001", "DS002"}),
        rejected_datasets={},
        receipts=(),
        split="train",
    )


def test_repository_official_distribution_inputs_are_bounded_and_complete() -> None:
    official = CAMPAIGN_ROOT / "official_inputs"
    events, _ = load_state_street_distributions(
        official / "state_street_spy_distribution_events_2006_2010.csv",
        "1993-01-22",
        "2010-12-31",
        split="train",
    )
    totals, _ = load_sec_distribution_totals(
        official / "sec_spy_distribution_fiscal_totals_1993_2009.csv",
        "1993-01-22",
        "2010-12-31",
        split="train",
    )
    audit = json.loads((official / "official_source_audit.json").read_text("utf-8"))
    assert len(events) == 19
    assert len(totals) == 17
    assert events["date"].max() == pd.Timestamp("2010-12-17")
    assert totals["period_end"].max() == pd.Timestamp("2009-09-30")
    assert audit["locked_opened"] is False
    assert audit["validation_boundary_incident"]["response_used"] is False


def test_price_candidate_and_benchmarks_are_exact_long_short_states() -> None:
    data = _long_fixture()
    package = _campaign()
    candidate = package.candidate_by_id()["STRAT0004"]
    signal = candidate_decisions(candidate, data)
    assert signal.first_evaluable_date is not None
    assert set(signal.decisions.unique()) <= {-1, 1}
    always_long = benchmark_decisions("always_long", data).decisions
    buy_hold = benchmark_decisions("buy_and_hold_spy_total_return", data).decisions
    always_short = benchmark_decisions("always_short", data).decisions
    assert always_long.equals(buy_hold)
    assert np.array_equal(always_short.to_numpy(), -always_long.to_numpy())


def test_warmup_is_not_misreported_as_missing_causal_coverage() -> None:
    data = _long_fixture()
    candidate = _campaign().candidate_by_id()["STRAT0004"]
    signal = candidate_decisions(candidate, data)
    assert signal.missing_fraction == pytest.approx(0.0)


def test_missing_candidate_dataset_is_explicit_rejection() -> None:
    data = _long_fixture()
    candidate = _campaign().candidate_by_id()["STRAT0019"]
    with pytest.raises(CandidateRejected, match="DATA_GATE_REJECTED"):
        candidate_decisions(candidate, data)


def test_monetary_candidate_uses_calendar_yoy_and_exact_long_short_states() -> None:
    base = _long_fixture()
    index = base.ledger.index
    data = PreparedMarketData(
        ledger=base.ledger,
        series={
            "DFF": pd.Series(4.0, index=index),
            "CPI": pd.Series(
                100.0 * np.exp(np.arange(len(index)) * 0.02 / 252.0),
                index=index,
            ),
        },
        available_dataset_ids=frozenset({"DS001", "DS002", "DS021", "DS033"}),
        rejected_datasets={},
        receipts=(),
        split="train",
    )
    candidate = _campaign().candidate_by_id()["STRAT0121"]
    signal = candidate_decisions(candidate, data)
    assert signal.first_evaluable_date is not None
    assert set(signal.decisions.unique()) <= {-1, 1}
    assert (signal.decisions.loc[signal.first_evaluable_date :] == -1).all()


def test_nfci_change_uses_four_calendar_weeks_not_four_daily_sessions() -> None:
    base = _long_fixture()
    index = base.ledger.index
    nfci = pd.Series(np.arange(len(index), dtype=float), index=index)
    data = PreparedMarketData(
        ledger=base.ledger,
        series={"NFCI": nfci},
        available_dataset_ids=frozenset({"DS001", "DS002", "DS025"}),
        rejected_datasets={},
        receipts=(),
        split="train",
    )
    candidate = _campaign().candidate_by_id()["STRAT0092"]
    signal = candidate_decisions(candidate, data)
    first = pd.Timestamp(signal.first_evaluable_date)
    assert first >= index[20]
    assert signal.decisions.loc[first] == -1


def test_financial_conditions_vote_uses_component_signs() -> None:
    base = _long_fixture()
    index = base.ledger.index
    data = PreparedMarketData(
        ledger=base.ledger,
        series={
            "NFCI": pd.Series(100.0, index=index),
            "ANFCI": pd.Series(-1.0, index=index),
            "OFR_FSI": pd.Series(-1.0, index=index),
        },
        available_dataset_ids=frozenset({"DS001", "DS002"}),
        rejected_datasets={},
        receipts=(),
        split="train",
    )
    candidate = _campaign().candidate_by_id()["STRAT0096"]
    signal = candidate_decisions(candidate, data)
    assert (signal.decisions == 1).all()


def test_under_specified_monetary_vote_is_rejected_precisely() -> None:
    base = _long_fixture()
    candidate = _campaign().candidate_by_id()["STRAT0126"]
    data = PreparedMarketData(
        ledger=base.ledger,
        series={
            name: pd.Series(1.0, index=base.ledger.index)
            for name in ("DFF", "CPI", "T10YIE", "WALCL", "M2")
        },
        available_dataset_ids=frozenset(candidate["required_datasets"]),
        rejected_datasets={},
        receipts=(),
        split="train",
    )
    with pytest.raises(
        CandidateRejected,
        match="INCOMPLETE_FROZEN_RULE_SPEC:INFLATION_ACCELERATION_HORIZON",
    ):
        candidate_decisions(candidate, data)


def test_preholiday_rule_does_not_treat_an_ordinary_weekend_as_a_holiday() -> None:
    data = _long_fixture()
    candidate = _campaign().candidate_by_id()["STRAT0143"]
    signal = candidate_decisions(candidate, data)
    index = data.ledger.index
    close = data.ledger["tr_close"]
    trend = close - close.rolling(200, min_periods=200).mean()
    next_dates = pd.Series(index, index=index).shift(-1)
    following_dates = next_dates.shift(-1)
    ordinary_friday = (
        (next_dates.dt.dayofweek == 4) & (following_dates.dt.dayofweek == 0) & (trend < 0)
    )
    assert ordinary_friday.any()
    assert (signal.decisions.loc[ordinary_friday] == -1).all()


def test_ensemble_uses_exact_frozen_component_ids_without_aliases() -> None:
    package = _campaign()
    candidate = package.candidate_by_id()["STRAT0027"]
    assert _frozen_component_ids(candidate) == (
        "STRAT0004",
        "STRAT0046",
        "STRAT0079",
        "STRAT0086",
    )
    assert candidate["parameters"]["components"] == [
        "SMA200",
        "BREAKOUT126",
        "CURVE",
        "CREDIT",
    ]


def test_ensemble_requires_registry_and_never_substitutes_aliases() -> None:
    base = _long_fixture()
    data = PreparedMarketData(
        ledger=base.ledger,
        series=base.series,
        available_dataset_ids=frozenset({"DS001", "DS002", "DS004", "DS009"}),
        rejected_datasets={},
        receipts=(),
        split="train",
    )
    candidate = _campaign().candidate_by_id()["STRAT0025"]
    with pytest.raises(
        CandidateRejected,
        match="ENSEMBLE_REQUIRES_FROZEN_CANDIDATE_REGISTRY",
    ):
        candidate_decisions(candidate, data)


def test_train_workload_evaluates_or_rejects_without_silent_drop() -> None:
    data = _long_fixture()
    package = _campaign()
    evaluated, records = TRAIN_WORKLOAD._evaluate(
        data,
        "STRAT0004",
        package.candidate_by_id()["STRAT0004"],
        "test-attempt",
    )
    rejected, rejected_records = TRAIN_WORKLOAD._evaluate(
        data,
        "STRAT0019",
        package.candidate_by_id()["STRAT0019"],
        "test-attempt",
    )
    assert evaluated["status"] == "evaluated"
    assert len(records) == 1
    assert evaluated["train_period_count"] >= 1000
    assert rejected["status"] == "rejected"
    assert rejected["rejection_reason"].startswith("DATA_GATE_REJECTED")
    assert rejected_records == ()


def test_campaign_partial_merge_waits_for_exact_coverage_and_rejects_conflicts(
    tmp_path: Path,
) -> None:
    data = _long_fixture()
    candidate = _campaign().candidate_by_id()["STRAT0004"]
    row, _ = TRAIN_WORKLOAD._evaluate(data, "STRAT0004", candidate, "merge-attempt-a")
    first = tmp_path / "first"
    first.mkdir()
    pq.write_table(
        pa.Table.from_pylist([row], schema=TRAIN_WORKLOAD.result_schema),
        first / TRAIN_WORKLOAD.result_filename,
    )
    partial = tmp_path / "partial"
    TRAIN_WORKLOAD.merge_group((first,), partial)
    assert (partial / TRAIN_WORKLOAD.result_filename).is_file()
    assert not (partial / "train_selection_freeze.json").exists()

    conflicting = dict(row)
    conflicting["source_attempt_id"] = "merge-attempt-b"
    conflicting["unit_output_sha256"] = "0" * 64
    second = tmp_path / "second"
    second.mkdir()
    pq.write_table(
        pa.Table.from_pylist([conflicting], schema=TRAIN_WORKLOAD.result_schema),
        second / TRAIN_WORKLOAD.result_filename,
    )
    with pytest.raises(ValueError, match="conflicting result for STRAT0004"):
        TRAIN_WORKLOAD.merge_group((first, second), tmp_path / "conflict")


def test_fixture_snapshot_round_trip_preserves_boundaries(tmp_path: Path) -> None:
    data = _long_fixture()
    write_fixture_snapshot(tmp_path, data.ledger)
    loaded = load_market_snapshot(tmp_path)
    assert loaded.split == "train"
    assert loaded.ledger.index.max() == pd.Timestamp("2010-12-31")
    assert loaded.available_dataset_ids == frozenset({"DS001", "DS002"})


def test_github_train_spec_passes_universal_preflight() -> None:
    report = validate_run_spec(REPO_ROOT / "config" / "sp500_long_short_daily_train_v3.yaml")
    assert report.valid, [item.model_dump() for item in report.violations]


def test_phase_workloads_are_bounded_and_have_exact_unit_counts() -> None:
    assert SMOKE_WORKLOAD.data_start == "2005-10-01"
    assert SMOKE_WORKLOAD.data_end == "2009-09-30"
    assert PILOT_WORKLOAD.data_end == "2010-12-31"
    assert TRAIN_WORKLOAD.data_end == "2010-12-31"
    assert len(SMOKE_WORKLOAD._unit_definitions()) == 7
    assert len(PILOT_WORKLOAD._unit_definitions()) == 23
    assert len(TRAIN_WORKLOAD._unit_definitions()) == 173
    assert {
        payload["family"]
        for _, payload, _ in PILOT_WORKLOAD._unit_definitions()
        if payload.get("unit_type") != "benchmark"
    } == set(PILOT_WORKLOAD.representative_families)


def test_every_frozen_family_is_implemented_or_has_a_precise_rejection() -> None:
    package_families = {row["family"] for row in _campaign().candidates}
    assert IMPLEMENTED_FAMILIES.isdisjoint(EXPLICIT_FAMILY_REJECTIONS)
    assert package_families <= IMPLEMENTED_FAMILIES | set(EXPLICIT_FAMILY_REJECTIONS)


def test_effective_trial_count_collapses_identical_strategies() -> None:
    values = np.linspace(-0.01, 0.01, 200)
    identical = pd.DataFrame({"a": values, "b": values, "c": values})
    independent = pd.DataFrame(
        np.random.default_rng(20260803).normal(size=(2000, 3)),
        columns=["a", "b", "c"],
    )
    assert effective_independent_trials(identical) == pytest.approx(1.0)
    assert effective_independent_trials(independent) > 2.8


def test_stationary_bootstrap_and_fdr_are_deterministic_and_bounded() -> None:
    left = _stationary_bootstrap_indices(np.random.default_rng(7), 200, 10)
    right = _stationary_bootstrap_indices(np.random.default_rng(7), 200, 10)
    assert np.array_equal(left, right)
    assert int(left.min()) >= 0
    assert int(left.max()) < 200
    qvalues = _benjamini_hochberg({"a": 0.01, "b": 0.04, "c": 0.03})
    assert qvalues == pytest.approx({"a": 0.03, "c": 0.04, "b": 0.04})


def test_reality_check_uses_stationary_bootstrap_and_reports_ci_and_fdr() -> None:
    rng = np.random.default_rng(11)
    frame = pd.DataFrame(
        {
            "a": rng.normal(0.001, 0.01, 220),
            "b": rng.normal(0.0002, 0.01, 220),
        }
    )
    result = reality_check_and_spa(frame, seed=13, samples=80, block_lengths=(5,))
    assert result["bootstrap_method"] == "politis_romano_stationary_circular"
    assert set(result["candidate_fdr_qvalues"]) == {"a", "b"}
    assert set(result["candidate_mean_differential_ci_95"]) == {"a", "b"}
    assert all(len(bounds) == 2 for bounds in result["candidate_mean_differential_ci_95"].values())


def test_train_freeze_hash_and_code_are_verified(tmp_path: Path) -> None:
    payload = {
        "schema_version": "1",
        "selection_closed": True,
        "validation_opened": False,
        "locked_opened": False,
        "train_end": "2010-12-31",
        "locked_start": "2021-01-01",
        "code_sha": "abc123",
        "finalists": [],
    }
    payload["freeze_sha256"] = canonical_json_hash(payload)
    path = tmp_path / "train_selection_freeze.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert verify_train_freeze(path, code_sha="abc123")["selection_closed"] is True
    with pytest.raises(ValidationGateError, match="CODE_SHA_MISMATCH"):
        verify_train_freeze(path, code_sha="different")
    payload["selection_closed"] = False
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValidationGateError, match="HASH_MISMATCH"):
        verify_train_freeze(path, code_sha="abc123")


def test_validation_ack_is_required_before_any_artifact_read(tmp_path: Path) -> None:
    with pytest.raises(ValidationGateError, match="VALIDATION_ACK_MISMATCH"):
        run_validation_once(
            train_results_dir=tmp_path / "missing-results",
            train_prepared_dir=tmp_path / "missing-train",
            validation_prepared_dir=tmp_path / "missing-validation",
            output_dir=tmp_path / "output",
            validation_ack=VALIDATION_ACK + "_WRONG",
        )


def test_validation_refuses_diagnostic_representatives_before_loading_data(tmp_path: Path) -> None:
    freeze = {
        "schema_version": "1",
        "selection_closed": True,
        "validation_opened": False,
        "locked_opened": False,
        "train_end": "2010-12-31",
        "locked_start": "2021-01-01",
        "code_sha": "LOCAL_TEST_ONLY",
        "finalists": [
            {
                "strategy_id": "STRAT0014",
                "canonical_hash": "a" * 64,
                "diagnostic_only": True,
                "eligible_for_validation": False,
            }
        ],
    }
    freeze["freeze_sha256"] = canonical_json_hash(freeze)
    train_results = tmp_path / "train-results"
    train_results.mkdir()
    (train_results / "train_selection_freeze.json").write_text(
        json.dumps(freeze), encoding="utf-8"
    )

    with pytest.raises(
        ValidationGateError,
        match="INELIGIBLE_FROZEN_FINALISTS_VALIDATION_MUST_NOT_OPEN:STRAT0014",
    ):
        run_validation_once(
            train_results_dir=train_results,
            train_prepared_dir=tmp_path / "missing-train",
            validation_prepared_dir=tmp_path / "missing-validation",
            output_dir=tmp_path / "output",
            validation_ack=VALIDATION_ACK,
            code_sha="LOCAL_TEST_ONLY",
        )


def test_authorized_diagnostic_freeze_is_traceable_and_validation_stays_locked(
    tmp_path: Path,
) -> None:
    candidate = _campaign().candidate_by_id()["STRAT0014"]
    source = tmp_path / "source-train-results"
    source.mkdir()
    source_freeze = {
        "schema_version": "1",
        "selection_closed": True,
        "validation_opened": False,
        "locked_opened": False,
        "train_end": "2010-12-31",
        "validation_start": "2011-01-01",
        "validation_end": "2020-12-31",
        "locked_start": "2021-01-01",
        "code_sha": "TRAIN_SOURCE_SHA",
        "finalists": [],
    }
    source_freeze["freeze_sha256"] = canonical_json_hash(source_freeze)
    (source / "train_selection_freeze.json").write_text(
        json.dumps(source_freeze), encoding="utf-8"
    )
    pd.DataFrame(
        [
            {
                "strategy_id": "STRAT0014",
                "canonical_hash": candidate["canonical_hash"],
                "status": "evaluated",
                "hard_train_eligible": True,
                "train_cagr_pct": 15.484,
                "train_sharpe": 0.7704,
                "train_calmar": 0.5864,
                "train_max_drawdown_pct": -26.4051,
                "spa_pvalue": 0.9240759,
            }
        ]
    ).to_csv(source / "candidate_metrics.csv", index=False)

    diagnostic = tmp_path / "diagnostic-freeze"
    path = build_diagnostic_train_freeze(
        source_train_results_dir=source,
        output_dir=diagnostic,
        strategy_id="STRAT0014",
        code_sha="LOCAL_TEST_ONLY",
    )
    freeze = verify_train_freeze(path, code_sha="LOCAL_TEST_ONLY")
    assert freeze["source_train_freeze_sha256"] == source_freeze["freeze_sha256"]
    assert freeze["locked_start"] == "2021-01-01"
    assert freeze["locked_opened"] is False
    assert freeze["finalists"] == [
        {
            "strategy_id": "STRAT0014",
            "canonical_hash": candidate["canonical_hash"],
            "diagnostic_only": True,
            "eligible_for_validation": False,
            "train_metrics": {
                "cagr_pct": 15.484,
                "sharpe": 0.7704,
                "calmar": 0.5864,
                "max_drawdown_pct": -26.4051,
                "spa_pvalue": 0.9240759,
            },
        }
    ]


def test_phase_snapshots_combine_without_opening_locked() -> None:
    train = _long_fixture()
    validation_dates = pd.bdate_range("2011-01-03", "2020-12-31")
    close = 100.0 * np.exp(np.arange(len(validation_dates)) * 0.0001)
    prices = pd.DataFrame(
        {
            "date": validation_dates,
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 1_000_000,
        }
    )
    validation_ledger, _ = build_total_return_ledger(prices)
    validation = PreparedMarketData(
        ledger=validation_ledger,
        series={},
        available_dataset_ids=frozenset({"DS001", "DS002"}),
        rejected_datasets={},
        receipts=(),
        split="validation",
    )
    combined = combine_phase_snapshots(train, validation)
    assert combined.ledger.index.min() == train.ledger.index.min()
    assert combined.ledger.index.max() == pd.Timestamp("2020-12-31")
    assert combined.ledger.index.max() < pd.Timestamp("2021-01-01")


def test_one_shot_validation_outputs_only_frozen_candidate_and_2011_2020(
    tmp_path: Path,
) -> None:
    package = _campaign()
    candidate = package.candidate_by_id()["STRAT0004"]
    train_data = _long_fixture()
    train_root = tmp_path / "train-prepared"
    write_fixture_snapshot(train_root, train_data.ledger)

    dates = pd.bdate_range("2011-01-03", "2020-12-31")
    close = 100.0 * np.exp(np.arange(len(dates)) * 0.0002)
    prices = pd.DataFrame(
        {
            "date": dates,
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 1_000_000,
        }
    )
    validation_ledger, _ = build_total_return_ledger(prices)
    validation_root = tmp_path / "validation-prepared"
    write_fixture_snapshot(validation_root, validation_ledger, split="validation")

    freeze = {
        "schema_version": "1",
        "selection_closed": True,
        "validation_opened": False,
        "locked_opened": False,
        "train_end": "2010-12-31",
        "validation_start": "2011-01-01",
        "validation_end": "2020-12-31",
        "locked_start": "2021-01-01",
        "code_sha": "LOCAL_TEST_ONLY",
        "finalists": [
            {
                "strategy_id": candidate["strategy_id"],
                "canonical_hash": candidate["canonical_hash"],
                "diagnostic_only": False,
                "train_metrics": {"sharpe": 0.5, "calmar": 0.5},
            }
        ],
    }
    freeze["freeze_sha256"] = canonical_json_hash(freeze)
    train_results = tmp_path / "train-results"
    train_results.mkdir()
    (train_results / "train_selection_freeze.json").write_text(json.dumps(freeze), encoding="utf-8")
    output = tmp_path / "validation-output"
    summary = run_validation_once(
        train_results_dir=train_results,
        train_prepared_dir=train_root,
        validation_prepared_dir=validation_root,
        output_dir=output,
        validation_ack=VALIDATION_ACK,
        code_sha="LOCAL_TEST_ONLY",
    )
    metrics = pd.read_csv(output / "validation_candidate_and_benchmark_metrics.csv")
    daily = pd.read_parquet(output / "validation_daily_returns.parquet")
    assert len(metrics) == 6
    assert set(metrics.loc[metrics["unit_type"] == "candidate", "strategy_id"]) == {"STRAT0004"}
    assert pd.to_datetime(daily["date"]).min() >= pd.Timestamp("2011-01-01")
    assert pd.to_datetime(daily["date"]).max() <= pd.Timestamp("2020-12-31")
    assert summary["locked_opened"] is False
    assert summary["validation_used_for_selection"] is False
    assert (output / "final_manifest.json").is_file()


def test_diagnostic_validation_evaluates_only_strat0014_without_promotion(
    tmp_path: Path,
) -> None:
    candidate = _campaign().candidate_by_id()["STRAT0014"]
    train_data = _long_fixture()
    train_root = tmp_path / "train-prepared"
    write_fixture_snapshot(train_root, train_data.ledger)

    dates = pd.bdate_range("2011-01-03", "2020-12-31")
    close = 100.0 * np.exp(np.arange(len(dates)) * 0.0002)
    prices = pd.DataFrame(
        {
            "date": dates,
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 1_000_000,
        }
    )
    validation_ledger, _ = build_total_return_ledger(prices)
    validation_root = tmp_path / "validation-prepared"
    write_fixture_snapshot(validation_root, validation_ledger, split="validation")

    freeze = {
        "schema_version": "1",
        "selection_closed": True,
        "validation_opened": False,
        "locked_opened": False,
        "train_end": "2010-12-31",
        "validation_start": "2011-01-01",
        "validation_end": "2020-12-31",
        "locked_start": "2021-01-01",
        "code_sha": "LOCAL_TEST_ONLY",
        "finalists": [
            {
                "strategy_id": "STRAT0014",
                "canonical_hash": candidate["canonical_hash"],
                "diagnostic_only": True,
                "eligible_for_validation": False,
                "train_metrics": {"sharpe": 0.7704, "calmar": 0.5864},
            }
        ],
    }
    freeze["freeze_sha256"] = canonical_json_hash(freeze)
    train_results = tmp_path / "train-results"
    train_results.mkdir()
    (train_results / "train_selection_freeze.json").write_text(
        json.dumps(freeze), encoding="utf-8"
    )

    output = tmp_path / "validation-output"
    summary = run_validation_once(
        train_results_dir=train_results,
        train_prepared_dir=train_root,
        validation_prepared_dir=validation_root,
        output_dir=output,
        validation_ack=DIAGNOSTIC_VALIDATION_ACK,
        code_sha="LOCAL_TEST_ONLY",
        allow_diagnostic=True,
    )
    metrics = pd.read_csv(output / "validation_candidate_and_benchmark_metrics.csv")
    daily = pd.read_parquet(output / "validation_daily_returns.parquet")
    assert set(metrics.loc[metrics["unit_type"] == "candidate", "strategy_id"]) == {
        "STRAT0014"
    }
    assert summary["result_status"] == "DIAGNOSTIC_VALIDATION_RESULT"
    assert summary["diagnostic_only"] is True
    assert summary["promotion_eligible"] is False
    assert summary["locked_opened"] is False
    assert pd.to_datetime(daily["date"]).max() <= pd.Timestamp("2020-12-31")


def test_diagnostic_validation_script_marks_yahoo_distribution_source() -> None:
    path = REPO_ROOT / "scripts" / "run_sp500_long_short_daily_validation.py"
    text = path.read_text(encoding="utf-8")
    assert "allow_diagnostic_yahoo_distributions=allow_diagnostic" in text
    source = (
        REPO_ROOT / "infra" / "sp500_long_short_daily" / "data.py"
    ).read_text(encoding="utf-8")
    assert "diagnostic_bounded_yahoo_distributions_not_official_sponsor" in source
    assert "promotion_eligible=false" in source


def test_workflow_exposes_fail_closed_one_shot_validation() -> None:
    path = REPO_ROOT / ".github" / "workflows" / "sp500-long-short-daily-campaign.yml"
    text = path.read_text(encoding="utf-8")
    parsed = yaml.safe_load(text)
    assert parsed
    assert "validation_once" in text
    assert "validation_diagnostic" in text
    assert "OPEN_VALIDATION_2011_2020_ONCE" in text
    assert "OPEN_VALIDATION_2011_2020_DIAGNOSTIC_STRAT0014" in text
    assert "train_prepared_run_id" in text
    assert '--source-mode "$source_mode"' in text
    assert 'source_mode="yahoo-fallback"' in text
    assert "train_selection_freeze.json" not in text or "train-results" in text
    assert "2021-01-01" not in text
    assert "C:\\" not in text
    assert "self-hosted" not in text
    assert "ubuntu-24.04" in text
    assert "stooq_windows" in text
    assert "merge_sp500_stooq_windows.py" in text
    assert "prepared_artifact_name" in text
    assert "max-parallel: 180" in text

    universal = (
        REPO_ROOT / ".github" / "workflows" / "_aurora-future-run-v3.yml"
    ).read_text(encoding="utf-8")
    assert yaml.safe_load(universal)
    assert "sp500_stooq_windows" in universal
    assert "sp500_prepare_data" in universal
    assert "merge_sp500_stooq_windows.py" in universal
    assert "SP500_STOOQ_HISTORY_CSV" in universal
    assert "max-parallel: 4" in universal
    assert "cursor.year + 3" in universal
    assert "dt.timedelta(days=44)" not in universal
    assert '--source-mode "yahoo-fallback"' in universal
    assert "prepared_artifact_run_id" in universal
    assert "AURORA_PREPARED_ARTIFACT_RUN_ID:" in universal
    universal_lines = [line.strip() for line in universal.splitlines()]
    assert universal_lines.count(
        "run-id: ${{ env.AURORA_PREPARED_ARTIFACT_RUN_ID }}"
    ) == 9
    assert universal_lines.count(
        "prepared-artifact-run-id: ${{ env.AURORA_PREPARED_ARTIFACT_RUN_ID }}"
    ) == 12
    assert "needs.prepare_data.result == 'success'" in universal

    retry_action = (
        REPO_ROOT / ".github" / "actions" / "aurora-retry-shard" / "action.yml"
    ).read_text(encoding="utf-8")
    assert yaml.safe_load(retry_action)
    assert "prepared-artifact-run-id" in retry_action
    assert "inputs.prepared-artifact-run-id || github.run_id" in retry_action
    assert "C:\\" not in universal
    assert "self-hosted" not in universal


def test_smoke_manifest_advertises_only_opened_train_dates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = yaml.safe_load(
        (REPO_ROOT / "config" / "sp500_long_short_daily_train_v3.yaml").read_text(encoding="utf-8")
    )
    payload["policy"]["policy_hash"] = "a" * 64
    spec = RunSpec.model_validate(payload)
    monkeypatch.setattr(
        SMOKE_WORKLOAD,
        "_prepare_dataset",
        lambda root: (("market_data_manifest.json",), "b" * 64),
    )
    prepared = SMOKE_WORKLOAD.prepare_shared_inputs(spec, tmp_path)
    manifest = pd.read_json(prepared.manifest_path, typ="series")
    assert manifest["campaign_phase"] == "smoke"
    assert manifest["max_date"] == "2009-09-30"
    assert manifest["validation_opened"] is False
    assert manifest["locked_opened"] is False


def test_smoke_reduction_cannot_create_train_freeze(tmp_path: Path) -> None:
    rows = []
    for key, payload, _ in SMOKE_WORKLOAD._unit_definitions():
        row = SMOKE_WORKLOAD._base_row(key, payload, "test-smoke")
        row["status"] = "rejected"
        row["rejection_reason"] = "TEST_REJECTION"
        from aurora.infra.github_performance.contracts import canonical_sha256

        row["unit_output_sha256"] = canonical_sha256(
            {name: value for name, value in row.items() if name != "source_attempt_id"}
        )
        rows.append(row)
    SMOKE_WORKLOAD._write_reduction(rows, tmp_path)
    summary = pd.read_json(tmp_path / "sp500_long_short_daily_smoke_summary.json", typ="series")
    assert summary["expected_units"] == 7
    assert summary["maximum_date"] == "2009-09-30"
    assert summary["validation_opened"] is False
    assert not (tmp_path / "train_selection_freeze.json").exists()


def test_full_coverage_reduction_keeps_every_terminal_unit(tmp_path: Path, monkeypatch) -> None:
    data = _long_fixture()
    write_fixture_snapshot(tmp_path, data.ledger)
    monkeypatch.setenv("AURORA_PREPARED_ROOT", str(tmp_path))
    definitions = TRAIN_WORKLOAD._unit_definitions()
    rows = []
    evaluated_keys = {
        "STRAT0001",
        *(key for key, _, _ in definitions if key.startswith("BENCHMARK::")),
    }
    for key, payload, _ in definitions:
        if key in evaluated_keys:
            row, _ = TRAIN_WORKLOAD._evaluate(data, key, payload, "test-reduction")
        else:
            row = TRAIN_WORKLOAD._base_row(key, payload, "test-reduction")
            row["status"] = "rejected"
            row["rejection_reason"] = "TEST_DISCLOSED_REJECTION"
            from aurora.infra.github_performance.contracts import canonical_sha256

            row["unit_output_sha256"] = canonical_sha256(
                {name: value for name, value in row.items() if name != "source_attempt_id"}
            )
        rows.append(row)
    output = tmp_path / "reduction"
    TRAIN_WORKLOAD._write_reduction(rows, output)
    summary = pd.read_json(output / "sp500_long_short_daily_train_summary.json", typ="series")
    eligibility = pd.read_csv(output / "eligibility_and_rejections.csv")
    assert len(eligibility) == 173
    assert int(summary["expected_candidates"]) == 168
    assert int(summary["expected_benchmarks"]) == 5
    assert int(summary["frozen_finalists"]) == 0
    assert int(summary["multiple_testing_eligible_candidates"]) == 0
    assert summary["result_status"] == "NEGATIVE_RESULT"
    assert (output / "train_selection_freeze.json").is_file()
    assert (output / "multiple_testing.json").is_file()
    required = {
        "RESULT_STATUS.md",
        "final_manifest.json",
        "candidate_and_benchmark_metrics.csv",
        "annual_returns.csv",
        "rolling_metrics.csv",
        "regime_metrics.csv",
        "fold_metrics.csv",
        "multiple_testing_results.json",
        "data_lineage.jsonl",
        "raw_manifest.jsonl",
        "scheduler_plan.json",
        "environment_lock.txt",
        "implementation_mapping.md",
        "official_source_audit.json",
        "candidate_strategy_pack.jsonl",
        "feature_catalog.csv",
        "family_coverage.csv",
        "proxy_limitations.csv",
        "dataset_limitations.csv",
        "scientific_warnings.md",
    }
    assert required <= {path.name for path in output.iterdir()}
    final_manifest = json.loads((output / "final_manifest.json").read_text("utf-8"))
    assert final_manifest["validation_opened"] is False
    assert final_manifest["locked_opened"] is False
    assert final_manifest["validation_authorization_used"] is False
    assert final_manifest["required_file_count"] == len(final_manifest["files"])
    assert len(final_manifest["authoritative_input_zip_sha256"]) == 64
    assert set(final_manifest["files"]) >= required - {"final_manifest.json"}
    metrics = pd.read_csv(output / "candidate_and_benchmark_metrics.csv")
    assert len(metrics) == 173
    assert metrics["strategy_id"].nunique() == 173
    folds = pd.read_csv(output / "fold_metrics.csv")
    assert {
        "outer_fold_id",
        "outer_train_end",
        "outer_test_start",
        "outer_test_end",
        "embargo_sessions",
        "fit_mode",
        "out_of_fold",
    } <= set(folds.columns)
    assert set(folds["fit_mode"]) == {"static_rule_no_fit"}
    assert pd.to_datetime(folds["outer_train_end"]).lt(
        pd.to_datetime(folds["outer_test_start"])
    ).all()
    assert not folds["validation_used"].astype(bool).any()
    assert pd.to_datetime(folds["outer_test_end"]).max() <= pd.Timestamp("2010-12-31")
    regimes = pd.read_csv(output / "regime_metrics.csv")
    assert set(regimes["status"]) == {"calculated"}
    assert set(regimes["regime"]) <= {
        "spy_above_sma200",
        "spy_at_or_below_sma200",
    }
    freeze = json.loads((output / "train_selection_freeze.json").read_text("utf-8"))
    assert freeze["finalists"] == []
    assert freeze["validation_authorization_required"] == "OPEN_VALIDATION_2011_2020_ONCE"
    assert freeze["validation_authorization_used"] is False
    assert freeze["freeze_created_at_utc"].endswith("+00:00")
    assert len(freeze["candidate_identities"]) == 168
    assert len({row["strategy_id"] for row in freeze["candidate_identities"]}) == 168
    assert len(freeze["definitive_train_ranking"]) == 1
    assert isinstance(freeze["pareto_frontier"], list)
    assert freeze["selection_criteria"]["multiple_testing_gate"] == {
        "spa_pvalue": "<= 0.10"
    }
    assert freeze["code_files_sha256"]
    assert all(len(value) == 64 for value in freeze["code_files_sha256"].values())
    assert freeze["position_contract"]["allowed_values"] == [-1, 1]
    assert freeze["costs"] == {
        "commission_bps": 0,
        "spread_bps": 0,
        "slippage_bps": 0,
        "market_impact_bps": 0,
        "borrow_bps": 0,
        "financing_bps": 0,
    }
    strategy_pack = [
        json.loads(line)
        for line in (output / "candidate_strategy_pack.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert len(strategy_pack) == 168
    assert {row["strategy_id"] for row in strategy_pack} == {
        row["strategy_id"] for row in freeze["candidate_identities"]
    }
    features = pd.read_csv(output / "feature_catalog.csv")
    assert len(features) == 168
    coverage = pd.read_csv(output / "family_coverage.csv")
    assert len(coverage) == 28
    assert set(coverage["expected_candidates"]) == {6}
    assert set(coverage["terminal_candidates"]) == {6}
    assert coverage["exact_coverage"].all()
    proxies = pd.read_csv(output / "proxy_limitations.csv")
    assert set(proxies["classification"]) == {"proxy_only"}
    assert set(proxies["dataset_id"]) == {"DS046", "DS070", "DS071", "DS072"}
    warnings = (output / "scientific_warnings.md").read_text(encoding="utf-8")
    assert "Validation was not opened" in warnings
    assert "contradictions_and_negative_results.md" in warnings
