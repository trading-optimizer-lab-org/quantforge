"""Bounded, fail-closed data acquisition for the SPY daily campaign."""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, replace
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
import websockets

from aurora.core.execution_policy import require_github_only_execution
from aurora.infra.sp500_long_short_daily.contracts import (
    LOCKED_START,
    TRAIN_END,
    VALIDATION_END,
    CampaignPackage,
    LockedBoundaryError,
    assert_frame_before_locked,
    canonical_json_hash,
)
from aurora.infra.sp500_long_short_daily.ledger import build_total_return_ledger


YAHOO_CHART_ENDPOINTS = (
    "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
    "https://query2.finance.yahoo.com/v8/finance/chart/{symbol}",
)
KIBOT_API_ENDPOINT = "https://api.kibot.com/"
STOOQ_HISTORY_PAGE = "https://stooq.com/q/d/"
STOOQ_VERIFY = "https://stooq.com/__verify"
FRED_DOWNLOAD = "https://fred.stlouisfed.org/graph/fredgraph.csv"
FRED_API_OBSERVATIONS = "https://api.stlouisfed.org/fred/series/observations"
CBOE_VXO_HISTORY = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VXO_History.csv"
SPY_RETURN_TOLERANCE = 5e-4
SPY_REQUIRED_TOLERANCE_FRACTION = 0.995
SPY_MEDIAN_ADJUDICATION_MAX_SPREAD = 2.5e-3
SPY_PRICE_TICK_USD = 0.01
SPY_PRIMARY_VOLUME_RELATIVE_TOLERANCE = 1e-5
SPY_ADJUDICATOR_VOLUME_OUTLIER_TOLERANCE = 1e-3
SPONSOR_EVENT_AMOUNT_TOLERANCE = 5.001e-4
SEC_FISCAL_EVENT_SUM_TOLERANCE = 0.005001
SEC_FISCAL_EVENT_SUM_TOLERANCE_MAX = 0.050001
STOOQ_MAX_BOUNDED_PAGES = 100
STOOQ_PUBLIC_HISTORY_ROW_CAP = 1000
STOOQ_PAGE_DELAY_SECONDS = 1.25
STOOQ_WINDOW_COOLDOWN_SECONDS = 60.0
STOOQ_MAX_WINDOW_YEARS = 3
STOOQ_RAW_OPERATION_PARAMS = {
    "c": "0",
    "o": "1111111",
    "o_s": "1",
    "o_d": "1",
    "o_p": "1",
    "o_n": "1",
    "o_o": "1",
    "o_m": "1",
    "o_x": "1",
}


class DataGateError(RuntimeError):
    """Raised when a required source cannot satisfy the frozen contract."""


@dataclass(frozen=True)
class DownloadReceipt:
    dataset_id: str
    url_template: str
    sha256: str
    byte_count: int
    minimum_date: str | None
    maximum_date: str | None
    status: str
    reason: str | None = None


@dataclass(frozen=True)
class PreparedMarketData:
    ledger: pd.DataFrame
    series: Mapping[str, pd.Series]
    available_dataset_ids: frozenset[str]
    rejected_datasets: Mapping[str, str]
    receipts: tuple[DownloadReceipt, ...]
    split: str


def _repo_campaign_root() -> Path:
    candidates: list[Path] = []
    github_workspace = os.environ.get("GITHUB_WORKSPACE", "").strip()
    if github_workspace:
        candidates.append(Path(github_workspace))
    candidates.extend((Path.cwd(), Path(__file__).resolve().parents[2]))
    for root in candidates:
        campaign = root / "campaigns" / "sp500_long_short_daily"
        if (campaign / "official_inputs").is_dir():
            return campaign.resolve()
    raise DataGateError("SP500_CAMPAIGN_OFFICIAL_INPUTS_NOT_FOUND")


def _epoch_seconds(value: pd.Timestamp) -> int:
    timestamp = pd.Timestamp(value, tz="UTC")
    return int(timestamp.timestamp())


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _store_raw(raw_dir: Path | None, filename: str, payload: bytes) -> None:
    if raw_dir is None:
        return
    root = Path(raw_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    target = root / filename
    target.write_bytes(payload)
    if _sha256(target.read_bytes()) != _sha256(payload):
        raise DataGateError(f"RAW_SNAPSHOT_HASH_MISMATCH:{filename}")


def _bounded_dates(start: Any, end: Any, *, split: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    start_date = pd.Timestamp(start).normalize()
    end_date = pd.Timestamp(end).normalize()
    if end_date >= LOCKED_START:
        raise LockedBoundaryError("TECHNICAL_FAILURE_LOCKED_BREACH:data_request")
    if split == "train" and end_date > TRAIN_END:
        raise DataGateError("TRAIN_REQUEST_CROSSES_SELECTION_BOUNDARY")
    if split == "validation" and (
        start_date < TRAIN_END + pd.Timedelta(days=1) or end_date > VALIDATION_END
    ):
        raise DataGateError("VALIDATION_REQUEST_OUTSIDE_FROZEN_BOUNDARY")
    return start_date, end_date


def _request_bytes(
    session: requests.Session,
    url: str,
    *,
    params: Mapping[str, Any] | None = None,
    attempts: int = 4,
    timeout: int = 60,
) -> bytes:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = session.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            payload = response.content
            if not payload:
                raise DataGateError("EMPTY_HTTP_RESPONSE")
            return payload
        except (requests.RequestException, DataGateError) as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(2**attempt)
    raise DataGateError(f"DOWNLOAD_FAILED:{type(last_error).__name__}")


def _solve_stooq_browser_verification(session: requests.Session, payload: bytes) -> bool:
    """Complete Stooq's deterministic JavaScript proof-of-work using HTTP only."""

    if b"/__verify" not in payload or b"crypto.subtle.digest" not in payload:
        return False
    match = re.search(rb'const c="([^"]+)",d=(\d+)', payload)
    if match is None:
        raise DataGateError("STOOQ_VERIFICATION_SCHEMA_MISMATCH")
    challenge = match.group(1).decode("ascii")
    difficulty = int(match.group(2))
    if difficulty < 1 or difficulty > 6:
        raise DataGateError("STOOQ_VERIFICATION_DIFFICULTY_OUT_OF_RANGE")
    prefix = "0" * difficulty
    nonce: int | None = None
    for candidate in range(10_000_000):
        digest = hashlib.sha256(f"{challenge}{candidate}".encode()).hexdigest()
        if digest.startswith(prefix):
            nonce = candidate
            break
    if nonce is None:
        raise DataGateError("STOOQ_VERIFICATION_NONCE_NOT_FOUND")
    try:
        response = session.post(
            STOOQ_VERIFY,
            data={"c": challenge, "n": str(nonce)},
            timeout=60,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise DataGateError("STOOQ_VERIFICATION_POST_FAILED") from exc
    return True


def _parse_csv(payload: bytes) -> pd.DataFrame:
    try:
        return pd.read_csv(io.BytesIO(payload))
    except Exception as exc:
        raise DataGateError("INVALID_CSV_RESPONSE") from exc


class _StooqHistoryHTMLParser(HTMLParser):
    """Collect Stooq history rows and pagination links from public HTML."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self.hrefs: list[str] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr":
            self._row = []
        elif tag in {"td", "th"} and self._row is not None:
            self._cell = []
        elif tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.hrefs.append(href)

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self._cell is not None and self._row is not None:
            self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if self._row:
                self.rows.append(self._row)
            self._row = None
            self._cell = None


_STOOQ_MONTHS = {
    "jan": 1,
    "sty": 1,
    "feb": 2,
    "lut": 2,
    "mar": 3,
    "apr": 4,
    "kwi": 4,
    "may": 5,
    "maj": 5,
    "jun": 6,
    "cze": 6,
    "jul": 7,
    "lip": 7,
    "aug": 8,
    "sie": 8,
    "sep": 9,
    "wrz": 9,
    "oct": 10,
    "paz": 10,
    "paź": 10,
    "nov": 11,
    "lis": 11,
    "dec": 12,
    "gru": 12,
}


def _parse_stooq_display_date(value: str) -> pd.Timestamp:
    parts = value.replace("\xa0", " ").strip().split()
    if len(parts) != 3:
        raise DataGateError("STOOQ_HTML_DATE_SCHEMA_MISMATCH")
    month = _STOOQ_MONTHS.get(parts[1].casefold())
    if month is None:
        raise DataGateError("STOOQ_HTML_MONTH_SCHEMA_MISMATCH")
    try:
        return pd.Timestamp(year=int(parts[2]), month=month, day=int(parts[0]))
    except ValueError as exc:
        raise DataGateError("STOOQ_HTML_INVALID_DATE") from exc


def _parse_stooq_html_history(payload: bytes) -> tuple[pd.DataFrame, int]:
    """Parse bounded Stooq OHLCV rows without retaining page-level current quotes."""

    decoded = payload.decode("utf-8", errors="ignore")
    if "Exceeded the daily site hits limit" in decoded:
        raise DataGateError("STOOQ_DAILY_HITS_LIMIT")
    parser = _StooqHistoryHTMLParser()
    parser.feed(decoded)
    parsed: list[dict[str, Any]] = []
    for cells in parser.rows:
        if len(cells) < 7 or not cells[0].replace(",", "").isdigit():
            continue
        try:
            parsed.append(
                {
                    "Date": _parse_stooq_display_date(cells[1]),
                    "Open": float(cells[2].replace(",", "")),
                    "High": float(cells[3].replace(",", "")),
                    "Low": float(cells[4].replace(",", "")),
                    "Close": float(cells[5].replace(",", "")),
                    "Volume": int(float(cells[-1].replace(",", ""))),
                }
            )
        except (ValueError, DataGateError) as exc:
            raise DataGateError("STOOQ_HTML_ROW_SCHEMA_MISMATCH") from exc
    if not parsed:
        raise DataGateError("STOOQ_HTML_HISTORY_ROWS_NOT_FOUND")
    page_numbers = [
        int(match.group(1))
        for href in parser.hrefs
        if "q/d/?" in href
        and (match := re.search(r"(?:[?&])l=(\d+)(?:&|$)", href)) is not None
    ]
    page_count = max(page_numbers, default=1)
    frame = pd.DataFrame(parsed).drop_duplicates(subset=["Date"], keep="last")
    return frame, page_count


def _request_stooq_history_page(
    client: requests.Session,
    params: Mapping[str, Any],
    *,
    browser_profile: Path | None,
) -> bytes:
    if browser_profile is None:
        return _request_bytes(
            client,
            STOOQ_HISTORY_PAGE,
            params=params,
            attempts=2,
            timeout=15,
        )
    browser = next(
        (
            path
            for executable in (
                "google-chrome",
                "google-chrome-stable",
                "chromium",
                "chromium-browser",
            )
            if (path := shutil.which(executable)) is not None
        ),
        None,
    )
    if browser is None:
        raise DataGateError("STOOQ_HEADLESS_BROWSER_NOT_AVAILABLE")
    prepared = requests.Request("GET", STOOQ_HISTORY_PAGE, params=params).prepare()
    if prepared.url is None:
        raise DataGateError("STOOQ_HEADLESS_URL_BUILD_FAILED")
    if os.environ.get("GITHUB_ACTIONS", "").strip().lower() == "true":
        return _request_stooq_history_page_via_cdp(
            browser,
            prepared.url,
            browser_profile,
            symbol=str(params.get("s", "spy.us")),
        )
    command = [
        browser,
        "--headless=new",
        "--no-sandbox",
        "--disable-gpu",
        "--disable-dev-shm-usage",
        "--disable-background-networking",
        "--virtual-time-budget=15000",
        f"--user-data-dir={browser_profile}",
        "--dump-dom",
        prepared.url,
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            timeout=90,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise DataGateError("STOOQ_HEADLESS_BROWSER_FAILED") from exc
    payload = bytes(completed.stdout)
    if completed.returncode != 0 or not payload:
        raise DataGateError(
            f"STOOQ_HEADLESS_BROWSER_FAILED:returncode={completed.returncode}"
        )
    return payload


async def _cdp_render_stooq_history(
    websocket_url: str,
    *,
    landing_url: str,
    history_url: str,
) -> bytes:
    """Render history after a same-session landing navigation.

    Stooq's operation mask is generated by its historical-data form. A direct
    top-level request can ignore that mask, so retain the landing-page session
    and referrer before requesting the bounded history page.
    """

    async with websockets.connect(
        websocket_url,
        max_size=16 * 1024 * 1024,
        open_timeout=20,
    ) as connection:
        command_id = 0
        session_id: str | None = None

        async def command(
            method: str,
            params: Mapping[str, Any] | None = None,
            *,
            browser_scope: bool = False,
        ) -> Any:
            nonlocal command_id
            command_id += 1
            expected_id = command_id
            message: dict[str, Any] = {
                "id": expected_id,
                "method": method,
                "params": dict(params or {}),
            }
            if session_id is not None and not browser_scope:
                message["sessionId"] = session_id
            await connection.send(
                json.dumps(message)
            )
            while True:
                message = json.loads(await asyncio.wait_for(connection.recv(), timeout=30))
                if message.get("id") != expected_id:
                    continue
                if "error" in message:
                    raise DataGateError(
                        f"STOOQ_HEADLESS_CDP_COMMAND_FAILED:{method}:{message['error']}"
                    )
                return message.get("result", {})

        async def navigate(url: str, *, referrer: str | None = None) -> None:
            payload: dict[str, Any] = {"url": url}
            if referrer is not None:
                payload["referrer"] = referrer
            await command("Page.navigate", payload)
            deadline = time.monotonic() + 30.0
            while time.monotonic() < deadline:
                state = await command(
                    "Runtime.evaluate",
                    {"expression": "document.readyState", "returnByValue": True},
                )
                value = state.get("result", {}).get("value")
                if value == "complete":
                    return
                await asyncio.sleep(0.1)
            raise DataGateError("STOOQ_HEADLESS_CDP_NAVIGATION_TIMEOUT")

        targets = await command("Target.getTargets", browser_scope=True)
        target_id = next(
            (
                info.get("targetId")
                for info in targets.get("targetInfos", [])
                if info.get("type") == "page" and info.get("targetId")
            ),
            None,
        )
        if not isinstance(target_id, str):
            raise DataGateError("STOOQ_HEADLESS_CDP_TARGET_NOT_FOUND")
        attached = await command(
            "Target.attachToTarget",
            {"targetId": target_id, "flatten": True},
            browser_scope=True,
        )
        attached_session = attached.get("sessionId")
        if not isinstance(attached_session, str):
            raise DataGateError("STOOQ_HEADLESS_CDP_ATTACH_FAILED")
        session_id = attached_session
        await command("Page.enable")
        await command("Runtime.enable")
        await command("Network.enable")
        cookie = await command(
            "Network.setCookie",
            {
                "name": "privacy",
                "value": str(int(time.time())),
                "url": "https://stooq.com/",
                "expires": time.time() + 30 * 24 * 60 * 60,
                "secure": True,
            },
        )
        if cookie.get("success") is not True:
            raise DataGateError("STOOQ_HEADLESS_PRIVACY_COOKIE_FAILED")
        await navigate(landing_url)
        await navigate(history_url, referrer=landing_url)
        rendered = await command(
            "Runtime.evaluate",
            {
                "expression": "document.documentElement.outerHTML",
                "returnByValue": True,
            },
        )
        html = rendered.get("result", {}).get("value")
        if not isinstance(html, str) or not html:
            raise DataGateError("STOOQ_HEADLESS_CDP_EMPTY_DOM")
        return html.encode("utf-8")


def _request_stooq_history_page_via_cdp(
    browser: str,
    history_url: str,
    browser_profile: Path,
    *,
    symbol: str,
) -> bytes:
    """Use one headless browser session so Stooq preserves form semantics."""

    command = [
        browser,
        "--headless=new",
        "--no-sandbox",
        "--disable-gpu",
        "--disable-dev-shm-usage",
        "--disable-background-networking",
        "--remote-debugging-port=0",
        "--remote-allow-origins=*",
        f"--user-data-dir={browser_profile}",
        "about:blank",
    ]
    port_file = browser_profile / "DevToolsActivePort"
    port_file.unlink(missing_ok=True)
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
    )
    except OSError as exc:
        raise DataGateError("STOOQ_HEADLESS_BROWSER_FAILED") from exc
    try:
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline and not port_file.is_file():
            if process.poll() is not None:
                raise DataGateError("STOOQ_HEADLESS_BROWSER_FAILED:early_exit")
            time.sleep(0.05)
        if not port_file.is_file():
            raise DataGateError("STOOQ_HEADLESS_CDP_PORT_TIMEOUT")
        port_lines = port_file.read_text(encoding="utf-8").splitlines()
        if len(port_lines) < 2:
            raise DataGateError("STOOQ_HEADLESS_CDP_PORT_INVALID")
        port = int(port_lines[0])
        browser_websocket_url = f"ws://127.0.0.1:{port}{port_lines[1]}"
        landing = requests.Request(
            "GET",
            STOOQ_HISTORY_PAGE,
            params={"s": symbol.lower()},
        ).prepare().url
        if landing is None:
            raise DataGateError("STOOQ_HEADLESS_URL_BUILD_FAILED")
        endpoint_deadline = time.monotonic() + 20.0
        last_connection_error: OSError | None = None
        while time.monotonic() < endpoint_deadline:
            try:
                return asyncio.run(
                    _cdp_render_stooq_history(
                        browser_websocket_url,
                        landing_url=landing,
                        history_url=history_url,
                    )
                )
            except OSError as exc:
                last_connection_error = exc
                time.sleep(0.05)
        raise DataGateError("STOOQ_HEADLESS_CDP_ENDPOINT_TIMEOUT") from last_connection_error
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise DataGateError("STOOQ_HEADLESS_CDP_FAILED") from exc
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


def _load_stooq_history_page(
    client: requests.Session,
    params: Mapping[str, Any],
    *,
    browser_profile: Path | None,
    attempts: int = 3,
) -> tuple[bytes, pd.DataFrame, int]:
    """Fetch and validate one page, retrying transient verification screens."""

    last_error: DataGateError | None = None
    for attempt in range(1, attempts + 1):
        payload = _request_stooq_history_page(
            client,
            params,
            browser_profile=browser_profile,
        )
        if browser_profile is None and _solve_stooq_browser_verification(client, payload):
            payload = _request_stooq_history_page(
                client,
                params,
                browser_profile=None,
            )
        try:
            frame, page_count = _parse_stooq_html_history(payload)
            return payload, frame, page_count
        except DataGateError as exc:
            if str(exc) != "STOOQ_HTML_HISTORY_ROWS_NOT_FOUND" or attempt == attempts:
                raise
            last_error = exc
            time.sleep(float(min(30, 5 * attempt)))
    raise DataGateError("STOOQ_HTML_HISTORY_ROWS_NOT_FOUND") from last_error


def _download_stooq_html_history(
    client: requests.Session,
    symbol: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> tuple[pd.DataFrame, bytes, str, int, int]:
    """Read raw Stooq history in bounded windows below its public row cap."""

    browser_profile_root: Path | None = None
    if isinstance(client, requests.Session):
        client.cookies.set(
            "privacy",
            str(int(time.time())),
            domain="stooq.com",
            path="/",
        )
    windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    window_start = start_date
    while window_start <= end_date:
        window_end = min(
            window_start + pd.DateOffset(years=STOOQ_MAX_WINDOW_YEARS) - pd.Timedelta(days=1),
            end_date,
        )
        windows.append((window_start, window_end))
        window_start = window_end + pd.Timedelta(days=1)

    frames: list[pd.DataFrame] = []
    payload_hashes: list[str] = []
    total_page_count = 0
    for window_number, (bounded_start, bounded_end) in enumerate(windows, start=1):
        if browser_profile_root is not None and window_number > 1:
            print(
                "[sp500-data] stooq public rate cooldown "
                f"seconds={STOOQ_WINDOW_COOLDOWN_SECONDS:g}",
                flush=True,
            )
            time.sleep(STOOQ_WINDOW_COOLDOWN_SECONDS)
        browser_profile = (
            browser_profile_root / f"window-{window_number:03d}"
            if browser_profile_root is not None
            else None
        )
        if browser_profile is not None:
            browser_profile.mkdir(parents=True, exist_ok=True)
        print(
            "[sp500-data] stooq window start "
            f"number={window_number}/{len(windows)} "
            f"start={bounded_start.date()} end={bounded_end.date()}",
            flush=True,
        )
        first_params: dict[str, Any] = {
            "s": symbol.lower(),
            "f": bounded_start.strftime("%Y%m%d"),
            "t": bounded_end.strftime("%Y%m%d"),
            **STOOQ_RAW_OPERATION_PARAMS,
        }
        first_payload, first_frame, page_count = _load_stooq_history_page(
            client,
            first_params,
            browser_profile=browser_profile,
        )
        if page_count > STOOQ_MAX_BOUNDED_PAGES:
            raise DataGateError(f"STOOQ_HTML_PAGE_COUNT_OUT_OF_RANGE:{page_count}")
        window_frames = [first_frame]
        payload_hashes.append(_sha256(first_payload))

        for page in range(2, page_count + 1):
            if isinstance(client, requests.Session):
                time.sleep(STOOQ_PAGE_DELAY_SECONDS)
            page_client = client
            owned_client: requests.Session | None = None
            if isinstance(client, requests.Session):
                owned_client = requests.Session()
                owned_client.headers.update(client.headers)
                owned_client.headers["Referer"] = STOOQ_HISTORY_PAGE
                owned_client.cookies.update(client.cookies)
                page_client = owned_client
            try:
                page_payload, page_frame, reported_page_count = _load_stooq_history_page(
                    page_client,
                    {
                        "s": symbol.lower(),
                        "f": bounded_start.strftime("%Y%m%d"),
                        "t": bounded_end.strftime("%Y%m%d"),
                        "l": page,
                        **STOOQ_RAW_OPERATION_PARAMS,
                    },
                    browser_profile=browser_profile,
                )
            except DataGateError as exc:
                accumulated_rows = sum(len(item) for item in window_frames)
                terminal_public_cap = (
                    str(exc) == "STOOQ_HTML_HISTORY_ROWS_NOT_FOUND"
                    and page == page_count
                    and accumulated_rows >= STOOQ_PUBLIC_HISTORY_ROW_CAP
                )
                if not terminal_public_cap:
                    raise DataGateError(
                        "STOOQ_WINDOW_PAGE_FAILED:"
                        f"window={window_number}:page={page}:"
                        f"start={bounded_start.date()}:end={bounded_end.date()}:"
                        f"cause={exc}"
                    ) from exc
                print(
                    "[sp500-data] stooq terminal empty page accepted "
                    f"after public row cap={accumulated_rows}",
                    flush=True,
                )
                break
            finally:
                if owned_client is not None:
                    owned_client.close()
            if reported_page_count > page_count:
                raise DataGateError("STOOQ_HTML_PAGINATION_CHANGED_DURING_DOWNLOAD")
            window_frames.append(page_frame)
            payload_hashes.append(_sha256(page_payload))
        frames.extend(window_frames)
        total_page_count += page_count
        print(
            "[sp500-data] stooq window complete "
            f"number={window_number}/{len(windows)} rows={sum(len(item) for item in window_frames)} "
            f"pages={page_count}",
            flush=True,
        )

    frame = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=["Date"], keep="last")
        .sort_values("Date", kind="mergesort")
        .reset_index(drop=True)
    )
    canonical_payload = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    response_chain_hash = _sha256("\n".join(payload_hashes).encode("ascii"))
    return frame, canonical_payload, response_chain_hash, total_page_count, len(windows)


def _assert_response_date_bound(
    frame: pd.DataFrame,
    *,
    date_column: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    label: str,
) -> pd.DataFrame:
    dates = pd.to_datetime(frame[date_column], errors="coerce")
    if dates.isna().any():
        raise DataGateError(f"INVALID_DATES:{label}")
    normalized = dates.dt.normalize()
    if len(normalized) and (normalized.min() < start or normalized.max() > end):
        raise DataGateError(f"UNBOUNDED_SOURCE_RESPONSE:{label}")
    result = frame.copy()
    result[date_column] = normalized
    assert_frame_before_locked(result, label=label)
    return result


def _parse_yahoo_chart(
    payload: bytes,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, bytes]:
    """Parse only bounded daily observations and discard Yahoo quote metadata."""

    try:
        document = json.loads(payload)
        chart = document["chart"]
        if chart.get("error") is not None:
            raise DataGateError("YAHOO_CHART_ERROR")
        results = chart.get("result") or []
        if len(results) != 1:
            raise DataGateError("YAHOO_CHART_RESULT_COUNT")
        result = results[0]
        timestamps = result.get("timestamp") or []
        quote = (result.get("indicators", {}).get("quote") or [])[0]
        adjusted_groups = result.get("indicators", {}).get("adjclose") or []
        adjusted = adjusted_groups[0].get("adjclose", []) if adjusted_groups else []
    except DataGateError:
        raise
    except (IndexError, KeyError, TypeError, ValueError) as exc:
        raise DataGateError("YAHOO_CHART_SCHEMA_MISMATCH") from exc

    columns = {
        "open": quote.get("open", []),
        "high": quote.get("high", []),
        "low": quote.get("low", []),
        "close": quote.get("close", []),
        "volume": quote.get("volume", []),
    }
    lengths = {len(timestamps), len(adjusted), *(len(values) for values in columns.values())}
    if len(lengths) != 1:
        raise DataGateError("YAHOO_CHART_ARRAY_LENGTH_MISMATCH")

    dates = pd.to_datetime(timestamps, unit="s", utc=True).tz_localize(None).normalize()
    prices = pd.DataFrame(
        {
            "date": dates,
            **columns,
            "adj_close": adjusted,
        }
    )[["date", "open", "high", "low", "close", "adj_close", "volume"]]
    prices = _assert_response_date_bound(
        prices,
        date_column="date",
        start=start,
        end=end,
        label="yahoo_history",
    )

    events = result.get("events") or {}

    def event_frame(name: str, value_name: str) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for event in (events.get(name) or {}).values():
            event_timestamp = event.get("date")
            if event_timestamp is None:
                raise DataGateError(f"YAHOO_EVENT_DATE_MISSING:{name}")
            if name == "splits":
                numerator = event.get("numerator")
                denominator = event.get("denominator")
                if numerator is None or denominator in (None, 0):
                    raise DataGateError("YAHOO_SPLIT_RATIO_MISSING")
                value = float(numerator) / float(denominator)
            else:
                value = event.get("amount")
            rows.append(
                {
                    "date": pd.to_datetime(event_timestamp, unit="s", utc=True)
                    .tz_localize(None)
                    .normalize(),
                    value_name: value,
                }
            )
        frame = pd.DataFrame(rows, columns=["date", value_name])
        if len(frame):
            frame[value_name] = pd.to_numeric(frame[value_name], errors="raise")
            frame = frame.sort_values("date", kind="mergesort").reset_index(drop=True)
        return _assert_response_date_bound(
            frame,
            date_column="date",
            start=start,
            end=end,
            label=f"yahoo_{name}",
        )

    dividends = event_frame("dividends", "distribution")
    splits = event_frame("splits", "split_ratio")
    bounded_prices = prices.assign(date=prices["date"].dt.strftime("%Y-%m-%d"))
    bounded_prices = bounded_prices.astype(object).where(pd.notna(bounded_prices), None)
    bounded_snapshot = json.dumps(
        {
            "prices": bounded_prices.to_dict(orient="records"),
            "dividends": dividends.assign(date=dividends["date"].dt.strftime("%Y-%m-%d")).to_dict(
                orient="records"
            ),
            "splits": splits.assign(date=splits["date"].dt.strftime("%Y-%m-%d")).to_dict(
                orient="records"
            ),
        },
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return prices, dividends, splits, bounded_snapshot


def download_yahoo_history(
    symbol: str,
    start: Any,
    end: Any,
    *,
    split: str,
    session: requests.Session | None = None,
    raw_dir: Path | None = None,
    include_events: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, tuple[DownloadReceipt, ...]]:
    """Download bounded raw OHLC and optional corporate events from Yahoo chart JSON."""

    start_date, end_date = _bounded_dates(start, end, split=split)
    client = session or requests.Session()
    client.headers.update({"User-Agent": "Mozilla/5.0 AuroraResearch bounded-chart-json"})
    params: dict[str, Any] = {
        "period1": _epoch_seconds(start_date),
        "period2": _epoch_seconds(end_date + pd.Timedelta(days=1)),
        "interval": "1d",
        "includeAdjustedClose": "true",
        "includePrePost": "false",
    }
    if include_events:
        params["events"] = "div,splits"
    payload: bytes | None = None
    selected_url: str | None = None
    errors: list[str] = []
    for endpoint in YAHOO_CHART_ENDPOINTS:
        selected_url = endpoint.format(symbol=symbol)
        try:
            payload = _request_bytes(client, selected_url, params=params, attempts=3)
            break
        except DataGateError as exc:
            errors.append(str(exc))
    if payload is None or selected_url is None:
        raise DataGateError(f"YAHOO_CHART_ENDPOINTS_FAILED:{'|'.join(errors)}")

    prices, dividends, splits, bounded_payload = _parse_yahoo_chart(
        payload,
        start=start_date,
        end=end_date,
    )
    _store_raw(raw_dir, f"yahoo_{symbol.lower()}_bounded_chart.json", bounded_payload)
    dates = prices["date"]
    receipts = (
        DownloadReceipt(
            dataset_id="YAHOO_SPY_BOUNDED_CHART",
            url_template=selected_url,
            sha256=_sha256(bounded_payload),
            byte_count=len(bounded_payload),
            minimum_date=dates.min().date().isoformat() if len(dates) else None,
            maximum_date=dates.max().date().isoformat() if len(dates) else None,
            status="downloaded_bounded_chart_json_current_metadata_discarded",
        ),
    )
    return prices, dividends, splits, receipts


def _parse_kibot_daily_history(
    payload: bytes,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Parse a bounded unadjusted Kibot daily response."""

    if payload.lstrip().startswith((b"400 ", b"401 ", b"402 ", b"403 ", b"404 ")):
        message = payload.decode("utf-8", errors="replace").strip().replace("\n", " ")
        raise DataGateError(f"KIBOT_HISTORY_ERROR:{message}")
    try:
        frame = pd.read_csv(
            io.BytesIO(payload),
            header=None,
            names=["date", "open", "high", "low", "close", "volume"],
        )
    except (OSError, pd.errors.ParserError) as exc:
        raise DataGateError("KIBOT_HISTORY_SCHEMA_MISMATCH") from exc
    if frame.empty or frame.shape[1] != 6:
        raise DataGateError("KIBOT_HISTORY_EMPTY_OR_INVALID")
    try:
        for column in ("open", "high", "low", "close", "volume"):
            frame[column] = pd.to_numeric(frame[column], errors="raise")
    except (TypeError, ValueError) as exc:
        raise DataGateError("KIBOT_HISTORY_NON_NUMERIC") from exc
    frame = _assert_response_date_bound(
        frame,
        date_column="date",
        start=start,
        end=end,
        label="kibot_spy_unadjusted_daily",
    )
    if frame["date"].duplicated().any():
        raise DataGateError("KIBOT_HISTORY_DUPLICATE_DATES")
    if (frame[["open", "high", "low", "close"]] <= 0.0).any().any():
        raise DataGateError("KIBOT_HISTORY_NON_POSITIVE_PRICE")
    if (frame["volume"] < 0.0).any():
        raise DataGateError("KIBOT_HISTORY_NEGATIVE_VOLUME")
    return frame.sort_values("date", kind="mergesort").reset_index(drop=True)


def download_kibot_unadjusted_history(
    symbol: str,
    start: Any,
    end: Any,
    *,
    split: str,
    session: requests.Session | None = None,
    raw_dir: Path | None = None,
) -> tuple[pd.DataFrame, DownloadReceipt]:
    """Download bounded raw ETF OHLCV from Kibot's documented guest API."""

    start_date, end_date = _bounded_dates(start, end, split=split)
    client = session or requests.Session()
    client.headers.update({"User-Agent": "Mozilla/5.0 AuroraResearch bounded-kibot-eod"})
    login_payload = _request_bytes(
        client,
        KIBOT_API_ENDPOINT,
        params={"action": "login", "user": "guest", "password": "guest"},
        attempts=3,
    )
    if not login_payload.lstrip().startswith(b"200 OK"):
        raise DataGateError("KIBOT_GUEST_LOGIN_FAILED")
    params = {
        "action": "history",
        "symbol": symbol,
        "interval": "Daily",
        "startdate": start_date.strftime("%m/%d/%Y"),
        "enddate": end_date.strftime("%m/%d/%Y"),
        "unadjusted": "1",
        "type": "etfs",
    }
    payload = _request_bytes(client, KIBOT_API_ENDPOINT, params=params, attempts=3)
    frame = _parse_kibot_daily_history(payload, start=start_date, end=end_date)
    filename = f"kibot_{symbol.lower()}_unadjusted_daily.csv"
    _store_raw(raw_dir, filename, payload)
    dates = frame["date"]
    return frame, DownloadReceipt(
        dataset_id="KIBOT_SPY_UNADJUSTED_ADJUDICATOR",
        url_template=KIBOT_API_ENDPOINT,
        sha256=_sha256(payload),
        byte_count=len(payload),
        minimum_date=dates.min().date().isoformat(),
        maximum_date=dates.max().date().isoformat(),
        status="downloaded_bounded_guest_daily_unadjusted",
        reason="third_source_open_field_adjudication_only",
    )


def _adjudicate_stooq_open_prices(
    yahoo_prices: pd.DataFrame,
    stooq_prices: pd.DataFrame,
    kibot_prices: pd.DataFrame,
) -> tuple[pd.DataFrame, Mapping[str, Any]]:
    """Repair Stooq execution and signal prices with three-source consensus."""

    yahoo = yahoo_prices.set_index("date").sort_index(kind="mergesort")
    stooq = stooq_prices.set_index("date").sort_index(kind="mergesort")
    kibot = kibot_prices.set_index("date").sort_index(kind="mergesort")
    expected = yahoo.index.intersection(stooq.index)
    common = expected.intersection(kibot.index)
    if len(common) != len(expected) or not common.equals(expected):
        missing = expected.difference(kibot.index)
        details = ",".join(date.date().isoformat() for date in missing[:20])
        raise DataGateError(f"KIBOT_ADJUDICATOR_INCOMPLETE:{details}")

    canonical = stooq.copy()
    field_audits: dict[str, Mapping[str, Any]] = {}
    for field in ("open", "close"):
        yahoo_value = yahoo.loc[common, field].astype(float)
        stooq_value = stooq.loc[common, field].astype(float)
        kibot_value = kibot.loc[common, field].astype(float)
        yahoo_stooq = (yahoo_value - stooq_value).abs() / yahoo_value.abs()
        yahoo_kibot = (yahoo_value - kibot_value).abs() / yahoo_value.abs()
        stooq_kibot = (stooq_value - kibot_value).abs() / kibot_value.abs()
        vendor_disagreement = yahoo_stooq > SPY_RETURN_TOLERANCE
        kibot_supports_yahoo = yahoo_kibot <= SPY_RETURN_TOLERANCE
        kibot_supports_stooq = stooq_kibot <= SPY_RETURN_TOLERANCE
        yahoo_repairs = vendor_disagreement & kibot_supports_yahoo & ~kibot_supports_stooq
        bridge_repairs = vendor_disagreement & kibot_supports_yahoo & kibot_supports_stooq
        retained_stooq = vendor_disagreement & ~kibot_supports_yahoo & kibot_supports_stooq
        no_pair_agrees = vendor_disagreement & ~kibot_supports_yahoo & ~kibot_supports_stooq
        vendor_values = pd.concat([yahoo_value, stooq_value, kibot_value], axis=1)
        median_value = vendor_values.median(axis=1)
        relative_spread = (
            vendor_values.max(axis=1) - vendor_values.min(axis=1)
        ) / median_value.abs()
        median_repairs = no_pair_agrees & (
            relative_spread <= SPY_MEDIAN_ADJUDICATION_MAX_SPREAD
        )
        yahoo_volume = yahoo.loc[common, "volume"].astype(float)
        stooq_volume = stooq.loc[common, "volume"].astype(float)
        kibot_volume = kibot.loc[common, "volume"].astype(float)
        primary_volume_scale = pd.concat(
            [yahoo_volume.abs(), stooq_volume.abs()], axis=1
        ).max(axis=1).clip(lower=1.0)
        primary_volume_gap = (yahoo_volume - stooq_volume).abs() / primary_volume_scale
        adjudicator_volume_scale = pd.concat(
            [yahoo_volume.abs(), kibot_volume.abs()], axis=1
        ).max(axis=1).clip(lower=1.0)
        adjudicator_volume_gap = (
            yahoo_volume - kibot_volume
        ).abs() / adjudicator_volume_scale
        primary_price_gap = (yahoo_value - stooq_value).abs()
        tick_boundary = primary_price_gap <= (
            SPY_RETURN_TOLERANCE * yahoo_value.abs() + SPY_PRICE_TICK_USD
        )
        primary_volume_supported_repairs = (
            no_pair_agrees
            & ~median_repairs
            & tick_boundary
            & (primary_volume_gap <= SPY_PRIMARY_VOLUME_RELATIVE_TOLERANCE)
            & (adjudicator_volume_gap > SPY_ADJUDICATOR_VOLUME_OUTLIER_TOLERANCE)
        )
        unresolved = no_pair_agrees & ~median_repairs & ~primary_volume_supported_repairs
        changed = (
            yahoo_repairs
            | bridge_repairs
            | median_repairs
            | primary_volume_supported_repairs
        )

        canonical.loc[yahoo_repairs, field] = yahoo_value.loc[yahoo_repairs]
        canonical.loc[bridge_repairs, field] = kibot_value.loc[bridge_repairs]
        canonical.loc[median_repairs, field] = median_value.loc[median_repairs]
        canonical.loc[primary_volume_supported_repairs, field] = yahoo_value.loc[
            primary_volume_supported_repairs
        ]

        def dates(mask: pd.Series) -> list[str]:
            return [date.date().isoformat() for date in common[mask.to_numpy()]]

        field_audits[field] = {
            "vendor_disagreement_count": int(vendor_disagreement.sum()),
            "changed_count": int(changed.sum()),
            "yahoo_supported_repair_count": int(yahoo_repairs.sum()),
            "kibot_bridge_repair_count": int(bridge_repairs.sum()),
            "three_source_median_repair_count": int(median_repairs.sum()),
            "primary_volume_supported_repair_count": int(
                primary_volume_supported_repairs.sum()
            ),
            "retained_stooq_count": int(retained_stooq.sum()),
            "unresolved_level_count": int(unresolved.sum()),
            "yahoo_supported_repair_dates": dates(yahoo_repairs),
            "kibot_bridge_repair_dates": dates(bridge_repairs),
            "three_source_median_repair_dates": dates(median_repairs),
            "primary_volume_supported_repair_dates": dates(
                primary_volume_supported_repairs
            ),
            "retained_stooq_dates": dates(retained_stooq),
            "unresolved_level_dates": dates(unresolved),
        }

    expanded_high = canonical[["open", "close"]].max(axis=1) > canonical["high"]
    expanded_low = canonical[["open", "close"]].min(axis=1) < canonical["low"]
    canonical.loc[expanded_high, "high"] = canonical.loc[
        expanded_high, ["open", "close"]
    ].max(axis=1)
    canonical.loc[expanded_low, "low"] = canonical.loc[
        expanded_low, ["open", "close"]
    ].min(axis=1)

    open_audit = field_audits["open"]
    close_audit = field_audits["close"]

    audit = {
        "method": "three_source_execution_and_signal_price_consensus_v4",
        "tolerance": SPY_RETURN_TOLERANCE,
        "median_adjudication_max_spread": SPY_MEDIAN_ADJUDICATION_MAX_SPREAD,
        "price_tick_usd": SPY_PRICE_TICK_USD,
        "primary_volume_relative_tolerance": SPY_PRIMARY_VOLUME_RELATIVE_TOLERANCE,
        "adjudicator_volume_outlier_tolerance": (
            SPY_ADJUDICATOR_VOLUME_OUTLIER_TOLERANCE
        ),
        "overlap_rows": len(common),
        "fields": field_audits,
        "vendor_disagreement_count": open_audit["vendor_disagreement_count"],
        "changed_open_count": open_audit["changed_count"],
        "changed_close_count": close_audit["changed_count"],
        "yahoo_supported_repair_count": open_audit["yahoo_supported_repair_count"],
        "kibot_bridge_repair_count": open_audit["kibot_bridge_repair_count"],
        "retained_stooq_count": open_audit["retained_stooq_count"],
        "unresolved_level_count": open_audit["unresolved_level_count"],
        "unresolved_close_level_count": close_audit["unresolved_level_count"],
        "expanded_high_count": int(expanded_high.sum()),
        "expanded_low_count": int(expanded_low.sum()),
        "yahoo_supported_repair_dates": open_audit["yahoo_supported_repair_dates"],
        "kibot_bridge_repair_dates": open_audit["kibot_bridge_repair_dates"],
        "retained_stooq_dates": open_audit["retained_stooq_dates"],
        "unresolved_level_dates": open_audit["unresolved_level_dates"],
        "expanded_high_dates": [
            date.date().isoformat() for date in canonical.index[expanded_high.to_numpy()]
        ],
        "expanded_low_dates": [
            date.date().isoformat() for date in canonical.index[expanded_low.to_numpy()]
        ],
    }
    return canonical.reset_index(names="date"), audit


def download_stooq_history(
    symbol: str,
    start: Any,
    end: Any,
    *,
    split: str,
    session: requests.Session | None = None,
    raw_dir: Path | None = None,
) -> tuple[pd.DataFrame, DownloadReceipt]:
    start_date, end_date = _bounded_dates(start, end, split=split)
    prebuilt_path_value = os.environ.get("SP500_STOOQ_HISTORY_CSV", "").strip()
    if prebuilt_path_value:
        prebuilt_path = Path(prebuilt_path_value).resolve()
        manifest_path_value = os.environ.get("SP500_STOOQ_HISTORY_MANIFEST", "").strip()
        if not prebuilt_path.is_file() or not manifest_path_value:
            raise DataGateError("STOOQ_PREBUILT_INPUTS_INCOMPLETE")
        manifest_path = Path(manifest_path_value).resolve()
        if not manifest_path.is_file():
            raise DataGateError("STOOQ_PREBUILT_MANIFEST_NOT_FOUND")
        payload = prebuilt_path.read_bytes()
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DataGateError("STOOQ_PREBUILT_MANIFEST_INVALID") from exc
        if manifest.get("merged_sha256") != _sha256(payload):
            raise DataGateError("STOOQ_PREBUILT_HASH_MISMATCH")
        frame = _parse_csv(payload)
        expected_columns = {"date", "open", "high", "low", "close", "volume"}
        frame.columns = [str(column).strip().lower() for column in frame.columns]
        if not expected_columns.issubset(frame.columns):
            raise DataGateError("STOOQ_PREBUILT_SCHEMA_MISMATCH")
        frame = _assert_response_date_bound(
            frame,
            date_column="date",
            start=start_date,
            end=end_date,
            label=f"stooq_prebuilt_{symbol}",
        )
        frame = (
            frame.loc[:, ["date", "open", "high", "low", "close", "volume"]]
            .drop_duplicates(subset=["date"], keep="last")
            .sort_values("date", kind="mergesort")
            .reset_index(drop=True)
        )
        if frame.empty:
            raise DataGateError("STOOQ_PREBUILT_EMPTY")
        dates = frame["date"]
        manifest_source = str(
            manifest.get("source") or "stooq_public_html_raw_unadjusted"
        )
        fallback_used = "fallback" in manifest_source
        prebuilt_transport = (
            "github_actions_sharded_provider_fallback"
            if fallback_used
            else "github_actions_sharded_headless_chrome"
        )
        _store_raw(raw_dir, f"stooq_{symbol.replace('.', '_').lower()}_history.csv", payload)
        return frame, DownloadReceipt(
            dataset_id="DS002",
            url_template=(
                YAHOO_CHART_ENDPOINTS[0]
                if "yahoo" in manifest_source
                else STOOQ_HISTORY_PAGE
            ),
            sha256=_sha256(payload),
            byte_count=len(payload),
            minimum_date=dates.min().date().isoformat(),
            maximum_date=dates.max().date().isoformat(),
            status=(
                "loaded_github_sharded_documented_free_fallback_raw_unadjusted"
                if fallback_used
                else "loaded_github_sharded_html_history_raw_unadjusted"
            ),
            reason=(
                f"window_count={int(manifest.get('window_count', 0))};"
                f"manifest_sha256={_sha256(manifest_path.read_bytes())};"
                f"effective_source={manifest_source};"
                f"transport={prebuilt_transport}"
            ),
        )
    client = session or requests.Session()
    client.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
    )
    frame, payload, html_chain_hash, page_count, window_count = _download_stooq_html_history(
        client,
        symbol,
        start_date,
        end_date,
    )
    frame = _assert_response_date_bound(
        frame,
        date_column="Date",
        start=start_date,
        end=end_date,
        label=f"stooq_{symbol}",
    ).rename(columns=str.lower)
    _store_raw(raw_dir, f"stooq_{symbol.replace('.', '_').lower()}_history.csv", payload)
    dates = frame["date"]
    receipt = DownloadReceipt(
        dataset_id="DS002",
        url_template=STOOQ_HISTORY_PAGE,
        sha256=_sha256(payload),
        byte_count=len(payload),
        minimum_date=dates.min().date().isoformat() if len(dates) else None,
        maximum_date=dates.max().date().isoformat() if len(dates) else None,
        status="downloaded_bounded_html_public_history_raw_unadjusted",
        reason=(
            f"window_count={window_count};page_count={page_count};"
            f"operation_adjustments_skipped={','.join(STOOQ_RAW_OPERATION_PARAMS)};"
            f"raw_response_chain_sha256={html_chain_hash};"
            "transport="
            + (
                "headless_chrome"
                if os.environ.get("GITHUB_ACTIONS", "").strip().lower() == "true"
                else "https"
            )
        ),
    )
    return frame[["date", "open", "high", "low", "close", "volume"]], receipt


def download_fred_series(
    series_id: str,
    dataset_id: str,
    start: Any,
    end: Any,
    *,
    split: str,
    session: requests.Session | None = None,
    raw_dir: Path | None = None,
) -> tuple[pd.Series, DownloadReceipt]:
    start_date, end_date = _bounded_dates(start, end, split=split)
    payload = _request_bytes(
        session or requests.Session(),
        FRED_DOWNLOAD,
        params={
            "id": series_id,
            "cosd": start_date.date().isoformat(),
            "coed": end_date.date().isoformat(),
        },
    )
    frame = _parse_csv(payload)
    if "DATE" not in frame.columns or series_id not in frame.columns:
        raise DataGateError(f"FRED_SCHEMA_MISMATCH:{series_id}")
    frame = _assert_response_date_bound(
        frame,
        date_column="DATE",
        start=start_date,
        end=end_date,
        label=f"fred_{series_id}",
    )
    _store_raw(raw_dir, f"fred_{dataset_id}_{series_id}.csv", payload)
    values = pd.to_numeric(frame[series_id].replace(".", pd.NA), errors="coerce")
    series = pd.Series(values.to_numpy(), index=pd.DatetimeIndex(frame["DATE"]), name=series_id)
    series = series.dropna().sort_index(kind="mergesort")
    receipt = DownloadReceipt(
        dataset_id=dataset_id,
        url_template=FRED_DOWNLOAD,
        sha256=_sha256(payload),
        byte_count=len(payload),
        minimum_date=series.index.min().date().isoformat() if len(series) else None,
        maximum_date=series.index.max().date().isoformat() if len(series) else None,
        status="downloaded",
    )
    return series, receipt


def download_alfred_initial_series(
    series_id: str,
    dataset_id: str,
    start: Any,
    end: Any,
    *,
    split: str,
    session: requests.Session | None = None,
    api_key: str | None = None,
    raw_dir: Path | None = None,
) -> tuple[pd.DataFrame, DownloadReceipt]:
    """Download initial releases only and retain their actual release dates."""

    start_date, end_date = _bounded_dates(start, end, split=split)
    key = api_key or os.environ.get("FRED_API_KEY", "").strip()
    if not key:
        raise DataGateError("FRED_API_KEY_REQUIRED_FOR_INITIAL_RELEASE_VINTAGES")
    payload = _request_bytes(
        session or requests.Session(),
        FRED_API_OBSERVATIONS,
        params={
            "series_id": series_id,
            "api_key": key,
            "file_type": "json",
            "output_type": 4,
            "observation_start": start_date.date().isoformat(),
            "observation_end": end_date.date().isoformat(),
            "realtime_start": "1776-07-04",
            "realtime_end": end_date.date().isoformat(),
            "limit": 100000,
            "sort_order": "asc",
        },
    )
    try:
        decoded = json.loads(payload)
        observations = decoded["observations"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise DataGateError(f"ALFRED_SCHEMA_MISMATCH:{series_id}") from exc
    rows = []
    for observation in observations:
        value = pd.to_numeric(observation.get("value"), errors="coerce")
        if pd.isna(value):
            continue
        observation_date = pd.Timestamp(observation["date"]).normalize()
        release_date = pd.Timestamp(observation["realtime_start"]).normalize()
        if observation_date < start_date or observation_date > end_date:
            raise DataGateError(f"UNBOUNDED_SOURCE_RESPONSE:alfred_{series_id}")
        if release_date > end_date:
            raise DataGateError(f"POST_PHASE_RELEASE_IN_RESPONSE:alfred_{series_id}")
        if release_date >= LOCKED_START:
            raise LockedBoundaryError("TECHNICAL_FAILURE_LOCKED_BREACH:alfred_release")
        rows.append(
            {
                "observation_date": observation_date,
                "release_date": release_date,
                "value": float(value),
            }
        )
    frame = pd.DataFrame(rows, columns=["observation_date", "release_date", "value"])
    if not frame.empty:
        frame = frame.sort_values(
            ["release_date", "observation_date"], kind="mergesort"
        ).drop_duplicates("release_date", keep="last")
    _store_raw(raw_dir, f"alfred_{dataset_id}_{series_id}.json", payload)
    receipt = DownloadReceipt(
        dataset_id=dataset_id,
        url_template=FRED_API_OBSERVATIONS,
        sha256=_sha256(payload),
        byte_count=len(payload),
        minimum_date=(frame["observation_date"].min().date().isoformat() if len(frame) else None),
        maximum_date=(frame["observation_date"].max().date().isoformat() if len(frame) else None),
        status="downloaded_initial_releases_only",
    )
    return frame, receipt


def load_cboe_vxo_history(
    path: Path,
    start: Any,
    end: Any,
    *,
    split: str,
) -> tuple[pd.Series, DownloadReceipt]:
    """Load a phase-bounded official Cboe VXO capture without opening later tiers."""

    start_date, end_date = _bounded_dates(start, end, split=split)
    source = Path(path).resolve()
    payload = source.read_bytes()
    frame = _parse_csv(payload)
    required_columns = {"DATE", "CLOSE"}
    if not required_columns.issubset(frame.columns):
        raise DataGateError("CBOE_VXO_SCHEMA_MISMATCH")
    frame = _assert_response_date_bound(
        frame,
        date_column="DATE",
        start=start_date,
        end=end_date,
        label=f"cboe_vxo_{split}",
    )
    values = pd.to_numeric(frame["CLOSE"], errors="coerce")
    if values.isna().any() or (values <= 0).any():
        raise DataGateError("CBOE_VXO_INVALID_CLOSE")
    series = pd.Series(
        values.to_numpy(dtype=float),
        index=pd.DatetimeIndex(frame["DATE"]),
        name="VXO",
    ).sort_index(kind="mergesort")
    if series.index.has_duplicates:
        raise DataGateError("CBOE_VXO_DUPLICATE_DATE")
    receipt = DownloadReceipt(
        dataset_id="DS005",
        url_template=CBOE_VXO_HISTORY,
        sha256=_sha256(payload),
        byte_count=len(payload),
        minimum_date=series.index.min().date().isoformat() if len(series) else None,
        maximum_date=series.index.max().date().isoformat() if len(series) else None,
        status="loaded_phase_bounded_official_cboe_history",
        reason=f"frozen_source_file={source.name}",
    )
    return series, receipt


def load_state_street_distributions(
    path: Path,
    start: Any,
    end: Any,
    *,
    split: str,
) -> tuple[pd.DataFrame, DownloadReceipt]:
    """Load a pre-frozen official sponsor export without touching current data."""

    start_date, end_date = _bounded_dates(start, end, split=split)
    source = Path(path).resolve()
    if not source.is_file():
        raise DataGateError("STATE_STREET_DISTRIBUTION_SNAPSHOT_REQUIRED")
    payload = source.read_bytes()
    frame = _parse_csv(payload)
    required = {"ex_date", "distribution"}
    if not required.issubset(frame.columns):
        raise DataGateError("STATE_STREET_DISTRIBUTION_SCHEMA_MISMATCH")
    dates = pd.to_datetime(frame["ex_date"], errors="coerce").dt.normalize()
    values = pd.to_numeric(frame["distribution"], errors="coerce")
    if dates.isna().any() or values.isna().any() or (values < 0).any():
        raise DataGateError("STATE_STREET_DISTRIBUTION_INVALID_VALUES")
    phase_limit = TRAIN_END if split == "train" else VALIDATION_END
    if len(dates) and dates.max() > phase_limit:
        raise DataGateError("STATE_STREET_SNAPSHOT_EXCEEDS_PHASE_BOUNDARY")
    if len(dates) and dates.max() >= LOCKED_START:
        raise LockedBoundaryError("TECHNICAL_FAILURE_LOCKED_BREACH:sponsor_snapshot")
    bounded = pd.DataFrame({"date": dates, "distribution": values})
    bounded = bounded.loc[(bounded["date"] >= start_date) & (bounded["date"] <= end_date)]
    bounded = bounded.sort_values("date", kind="mergesort").reset_index(drop=True)
    if bounded["date"].duplicated().any():
        raise DataGateError("STATE_STREET_DISTRIBUTION_DUPLICATE_EX_DATE")
    receipt = DownloadReceipt(
        dataset_id="DS001",
        url_template="PINNED_STATE_STREET_SPY_DISTRIBUTION_EXPORT",
        sha256=_sha256(payload),
        byte_count=len(payload),
        minimum_date=bounded["date"].min().date().isoformat() if len(bounded) else None,
        maximum_date=bounded["date"].max().date().isoformat() if len(bounded) else None,
        status="loaded_official_frozen_snapshot",
    )
    return bounded, receipt


def load_sec_distribution_totals(
    path: Path,
    start: Any,
    end: Any,
    *,
    split: str,
) -> tuple[pd.DataFrame, DownloadReceipt]:
    """Load audited SPY distribution totals for explicitly bounded fiscal periods."""

    start_date, end_date = _bounded_dates(start, end, split=split)
    source = Path(path).resolve()
    if not source.is_file():
        raise DataGateError("SEC_DISTRIBUTION_TOTALS_SNAPSHOT_REQUIRED")
    payload = source.read_bytes()
    frame = _parse_csv(payload)
    required = {"period_start", "period_end", "distribution_total"}
    if not required.issubset(frame.columns):
        raise DataGateError("SEC_DISTRIBUTION_TOTALS_SCHEMA_MISMATCH")
    starts = pd.to_datetime(frame["period_start"], errors="coerce").dt.normalize()
    ends = pd.to_datetime(frame["period_end"], errors="coerce").dt.normalize()
    totals = pd.to_numeric(frame["distribution_total"], errors="coerce")
    if (
        starts.isna().any()
        or ends.isna().any()
        or totals.isna().any()
        or (starts > ends).any()
        or (totals < 0).any()
    ):
        raise DataGateError("SEC_DISTRIBUTION_TOTALS_INVALID_VALUES")
    phase_limit = TRAIN_END if split == "train" else VALIDATION_END
    if len(ends) and ends.max() > phase_limit:
        raise DataGateError("SEC_DISTRIBUTION_TOTALS_EXCEED_PHASE_BOUNDARY")
    if len(ends) and ends.max() >= LOCKED_START:
        raise LockedBoundaryError("TECHNICAL_FAILURE_LOCKED_BREACH:sec_distribution_totals")
    bounded = frame.copy()
    bounded["period_start"] = starts
    bounded["period_end"] = ends
    bounded["distribution_total"] = totals
    if "event_sum_tolerance" in bounded.columns:
        event_sum_tolerance = pd.to_numeric(
            bounded["event_sum_tolerance"], errors="coerce"
        ).fillna(SEC_FISCAL_EVENT_SUM_TOLERANCE)
    else:
        event_sum_tolerance = pd.Series(
            SEC_FISCAL_EVENT_SUM_TOLERANCE,
            index=bounded.index,
            dtype=float,
        )
    if (
        (event_sum_tolerance < SEC_FISCAL_EVENT_SUM_TOLERANCE).any()
        or (event_sum_tolerance > SEC_FISCAL_EVENT_SUM_TOLERANCE_MAX).any()
    ):
        raise DataGateError("SEC_DISTRIBUTION_TOTALS_INVALID_EVENT_SUM_TOLERANCE")
    bounded["event_sum_tolerance"] = event_sum_tolerance
    bounded = (
        bounded.loc[(bounded["period_end"] >= start_date) & (bounded["period_start"] <= end_date)]
        .sort_values("period_start", kind="mergesort")
        .reset_index(drop=True)
    )
    if bounded["period_start"].duplicated().any() or bounded["period_end"].duplicated().any():
        raise DataGateError("SEC_DISTRIBUTION_TOTALS_DUPLICATE_PERIOD")
    previous_end: pd.Timestamp | None = None
    for row in bounded.itertuples(index=False):
        if previous_end is not None and pd.Timestamp(row.period_start) <= previous_end:
            raise DataGateError("SEC_DISTRIBUTION_TOTALS_OVERLAPPING_PERIODS")
        previous_end = pd.Timestamp(row.period_end)
    receipt = DownloadReceipt(
        dataset_id="DS001_SEC_AUDITED_TOTALS",
        url_template="PINNED_SEC_AUDITED_SPY_DISTRIBUTION_TOTALS",
        sha256=_sha256(payload),
        byte_count=len(payload),
        minimum_date=(bounded["period_start"].min().date().isoformat() if len(bounded) else None),
        maximum_date=(bounded["period_end"].max().date().isoformat() if len(bounded) else None),
        status="loaded_official_frozen_audited_totals",
    )
    return bounded, receipt


def reconcile_sponsor_distributions(
    sponsor: pd.DataFrame,
    yahoo: pd.DataFrame,
    *,
    absolute_tolerance: float = SPONSOR_EVENT_AMOUNT_TOLERANCE,
) -> Mapping[str, Any]:
    left = sponsor.set_index("date")["distribution"].sort_index()
    right = yahoo.set_index("date")["distribution"].sort_index()
    if not left.index.equals(right.index):
        missing_sponsor = right.index.difference(left.index)
        missing_yahoo = left.index.difference(right.index)
        raise DataGateError(
            "SPONSOR_DISTRIBUTION_DATE_MISMATCH:"
            f"missing_sponsor={len(missing_sponsor)}:missing_yahoo={len(missing_yahoo)}"
        )
    differences = (left - right).abs()
    if len(differences) and bool((differences > absolute_tolerance).any()):
        raise DataGateError("SPONSOR_DISTRIBUTION_AMOUNT_MISMATCH")
    return {
        "event_count": int(len(left)),
        "maximum_absolute_amount_difference": (
            float(differences.max()) if len(differences) else 0.0
        ),
        "absolute_tolerance": absolute_tolerance,
    }


def reconcile_official_distribution_audit(
    exact_events: pd.DataFrame,
    fiscal_totals: pd.DataFrame,
    yahoo: pd.DataFrame,
    *,
    exact_tolerance: float = SPONSOR_EVENT_AMOUNT_TOLERANCE,
    rounded_total_tolerance: float = SEC_FISCAL_EVENT_SUM_TOLERANCE,
) -> Mapping[str, Any]:
    """Verify operational Yahoo events against official event and audited totals."""

    operational = yahoo.loc[:, ["date", "distribution"]].copy()
    operational["date"] = pd.to_datetime(operational["date"]).dt.normalize()
    operational["distribution"] = pd.to_numeric(operational["distribution"], errors="coerce")
    operational = operational.sort_values("date", kind="mergesort").reset_index(drop=True)
    if (
        operational["date"].isna().any()
        or operational["distribution"].isna().any()
        or operational["date"].duplicated().any()
    ):
        raise DataGateError("OPERATIONAL_DISTRIBUTION_EVENTS_INVALID")
    if operational.empty:
        raise DataGateError("OPERATIONAL_DISTRIBUTION_EVENTS_EMPTY")

    exact_audit: Mapping[str, Any]
    if exact_events.empty:
        exact_audit = {
            "event_count": 0,
            "maximum_absolute_amount_difference": 0.0,
            "absolute_tolerance": exact_tolerance,
        }
    else:
        exact_start = pd.Timestamp(exact_events["date"].min()).normalize()
        exact_end = pd.Timestamp(exact_events["date"].max()).normalize()
        operational_exact_window = operational.loc[
            operational["date"].between(exact_start, exact_end)
        ]
        exact_audit = reconcile_sponsor_distributions(
            exact_events,
            operational_exact_window,
            absolute_tolerance=exact_tolerance,
        )

    period_rows: list[dict[str, Any]] = []
    covered_dates: set[pd.Timestamp] = set(exact_events["date"])
    for row in fiscal_totals.itertuples(index=False):
        period_start = pd.Timestamp(row.period_start).normalize()
        period_end = pd.Timestamp(row.period_end).normalize()
        in_period = operational.loc[operational["date"].between(period_start, period_end)]
        observed = float(in_period["distribution"].sum())
        expected = float(row.distribution_total)
        difference = abs(observed - expected)
        period_tolerance = float(
            getattr(row, "event_sum_tolerance", rounded_total_tolerance)
        )
        if not rounded_total_tolerance <= period_tolerance <= SEC_FISCAL_EVENT_SUM_TOLERANCE_MAX:
            raise DataGateError("SEC_DISTRIBUTION_TOTALS_INVALID_EVENT_SUM_TOLERANCE")
        if difference > period_tolerance:
            raise DataGateError(
                "SEC_DISTRIBUTION_FISCAL_TOTAL_MISMATCH:"
                f"{period_start.date()}:{period_end.date()}:"
                f"observed={observed:.8f}:expected={expected:.8f}"
            )
        covered_dates.update(pd.Timestamp(value) for value in in_period["date"])
        period_rows.append(
            {
                "period_start": period_start.date().isoformat(),
                "period_end": period_end.date().isoformat(),
                "event_count": int(len(in_period)),
                "observed_total": observed,
                "audited_total": expected,
                "absolute_difference": difference,
                "event_sum_tolerance": period_tolerance,
            }
        )

    uncovered = [
        date.date().isoformat()
        for date in operational["date"]
        if pd.Timestamp(date) not in covered_dates
    ]
    if uncovered:
        raise DataGateError(
            "OPERATIONAL_DISTRIBUTION_EVENT_WITHOUT_OFFICIAL_COVERAGE:" + ",".join(uncovered[:10])
        )
    return {
        "operational_source": "Yahoo Finance bounded event endpoint",
        "official_verification_sources": [
            "State Street exact distribution events",
            "SEC audited fiscal distribution totals",
        ],
        "operational_event_count": int(len(operational)),
        "exact_event_audit": exact_audit,
        "fiscal_period_audit": period_rows,
        "uncovered_event_count": 0,
        "exact_tolerance": exact_tolerance,
        "rounded_total_tolerance": rounded_total_tolerance,
    }


FRED_DATASETS: Mapping[str, tuple[str, str]] = {
    "DS004": ("VIXCLS", "VIX"),
    "DS016": ("DGS10", "DGS10"),
    "DS017": ("DGS2", "DGS2"),
    "DS018": ("DGS3MO", "DGS3MO"),
    "DS019": ("T10Y2Y", "T10Y2Y"),
    "DS020": ("T10Y3M", "T10Y3M"),
    "DS021": ("DFF", "DFF"),
    "DS022": ("BAA10YM", "BAA10Y"),
    "DS023": ("BAMLC0A0CM", "IG_OAS"),
    "DS024": ("BAMLH0A0HYM2", "HY_OAS"),
    "DS025": ("NFCI", "NFCI"),
    "DS026": ("ANFCI", "ANFCI"),
    "DS027": ("STLFSI4", "STLFSI"),
    "DS028": ("OFRFSI", "OFR_FSI"),
    "DS033": ("CPIAUCSL", "CPI"),
    "DS034": ("PCEPI", "PCE"),
    "DS038": ("WALCL", "WALCL"),
    "DS039": ("M2SL", "M2"),
    "DS040": ("T10YIE", "T10YIE"),
    "DS045": ("USEPUINDXD", "EPU"),
    "DS050": ("DTWEXBGS", "USD"),
    "DS051": ("DCOILWTICO", "WTI"),
    "DS052": ("GOLDAMGBD228NLBM", "GOLD"),
}

FRED_FIRST_DISSEMINATION: Mapping[str, pd.Timestamp] = {
    "DS004": pd.Timestamp("2003-09-22"),
}


def _align_causal(
    series: pd.Series, sessions: pd.DatetimeIndex, *, lag_sessions: int = 1
) -> pd.Series:
    values = series.copy().sort_index(kind="mergesort")
    values.index = pd.DatetimeIndex(values.index).normalize()
    aligned = values.reindex(sessions, method="ffill")
    if lag_sessions:
        aligned = aligned.shift(lag_sessions)
    return aligned


def _align_initial_releases(
    releases: pd.DataFrame,
    sessions: pd.DatetimeIndex,
    *,
    first_dissemination: pd.Timestamp | None = None,
) -> pd.Series:
    if releases.empty:
        return pd.Series(np.nan, index=sessions, dtype=float)
    values = releases.set_index("release_date")["value"].sort_index(kind="mergesort")
    aligned = values.reindex(sessions, method="ffill").shift(1)
    if first_dissemination is not None:
        aligned.loc[aligned.index < first_dissemination] = np.nan
    return aligned


def _reconcile_spy_sources(
    yahoo_prices: pd.DataFrame,
    stooq: pd.DataFrame,
    distributions: pd.DataFrame,
    splits: pd.DataFrame,
    *,
    minimum_overlap: int = 1000,
    close_consensus_dates: Iterable[Any] | None = None,
) -> Mapping[str, Any]:
    yahoo_prices = yahoo_prices.set_index("date").sort_index(kind="mergesort")
    comparison = stooq.set_index("date").sort_index(kind="mergesort")
    common = yahoo_prices.index.intersection(comparison.index)
    if len(common) < minimum_overlap:
        raise DataGateError("SPY_RECONCILIATION_TOO_SHORT")

    event_dates = set(pd.to_datetime(distributions.get("date", pd.Series(dtype="datetime64[ns]"))))
    event_dates.update(pd.to_datetime(splits.get("date", pd.Series(dtype="datetime64[ns]"))))
    if any(date not in common for date in event_dates):
        raise DataGateError("SPY_CORPORATE_ACTION_SESSION_MISSING_FROM_SOURCE_OVERLAP")

    bounded_distributions = distributions.loc[
        pd.to_datetime(distributions.get("date", pd.Series(dtype="datetime64[ns]"))).isin(common)
    ]
    bounded_splits = splits.loc[
        pd.to_datetime(splits.get("date", pd.Series(dtype="datetime64[ns]"))).isin(common)
    ]
    yahoo_ledger, _ = build_total_return_ledger(
        yahoo_prices.loc[common].reset_index(names="date"),
        bounded_distributions,
        bounded_splits,
    )
    stooq_ledger, _ = build_total_return_ledger(
        comparison.loc[common].reset_index(names="date"),
        bounded_distributions,
        bounded_splits,
    )
    yahoo_returns = yahoo_ledger["long_return"]
    stooq_returns = stooq_ledger["long_return"]
    valid = yahoo_returns.notna() & stooq_returns.notna()
    yahoo_returns = yahoo_returns.loc[valid]
    stooq_returns = stooq_returns.loc[valid]
    differences = (yahoo_returns - stooq_returns).abs()
    within = differences <= SPY_RETURN_TOLERANCE
    raw_within_fraction = float(within.mean()) if len(within) else 0.0
    outlier_dates = differences.index[~within]
    median_abs_difference = float(differences.median())

    level_differences = pd.DataFrame(index=common)
    for column in ("open", "high", "low", "close"):
        denominator = yahoo_prices.loc[common, column].abs().replace(0.0, np.nan)
        level_differences[column] = (
            (yahoo_prices.loc[common, column] - comparison.loc[common, column]).abs()
            / denominator
        )
    isolated_open_dates = set(
        level_differences.index[
            (level_differences[["high", "low", "close"]] <= SPY_RETURN_TOLERANCE).all(axis=1)
            & (level_differences["open"] > SPY_RETURN_TOLERANCE)
            & (comparison.loc[common, "open"] <= yahoo_prices.loc[common, "high"])
            & (comparison.loc[common, "open"] >= yahoo_prices.loc[common, "low"])
        ]
    )
    reconciled_execution_dates: list[pd.Timestamp] = []
    unreconciled: list[pd.Timestamp] = []
    for date in outlier_dates:
        location = common.get_loc(date)
        following = common[location + 1] if location + 1 < len(common) else None
        endpoints = [endpoint for endpoint in (date, following) if endpoint is not None]
        bounded_level_difference = all(
            bool(level_differences.loc[endpoint, "open"] <= SPY_RETURN_TOLERANCE)
            or endpoint in isolated_open_dates
            for endpoint in endpoints
        )
        if bounded_level_difference:
            reconciled_execution_dates.append(pd.Timestamp(date))
        else:
            unreconciled.append(pd.Timestamp(date))
    reconciled_mask = within.copy()
    reconciled_mask.loc[reconciled_execution_dates] = True
    within_fraction = float(reconciled_mask.mean()) if len(reconciled_mask) else 0.0
    clean = within
    correlation = float(yahoo_returns.loc[clean].corr(stooq_returns.loc[clean]))
    raw_correlation = float(yahoo_returns.corr(stooq_returns))
    if within_fraction < SPY_REQUIRED_TOLERANCE_FRACTION or unreconciled:
        unreconciled_details = ",".join(
            f"{pd.Timestamp(date).date().isoformat()}="
            f"{differences.loc[date]:.9f}/"
            f"yahoo={yahoo_returns.loc[date]:.9f}/"
            f"stooq={stooq_returns.loc[date]:.9f}"
            for date in unreconciled
        )
        raise DataGateError(
            "SPY_RECONCILIATION_99_5_PERCENT_GATE_FAILED:"
            "basis=open_to_open_total_return:"
            f"within_fraction={within_fraction:.9f}:"
            f"raw_within_fraction={raw_within_fraction:.9f}:"
            f"outliers={len(outlier_dates)}:"
            f"reconciled={len(reconciled_execution_dates)}:"
            f"unreconciled={len(unreconciled)}:"
            f"correlation={correlation:.9f}:"
            f"raw_correlation={raw_correlation:.9f}:"
            f"median_abs_difference={median_abs_difference:.12f}:"
            f"unreconciled_details={unreconciled_details}"
        )
    if correlation < 0.999:
        raise DataGateError("SPY_RECONCILIATION_FAILED")

    yahoo_close_returns = yahoo_prices.loc[common, "close"].pct_change()
    stooq_close_returns = comparison.loc[common, "close"].pct_change()
    close_valid = yahoo_close_returns.notna() & stooq_close_returns.notna()
    yahoo_close_returns = yahoo_close_returns.loc[close_valid]
    stooq_close_returns = stooq_close_returns.loc[close_valid]
    close_differences = (yahoo_close_returns - stooq_close_returns).abs()
    close_outlier_dates = list(close_differences.index[close_differences > SPY_RETURN_TOLERANCE])

    if close_consensus_dates is None:
        close_only_dates = set(
            level_differences.index[
                (level_differences[["open", "high", "low"]] <= SPY_RETURN_TOLERANCE).all(
                    axis=1
                )
                & (level_differences["close"] > SPY_RETURN_TOLERANCE)
                & (comparison.loc[common, "close"] <= yahoo_prices.loc[common, "high"])
                & (comparison.loc[common, "close"] >= yahoo_prices.loc[common, "low"])
            ]
        )
    else:
        close_only_dates = set(pd.to_datetime(list(close_consensus_dates)))
        if any(date not in common for date in close_only_dates):
            raise DataGateError("SPY_CLOSE_CONSENSUS_DATE_OUTSIDE_OVERLAP")
    close_unreconciled: list[pd.Timestamp] = []
    for date in close_outlier_dates:
        location = common.get_loc(date)
        previous = common[location - 1] if location > 0 else None
        endpoints = [endpoint for endpoint in (previous, date) if endpoint is not None]
        bounded_level_difference = all(
            bool(level_differences.loc[endpoint, "close"] <= SPY_RETURN_TOLERANCE)
            or endpoint in close_only_dates
            for endpoint in endpoints
        )
        if date not in event_dates and not bounded_level_difference:
            close_unreconciled.append(pd.Timestamp(date))
    if close_unreconciled:
        details = ",".join(date.date().isoformat() for date in close_unreconciled)
        raise DataGateError(f"SPY_UNRECONCILED_CLOSE_RETURN_OUTLIERS:{details}")

    return {
        "overlap_rows": len(common),
        "daily_return_correlation": correlation,
        "raw_daily_return_correlation": raw_correlation,
        "median_abs_return_difference": median_abs_difference,
        "within_5_bps_fraction": within_fraction,
        "raw_within_5_bps_fraction": raw_within_fraction,
        "outlier_count": int(len(outlier_dates)),
        "reconciled_outlier_count": len(reconciled_execution_dates),
        "unreconciled_outlier_count": len(unreconciled),
        "return_tolerance": SPY_RETURN_TOLERANCE,
        "required_tolerance_fraction": SPY_REQUIRED_TOLERANCE_FRACTION,
        "comparison_basis": "open_to_open_total_return",
        "canonical_price_source": "stooq_raw_ohlcv",
        "independent_reconciliation_source": "yahoo_raw_ohlcv",
        "isolated_yahoo_open_discrepancy_dates": [
            date.date().isoformat() for date in sorted(isolated_open_dates)
        ],
        "close_return_within_5_bps_fraction": float(
            (close_differences <= SPY_RETURN_TOLERANCE).mean()
        ),
        "close_return_outlier_count": len(close_outlier_dates),
        "close_only_vendor_discrepancy_dates": [
            date.date().isoformat() for date in sorted(close_only_dates)
        ],
        "close_return_unreconciled_outlier_count": 0,
        "field_level_difference_diagnostics": {
            column: {
                "median_relative_difference": float(level_differences[column].median()),
                "maximum_relative_difference": float(level_differences[column].max()),
                "over_5_bps_count": int(
                    (level_differences[column] > SPY_RETURN_TOLERANCE).sum()
                ),
            }
            for column in ("open", "high", "low", "close")
        },
    }


def prepare_market_snapshot(
    root: Path,
    package: CampaignPackage,
    *,
    start: str,
    end: str,
    split: str,
    allow_diagnostic_yahoo_distributions: bool = False,
    skip_independent_price_sources: bool = False,
) -> Mapping[str, Any]:
    """Acquire one immutable bounded snapshot on GitHub Actions.

    The benchmark-only fast path keeps the bounded Yahoo series and official
    distribution audit but does not wait for optional Stooq/Kibot adjudication.
    Production callers retain the full multi-source path by default.
    """

    require_github_only_execution("SP500_LONG_SHORT_DAILY_PREPARE")
    start_date, end_date = _bounded_dates(start, end, split=split)
    if allow_diagnostic_yahoo_distributions and split != "validation":
        raise DataGateError("DIAGNOSTIC_YAHOO_DISTRIBUTIONS_VALIDATION_ONLY")
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    raw_root = root / "raw"
    raw_root.mkdir(exist_ok=True)
    client = requests.Session()
    print(f"[sp500-data] yahoo start={start_date.date()} end={end_date.date()}", flush=True)
    prices, yahoo_dividends, splits, yahoo_receipts = download_yahoo_history(
        "SPY",
        start_date,
        end_date,
        split=split,
        session=client,
        raw_dir=raw_root,
    )
    if not splits.empty:
        raise DataGateError("SPY_SPLIT_REQUIRES_EXPLICIT_RAW_PRICE_REPAIR")
    print(f"[sp500-data] yahoo complete rows={len(prices)}", flush=True)
    if split == "train":
        exact_path = Path(
            os.environ.get("SP500_STATE_STREET_DISTRIBUTIONS_CSV", "").strip()
            or _repo_campaign_root()
            / "official_inputs"
            / "state_street_spy_distribution_events_2006_2010.csv"
        )
        totals_path = Path(
            os.environ.get("SP500_SEC_DISTRIBUTION_TOTALS_CSV", "").strip()
            or _repo_campaign_root()
            / "official_inputs"
            / "sec_spy_distribution_fiscal_totals_1993_2009.csv"
        )
        exact_events, sponsor_receipt = load_state_street_distributions(
            exact_path, start_date, end_date, split=split
        )
        fiscal_totals, totals_receipt = load_sec_distribution_totals(
            totals_path, start_date, end_date, split=split
        )
        sponsor_reconciliation = reconcile_official_distribution_audit(
            exact_events, fiscal_totals, yahoo_dividends
        )
        print("[sp500-data] official distribution audit complete", flush=True)
        dividends = yahoo_dividends.copy()
        _store_raw(raw_root, exact_path.name, exact_path.read_bytes())
        _store_raw(raw_root, totals_path.name, totals_path.read_bytes())
        distribution_receipts = [sponsor_receipt, totals_receipt]
    elif allow_diagnostic_yahoo_distributions:
        dividends = yahoo_dividends.copy()
        sponsor_reconciliation = {
            "diagnostic_only": True,
            "official_sponsor_snapshot_used": False,
            "source": "bounded_yahoo_chart_events",
            "minimum_date": (
                dividends["date"].min().date().isoformat() if len(dividends) else None
            ),
            "maximum_date": (
                dividends["date"].max().date().isoformat() if len(dividends) else None
            ),
            "event_count": len(dividends),
        }
        distribution_receipts = [
            replace(
                yahoo_receipts[0],
                dataset_id="DS001",
                status="diagnostic_bounded_yahoo_distributions_not_official_sponsor",
                reason="owner_authorized_diagnostic_validation_only;promotion_eligible=false",
            )
        ]
    else:
        sponsor_path = Path(
            os.environ.get("SP500_STATE_STREET_DISTRIBUTIONS_CSV", "").strip()
            or _repo_campaign_root()
            / "official_inputs"
            / "state_street_spy_distributions_2011_2020.csv"
        )
        dividends, sponsor_receipt = load_state_street_distributions(
            sponsor_path, start_date, end_date, split=split
        )
        sponsor_reconciliation = reconcile_sponsor_distributions(dividends, yahoo_dividends)
        _store_raw(
            raw_root,
            f"state_street_spy_distributions_{split}.csv",
            sponsor_path.read_bytes(),
        )
        distribution_receipts = [sponsor_receipt]
    price_adjudication: Mapping[str, Any]
    reconciliation: Mapping[str, Any]
    if skip_independent_price_sources:
        # Explicit benchmark-only fast path. The bounded Yahoo series and
        # official distribution audit remain in force; only optional provider
        # adjudication is omitted. Production callers use the full path.
        stooq = prices.loc[:, ["date", "open", "high", "low", "close", "volume"]].copy()
        price_adjudication = {
            "mode": "benchmark_primary_yahoo_without_independent_adjudication",
            "changed_open_count": 0,
            "changed_close_count": 0,
        }
        stooq_receipt = replace(
            yahoo_receipts[0],
            dataset_id="DS002",
            status="benchmark_yahoo_primary_raw_ohlcv",
            reason="bounded_method_benchmark_fast_path;stooq_kibot_not_requested",
        )
        kibot_receipt = DownloadReceipt(
            dataset_id="KIBOT_SPY_UNADJUSTED_ADJUDICATOR",
            url_template=KIBOT_API_ENDPOINT,
            sha256="",
            byte_count=0,
            minimum_date=None,
            maximum_date=None,
            status="not_requested_benchmark_fast_path",
            reason="optional_independent_price_adjudication_skipped",
        )
        reconciliation = {
            "canonical_price_source": "yahoo_raw_ohlcv_bounded_benchmark",
            "independent_reconciliation_sources": [],
            "independent_price_adjudication_skipped": True,
            "price_adjudication": price_adjudication,
        }
    else:
        print("[sp500-data] stooq history start", flush=True)
        stooq, stooq_receipt = download_stooq_history(
            "spy.us",
            start_date,
            end_date,
            split=split,
            session=client,
            raw_dir=raw_root,
        )
        print(f"[sp500-data] stooq history complete rows={len(stooq)}", flush=True)
        print("[sp500-data] kibot adjudication history start", flush=True)
        kibot, kibot_receipt = download_kibot_unadjusted_history(
            "SPY",
            start_date,
            end_date,
            split=split,
            session=client,
            raw_dir=raw_root,
        )
        stooq, price_adjudication = _adjudicate_stooq_open_prices(prices, stooq, kibot)
        _store_raw(
            raw_root,
            "spy_price_adjudication.json",
            json.dumps(price_adjudication, sort_keys=True, separators=(",", ":")).encode("utf-8"),
        )
        print(
            "[sp500-data] kibot adjudication complete "
            f"changed_opens={price_adjudication['changed_open_count']} "
            f"changed_closes={price_adjudication['changed_close_count']}",
            flush=True,
        )
        reconciliation = _reconcile_spy_sources(
            prices,
            stooq,
            dividends,
            splits,
            minimum_overlap=min(1000, max(200, len(prices) - 1)),
            close_consensus_dates=(
                price_adjudication["fields"]["close"]["yahoo_supported_repair_dates"]
                + price_adjudication["fields"]["close"]["kibot_bridge_repair_dates"]
                + price_adjudication["fields"]["close"]["three_source_median_repair_dates"]
                + price_adjudication["fields"]["close"][
                    "primary_volume_supported_repair_dates"
                ]
                + price_adjudication["fields"]["close"]["retained_stooq_dates"]
            ),
        )
        stooq_fallback_used = "fallback" in stooq_receipt.status
        yahoo_fallback_used = stooq_fallback_used and "yahoo" in (stooq_receipt.reason or "")
        reconciliation = {
            **reconciliation,
            "canonical_price_source": (
                "yahoo_raw_ohlcv_fallback_with_kibot_reconciliation"
                if yahoo_fallback_used
                else "stooq_raw_ohlcv_with_kibot_adjudicated_open_and_close"
            ),
            "independent_reconciliation_sources": (
                ["kibot_unadjusted_daily_ohlcv"]
                if yahoo_fallback_used
                else ["yahoo_raw_ohlcv", "kibot_unadjusted_daily_ohlcv"]
            ),
            "stooq_provider_outage_fallback_used": stooq_fallback_used,
            "stooq_provider_outage_fallback_source": (
                "yahoo_raw_ohlcv" if yahoo_fallback_used else None
            ),
            "price_adjudication": price_adjudication,
        }
    ledger, audit = build_total_return_ledger(stooq, dividends, splits)

    receipts: list[DownloadReceipt] = [
        *yahoo_receipts,
        *distribution_receipts,
        stooq_receipt,
        kibot_receipt,
    ]
    series: dict[str, pd.Series] = {}
    available = {"DS001", "DS002"}
    rejected: dict[str, str] = {}
    required = set(package.required_dataset_ids())
    static_rejections = {
        "DS009": "FIRST_DISSEMINATION_AFTER_TRAIN_END:VIX3M_2013_09_30",
        "DS011": "NO_BOUNDED_CAUSAL_VIX_FUTURES_ADAPTER",
        "DS071": "PROXY_ONLY_NOT_EXECUTION_GRADE",
    }
    for dataset_id, reason in static_rejections.items():
        if dataset_id in required:
            rejected[dataset_id] = reason
    if "DS005" in required:
        vxo_filename = (
            "cboe_vxo_daily_1993_2010.csv"
            if split == "train"
            else "cboe_vxo_daily_2011_2020.csv"
        )
        vxo_path = Path(
            os.environ.get("SP500_CBOE_VXO_HISTORY_CSV", "").strip()
            or _repo_campaign_root() / "official_inputs" / vxo_filename
        )
        try:
            print(f"[sp500-data] cboe DS005 start file={vxo_filename}", flush=True)
            downloaded, receipt = load_cboe_vxo_history(
                vxo_path,
                start_date,
                end_date,
                split=split,
            )
            aligned = _align_causal(downloaded, ledger.index, lag_sessions=1)
            if not aligned.notna().any():
                raise DataGateError("NO_CAUSAL_VALUES_IN_PHASE:DS005")
            series["VXO"] = aligned
            receipts.append(receipt)
            available.add("DS005")
            _store_raw(raw_root, vxo_path.name, vxo_path.read_bytes())
            print("[sp500-data] cboe DS005 complete", flush=True)
        except (DataGateError, OSError) as exc:
            rejected["DS005"] = str(exc)
            print(f"[sp500-data] cboe DS005 rejected={exc}", flush=True)
    for dataset_id, (fred_id, logical_name) in FRED_DATASETS.items():
        if dataset_id not in required:
            continue
        try:
            print(f"[sp500-data] alfred {dataset_id} start", flush=True)
            downloaded, receipt = download_alfred_initial_series(
                fred_id,
                dataset_id,
                start_date,
                end_date,
                split=split,
                session=client,
                raw_dir=raw_root,
            )
            first_dissemination = FRED_FIRST_DISSEMINATION.get(dataset_id)
            aligned = _align_initial_releases(
                downloaded,
                ledger.index,
                first_dissemination=first_dissemination,
            )
            if not aligned.notna().any():
                raise DataGateError(f"NO_CAUSAL_VALUES_IN_PHASE:{dataset_id}")
            series[logical_name] = aligned
            receipts.append(receipt)
            available.add(dataset_id)
            print(f"[sp500-data] alfred {dataset_id} complete", flush=True)
        except DataGateError as exc:
            rejected[dataset_id] = str(exc)
            print(f"[sp500-data] alfred {dataset_id} rejected={exc}", flush=True)

    for dataset_id in sorted(required - available - set(rejected)):
        rejected[dataset_id] = "NO_BOUNDED_CAUSAL_ADAPTER"

    prices_path = root / "spy_ledger.parquet"
    pq.write_table(
        pa.Table.from_pandas(ledger.reset_index(names="date"), preserve_index=False), prices_path
    )
    series_rows = []
    for name, values in sorted(series.items()):
        for date, value in values.items():
            series_rows.append({"date": date, "series": name, "value": value})
    series_path = root / "causal_series.parquet"
    pq.write_table(pa.Table.from_pylist(series_rows), series_path)
    receipt_payload = [receipt.__dict__ for receipt in receipts]
    (root / "raw_manifest.jsonl").write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in receipt_payload
        ),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "1",
        "split": split,
        "minimum_date": ledger.index.min().date().isoformat(),
        "maximum_date": ledger.index.max().date().isoformat(),
        "locked_opened": False,
        "available_dataset_ids": sorted(available),
        "rejected_datasets": dict(sorted(rejected.items())),
        "receipts": receipt_payload,
        "ledger_audit": audit.__dict__,
        "spy_reconciliation": reconciliation,
        "sponsor_distribution_reconciliation": sponsor_reconciliation,
        "candidate_pack_sha256": canonical_json_hash(list(package.candidates)),
    }
    manifest["snapshot_sha256"] = canonical_json_hash(manifest)
    (root / "market_data_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def load_market_snapshot(root: Path) -> PreparedMarketData:
    root = Path(root).resolve()
    manifest = json.loads((root / "market_data_manifest.json").read_text("utf-8"))
    if manifest.get("locked_opened") is not False:
        raise LockedBoundaryError("TECHNICAL_FAILURE_LOCKED_BREACH:snapshot_manifest")
    ledger = pq.read_table(root / "spy_ledger.parquet").to_pandas()
    ledger["date"] = pd.to_datetime(ledger["date"])
    ledger = ledger.set_index("date").sort_index(kind="mergesort")
    assert_frame_before_locked(ledger, label="loaded_spy_ledger")
    series_table = pq.read_table(root / "causal_series.parquet").to_pandas()
    series: dict[str, pd.Series] = {}
    if len(series_table):
        series_table["date"] = pd.to_datetime(series_table["date"])
        assert_frame_before_locked(series_table, label="loaded_causal_series")
        for name, group in series_table.groupby("series", sort=True):
            series[str(name)] = pd.Series(
                group["value"].to_numpy(dtype=float),
                index=pd.DatetimeIndex(group["date"]),
                name=str(name),
            ).sort_index(kind="mergesort")
    return PreparedMarketData(
        ledger=ledger,
        series=series,
        available_dataset_ids=frozenset(manifest["available_dataset_ids"]),
        rejected_datasets=manifest["rejected_datasets"],
        receipts=tuple(DownloadReceipt(**row) for row in manifest["receipts"]),
        split=str(manifest["split"]),
    )


def write_fixture_snapshot(
    root: Path,
    ledger: pd.DataFrame,
    *,
    split: str = "train",
    series: Mapping[str, pd.Series] | None = None,
    available_dataset_ids: Iterable[str] = ("DS001", "DS002"),
) -> None:
    """Write a deterministic test fixture without invoking acquisition."""

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    assert_frame_before_locked(ledger, label="fixture_ledger")
    pq.write_table(
        pa.Table.from_pandas(ledger.reset_index(names="date"), preserve_index=False),
        root / "spy_ledger.parquet",
    )
    rows = []
    for name, values in sorted((series or {}).items()):
        for date, value in values.items():
            rows.append({"date": date, "series": name, "value": float(value)})
    pq.write_table(
        pa.Table.from_pylist(
            rows,
            schema=pa.schema(
                [
                    pa.field("date", pa.timestamp("ns")),
                    pa.field("series", pa.string()),
                    pa.field("value", pa.float64()),
                ]
            ),
        ),
        root / "causal_series.parquet",
    )
    manifest = {
        "schema_version": "1",
        "split": split,
        "minimum_date": ledger.index.min().date().isoformat(),
        "maximum_date": ledger.index.max().date().isoformat(),
        "locked_opened": False,
        "available_dataset_ids": sorted(set(available_dataset_ids)),
        "rejected_datasets": {},
        "receipts": [],
        "snapshot_sha256": canonical_json_hash({"fixture": True, "rows": len(ledger)}),
    }
    (root / "market_data_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (root / "raw_manifest.jsonl").write_text("", encoding="utf-8")
