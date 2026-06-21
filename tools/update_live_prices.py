from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv("DATA_DIR", PACKAGE_ROOT / "data_live"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", PACKAGE_ROOT / "runtime_outputs"))
REPORT_PATH = OUTPUT_DIR / "LATEST_PRICE_UPDATE_REPORT.json"
DATA_REPORT_PATH = DATA_DIR / "LAST_PRICE_UPDATE_REPORT.json"
MANIFEST_PATH = DATA_DIR / "DATA_MANIFEST.json"
PROGRESS_PATH = OUTPUT_DIR / "LATEST_MARKET_UPDATE_PROGRESS.txt"
PROGRESS_JSON_PATH = OUTPUT_DIR / "LATEST_MARKET_UPDATE_PROGRESS.json"
PRICE_COLS = ["date", "symbol", "open", "high", "low", "close", "adj_close", "volume"]
PROGRESS_EVENTS: list[dict[str, Any]] = []


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def clean(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean(v) for v in obj]
    if isinstance(obj, tuple):
        return [clean(v) for v in obj]
    if isinstance(obj, pd.Timestamp):
        return obj.strftime("%Y-%m-%d")
    if pd.isna(obj) if not isinstance(obj, (dict, list, tuple)) else False:
        return None
    return obj


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(obj), ensure_ascii=False, indent=2), encoding="utf-8")


def progress(message: str, reset: bool = False, **fields: Any) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if reset:
        PROGRESS_EVENTS.clear()
        PROGRESS_PATH.write_text("", encoding="utf-8-sig")
    event = {"time": now_iso(), "message": message}
    event.update(fields)
    PROGRESS_EVENTS.append(event)
    line = f"[{event['time']}] {message}"
    print(line, flush=True)
    with PROGRESS_PATH.open("a", encoding="utf-8-sig") as fh:
        fh.write(line + "\n")
    write_json(PROGRESS_JSON_PATH, {"updated_at": now_iso(), "events": PROGRESS_EVENTS[-300:]})


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def is_rate_limited(text: str) -> bool:
    return "rate_limited" in text or "Too Many Requests" in text or "Rate limited" in text or "YFRateLimitError" in text


def safe_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    tmp.replace(path)


def normalize_price_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=PRICE_COLS)
    out = df.copy()
    out.columns = [str(c).strip().lower().replace(" ", "_") for c in out.columns]
    if "datetime" in out.columns and "date" not in out.columns:
        out = out.rename(columns={"datetime": "date"})
    if "adj_close" not in out.columns and "adjclose" in out.columns:
        out = out.rename(columns={"adjclose": "adj_close"})
    for col in PRICE_COLS:
        if col not in out.columns:
            out[col] = pd.NA
    out = out[PRICE_COLS].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.tz_localize(None)
    out["symbol"] = out["symbol"].astype(str).str.strip()
    for col in ["open", "high", "low", "close", "adj_close", "volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["date", "symbol"])
    return out


def yahoo_symbol(local_symbol: str, market: str) -> str:
    sym = str(local_symbol).strip()
    if market == "TW":
        return sym
    if sym.startswith("^"):
        return sym
    return sym.replace(".", "-")


def extract_yfinance_one(raw: pd.DataFrame, yf_symbol: str, local_symbol: str) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame(columns=PRICE_COLS)
    frame = raw.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        level0 = [str(x) for x in frame.columns.get_level_values(0).unique()]
        level1 = [str(x) for x in frame.columns.get_level_values(1).unique()]
        if yf_symbol in level0:
            frame = frame[yf_symbol].copy()
        elif local_symbol in level0:
            frame = frame[local_symbol].copy()
        elif yf_symbol in level1:
            frame = frame.xs(yf_symbol, level=1, axis=1).copy()
        elif local_symbol in level1:
            frame = frame.xs(local_symbol, level=1, axis=1).copy()
        else:
            return pd.DataFrame(columns=PRICE_COLS)
    frame = frame.reset_index()
    frame.columns = [str(c).strip().lower().replace(" ", "_") for c in frame.columns]
    if "adj_close" not in frame.columns and "adj_close_" in frame.columns:
        frame = frame.rename(columns={"adj_close_": "adj_close"})
    if "adj_close" not in frame.columns and "adjclose" in frame.columns:
        frame = frame.rename(columns={"adjclose": "adj_close"})
    if "date" not in frame.columns and "datetime" in frame.columns:
        frame = frame.rename(columns={"datetime": "date"})
    frame["symbol"] = local_symbol
    return normalize_price_frame(frame)


def download_chart_one(local_symbol: str, market: str, start: str, timeout: int) -> tuple[pd.DataFrame, str]:
    try:
        import requests
    except Exception as exc:
        return pd.DataFrame(columns=PRICE_COLS), f"import requests failed: {exc}"

    yf_sym = yahoo_symbol(local_symbol, market)
    period1 = int(pd.Timestamp(start).timestamp())
    period2 = int((pd.Timestamp(datetime.now().date()) + pd.Timedelta(days=3)).timestamp())
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yf_sym}"
    params = {
        "period1": period1,
        "period2": period2,
        "interval": "1d",
        "events": "history",
        "includeAdjustedClose": "true",
    }
    try:
        resp = requests.get(url, params=params, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
    except Exception as exc:
        return pd.DataFrame(columns=PRICE_COLS), str(exc)
    if resp.status_code == 429:
        return pd.DataFrame(columns=PRICE_COLS), "rate_limited: Yahoo chart returned 429"
    if resp.status_code >= 400:
        return pd.DataFrame(columns=PRICE_COLS), f"chart_http_{resp.status_code}"
    try:
        payload = resp.json()
        result = (payload.get("chart", {}).get("result") or [None])[0]
        error = payload.get("chart", {}).get("error")
    except Exception as exc:
        return pd.DataFrame(columns=PRICE_COLS), f"chart_json_error: {exc}"
    if error:
        return pd.DataFrame(columns=PRICE_COLS), f"chart_error: {error}"
    if not result:
        return pd.DataFrame(columns=PRICE_COLS), "chart_empty"
    timestamps = result.get("timestamp") or []
    quote = ((result.get("indicators") or {}).get("quote") or [{}])[0]
    adj = ((result.get("indicators") or {}).get("adjclose") or [{}])[0].get("adjclose") or []
    rows: list[dict[str, Any]] = []
    for i, ts in enumerate(timestamps):
        close = (quote.get("close") or [None] * len(timestamps))[i]
        if close is None or pd.isna(close):
            continue
        rows.append(
            {
                "date": pd.to_datetime(int(ts), unit="s", utc=True).tz_convert(None).normalize(),
                "symbol": local_symbol,
                "open": (quote.get("open") or [None] * len(timestamps))[i],
                "high": (quote.get("high") or [None] * len(timestamps))[i],
                "low": (quote.get("low") or [None] * len(timestamps))[i],
                "close": close,
                "adj_close": adj[i] if i < len(adj) else close,
                "volume": (quote.get("volume") or [None] * len(timestamps))[i],
            }
        )
    if not rows:
        return pd.DataFrame(columns=PRICE_COLS), "chart_no_price_rows"
    return normalize_price_frame(pd.DataFrame(rows)), ""


def download_chart_chunk(local_symbols: list[str], market: str, start: str, timeout: int) -> tuple[pd.DataFrame, str]:
    parts: list[pd.DataFrame] = []
    errors: list[str] = []
    for sym in local_symbols:
        data, err = download_chart_one(sym, market=market, start=start, timeout=timeout)
        if not data.empty:
            parts.append(data)
        elif err:
            errors.append(f"{sym}:{err}")
    if parts:
        return pd.concat(parts, ignore_index=True), ""
    return pd.DataFrame(columns=PRICE_COLS), "; ".join(errors[:5]) if errors else "chart_empty"


def download_chunk(local_symbols: list[str], market: str, start: str, timeout: int) -> tuple[pd.DataFrame, str]:
    try:
        import logging
        import yfinance as yf
    except Exception as exc:
        return pd.DataFrame(columns=PRICE_COLS), f"import yfinance failed: {exc}"

    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    mapping = {sym: yahoo_symbol(sym, market) for sym in local_symbols}
    tickers = list(dict.fromkeys(mapping.values()))
    captured = io.StringIO()
    try:
        with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
            raw = yf.download(
                tickers=tickers if len(tickers) > 1 else tickers[0],
                start=start,
                auto_adjust=False,
                progress=False,
                threads=True,
                group_by="ticker",
                timeout=timeout,
            )
    except Exception as exc:
        return pd.DataFrame(columns=PRICE_COLS), str(exc)

    parts = []
    for local, yf_sym in mapping.items():
        part = extract_yfinance_one(raw, yf_sym, local)
        if not part.empty:
            parts.append(part)
    if not parts:
        captured_text = captured.getvalue()
        max_chart_symbols = int(os.getenv("LIVE_UPDATE_CHART_FALLBACK_MAX_SYMBOLS", "10"))
        skip_rate_limit_chart = env_bool("LIVE_UPDATE_SKIP_CHART_ON_RATE_LIMIT", True)
        disable_chart = env_bool("LIVE_UPDATE_DISABLE_CHART_FALLBACK", False)
        rate_limited = is_rate_limited(captured_text)
        if disable_chart or (skip_rate_limit_chart and rate_limited) or (max_chart_symbols and len(local_symbols) > max_chart_symbols):
            reason = "rate_limited" if rate_limited else "empty"
            return pd.DataFrame(columns=PRICE_COLS), f"{reason}; chart_fallback=skipped"
        chart_data, chart_err = download_chart_chunk(local_symbols, market=market, start=start, timeout=timeout)
        if not chart_data.empty:
            return chart_data, ""
        if rate_limited:
            return pd.DataFrame(columns=PRICE_COLS), f"rate_limited: yfinance Too Many Requests; chart_fallback={chart_err}"
        return pd.DataFrame(columns=PRICE_COLS), f"empty; chart_fallback={chart_err}"
    return pd.concat(parts, ignore_index=True), ""


def remote_latest_date(market: str, timeout: int) -> str | None:
    defaults = {
        "US": "SPY,QQQ",
        "TW": "2330.TW,^TWII",
    }
    symbols = [
        item.strip()
        for item in os.getenv(f"LIVE_UPDATE_{market}_ANCHORS", defaults.get(market, "")).split(",")
        if item.strip()
    ]
    start = (datetime.now() - timedelta(days=int(os.getenv("LIVE_UPDATE_PREFLIGHT_LOOKBACK_DAYS", "14")))).strftime("%Y-%m-%d")
    latest: pd.Timestamp | None = None
    for symbol in symbols:
        data, _err = download_chart_one(
            symbol,
            market=market,
            start=start,
            timeout=min(timeout, int(os.getenv("LIVE_UPDATE_PREFLIGHT_TIMEOUT", "8"))),
        )
        if data.empty:
            continue
        ts = pd.to_datetime(data["date"], errors="coerce").max()
        if pd.isna(ts):
            continue
        latest = ts if latest is None else max(latest, ts)
    return latest.strftime("%Y-%m-%d") if latest is not None else None


def file_date_stats(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    df = pd.read_csv(path, usecols=["date", "symbol"], dtype={"symbol": str}, parse_dates=["date"], low_memory=False)
    return {
        "path": str(path),
        "exists": True,
        "rows": int(len(df)),
        "symbols": int(df["symbol"].nunique()),
        "min_date": str(df["date"].min().date()) if len(df) else None,
        "max_date": str(df["date"].max().date()) if len(df) else None,
    }


def update_price_file(
    path: Path,
    market: str,
    label: str,
    tail_rows: int,
    chunk_size: int,
    sleep_seconds: float,
    timeout: int,
    max_symbols: int = 0,
) -> dict[str, Any]:
    before = file_date_stats(path)
    progress(
        f"{label}: start update file; local max={before.get('max_date')} symbols={before.get('symbols')}",
        label=label,
        stage="start_file",
        local_max_date=before.get("max_date"),
        symbols=before.get("symbols"),
    )
    if not path.exists():
        progress(f"{label}: missing file, skipped", label=label, stage="missing_file")
        return {"label": label, "status": "missing_file", "path": str(path), "before": before}
    if env_bool("LIVE_UPDATE_PREFLIGHT", True):
        progress(f"{label}: preflight remote market date check", label=label, stage="preflight_start")
        remote_date = remote_latest_date(market, timeout=timeout)
        local_date = before.get("max_date")
        if remote_date and local_date and str(local_date) >= str(remote_date):
            progress(
                f"{label}: skipped; local date {local_date} already >= remote date {remote_date}",
                label=label,
                stage="preflight_skip",
                local_date=local_date,
                remote_date=remote_date,
            )
            return {
                "label": label,
                "status": "skipped_preflight_no_new_market_date",
                "path": str(path),
                "market": market,
                "before": before,
                "after": before,
                "remote_latest_date": remote_date,
                "attempted_symbols": 0,
                "chunks_attempted": 0,
                "new_rows": 0,
                "updated_symbols": 0,
                "error_count": 0,
                "errors_sample": [],
            }
        progress(
            f"{label}: preflight done; local={local_date} remote={remote_date or 'unknown'}, scanning symbols",
            label=label,
            stage="preflight_done",
            local_date=local_date,
            remote_date=remote_date,
        )

    old = pd.read_csv(path, dtype={"symbol": str}, parse_dates=["date"], low_memory=False)
    old = normalize_price_frame(old)
    if old.empty:
        progress(f"{label}: empty file, skipped", label=label, stage="empty_file")
        return {"label": label, "status": "empty_file", "path": str(path), "before": before}

    last_by_symbol = old.groupby("symbol")["date"].max().to_dict()
    symbols = sorted(last_by_symbol)
    if max_symbols and max_symbols > 0:
        symbols = symbols[:max_symbols]
    total_chunks = (len(symbols) + max(chunk_size, 1) - 1) // max(chunk_size, 1)
    progress(
        f"{label}: updating {len(symbols)} symbols in {total_chunks} chunks; chunk={chunk_size}, sleep={sleep_seconds}s, timeout={timeout}s",
        label=label,
        stage="chunks_start",
        attempted_symbols=len(symbols),
        total_chunks=total_chunks,
        chunk_size=chunk_size,
        sleep_seconds=sleep_seconds,
        timeout=timeout,
    )

    new_parts: list[pd.DataFrame] = []
    errors: list[dict[str, Any]] = []
    chunks_attempted = 0
    consecutive_errors = 0
    aborted_reason = ""
    started_clock = time.monotonic()
    max_runtime_seconds = int(os.getenv("LIVE_UPDATE_MAX_RUNTIME_SECONDS", "0"))
    consecutive_error_limit = int(os.getenv("LIVE_UPDATE_CONSECUTIVE_ERROR_LIMIT", "4"))
    abort_on_rate_limit = env_bool("LIVE_UPDATE_ABORT_ON_RATE_LIMIT", True)
    today = pd.Timestamp(datetime.now().date())

    for i in range(0, len(symbols), chunk_size):
        if max_runtime_seconds and (time.monotonic() - started_clock) > max_runtime_seconds:
            aborted_reason = "aborted_max_runtime_seconds"
            break
        chunk_symbols = symbols[i : i + chunk_size]
        starts = []
        for sym in chunk_symbols:
            last = pd.Timestamp(last_by_symbol[sym])
            if last.date() <= today.date():
                starts.append(last + timedelta(days=1))
        if not starts:
            continue
        start = min(starts).strftime("%Y-%m-%d")
        chunks_attempted += 1
        done_symbols = min(i + len(chunk_symbols), len(symbols))
        progress(
            f"{label}: chunk {chunks_attempted}/{total_chunks}, symbols {done_symbols}/{len(symbols)}, start={start}, sample={','.join(chunk_symbols[:5])}",
            label=label,
            stage="chunk_start",
            chunk=chunks_attempted,
            total_chunks=total_chunks,
            done_symbols=done_symbols,
            total_symbols=len(symbols),
            start=start,
            sample=chunk_symbols[:5],
        )
        data, err = download_chunk(chunk_symbols, market=market, start=start, timeout=timeout)
        if err and data.empty:
            errors.append({"chunk": chunks_attempted, "start": start, "symbols": chunk_symbols[:8], "error": err})
            consecutive_errors += 1
            progress(
                f"{label}: chunk {chunks_attempted}/{total_chunks} failed; consecutive_errors={consecutive_errors}; {str(err)[:180]}",
                label=label,
                stage="chunk_error",
                chunk=chunks_attempted,
                consecutive_errors=consecutive_errors,
                error=str(err)[:300],
            )
            if abort_on_rate_limit and is_rate_limited(str(err)):
                aborted_reason = "aborted_rate_limited"
                progress(f"{label}: aborting because source is rate limited", label=label, stage="abort", reason=aborted_reason)
                break
            if consecutive_error_limit and consecutive_errors >= consecutive_error_limit:
                aborted_reason = "aborted_consecutive_errors"
                progress(
                    f"{label}: aborting after {consecutive_errors} consecutive errors",
                    label=label,
                    stage="abort",
                    reason=aborted_reason,
                )
                break
        if not data.empty:
            consecutive_errors = 0
            keep_rows = []
            for sym, part in data.groupby("symbol", sort=False):
                last = pd.Timestamp(last_by_symbol.get(sym, pd.Timestamp("1900-01-01")))
                part = part[part["date"] > last].copy()
                if not part.empty:
                    keep_rows.append(part)
            if keep_rows:
                chunk_new = pd.concat(keep_rows, ignore_index=True)
                new_parts.append(chunk_new)
                progress(
                    f"{label}: chunk {chunks_attempted}/{total_chunks} added {len(chunk_new)} rows for {chunk_new['symbol'].nunique()} symbols",
                    label=label,
                    stage="chunk_success",
                    chunk=chunks_attempted,
                    new_rows=len(chunk_new),
                    updated_symbols=int(chunk_new["symbol"].nunique()),
                )
            else:
                progress(
                    f"{label}: chunk {chunks_attempted}/{total_chunks} returned data but no rows newer than local cache",
                    label=label,
                    stage="chunk_no_new_rows",
                    chunk=chunks_attempted,
                )
        if sleep_seconds > 0 and i + chunk_size < len(symbols):
            time.sleep(sleep_seconds)

    if new_parts:
        new_df = pd.concat(new_parts, ignore_index=True)
        if aborted_reason and not env_bool("LIVE_UPDATE_WRITE_PARTIAL_ON_ABORT", False):
            new_df = pd.DataFrame(columns=PRICE_COLS)
            combined = old.copy()
        else:
            combined = pd.concat([old, new_df], ignore_index=True)
    else:
        new_df = pd.DataFrame(columns=PRICE_COLS)
        combined = old.copy()

    combined = normalize_price_frame(combined)
    combined = combined.drop_duplicates(["date", "symbol"], keep="last").sort_values(["symbol", "date"])
    combined = combined.groupby("symbol", group_keys=False).tail(tail_rows).sort_values(["date", "symbol"])
    progress(f"{label}: writing updated cache file", label=label, stage="write_start")
    safe_csv(combined, path)

    after = file_date_stats(path)
    if aborted_reason:
        status = f"{aborted_reason}_partial_written" if len(new_df) else aborted_reason
    else:
        status = "updated" if len(new_df) else "no_new_rows"
    progress(
        f"{label}: done status={status}, new_rows={len(new_df)}, updated_symbols={int(new_df['symbol'].nunique()) if len(new_df) else 0}, after_max={after.get('max_date')}",
        label=label,
        stage="file_done",
        status=status,
        new_rows=int(len(new_df)),
        updated_symbols=int(new_df["symbol"].nunique()) if len(new_df) else 0,
        after_max_date=after.get("max_date"),
    )
    return {
        "label": label,
        "status": status,
        "path": str(path),
        "market": market,
        "before": before,
        "after": after,
        "attempted_symbols": int(len(symbols)),
        "chunks_attempted": int(chunks_attempted),
        "new_rows": int(len(new_df)),
        "updated_symbols": int(new_df["symbol"].nunique()) if len(new_df) else 0,
        "error_count": int(len(errors)),
        "errors_sample": errors[:8],
        "aborted": bool(aborted_reason),
        "abort_reason": aborted_reason,
        "elapsed_seconds": round(time.monotonic() - started_clock, 2),
    }


def sync_us_execution_from_scan(tail_rows: int) -> dict[str, Any]:
    exec_path = DATA_DIR / "us_execution_prices_tail.csv"
    scan_path = DATA_DIR / "us_scan_prices_tail.csv"
    before = file_date_stats(exec_path)
    progress("US_EXECUTION: syncing from US_SCAN cache", label="US_EXECUTION", stage="sync_start")
    if not exec_path.exists() or not scan_path.exists():
        progress("US_EXECUTION: missing file, sync skipped", label="US_EXECUTION", stage="sync_missing")
        return {"label": "US_EXECUTION", "status": "missing_file", "before": before}
    exec_symbols = pd.read_csv(exec_path, usecols=["symbol"], dtype={"symbol": str}, low_memory=False)["symbol"].dropna().unique()
    scan = pd.read_csv(scan_path, dtype={"symbol": str}, parse_dates=["date"], low_memory=False)
    filt = normalize_price_frame(scan[scan["symbol"].isin(exec_symbols)].copy())
    missing = sorted(set(exec_symbols) - set(filt["symbol"].unique()))
    if not missing and not filt.empty:
        filt = filt.groupby("symbol", group_keys=False).tail(tail_rows).sort_values(["date", "symbol"])
        safe_csv(filt, exec_path)
        progress("US_EXECUTION: sync complete", label="US_EXECUTION", stage="sync_done")
        return {
            "label": "US_EXECUTION",
            "status": "synced_from_us_scan",
            "before": before,
            "after": file_date_stats(exec_path),
            "attempted_symbols": 0,
            "new_rows": 0,
            "updated_symbols": 0,
            "missing_from_scan": [],
        }
    return {
        "label": "US_EXECUTION",
        "status": "sync_incomplete",
        "before": before,
        "after": file_date_stats(exec_path),
        "missing_from_scan": missing[:30],
    }


def update_tw_map_latest() -> dict[str, Any]:
    map_path = DATA_DIR / "tw_strategy_group_map.csv"
    price_path = DATA_DIR / "tw_strategy_prices_tail.csv"
    progress("TW_MAP: updating latest close/volume fields", label="TW_MAP", stage="map_start")
    if not map_path.exists() or not price_path.exists():
        progress("TW_MAP: missing file, skipped", label="TW_MAP", stage="map_missing")
        return {"status": "missing_file"}
    symbol_map = pd.read_csv(map_path, dtype={"symbol": str}, low_memory=False)
    prices = pd.read_csv(
        price_path,
        usecols=["date", "symbol", "close", "volume"],
        dtype={"symbol": str},
        parse_dates=["date"],
        low_memory=False,
    )
    prices["close"] = pd.to_numeric(prices["close"], errors="coerce")
    prices["volume"] = pd.to_numeric(prices["volume"], errors="coerce")
    latest = prices.sort_values(["symbol", "date"]).dropna(subset=["symbol"]).groupby("symbol", as_index=False).tail(1)
    latest = latest.set_index("symbol")
    for col in ["latest_price_date", "latest_close", "latest_volume"]:
        if col not in symbol_map.columns:
            symbol_map[col] = pd.NA
    symbol_map["latest_price_date"] = symbol_map["symbol"].map(latest["date"].dt.strftime("%Y-%m-%d")).fillna(symbol_map["latest_price_date"])
    symbol_map["latest_close"] = symbol_map["symbol"].map(latest["close"]).fillna(symbol_map["latest_close"])
    symbol_map["latest_volume"] = symbol_map["symbol"].map(latest["volume"]).fillna(symbol_map["latest_volume"])
    safe_csv(symbol_map, map_path)
    progress("TW_MAP: updated", label="TW_MAP", stage="map_done")
    return {
        "status": "updated",
        "symbols": int(len(latest)),
        "max_latest_price_date": str(pd.to_datetime(symbol_map["latest_price_date"], errors="coerce").max().date()),
    }


def update_manifest(report: dict[str, Any]) -> None:
    manifest = read_json(MANIFEST_PATH, {})
    manifest["last_price_update"] = {
        "finished_at": report.get("finished_at"),
        "status": report.get("status"),
        "new_rows_total": report.get("new_rows_total"),
        "max_dates_after": report.get("max_dates_after"),
    }
    write_json(MANIFEST_PATH, manifest)


def update_live_prices(
    markets: set[str] | None = None,
    tail_rows: int | None = None,
    chunk_size: int | None = None,
    sleep_seconds: float | None = None,
    timeout: int | None = None,
    max_symbols_per_file: int | None = None,
    daily_safe: bool | None = None,
) -> dict[str, Any]:
    markets = markets or {"us", "tw"}
    manifest = read_json(MANIFEST_PATH, {})
    tail_rows = int(tail_rows or manifest.get("tail_rows_per_symbol") or os.getenv("LIVE_UPDATE_TAIL_ROWS", "420"))
    chunk_size = int(chunk_size or os.getenv("LIVE_UPDATE_CHUNK_SIZE", "80"))
    sleep_seconds = float(sleep_seconds if sleep_seconds is not None else os.getenv("LIVE_UPDATE_SLEEP_SECONDS", "0.8"))
    timeout = int(timeout or os.getenv("LIVE_UPDATE_TIMEOUT", "30"))
    max_symbols = int(max_symbols_per_file if max_symbols_per_file is not None else os.getenv("LIVE_UPDATE_MAX_SYMBOLS", "0"))
    if daily_safe is None:
        daily_safe = os.getenv("LIVE_UPDATE_DAILY_SAFE", "0") == "1"
    if daily_safe:
        chunk_size = int(os.getenv("LIVE_UPDATE_SAFE_CHUNK_SIZE", str(min(chunk_size, 5))))
        sleep_seconds = float(os.getenv("LIVE_UPDATE_SAFE_SLEEP_SECONDS", str(max(sleep_seconds, 3.0))))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    started = now_iso()
    progress(
        f"market update started; markets={','.join(sorted(markets))}, daily_safe={bool(daily_safe)}, chunk={chunk_size}, sleep={sleep_seconds}, timeout={timeout}",
        reset=True,
        stage="run_start",
        markets=sorted(markets),
        daily_safe=bool(daily_safe),
        chunk_size=chunk_size,
        sleep_seconds=sleep_seconds,
        timeout=timeout,
    )
    files: list[dict[str, Any]] = []
    map_update: dict[str, Any] = {}

    try:
        if "us" in markets:
            if daily_safe:
                files.append(
                    update_price_file(
                        DATA_DIR / "us_execution_prices_tail.csv",
                        market="US",
                        label="US_EXECUTION",
                        tail_rows=tail_rows,
                        chunk_size=chunk_size,
                        sleep_seconds=sleep_seconds,
                        timeout=timeout,
                        max_symbols=max_symbols,
                    )
                )
                files.append(
                    {
                        "label": "US_SCAN",
                        "status": "skipped_daily_safe_mode",
                        "before": file_date_stats(DATA_DIR / "us_scan_prices_tail.csv"),
                        "after": file_date_stats(DATA_DIR / "us_scan_prices_tail.csv"),
                        "new_rows": 0,
                        "error_count": 0,
                    }
                )
            else:
                files.append(
                    update_price_file(
                        DATA_DIR / "us_scan_prices_tail.csv",
                        market="US",
                        label="US_SCAN",
                        tail_rows=tail_rows,
                        chunk_size=chunk_size,
                        sleep_seconds=sleep_seconds,
                        timeout=timeout,
                        max_symbols=max_symbols,
                    )
                )
                if not max_symbols:
                    files.append(sync_us_execution_from_scan(tail_rows=tail_rows))
                else:
                    files.append(
                        update_price_file(
                            DATA_DIR / "us_execution_prices_tail.csv",
                            market="US",
                            label="US_EXECUTION",
                            tail_rows=tail_rows,
                            chunk_size=chunk_size,
                            sleep_seconds=sleep_seconds,
                            timeout=timeout,
                            max_symbols=max_symbols,
                        )
                    )
        if "tw" in markets:
            if daily_safe:
                files.append(
                    {
                        "label": "TW_STOCKS",
                        "status": "skipped_daily_safe_mode",
                        "before": file_date_stats(DATA_DIR / "tw_strategy_prices_tail.csv"),
                        "after": file_date_stats(DATA_DIR / "tw_strategy_prices_tail.csv"),
                        "new_rows": 0,
                        "error_count": 0,
                    }
                )
                files.append(
                    update_price_file(
                        DATA_DIR / "tw_index_prices_tail.csv",
                        market="TW",
                        label="TW_INDEX",
                        tail_rows=tail_rows,
                        chunk_size=chunk_size,
                        sleep_seconds=sleep_seconds,
                        timeout=timeout,
                        max_symbols=max_symbols,
                    )
                )
            else:
                files.append(
                    update_price_file(
                        DATA_DIR / "tw_strategy_prices_tail.csv",
                        market="TW",
                        label="TW_STOCKS",
                        tail_rows=tail_rows,
                        chunk_size=chunk_size,
                        sleep_seconds=sleep_seconds,
                        timeout=timeout,
                        max_symbols=max_symbols,
                    )
                )
                files.append(
                    update_price_file(
                        DATA_DIR / "tw_index_prices_tail.csv",
                        market="TW",
                        label="TW_INDEX",
                        tail_rows=tail_rows,
                        chunk_size=chunk_size,
                        sleep_seconds=sleep_seconds,
                        timeout=timeout,
                        max_symbols=max_symbols,
                    )
                )
            map_update = update_tw_map_latest()

        new_rows_total = int(sum(int(item.get("new_rows", 0) or 0) for item in files))
        max_dates_after = {
            item.get("label", ""): (item.get("after") or {}).get("max_date")
            for item in files
            if isinstance(item.get("after"), dict)
        }
        hard_errors = [item for item in files if item.get("status") in {"missing_file", "empty_file"}]
        aborted_files = [item for item in files if item.get("aborted") or str(item.get("status", "")).startswith("aborted_")]
        all_error_text = json.dumps(files, ensure_ascii=False)
        source_rate_limited = "rate_limited" in all_error_text or "Too Many Requests" in all_error_text or "Rate limited" in all_error_text
        source_empty = bool(files) and new_rows_total == 0 and any(int(item.get("chunks_attempted", 0) or 0) > 0 for item in files)
        status = "updated" if new_rows_total else "no_new_rows"
        if aborted_files:
            status = "aborted"
        if source_rate_limited and not new_rows_total:
            status = "source_rate_limited"
        if hard_errors:
            status = "partial_error"
        report = {
            "ok": not bool(hard_errors),
            "status": status,
            "source": "yfinance",
            "started_at": started,
            "finished_at": now_iso(),
            "data_dir": str(DATA_DIR),
            "tail_rows_per_symbol": tail_rows,
            "chunk_size": chunk_size,
            "sleep_seconds": sleep_seconds,
            "timeout": timeout,
            "max_symbols_per_file": max_symbols,
            "daily_safe": bool(daily_safe),
            "new_rows_total": new_rows_total,
            "possible_source_limit_or_no_new_data": source_empty,
            "source_rate_limited": source_rate_limited,
            "max_dates_after": max_dates_after,
            "files": files,
            "tw_map_update": map_update,
            "progress_path": str(PROGRESS_PATH),
            "progress_json_path": str(PROGRESS_JSON_PATH),
        }
    except Exception as exc:
        report = {
            "ok": False,
            "status": "failed",
            "source": "yfinance",
            "started_at": started,
            "finished_at": now_iso(),
            "error": str(exc),
            "traceback": traceback.format_exc(limit=8),
            "files": files,
            "progress_path": str(PROGRESS_PATH),
            "progress_json_path": str(PROGRESS_JSON_PATH),
        }

    write_json(REPORT_PATH, report)
    write_json(DATA_REPORT_PATH, report)
    update_manifest(report)
    progress(
        f"market update finished; status={report.get('status')}, new_rows={report.get('new_rows_total')}, max_dates={report.get('max_dates_after')}",
        stage="run_done",
        status=report.get("status"),
        new_rows_total=report.get("new_rows_total"),
        max_dates_after=report.get("max_dates_after"),
    )
    return report


def parse_markets(value: str) -> set[str]:
    value = value.strip().lower()
    if value in {"all", ""}:
        return {"us", "tw"}
    return {part.strip() for part in value.split(",") if part.strip() in {"us", "tw"}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--markets", default=os.getenv("LIVE_UPDATE_MARKETS", "all"), help="all, us, tw, or us,tw")
    parser.add_argument("--tail-rows", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--sleep-seconds", type=float, default=None)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--max-symbols-per-file", type=int, default=None)
    parser.add_argument("--daily-safe", action="store_true", help="Update only execution-sized files and skip large scan pools to avoid Yahoo rate limits.")
    args = parser.parse_args()
    report = update_live_prices(
        markets=parse_markets(args.markets),
        tail_rows=args.tail_rows,
        chunk_size=args.chunk_size,
        sleep_seconds=args.sleep_seconds,
        timeout=args.timeout,
        max_symbols_per_file=args.max_symbols_per_file,
        daily_safe=args.daily_safe,
    )
    print(json.dumps(clean(report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
