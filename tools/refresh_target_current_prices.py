from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import build_dashboard  # noqa: E402
from tools.update_live_prices import clean, download_chunk, write_json  # noqa: E402


REPORT_PATH = build_dashboard.OUTPUT_DIR / "LATEST_TARGET_PRICE_REFRESH_REPORT.json"
DATA_REPORT_PATH = build_dashboard.DATA_DIR / "LAST_TARGET_PRICE_REFRESH_REPORT.json"
PROGRESS_PATH = build_dashboard.OUTPUT_DIR / "LATEST_TARGET_PRICE_REFRESH_PROGRESS.txt"
DEFAULT_LOOKBACK_DAYS = int(os.getenv("TARGET_PRICE_LOOKBACK_DAYS", "21"))
DEFAULT_CHUNK_SIZE = int(os.getenv("TARGET_PRICE_CHUNK_SIZE", "1"))
DEFAULT_SLEEP_SECONDS = float(os.getenv("TARGET_PRICE_SLEEP_SECONDS", "1"))
DEFAULT_TIMEOUT = int(os.getenv("TARGET_PRICE_TIMEOUT", "30"))
DEFAULT_RETRY_ATTEMPTS = int(os.getenv("TARGET_PRICE_RETRY_ATTEMPTS", "2"))
DEFAULT_RETRY_SLEEP_SECONDS = float(os.getenv("TARGET_PRICE_RETRY_SLEEP_SECONDS", "12"))


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def progress(message: str, reset: bool = False) -> None:
    build_dashboard.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if reset:
        PROGRESS_PATH.write_text("", encoding="utf-8-sig")
    line = f"[{now_iso()}] {message}"
    print(line, flush=True)
    with PROGRESS_PATH.open("a", encoding="utf-8-sig") as fh:
        fh.write(line + "\n")


def parse_date(value: Any) -> pd.Timestamp | None:
    try:
        ts = pd.Timestamp(value)
    except Exception:
        return None
    if pd.isna(ts):
        return None
    return ts.tz_localize(None) if getattr(ts, "tzinfo", None) is not None else ts


def read_target_symbols(market: str) -> list[str]:
    path = build_dashboard.OUTPUT_DIR / market / "latest_target_position.csv"
    symbols: list[str] = []
    if path.exists():
        try:
            df = pd.read_csv(path, dtype={"symbol": str}, low_memory=False)
            if "target_weight" not in df.columns:
                df["target_weight"] = 0.0
            df["target_weight"] = pd.to_numeric(df["target_weight"], errors="coerce").fillna(0.0)
            for symbol in df.loc[df["target_weight"].gt(0), "symbol"].dropna().astype(str):
                symbol = symbol.strip()
                if symbol and symbol.upper() != "CASH":
                    symbols.append(symbol)
        except Exception:
            pass

    for row in build_dashboard.read_current_holdings():
        if str(row.get("market", "")).upper() != market:
            continue
        symbol = str(row.get("symbol", "")).strip()
        if symbol and symbol.upper() != "CASH":
            symbols.append(symbol)
    return sorted(dict.fromkeys(symbols))


def latest_reference_dates(market: str, symbols: list[str]) -> dict[str, pd.Timestamp]:
    price_map = build_dashboard.latest_price_map(market)
    out: dict[str, pd.Timestamp] = {}
    for symbol in symbols:
        row = price_map.get(symbol, {})
        candidates = [parse_date(row.get("date")), parse_date(row.get("current_price_date"))]
        candidates = [ts for ts in candidates if ts is not None]
        if candidates:
            out[symbol] = max(candidates)
    return out


def refresh_market(
    market: str,
    symbols: list[str],
    lookback_days: int,
    chunk_size: int,
    sleep_seconds: float,
    timeout: int,
    retry_attempts: int,
    retry_sleep_seconds: float,
) -> dict[str, Any]:
    if not symbols:
        progress(f"{market}: no target/holding symbols to refresh")
        return {
            "market": market,
            "status": "no_symbols",
            "symbols": [],
            "updated_symbols": 0,
            "skipped_older": [],
            "errors": [],
            "rows": [],
        }

    start = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    progress(f"{market}: refresh {len(symbols)} symbols from {start}; chunk={chunk_size}, sleep={sleep_seconds}s")
    reference_dates = latest_reference_dates(market, symbols)
    updated_rows: list[dict[str, Any]] = []
    skipped_older: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    chunks_attempted = 0
    download_attempts = 0

    for idx in range(0, len(symbols), chunk_size):
        chunk_symbols = symbols[idx : idx + chunk_size]
        chunks_attempted += 1
        progress(f"{market}: quote chunk {chunks_attempted}, symbols={','.join(chunk_symbols)}")
        data = pd.DataFrame()
        err = ""
        attempts_used = 0
        for attempt in range(1, max(1, retry_attempts) + 1):
            attempts_used = attempt
            download_attempts += 1
            data, err = download_chunk(chunk_symbols, market=market, start=start, timeout=timeout)
            if not data.empty:
                break
            if attempt < max(1, retry_attempts):
                err_text = str(err or "empty")
                multiplier = 2.0 if ("Too Many Requests" in err_text or "Rate limited" in err_text) else 1.0
                time.sleep(retry_sleep_seconds * attempt * multiplier)
        if err and data.empty:
            errors.append({"chunk": chunks_attempted, "symbols": chunk_symbols, "attempts": attempts_used, "error": err})
            progress(f"{market}: quote chunk {chunks_attempted} failed: {str(err)[:180]}")
        if not data.empty:
            data = data.dropna(subset=["date", "symbol", "close"]).copy()
            data["close"] = pd.to_numeric(data["close"], errors="coerce")
            data = data[data["close"].gt(0)]
            for symbol, part in data.groupby("symbol", sort=False):
                if part.empty:
                    continue
                quote = part.sort_values("date").tail(1).iloc[0]
                quote_date = parse_date(quote["date"])
                reference_date = reference_dates.get(str(symbol))
                if quote_date is None:
                    continue
                if reference_date is not None and quote_date < reference_date:
                    skipped_older.append(
                        {
                            "symbol": str(symbol),
                            "quote_date": quote_date.strftime("%Y-%m-%d"),
                            "reference_date": reference_date.strftime("%Y-%m-%d"),
                        }
                    )
                    continue
                updated_rows.append(
                    {
                        "market": market,
                        "symbol": str(symbol),
                        "current_price": float(quote["close"]),
                        "price_date": quote_date.strftime("%Y-%m-%d"),
                        "source": "yfinance_target_daily",
                        "updated_at": now_iso(),
                    }
                )
                progress(f"{market}: updated {symbol} price={float(quote['close']):.4f} date={quote_date.strftime('%Y-%m-%d')}")
        if sleep_seconds > 0 and idx + chunk_size < len(symbols):
            time.sleep(sleep_seconds)

    status = "updated" if updated_rows else "no_new_quotes"
    if errors and not updated_rows:
        status = "source_error"
    return {
        "market": market,
        "status": status,
        "start": start,
        "symbols": symbols,
        "attempted_symbols": len(symbols),
        "chunks_attempted": chunks_attempted,
        "download_attempts": download_attempts,
        "retry_attempts": retry_attempts,
        "retry_sleep_seconds": retry_sleep_seconds,
        "updated_symbols": len({row["symbol"] for row in updated_rows}),
        "skipped_older": skipped_older[:20],
        "errors": errors[:10],
        "rows": updated_rows,
    }


def refresh_target_current_prices(
    markets: set[str] | None = None,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    sleep_seconds: float = DEFAULT_SLEEP_SECONDS,
    timeout: int = DEFAULT_TIMEOUT,
    retry_attempts: int = DEFAULT_RETRY_ATTEMPTS,
    retry_sleep_seconds: float = DEFAULT_RETRY_SLEEP_SECONDS,
) -> dict[str, Any]:
    markets = markets or {"TW", "US"}
    started = now_iso()
    progress(f"target quote refresh started; markets={','.join(sorted(markets))}", reset=True)
    market_reports: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    try:
        for market in ["TW", "US"]:
            if market not in markets:
                continue
            report = refresh_market(
                market=market,
                symbols=read_target_symbols(market),
                lookback_days=lookback_days,
                chunk_size=chunk_size,
                sleep_seconds=sleep_seconds,
                timeout=timeout,
                retry_attempts=retry_attempts,
                retry_sleep_seconds=retry_sleep_seconds,
            )
            rows.extend(report.pop("rows", []))
            market_reports.append(report)
        merged = build_dashboard.merge_current_prices(rows) if rows else build_dashboard.read_current_prices()
        updated_symbols = sorted({f"{row['market']}:{row['symbol']}" for row in rows})
        status = "updated" if rows else "no_new_quotes"
        if not rows and any(report.get("status") == "source_error" for report in market_reports):
            status = "source_error"
        report = {
            "ok": True,
            "status": status,
            "source": "yfinance_target_only",
            "started_at": started,
            "finished_at": now_iso(),
            "lookback_days": lookback_days,
            "chunk_size": chunk_size,
            "sleep_seconds": sleep_seconds,
            "timeout": timeout,
            "retry_attempts": retry_attempts,
            "retry_sleep_seconds": retry_sleep_seconds,
            "updated_symbols": updated_symbols,
            "updated_symbol_count": len(updated_symbols),
            "current_price_rows": len(merged),
            "markets": market_reports,
            "notes": "Target-only quote refresh updates state/current_prices.csv and entry judgment only. It does not update the full scan pool or retrain strategy rules.",
        }
    except Exception as exc:
        report = {
            "ok": False,
            "status": "failed",
            "source": "yfinance_target_only",
            "started_at": started,
            "finished_at": now_iso(),
            "error": str(exc),
            "traceback": traceback.format_exc(limit=8),
            "markets": market_reports,
        }

    write_json(REPORT_PATH, report)
    write_json(DATA_REPORT_PATH, report)
    progress(f"target quote refresh finished; status={report.get('status')} updated={report.get('updated_symbol_count')}")
    return report


def parse_markets(value: str) -> set[str]:
    value = value.strip().upper()
    if value in {"", "ALL"}:
        return {"TW", "US"}
    return {part.strip() for part in value.split(",") if part.strip() in {"TW", "US"}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--markets", default="all", help="all, TW, US, or TW,US")
    parser.add_argument("--lookback-days", type=int, default=DEFAULT_LOOKBACK_DAYS)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--sleep-seconds", type=float, default=DEFAULT_SLEEP_SECONDS)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--retry-attempts", type=int, default=DEFAULT_RETRY_ATTEMPTS)
    parser.add_argument("--retry-sleep-seconds", type=float, default=DEFAULT_RETRY_SLEEP_SECONDS)
    args = parser.parse_args()
    report = refresh_target_current_prices(
        markets=parse_markets(args.markets),
        lookback_days=args.lookback_days,
        chunk_size=args.chunk_size,
        sleep_seconds=args.sleep_seconds,
        timeout=args.timeout,
        retry_attempts=args.retry_attempts,
        retry_sleep_seconds=args.retry_sleep_seconds,
    )
    print(json.dumps(clean(report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
