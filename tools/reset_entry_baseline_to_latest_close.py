from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import build_dashboard  # noqa: E402
from tools.update_live_prices import is_incomplete_market_date  # noqa: E402


REPORT_PATH = build_dashboard.OUTPUT_DIR / "LATEST_ENTRY_BASELINE_RESET_REPORT.json"
LIVE_CYCLE_BASELINE_PATH = build_dashboard.STATE_DIR / "live_cycle_baseline.csv"
LIVE_CYCLE_BASELINE_COLS = [
    "market",
    "cycle_rebalance_date",
    "source_signal_date",
    "rebalance_step_trading_days",
    "baseline_type",
    "created_at",
]


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def target_symbols(market: str) -> list[str]:
    rows = build_dashboard.read_csv(build_dashboard.OUTPUT_DIR / market / "latest_target_position.csv")
    symbols: list[str] = []
    for row in rows:
        symbol = str(row.get("symbol", "")).strip()
        if not symbol or symbol.upper() == "CASH":
            continue
        weight = build_dashboard.as_float(row.get("target_weight")) or 0.0
        if weight > 1e-12:
            symbols.append(symbol)
    return symbols


def market_source_signal_date(market: str) -> str:
    rows = build_dashboard.read_csv(build_dashboard.OUTPUT_DIR / market / "latest_market_state.csv")
    return str(
        build_dashboard.state_value(
            rows,
            "component_signal_date",
            build_dashboard.state_value(rows, "latest_price_date", ""),
        )
        or ""
    )


def market_rebalance_step(market: str) -> int:
    rows = build_dashboard.read_csv(build_dashboard.OUTPUT_DIR / market / "latest_market_state.csv")
    step = build_dashboard.as_int(build_dashboard.state_value(rows, "rebalance_step_trading_days", 0), 0)
    return int(step or 0)


def latest_close_rows(market: str, symbols: list[str]) -> list[dict[str, Any]]:
    history = build_dashboard.price_history_frame(market, set(symbols))
    latest_by_symbol: dict[str, dict[str, Any]] = {}
    if not history.empty:
        history = history.copy()
        history["_date_ts"] = pd.to_datetime(history["date"], errors="coerce")
        history = history.loc[
            history["_date_ts"].notna()
            & ~history["_date_ts"].map(lambda ts: is_incomplete_market_date(ts, market))
        ].drop(columns=["_date_ts"])
        latest = history.sort_values(["symbol", "date"]).groupby("symbol", as_index=False).tail(1)
        for row in latest.to_dict(orient="records"):
            latest_by_symbol[str(row["symbol"])] = {
                "signal_date": pd.Timestamp(row["date"]).strftime("%Y-%m-%d"),
                "signal_close": float(row["close"]),
            }

    price_map = build_dashboard.latest_price_map(market)
    out: list[dict[str, Any]] = []
    source_signal_date = market_source_signal_date(market)
    created_at = now_iso()
    for symbol in symbols:
        rec = latest_by_symbol.get(symbol)
        if rec is None:
            fallback = price_map.get(symbol, {})
            close = build_dashboard.as_float(fallback.get("close") or fallback.get("current_price"))
            date = str(fallback.get("date") or fallback.get("current_price_date") or "")
            if close is None or close <= 0 or not date:
                continue
            if is_incomplete_market_date(date, market):
                continue
            rec = {"signal_date": date, "signal_close": close}
        out.append(
            {
                "market": market,
                "symbol": symbol,
                "signal_date": rec["signal_date"],
                "signal_close": rec["signal_close"],
                "source_signal_date": source_signal_date or rec["signal_date"],
                "baseline_type": "manual_latest_close",
                "created_at": created_at,
            }
        )
    return out


def write_live_cycle_baselines(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_market: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_market.setdefault(str(row.get("market", "")).upper(), []).append(row)

    created_at = now_iso()
    cycle_rows: list[dict[str, Any]] = []
    for market in ["TW", "US"]:
        market_rows = by_market.get(market, [])
        dates = sorted({str(row.get("signal_date", "")) for row in market_rows if str(row.get("signal_date", ""))})
        if not dates:
            continue
        cycle_rows.append(
            {
                "market": market,
                "cycle_rebalance_date": dates[-1],
                "source_signal_date": dates[-1],
                "rebalance_step_trading_days": market_rebalance_step(market),
                "baseline_type": "manual_go_live_close",
                "created_at": created_at,
            }
        )
    build_dashboard.STATE_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(cycle_rows, columns=LIVE_CYCLE_BASELINE_COLS)
    tmp = LIVE_CYCLE_BASELINE_PATH.with_suffix(".csv.tmp")
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    tmp.replace(LIVE_CYCLE_BASELINE_PATH)
    return cycle_rows


def reset_entry_baseline_to_latest_close() -> dict[str, Any]:
    all_rows: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "ok": True,
        "status": "reset_to_latest_close",
        "generated_at": now_iso(),
        "markets": {},
    }
    for market in ["TW", "US"]:
        symbols = target_symbols(market)
        rows = latest_close_rows(market, symbols)
        all_rows.extend(rows)
        report["markets"][market] = {
            "target_symbols": symbols,
            "reset_count": len(rows),
            "source_signal_date": market_source_signal_date(market),
            "baseline_dates": sorted({str(row["signal_date"]) for row in rows}),
        }
    build_dashboard.write_entry_baselines(all_rows)
    cycle_rows = write_live_cycle_baselines(all_rows)
    report["baseline_path"] = str(build_dashboard.ENTRY_BASELINE_PATH)
    report["live_cycle_baseline_path"] = str(LIVE_CYCLE_BASELINE_PATH)
    report["live_cycle_baselines"] = cycle_rows
    report["total_reset_count"] = len(all_rows)
    write_json(REPORT_PATH, report)
    return report


def main() -> None:
    report = reset_entry_baseline_to_latest_close()
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
