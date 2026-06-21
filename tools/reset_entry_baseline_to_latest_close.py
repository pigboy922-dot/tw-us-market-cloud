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


REPORT_PATH = build_dashboard.OUTPUT_DIR / "LATEST_ENTRY_BASELINE_RESET_REPORT.json"


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


def latest_close_rows(market: str, symbols: list[str]) -> list[dict[str, Any]]:
    history = build_dashboard.price_history_frame(market, set(symbols))
    latest_by_symbol: dict[str, dict[str, Any]] = {}
    if not history.empty:
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


def main() -> None:
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
    report["baseline_path"] = str(build_dashboard.ENTRY_BASELINE_PATH)
    report["total_reset_count"] = len(all_rows)
    write_json(REPORT_PATH, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
