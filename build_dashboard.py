from __future__ import annotations

import html
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", BASE_DIR / "runtime_outputs"))
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data_live"))
STATE_DIR = Path(os.getenv("STATE_DIR", BASE_DIR / "state"))
DASHBOARD_PATH = OUTPUT_DIR / "DAILY_MARKET_POOL_DASHBOARD.html"
HOLDINGS_PATH = STATE_DIR / "current_holdings.csv"
CURRENT_PRICES_PATH = STATE_DIR / "current_prices.csv"
ENTRY_BASELINE_PATH = STATE_DIR / "entry_signal_baseline.csv"
ORDER_PLAN_PATH = OUTPUT_DIR / "LATEST_ORDER_PLAN.csv"
HOLDING_COLS = ["market", "symbol", "shares", "avg_cost", "note"]
CURRENT_PRICE_COLS = ["market", "symbol", "current_price", "price_date", "source", "updated_at"]
ENTRY_BASELINE_COLS = [
    "market",
    "symbol",
    "signal_date",
    "signal_close",
    "source_signal_date",
    "baseline_type",
    "created_at",
]
DEFAULT_CAPITAL = {"TW": 1_000_000.0, "US": 100_000.0}
PRICE_CURRENT_KEYS = ["live_price", "current_price", "last_price", "regular_market_price", "realtime_price"]
ENTRY_RULE_DEFAULTS = {
    "TW": {"floor": -7.0, "full": 3.0, "partial": 3.0, "hard": 3.0},
    "US": {"floor": -5.0, "full": 5.0, "partial": 5.0, "hard": 5.0},
}
ENTRY_RULE_LOCKS = {
    "TW": {"floor": -7.0, "full": 3.0, "partial": 3.0, "hard": 3.0},
    "US": {"floor": -5.0, "full": 5.0, "partial": 5.0, "hard": 5.0},
}


def zh(value: str) -> str:
    return value.encode("ascii").decode("unicode_escape")


ROLE_LABELS = {
    "core": zh(r"\u6838\u5fc3"),
    "core_leader": zh(r"\u4e3b\u984c\u9f8d\u982d"),
    "accelerator": zh(r"\u52a0\u901f"),
    "theme_leader": zh(r"\u984c\u6750\u9f8d\u982d"),
    "residual_cash": zh(r"\u4fdd\u7559\u73fe\u91d1"),
    "risk_off_cash": zh(r"\u9632\u5b88\u73fe\u91d1"),
    "weak_or_residual_cash": zh(r"\u9632\u5b88\u73fe\u91d1"),
}

THEME_LABELS = {
    "ai_compute": "AI " + zh(r"\u904b\u7b97"),
    "memory_storage": zh(r"\u8a18\u61b6\u9ad4/\u5132\u5b58"),
    "server_network": zh(r"\u4f3a\u670d\u5668/\u7db2\u901a"),
    "space_ev_defense": zh(r"\u592a\u7a7a/EV/\u8ecd\u5de5"),
    "thermal_power": zh(r"\u6563\u71b1/\u96fb\u529b"),
    "grid_engineering": zh(r"\u96fb\u7db2\u5de5\u7a0b"),
    "semi_equipment": zh(r"\u534a\u5c0e\u9ad4\u8a2d\u5099"),
    "software_platform": zh(r"\u8edf\u9ad4\u5e73\u53f0"),
    "optical_communication": zh(r"\u5149\u901a\u8a0a"),
    "compound_semiconductor": zh(r"\u5316\u5408\u7269\u534a\u5c0e\u9ad4"),
    "power_semiconductor": zh(r"\u529f\u7387\u534a\u5c0e\u9ad4"),
    "semi_foundry_idm": zh(r"\u6676\u5713/IDM"),
    "cloud_infra": zh(r"\u96f2\u7aef\u57fa\u5efa"),
    "cloud_ai_infra": "AI " + zh(r"\u96f2\u7aef\u57fa\u5efa"),
    "electronics_manufacturing": zh(r"\u96fb\u5b50\u88fd\u9020"),
    "pcb_material": "PCB " + zh(r"\u6750\u6599"),
    "substrate": zh(r"\u8f09\u677f"),
    "wafer_material": zh(r"\u6676\u5713\u6750\u6599"),
    "foundry": zh(r"\u6676\u5713\u4ee3\u5de5"),
    "dram_nand": "DRAM/NAND",
    "semiconductor_equipment": zh(r"\u534a\u5c0e\u9ad4\u8a2d\u5099"),
    "storage_device": zh(r"\u5132\u5b58\u88dd\u7f6e"),
    "passive_component": zh(r"\u88ab\u52d5\u5143\u4ef6"),
    "packaging_test": zh(r"\u5c01\u6e2c"),
    "core_2330": zh(r"\u53f0\u7a4d\u96fb\u6838\u5fc3"),
    "cash": zh(r"\u73fe\u91d1"),
    "us_unclassified": zh(r"\u672a\u5206\u985e\u5f37\u52e2\u80a1"),
    "us981a_ai_theme": "US 981A " + zh(r"\u5f0fAI\u4e3b\u984c"),
}

TXT = {
    "no_target": zh(r"\u76ee\u524d\u6c92\u6709\u76ee\u6a19\u6301\u5009"),
    "symbol": zh(r"\u6a19\u7684"),
    "theme": zh(r"\u4e3b\u984c"),
    "role": zh(r"\u5b9a\u4f4d"),
    "target_weight": zh(r"\u76ee\u6a19\u6b0a\u91cd"),
    "source": zh(r"\u4f86\u6e90"),
    "exec_group": zh(r"\u5c64\u7d1a"),
    "main_position": zh(r"\u4e3b\u6301\u5009"),
    "micro_position": zh(r"\u5c0f\u6b0a\u91cd"),
    "amount_calc": zh(r"\u8cc7\u91d1\u8a66\u7b97"),
    "no_order": zh(r"\u6c92\u6709\u6b63\u5f0f\u8cb7\u55ae"),
    "yes": zh(r"\u662f"),
    "no": zh(r"\u5426"),
    "no_data": zh(r"\u6c92\u6709\u8cc7\u6599"),
    "stock": zh(r"\u80a1\u7968"),
    "score": zh(r"\u5206\u6578"),
    "status": zh(r"\u72c0\u614b"),
    "official_pool": zh(r"\u6b63\u5f0f\u6c60"),
    "watch": zh(r"\u89c0\u5bdf"),
    "no_watch": zh(r"\u6c92\u6709\u89c0\u5bdf\u8b66\u793a"),
    "included": zh(r"\u5df2\u7d0d\u5165"),
    "no_theme": zh(r"\u6c92\u6709\u4e3b\u984c\u6392\u884c"),
    "today_orders": zh(r"\u4eca\u65e5\u6b63\u5f0f\u8cb7\u55ae"),
    "top_note": "正式買單使用台股攻擊/推薦融合 H46 乾淨版；攻擊腿明顯勝出才切攻擊，否則用推薦腿。",
    "rebalance_due": zh(r"\u4e0b\u6b21\u4ea4\u6613\u65e5\u63db\u5009"),
    "not_rebalance": zh(r"\u975e\u63db\u5009\u65e5"),
    "us_risk_on": zh(r"\u7f8e\u80a1\u98a8\u96aa\u958b"),
    "us_defense": zh(r"\u7f8e\u80a1\u9632\u5b88"),
    "tw_strong": zh(r"\u53f0\u80a1\u5f37\u52e2"),
    "tw": zh(r"\u53f0\u80a1"),
    "us": zh(r"\u7f8e\u80a1"),
    "data_date": zh(r"\u8cc7\u6599\u65e5"),
    "gross": zh(r"\u7e3d\u66dd\u96aa"),
    "scan_pool": zh(r"\u6383\u63cf\u6c60"),
    "unit_symbols": zh(r"\u6a94"),
    "summary": zh(r"\u5feb\u901f\u6301\u5009\u6458\u8981"),
    "mode": zh(r"\u6a21\u5f0f"),
    "tw_target": "台股正式目標 等權",
    "tw_target_note": "台股攻擊/推薦融合 H46 決定股票名單；正式下單採等權執行，不重新訓練。",
    "us_target": "美股正式目標 Top4 上限25%",
    "us_target_note": zh(r"\u6b63\u5f0f\u7b56\u7565\uff1a\u7f8e\u80a1 Top1 \u52d5\u614b\u5e02\u5834\u7248\uff1b\u6bcf\u65e5\u7528 Top800 \u6d41\u52d5\u5e02\u5834\u6c60\u91cd\u7b97\u3002"),
    "tw_capital": zh(r"\u53f0\u80a1\u8cc7\u91d1"),
    "us_capital": zh(r"\u7f8e\u80a1\u8cc7\u91d1"),
    "us_watch": zh(r"\u7f8e\u80a1\u5168\u5834\u71b1\u9580\u89c0\u5bdf"),
    "us_watch_note": zh(r"\u6383\u63cf\u699c\u7528\u4f86\u767c\u73fe\u65b0\u4e3b\u7dda\uff1b\u7b26\u5408\u51cd\u7d50\u898f\u5247\u624d\u9032\u52d5\u614b\u5e02\u5834\u7b56\u7565\u3002"),
    "watch_layer": zh(r"\u89c0\u5bdf\u5c64"),
    "theme_heat": zh(r"\u4e3b\u984c\u71b1\u5ea6"),
    "stock_accel": zh(r"\u80a1\u7968\u52a0\u901f\u699c"),
    "tw_theme_rank": zh(r"\u53f0\u80a1\u984c\u6750\u6392\u884c"),
    "us_theme_rank": zh(r"\u7f8e\u80a1\u6b63\u5f0f\u984c\u6750\u6392\u884c"),
    "strong_stocks": zh(r"\u5f37\u52e2\u80a1"),
    "tw_candidates": zh(r"\u53f0\u80a1\u5019\u9078\u6c60"),
    "name": zh(r"\u540d\u7a31"),
    "selected_today": zh(r"\u4eca\u65e5\u9078\u5165"),
    "amount_rank": zh(r"\u91cf\u6392\u884c"),
    "us_candidates": zh(r"\u7f8e\u80a1\u6b63\u5f0f\u5019\u9078\u6c60"),
    "eligible": zh(r"\u5408\u683c"),
    "us_alerts": zh(r"\u7f8e\u80a1\u89c0\u5bdf\u8b66\u793a"),
    "alerts_note": zh(r"\u6b63\u5f0f\u7b56\u7565\u5916\u7684\u71b1\u9580\u80a1\u7968\uff0c\u53ea\u80fd\u7576\u89c0\u5bdf\u6216\u4e0b\u4e00\u7248\u7814\u7a76\u4f86\u6e90\u3002"),
    "us_scan_theme_rank": zh(r"\u7f8e\u80a1\u6383\u63cf\u4e3b\u984c\u6392\u884c"),
    "us_scan_turnover": zh(r"\u7f8e\u80a1 Top800 \u9032\u51fa"),
    "us_scan_turnover_note": zh(r"\u9019\u662f\u6383\u63cf\u6c60\u9032\u51fa\uff0c\u4e0d\u662f\u6b63\u5f0f\u8cb7\u55ae\u3002entered_top800 \u4ee3\u8868\u65b0\u9032\u96f7\u9054\uff0cexited_top800 \u4ee3\u8868\u88ab\u5254\u51fa\u96f7\u9054\u3002"),
    "us_upgrade_candidates": zh(r"\u7f8e\u80a1\u5347\u7d1a\u5019\u9078\u6c60"),
    "us_upgrade_candidates_note": zh(r"\u9019\u662f Top800 \u96f7\u9054\u6311\u51fa\u7684\u5f37\u52e2\u984c\u6750\u5019\u9078\uff0c\u4e0d\u662f\u4eca\u65e5\u6b63\u5f0f\u8cb7\u55ae\u3002\u5019\u9078\u80a1\u8981\u7d93\u904e\u4e0b\u4e00\u7248\u4e7e\u6de8\u8a13\u7df4\u9a57\u8b49\uff0c\u624d\u80fd\u9032\u6b63\u5f0f\u7b56\u7565\u3002"),
    "us_dynamic_formal_pool": zh(r"\u7f8e\u80a1\u6d41\u52d5\u6b63\u5f0f\u5019\u9078\u6c60"),
    "us_dynamic_formal_pool_note": zh(r"\u9019\u662f\u820a\u5019\u9078\u53c3\u8003\u8868\uff1b\u73fe\u5728\u6b63\u5f0f\u8cb7\u55ae\u4f7f\u7528 Top1 \u52d5\u614b\u5e02\u5834\u7248\u3002"),
    "change": zh(r"\u8b8a\u52d5"),
    "new_rank": zh(r"\u65b0\u6392\u540d"),
    "old_rank": zh(r"\u820a\u6392\u540d"),
    "members": zh(r"\u6210\u54e1"),
    "update": zh(r"\u66f4\u65b0\u5e02\u5834"),
    "updating": zh(r"\u66f4\u65b0\u4e2d..."),
    "done_reload": zh(r"\u5b8c\u6210\uff0c\u91cd\u65b0\u8f09\u5165"),
    "excel_record": zh(r"\u4e0b\u8f09Excel\u7d00\u9304"),
    "price_update": zh(r"\u884c\u60c5\u66f4\u65b0\u72c0\u614b"),
    "price_update_note": zh(r"\u9019\u88e1\u986f\u793a\u672c\u6b21\u662f\u5426\u771f\u7684\u5beb\u5165\u65b0\u884c\u60c5\u3002\u82e5\u70ba 0\uff0c\u4ee3\u8868\u8cc7\u6599\u6e90\u6c92\u6709\u56de\u65b0\u8cc7\u6599\u6216\u88ab\u9650\u6d41\u3002"),
    "price_updated": zh(r"\u884c\u60c5\u5df2\u66f4\u65b0"),
    "price_no_new": zh(r"\u672a\u6293\u5230\u65b0\u884c\u60c5"),
    "price_no_report": zh(r"\u5c1a\u672a\u57f7\u884c\u6293\u884c\u60c5"),
    "file": zh(r"\u6a94\u6848"),
    "new_rows": zh(r"\u65b0\u589e\u7b46\u6578"),
    "latest_date": zh(r"\u6700\u5f8c\u65e5\u671f"),
    "rows": zh(r"\u7b46\u6578"),
    "errors": zh(r"\u932f\u8aa4"),
}

TXT.update(
    {
        "top_note": "正式買單使用台股攻擊/推薦融合 H46 乾淨版；攻擊腿明顯勝出才切攻擊，否則用推薦腿。",
        "us_target": zh(r"\u7f8e\u80a1\u6b63\u5f0f\u76ee\u6a19 Top4 \u4e0a\u965025%"),
        "us_target_note": zh(
            r"\u6b63\u5f0f\u7b56\u7565\uff1a\u7f8e\u80a1 Top1 \u52d5\u614b\u5e02\u5834\u7248\uff1b"
            r"\u6bcf\u65e5\u7528 Top800 \u6d41\u52d5\u5e02\u5834\u6c60\u91cd\u7b97\uff0c"
            r"Top1 component \u6c7a\u5b9a\u6301\u5009\u3002"
        ),
        "us_watch_note": zh(
            r"\u6383\u63cf\u699c\u7528\u4f86\u767c\u73fe\u65b0\u4e3b\u7dda\uff1b"
            r"\u7b26\u5408\u51cd\u7d50\u898f\u5247\u624d\u9032\u52d5\u614b\u5e02\u5834\u7b56\u7565\uff0c"
            r"\u4e0d\u662f\u56fa\u5b9a\u5019\u9078\u6c60\u3002"
        ),
        "us_dynamic_formal_pool": zh(r"\u7f8e\u80a1\u52d5\u614b\u5019\u9078\u53c3\u8003"),
        "us_dynamic_formal_pool_note": zh(
            r"\u6b63\u5f0f\u8cb7\u55ae\u5df2\u6539\u7528 Top1 \u52d5\u614b\u5e02\u5834\u7248\uff1b"
            r"\u6b64\u8868\u53ea\u4fdd\u7559\u820a\u5019\u9078\u6a94\u53c3\u8003\uff0c"
            r"\u4e0d\u662f\u73fe\u5728\u7684\u6b63\u5f0f\u8cb7\u55ae\u4f86\u6e90\u3002"
        ),
    }
)


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def read_csv(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    df = pd.read_csv(path, low_memory=False)
    if limit is not None:
        df = df.head(limit)
    return df.where(pd.notna(df), "").to_dict(orient="records")


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def as_float(value: Any) -> float | None:
    try:
        if value == "":
            return None
        return float(value)
    except Exception:
        return None


def as_int(value: Any, default: int = 0) -> int:
    try:
        if value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


def read_current_holdings() -> list[dict[str, Any]]:
    if not HOLDINGS_PATH.exists():
        return []
    df = pd.read_csv(HOLDINGS_PATH, dtype={"market": str, "symbol": str}, low_memory=False)
    for col in HOLDING_COLS:
        if col not in df.columns:
            df[col] = ""
    df = df[HOLDING_COLS].copy()
    df["market"] = df["market"].fillna("").astype(str).str.upper().str.strip()
    df["symbol"] = df["symbol"].fillna("").astype(str).str.strip()
    df["shares"] = pd.to_numeric(df["shares"], errors="coerce").fillna(0.0)
    df["avg_cost"] = pd.to_numeric(df["avg_cost"], errors="coerce")
    df = df[(df["market"].isin(["TW", "US"])) & (df["symbol"] != "") & (df["shares"].abs() > 1e-12)]
    return df.where(pd.notna(df), "").to_dict(orient="records")


def write_current_holdings(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    clean_rows: list[dict[str, Any]] = []
    for row in rows:
        market = str(row.get("market", "")).upper().strip()
        symbol = str(row.get("symbol", "")).strip()
        shares = as_float(row.get("shares")) or 0.0
        if market not in {"TW", "US"} or not symbol or abs(shares) <= 1e-12:
            continue
        avg_cost = as_float(row.get("avg_cost"))
        clean_rows.append(
            {
                "market": market,
                "symbol": symbol,
                "shares": shares,
                "avg_cost": "" if avg_cost is None else avg_cost,
                "note": str(row.get("note", "")),
            }
        )
    df = pd.DataFrame(clean_rows, columns=HOLDING_COLS)
    tmp = HOLDINGS_PATH.with_suffix(".csv.tmp")
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    tmp.replace(HOLDINGS_PATH)
    return clean_rows


def read_current_prices() -> list[dict[str, Any]]:
    if not CURRENT_PRICES_PATH.exists():
        return []
    df = pd.read_csv(CURRENT_PRICES_PATH, dtype={"market": str, "symbol": str}, low_memory=False)
    for col in CURRENT_PRICE_COLS:
        if col not in df.columns:
            df[col] = ""
    df = df[CURRENT_PRICE_COLS].copy()
    df["market"] = df["market"].fillna("").astype(str).str.upper().str.strip()
    df["symbol"] = df["symbol"].fillna("").astype(str).str.strip()
    df["current_price"] = pd.to_numeric(df["current_price"], errors="coerce")
    df = df[(df["market"].isin(["TW", "US"])) & (df["symbol"] != "") & (df["current_price"].gt(0))]
    return df.where(pd.notna(df), "").to_dict(orient="records")


def normalize_current_price_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    clean_rows: list[dict[str, Any]] = []
    now = pd.Timestamp.now().isoformat(timespec="seconds")
    for row in rows:
        market = str(row.get("market", "")).upper().strip()
        symbol = str(row.get("symbol", "")).strip()
        current_price = as_float(row.get("current_price"))
        if market not in {"TW", "US"} or not symbol or current_price is None or current_price <= 0:
            continue
        clean_rows.append(
            {
                "market": market,
                "symbol": symbol,
                "current_price": current_price,
                "price_date": str(row.get("price_date", "") or pd.Timestamp.now().strftime("%Y-%m-%d")),
                "source": str(row.get("source", "") or "manual"),
                "updated_at": str(row.get("updated_at", "") or now),
            }
        )
    return clean_rows


def write_current_prices(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    clean_rows = normalize_current_price_rows(rows)
    df = pd.DataFrame(clean_rows, columns=CURRENT_PRICE_COLS)
    tmp = CURRENT_PRICES_PATH.with_suffix(".csv.tmp")
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    tmp.replace(CURRENT_PRICES_PATH)
    return clean_rows


def merge_current_prices(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    clean_rows = normalize_current_price_rows(rows)
    if not clean_rows:
        return read_current_prices()
    merged = {
        (str(row.get("market", "")).upper(), str(row.get("symbol", ""))): row
        for row in read_current_prices()
    }
    for row in clean_rows:
        merged[(str(row["market"]).upper(), str(row["symbol"]))] = row
    return write_current_prices(list(merged.values()))


def latest_price_map(market: str) -> dict[str, dict[str, Any]]:
    if market == "TW":
        paths = [DATA_DIR / "tw_strategy_prices_tail.csv"]
    else:
        paths = [DATA_DIR / "us_scan_prices_tail.csv", DATA_DIR / "us_execution_prices_tail.csv"]
    frames = []
    for path in paths:
        if not path.exists():
            continue
        try:
            header = pd.read_csv(path, nrows=0)
            available = {str(col) for col in header.columns}
            usecols = ["date", "symbol", "close"] + [col for col in PRICE_CURRENT_KEYS if col in available]
            part = pd.read_csv(
                path,
                usecols=usecols,
                dtype={"symbol": str},
                parse_dates=["date"],
                low_memory=False,
            )
        except Exception:
            continue
        part["close"] = pd.to_numeric(part["close"], errors="coerce")
        for col in PRICE_CURRENT_KEYS:
            if col in part.columns:
                part[col] = pd.to_numeric(part[col], errors="coerce")
        part = part.dropna(subset=["date", "symbol", "close"])
        if not part.empty:
            frames.append(part)
    if not frames:
        return {}
    df = pd.concat(frames, ignore_index=True)
    latest = df.sort_values(["symbol", "date"]).groupby("symbol", as_index=False).tail(1)
    out: dict[str, dict[str, Any]] = {}
    for row in latest.to_dict(orient="records"):
        item = {"close": float(row["close"]), "date": pd.Timestamp(row["date"]).strftime("%Y-%m-%d")}
        for col in PRICE_CURRENT_KEYS:
            value = as_float(row.get(col))
            if value is not None and value > 0:
                item[col] = value
        out[str(row["symbol"])] = item
    for row in read_current_prices():
        if str(row.get("market", "")).upper() != market:
            continue
        symbol = str(row.get("symbol", "")).strip()
        current_price = as_float(row.get("current_price"))
        if not symbol or current_price is None or current_price <= 0:
            continue
        item = out.setdefault(symbol, {"close": current_price, "date": str(row.get("price_date", ""))})
        item["current_price"] = current_price
        item["live_price"] = current_price
        item["current_price_date"] = str(row.get("price_date", "") or item.get("date", ""))
        item["current_price_source"] = str(row.get("source", "") or "manual")
    return out


def price_history_frame(market: str, symbols: set[str]) -> pd.DataFrame:
    if not symbols:
        return pd.DataFrame(columns=["date", "symbol", "close"])
    if market == "TW":
        paths = [DATA_DIR / "tw_strategy_prices_tail.csv"]
    else:
        paths = [DATA_DIR / "us_scan_prices_tail.csv", DATA_DIR / "us_execution_prices_tail.csv"]
    frames = []
    for path in paths:
        if not path.exists():
            continue
        try:
            part = pd.read_csv(
                path,
                usecols=["date", "symbol", "close"],
                dtype={"symbol": str},
                parse_dates=["date"],
                low_memory=False,
            )
        except Exception:
            continue
        part = part[part["symbol"].isin(symbols)].copy()
        part["close"] = pd.to_numeric(part["close"], errors="coerce")
        part = part.dropna(subset=["date", "symbol", "close"])
        if not part.empty:
            frames.append(part)
    if not frames:
        return pd.DataFrame(columns=["date", "symbol", "close"])
    df = pd.concat(frames, ignore_index=True)
    return df.sort_values(["symbol", "date"]).drop_duplicates(["symbol", "date"], keep="last")


def price_at_or_before(
    history: pd.DataFrame,
    symbol: str,
    date_text: Any,
    fallback_close: Any,
    fallback_date: Any,
) -> dict[str, Any]:
    close = as_float(fallback_close)
    date_out = "" if fallback_date is None else str(fallback_date)
    try:
        target_date = pd.Timestamp(date_text)
    except Exception:
        return {"close": close, "date": date_out}
    if history.empty:
        return {"close": close, "date": date_out}
    rows = history[(history["symbol"] == symbol) & (history["date"] <= target_date)]
    if rows.empty:
        return {"close": close, "date": date_out}
    row = rows.tail(1).iloc[0]
    return {"close": float(row["close"]), "date": pd.Timestamp(row["date"]).strftime("%Y-%m-%d")}


def read_entry_baselines() -> list[dict[str, Any]]:
    if not ENTRY_BASELINE_PATH.exists():
        return []
    df = pd.read_csv(ENTRY_BASELINE_PATH, dtype={"market": str, "symbol": str}, low_memory=False)
    for col in ENTRY_BASELINE_COLS:
        if col not in df.columns:
            df[col] = ""
    df = df[ENTRY_BASELINE_COLS].copy()
    df["market"] = df["market"].fillna("").astype(str).str.upper().str.strip()
    df["symbol"] = df["symbol"].fillna("").astype(str).str.strip()
    df["signal_close"] = pd.to_numeric(df["signal_close"], errors="coerce")
    df = df[(df["market"].isin(["TW", "US"])) & (df["symbol"] != "") & df["signal_close"].gt(0)]
    return df.where(pd.notna(df), "").to_dict(orient="records")


def write_entry_baselines(rows: list[dict[str, Any]]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=ENTRY_BASELINE_COLS)
    tmp = ENTRY_BASELINE_PATH.with_suffix(".csv.tmp")
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    tmp.replace(ENTRY_BASELINE_PATH)


def load_entry_rules() -> dict[str, dict[str, float]]:
    rules = {market: dict(vals) for market, vals in ENTRY_RULE_DEFAULTS.items()}
    cap_candidates = [
        OUTPUT_DIR / "ENTRY_GAP_THRESHOLD_BACKTEST" / "ENTRY_GAP_THRESHOLD_SUMMARY.json",
        BASE_DIR.parent / "FINAL_RULE_ANALYSIS_REPORTS" / "ENTRY_GAP_THRESHOLD_BACKTEST" / "ENTRY_GAP_THRESHOLD_SUMMARY.json",
    ]
    for path in cap_candidates:
        data = read_json(path, {})
        recs = data.get("recommendations", {}) if isinstance(data, dict) else {}
        if not isinstance(recs, dict):
            continue
        for market in ["TW", "US"]:
            cap = as_float((recs.get(market) or {}).get("entry_gap_cap_pct"))
            if cap is not None and cap > 0:
                rules[market]["full"] = cap
        break
    floor_candidates = [
        OUTPUT_DIR / "ENTRY_GAP_THRESHOLD_BACKTEST" / "ENTRY_DROP_FLOOR_SUMMARY.json",
        BASE_DIR.parent / "FINAL_RULE_ANALYSIS_REPORTS" / "ENTRY_GAP_THRESHOLD_BACKTEST" / "ENTRY_DROP_FLOOR_SUMMARY.json",
    ]
    for path in floor_candidates:
        data = read_json(path, {})
        recs = data.get("recommendations", {}) if isinstance(data, dict) else {}
        if not isinstance(recs, dict):
            continue
        for market in ["TW", "US"]:
            floor = as_float((recs.get(market) or {}).get("entry_gap_floor_pct"))
            if floor is not None:
                rules[market]["floor"] = floor
        break
    for market, locked in ENTRY_RULE_LOCKS.items():
        rules.setdefault(market, {}).update(locked)
    return rules


def sync_entry_baselines(
    all_rows_by_market: dict[str, list[dict[str, Any]]],
    signal_dates: dict[str, Any],
) -> dict[tuple[str, str], dict[str, Any]]:
    existing = {
        (str(row.get("market", "")).upper(), str(row.get("symbol", ""))): row
        for row in read_entry_baselines()
    }
    out_rows: list[dict[str, Any]] = []
    now = pd.Timestamp.now().isoformat(timespec="seconds")
    reset_on_signal_change = {
        item.strip().upper()
        for item in os.getenv("ENTRY_BASELINE_RESET_ON_SIGNAL_CHANGE_MARKETS", "TW,US").split(",")
        if item.strip()
    }
    for market, rows in all_rows_by_market.items():
        market_signal_date = str(signal_dates.get(market, "") or "")
        symbols = {
            str(row.get("symbol", ""))
            for row in rows
            if str(row.get("symbol", "")).upper() != "CASH" and (as_float(row.get("target_weight")) or 0.0) > 1e-12
        }
        history = price_history_frame(market, symbols)
        for row in rows:
            symbol = str(row.get("symbol", ""))
            if symbol.upper() == "CASH" or (as_float(row.get("target_weight")) or 0.0) <= 1e-12:
                continue
            key = (market, symbol)
            base = existing.get(key)
            base_source_signal_date = str(base.get("source_signal_date", "") or base.get("signal_date", "")) if base else ""
            should_reset = (
                market in reset_on_signal_change
                and market_signal_date
                and base_source_signal_date != market_signal_date
            ) if base else False
            if base and not should_reset:
                out_rows.append(
                    {
                        "market": market,
                        "symbol": symbol,
                        "signal_date": base.get("signal_date", ""),
                        "signal_close": base.get("signal_close", ""),
                        "source_signal_date": base.get("source_signal_date", "") or base.get("signal_date", ""),
                        "baseline_type": base.get("baseline_type", "") or "strategy_signal_close",
                        "created_at": base.get("created_at", ""),
                    }
                )
                continue
            signal = price_at_or_before(
                history,
                symbol,
                signal_dates.get(market) or row.get("price_date", ""),
                row.get("close", ""),
                row.get("price_date", ""),
            )
            if as_float(signal.get("close")) is None:
                continue
            out_rows.append(
                {
                    "market": market,
                    "symbol": symbol,
                    "signal_date": signal.get("date", ""),
                    "signal_close": signal.get("close", ""),
                    "source_signal_date": market_signal_date or signal.get("date", ""),
                    "baseline_type": "strategy_signal_close",
                    "created_at": now,
                }
            )
    write_entry_baselines(out_rows)
    return {(str(row["market"]), str(row["symbol"])): row for row in out_rows}


def current_price_for_entry(row: dict[str, Any]) -> float | None:
    for key in PRICE_CURRENT_KEYS + ["close"]:
        value = as_float(row.get(key))
        if value is not None and value > 0:
            return value
    return None


def entry_status_for_row(row: dict[str, Any], market: str, baseline: dict[str, Any] | None, rules: dict[str, dict[str, float]]) -> dict[str, Any]:
    symbol = str(row.get("symbol", ""))
    if symbol.upper() == "CASH":
        return {"entry_status": "保留現金", "entry_class": "entry-neutral", "entry_note": "", "entry_gap_pct": ""}
    target_weight = as_float(row.get("target_weight")) or 0.0
    diff = as_float(row.get("diff_shares"))
    threshold = 0.5 if market == "TW" else 0.00005
    if target_weight <= 1e-12:
        return {"entry_status": "非目標", "entry_class": "entry-neutral", "entry_note": "不在正式目標，不新買。", "entry_gap_pct": ""}
    if diff is not None and diff <= threshold:
        return {"entry_status": "不用新買", "entry_class": "entry-neutral", "entry_note": "已持有或不需加碼，照換倉規則。", "entry_gap_pct": ""}
    current_price = current_price_for_entry(row)
    signal_close = as_float((baseline or {}).get("signal_close"))
    signal_date = (baseline or {}).get("signal_date", "")
    if current_price is None or signal_close is None or signal_close <= 0:
        return {"entry_status": "缺基準價", "entry_class": "entry-warn", "entry_note": "缺少訊號收盤價，不能判斷追高。", "entry_gap_pct": ""}
    gap_pct = (current_price / signal_close - 1.0) * 100.0
    rule = rules.get(market, ENTRY_RULE_DEFAULTS[market])
    buy_lower_price = signal_close * (1.0 + float(rule["floor"]) / 100.0)
    buy_upper_price = signal_close * (1.0 + float(rule["full"]) / 100.0)
    partial_upper_price = signal_close * (1.0 + float(rule["partial"]) / 100.0)
    hard_upper_price = signal_close * (1.0 + float(rule["hard"]) / 100.0)
    if gap_pct < float(rule["floor"]):
        status, cls = "跌破不買", "entry-danger"
    elif gap_pct <= float(rule["full"]):
        status, cls = "可買滿", "entry-ok"
    elif gap_pct <= float(rule["partial"]):
        status, cls = "分批", "entry-warn"
    elif gap_pct <= float(rule["hard"]):
        status, cls = "等拉回", "entry-warn"
    else:
        status, cls = "禁止新買", "entry-danger"
    note = f"盤中/最新價較訊號收盤 {gap_pct:+.2f}%；允許 {float(rule['floor']):+.0f}% 到 +{float(rule['full']):.0f}% 買滿；訊號 {signal_date} @ {signal_close:.2f}"
    return {
        "entry_status": status,
        "entry_class": cls,
        "entry_note": note,
        "entry_gap_pct": gap_pct,
        "entry_current_price": current_price,
        "entry_signal_date": signal_date,
        "entry_signal_close": signal_close,
        "buy_lower_price": buy_lower_price,
        "buy_upper_price": buy_upper_price,
        "partial_upper_price": partial_upper_price,
        "hard_upper_price": hard_upper_price,
        "tracking_base_date": signal_date,
        "tracking_base_price": signal_close,
        "tracking_return_pct": gap_pct,
        "entry_floor_pct": float(rule["floor"]),
        "entry_full_cap_pct": float(rule["full"]),
        "entry_partial_cap_pct": float(rule["partial"]),
        "entry_hard_cap_pct": float(rule["hard"]),
    }


def apply_entry_status(
    rows: list[dict[str, Any]],
    market: str,
    baselines: dict[tuple[str, str], dict[str, Any]],
    rules: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        rr = dict(row)
        rr.update(entry_status_for_row(rr, market, baselines.get((market, str(rr.get("symbol", "")))), rules))
        out.append(rr)
    return out


def entry_baseline_date_text(
    baselines: dict[tuple[str, str], dict[str, Any]],
    market: str,
) -> str:
    dates = sorted(
        {
            str(row.get("signal_date", ""))
            for (row_market, _symbol), row in baselines.items()
            if row_market == market and str(row.get("signal_date", ""))
        }
    )
    if not dates:
        return ""
    if len(dates) == 1:
        return dates[0]
    return f"{dates[0]}~{dates[-1]}"


def holdings_map(holdings: list[dict[str, Any]], market: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for row in holdings:
        if str(row.get("market", "")).upper() != market:
            continue
        symbol = str(row.get("symbol", "")).strip()
        shares = as_float(row.get("shares")) or 0.0
        if symbol:
            out[symbol] = out.get(symbol, 0.0) + shares
    return out


def build_order_rows(
    target_rows: list[dict[str, Any]],
    market: str,
    price_map: dict[str, dict[str, Any]],
    hold_map: dict[str, float],
    capital: float,
) -> list[dict[str, Any]]:
    targets = {str(row.get("symbol", "")): dict(row) for row in target_rows if str(row.get("symbol", ""))}
    symbols = [str(row.get("symbol", "")) for row in target_rows if str(row.get("symbol", ""))]
    for symbol in hold_map:
        if symbol not in targets:
            symbols.append(symbol)
            targets[symbol] = {"symbol": symbol, "target_weight": 0.0, "role": "not_in_target", "theme": ""}
    rows: list[dict[str, Any]] = []
    for symbol in dict.fromkeys(symbols):
        row = targets.get(symbol, {})
        price = price_map.get(symbol, {})
        close = as_float(price.get("close"))
        sizing_price = current_price_for_entry(price) or close
        current_fields = {key: price.get(key, "") for key in PRICE_CURRENT_KEYS if key in price}
        weight = as_float(row.get("target_weight")) or 0.0
        target_amount = capital * weight
        if symbol.upper() == "CASH":
            target_shares = ""
        elif sizing_price and sizing_price > 0:
            target_shares = int(target_amount // sizing_price) if market == "TW" else target_amount / sizing_price
        else:
            target_shares = ""
        current_shares = float(hold_map.get(symbol, 0.0))
        diff_shares = "" if target_shares == "" else float(target_shares) - current_shares
        if target_shares == "":
            action = "保留現金"
        elif abs(float(diff_shares)) < (1e-9 if market == "TW" else 0.00005):
            action = "持有"
        elif current_shares <= 1e-12 and float(target_shares) > 0:
            action = "新增"
        elif float(target_shares) <= 1e-12 and current_shares > 0:
            action = "清倉"
        elif float(diff_shares) > 0:
            action = "買進"
        else:
            action = "賣出"
        rows.append(
            {
                **row,
                **current_fields,
                "market": market,
                "symbol": symbol,
                "close": "" if close is None else close,
                "price_date": price.get("date", ""),
                "current_price_date": price.get("current_price_date", ""),
                "current_price_source": price.get("current_price_source", ""),
                "target_amount": target_amount,
                "target_shares": target_shares,
                "current_shares": current_shares,
                "diff_shares": diff_shares,
                "action": action,
                "currency": "TWD" if market == "TW" else "USD",
            }
        )
    return rows


def write_order_plan(rows: list[dict[str, Any]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cols = [
        "market",
        "symbol",
        "name",
        "theme",
        "target_weight",
        "close",
        *PRICE_CURRENT_KEYS,
        "price_date",
        "current_price_date",
        "current_price_source",
        "target_amount",
        "target_shares",
        "current_shares",
        "diff_shares",
        "action",
        "entry_status",
        "entry_gap_pct",
        "entry_current_price",
        "entry_signal_date",
        "entry_signal_close",
        "buy_lower_price",
        "buy_upper_price",
        "partial_upper_price",
        "hard_upper_price",
        "tracking_base_date",
        "tracking_base_price",
        "tracking_return_pct",
        "entry_floor_pct",
        "entry_full_cap_pct",
        "entry_partial_cap_pct",
        "entry_hard_cap_pct",
    ]
    df = pd.DataFrame(rows)
    for col in cols:
        if col not in df.columns:
            df[col] = ""
    df[cols].to_csv(ORDER_PLAN_PATH, index=False, encoding="utf-8-sig")


def fmt_weight(value: Any) -> str:
    number = as_float(value)
    if number is None:
        return ""
    return f"{number * 100:.2f}%"


def fmt_pct(value: Any) -> str:
    number = as_float(value)
    if number is None:
        return ""
    return f"{number:.2f}%"


def fmt_num(value: Any, digits: int = 2) -> str:
    number = as_float(value)
    if number is None:
        return esc(value)
    return f"{number:.{digits}f}"


def fmt_money_value(value: Any, currency: str) -> str:
    number = as_float(value)
    if number is None:
        return ""
    prefix = "NT$ " if currency == "TWD" else "$ "
    return prefix + f"{number:,.0f}"


def fmt_compact_usd(value: Any) -> str:
    number = as_float(value)
    if number is None:
        return ""
    if abs(number) >= 1_000_000_000:
        return f"${number / 1_000_000_000:.2f}B"
    if abs(number) >= 1_000_000:
        return f"${number / 1_000_000:.1f}M"
    return f"${number:,.0f}"


def fmt_shares(value: Any, market: str) -> str:
    number = as_float(value)
    if number is None:
        return ""
    if market == "TW":
        return f"{int(round(number)):,}"
    return f"{number:,.4f}".rstrip("0").rstrip(".")


def action_class(action: Any) -> str:
    raw = str(action)
    if raw in {"新增", "買進"}:
        return "buy-action"
    if raw in {"清倉", "賣出"}:
        return "sell-action"
    return "hold-action"


def label_theme(theme: Any) -> str:
    raw = "" if theme is None else str(theme)
    zh_label = THEME_LABELS.get(raw, raw)
    if raw and zh_label != raw:
        return f"{esc(zh_label)}<span class='raw'>{esc(raw)}</span>"
    return esc(raw)


def label_role(role: Any) -> str:
    raw = "" if role is None else str(role)
    if "|" not in raw:
        return ROLE_LABELS.get(raw, raw)
    parts = [part for part in raw.split("|") if part]
    labels = []
    for part in parts:
        if part == "equal_weight_exec":
            labels.append("等權執行")
        elif part == "top6_cap25_exec":
            labels.append("Top6 上限25%")
        elif part == "top4_cap25_exec":
            labels.append("Top4 上限25%")
        else:
            labels.append(ROLE_LABELS.get(part, part))
    return " / ".join(labels)


def label_exec_group(value: Any) -> str:
    raw = "" if value is None else str(value)
    if raw == "main":
        return TXT["main_position"]
    if raw == "micro":
        return TXT["micro_position"]
    return raw


def bool_text(value: Any) -> str:
    return TXT["yes"] if str(value).lower() == "true" or value is True else TXT["no"]


def format_cell(key: str, value: Any) -> str:
    if isinstance(value, bool):
        return bool_text(value)
    if key == "target_weight" or key.endswith("_weight"):
        return fmt_weight(value)
    if key.endswith("_pct"):
        return fmt_pct(value)
    if key.endswith("score") or key in {"theme_score", "scan_score"}:
        return fmt_num(value, 2)
    if key in {"dollar_volume20", "median_dollar_volume_60d"}:
        return esc(fmt_compact_usd(value))
    if key in {"theme", "scan_theme"}:
        return label_theme(value)
    if key == "role":
        return esc(label_role(value))
    if key == "execution_group":
        return esc(label_exec_group(value))
    if key in {"selected", "eligible", "in_execution_pool", "selected_today"}:
        return bool_text(value)
    return esc(value)


def pill(text: str, kind: str = "neutral") -> str:
    return f"<span class='pill {kind}'>{html.escape(text)}</span>"


def state_value(rows: list[dict[str, Any]], key: str, default: Any = "") -> Any:
    if not rows:
        return default
    return rows[0].get(key, default)


def target_table(rows: list[dict[str, Any]], market: str, currency: str) -> str:
    if not rows:
        return f"<div class='empty'>{TXT['no_target']}</div>"
    body = []
    for row in rows:
        symbol = str(row.get("symbol", ""))
        weight = as_float(row.get("target_weight")) or 0.0
        close = current_price_for_entry(row) or 0.0
        action = row.get("action", "")
        entry_status = row.get("entry_status", "")
        entry_class = row.get("entry_class", "entry-neutral")
        entry_gap = as_float(row.get("entry_gap_pct"))
        entry_note = row.get("entry_note", "")
        entry_floor = as_float(row.get("entry_floor_pct"))
        entry_full = as_float(row.get("entry_full_cap_pct"))
        entry_partial = as_float(row.get("entry_partial_cap_pct"))
        entry_hard = as_float(row.get("entry_hard_cap_pct"))
        price_date = row.get("price_date", "")
        is_cash = symbol.upper() == "CASH"
        name = row.get("name", "")
        name_line = f"<span class='name'>{esc(name)}</span>" if name else ""
        row_class = "cash-row" if is_cash else ""
        source = row.get("source_sleeve", "")
        source_line = f"<span class='raw'>{esc(source)}</span>" if source else ""
        price_cell = "" if is_cash or close <= 0 else f"{fmt_num(close, 2)}<span class='raw'>{esc(price_date)}</span>"
        role_cell = f"<span class='role-chip'>{esc(label_role(row.get('role', '')))}</span>{source_line}"
        body.append(
            f"<tr class='target-row {row_class}' data-market='{market}' data-symbol='{esc(symbol)}' "
            f"data-weight='{weight:.10f}' data-close='{close:.10f}' "
            f"data-entry-gap='{'' if entry_gap is None else f'{entry_gap:.6f}'}' "
            f"data-entry-floor='{'' if entry_floor is None else f'{entry_floor:.6f}'}' "
            f"data-entry-full='{'' if entry_full is None else f'{entry_full:.6f}'}' "
            f"data-entry-partial='{'' if entry_partial is None else f'{entry_partial:.6f}'}' "
            f"data-entry-hard='{'' if entry_hard is None else f'{entry_hard:.6f}'}'>"
            f"<td data-label='標的'><span class='symbol'>{esc(symbol)}</span>{name_line}</td>"
            f"<td data-label='入場'><span class='entry-chip {esc(entry_class)}'>{esc(entry_status)}</span><span class='raw entry-note'>{esc(entry_note)}</span></td>"
            f"<td data-label='權重' class='weight'>{fmt_weight(row.get('target_weight'))}</td>"
            f"<td data-label='金額' class='amount' data-currency='{currency}'>{fmt_money_value(row.get('target_amount'), currency)}</td>"
            f"<td data-label='現價'>{price_cell}</td>"
            f"<td data-label='主題' class='theme-cell'>{label_theme(row.get('theme', ''))}</td>"
            f"<td data-label='定位'>{role_cell}</td>"
            "</tr>"
        )
    return (
        "<table class='target-table'><thead><tr>"
        f"<th>{TXT['symbol']}</th><th>入場判斷</th><th>{TXT['target_weight']}</th><th>{TXT['amount_calc']}</th><th>現價</th>"
        f"<th>{TXT['theme']}</th><th>{TXT['role']}</th>"
        f"</tr></thead><tbody>{''.join(body)}</tbody></table>"
    )


def entry_board(rows: list[dict[str, Any]], market: str, currency: str) -> str:
    cards = []
    for row in rows:
        symbol = str(row.get("symbol", ""))
        if not symbol or symbol.upper() == "CASH":
            continue
        entry_status = str(row.get("entry_status", ""))
        entry_class = str(row.get("entry_class", "entry-neutral"))
        entry_gap = as_float(row.get("entry_gap_pct"))
        gap_text = "" if entry_gap is None else f"{entry_gap:+.2f}%"
        signal_date = str(row.get("entry_signal_date", ""))
        signal_close = as_float(row.get("entry_signal_close"))
        signal_text = ""
        if signal_date and signal_close is not None:
            signal_text = f"訊號 {signal_date} @ {signal_close:.2f}"
        amount = fmt_money_value(row.get("target_amount"), currency)
        cards.append(
            f"<article class='entry-card {esc(entry_class)}'>"
            f"<div class='entry-card-top'><span class='symbol'>{esc(symbol)}</span><span class='entry-chip {esc(entry_class)}'>{esc(entry_status)}</span></div>"
            f"<div class='entry-card-main'>{esc(gap_text)}</div>"
            f"<div class='entry-card-meta'><span>{fmt_weight(row.get('target_weight'))}</span><span>{amount}</span></div>"
            f"<div class='entry-card-note'>{esc(signal_text)}</div>"
            "</article>"
        )
    if not cards:
        return f"<div class='empty'>{TXT['no_target']}</div>"
    return "<div class='entry-board'>" + "".join(cards) + "</div>"


def compact_targets(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return f"<div class='empty'>{TXT['no_order']}</div>"
    items = []
    for row in rows:
        if str(row.get("execution_group", "main")) == "micro":
            continue
        symbol = str(row.get("symbol", ""))
        entry_status = str(row.get("entry_status", ""))
        kind = "cash" if symbol.upper() == "CASH" else str(row.get("entry_class", "entry-neutral"))
        items.append(
            f"<span class='target-chip {kind}'><strong>{esc(symbol)}</strong><em>{fmt_weight(row.get('target_weight'))}</em><small>{esc(entry_status)}</small></span>"
        )
    if not items:
        return f"<div class='empty'>{TXT['no_order']}</div>"
    return "<div class='target-strip'>" + "".join(items) + "</div>"


def entry_headline(rows: list[dict[str, Any]]) -> tuple[str, str]:
    real_rows = [row for row in rows if str(row.get("symbol", "")).upper() != "CASH"]
    ok = sum(str(row.get("entry_status", "")) == "可買滿" for row in real_rows)
    warn = sum(str(row.get("entry_status", "")) in {"分批", "等拉回"} for row in real_rows)
    danger = sum(str(row.get("entry_status", "")) in {"禁止新買", "跌破不買"} for row in real_rows)
    if ok and not warn and not danger:
        return f"可買 {ok} 檔", "ok"
    if ok or warn:
        return f"可買 {ok} / 等 {warn}", "warn"
    if danger:
        return "今天先不買", "danger"
    return "等待訊號", "neutral"


def clean_status_groups(rows: list[dict[str, Any]]) -> str:
    groups: dict[str, list[str]] = {}
    for row in rows:
        symbol = str(row.get("symbol", ""))
        if not symbol or symbol.upper() == "CASH":
            continue
        status = str(row.get("entry_status", ""))
        groups.setdefault(status, []).append(symbol)
    parts = []
    for status in ["禁止新買", "跌破不買", "等拉回", "分批", "可買滿"]:
        symbols = groups.get(status, [])
        if symbols:
            parts.append(f"<div><strong>{esc(status)}</strong><span>{esc('、'.join(symbols))}</span></div>")
    return "<div class='clean-reason'>" + "".join(parts) + "</div>"


def legacy_clean_order_panel(
    rows: list[dict[str, Any]],
    market_label: str,
    market: str,
    currency: str,
    date_text: Any,
    capital_label: str,
    capital_id: str,
    capital_value: float,
    capital_step: int,
    schedule_text: str,
    headline_override: str | None = None,
) -> str:
    headline, headline_kind = entry_headline(rows)
    display_headline = headline_override or headline
    entry_note = f" / 入場：{headline}" if headline_override else ""
    items = []
    for row in rows:
        symbol = str(row.get("symbol", ""))
        if not symbol:
            continue
        is_cash = symbol.upper() == "CASH"
        entry_status = str(row.get("entry_status", ""))
        entry_class = "cash" if is_cash else str(row.get("entry_class", "entry-neutral"))
        name = str(row.get("name", ""))
        name_line = f"<span class='clean-name'>{esc(name)}</span>" if name else ""
        signal_date = str(row.get("entry_signal_date", "") or row.get("tracking_base_date", ""))
        signal_close = as_float(row.get("entry_signal_close"))
        lower = as_float(row.get("buy_lower_price"))
        upper = as_float(row.get("buy_upper_price"))
        close = current_price_for_entry(row)
        gap = as_float(row.get("tracking_return_pct"))
        price_date = str(row.get("price_date", ""))
        signal_text = "-" if is_cash or not signal_date or signal_close is None else f"{signal_date} @ {signal_close:.2f}"
        band_text = "-" if is_cash or lower is None or upper is None else f"{lower:.2f} - {upper:.2f}"
        track_text = "-" if is_cash or close is None or gap is None else f"{close:.2f} / {gap:+.2f}%"
        price_meta = "" if is_cash or not price_date else f"<small class='price-meta'>現價日 {esc(price_date)}</small>"
        price_input = (
            ""
            if is_cash or close is None
            else (
                f"<input class='live-price-input' data-market='{market}' data-symbol='{esc(symbol)}' "
                f"type='number' min='0' step='0.01' inputmode='decimal' value='{close:.2f}' aria-label='{esc(symbol)} 現價'>"
            )
        )
        items.append(
            f"<article class='clean-row target-row {esc(entry_class)}' data-market='{market}' data-symbol='{esc(symbol)}' "
            f"data-weight='{(as_float(row.get('target_weight')) or 0.0):.10f}'>"
            "<div class='clean-symbol'>"
            f"<strong>{esc(symbol)}</strong>{name_line}"
            "</div>"
            f"<span class='clean-status {esc(entry_class)}'>{esc(entry_status)}</span>"
            f"<div class='track-cell'><span>進場基準日/價</span><strong>{esc(signal_text)}</strong></div>"
            f"<div class='track-cell buy-band'><span>買入區間</span><strong>{esc(band_text)}</strong></div>"
            f"<div class='track-cell live-price-cell'><span>現價/偏離</span><strong>{esc(track_text)}</strong>{price_meta}{price_input}</div>"
            f"<div class='track-cell'><span>權重/金額</span><strong><em>{fmt_weight(row.get('target_weight'))}</em><b class='amount clean-amount' data-currency='{currency}'>{fmt_money_value(row.get('target_amount'), currency)}</b></strong></div>"
            "</article>"
        )
    list_html = f"<div class='clean-list'>{''.join(items)}</div>"
    return (
        f"<section class='clean-panel {headline_kind}'>"
        "<div class='clean-head'>"
        "<div>"
        f"<div class='clean-market'>{esc(market_label)}</div>"
        f"<h2>{esc(display_headline)}</h2>"
        f"<p>資料日 {esc(date_text)} / {esc(schedule_text)}{esc(entry_note)}</p>"
        "</div>"
        f"<label class='clean-capital'><span>{esc(capital_label)}</span><input id='{capital_id}' type='number' min='0' step='{capital_step}' value='{capital_value:.0f}'></label>"
        "</div>"
        f"{list_html}"
        "</section>"
    )


def clean_order_panel(
    rows: list[dict[str, Any]],
    market_label: str,
    market: str,
    currency: str,
    date_text: Any,
    capital_label: str,
    capital_id: str,
    capital_value: float,
    capital_step: int,
    schedule_text: str,
    headline_override: str | None = None,
) -> str:
    headline, headline_kind = entry_headline(rows)
    display_headline = headline_override or headline
    entry_suffix = f" / {headline}" if headline_override else ""
    cards: list[str] = []
    label_signal = zh(r"\u9032\u5834\u57fa\u6e96\u65e5/\u50f9")
    label_band = zh(r"\u8cb7\u5165\u5340\u9593")
    label_current = zh(r"\u73fe\u50f9/\u504f\u96e2")
    label_weight = zh(r"\u6b0a\u91cd")
    label_amount = zh(r"\u76ee\u6a19\u91d1\u984d")
    label_adjust = zh(r"\u73fe\u50f9\u6821\u6b63")
    label_data = zh(r"\u8cc7\u6599\u65e5")

    for row in rows:
        symbol = str(row.get("symbol", ""))
        if not symbol:
            continue
        is_cash = symbol.upper() == "CASH"
        entry_status = str(row.get("entry_status", ""))
        entry_class = "cash" if is_cash else str(row.get("entry_class", "entry-neutral"))
        name = str(row.get("name", ""))
        name_html = f"<span>{esc(name)}</span>" if name else ""
        signal_date = str(row.get("entry_signal_date", "") or row.get("tracking_base_date", ""))
        signal_close = as_float(row.get("entry_signal_close"))
        lower = as_float(row.get("buy_lower_price"))
        upper = as_float(row.get("buy_upper_price"))
        close = current_price_for_entry(row)
        gap = as_float(row.get("tracking_return_pct"))
        price_date = str(row.get("price_date", ""))
        current_price_date = str(row.get("current_price_date", "") or price_date)
        current_price_source = str(row.get("current_price_source", ""))
        if current_price_source == "manual_ui" and current_price_date and current_price_date != price_date:
            price_meta = f"校正價 {current_price_date} / 收盤日 {price_date}"
        elif current_price_source == "manual_ui":
            price_meta = f"校正價 {current_price_date or price_date}"
        elif current_price_source.endswith("_intraday"):
            price_meta = f"盤中/最新 {current_price_date or price_date}"
        else:
            price_meta = f"收盤日 {price_date}"
        signal_text = "-" if is_cash or not signal_date or signal_close is None else f"{signal_date} @ {signal_close:.2f}"
        band_text = "-" if is_cash or lower is None or upper is None else f"{lower:.2f} - {upper:.2f}"
        current_text = "-" if is_cash or close is None else f"{close:.2f}"
        gap_text = "" if is_cash or gap is None else f"{gap:+.2f}%"
        price_input = ""
        if not is_cash and close is not None:
            price_input = (
                "<details class='trade-price-edit'>"
                f"<summary>{label_adjust}</summary>"
                f"<input class='live-price-input' data-market='{market}' data-symbol='{esc(symbol)}' "
                f"type='number' min='0' step='0.01' inputmode='decimal' value='{close:.2f}' aria-label='{esc(symbol)} current price'>"
                "</details>"
            )
        cards.append(
            f"<article class='trade-card target-row {esc(entry_class)}' data-market='{market}' data-symbol='{esc(symbol)}' "
            f"data-weight='{(as_float(row.get('target_weight')) or 0.0):.10f}'>"
            "<div class='trade-card-head'>"
            f"<div class='trade-symbol'><strong>{esc(symbol)}</strong>{name_html}</div>"
            f"<span class='trade-status {esc(entry_class)}'>{esc(entry_status)}</span>"
            "</div>"
            "<div class='trade-facts'>"
            f"<div class='trade-fact'><span>{label_signal}</span><strong>{esc(signal_text)}</strong></div>"
            f"<div class='trade-fact trade-band'><span>{label_band}</span><strong>{esc(band_text)}</strong></div>"
            f"<div class='trade-fact'><span>{label_current}</span><strong>{esc(current_text)} <em>{esc(gap_text)}</em></strong><small>{esc(price_meta)}</small></div>"
            f"<div class='trade-fact'><span>{label_weight}</span><strong>{fmt_weight(row.get('target_weight'))}</strong></div>"
            f"<div class='trade-fact trade-amount-box'><span>{label_amount}</span><strong class='amount' data-currency='{currency}'>{fmt_money_value(row.get('target_amount'), currency)}</strong></div>"
            "</div>"
            f"{price_input}"
            "</article>"
        )

    list_html = "<div class='trade-list'>" + "".join(cards) + "</div>" if cards else f"<div class='empty'>{TXT['no_target']}</div>"
    return (
        f"<section class='clean-panel {headline_kind}'>"
        "<div class='clean-head'>"
        "<div>"
        f"<div class='clean-market'>{esc(market_label)}</div>"
        f"<h2>{esc(display_headline)}</h2>"
        f"<p>{label_data} {esc(date_text)} / {esc(schedule_text)}{esc(entry_suffix)}</p>"
        "</div>"
        f"<label class='clean-capital'><span>{esc(capital_label)}</span><input id='{capital_id}' type='number' min='0' step='{capital_step}' value='{capital_value:.0f}'></label>"
        "</div>"
        f"{list_html}"
        "</section>"
    )


def simple_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]], empty: str | None = None) -> str:
    if not rows:
        return f"<div class='empty'>{esc(empty or TXT['no_data'])}</div>"
    head = "".join(f"<th>{esc(label)}</th>" for _key, label in columns)
    body_rows = []
    for row in rows:
        cells = [f"<td>{format_cell(key, row.get(key, ''))}</td>" for key, _label in columns]
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def watch_table(rows: list[dict[str, Any]], limit: int = 12, target_symbols: set[str] | None = None) -> str:
    if not rows:
        return f"<div class='empty'>{TXT['no_watch']}</div>"
    target_symbols = target_symbols or set()
    body = []
    for row in rows[:limit]:
        symbol = str(row.get("symbol", ""))
        in_target = symbol in target_symbols
        in_pool = str(row.get("in_execution_pool", "")).lower() == "true"
        badge = pill(zh(r"\u6b63\u5f0f\u8cb7\u55ae"), "ok") if in_target else (pill(TXT["official_pool"], "ok") if in_pool else pill(TXT["watch"], "watch"))
        body.append(
            "<tr>"
            f"<td>{esc(row.get('rank', ''))}</td>"
            f"<td><span class='symbol'>{esc(symbol)}</span></td>"
            f"<td>{label_theme(row.get('scan_theme', row.get('theme', '')))}</td>"
            f"<td>{fmt_num(row.get('scan_score', ''), 2)}</td>"
            f"<td>{fmt_pct(row.get('mom63_skip5_pct', ''))}</td>"
            f"<td>{badge}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr>"
        f"<th>#</th><th>{TXT['stock']}</th><th>{TXT['theme']}</th><th>{TXT['score']}</th>"
        f"<th>63D</th><th>{TXT['status']}</th>"
        f"</tr></thead><tbody>{''.join(body)}</tbody></table>"
    )


def theme_cards(rows: list[dict[str, Any]], theme_key: str, score_key: str, symbol_key: str, limit: int = 6) -> str:
    if not rows:
        return f"<div class='empty'>{TXT['no_theme']}</div>"
    cards = []
    for row in rows[:limit]:
        selected = str(row.get("selected", "")).lower() == "true" or row.get("selected") is True
        cards.append(
            "<article class='theme-tile'>"
            f"<div class='theme-rank'>#{esc(row.get('rank', ''))}</div>"
            f"<div class='theme-name'>{label_theme(row.get(theme_key, ''))}</div>"
            f"<div class='theme-score'>{fmt_num(row.get(score_key, ''), 2)}</div>"
            f"<div class='theme-symbols'>{esc(row.get(symbol_key, ''))}</div>"
            f"{pill(TXT['included'], 'ok') if selected else ''}"
            "</article>"
        )
    return "<div class='theme-grid'>" + "".join(cards) + "</div>"


def price_update_table(report: dict[str, Any]) -> str:
    if not report:
        return f"<div class='empty'>{TXT['price_no_report']}</div>"
    rows = []
    for item in report.get("files", []):
        after = item.get("after") if isinstance(item.get("after"), dict) else {}
        rows.append(
            {
                "label": item.get("label", ""),
                "status": item.get("status", ""),
                "new_rows": item.get("new_rows", 0),
                "max_date": after.get("max_date", ""),
                "symbols": after.get("symbols", ""),
                "error_count": item.get("error_count", 0),
            }
        )
    if not rows:
        rows = [
            {
                "label": "PRICE_UPDATE",
                "status": report.get("status", ""),
                "new_rows": report.get("new_rows_total", 0),
                "max_date": "",
                "symbols": "",
                "error_count": "",
            }
        ]
    return simple_table(
        rows,
        [
            ("label", TXT["file"]),
            ("status", TXT["status"]),
            ("new_rows", TXT["new_rows"]),
            ("max_date", TXT["latest_date"]),
            ("symbols", TXT["unit_symbols"]),
            ("error_count", TXT["errors"]),
        ],
    )


def detail_section(title: str, body: str, subtitle: str = "") -> str:
    sub = f"<p class='section-subtitle'>{esc(subtitle)}</p>" if subtitle else ""
    return f"<section class='section'><div class='section-head'><h2>{esc(title)}</h2>{sub}</div>{body}</section>"


def has_saved_holdings(hold_map: dict[str, float]) -> bool:
    return any(abs(float(v)) > 1e-12 for v in hold_map.values())


def order_diff_summary(rows: list[dict[str, Any]], market: str) -> dict[str, Any]:
    threshold = 0.5 if market == "TW" else 0.00005
    buy_rows: list[dict[str, Any]] = []
    sell_rows: list[dict[str, Any]] = []
    for row in rows:
        symbol = str(row.get("symbol", ""))
        if symbol.upper() == "CASH":
            continue
        diff = as_float(row.get("diff_shares"))
        if diff is None or abs(diff) <= threshold:
            continue
        if diff > 0:
            buy_rows.append(row)
        else:
            sell_rows.append(row)
    return {
        "buy_count": len(buy_rows),
        "sell_count": len(sell_rows),
        "change_count": len(buy_rows) + len(sell_rows),
        "buy_symbols": [str(row.get("symbol", "")) for row in buy_rows[:5]],
        "sell_symbols": [str(row.get("symbol", "")) for row in sell_rows[:5]],
    }


def rebalance_card(
    market_label: str,
    rows: list[dict[str, Any]],
    market: str,
    has_holdings: bool,
    schedule_due: bool | None,
    date_text: Any,
) -> str:
    summary = order_diff_summary(rows, market)
    change_count = int(summary["change_count"])
    if not has_holdings:
        if schedule_due is False:
            status = "觀察不換倉"
            kind = "warn"
            detail = "還沒到策略換倉節奏；畫面只更新行情、權重與入場狀態。"
        else:
            status = "依目標配置"
            kind = "ok"
            detail = "畫面改為乾淨買單模式，只顯示權重、資金與入場狀態，不再顯示數量欄。"
    elif schedule_due is False:
        status = "非換倉日"
        kind = "warn"
        detail = "策略節奏未到，原則上先不主動調倉；只更新行情與觀察。"
    elif change_count == 0:
        status = "不用換倉"
        kind = "ok"
        detail = "目前持倉已接近正式目標。"
    else:
        status = "需要換倉"
        kind = "danger"
        buy = int(summary["buy_count"])
        sell = int(summary["sell_count"])
        buy_symbols = ", ".join(summary["buy_symbols"])
        sell_symbols = ", ".join(summary["sell_symbols"])
        detail = f"買進/加碼 {buy} 檔，賣出/減碼 {sell} 檔。"
        if buy_symbols:
            detail += f" 買：{buy_symbols}。"
        if sell_symbols:
            detail += f" 賣：{sell_symbols}。"
    schedule_text = "固定策略交易日節奏" if schedule_due is None else ("到換倉節奏" if schedule_due else "未到換倉節奏")
    return (
        f"<article class='rebalance-card {kind}'>"
        f"<div class='rebalance-market'>{esc(market_label)}</div>"
        f"<div class='rebalance-status'>{esc(status)}</div>"
        f"<div class='rebalance-detail'>{esc(detail)}</div>"
        f"<div class='rebalance-meta'>資料日 {esc(date_text)} / {esc(schedule_text)}</div>"
        "</article>"
    )


def main() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = read_json(OUTPUT_DIR / "LATEST_DAILY_MARKET_POOL_SUMMARY.json", {})
    price_report = read_json(OUTPUT_DIR / "LATEST_PRICE_UPDATE_REPORT.json", {})
    tw_state = read_csv(OUTPUT_DIR / "TW" / "latest_market_state.csv")
    us_state = read_csv(OUTPUT_DIR / "US" / "latest_market_state.csv")
    tw_target = read_csv(OUTPUT_DIR / "TW" / "latest_target_position.csv")
    us_target = read_csv(OUTPUT_DIR / "US" / "latest_target_position.csv")
    holdings = read_current_holdings()
    tw_price_map = latest_price_map("TW")
    us_price_map = latest_price_map("US")
    tw_hold_map = holdings_map(holdings, "TW")
    us_hold_map = holdings_map(holdings, "US")
    tw_order_rows = build_order_rows(tw_target, "TW", tw_price_map, tw_hold_map, DEFAULT_CAPITAL["TW"])
    us_order_rows = build_order_rows(us_target, "US", us_price_map, us_hold_map, DEFAULT_CAPITAL["US"])
    entry_rules = load_entry_rules()
    entry_baselines = sync_entry_baselines(
        {"TW": tw_order_rows, "US": us_order_rows},
        {
            "TW": state_value(tw_state, "component_signal_date", state_value(tw_state, "latest_price_date")),
            "US": state_value(us_state, "component_signal_date", state_value(us_state, "latest_price_date")),
        },
    )
    tw_order_rows = apply_entry_status(tw_order_rows, "TW", entry_baselines, entry_rules)
    us_order_rows = apply_entry_status(us_order_rows, "US", entry_baselines, entry_rules)
    write_order_plan(tw_order_rows + us_order_rows)
    us_target_symbols = {
        str(row.get("symbol", ""))
        for row in us_order_rows
        if str(row.get("symbol", "")) and str(row.get("symbol", "")) != "CASH" and (as_float(row.get("target_weight")) or 0.0) > 0.0
    }
    tw_themes = read_csv(OUTPUT_DIR / "TW" / "latest_theme_rank.csv", 12)
    us_themes = read_csv(OUTPUT_DIR / "US" / "latest_theme_rank.csv", 12)
    tw_candidates = read_csv(OUTPUT_DIR / "TW" / "latest_candidate_pool.csv", 30)
    us_candidates = read_csv(OUTPUT_DIR / "US" / "latest_candidate_pool.csv", 30)
    us_scan_state = read_csv(OUTPUT_DIR / "US_SCAN" / "latest_market_state.csv")
    us_scan_stocks = read_csv(OUTPUT_DIR / "US_SCAN" / "latest_scan_stock_rank.csv", 40)
    us_scan_themes = read_csv(OUTPUT_DIR / "US_SCAN" / "latest_scan_theme_rank.csv", 14)
    us_scan_turnover = read_csv(OUTPUT_DIR / "US_SCAN" / "latest_scan_pool_turnover.csv", 40)
    us_upgrade_candidates = read_csv(DATA_DIR / "us_theme_promotion_candidates.csv", 20)
    us_dynamic_formal_pool = read_csv(DATA_DIR / "us_dynamic_formal_candidate_pool.csv", 30)
    us_scan_alerts = read_json(OUTPUT_DIR / "US_SCAN" / "latest_scan_alerts.json", {})
    us_scan_outside = us_scan_alerts.get("top_outside_execution_symbols", []) if isinstance(us_scan_alerts, dict) else []

    generated_at = str(summary.get("generated_at", ""))
    tw_date = state_value(tw_state, "latest_price_date")
    us_date = state_value(us_state, "latest_price_date")
    us_scan_count = state_value(us_scan_state, "scan_universe_symbols", "")
    us_regime = str(state_value(us_state, "regime", ""))
    us_rebalance_today = as_bool(state_value(us_state, "rebalance_due_today_by_counter", False))
    us_rebalance = as_bool(state_value(us_state, "rebalance_due_next_session_by_counter", False))
    us_rebalance_date = str(state_value(us_state, "component_rebalance_date", ""))
    us_signal_date = str(state_value(us_state, "component_signal_date", ""))
    us_days_since = as_int(state_value(us_state, "trading_days_since_rebalance", 0))
    us_rebalance_step = as_int(state_value(us_state, "rebalance_step_trading_days", 5), 5)
    tw_rebalance_today = as_bool(state_value(tw_state, "rebalance_due_today_by_counter", False))
    tw_rebalance_next = as_bool(state_value(tw_state, "rebalance_due_next_session_by_counter", False))
    tw_rebalance_date = str(state_value(tw_state, "component_rebalance_date", ""))
    tw_signal_date = str(state_value(tw_state, "component_signal_date", ""))
    tw_days_since = as_int(state_value(tw_state, "trading_days_since_rebalance", 0))
    tw_days_since_signal = as_int(state_value(tw_state, "trading_days_since_signal", 0))
    tw_rebalance_step = as_int(state_value(tw_state, "rebalance_step_trading_days", 5), 5)
    tw_regime = str(state_value(tw_state, "regime_state_label", ""))
    tw_mode = str(state_value(tw_state, "outer_mode", ""))
    gross_cap = as_float(state_value(us_state, "gross_cap", "")) or 0.0
    smh_mom = state_value(us_state, "smh_mom63_pct", "")
    us_dynamic_active = as_bool(state_value(us_state, "dynamic_formal_gate_active", False))
    us_dynamic_reason = str(state_value(us_state, "dynamic_formal_gate_reason", ""))
    us_dynamic_weight = as_float(state_value(us_state, "dynamic_formal_weight", 0.0)) or 0.0
    us_dynamic_date = str(state_value(us_state, "dynamic_formal_latest_price_date", ""))
    us_dynamic_pool_count = state_value(us_state, "dynamic_formal_pool_count", "")
    us_strategy_mode = str(state_value(us_state, "strategy_mode", ""))
    us_is_dynamic_market = us_strategy_mode == "top1_dynamic_market_like_tw"
    us_selected_component = str(state_value(us_state, "selected_component", ""))
    us_dynamic_market_pool = str(state_value(us_state, "dynamic_market_pool", ""))
    us_rebalance_remaining = as_int(state_value(us_state, "rebalance_days_remaining", 0))
    price_new_rows = int(price_report.get("new_rows_total", 0) or 0) if isinstance(price_report, dict) else 0
    price_status = str(price_report.get("status", "")) if isinstance(price_report, dict) else ""
    tw_entry_baseline_date = entry_baseline_date_text(entry_baselines, "TW")
    us_entry_baseline_date = entry_baseline_date_text(entry_baselines, "US")

    tw_schedule_status = "今天換倉檢查" if tw_rebalance_today else ("下一交易日換倉檢查" if tw_rebalance_next else "未到換倉節奏")
    tw_schedule_text = (
        f"{tw_schedule_status} / 上次換倉 {tw_rebalance_date or '-'} / "
        f"已走 {tw_days_since}/{tw_rebalance_step} 交易日"
    )
    if tw_signal_date:
        tw_schedule_text += f" / 策略訊號日 {tw_signal_date} 已走 {tw_days_since_signal} 日"
    if tw_entry_baseline_date:
        tw_schedule_text += f" / 入場基準 {tw_entry_baseline_date}"
    us_schedule_status = "今天換倉檢查" if us_rebalance_today else ("下一交易日換倉檢查" if us_rebalance else "未到換倉節奏")
    us_schedule_text = (
        f"{us_schedule_status} / 上次換倉 {us_rebalance_date or '-'} / "
        f"已走 {us_days_since}/{us_rebalance_step} 交易日"
    )
    if us_signal_date:
        us_schedule_text += f" / 策略訊號日 {us_signal_date}"
    if us_entry_baseline_date:
        us_schedule_text += f" / 入場基準 {us_entry_baseline_date}"
    rebalance_pill = pill(TXT["rebalance_due"], "ok") if us_rebalance else pill(TXT["not_rebalance"], "warn")
    regime_pill = pill(TXT["us_risk_on"], "ok") if us_regime == "risk_on" else pill(TXT["us_defense"], "danger")
    us_top1_label = zh(r"\u7f8e\u80a1Top1\u52d5\u614b\u5e02\u5834")
    us_strategy_label = zh(r"\u7f8e\u80a1\u7b56\u7565")
    us_component_label = zh(r"\u76ee\u524d\u7b56\u7565\u817f")
    next_rebalance_label = zh(r"\u4e0b\u6b21\u63db\u5009")
    trading_day_label = zh(r"\u4ea4\u6613\u65e5")
    if us_is_dynamic_market:
        dynamic_pill = pill(us_top1_label, "ok")
        dynamic_metric_1 = (
            f"<div class='metric'><span>{us_strategy_label}</span>"
            f"<strong>Top1 Dynamic</strong></div>"
        )
        dynamic_metric_2 = (
            f"<div class='metric'><span>{us_component_label}</span>"
            f"<strong>{esc(us_selected_component or '-')}</strong></div>"
        )
        dynamic_metric_3 = (
            f"<div class='metric'><span>{next_rebalance_label}</span>"
            f"<strong>{esc(str(us_rebalance_remaining))} {trading_day_label}</strong></div>"
        )
    else:
        dynamic_pill = pill("Formal99 ON", "ok") if us_dynamic_active else pill(f"Formal99 OFF: {us_dynamic_reason}", "warn")
        dynamic_metric_1 = (
            f"<div class='metric'><span>Formal99</span>"
            f"<strong>{'ON' if us_dynamic_active else 'OFF'} / {us_dynamic_weight * 100:.0f}%</strong></div>"
        )
        dynamic_metric_2 = (
            f"<div class='metric'><span>Formal99 Data</span>"
            f"<strong>{esc(us_dynamic_date)} / {esc(us_dynamic_pool_count)} {TXT['unit_symbols']}</strong></div>"
        )
        dynamic_metric_3 = ""
    tw_pill = pill(TXT["tw_strong"], "ok") if tw_regime == "strong" else pill(f"{TXT['tw']} {tw_regime}", "warn")
    if not price_report:
        price_pill = pill(TXT["price_no_report"], "warn")
    elif price_new_rows > 0:
        price_pill = pill(TXT["price_updated"], "ok")
    else:
        price_pill = pill(TXT["price_no_new"], "warn")

    style = """
    :root {
      --bg: #f4f6f8; --panel: #ffffff; --panel-2: #f9fafb; --text: #17202c;
      --muted: #667085; --line: #d7dde7; --line-soft: #ebeff5; --green: #087f5b;
      --green-bg: #e7f6ef; --blue: #1f5eff; --blue-bg: #edf3ff; --amber: #b45f06;
      --amber-bg: #fff4df; --red: #b42318; --red-bg: #ffe9e6; --ink: #111827;
    }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: Arial, "Microsoft JhengHei", "PingFang TC", sans-serif; color: var(--text); background: var(--bg); -webkit-text-size-adjust: 100%; }
    header { position: sticky; top: 0; z-index: 10; display: flex; justify-content: space-between; align-items: center; gap: 18px; padding: 14px 22px; border-bottom: 1px solid var(--line); background: rgba(255,255,255,.96); backdrop-filter: blur(10px); }
    h1 { margin: 0; font-size: 21px; letter-spacing: 0; }
    h2 { margin: 0; font-size: 17px; letter-spacing: 0; }
    h3 { margin: 0 0 10px; font-size: 15px; letter-spacing: 0; }
    main { max-width: 1380px; margin: 0 auto; padding: 20px; }
    button { border: 1px solid #0f766e; background: #0f766e; color: #fff; border-radius: 6px; padding: 10px 14px; font-weight: 700; cursor: pointer; white-space: nowrap; }
    button:disabled { opacity: .6; cursor: wait; }
    .download-link { display: inline-flex; align-items: center; justify-content: center; border: 1px solid var(--line); background: #fff; color: var(--text); border-radius: 6px; padding: 9px 12px; font-size: 13px; font-weight: 800; text-decoration: none; white-space: nowrap; }
    .download-link:hover { border-color: #0f766e; color: #0f766e; }
    input { width: 150px; border: 1px solid var(--line); border-radius: 6px; padding: 8px 10px; font-size: 14px; background: #fff; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th, td { padding: 9px 8px; border-bottom: 1px solid var(--line-soft); text-align: left; vertical-align: top; }
    th { color: var(--muted); font-weight: 700; white-space: nowrap; }
    td { line-height: 1.35; }
    .meta { display: none; }
    .top-actions { display: flex; flex-direction: column; align-items: flex-end; gap: 5px; }
    .status { color: var(--muted); font-size: 13px; min-height: 18px; text-align: right; }
    .hero { display: grid; grid-template-columns: 1.1fr .9fr; gap: 16px; align-items: stretch; margin-bottom: 16px; }
    .decision, .section { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 16px; box-shadow: 0 1px 2px rgba(16,24,40,.04); }
    .section { margin-bottom: 16px; overflow: auto; }
    .rebalance-panel { margin-bottom: 16px; }
    .rebalance-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    .rebalance-card { border: 1px solid var(--line); border-radius: 8px; padding: 14px; background: var(--panel-2); }
    .rebalance-card.ok { border-color: #b7e4cf; background: var(--green-bg); }
    .rebalance-card.warn { border-color: #f5d08a; background: var(--amber-bg); }
    .rebalance-card.danger { border-color: #ffb4ad; background: var(--red-bg); }
    .rebalance-market { color: var(--muted); font-size: 13px; font-weight: 800; }
    .rebalance-status { margin-top: 6px; font-size: 28px; font-weight: 900; color: var(--ink); }
    .rebalance-detail { margin-top: 8px; min-height: 38px; line-height: 1.45; }
    .rebalance-meta { margin-top: 10px; color: var(--muted); font-size: 12px; }
    .details-collapse { margin-bottom: 18px; }
    .details-collapse > summary { cursor: pointer; list-style: none; border: 1px solid var(--line); border-radius: 8px; padding: 13px 16px; background: var(--panel); font-weight: 900; }
    .details-collapse > summary::-webkit-details-marker { display: none; }
    .details-collapse[open] > summary { margin-bottom: 12px; }
    .decision-title, .section-head { display: flex; justify-content: space-between; align-items: flex-start; gap: 12px; margin-bottom: 12px; }
    .decision-title p, .section-subtitle { margin: 5px 0 0; color: var(--muted); font-size: 13px; line-height: 1.45; }
    .buy-grid, .details-grid, .entry-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    .metrics { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; }
    .metric { border: 1px solid var(--line-soft); background: var(--panel-2); border-radius: 8px; padding: 12px; min-height: 76px; }
    .metric span { display: block; color: var(--muted); font-size: 12px; margin-bottom: 6px; }
    .metric strong { display: block; font-size: 20px; color: var(--ink); word-break: break-word; }
    .pill { display: inline-flex; align-items: center; border-radius: 999px; padding: 4px 9px; font-size: 12px; font-weight: 700; white-space: nowrap; margin-left: 4px; }
    .pill.ok { color: var(--green); background: var(--green-bg); }
    .pill.warn { color: var(--amber); background: var(--amber-bg); }
    .pill.danger { color: var(--red); background: var(--red-bg); }
    .pill.watch { color: var(--blue); background: var(--blue-bg); }
    .pill.neutral { color: #344054; background: #eef1f5; }
    .target-strip { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 12px; }
    .target-chip { display: inline-flex; align-items: center; gap: 8px; border-radius: 999px; padding: 7px 10px; font-size: 13px; background: var(--blue-bg); color: #174ea6; border: 1px solid #cfe0ff; }
    .target-chip.entry-ok { border-color: #b7e4cf; }
    .target-chip.entry-warn { border-color: #f5d08a; }
    .target-chip.entry-danger { border-color: #ffb4ad; }
    .target-chip.cash { color: #5f6368; background: #f1f3f4; border-color: #dadddf; }
    .target-chip em { font-style: normal; font-weight: 700; }
    .target-chip small { font-size: 11px; font-weight: 900; }
    .entry-grid { margin-bottom: 16px; }
    .entry-board { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }
    .entry-card { border: 1px solid var(--line-soft); border-radius: 8px; padding: 12px; background: var(--panel-2); min-height: 116px; }
    .entry-card.entry-ok { border-color: #b7e4cf; background: var(--green-bg); }
    .entry-card.entry-warn { border-color: #f5d08a; background: var(--amber-bg); }
    .entry-card.entry-danger { border-color: #ffb4ad; background: var(--red-bg); }
    .entry-card-top { display: flex; justify-content: space-between; align-items: flex-start; gap: 8px; }
    .entry-card-main { margin-top: 10px; font-size: 24px; line-height: 1; font-weight: 900; color: var(--ink); }
    .entry-card-meta { display: flex; gap: 10px; margin-top: 10px; color: var(--muted); font-size: 12px; font-weight: 800; }
    .entry-card-note { margin-top: 8px; color: var(--muted); font-size: 11px; }
    .target-table { font-size: 13px; }
    .target-table th { background: #fbfcfe; }
    .target-table td { padding-top: 11px; padding-bottom: 11px; }
    .target-row:hover td { background: #fbfdff; }
    .symbol { display: block; font-size: 15px; font-weight: 800; color: var(--ink); }
    .name { display: block; margin-top: 2px; color: var(--muted); font-size: 12px; }
    .raw { display: block; color: var(--muted); font-size: 11px; margin-top: 2px; }
    .weight { font-weight: 800; color: var(--green); white-space: nowrap; }
    .amount { font-weight: 800; white-space: nowrap; }
    .cash-row .weight, .cash-row .amount { color: #5f6368; }
    .capital-bar { display: flex; align-items: center; justify-content: flex-end; gap: 8px; color: var(--muted); font-size: 13px; margin-bottom: 10px; white-space: nowrap; }
    .capital-bar input { width: 132px; }
    .theme-cell { min-width: 120px; }
    .role-chip { display: inline-flex; align-items: center; border-radius: 999px; padding: 4px 8px; background: #eef1f5; color: #344054; font-size: 12px; font-weight: 800; white-space: nowrap; }
    .action-chip { display: inline-flex; align-items: center; justify-content: center; min-width: 46px; border-radius: 999px; padding: 4px 8px; font-size: 12px; font-weight: 800; white-space: nowrap; }
    .entry-chip { display: inline-flex; align-items: center; justify-content: center; min-width: 72px; border-radius: 999px; padding: 5px 9px; font-size: 12px; font-weight: 900; white-space: nowrap; }
    .buy-action { color: var(--green); background: var(--green-bg); }
    .sell-action { color: var(--red); background: var(--red-bg); }
    .hold-action { color: #344054; background: #eef1f5; }
    .entry-ok { color: var(--green); background: var(--green-bg); }
    .entry-warn { color: var(--amber); background: var(--amber-bg); }
    .entry-danger { color: var(--red); background: var(--red-bg); }
    .entry-neutral { color: #344054; background: #eef1f5; }
    .watch-layout { display: grid; grid-template-columns: .86fr 1.14fr; gap: 16px; align-items: start; }
    .scan-turnover { margin-top: 14px; }
    .theme-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; }
    .theme-tile { border: 1px solid var(--line-soft); background: var(--panel-2); border-radius: 8px; padding: 11px; min-height: 118px; }
    .theme-rank { color: var(--muted); font-size: 12px; font-weight: 700; }
    .theme-name { margin-top: 6px; font-weight: 800; min-height: 36px; }
    .theme-score { margin-top: 8px; color: var(--green); font-weight: 800; }
    .theme-symbols { margin: 6px 0; color: var(--muted); font-size: 12px; line-height: 1.35; word-break: break-word; }
    .clean-shell { display: grid; gap: 14px; min-width: 0; }
    .clean-title { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 16px; }
    .clean-title h2 { font-size: 22px; margin-bottom: 5px; }
    .clean-title p { margin: 0; color: var(--muted); font-size: 13px; line-height: 1.45; }
    .clean-grid { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 14px; min-width: 0; }
    .clean-panel { min-width: 0; background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 16px; box-shadow: 0 1px 2px rgba(16,24,40,.04); }
    .clean-panel.ok { border-top: 4px solid #38a169; }
    .clean-panel.warn { border-top: 4px solid #d99021; }
    .clean-panel.danger { border-top: 4px solid #d92d20; }
    .clean-head { display: flex; justify-content: space-between; align-items: flex-start; gap: 12px; margin-bottom: 12px; }
    .clean-market { color: var(--muted); font-size: 12px; font-weight: 900; letter-spacing: .02em; }
    .clean-head h2 { font-size: 28px; margin: 4px 0 6px; }
    .clean-head p { margin: 0; color: var(--muted); font-size: 12px; line-height: 1.4; }
    .clean-capital { display: grid; gap: 5px; min-width: 148px; color: var(--muted); font-size: 12px; font-weight: 800; }
    .clean-capital input { width: 148px; }
    .clean-list { display: grid; gap: 8px; min-width: 0; }
    .trade-list { display: grid; gap: 10px; min-width: 0; }
    .trade-card { min-width: 0; border: 1px solid var(--line-soft); border-left-width: 4px; border-radius: 8px; padding: 14px; background: #fff; }
    .trade-card.entry-ok { border-color: #b7e4cf; background: #f6fffa; }
    .trade-card.entry-warn { border-color: #f5d08a; background: #fffaf0; }
    .trade-card.entry-danger { border-color: #ffb4ad; background: #fff8f7; }
    .trade-card.cash { background: #f7f8fa; }
    .trade-card-head { display: grid; grid-template-columns: minmax(0, 1fr) auto; align-items: start; gap: 12px; margin-bottom: 12px; }
    .trade-symbol { min-width: 0; display: grid; gap: 3px; }
    .trade-symbol strong { color: var(--ink); font-size: 18px; line-height: 1.1; }
    .trade-symbol span { color: var(--muted); font-size: 12px; line-height: 1.25; max-width: 22rem; }
    .trade-status { justify-self: end; border-radius: 999px; padding: 6px 10px; font-size: 12px; font-weight: 900; white-space: nowrap; }
    .trade-status.entry-ok { color: var(--green); background: var(--green-bg); }
    .trade-status.entry-warn { color: var(--amber); background: var(--amber-bg); }
    .trade-status.entry-danger { color: var(--red); background: var(--red-bg); }
    .trade-status.cash { color: #344054; background: #eef1f5; }
    .trade-facts { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px 14px; align-items: start; }
    .trade-fact { min-width: 0; border-left: 3px solid #e5eaf2; padding: 2px 0 2px 10px; display: grid; align-content: start; gap: 4px; }
    .trade-fact span { color: var(--muted); font-size: 11px; font-weight: 800; line-height: 1.15; }
    .trade-fact strong { color: var(--ink); font-size: 13px; font-weight: 900; line-height: 1.25; overflow-wrap: anywhere; }
    .trade-fact em { color: var(--green); font-style: normal; white-space: nowrap; }
    .trade-fact small { color: var(--muted); font-size: 10px; font-weight: 800; line-height: 1.2; }
    .trade-band strong { color: #174ea6; }
    .trade-amount-box strong { white-space: normal; }
    .trade-price-edit { margin-top: 10px; color: var(--muted); font-size: 11px; font-weight: 900; }
    .trade-price-edit summary { cursor: pointer; width: fit-content; list-style: none; border: 1px solid var(--line-soft); border-radius: 999px; padding: 5px 9px; background: rgba(255,255,255,.72); }
    .trade-price-edit summary::-webkit-details-marker { display: none; }
    .trade-price-edit[open] { display: grid; grid-template-columns: auto minmax(120px, 150px); justify-content: end; align-items: center; gap: 8px; }
    .trade-price-edit[open] summary { align-self: center; }
    .clean-reason { display: grid; gap: 8px; border: 1px solid var(--line-soft); border-radius: 8px; padding: 12px; background: #fff8f7; }
    .clean-reason div { display: grid; grid-template-columns: 86px minmax(0, 1fr); gap: 10px; align-items: start; }
    .clean-reason strong { color: var(--red); font-size: 12px; white-space: nowrap; }
    .clean-reason span { color: var(--ink); font-size: 13px; font-weight: 800; line-height: 1.45; }
    .clean-row { min-width: 0; display: grid; grid-template-columns: minmax(92px, 1.1fr) 82px minmax(126px, 1fr) minmax(116px, 1fr) minmax(104px, .9fr) minmax(116px, .9fr); align-items: center; gap: 8px; border: 1px solid var(--line-soft); border-radius: 8px; padding: 10px; background: #fff; }
    .clean-row.entry-ok { background: #f7fffb; }
    .clean-row.entry-warn { background: #fffaf0; }
    .clean-row.entry-danger { background: #fff8f7; }
    .clean-symbol strong { display: block; color: var(--ink); font-size: 15px; }
    .clean-name { display: block; margin-top: 2px; color: var(--muted); font-size: 11px; }
    .clean-status { justify-self: start; border-radius: 999px; padding: 5px 8px; font-size: 12px; font-weight: 900; white-space: nowrap; }
    .clean-status.entry-ok { color: var(--green); background: var(--green-bg); }
    .clean-status.entry-warn { color: var(--amber); background: var(--amber-bg); }
    .clean-status.entry-danger { color: var(--red); background: var(--red-bg); }
    .clean-status.cash { color: #344054; background: #eef1f5; }
    .track-cell { display: grid; gap: 3px; min-width: 0; }
    .track-cell span { color: var(--muted); font-size: 11px; font-weight: 800; }
    .track-cell strong { color: var(--ink); font-size: 12px; font-weight: 900; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .track-cell em { color: var(--green); font-style: normal; margin-right: 8px; }
    .track-cell b { font: inherit; }
    .price-meta { color: var(--muted); font-size: 10px; font-weight: 800; }
    .live-price-cell { align-items: start; }
    .live-price-input { width: 98px; min-height: 32px; padding: 6px 8px; border-radius: 6px; font-size: 13px; font-weight: 800; }
    .buy-band strong { color: #1f4f8f; }
    .clean-gap { color: var(--muted); font-size: 12px; font-weight: 800; text-align: right; }
    .clean-weight { color: var(--green); font-size: 12px; font-weight: 900; text-align: right; }
    .clean-amount { color: var(--ink); font-size: 13px; text-align: right; white-space: nowrap; }
    .detail-grid-plain { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    .empty { color: var(--muted); padding: 12px 0; }
    @media (max-width: 1100px) {
      .hero, .buy-grid, .watch-layout, .details-grid, .entry-grid { grid-template-columns: 1fr; }
      .clean-grid, .detail-grid-plain { grid-template-columns: 1fr; }
      .rebalance-grid { grid-template-columns: 1fr; }
      .metrics, .theme-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .trade-facts { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
    @media (max-width: 720px) {
      header { display: grid; grid-template-columns: minmax(0, 1fr) auto; align-items: center; gap: 10px; padding: 10px 12px; }
      h1 { font-size: 18px; line-height: 1.15; }
      h2 { font-size: 16px; }
      h3 { font-size: 14px; margin-bottom: 8px; }
      main { padding: 10px; }
      button { min-height: 40px; padding: 9px 12px; border-radius: 7px; }
      input { width: 100%; min-height: 40px; }
      .meta { display: none; }
      .top-actions { align-items: stretch; }
      .status { display: block; text-align: left; white-space: normal; line-height: 1.35; }
      .download-link { min-height: 40px; padding: 9px 12px; }
      .hero { gap: 10px; margin-bottom: 10px; }
      .decision, .section { padding: 12px; border-radius: 8px; margin-bottom: 10px; }
      .clean-shell { gap: 10px; }
      .clean-grid, .detail-grid-plain { grid-template-columns: minmax(0, 1fr); }
      .clean-title { padding: 12px; }
      .clean-title h2 { font-size: 18px; }
      .clean-title p { font-size: 12px; }
      .clean-panel { padding: 12px; }
      .clean-head { display: grid; gap: 10px; margin-bottom: 10px; }
      .clean-head h2 { font-size: 24px; margin: 3px 0 4px; }
      .clean-capital { min-width: 0; }
      .clean-capital input { width: 100%; }
      .clean-reason { padding: 10px; }
      .clean-reason div { grid-template-columns: 72px minmax(0, 1fr); gap: 8px; }
      .trade-card { padding: 12px; }
      .trade-card-head { grid-template-columns: minmax(0, 1fr); gap: 8px; margin-bottom: 10px; }
      .trade-symbol strong { font-size: 17px; }
      .trade-status { justify-self: start; padding: 5px 8px; font-size: 11px; }
      .trade-facts { grid-template-columns: 1fr; gap: 8px; }
      .trade-fact { padding: 2px 0 2px 9px; }
      .trade-fact strong { font-size: 12px; }
      .trade-price-edit[open] { grid-template-columns: 1fr; justify-content: stretch; }
      .trade-price-edit .live-price-input { width: 100%; }
      .clean-row { grid-template-columns: minmax(0, 1fr) auto; gap: 8px 10px; padding: 10px; }
      .clean-status { justify-self: end; }
      .track-cell { grid-column: span 2; grid-template-columns: 72px minmax(0, 1fr); align-items: baseline; gap: 8px; }
      .track-cell span { font-size: 11px; }
      .track-cell strong { font-size: 12px; white-space: normal; }
      .track-cell em { margin-right: 6px; }
      .clean-gap, .clean-weight, .clean-amount { text-align: left; }
      .clean-gap::before { content: "距離 "; color: var(--muted); font-weight: 700; }
      .clean-weight::before { content: "權重 "; color: var(--muted); font-weight: 700; }
      .clean-amount::before { content: "金額 "; color: var(--muted); font-weight: 700; }
      .live-price-input { width: min(132px, 100%); min-height: 36px; }
      .section { overflow: visible; }
      .rebalance-panel { margin-bottom: 10px; }
      .decision-title, .section-head { flex-direction: column; gap: 8px; margin-bottom: 10px; }
      .decision-title p, .section-subtitle { font-size: 12px; line-height: 1.4; }
      .hero .decision:first-child .decision-title p { display: none; }
      .pill { margin: 2px 3px 0 0; padding: 4px 8px; font-size: 11px; }
      .metrics { grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; }
      .metric { min-height: 62px; padding: 9px; }
      .metric span { font-size: 11px; margin-bottom: 5px; }
      .metric strong { font-size: 18px; }
      .target-strip { gap: 6px; margin-bottom: 10px; }
      .target-chip { flex: 0 0 calc(50% - 3px); justify-content: space-between; gap: 5px; border-radius: 8px; padding: 7px 8px; font-size: 12px; }
      .target-chip strong { font-size: 12px; }
      .target-chip em { font-size: 11px; }
      .target-chip small { font-size: 11px; }
      .entry-grid { gap: 10px; margin-bottom: 10px; }
      .entry-board { grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; }
      .entry-card { min-height: 98px; padding: 10px; }
      .entry-card-top { gap: 6px; }
      .entry-card-main { margin-top: 8px; font-size: 22px; }
      .entry-card-meta { flex-wrap: wrap; gap: 5px 8px; margin-top: 8px; font-size: 11px; }
      .entry-card-note { margin-top: 6px; font-size: 10px; line-height: 1.25; }
      .symbol { font-size: 15px; }
      .entry-chip { min-width: auto; padding: 4px 7px; font-size: 11px; }
      .capital-bar { justify-content: flex-start; flex-wrap: wrap; gap: 6px; width: 100%; }
      .capital-bar input { flex: 1 1 160px; width: auto; }
      .target-table, .target-table thead, .target-table tbody, .target-table tr, .target-table td { display: block; width: 100%; }
      .target-table thead { display: none; }
      .target-table tr { border: 1px solid var(--line); border-radius: 8px; margin: 0 0 10px; padding: 10px; background: #fff; }
      .target-table tr:last-child { margin-bottom: 0; }
      .target-table td { display: grid; grid-template-columns: 64px minmax(0, 1fr); align-items: start; gap: 8px; border: 0; padding: 6px 0; line-height: 1.35; }
      .target-table td::before { content: attr(data-label); color: var(--muted); font-size: 11px; font-weight: 900; padding-top: 2px; }
      .target-table .entry-note { display: none; }
      .target-table .theme-cell .raw, .target-table td[data-label="定位"] .raw { display: none; }
      .target-row:hover td { background: transparent; }
      .theme-cell { min-width: 0; word-break: break-word; }
      .role-chip { white-space: normal; border-radius: 6px; line-height: 1.35; }
      .raw { font-size: 10px; }
      .theme-grid { grid-template-columns: 1fr; }
      th, td { padding: 8px 6px; }
    }
    @media (max-width: 380px) {
      header { grid-template-columns: 1fr; }
      .top-actions { align-items: stretch; }
      .target-chip { flex-basis: 100%; }
      .entry-board { grid-template-columns: 1fr; }
      .metrics { grid-template-columns: 1fr 1fr; }
    }
    """

    script = f"""
    const btn = document.getElementById('updateBtn');
    const savePricesBtn = document.getElementById('savePricesBtn');
    const statusEl = document.getElementById('status');
    const twCapital = document.getElementById('twCapital');
    const usCapital = document.getElementById('usCapital');
    function fmtMoney(value, currency) {{
      const prefix = currency === 'TWD' ? 'NT$ ' : '$ ';
      return prefix + Math.round(value).toLocaleString();
    }}
    function refreshAmounts() {{
      const totals = {{ TW: Number(twCapital?.value || 0), US: Number(usCapital?.value || 0) }};
      document.querySelectorAll('.target-row').forEach((row) => {{
        const market = row.dataset.market;
        const weight = Number(row.dataset.weight || 0);
        const currency = row.querySelector('.amount')?.dataset.currency || 'USD';
        const cell = row.querySelector('.amount');
        const targetAmount = (totals[market] || 0) * weight;
        if (cell) cell.textContent = fmtMoney(targetAmount, currency);
      }});
    }}
    [twCapital, usCapital].forEach((el) => el && el.addEventListener('input', refreshAmounts));
    refreshAmounts();
    savePricesBtn.addEventListener('click', async () => {{
      const prices = Array.from(document.querySelectorAll('.live-price-input'))
        .map((input) => ({{
          market: input.dataset.market,
          symbol: input.dataset.symbol,
          current_price: Number(input.value || 0),
          price_date: new Date().toISOString().slice(0, 10),
          source: 'manual_ui'
        }}))
        .filter((row) => row.market && row.symbol && Number.isFinite(row.current_price) && row.current_price > 0);
      savePricesBtn.disabled = true;
      statusEl.textContent = '套用現價中';
      try {{
        const res = await fetch('/api/current-prices', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify({{ prices }})
        }});
        const data = await res.json();
        if (!res.ok || !data.ok) throw new Error(data.error || 'save prices failed');
        statusEl.textContent = '現價已套用，重新整理';
        window.location.reload();
      }} catch (err) {{
        statusEl.textContent = String(err);
        savePricesBtn.disabled = false;
      }}
    }});
    function sleep(ms) {{
      return new Promise((resolve) => setTimeout(resolve, ms));
    }}
    function lastLine(lines) {{
      if (!Array.isArray(lines) || !lines.length) return '';
      return String(lines[lines.length - 1] || '').replace(/^\\[[^\\]]+\\]\\s*/, '');
    }}
    function cloudStatusText(data) {{
      const job = data.job || {{}};
      const status = job.status || (data.running ? 'running' : 'idle');
      const progress = lastLine(data.market_progress_tail) || lastLine(data.target_progress_tail) || lastLine(data.cloud_progress_tail) || job.message || '';
      if (status === 'finished') return '雲端更新完成，正在重新整理';
      if (status === 'failed') return '雲端更新失敗：' + (job.message || '請看狀態檔');
      if (status === 'queued') return '雲端更新排隊中';
      if (data.running || status === 'running') return '雲端更新中：' + progress;
      return String(status || '');
    }}
    async function pollCloudUpdate() {{
      for (let i = 0; i < 720; i += 1) {{
        const res = await fetch('/api/cloud-update/status?ts=' + Date.now());
        const data = await res.json();
        if (!res.ok || !data.ok) throw new Error(data.error || 'cloud status failed');
        statusEl.textContent = cloudStatusText(data);
        const jobStatus = data.job && data.job.status;
        if (!data.running && jobStatus === 'finished') {{
          statusEl.textContent = '雲端更新完成，重新整理';
          await sleep(800);
          window.location.reload();
          return;
        }}
        if (!data.running && jobStatus === 'failed') {{
          throw new Error((data.job && data.job.message) || 'cloud update failed');
        }}
        await sleep(5000);
      }}
      throw new Error('cloud update still running; open /api/cloud-update/status to check');
    }}
    btn.addEventListener('click', async () => {{
      btn.disabled = true;
      statusEl.textContent = '雲端更新啟動中';
      try {{
        const res = await fetch('/api/cloud-update/start', {{ method: 'POST' }});
        const data = await res.json();
        if (!res.ok || !data.ok) throw new Error(data.error || 'cloud update start failed');
        statusEl.textContent = data.status === 'already_running' ? '雲端已有更新在跑' : '雲端更新已啟動';
        await pollCloudUpdate();
      }} catch (err) {{
        statusEl.textContent = String(err);
        btn.disabled = false;
      }}
    }});
    """

    overview = (
        "<section class='hero'>"
        "<div class='decision'>"
        f"<div class='decision-title'><div><h2>{TXT['today_orders']}</h2><p>{TXT['top_note']}</p></div>"
        f"<div>{tw_pill} {regime_pill} {dynamic_pill} {rebalance_pill} {price_pill}</div></div>"
        "<div class='metrics'>"
        f"<div class='metric'><span>{TXT['tw']} {TXT['data_date']}</span><strong>{esc(tw_date)}</strong></div>"
        f"<div class='metric'><span>{TXT['us']} {TXT['data_date']}</span><strong>{esc(us_date)}</strong></div>"
        f"<div class='metric'><span>{TXT['us']} {TXT['gross']}</span><strong>{gross_cap * 100:.0f}%</strong></div>"
        f"<div class='metric'><span>{TXT['us']} {TXT['scan_pool']}</span><strong>{esc(us_scan_count)} {TXT['unit_symbols']}</strong></div>"
        f"{dynamic_metric_1}"
        f"{dynamic_metric_2}"
        f"{dynamic_metric_3}"
        f"<div class='metric'><span>{TXT['price_update']}</span><strong>{esc(price_new_rows)} {TXT['rows']}</strong></div>"
        "</div></div>"
        "<div class='decision'>"
        f"<div class='decision-title'><div><h2>{TXT['summary']}</h2>"
        f"<p>{TXT['tw']} {TXT['mode']}: {esc(tw_mode)} / {TXT['us']} regime: {esc(us_regime)} / SMH 63D: {fmt_pct(smh_mom)}</p></div></div>"
        f"<h3>{TXT['tw']}</h3>{compact_targets(tw_order_rows)}"
        f"<h3>{TXT['us']}</h3>{compact_targets(us_order_rows)}"
        "</div></section>"
    )

    rebalance_section = (
        "<section class='decision rebalance-panel'>"
        "<div class='decision-title'><div><h2>換倉結論</h2>"
        "<p>先看這裡。正式表只顯示權重、資金試算與入場狀態；數量欄與持倉輸入已從畫面拿掉。</p></div></div>"
        "<div class='rebalance-grid'>"
        + rebalance_card("台股", tw_order_rows, "TW", has_saved_holdings(tw_hold_map), None, tw_date)
        + rebalance_card("美股", us_order_rows, "US", has_saved_holdings(us_hold_map), us_rebalance, us_date)
        + "</div></section>"
    )

    entry_section = (
        "<div class='entry-grid'>"
        "<section class='section'>"
        "<div class='section-head'><div><h2>台股入場判斷</h2><p class='section-subtitle'>用盤中/最新價對進場基準日收盤價判斷，落在 -7% 到 +3% 才能新買。</p></div></div>"
        f"{entry_board(tw_order_rows, 'TW', 'TWD')}"
        "</section>"
        "<section class='section'>"
        "<div class='section-head'><div><h2>美股入場判斷</h2><p class='section-subtitle'>用盤中/最新價對進場基準日收盤價判斷，落在 -5% 到 +5% 才能新買。</p></div></div>"
        f"{entry_board(us_order_rows, 'US', 'USD')}"
        "</section></div>"
    )

    price_update_section = detail_section(
        TXT["price_update"],
        price_update_table(price_report),
        f"{TXT['price_update_note']} status={price_status or 'none'}",
    )

    buy_sections = (
        "<div class='buy-grid'>"
        "<section class='section'>"
        f"<div class='section-head'><div><h2>{TXT['tw_target']}</h2><p class='section-subtitle'>{TXT['tw_target_note']} 輸入資金後只估算目標金額，不顯示數量。</p></div>"
        f"<div class='capital-bar'><span>{TXT['tw_capital']}</span><input id='twCapital' type='number' min='0' step='10000' value='1000000'></div></div>"
        f"{target_table(tw_order_rows, 'TW', 'TWD')}"
        "</section>"
        "<section class='section'>"
        f"<div class='section-head'><div><h2>{TXT['us_target']}</h2><p class='section-subtitle'>{TXT['us_target_note']} 輸入資金後只估算目標金額，不顯示數量。</p></div>"
        f"<div class='capital-bar'><span>{TXT['us_capital']}</span><input id='usCapital' type='number' min='0' step='1000' value='100000'></div></div>"
        f"{target_table(us_order_rows, 'US', 'USD')}"
        "</section></div>"
    )

    clean_dashboard = (
        "<section class='clean-shell'>"
        "<div class='clean-title'>"
        "<h2>推薦與持倉追蹤</h2>"
        f"<p>每檔用進入持倉追蹤時的收盤價當基準，直接顯示買入上下限與盤中/最新價偏離；同一輪持倉內，不會每天重設基準價。台股資料日 {esc(tw_date)}，美股資料日 {esc(us_date)}。</p>"
        "</div>"
        "<div class='clean-grid'>"
        + clean_order_panel(
            tw_order_rows,
            "台股",
            "TW",
            "TWD",
            tw_date,
            TXT["tw_capital"],
            "twCapital",
            DEFAULT_CAPITAL["TW"],
            10000,
            tw_schedule_text,
            "今天換倉檢查" if tw_rebalance_today else ("下一交易日換倉檢查" if tw_rebalance_next else None),
        )
        + clean_order_panel(
            us_order_rows,
            "美股",
            "US",
            "USD",
            us_date,
            TXT["us_capital"],
            "usCapital",
            DEFAULT_CAPITAL["US"],
            1000,
            us_schedule_text,
            "今天換倉檢查" if us_rebalance_today else ("下一交易日換倉檢查" if us_rebalance else None),
        )
        + "</div></section>"
    )

    full_target_details = (
        "<div class='detail-grid-plain'>"
        + detail_section("台股完整目標", target_table(tw_order_rows, "TW", "TWD"))
        + detail_section("美股完整目標", target_table(us_order_rows, "US", "USD"))
        + "</div>"
    )

    scan_turnover_table = simple_table(
        us_scan_turnover,
        [
            ("change", TXT["change"]),
            ("symbol", TXT["stock"]),
            ("name", TXT["name"]),
            ("trade_group", TXT["theme"]),
            ("old_rank", TXT["old_rank"]),
            ("new_rank", TXT["new_rank"]),
            ("median_dollar_volume_60d", "60D $Vol"),
        ],
    )
    upgrade_candidate_table = simple_table(
        us_upgrade_candidates,
        [
            ("rank", "#"),
            ("symbol", TXT["stock"]),
            ("name", TXT["name"]),
            ("scan_theme", TXT["theme"]),
            ("scan_score", TXT["score"]),
            ("mom63_skip5_pct", "63D"),
            ("dollar_volume20", "20D $Vol"),
            ("obs", "Obs"),
            ("admission_stage", TXT["status"]),
            ("auto_buy", "Auto Buy"),
        ],
    )
    dynamic_formal_pool_table = simple_table(
        us_dynamic_formal_pool,
        [
            ("formal_pool_rank", "#"),
            ("symbol", TXT["stock"]),
            ("name", TXT["name"]),
            ("scan_theme", TXT["theme"]),
            ("scan_score", TXT["score"]),
            ("mom63_skip5_pct", "63D"),
            ("mom126_skip5_pct", "126D"),
            ("dollar_volume20", "20D $Vol"),
            ("obs", "Obs"),
            ("in_execution_pool", "Old Pool"),
            ("auto_buy", "Auto Buy"),
        ],
    )
    dynamic_formal_pool_section = ""
    if not us_is_dynamic_market:
        dynamic_formal_pool_section = (
            f"<div class='scan-turnover'><h3>{TXT['us_dynamic_formal_pool']}</h3>"
            f"<p class='section-subtitle'>{TXT['us_dynamic_formal_pool_note']}</p>{dynamic_formal_pool_table}</div>"
        )

    watch_section = (
        "<section class='section'>"
        f"<div class='section-head'><div><h2>{TXT['us_watch']}</h2><p class='section-subtitle'>{TXT['us_watch_note']}</p></div>{pill(TXT['watch_layer'], 'watch')}</div>"
        "<div class='watch-layout'>"
        f"<div><h3>{TXT['theme_heat']}</h3>{theme_cards(us_scan_themes, 'scan_theme', 'theme_score', 'top_symbols', 6)}</div>"
        f"<div><h3>{TXT['stock_accel']}</h3>{watch_table(us_scan_stocks, 15, us_target_symbols)}</div>"
        "</div>"
        f"{dynamic_formal_pool_section}"
        f"<div class='scan-turnover'><h3>{TXT['us_upgrade_candidates']}</h3><p class='section-subtitle'>{TXT['us_upgrade_candidates_note']}</p>{upgrade_candidate_table}</div>"
        f"<div class='scan-turnover'><h3>{TXT['us_scan_turnover']}</h3><p class='section-subtitle'>{TXT['us_scan_turnover_note']}</p>{scan_turnover_table}</div>"
        "</section>"
    )

    details = (
        "<details class='details-collapse'><summary>展開詳細資料</summary>"
        + overview
        + rebalance_section
        + entry_section
        + full_target_details
        + price_update_section
        + watch_section
        + "<div class='details-grid'>"
        + detail_section(
            TXT["tw_theme_rank"],
            simple_table(tw_themes, [("rank", "#"), ("theme", TXT["theme"]), ("theme_score", TXT["score"]), ("selected", TXT["included"])]),
        )
        + detail_section(
            TXT["us_theme_rank"],
            simple_table(
                us_themes,
                [
                    ("rank", "#"),
                    ("theme", TXT["theme"]),
                    ("theme_score", TXT["score"]),
                    ("top_symbols_by_momentum", TXT["strong_stocks"]),
                    ("selected", TXT["included"]),
                ],
            ),
        )
        + detail_section(
            TXT["tw_candidates"],
            simple_table(
                tw_candidates,
                [
                    ("rank", "#"),
                    ("symbol", TXT["stock"]),
                    ("name", TXT["name"]),
                    ("theme", TXT["theme"]),
                    ("selected_today", TXT["selected_today"]),
                    ("ret60_pct", "60D"),
                    ("amount_rank60", TXT["amount_rank"]),
                ],
            ),
        )
        + detail_section(
            TXT["us_candidates"],
            simple_table(
                us_candidates,
                [
                    ("source_sleeve", TXT["source"]),
                    ("symbol", TXT["stock"]),
                    ("theme", TXT["theme"]),
                    ("eligible", TXT["eligible"]),
                    ("mom63_skip5_pct", "63D"),
                    ("mom126_skip5_pct", "126D"),
                ],
            ),
        )
        + detail_section(TXT["us_alerts"], watch_table(us_scan_outside, 12, us_target_symbols), TXT["alerts_note"])
        + detail_section(
            TXT["us_scan_theme_rank"],
            simple_table(
                us_scan_themes,
                [
                    ("rank", "#"),
                    ("scan_theme", TXT["theme"]),
                    ("theme_score", TXT["score"]),
                    ("member_count", TXT["members"]),
                    ("top_symbols", TXT["strong_stocks"]),
                ],
            ),
        )
        + "</div></details>"
    )

    html_text = f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Daily Market Pool Cloud</title>
  <style>{style}</style>
</head>
<body>
  <header>
    <div>
      <h1>Daily Market Pool Cloud</h1>
      <div class="meta">generated_at: {esc(generated_at)} / data_mode: {esc(summary.get("data_mode", ""))}</div>
    </div>
    <div class="top-actions">
      <a class="download-link" href="/files/DAILY_MARKET_RECORD.xlsx">{TXT['excel_record']}</a>
      <button id="savePricesBtn" type="button">套用現價</button>
      <button id="updateBtn" type="button">{TXT['update']}</button>
      <div id="status" class="status"></div>
    </div>
  </header>
  <main>
    {clean_dashboard}
    {details}
  </main>
  <script>{script}</script>
</body>
</html>
"""
    DASHBOARD_PATH.write_text(html_text, encoding="utf-8")
    return DASHBOARD_PATH


if __name__ == "__main__":
    print(main())
