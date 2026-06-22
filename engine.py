from __future__ import annotations

import json
import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data_live"))
RULE_DIR = Path(os.getenv("RULE_DIR", BASE_DIR / "frozen_rules"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", BASE_DIR / "runtime_outputs"))
STATE_PATH = Path(os.getenv("STATE_PATH", BASE_DIR / "state" / "live_state.json"))
STATE_DIR = STATE_PATH.parent
LIVE_CYCLE_BASELINE_PATH = Path(os.getenv("LIVE_CYCLE_BASELINE_PATH", STATE_DIR / "live_cycle_baseline.csv"))

US_OUT = OUTPUT_DIR / "US"
US_SCAN_OUT = OUTPUT_DIR / "US_SCAN"
TW_OUT = OUTPUT_DIR / "TW"

US_SCAN_THEME_OVERRIDES = {
    "AAOI": "optical_communication",
    "AAPL": "consumer_platform",
    "AEHR": "semi_equipment",
    "ALAB": "ai_data_center_connectivity",
    "AMAT": "semi_equipment",
    "AMD": "ai_compute",
    "AMZN": "software_platform",
    "ANET": "server_network",
    "ARM": "ai_compute",
    "ASTS": "space_ev_defense",
    "AVGO": "ai_compute",
    "AXTI": "compound_semiconductor",
    "BE": "clean_power",
    "CEG": "power_generation",
    "CIEN": "server_network",
    "COHR": "optical_communication",
    "CRDO": "ai_data_center_connectivity",
    "DELL": "server_network",
    "DOCN": "cloud_infra",
    "FLEX": "electronics_manufacturing",
    "GEV": "grid_engineering",
    "GFS": "semi_foundry_idm",
    "GOOGL": "software_platform",
    "HUT": "crypto_infra",
    "INTC": "semi_foundry_idm",
    "IONQ": "quantum_compute",
    "IREN": "crypto_ai_power",
    "KLAC": "semi_equipment",
    "LITE": "optical_communication",
    "LRCX": "semi_equipment",
    "META": "software_platform",
    "MRVL": "ai_compute",
    "MSFT": "software_platform",
    "MU": "memory_storage",
    "NBIS": "cloud_ai_infra",
    "NVDA": "ai_compute",
    "NVTS": "power_semiconductor",
    "PL": "space_ev_defense",
    "PLTR": "space_ev_defense",
    "POET": "optical_communication",
    "RKLB": "space_ev_defense",
    "SMCI": "server_network",
    "SNDK": "memory_storage",
    "STM": "power_semiconductor",
    "STX": "memory_storage",
    "TER": "semi_equipment",
    "TSLA": "space_ev_defense",
    "TSM": "semi_foundry_idm",
    "VRT": "thermal_power",
    "WDC": "memory_storage",
}


def clean(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean(v) for v in obj]
    if isinstance(obj, tuple):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        return None if not np.isfinite(val) else val
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.strftime("%Y-%m-%d")
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(obj), ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def pct(x: float) -> float | None:
    return None if not np.isfinite(x) else float(x * 100.0)


def px_frame(df: pd.DataFrame, symbols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = df.copy()
    for col in ["open", "close", "adj_close", "volume"]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    if "adj_close" not in work.columns:
        work["adj_close"] = work["close"]
    work["px"] = work["adj_close"].where(work["adj_close"].gt(0), work["close"])
    if "open" in work.columns:
        factor = work["adj_close"] / work["close"]
        work["adj_open"] = work["open"] * factor.replace([np.inf, -np.inf], np.nan)
        work["adj_open"] = work["adj_open"].where(work["adj_open"].gt(0), work["open"])
    else:
        work["adj_open"] = np.nan
    dates = pd.DatetimeIndex(sorted(work["date"].dropna().unique()))
    close = work.pivot_table(index="date", columns="symbol", values="px", aggfunc="last").reindex(index=dates, columns=symbols).ffill()
    open_ = work.pivot_table(index="date", columns="symbol", values="adj_open", aggfunc="last").reindex(index=dates, columns=symbols)
    volume = work.pivot_table(index="date", columns="symbol", values="volume", aggfunc="last").reindex(index=dates, columns=symbols)
    return close, open_, volume


def ret_at(px: pd.Series, last_pos: int, window: int, skip: int = 0) -> float:
    a = last_pos - skip
    b = last_pos - skip - window
    if a < 0 or b < 0:
        return np.nan
    va = float(px.iloc[a])
    vb = float(px.iloc[b])
    if not (np.isfinite(va) and np.isfinite(vb) and vb > 0):
        return np.nan
    return va / vb - 1.0


def confirmed_last(condition: pd.Series, bars: int = 3) -> bool:
    if len(condition) < bars:
        return False
    return bool(condition.tail(bars).fillna(False).astype(bool).all())


def update_step_counter(state_market: dict[str, Any], latest_date: pd.Timestamp, dates: pd.DatetimeIndex, fallback_step: int) -> int:
    latest_str = latest_date.strftime("%Y-%m-%d")
    prev_date = state_market.get("latest_price_date")
    prev_step = state_market.get("latest_backtest_step")
    if prev_date and prev_step is not None:
        try:
            prev_ts = pd.Timestamp(prev_date)
            added = int((dates > prev_ts).sum())
            return int(prev_step) + max(0, added)
        except Exception:
            return int(prev_step)
    state_market["latest_price_date"] = latest_str
    return int(fallback_step)


def rebalance_counter_fields(
    latest_step: int,
    latest_index: int,
    dates: pd.DatetimeIndex,
    rebalance_step: int,
) -> dict[str, Any]:
    step = max(1, int(rebalance_step))
    latest_step_int = int(latest_step)
    days_since = int(latest_step_int % step)
    last_rebalance_step = latest_step_int - days_since
    step_offset = latest_step_int - int(latest_index)
    last_rebalance_index = max(0, min(int(latest_index), last_rebalance_step - step_offset))
    return {
        "component_rebalance_date": dates[last_rebalance_index].strftime("%Y-%m-%d"),
        "trading_days_since_rebalance": days_since,
        "rebalance_days_remaining": max(0, step - days_since),
        "rebalance_due_today_by_counter": days_since == 0,
        "rebalance_due_next_session_by_counter": bool((latest_step_int + 1) % step == 0),
        "rebalance_count_basis": "latest_backtest_step_counter",
    }


def live_cycle_baseline(market: str) -> dict[str, Any] | None:
    if not LIVE_CYCLE_BASELINE_PATH.exists():
        return None
    try:
        df = pd.read_csv(LIVE_CYCLE_BASELINE_PATH, dtype=str, low_memory=False)
    except Exception:
        return None
    if df.empty or "market" not in df.columns:
        return None
    df["market"] = df["market"].fillna("").astype(str).str.upper().str.strip()
    part = df[df["market"] == market.upper()].copy()
    if part.empty:
        return None
    if "created_at" in part.columns:
        part = part.sort_values("created_at")
    return part.tail(1).iloc[0].where(pd.notna(part.tail(1).iloc[0]), "").to_dict()


def apply_live_cycle_baseline(market_state: dict[str, Any], market: str, dates: pd.DatetimeIndex) -> dict[str, Any]:
    rec = live_cycle_baseline(market)
    if not rec:
        return market_state
    start_text = str(rec.get("cycle_rebalance_date") or rec.get("baseline_date") or "").strip()
    if not start_text:
        return market_state
    try:
        start = pd.Timestamp(start_text).normalize()
        latest = pd.Timestamp(market_state.get("latest_price_date")).normalize()
    except Exception:
        return market_state
    if pd.isna(start) or pd.isna(latest) or start > latest:
        return market_state

    date_index = pd.DatetimeIndex(pd.to_datetime(dates)).normalize()
    if date_index.empty:
        return market_state
    days_since = int(((date_index > start) & (date_index <= latest)).sum())
    step = int(market_state.get("rebalance_step_trading_days") or rec.get("rebalance_step_trading_days") or 1)
    step = max(1, step)
    due_today = days_since >= step
    due_next = (not due_today) and (days_since + 1 >= step)

    market_state.setdefault("research_component_rebalance_date", market_state.get("component_rebalance_date", ""))
    market_state["component_rebalance_date"] = start.strftime("%Y-%m-%d")
    market_state["live_cycle_rebalance_date"] = start.strftime("%Y-%m-%d")
    market_state["live_cycle_source_signal_date"] = str(rec.get("source_signal_date") or "")
    market_state["live_cycle_baseline_type"] = str(rec.get("baseline_type") or "manual_go_live_close")
    market_state["trading_days_since_rebalance"] = days_since
    market_state["rebalance_days_remaining"] = max(0, step - days_since)
    market_state["rebalance_due_today_by_counter"] = due_today
    market_state["rebalance_due_next_session_by_counter"] = due_next
    market_state["rebalance_count_basis"] = "live_go_live_cycle"
    if "action_signal" in market_state:
        if due_today:
            market_state["action_signal"] = "rebalance_live_cycle_target"
        elif due_next:
            market_state["action_signal"] = "next_session_live_cycle_rebalance_pending"
        else:
            market_state["action_signal"] = "hold_live_cycle_target"
    notes = str(market_state.get("notes") or "")
    live_note = f" Live cycle starts from {start.strftime('%Y-%m-%d')} because formal deployment was reset manually."
    if live_note.strip() not in notes:
        market_state["notes"] = (notes + live_note).strip()
    return market_state


def load_us_prices(symbols: list[str]) -> pd.DataFrame:
    path = DATA_DIR / "us_execution_prices_tail.csv"
    if not path.exists():
        path = DATA_DIR / "us_strategy_prices_tail.csv"
    df = pd.read_csv(path, dtype={"symbol": str}, parse_dates=["date"], low_memory=False)
    df = df[df["symbol"].isin(symbols)].copy()
    if df.empty:
        raise RuntimeError(f"no US prices found in {path}")
    return df


def run_us_v30_daily(state: dict[str, Any]) -> dict[str, Any]:
    US_OUT.mkdir(parents=True, exist_ok=True)
    spec = json.loads((RULE_DIR / "us_rule.json").read_text(encoding="utf-8"))
    themes: dict[str, list[str]] = spec["universe"]["themes"]
    signal_rules = spec["signal_rules"]
    overlay_rules = spec["overlay_rules"]
    risk_limits = spec["risk_limits"]
    all_symbols = sorted(set(sum(themes.values(), [])) | {"SMH", "XLK", "SPY", "QQQ", "SOXX"})
    prices = load_us_prices(all_symbols)
    close, _open, volume = px_frame(prices, all_symbols)
    dates = close.index
    if len(dates) < 252:
        raise RuntimeError("US live data needs at least 252 trading rows for the frozen eligibility rule")
    latest_date = dates[-1]
    last = len(dates) - 1
    lookback = int(signal_rules["momentum_lookback_trading_days"])
    skip = int(signal_rules["momentum_skip_trading_days"])

    smh = close["SMH"]
    xlk = close["XLK"]
    smh_ma200 = smh.rolling(200, min_periods=150).mean()
    smh_mom63 = ret_at(smh, last, 63, 0)
    base_risk_on = bool(np.isfinite(smh.iloc[-1]) and np.isfinite(smh_ma200.iloc[-1]) and smh.iloc[-1] > smh_ma200.iloc[-1])

    xlk_ma50 = xlk.rolling(50, min_periods=30).mean()
    xlk_ma150 = xlk.rolling(150, min_periods=100).mean()
    xlk_mom21 = xlk / xlk.shift(21) - 1.0
    xlk_mom63 = xlk / xlk.shift(63) - 1.0
    weak_condition = (xlk < xlk_ma50) | (xlk_mom21 < 0.0)
    hard_condition = (xlk < xlk_ma150) & (xlk_mom63 < -0.10)
    weak_confirmed = confirmed_last(weak_condition)
    hard_confirmed = confirmed_last(hard_condition)
    if not base_risk_on:
        gross_cap = 0.0
        regime = "risk_off"
    elif hard_confirmed:
        gross_cap = float(overlay_rules["hard_gross_cap"])
        regime = "hard_weak"
    elif weak_confirmed:
        gross_cap = float(overlay_rules["weak_gross_cap"])
        regime = "weak"
    else:
        gross_cap = float(overlay_rules["normal_gross_cap"])
        regime = "risk_on"

    theme_rows: list[dict[str, Any]] = []
    stock_rows: list[dict[str, Any]] = []
    min_obs = int(spec["universe"]["eligibility"]["dynamic_min_observations_at_decision_bar"])
    for theme_name, syms in themes.items():
        stock_moms: list[tuple[str, float]] = []
        mom63_vals: list[float] = []
        mom126_vals: list[float] = []
        for sym in syms:
            if sym not in close.columns:
                continue
            obs = int(close[sym].dropna().shape[0])
            m63 = ret_at(close[sym], last, lookback, skip)
            m126 = ret_at(close[sym], last, 126, skip)
            dollar_volume20 = float((close[sym] * volume[sym]).tail(20).median()) if sym in volume else np.nan
            eligible = bool(obs >= min_obs and np.isfinite(m63))
            if eligible:
                stock_moms.append((sym, m63))
                mom63_vals.append(m63)
                if np.isfinite(m126):
                    mom126_vals.append(m126)
            stock_rows.append(
                {
                    "symbol": sym,
                    "theme": theme_name,
                    "eligible": eligible,
                    "mom63_skip5_pct": pct(m63),
                    "mom126_skip5_pct": pct(m126),
                    "dollar_volume20": dollar_volume20,
                }
            )
        stock_moms.sort(key=lambda x: x[1], reverse=True)
        top2_mean = float(np.mean([x[1] for x in stock_moms[:2]])) if stock_moms else np.nan
        theme_mom63 = float(np.nanmean(mom63_vals)) if mom63_vals else np.nan
        theme_mom126 = float(np.nanmean(mom126_vals)) if mom126_vals else np.nan
        positive_breadth = float(np.mean([x > 0 for x in mom63_vals])) if mom63_vals else 0.0
        score = top2_mean + 0.30 * theme_mom63 + 0.15 * theme_mom126 + 0.05 * positive_breadth if np.isfinite(top2_mean) else np.nan
        theme_rows.append(
            {
                "theme": theme_name,
                "theme_score": score,
                "top2_stock_momentum_mean_pct": pct(top2_mean),
                "theme_mom63_pct": pct(theme_mom63),
                "theme_mom126_pct": pct(theme_mom126),
                "positive_breadth": positive_breadth,
                "top_symbols_by_momentum": "|".join(sym for sym, _ in stock_moms[:6]),
            }
        )

    theme_rows = sorted(theme_rows, key=lambda x: -999.0 if not np.isfinite(x["theme_score"]) else -float(x["theme_score"]))
    for rank, row in enumerate(theme_rows, 1):
        row["rank"] = rank
        row["selected"] = rank <= int(signal_rules["selected_theme_count"])

    selected_themes = [r["theme"] for r in theme_rows[: int(signal_rules["selected_theme_count"])]]
    accel_enabled = bool(
        selected_themes
        and np.isfinite(theme_rows[0]["theme_score"])
        and float(theme_rows[0]["theme_score"]) >= float(signal_rules["accelerator_enabled_when"]["top_theme_score_at_least"])
        and np.isfinite(smh_mom63)
        and smh_mom63 >= float(signal_rules["accelerator_enabled_when"]["SMH_63d_momentum_at_least"])
    )

    stock_rank = pd.DataFrame(stock_rows)
    if not stock_rank.empty:
        stock_rank["mom63_num"] = pd.to_numeric(stock_rank["mom63_skip5_pct"], errors="coerce")
        stock_rank = stock_rank.sort_values(["theme", "eligible", "mom63_num"], ascending=[True, False, False])

    targets: list[dict[str, Any]] = []
    if gross_cap <= 0.0:
        targets.append({"symbol": "CASH", "theme": "cash", "role": "risk_off_cash", "target_weight": 1.0})
    else:
        raw_targets: list[dict[str, Any]] = []
        theme_raw_weights = [float(signal_rules["top_theme_weight_before_caps"]), float(signal_rules["second_theme_weight_before_caps"])]
        for theme_idx, theme_name in enumerate(selected_themes):
            pool = [
                row
                for row in stock_rows
                if row["theme"] == theme_name and row["eligible"] and row["mom63_skip5_pct"] is not None
            ]
            pool.sort(key=lambda x: float(x["mom63_skip5_pct"]), reverse=True)
            keep_n = int(signal_rules["core_leader_count_per_selected_theme"])
            if accel_enabled:
                keep_n += int(signal_rules["accelerator_extra_count"])
            keep = pool[: max(1, keep_n)]
            if not keep:
                continue
            theme_gross = min(
                theme_raw_weights[min(theme_idx, len(theme_raw_weights) - 1)] * gross_cap,
                float(risk_limits["single_theme_gross_cap"]),
            )
            per = theme_gross / len(keep)
            for k, row in enumerate(keep):
                raw_targets.append(
                    {
                        "symbol": row["symbol"],
                        "theme": theme_name,
                        "role": "core_leader" if k < int(signal_rules["core_leader_count_per_selected_theme"]) else "accelerator",
                        "raw_weight": per,
                    }
                )
        by_symbol: dict[str, dict[str, Any]] = {}
        for row in raw_targets:
            sym = row["symbol"]
            if sym not in by_symbol:
                by_symbol[sym] = dict(row)
            else:
                by_symbol[sym]["raw_weight"] += row["raw_weight"]
        used = 0.0
        for row in sorted(by_symbol.values(), key=lambda x: x["raw_weight"], reverse=True):
            weight = min(float(row["raw_weight"]), float(risk_limits["single_symbol_weight_cap"]))
            used += weight
            targets.append({"symbol": row["symbol"], "theme": row["theme"], "role": row["role"], "target_weight": weight})
        cash = max(0.0, 1.0 - used)
        if cash > 1e-9:
            targets.append({"symbol": "CASH", "theme": "cash", "role": "residual_cash", "target_weight": cash})

    us_state = state.setdefault("US", {})
    latest_step = update_step_counter(us_state, latest_date, dates, fallback_step=len(dates) - 1)
    rebalance_step = int(signal_rules["rebalance_step_trading_days"])
    rebalance_fields = rebalance_counter_fields(latest_step, len(dates) - 1, dates, rebalance_step)
    us_state.update(
        {
            "latest_price_date": latest_date.strftime("%Y-%m-%d"),
            "latest_backtest_step": latest_step,
            "regime": regime,
            "target_position_if_rebalanced": targets,
        }
    )

    market_state = {
        "market": "US",
        "latest_price_date": latest_date,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "frozen_strategy": spec["production_name"],
        "regime": regime,
        "base_risk_on": base_risk_on,
        "weak_confirmed_3d": weak_confirmed,
        "hard_confirmed_3d": hard_confirmed,
        "gross_cap": gross_cap,
        "smh_mom63_pct": pct(smh_mom63),
        "xlk_mom21_pct": pct(float(xlk_mom21.iloc[-1])),
        "xlk_mom63_pct": pct(float(xlk_mom63.iloc[-1])),
        "accelerator_enabled_if_rebalanced": accel_enabled,
        "latest_backtest_step": latest_step,
        "rebalance_step_trading_days": rebalance_step,
        **rebalance_fields,
        "uses_2025_2026_for_tuning": False,
        "notes": "Cloud package uses compact tail data and frozen V30 parameters. It does not retrain.",
    }
    market_state = apply_live_cycle_baseline(market_state, "US", dates)

    pd.DataFrame([market_state]).to_csv(US_OUT / "latest_market_state.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(theme_rows).to_csv(US_OUT / "latest_theme_rank.csv", index=False, encoding="utf-8-sig")
    stock_rank.drop(columns=["mom63_num"], errors="ignore").to_csv(US_OUT / "latest_candidate_pool.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(targets).to_csv(US_OUT / "latest_target_position.csv", index=False, encoding="utf-8-sig")
    action_report = {
        "market_state": market_state,
        "target_position_if_rebalanced": targets,
        "top_themes": theme_rows[:10],
        "top_candidates": stock_rank.drop(columns=["mom63_num"], errors="ignore").head(50).to_dict(orient="records"),
    }
    write_json(US_OUT / "latest_action_report.json", action_report)
    return action_report


US_SIGNAL_SYMBOLS = {"SMH", "XLK", "SPY", "QQQ", "SOXX"}
US_COMPONENTS = ["V30_CAP65_ATTACK", "V31_MAX_PROFIT", "US981A_TRAIN_ONLY"]

DYNAMIC_FORMAL_THEMES = {
    "ai_compute",
    "ai_data_center_connectivity",
    "cloud_ai_infra",
    "cloud_infra",
    "compound_semiconductor",
    "data_center_reit",
    "electronics_manufacturing",
    "grid_engineering",
    "industrial_automation",
    "memory_storage",
    "optical_communication",
    "power_generation",
    "power_semiconductor",
    "semi_equipment",
    "semi_foundry_idm",
    "server_network",
    "software_platform",
    "space_ev_defense",
    "thermal_power",
    "us981a_ai_theme",
}

DYNAMIC_FORMAL_EXCLUDED_THEMES = {
    "crypto_ai_power",
    "crypto_infra",
    "market_index_etf",
    "proxy",
    "semiconductor_etf",
    "technology_etf",
    "technology_index_etf",
    "us_unclassified",
    "us_unclassified_stock",
}

DYNAMIC_FORMAL_DEFAULT_CONFIG: dict[str, Any] = {
    "enabled": True,
    "strategy_name": "FORMAL99_RB5_T3_S3_N8_SC30_SMH200",
    "source_sleeve": "DYNAMIC_FORMAL99",
    "scan_top_n": 800,
    "formal_pool_max_symbols": 99,
    "lookback": 63,
    "skip": 5,
    "rebalance_n": 5,
    "selected_theme_count": 3,
    "stocks_per_theme": 3,
    "final_top_n": 8,
    "min_obs": 252,
    "min_dollar_volume20": 20_000_000,
    "min_mom63": 0.0,
    "min_scan_score": 0.30,
    "single_symbol_cap": 0.25,
    "dynamic_weight": 0.35,
    "gate_ma_window": 20,
    "gate_mom_window": 20,
    "gate_min_mom": 0.05,
    "gate_dd_window": 60,
    "gate_max_dd": 0.20,
    "max_scan_staleness_days": 3,
    "risk_mode": "hard_smh_ma200",
}


def us_rule_symbols(spec: dict[str, Any]) -> list[str]:
    universe = spec.get("universe", {})
    symbols: set[str] = set(US_SIGNAL_SYMBOLS)
    for themes in [
        universe.get("themes", {}),
        spec.get("v30_rule", {}).get("themes", {}),
        spec.get("v31_rule", {}).get("themes", {}),
    ]:
        if isinstance(themes, dict):
            for syms in themes.values():
                symbols.update(str(s) for s in syms)
    for key in ["us981a_symbols", "symbols"]:
        vals = universe.get(key, [])
        if isinstance(vals, list):
            symbols.update(str(s) for s in vals)
    vals = spec.get("us981a_rule", {}).get("symbols", [])
    if isinstance(vals, list):
        symbols.update(str(s) for s in vals)
    return sorted(s for s in symbols if s)


def confirmed_at(condition: pd.Series, pos: int, bars: int = 3) -> bool:
    if pos < bars - 1:
        return False
    return bool(condition.iloc[pos - bars + 1 : pos + 1].fillna(False).astype(bool).all())


def us_market_budget_for_signal(
    close: pd.DataFrame,
    signal_i: int,
    normal_gross: float,
    weak_gross: float,
    hard_gross: float,
    risk_mode: str = "hard_smh_ma200",
    overlay_symbol: str = "XLK",
    confirm_days: int = 3,
) -> dict[str, Any]:
    smh = close["SMH"]
    xlk = close[overlay_symbol]
    smh_ma200 = smh.rolling(200, min_periods=150).mean()
    smh_mom63 = ret_at(smh, signal_i, 63, 0)
    base_risk_on = bool(
        np.isfinite(smh.iloc[signal_i])
        and np.isfinite(smh_ma200.iloc[signal_i])
        and smh.iloc[signal_i] > smh_ma200.iloc[signal_i]
    )
    xlk_ma50 = xlk.rolling(50, min_periods=30).mean()
    xlk_ma150 = xlk.rolling(150, min_periods=100).mean()
    xlk_mom21 = xlk / xlk.shift(21) - 1.0
    xlk_mom63 = xlk / xlk.shift(63) - 1.0
    weak_condition = (xlk < xlk_ma50) | (xlk_mom21 < 0.0)
    hard_condition = (xlk < xlk_ma150) & (xlk_mom63 < -0.10)
    weak_confirmed = confirmed_at(weak_condition, signal_i, confirm_days)
    hard_confirmed = confirmed_at(hard_condition, signal_i, confirm_days)

    if risk_mode == "hard_smh_ma200" and not base_risk_on:
        gross_cap = 0.0
        regime = "risk_off"
    elif hard_confirmed:
        gross_cap = hard_gross
        regime = "hard_weak"
    elif weak_confirmed:
        gross_cap = weak_gross
        regime = "weak"
    else:
        gross_cap = normal_gross
        regime = "risk_on" if base_risk_on else "risk_on_soft"
    return {
        "gross_cap": float(gross_cap),
        "regime": regime,
        "base_risk_on": base_risk_on,
        "weak_confirmed_3d": weak_confirmed,
        "hard_confirmed_3d": hard_confirmed,
        "smh_mom63": smh_mom63,
        "xlk_mom21": float(xlk_mom21.iloc[signal_i]) if np.isfinite(xlk_mom21.iloc[signal_i]) else np.nan,
        "xlk_mom63": float(xlk_mom63.iloc[signal_i]) if np.isfinite(xlk_mom63.iloc[signal_i]) else np.nan,
    }


def us_score_theme_rows(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    themes: dict[str, list[str]],
    lookback: int,
    skip: int,
    min_obs: int,
    signal_i: int,
    source_sleeve: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    theme_rows: list[dict[str, Any]] = []
    stock_rows: list[dict[str, Any]] = []
    for theme_name, syms in themes.items():
        stock_moms: list[tuple[str, float]] = []
        mom_vals: list[float] = []
        mom126_vals: list[float] = []
        for sym in syms:
            if sym not in close.columns:
                continue
            px = close[sym]
            obs = int(px.iloc[: signal_i + 1].dropna().shape[0])
            mom = ret_at(px, signal_i, lookback, skip)
            mom126 = ret_at(px, signal_i, 126, skip)
            dollar_volume20 = (
                float((close[sym] * volume[sym]).iloc[: signal_i + 1].tail(20).median())
                if sym in volume
                else np.nan
            )
            eligible = bool(obs >= min_obs and np.isfinite(mom))
            if eligible:
                stock_moms.append((sym, mom))
                mom_vals.append(mom)
                if np.isfinite(mom126):
                    mom126_vals.append(mom126)
            stock_rows.append(
                {
                    "source_sleeve": source_sleeve,
                    "symbol": sym,
                    "theme": theme_name,
                    "eligible": eligible,
                    "mom63_skip5_pct": pct(mom),
                    "mom126_skip5_pct": pct(mom126),
                    "dollar_volume20": dollar_volume20,
                }
            )
        stock_moms.sort(key=lambda x: x[1], reverse=True)
        top2_mean = float(np.mean([x[1] for x in stock_moms[:2]])) if stock_moms else np.nan
        theme_mom = float(np.nanmean(mom_vals)) if mom_vals else np.nan
        theme_mom126 = float(np.nanmean(mom126_vals)) if mom126_vals else np.nan
        breadth = float(np.mean([x > 0 for x in mom_vals])) if mom_vals else 0.0
        score = top2_mean + 0.30 * theme_mom + 0.15 * theme_mom126 + 0.05 * breadth if np.isfinite(top2_mean) else np.nan
        theme_rows.append(
            {
                "source_sleeve": source_sleeve,
                "theme": theme_name,
                "theme_score": score,
                "top2_stock_momentum_mean_pct": pct(top2_mean),
                "theme_mom63_pct": pct(theme_mom),
                "theme_mom126_pct": pct(theme_mom126),
                "positive_breadth": breadth,
                "top_symbols_by_momentum": "|".join(sym for sym, _ in stock_moms[:6]),
            }
        )
    theme_rows = sorted(theme_rows, key=lambda x: -999.0 if not np.isfinite(x["theme_score"]) else -float(x["theme_score"]))
    return theme_rows, stock_rows


def us_theme_component_targets(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    component: str,
    themes: dict[str, list[str]],
    params: dict[str, Any],
    signal_i: int,
) -> dict[str, Any]:
    budget = us_market_budget_for_signal(
        close,
        signal_i,
        normal_gross=float(params["gross_cap"]),
        weak_gross=float(params.get("weak_gross_cap", 1.0)),
        hard_gross=float(params.get("hard_gross_cap", 0.75)),
        risk_mode=str(params.get("risk_mode", "hard_smh_ma200")),
        overlay_symbol=str(params.get("overlay_signal_symbol", "XLK")),
        confirm_days=int(params.get("overlay_confirm_days", 3)),
    )
    theme_rows, stock_rows = us_score_theme_rows(
        close,
        volume,
        themes,
        int(params["lookback"]),
        int(params.get("skip", 5)),
        int(params.get("dynamic_min_obs", 252)),
        signal_i,
        component,
    )
    for rank, row in enumerate(theme_rows, 1):
        row["rank"] = rank
        row["selected"] = rank <= int(params.get("selected_theme_count", 2))

    selected_themes = [r["theme"] for r in theme_rows[: int(params.get("selected_theme_count", 2))]]
    top_score = float(theme_rows[0]["theme_score"]) if theme_rows and np.isfinite(theme_rows[0]["theme_score"]) else np.nan
    smh_mom63 = budget["smh_mom63"]
    accel_enabled = bool(
        selected_themes
        and np.isfinite(top_score)
        and top_score >= float(params.get("accel_score_threshold", 0.35))
        and np.isfinite(smh_mom63)
        and smh_mom63 >= float(params.get("accel_regime_mom63_threshold", 0.0))
    )

    gross_cap = float(budget["gross_cap"])
    targets: list[dict[str, Any]] = []
    if gross_cap <= 0.0:
        targets.append(
            {
                "symbol": "CASH",
                "theme": "cash",
                "role": "risk_off_cash",
                "target_weight": 1.0,
                "source_sleeve": component,
            }
        )
    else:
        raw_targets: list[dict[str, Any]] = []
        theme_weights = [float(params.get("top_theme_weight", 0.85)), float(params.get("second_theme_weight", 0.15))]
        for theme_idx, theme_name in enumerate(selected_themes):
            pool = [
                row
                for row in stock_rows
                if row["theme"] == theme_name and row["eligible"] and row["mom63_skip5_pct"] is not None
            ]
            pool.sort(key=lambda x: float(x["mom63_skip5_pct"]), reverse=True)
            keep_n = int(params.get("core_leader_count", 1))
            if accel_enabled:
                keep_n += int(params.get("accelerator_extra_count", 4))
            keep = pool[: max(1, keep_n)]
            if not keep:
                continue
            theme_gross = min(
                theme_weights[min(theme_idx, len(theme_weights) - 1)] * gross_cap,
                float(params.get("theme_gross_cap", gross_cap)),
            )
            per = theme_gross / len(keep)
            for k, row in enumerate(keep):
                raw_targets.append(
                    {
                        "symbol": row["symbol"],
                        "theme": theme_name,
                        "role": "core_leader" if k < int(params.get("core_leader_count", 1)) else "accelerator",
                        "raw_weight": per,
                        "source_sleeve": component,
                    }
                )
        by_symbol: dict[str, dict[str, Any]] = {}
        for row in raw_targets:
            sym = row["symbol"]
            if sym not in by_symbol:
                by_symbol[sym] = dict(row)
            else:
                by_symbol[sym]["raw_weight"] += row["raw_weight"]
        used = 0.0
        for row in sorted(by_symbol.values(), key=lambda x: x["raw_weight"], reverse=True):
            weight = min(float(row["raw_weight"]), float(params.get("single_weight_cap", 0.5)))
            used += weight
            targets.append(
                {
                    "symbol": row["symbol"],
                    "theme": row["theme"],
                    "role": row["role"],
                    "target_weight": weight,
                    "source_sleeve": component,
                }
            )
        cash = max(0.0, 1.0 - used)
        if cash > 1e-9:
            targets.append(
                {
                    "symbol": "CASH",
                    "theme": "cash",
                    "role": "residual_cash",
                    "target_weight": cash,
                    "source_sleeve": component,
                }
            )

    return {
        "component": component,
        "targets": targets,
        "theme_rows": theme_rows,
        "stock_rows": stock_rows,
        "regime": budget["regime"],
        "gross_cap": gross_cap,
        "component_score": top_score,
        "accelerator_enabled": accel_enabled,
        "budget": budget,
    }


def us_rank_weights(scores: pd.Series, gross: float, cap: float, mode: str) -> pd.Series:
    scores = scores.dropna()
    if scores.empty:
        return scores
    if mode == "rank":
        raw = pd.Series(np.arange(len(scores), 0, -1, dtype=float), index=scores.index)
    elif mode == "equal":
        raw = pd.Series(1.0, index=scores.index)
    else:
        raw = scores.clip(lower=0.0)
        if raw.sum() <= 0:
            raw = pd.Series(1.0, index=scores.index)
    target = raw / raw.sum() * gross
    capped = target.copy()
    for _ in range(20):
        over = capped > cap
        if not over.any():
            break
        excess = float((capped[over] - cap).sum())
        capped[over] = cap
        under = ~over
        if not under.any() or capped[under].sum() <= 0:
            break
        capped[under] += capped[under] / capped[under].sum() * excess
    return capped.clip(upper=cap)


def us981a_score_frame(close: pd.DataFrame, symbols: list[str], score_name: str) -> pd.DataFrame:
    px = close[symbols]
    r20 = px / px.shift(20) - 1.0
    r60 = px / px.shift(60) - 1.0
    r120 = px / px.shift(120) - 1.0
    r252 = px / px.shift(252) - 1.0
    if score_name == "ai_mid_trend":
        return 0.15 * r20 + 0.55 * r60 + 0.30 * r120
    if score_name == "long_trend":
        return 0.20 * r60 + 0.45 * r120 + 0.35 * r252
    if score_name == "accel_20_60_120":
        return 0.30 * r20 + 0.45 * r60 + 0.25 * r120
    return 0.15 * r20 + 0.55 * r60 + 0.30 * r120


def us981a_component_targets(close: pd.DataFrame, component: str, rule: dict[str, Any], signal_i: int) -> dict[str, Any]:
    symbols = [s for s in rule["symbols"] if s in close.columns]
    score = us981a_score_frame(close, symbols, str(rule.get("score_name", "ai_mid_trend")))
    r60 = close[symbols] / close[symbols].shift(60) - 1.0
    r120 = close[symbols] / close[symbols].shift(120) - 1.0
    s = score.iloc[signal_i].dropna()
    s = s[(r60.iloc[signal_i].reindex(s.index) >= float(rule.get("min_ret60", 0.1))) & (r120.iloc[signal_i].reindex(s.index) >= float(rule.get("min_ret120", 0.05)))]
    s = s.sort_values(ascending=False).head(int(rule.get("top_n", 3)))
    weights = us_rank_weights(s, float(rule.get("gross", 1.25)), float(rule.get("max_single_weight", 0.4)), str(rule.get("weight_mode", "rank")))
    targets: list[dict[str, Any]] = []
    for sym, weight in weights.items():
        targets.append(
            {
                "symbol": str(sym),
                "theme": str(rule.get("theme", "us981a_ai_theme")),
                "role": "theme_leader",
                "target_weight": float(weight),
                "source_sleeve": component,
            }
        )
    used = float(weights.sum()) if len(weights) else 0.0
    if used <= 0.0:
        targets.append({"symbol": "CASH", "theme": "cash", "role": "risk_off_cash", "target_weight": 1.0, "source_sleeve": component})
    elif used < 1.0:
        targets.append({"symbol": "CASH", "theme": "cash", "role": "residual_cash", "target_weight": 1.0 - used, "source_sleeve": component})

    stock_rows = []
    for sym in symbols:
        stock_rows.append(
            {
                "source_sleeve": component,
                "symbol": sym,
                "theme": str(rule.get("theme", "us981a_ai_theme")),
                "eligible": bool(sym in weights.index),
                "mom63_skip5_pct": pct(float(r60[sym].iloc[signal_i])) if np.isfinite(r60[sym].iloc[signal_i]) else None,
                "mom126_skip5_pct": pct(float(r120[sym].iloc[signal_i])) if np.isfinite(r120[sym].iloc[signal_i]) else None,
                "component_score": float(score[sym].iloc[signal_i]) if np.isfinite(score[sym].iloc[signal_i]) else np.nan,
            }
        )
    top_score = float(s.iloc[0]) if len(s) else np.nan
    return {
        "component": component,
        "targets": targets,
        "theme_rows": [
            {
                "source_sleeve": component,
                "theme": str(rule.get("theme", "us981a_ai_theme")),
                "theme_score": top_score,
                "top_symbols_by_momentum": "|".join(s.index.astype(str).tolist()),
                "selected": len(s) > 0,
                "rank": 1,
            }
        ],
        "stock_rows": stock_rows,
        "regime": "risk_on" if used > 0.0 else "risk_off",
        "gross_cap": used,
        "component_score": top_score,
        "accelerator_enabled": len(s) > 0,
        "budget": {},
    }


def us_component_targets_for_signal(
    component: str,
    spec: dict[str, Any],
    close: pd.DataFrame,
    volume: pd.DataFrame,
    signal_i: int,
) -> dict[str, Any]:
    if component == "V30_CAP65_ATTACK":
        rule = spec["v30_rule"]
        return us_theme_component_targets(close, volume, component, rule["themes"], rule["params"], signal_i)
    if component == "V31_MAX_PROFIT":
        rule = spec["v31_rule"]
        return us_theme_component_targets(close, volume, component, rule["themes"], rule["params"], signal_i)
    if component == "US981A_TRAIN_ONLY":
        return us981a_component_targets(close, component, spec["us981a_rule"], signal_i)
    raise ValueError(component)


def us_component_rebalance_n(spec: dict[str, Any], component: str) -> int:
    if component == "US981A_TRAIN_ONLY":
        return int(spec["us981a_rule"].get("rebalance_n", 5))
    if component == "V31_MAX_PROFIT":
        return int(spec["v31_rule"]["params"].get("rebalance_step", 10))
    return int(spec["v30_rule"]["params"].get("rebalance_step", 10))


def simulate_us_component_equities(
    spec: dict[str, Any],
    close: pd.DataFrame,
    volume: pd.DataFrame,
) -> pd.DataFrame:
    returns = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out: dict[str, pd.Series] = {}
    for component in US_COMPONENTS:
        rebalance_n = us_component_rebalance_n(spec, component)
        start_i = max(252, rebalance_n + 1)
        current_w: dict[str, float] = {}
        equity = 1.0
        vals: list[float] = []
        for i, _dt in enumerate(close.index):
            if i == 0:
                vals.append(equity)
                continue
            if i >= start_i and (i - start_i) % rebalance_n == 0:
                signal_i = i - 1
                target_pack = us_component_targets_for_signal(component, spec, close, volume, signal_i)
                current_w = {
                    str(row["symbol"]): float(row["target_weight"])
                    for row in target_pack["targets"]
                    if str(row["symbol"]) != "CASH"
                }
            day_ret = 0.0
            for sym, weight in current_w.items():
                if sym in returns.columns:
                    day_ret += weight * float(returns[sym].iloc[i])
            equity *= 1.0 + day_ret
            vals.append(equity)
        out[component] = pd.Series(vals, index=close.index, dtype=float)
    return pd.DataFrame(out)


def us_component_score_frame(eq: pd.DataFrame, score_name: str) -> pd.DataFrame:
    r20 = eq / eq.shift(20) - 1.0
    r60 = eq / eq.shift(60) - 1.0
    r120 = eq / eq.shift(120) - 1.0
    r252 = eq / eq.shift(252) - 1.0
    dd60 = eq / eq.rolling(60).max() - 1.0
    if score_name == "long":
        return 0.20 * r60 + 0.45 * r120 + 0.35 * r252
    if score_name == "mid":
        return 0.15 * r20 + 0.55 * r60 + 0.30 * r120
    if score_name == "recent":
        return 0.50 * r20 + 0.35 * r60 + 0.15 * r120
    if score_name == "balanced":
        return 0.25 * r20 + 0.35 * r60 + 0.30 * r120 + 0.10 * dd60
    return 0.15 * r20 + 0.55 * r60 + 0.30 * r120


def us_weights_for_profile(profile: str, count: int) -> np.ndarray:
    table = {
        "top1": [1.0],
        "eq2": [0.5, 0.5],
        "eq3": [1 / 3, 1 / 3, 1 / 3],
        "p50_30_20": [0.5, 0.3, 0.2],
    }
    vals = table.get(profile, table["top1"])[:count]
    w = np.array(vals, dtype=float)
    return w / w.sum()


def select_us_component_weights(eq: pd.DataFrame, rule: dict[str, Any], signal_i: int) -> dict[str, float]:
    scores = us_component_score_frame(eq, str(rule.get("score_name", "mid")))
    s = scores.iloc[signal_i].dropna().sort_values(ascending=False)
    fallback = str(rule.get("fallback", "V30_CAP65_ATTACK"))
    if s.empty or float(s.iloc[0]) < float(rule.get("min_top_score", 0.0)):
        return {fallback: 1.0}
    mode = str(rule.get("mode", "top1"))
    profile = str(rule.get("weight_profile", "top1"))
    if mode == "top3":
        selected = list(s.head(3).index)
        profile = "eq3" if profile == "eq3" else profile
    elif mode == "top2":
        selected = list(s.head(2).index)
        profile = "eq2"
    elif mode == "gap_blend" and len(s) > 1 and float(s.iloc[0] - s.iloc[1]) <= float(rule.get("gap_threshold", 0.0)):
        selected = list(s.head(2).index)
        profile = "eq2"
    else:
        selected = [str(s.index[0])]
        profile = "top1"
    weights = us_weights_for_profile(profile, len(selected))
    return {str(sym): float(w) for sym, w in zip(selected, weights)}


def combine_us_component_weights(stable: dict[str, float], old: dict[str, float], stable_weight: float, old_weight: float) -> dict[str, float]:
    out = {component: 0.0 for component in US_COMPONENTS}
    for comp, w in stable.items():
        out[comp] = out.get(comp, 0.0) + stable_weight * float(w)
    for comp, w in old.items():
        out[comp] = out.get(comp, 0.0) + old_weight * float(w)
    return {k: v for k, v in out.items() if v > 1e-12}


def combine_us_targets(component_targets: dict[str, list[dict[str, Any]]], component_weights: dict[str, float], single_cap: float) -> list[dict[str, Any]]:
    by_symbol: dict[str, dict[str, Any]] = {}
    for component, sleeve_weight in component_weights.items():
        for row in component_targets.get(component, []):
            sym = str(row.get("symbol", ""))
            if not sym:
                continue
            if sym == "CASH":
                continue
            weight = float(row.get("target_weight", 0.0)) * float(sleeve_weight)
            if abs(weight) <= 1e-12:
                continue
            if sym not in by_symbol:
                by_symbol[sym] = {
                    "symbol": sym,
                    "theme": str(row.get("theme", "")),
                    "role": str(row.get("role", "")),
                    "target_weight": 0.0,
                    "source_sleeve": [],
                }
            by_symbol[sym]["target_weight"] += weight
            src = by_symbol[sym]["source_sleeve"]
            if component not in src:
                src.append(component)
            if by_symbol[sym]["theme"] == "cash" and str(row.get("theme", "")) != "cash":
                by_symbol[sym]["theme"] = str(row.get("theme", ""))
    excess = 0.0
    for sym, row in by_symbol.items():
        if sym == "CASH":
            continue
        if float(row["target_weight"]) > single_cap:
            excess += float(row["target_weight"]) - single_cap
            row["target_weight"] = single_cap
            row["role"] = f"{row['role']}|global_cap"
    if excess > 1e-12:
        cash = by_symbol.setdefault(
            "CASH",
            {"symbol": "CASH", "theme": "cash", "role": "residual_cash", "target_weight": 0.0, "source_sleeve": []},
        )
        cash["target_weight"] += excess
        if "global_cap" not in cash["source_sleeve"]:
            cash["source_sleeve"].append("global_cap")
    stock_gross = sum(float(row["target_weight"]) for sym, row in by_symbol.items() if sym != "CASH")
    if stock_gross < 1.0 - 1e-12:
        cash = by_symbol.setdefault(
            "CASH",
            {"symbol": "CASH", "theme": "cash", "role": "residual_cash", "target_weight": 0.0, "source_sleeve": []},
        )
        cash["target_weight"] += 1.0 - stock_gross
        if "residual" not in cash["source_sleeve"]:
            cash["source_sleeve"].append("residual")
    rows = []
    for row in by_symbol.values():
        row = dict(row)
        row["source_sleeve"] = "+".join(row["source_sleeve"])
        row["execution_group"] = "main" if str(row["symbol"]) == "CASH" or float(row["target_weight"]) >= 0.025 else "micro"
        rows.append(row)
    return sorted(rows, key=lambda x: (str(x["symbol"]) == "CASH", -float(x["target_weight"])))


def cap_weights_pro_rata(raw_weights: dict[str, float], target_total: float, cap: float) -> dict[str, float]:
    clean = {str(sym): max(0.0, float(weight)) for sym, weight in raw_weights.items() if float(weight) > 1e-12}
    if not clean or target_total <= 1e-12:
        return {}
    target_total = min(float(target_total), float(cap) * len(clean))
    fixed: dict[str, float] = {}
    active = dict(clean)
    remaining = target_total
    while active and remaining > 1e-12:
        active_total = sum(active.values())
        if active_total <= 1e-12:
            proposed = {sym: remaining / len(active) for sym in active}
        else:
            proposed = {sym: remaining * weight / active_total for sym, weight in active.items()}
        over_cap = [sym for sym, weight in proposed.items() if weight > cap + 1e-12]
        if not over_cap:
            fixed.update(proposed)
            break
        for sym in over_cap:
            fixed[sym] = float(cap)
            remaining -= float(cap)
            active.pop(sym, None)
    for sym in active:
        fixed.setdefault(sym, 0.0)
    drift = target_total - sum(fixed.values())
    if abs(drift) > 1e-10:
        room = [sym for sym, weight in fixed.items() if weight + drift <= cap + 1e-10 and weight + drift >= -1e-10]
        if room:
            fixed[room[0]] = max(0.0, fixed[room[0]] + drift)
    return fixed


def apply_us_top_n_cap_execution_overlay(targets: list[dict[str, Any]], top_n: int = 6, cap: float = 0.25) -> list[dict[str, Any]]:
    raw_rows = [dict(row) for row in targets]
    stock_rows = [
        row
        for row in raw_rows
        if str(row.get("symbol", "")) != "CASH" and float(row.get("target_weight", 0.0) or 0.0) > 1e-12
    ]
    if not stock_rows:
        cash = next((dict(row) for row in raw_rows if str(row.get("symbol", "")) == "CASH"), None)
        if cash is None:
            cash = {"symbol": "CASH", "theme": "cash", "role": "risk_off_cash", "target_weight": 1.0}
        cash["execution_group"] = "main"
        cash["execution_overlay"] = f"US_TOP{top_n}_CAP{int(cap * 100)}"
        return [cash]

    stock_gross = sum(float(row.get("target_weight", 0.0) or 0.0) for row in stock_rows)
    target_gross = min(1.0, max(0.0, stock_gross))
    selected = sorted(stock_rows, key=lambda row: float(row.get("target_weight", 0.0) or 0.0), reverse=True)[:top_n]
    capped = cap_weights_pro_rata(
        {str(row["symbol"]): float(row.get("target_weight", 0.0) or 0.0) for row in selected},
        target_total=target_gross,
        cap=cap,
    )

    out: list[dict[str, Any]] = []
    overlay_name = f"US_TOP{top_n}_CAP{int(cap * 100)}"
    for row in selected:
        sym = str(row["symbol"])
        weight = float(capped.get(sym, 0.0))
        if weight <= 1e-12:
            continue
        rr = dict(row)
        rr["raw_target_weight"] = float(row.get("target_weight", 0.0) or 0.0)
        rr["target_weight"] = weight
        rr["role"] = f"{rr.get('role', '')}|top{top_n}_cap{int(cap * 100)}_exec"
        rr["execution_group"] = "main"
        rr["execution_overlay"] = overlay_name
        out.append(rr)

    stock_total = sum(float(row["target_weight"]) for row in out)
    if target_gross < 1.0 - 1e-12:
        out.append(
            {
                "symbol": "CASH",
                "theme": "cash",
                "role": "residual_cash",
                "target_weight": 1.0 - target_gross,
                "source_sleeve": "execution_overlay",
                "execution_group": "main",
                "execution_overlay": overlay_name,
            }
        )
    elif abs(stock_total - target_gross) > 1e-8 and out:
        out[-1]["target_weight"] = max(0.0, float(out[-1]["target_weight"]) + (target_gross - stock_total))

    return sorted(out, key=lambda x: (str(x["symbol"]) == "CASH", -float(x["target_weight"])))


def us_combo_pack_for_signal(
    spec: dict[str, Any],
    close: pd.DataFrame,
    volume: pd.DataFrame,
    component_eq: pd.DataFrame,
    signal_i: int,
) -> dict[str, Any]:
    current_packs = {
        component: us_component_targets_for_signal(component, spec, close, volume, signal_i)
        for component in US_COMPONENTS
    }
    stable_rule = spec["meta_rules"]["stable_top3"]
    old_rule = spec["meta_rules"]["old_top1"]
    stable_component_weights = select_us_component_weights(component_eq, stable_rule, signal_i)
    old_component_weights = select_us_component_weights(component_eq, old_rule, signal_i)
    meta_weights = spec.get("meta_weights", {})
    component_weights = combine_us_component_weights(
        stable_component_weights,
        old_component_weights,
        float(meta_weights.get("META_STABLE_TOP3_SELECTED", 0.85)),
        float(meta_weights.get("OLD_STRICT_TOP1_SWITCH", 0.15)),
    )
    raw_targets = combine_us_targets(
        {k: v["targets"] for k, v in current_packs.items()},
        component_weights,
        float(spec.get("risk_limits", {}).get("combo_single_symbol_weight_cap", 0.5)),
    )
    targets = apply_us_top_n_cap_execution_overlay(raw_targets, top_n=6, cap=0.25)
    return {
        "current_packs": current_packs,
        "stable_component_weights": stable_component_weights,
        "old_component_weights": old_component_weights,
        "component_weights": component_weights,
        "raw_targets": raw_targets,
        "targets": targets,
    }


def simulate_us_combo_base_equity(
    spec: dict[str, Any],
    close: pd.DataFrame,
    volume: pd.DataFrame,
    component_eq: pd.DataFrame,
) -> pd.Series:
    returns = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    min_rebalance = min(us_component_rebalance_n(spec, c) for c in US_COMPONENTS)
    start_i = max(252, min_rebalance + 1)
    current_w: dict[str, float] = {}
    equity = 1.0
    vals: list[float] = []
    for i, _dt in enumerate(close.index):
        if i > 0:
            day_ret = 0.0
            for sym, weight in current_w.items():
                if sym in returns.columns:
                    day_ret += float(weight) * float(returns[sym].iloc[i])
            equity *= 1.0 + day_ret
        vals.append(float(equity))
        if i >= start_i and (i - start_i) % min_rebalance == 0:
            signal_i = i - 1
            pack = us_combo_pack_for_signal(spec, close, volume, component_eq, signal_i)
            current_w = {
                str(row["symbol"]): float(row["target_weight"])
                for row in pack["targets"]
                if str(row.get("symbol", "")) != "CASH"
            }
    return pd.Series(vals, index=close.index, name="US_COMBO_BASE_LIVE", dtype=float)


def us_dynamic_formal_config(spec: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(DYNAMIC_FORMAL_DEFAULT_CONFIG)
    user_cfg = spec.get("dynamic_formal_sleeve", {})
    if isinstance(user_cfg, dict):
        cfg.update(user_cfg)
    return cfg


def us_scan_theme_series(universe: pd.DataFrame, symbols: list[str]) -> pd.Series:
    meta = universe.drop_duplicates("symbol", keep="first").set_index("symbol").reindex(symbols)
    meta["symbol"] = meta.index.astype(str)
    for col in ["trade_group", "parent_group", "execution_theme", "name"]:
        if col not in meta.columns:
            meta[col] = ""
        meta[col] = meta[col].fillna("").astype(str)
    return meta.apply(choose_scan_theme, axis=1).astype(str)


def us_formal_ret(close: pd.DataFrame, signal_i: int, window: int, skip: int) -> pd.Series:
    end = int(signal_i) - int(skip)
    start = end - int(window)
    if start < 0 or end < 0:
        return pd.Series(np.nan, index=close.columns, dtype=float)
    prev = close.iloc[start].replace(0, np.nan)
    now = close.iloc[end]
    return now / prev - 1.0


def us_dynamic_formal_select_targets(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    universe: pd.DataFrame,
    signal_i: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    source_sleeve = str(config.get("source_sleeve", "DYNAMIC_FORMAL99"))
    symbols = close.columns.astype(str).tolist()
    latest_date = close.index[signal_i]
    budget = us_market_budget_for_signal(
        close,
        signal_i,
        normal_gross=1.0,
        weak_gross=1.0,
        hard_gross=0.75,
        risk_mode=str(config.get("risk_mode", "hard_smh_ma200")),
        overlay_symbol="XLK",
        confirm_days=1,
    )
    gross = float(budget["gross_cap"])

    theme = us_scan_theme_series(universe, symbols)
    meta = universe.drop_duplicates("symbol", keep="first").set_index("symbol").reindex(symbols)
    meta["symbol"] = meta.index.astype(str)
    for col in ["name", "trade_group", "parent_group", "scan_rank_60d_dollar_volume", "median_dollar_volume_60d"]:
        if col not in meta.columns:
            meta[col] = ""
    scan_rank = pd.to_numeric(meta["scan_rank_60d_dollar_volume"], errors="coerce")

    dollar = close * volume
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        dollar20 = dollar.iloc[max(0, signal_i - 19) : signal_i + 1].median()
        dollar60 = dollar.iloc[max(0, signal_i - 59) : signal_i + 1].median()
    dollar_accel = dollar20 / dollar60.replace(0, np.nan) - 1.0
    obs = close.iloc[: signal_i + 1].notna().sum()
    mom20 = us_formal_ret(close, signal_i, 20, 0)
    mom63 = us_formal_ret(close, signal_i, int(config.get("lookback", 63)), int(config.get("skip", 5)))
    mom126 = us_formal_ret(close, signal_i, 126, int(config.get("skip", 5)))
    score = (
        0.45 * mom63
        + 0.25 * mom126.fillna(0.0)
        + 0.20 * mom20.fillna(0.0)
        + 0.10 * dollar_accel.clip(lower=-1.0, upper=5.0).fillna(0.0)
    )

    focused = theme.isin(DYNAMIC_FORMAL_THEMES) & ~theme.isin(DYNAMIC_FORMAL_EXCLUDED_THEMES)
    in_top800 = scan_rank.notna() & scan_rank.le(float(config.get("scan_top_n", 800)))
    eligible = (
        in_top800
        & focused
        & obs.ge(int(config.get("min_obs", 252)))
        & dollar20.ge(float(config.get("min_dollar_volume20", 20_000_000)))
        & mom63.ge(float(config.get("min_mom63", 0.0)))
        & score.ge(float(config.get("min_scan_score", 0.30)))
        & score.replace([np.inf, -np.inf], np.nan).notna()
    )

    stock_rows: list[dict[str, Any]] = []
    for sym in symbols:
        stock_rows.append(
            {
                "source_sleeve": source_sleeve,
                "symbol": sym,
                "name": str(meta.at[sym, "name"]) if sym in meta.index else "",
                "theme": str(theme.get(sym, "us_unclassified")),
                "scan_rank_60d_dollar_volume": float(scan_rank.get(sym, np.nan)) if np.isfinite(scan_rank.get(sym, np.nan)) else None,
                "eligible": bool(eligible.get(sym, False)),
                "scan_score": float(score.get(sym, np.nan)) if np.isfinite(score.get(sym, np.nan)) else None,
                "mom20_pct": pct(float(mom20.get(sym, np.nan))),
                "mom63_skip5_pct": pct(float(mom63.get(sym, np.nan))),
                "mom126_skip5_pct": pct(float(mom126.get(sym, np.nan))),
                "dollar_volume20": float(dollar20.get(sym, np.nan)) if np.isfinite(dollar20.get(sym, np.nan)) else None,
                "obs": int(obs.get(sym, 0)),
                "selected_today": False,
            }
        )

    pool = pd.DataFrame(stock_rows)
    pool["score_num"] = pd.to_numeric(pool["scan_score"], errors="coerce")
    pool["mom63_num"] = pd.to_numeric(pool["mom63_skip5_pct"], errors="coerce")
    formal_pool = (
        pool[pool["eligible"].astype(bool)]
        .sort_values(["score_num", "mom63_num", "scan_rank_60d_dollar_volume"], ascending=[False, False, True])
        .head(int(config.get("formal_pool_max_symbols", 99)))
        .copy()
    )

    if gross <= 0.0 or formal_pool.empty:
        reason = str(budget["regime"]) if gross <= 0.0 else "no_formal99_candidates"
        return {
            "targets": [{"symbol": "CASH", "theme": "cash", "role": reason, "target_weight": 1.0, "source_sleeve": source_sleeve}],
            "raw_targets": [],
            "theme_rows": [],
            "stock_rows": stock_rows,
            "regime": reason,
            "gross_cap": 0.0,
            "active_stock_count": 0,
            "formal_pool_count": int(len(formal_pool)),
            "latest_price_date": latest_date.strftime("%Y-%m-%d"),
            "budget": budget,
        }

    theme_rows: list[dict[str, Any]] = []
    for theme_name, grp in formal_pool.groupby("theme", dropna=False):
        if len(grp) < 1:
            continue
        ranked = grp.sort_values("score_num", ascending=False)
        top2 = ranked.head(2)
        theme_score = (
            float(top2["score_num"].mean())
            + 0.20 * float(pd.to_numeric(grp["mom63_skip5_pct"], errors="coerce").median() / 100.0)
            + 0.10 * float((pd.to_numeric(grp["mom63_skip5_pct"], errors="coerce") > 0).mean())
        )
        theme_rows.append(
            {
                "source_sleeve": source_sleeve,
                "theme": str(theme_name),
                "theme_score": float(theme_score),
                "member_count": int(len(grp)),
                "top_symbols_by_momentum": "|".join(ranked["symbol"].astype(str).head(8)),
            }
        )
    theme_rows = sorted(theme_rows, key=lambda row: float(row["theme_score"]), reverse=True)
    selected_theme_count = int(config.get("selected_theme_count", 3))
    selected_themes = theme_rows[:selected_theme_count]
    for rank, row in enumerate(theme_rows, 1):
        row["rank"] = rank
        row["selected"] = rank <= selected_theme_count

    if len(selected_themes) == 1:
        theme_weights = [1.0]
    elif len(selected_themes) == 2:
        theme_weights = [0.85, 0.15]
    else:
        theme_weights = [0.70, 0.20, 0.10]

    raw: dict[str, float] = {}
    raw_rows: list[dict[str, Any]] = []
    stocks_per_theme = int(config.get("stocks_per_theme", 3))
    for theme_rank, (theme_row, theme_weight) in enumerate(zip(selected_themes, theme_weights), 1):
        candidates = formal_pool[formal_pool["theme"].astype(str).eq(str(theme_row["theme"]))].sort_values("score_num", ascending=False)
        keep = candidates.head(stocks_per_theme)
        if keep.empty:
            continue
        per = gross * float(theme_weight) / len(keep)
        for symbol_rank, row in enumerate(keep.to_dict(orient="records"), 1):
            sym = str(row["symbol"])
            raw[sym] = raw.get(sym, 0.0) + per
            raw_rows.append(
                {
                    "symbol": sym,
                    "theme": str(theme_row["theme"]),
                    "role": "dynamic_leader" if symbol_rank == 1 else "dynamic_accelerator",
                    "raw_weight": float(per),
                    "source_sleeve": source_sleeve,
                    "theme_rank": int(theme_rank),
                    "symbol_rank": int(symbol_rank),
                    "scan_score": row.get("scan_score"),
                    "mom63_skip5_pct": row.get("mom63_skip5_pct"),
                }
            )

    if not raw:
        return {
            "targets": [{"symbol": "CASH", "theme": "cash", "role": "no_dynamic_symbols", "target_weight": 1.0, "source_sleeve": source_sleeve}],
            "raw_targets": raw_rows,
            "theme_rows": theme_rows,
            "stock_rows": stock_rows,
            "regime": "no_dynamic_symbols",
            "gross_cap": 0.0,
            "active_stock_count": 0,
            "formal_pool_count": int(len(formal_pool)),
            "latest_price_date": latest_date.strftime("%Y-%m-%d"),
            "budget": budget,
        }

    final_top_n = int(config.get("final_top_n", 8))
    ranked_symbols = sorted(raw, key=lambda sym: raw[sym], reverse=True)[:final_top_n]
    capped = cap_weights_pro_rata(
        {sym: raw[sym] for sym in ranked_symbols},
        target_total=min(gross, 1.0),
        cap=float(config.get("single_symbol_cap", 0.25)),
    )
    targets: list[dict[str, Any]] = []
    for sym in sorted(capped, key=lambda s: capped[s], reverse=True):
        raw_row = next((row for row in raw_rows if row["symbol"] == sym), {})
        targets.append(
            {
                "symbol": sym,
                "theme": str(raw_row.get("theme", "")),
                "role": str(raw_row.get("role", "dynamic_formal")),
                "target_weight": float(capped[sym]),
                "source_sleeve": source_sleeve,
                "scan_score": raw_row.get("scan_score"),
                "mom63_skip5_pct": raw_row.get("mom63_skip5_pct"),
            }
        )
    used = float(sum(capped.values()))
    if used < 1.0 - 1e-12:
        targets.append({"symbol": "CASH", "theme": "cash", "role": "residual_cash", "target_weight": 1.0 - used, "source_sleeve": source_sleeve})

    selected = {row["symbol"] for row in targets if row["symbol"] != "CASH"}
    for row in stock_rows:
        row["selected_today"] = row["symbol"] in selected
    return {
        "targets": targets,
        "raw_targets": raw_rows,
        "theme_rows": theme_rows,
        "stock_rows": stock_rows,
        "regime": str(budget["regime"]),
        "gross_cap": used,
        "active_stock_count": len(selected),
        "formal_pool_count": int(len(formal_pool)),
        "latest_price_date": latest_date.strftime("%Y-%m-%d"),
        "budget": budget,
    }


def simulate_us_dynamic_formal_equity(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    universe: pd.DataFrame,
    config: dict[str, Any],
) -> pd.Series:
    returns = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    rebalance_n = int(config.get("rebalance_n", 5))
    start_i = max(252, int(config.get("lookback", 63)) + int(config.get("skip", 5)) + 1, rebalance_n + 1)
    current_w: dict[str, float] = {}
    equity = 1.0
    vals: list[float] = []
    for i, _dt in enumerate(close.index):
        if i > 0:
            day_ret = 0.0
            for sym, weight in current_w.items():
                if sym in returns.columns:
                    day_ret += float(weight) * float(returns[sym].iloc[i])
            equity *= 1.0 + day_ret
        vals.append(float(equity))
        if i >= start_i and (i - start_i) % rebalance_n == 0:
            signal_i = i - 1
            pack = us_dynamic_formal_select_targets(close, volume, universe, signal_i, config)
            current_w = {
                str(row["symbol"]): float(row["target_weight"])
                for row in pack["targets"]
                if str(row.get("symbol", "")) != "CASH"
            }
    return pd.Series(vals, index=close.index, name=str(config.get("strategy_name", "FORMAL99")), dtype=float)


def us_dynamic_gate_decision(
    base_equity: pd.Series,
    dynamic_equity: pd.Series,
    config: dict[str, Any],
    base_latest_date: pd.Timestamp,
    scan_latest_date: pd.Timestamp,
) -> dict[str, Any]:
    max_stale = int(config.get("max_scan_staleness_days", 3))
    stale_days = int(max(0, (pd.Timestamp(base_latest_date).date() - pd.Timestamp(scan_latest_date).date()).days))
    if stale_days > max_stale:
        return {
            "active": False,
            "dynamic_weight": 0.0,
            "reason": "scan_data_stale",
            "scan_staleness_days": stale_days,
            "max_scan_staleness_days": max_stale,
        }
    common = base_equity.index.intersection(dynamic_equity.index)
    base = base_equity.reindex(common).astype(float).replace([np.inf, -np.inf], np.nan).ffill()
    dyn = dynamic_equity.reindex(common).astype(float).replace([np.inf, -np.inf], np.nan).ffill()
    if len(common) < max(int(config.get("gate_ma_window", 20)), int(config.get("gate_dd_window", 60))) + 2:
        return {"active": False, "dynamic_weight": 0.0, "reason": "not_enough_gate_history", "scan_staleness_days": stale_days}
    rel = dyn / base.replace(0, np.nan)
    ma = rel.rolling(int(config.get("gate_ma_window", 20)), min_periods=int(config.get("gate_ma_window", 20))).mean()
    mom = dyn / dyn.shift(int(config.get("gate_mom_window", 20))) - 1.0
    dd = dyn / dyn.rolling(int(config.get("gate_dd_window", 60)), min_periods=20).max() - 1.0
    last = common[-1]
    rel_val = float(rel.loc[last]) if np.isfinite(rel.loc[last]) else np.nan
    ma_val = float(ma.loc[last]) if np.isfinite(ma.loc[last]) else np.nan
    mom_val = float(mom.loc[last]) if np.isfinite(mom.loc[last]) else np.nan
    dd_val = float(dd.loc[last]) if np.isfinite(dd.loc[last]) else np.nan
    active = bool(
        np.isfinite(rel_val)
        and np.isfinite(ma_val)
        and np.isfinite(mom_val)
        and np.isfinite(dd_val)
        and rel_val > ma_val
        and mom_val >= float(config.get("gate_min_mom", 0.05))
        and dd_val >= -float(config.get("gate_max_dd", 0.20))
    )
    return {
        "active": active,
        "dynamic_weight": float(config.get("dynamic_weight", 0.35)) if active else 0.0,
        "reason": "gate_on" if active else "gate_off",
        "scan_staleness_days": stale_days,
        "relative_value": rel_val,
        "relative_ma": ma_val,
        "dynamic_mom_pct": pct(mom_val),
        "dynamic_dd_pct": pct(dd_val),
    }


def combine_us_dynamic_sleeve_targets(
    base_targets: list[dict[str, Any]],
    dynamic_targets: list[dict[str, Any]],
    dynamic_weight: float,
) -> list[dict[str, Any]]:
    dyn_w = max(0.0, min(1.0, float(dynamic_weight)))
    base_w = 1.0 - dyn_w
    by_symbol: dict[str, dict[str, Any]] = {}
    for sleeve_name, sleeve_weight, rows in [
        ("base_core", base_w, base_targets),
        ("dynamic_formal99", dyn_w, dynamic_targets),
    ]:
        for row in rows:
            sym = str(row.get("symbol", ""))
            if not sym:
                continue
            weight = float(row.get("target_weight", 0.0) or 0.0) * sleeve_weight
            if weight <= 1e-12:
                continue
            if sym not in by_symbol:
                by_symbol[sym] = {
                    "symbol": sym,
                    "theme": str(row.get("theme", "")),
                    "role": str(row.get("role", "")),
                    "target_weight": 0.0,
                    "source_sleeve": [],
                    "execution_group": str(row.get("execution_group", "main")),
                }
            by_symbol[sym]["target_weight"] += weight
            if sleeve_name not in by_symbol[sym]["source_sleeve"]:
                by_symbol[sym]["source_sleeve"].append(sleeve_name)
            if str(by_symbol[sym].get("theme", "")) in {"", "cash"} and str(row.get("theme", "")) not in {"", "cash"}:
                by_symbol[sym]["theme"] = str(row.get("theme", ""))
    out = []
    for row in by_symbol.values():
        rr = dict(row)
        rr["source_sleeve"] = "+".join(rr["source_sleeve"])
        rr["execution_overlay"] = "US_BASE_PLUS_DYNAMIC_FORMAL99"
        out.append(rr)
    stock_total = sum(float(row["target_weight"]) for row in out if str(row["symbol"]) != "CASH")
    cash = next((row for row in out if str(row["symbol"]) == "CASH"), None)
    if cash is None and stock_total < 1.0 - 1e-12:
        out.append(
            {
                "symbol": "CASH",
                "theme": "cash",
                "role": "residual_cash",
                "target_weight": 1.0 - stock_total,
                "source_sleeve": "residual",
                "execution_group": "main",
                "execution_overlay": "US_BASE_PLUS_DYNAMIC_FORMAL99",
            }
        )
    return sorted(out, key=lambda x: (str(x["symbol"]) == "CASH", -float(x["target_weight"])))


US_DYNAMIC_MARKET_THEMES_CORE = {
    "ai_compute",
    "ai_data_center_connectivity",
    "cloud_ai_infra",
    "cloud_infra",
    "compound_semiconductor",
    "data_center_reit",
    "electronics_manufacturing",
    "grid_engineering",
    "industrial_automation",
    "memory_storage",
    "optical_communication",
    "power_generation",
    "power_semiconductor",
    "semi_equipment",
    "semi_foundry_idm",
    "server_network",
    "software_platform",
    "space_ev_defense",
    "thermal_power",
}

US_DYNAMIC_MARKET_THEMES_BROAD = US_DYNAMIC_MARKET_THEMES_CORE | {
    "consumer",
    "energy",
    "finance",
    "healthcare",
    "industrial_transport",
    "materials_commodity",
}

US_DYNAMIC_MARKET_COMPONENT_RULES = [
    {
        "name": "DEFENSE_R300_N3_COMBO",
        "theme_set": "core",
        "radar_n": 300,
        "score": "combo",
        "n": 3,
        "rebalance_n": 10,
        "min_obs": 252,
        "min_ret20": -0.10,
        "min_ret60": 0.05,
        "min_ret120": 0.00,
        "min_high120_dd": -0.60,
        "min_amount_ratio": 0.70,
        "min_dollar60": 20_000_000,
        "max_single_weight": 0.25,
        "risk_mode": "xlk_hard",
    },
    {
        "name": "VALMAX_R500_N3_ACCEL60",
        "theme_set": "core",
        "radar_n": 500,
        "score": "accel60",
        "n": 3,
        "rebalance_n": 10,
        "min_obs": 252,
        "min_ret20": -0.02,
        "min_ret60": 0.05,
        "min_ret120": 0.05,
        "min_high120_dd": -0.60,
        "min_amount_ratio": 0.00,
        "min_dollar60": 20_000_000,
        "max_single_weight": 0.25,
        "risk_mode": "none",
    },
    {
        "name": "TOP5_R800_N5_COMBO",
        "theme_set": "core",
        "radar_n": 800,
        "score": "combo",
        "n": 5,
        "rebalance_n": 10,
        "min_obs": 252,
        "min_ret20": -0.10,
        "min_ret60": 0.00,
        "min_ret120": 0.05,
        "min_high120_dd": -0.60,
        "min_amount_ratio": 0.70,
        "min_dollar60": 20_000_000,
        "max_single_weight": 0.25,
        "risk_mode": "none",
    },
    {
        "name": "ATTACK_R800_N6_ULTRA20",
        "theme_set": "core",
        "radar_n": 800,
        "score": "ultra20",
        "n": 6,
        "rebalance_n": 10,
        "min_obs": 252,
        "min_ret20": -0.02,
        "min_ret60": 0.00,
        "min_ret120": 0.05,
        "min_high120_dd": -0.60,
        "min_amount_ratio": 0.00,
        "min_dollar60": 20_000_000,
        "max_single_weight": 0.25,
        "risk_mode": "none",
    },
    {
        "name": "BROAD_R800_N6_COMBO",
        "theme_set": "broad",
        "radar_n": 800,
        "score": "combo",
        "n": 6,
        "rebalance_n": 10,
        "min_obs": 252,
        "min_ret20": -0.10,
        "min_ret60": 0.00,
        "min_ret120": 0.00,
        "min_high120_dd": -0.60,
        "min_amount_ratio": 0.70,
        "min_dollar60": 20_000_000,
        "max_single_weight": 0.25,
        "risk_mode": "none",
    },
    {
        "name": "BROAD_ATTACK_R800_N8_ULTRA20",
        "theme_set": "broad",
        "radar_n": 800,
        "score": "ultra20",
        "n": 8,
        "rebalance_n": 10,
        "min_obs": 252,
        "min_ret20": -0.02,
        "min_ret60": 0.00,
        "min_ret120": 0.00,
        "min_high120_dd": -0.60,
        "min_amount_ratio": 0.00,
        "min_dollar60": 20_000_000,
        "max_single_weight": 0.25,
        "risk_mode": "none",
    },
    {
        "name": "LIQUID_R200_N3_ACCEL60",
        "theme_set": "broad",
        "radar_n": 200,
        "score": "accel60",
        "n": 3,
        "rebalance_n": 10,
        "min_obs": 252,
        "min_ret20": -0.02,
        "min_ret60": 0.05,
        "min_ret120": 0.05,
        "min_high120_dd": -0.60,
        "min_amount_ratio": 0.00,
        "min_dollar60": 20_000_000,
        "max_single_weight": 0.25,
        "risk_mode": "none",
    },
    {
        "name": "LONG_R800_N5_LONG",
        "theme_set": "broad",
        "radar_n": 800,
        "score": "long",
        "n": 5,
        "rebalance_n": 10,
        "min_obs": 252,
        "min_ret20": -0.05,
        "min_ret60": 0.05,
        "min_ret120": 0.05,
        "min_high120_dd": -0.60,
        "min_amount_ratio": 0.00,
        "min_dollar60": 20_000_000,
        "max_single_weight": 0.25,
        "risk_mode": "smh_ma200",
    },
    {
        "name": "HYPER_R800_N6_ULTRA20",
        "theme_set": "broad",
        "radar_n": 800,
        "score": "ultra20",
        "n": 6,
        "rebalance_n": 10,
        "min_obs": 252,
        "min_ret20": 0.00,
        "min_ret60": 0.05,
        "min_ret120": 0.05,
        "min_high120_dd": -0.60,
        "min_amount_ratio": 0.00,
        "min_dollar60": 20_000_000,
        "max_single_weight": 0.25,
        "risk_mode": "none",
    },
    {
        "name": "REVACCEL_R800_N5_ACCEL60",
        "theme_set": "broad",
        "radar_n": 800,
        "score": "accel60",
        "n": 5,
        "rebalance_n": 10,
        "min_obs": 252,
        "min_ret20": 0.05,
        "min_ret60": -0.05,
        "min_ret120": -0.05,
        "min_high120_dd": -0.60,
        "min_amount_ratio": 0.00,
        "min_dollar60": 20_000_000,
        "max_single_weight": 0.25,
        "risk_mode": "none",
    },
]

US_DYNAMIC_MARKET_SWITCH_RULE = {
    "strategy_name": "US_TOP1_DYNAMIC_MARKET_SW_recent_RB10_MIN10_STAY0",
    "score": "recent",
    "rebalance_n": 10,
    "min_top_score": 0.10,
    "stay_bonus": 0.00,
    "fallback": "LONG_R800_N5_LONG",
}


def us_dynamic_market_component_rules(spec: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    rules = [dict(rule) for rule in US_DYNAMIC_MARKET_COMPONENT_RULES]
    if not spec:
        return rules
    raw_top_n = spec.get("execution_top_n", spec.get("top_n"))
    if raw_top_n in (None, ""):
        return rules
    try:
        top_n = int(raw_top_n)
    except (TypeError, ValueError):
        return rules
    if top_n < 1:
        return rules
    top_n = min(top_n, 20)
    for rule in rules:
        rule["original_n"] = int(rule.get("n", top_n))
        rule["n"] = top_n
        rule["execution_top_n_override"] = top_n
    return rules


def us_dynamic_market_theme_meta(universe: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    meta = universe.drop_duplicates("symbol", keep="first").set_index("symbol").reindex(symbols)
    meta["symbol"] = meta.index.astype(str)
    for col in ["name", "trade_group", "parent_group", "execution_theme", "etf"]:
        if col not in meta.columns:
            meta[col] = ""
        meta[col] = meta[col].fillna("").astype(str)
    meta["theme"] = meta.apply(choose_scan_theme, axis=1)
    name_lower = meta["name"].astype(str).str.lower()
    blocked = (
        name_lower.str.contains(" warrant", regex=False)
        | name_lower.str.contains(" warrants", regex=False)
        | name_lower.str.contains(" unit", regex=False)
        | name_lower.str.contains(" units", regex=False)
        | name_lower.str.contains(" right", regex=False)
        | name_lower.str.contains(" rights", regex=False)
        | name_lower.str.contains(" preferred", regex=False)
    )
    sym = meta["symbol"].astype(str)
    meta["is_common_stock"] = ~meta["etf"].astype(str).str.upper().eq("Y") & ~blocked & ~sym.str.endswith(("W", "U", "R"))
    return meta


def us_dynamic_market_feature_pack(close: pd.DataFrame, volume: pd.DataFrame) -> dict[str, pd.DataFrame]:
    dollar = close * volume
    amount20 = dollar.rolling(20, min_periods=10).mean()
    amount60 = dollar.rolling(60, min_periods=30).mean()
    amount_rank60 = dollar.rolling(60, min_periods=30).median().rank(axis=1, ascending=False, method="first")
    amount_ratio = amount20 / amount60.replace(0, np.nan)
    r20 = close / close.shift(20) - 1.0
    r60 = close / close.shift(60) - 1.0
    r120 = close / close.shift(120) - 1.0
    r252 = close / close.shift(252) - 1.0
    high120_dd = close / close.rolling(120, min_periods=40).max() - 1.0
    amount_bonus = amount_ratio.rank(axis=1, pct=True).fillna(0.0)
    observed = close.notna().cumsum()
    return {
        "amount60": amount60,
        "amount_rank60": amount_rank60,
        "amount_ratio": amount_ratio,
        "r20": r20,
        "r60": r60,
        "r120": r120,
        "r252": r252,
        "high120_dd": high120_dd,
        "obs_count": observed,
        "ultra20": 0.65 * r20 + 0.25 * r60 + 0.10 * r120,
        "accel60": 0.35 * r20 + 0.45 * r60 + 0.20 * r120,
        "combo": 0.20 * r20 + 0.30 * r60 + 0.40 * r120 + 0.10 * amount_bonus,
        "long": 0.15 * r60 + 0.35 * r120 + 0.50 * r252,
    }


def us_dynamic_market_risk_multiplier(rule: dict[str, Any], close: pd.DataFrame, signal_i: int) -> float:
    mode = str(rule.get("risk_mode", "none"))
    if mode == "none" or signal_i < 200 or "SMH" not in close.columns or "XLK" not in close.columns:
        return 1.0
    smh = close["SMH"]
    xlk = close["XLK"]
    smh_ma200 = smh.rolling(200, min_periods=150).mean().iloc[signal_i]
    xlk_ma150 = xlk.rolling(150, min_periods=100).mean().iloc[signal_i]
    xlk_m63 = xlk.iloc[signal_i] / xlk.iloc[signal_i - 63] - 1.0 if signal_i >= 63 and xlk.iloc[signal_i - 63] > 0 else np.nan
    if mode == "smh_ma200" and np.isfinite(smh_ma200) and smh.iloc[signal_i] < smh_ma200:
        return 0.0
    if mode == "xlk_hard" and np.isfinite(xlk_ma150) and np.isfinite(xlk_m63) and xlk.iloc[signal_i] < xlk_ma150 and xlk_m63 < -0.10:
        return 0.50
    return 1.0


def us_dynamic_market_component_targets(
    rule: dict[str, Any],
    close: pd.DataFrame,
    volume: pd.DataFrame,
    meta: pd.DataFrame,
    features: dict[str, pd.DataFrame],
    signal_i: int,
) -> dict[str, Any]:
    symbols = close.columns.astype(str).tolist()
    allowed_themes = US_DYNAMIC_MARKET_THEMES_CORE if str(rule.get("theme_set")) == "core" else US_DYNAMIC_MARKET_THEMES_BROAD
    allowed = meta["is_common_stock"].fillna(False).astype(bool) & meta["theme"].astype(str).isin(allowed_themes)
    pool_symbols = list(allowed[allowed].index.astype(str))
    score = features[str(rule["score"])].iloc[signal_i].reindex(pool_symbols)
    ok = (
        score.replace([np.inf, -np.inf], np.nan).notna()
        & features["amount_rank60"].iloc[signal_i].reindex(pool_symbols).le(float(rule["radar_n"])).fillna(False)
        & features["amount60"].iloc[signal_i].reindex(pool_symbols).ge(float(rule["min_dollar60"])).fillna(False)
        & features["obs_count"].iloc[signal_i].reindex(pool_symbols).ge(float(rule["min_obs"])).fillna(False)
        & features["r20"].iloc[signal_i].reindex(pool_symbols).ge(float(rule["min_ret20"])).fillna(False)
        & features["r60"].iloc[signal_i].reindex(pool_symbols).ge(float(rule["min_ret60"])).fillna(False)
        & features["r120"].iloc[signal_i].reindex(pool_symbols).ge(float(rule["min_ret120"])).fillna(False)
        & features["high120_dd"].iloc[signal_i].reindex(pool_symbols).ge(float(rule["min_high120_dd"])).fillna(False)
        & features["amount_ratio"].iloc[signal_i].reindex(pool_symbols).ge(float(rule["min_amount_ratio"])).fillna(False)
    )
    chosen = score[ok].sort_values(ascending=False).head(int(rule["n"]))
    gross = us_dynamic_market_risk_multiplier(rule, close, signal_i)
    targets: list[dict[str, Any]] = []
    if len(chosen) and gross > 0:
        raw = {str(sym): float(rank) for sym, rank in zip(chosen.index.astype(str), range(len(chosen), 0, -1))}
        capped = cap_weights_pro_rata(raw, target_total=min(gross, float(rule["max_single_weight"]) * len(chosen)), cap=float(rule["max_single_weight"]))
        for sym in sorted(capped, key=lambda s: capped[s], reverse=True):
            targets.append(
                {
                    "symbol": sym,
                    "name": str(meta.at[sym, "name"]) if sym in meta.index else "",
                    "theme": str(meta.at[sym, "theme"]) if sym in meta.index else "",
                    "role": f"top{int(rule['n'])}_{rule['name']}",
                    "target_weight": float(capped[sym]),
                    "source_sleeve": str(rule["name"]),
                    "scan_score": float(score.get(sym, np.nan)) if np.isfinite(score.get(sym, np.nan)) else None,
                }
            )
    used = float(sum(float(row["target_weight"]) for row in targets))
    if used < 1.0 - 1e-12:
        targets.append({"symbol": "CASH", "name": "Cash", "theme": "cash", "role": "residual_cash", "target_weight": 1.0 - used, "source_sleeve": str(rule["name"])})

    candidate_rows: list[dict[str, Any]] = []
    ranked = score.sort_values(ascending=False).head(120)
    for rank, (sym, val) in enumerate(ranked.items(), 1):
        candidate_rows.append(
            {
                "rank": rank,
                "component": str(rule["name"]),
                "symbol": str(sym),
                "name": str(meta.at[sym, "name"]) if sym in meta.index else "",
                "theme": str(meta.at[sym, "theme"]) if sym in meta.index else "",
                "eligible": bool(ok.get(sym, False)),
                "selected_today": str(sym) in {row["symbol"] for row in targets if row["symbol"] != "CASH"},
                "component_score": float(val) if np.isfinite(val) else None,
                "final_target_weight": next((float(row["target_weight"]) for row in targets if row["symbol"] == sym), 0.0),
                "scan_rank_60d_dollar_volume": float(features["amount_rank60"].iloc[signal_i].get(sym, np.nan))
                if np.isfinite(features["amount_rank60"].iloc[signal_i].get(sym, np.nan))
                else None,
            }
        )
    return {"targets": targets, "candidate_rows": candidate_rows, "gross": used}


def simulate_us_dynamic_market_components(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    meta: pd.DataFrame,
    features: dict[str, pd.DataFrame],
    component_rules: list[dict[str, Any]] | None = None,
) -> tuple[pd.DataFrame, dict[str, list[dict[str, Any]]]]:
    returns = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    equities: dict[str, pd.Series] = {}
    latest_targets: dict[str, list[dict[str, Any]]] = {}
    for rule in (component_rules or US_DYNAMIC_MARKET_COMPONENT_RULES):
        rebalance_n = int(rule.get("rebalance_n", 10))
        start_i = max(252, rebalance_n + 1)
        weights: dict[str, float] = {}
        vals: list[float] = []
        equity = 1.0
        last_targets: list[dict[str, Any]] = [{"symbol": "CASH", "theme": "cash", "role": "not_started", "target_weight": 1.0}]
        for i, _dt in enumerate(close.index):
            if i > 0:
                day_ret = 0.0
                for sym, weight in weights.items():
                    if sym in returns.columns:
                        day_ret += float(weight) * float(returns[sym].iloc[i])
                equity *= 1.0 + day_ret
            vals.append(float(equity))
            if i >= start_i and (i - start_i) % rebalance_n == 0:
                pack = us_dynamic_market_component_targets(rule, close, volume, meta, features, i - 1)
                last_targets = pack["targets"]
                weights = {
                    str(row["symbol"]): float(row["target_weight"])
                    for row in last_targets
                    if str(row.get("symbol", "")) != "CASH"
                }
        equities[str(rule["name"])] = pd.Series(vals, index=close.index, dtype=float)
        latest_targets[str(rule["name"])] = last_targets
    return pd.DataFrame(equities), latest_targets


def us_dynamic_market_switch_score(component_eq: pd.DataFrame, mode: str) -> pd.DataFrame:
    r20 = component_eq / component_eq.shift(20) - 1.0
    r60 = component_eq / component_eq.shift(60) - 1.0
    r120 = component_eq / component_eq.shift(120) - 1.0
    dd60 = component_eq / component_eq.rolling(60, min_periods=20).max() - 1.0
    if mode == "recent":
        return 0.50 * r20 + 0.35 * r60 + 0.15 * r120
    if mode == "long":
        return 0.20 * r60 + 0.45 * r120 + 0.35 * (component_eq / component_eq.shift(252) - 1.0)
    return 0.25 * r20 + 0.35 * r60 + 0.30 * r120 + 0.10 * dd60


def select_us_dynamic_market_component(component_eq: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    rule = US_DYNAMIC_MARKET_SWITCH_RULE
    scores = us_dynamic_market_switch_score(component_eq, str(rule["score"]))
    fallback = str(rule["fallback"])
    current = fallback if fallback in component_eq.columns else str(component_eq.columns[0])
    rows: list[dict[str, Any]] = []
    start_i = max(252, int(rule["rebalance_n"]) + 1)
    for i, dt in enumerate(component_eq.index):
        if i >= start_i and (i - start_i) % int(rule["rebalance_n"]) == 0:
            signal_i = i - 1
            sig = scores.iloc[signal_i].replace([np.inf, -np.inf], np.nan).dropna()
            if current in sig.index:
                sig.at[current] = float(sig.at[current]) + float(rule["stay_bonus"])
            selected = fallback
            top_score = np.nan
            if len(sig):
                ranked = sig.sort_values(ascending=False)
                top_score = float(ranked.iloc[0])
                selected = str(ranked.index[0]) if top_score >= float(rule["min_top_score"]) else fallback
            current = selected
            rows.append(
                {
                    "date": pd.Timestamp(dt).strftime("%Y-%m-%d"),
                    "signal_date": component_eq.index[signal_i].strftime("%Y-%m-%d"),
                    "selected_component": current,
                    "top_score": top_score,
                }
            )
    return current, pd.DataFrame(rows)


def run_us_dynamic_market_daily(state: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    US_OUT.mkdir(parents=True, exist_ok=True)
    prices, universe = load_us_scan_data()
    symbols = sorted(prices["symbol"].dropna().astype(str).unique())
    close, _open, volume = px_frame(prices, symbols)
    if len(close.index) < 252:
        raise RuntimeError("US dynamic market live data needs at least 252 trading rows")
    latest_date = close.index[-1]
    meta = us_dynamic_market_theme_meta(universe, symbols)
    features = us_dynamic_market_feature_pack(close, volume)
    component_rules = us_dynamic_market_component_rules(spec)
    execution_top_n = int(component_rules[0].get("n", 0)) if component_rules else 0
    execution_cap = float(component_rules[0].get("max_single_weight", 0.25)) if component_rules else 0.25
    execution_overlay = f"US_DYNAMIC_MARKET_TOP{execution_top_n}_CAP{int(execution_cap * 100)}"
    component_eq, latest_targets_by_component = simulate_us_dynamic_market_components(close, volume, meta, features, component_rules)
    selected_component, switch_modes = select_us_dynamic_market_component(component_eq)
    targets = latest_targets_by_component.get(selected_component, [{"symbol": "CASH", "theme": "cash", "role": "missing_component", "target_weight": 1.0}])
    signal_i = len(close.index) - 1
    selected_rule = next((r for r in component_rules if str(r["name"]) == selected_component), component_rules[0])
    latest_pack = us_dynamic_market_component_targets(selected_rule, close, volume, meta, features, signal_i)
    latest_scores = us_dynamic_market_switch_score(component_eq, str(US_DYNAMIC_MARKET_SWITCH_RULE["score"])).iloc[-1].dropna().sort_values(ascending=False)
    theme_rows = []
    for rank, (name, score) in enumerate(latest_scores.items(), 1):
        theme_rows.append({"rank": rank, "theme": str(name), "theme_score": float(score), "selected": str(name) == selected_component})

    us_state = state.setdefault("US", {})
    latest_step = update_step_counter(us_state, latest_date, close.index, fallback_step=len(close.index) - 1)
    rebalance_fields = rebalance_counter_fields(latest_step, len(close.index) - 1, close.index, int(US_DYNAMIC_MARKET_SWITCH_RULE["rebalance_n"]))
    target_gross = float(sum(float(row.get("target_weight", 0.0)) for row in targets if str(row.get("symbol", "")) != "CASH"))
    regime = "risk_on" if target_gross > 0 else "risk_off"
    us_state.update(
        {
            "latest_price_date": latest_date.strftime("%Y-%m-%d"),
        "latest_backtest_step": latest_step,
        "regime": regime,
        "target_position_if_rebalanced": targets,
        "dynamic_market_selected_component": selected_component,
        "dynamic_market_switch_rule": US_DYNAMIC_MARKET_SWITCH_RULE,
        "dynamic_market_execution_top_n": execution_top_n,
    }
    )
    market_state = {
        "market": "US",
        "latest_price_date": latest_date,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "frozen_strategy": spec.get("production_name", "US_TOP1_DYNAMIC_MARKET_LIKE_TW"),
        "strategy_mode": "top1_dynamic_market_like_tw",
        "execution_overlay": execution_overlay,
        "execution_top_n": execution_top_n,
        "regime": regime,
        "selected_component": selected_component,
        "component_signal_date": switch_modes.tail(1).iloc[0]["signal_date"] if not switch_modes.empty else "",
        "component_rebalance_date": switch_modes.tail(1).iloc[0]["date"] if not switch_modes.empty else "",
        "component_top_score": switch_modes.tail(1).iloc[0]["top_score"] if not switch_modes.empty else np.nan,
        "gross_cap": target_gross,
        "scan_universe_symbols": int(len(symbols)),
        "dynamic_market_pool": "rolling Top800 by 60D median dollar volume",
        "rebalance_step_trading_days": int(US_DYNAMIC_MARKET_SWITCH_RULE["rebalance_n"]),
        **rebalance_fields,
        "uses_2025_2026_for_tuning": False,
        "notes": f"US production target uses the clean dynamic-market switch with Top{execution_top_n} execution and {int(execution_cap * 100)}% single-stock cap. Rules are frozen; no cloud retraining.",
    }
    market_state = apply_live_cycle_baseline(market_state, "US", close.index)
    pd.DataFrame([market_state]).to_csv(US_OUT / "latest_market_state.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(theme_rows).to_csv(US_OUT / "latest_theme_rank.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(latest_pack["candidate_rows"]).to_csv(US_OUT / "latest_candidate_pool.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(targets).to_csv(US_OUT / "latest_target_position.csv", index=False, encoding="utf-8-sig")
    switch_modes.to_csv(US_OUT / "latest_top1_dynamic_market_switch_modes.csv", index=False, encoding="utf-8-sig")
    action_report = {
        "market_state": market_state,
        "target_position_if_rebalanced": targets,
        "component_scores": theme_rows,
        "component_candidates": latest_pack["candidate_rows"][:80],
        "component_equity_tail": component_eq.tail(5).reset_index().to_dict(orient="records"),
    }
    write_json(US_OUT / "latest_action_report.json", action_report)
    return action_report


def run_us_combo_daily(state: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    US_OUT.mkdir(parents=True, exist_ok=True)
    all_symbols = us_rule_symbols(spec)
    prices = load_us_prices(all_symbols)
    close, _open, volume = px_frame(prices, all_symbols)
    dates = close.index
    if len(dates) < 252:
        raise RuntimeError("US live data needs at least 252 trading rows for combo component scoring")
    latest_date = dates[-1]
    last = len(dates) - 1

    component_eq = simulate_us_component_equities(spec, close, volume)
    combo_pack = us_combo_pack_for_signal(spec, close, volume, component_eq, last)
    current_packs = combo_pack["current_packs"]
    stable_component_weights = combo_pack["stable_component_weights"]
    old_component_weights = combo_pack["old_component_weights"]
    component_weights = combo_pack["component_weights"]
    raw_targets = combo_pack["raw_targets"]
    base_targets = combo_pack["targets"]
    targets = [dict(row) for row in base_targets]
    dynamic_config = us_dynamic_formal_config(spec)
    dynamic_pack: dict[str, Any] = {
        "targets": [],
        "raw_targets": [],
        "theme_rows": [],
        "stock_rows": [],
        "regime": "disabled",
        "gross_cap": 0.0,
        "active_stock_count": 0,
        "formal_pool_count": 0,
        "latest_price_date": "",
    }
    dynamic_gate: dict[str, Any] = {"active": False, "dynamic_weight": 0.0, "reason": "disabled"}
    base_equity_tail: pd.Series | None = None
    dynamic_equity_tail: pd.Series | None = None
    if bool(dynamic_config.get("enabled", True)):
        try:
            scan_prices, scan_universe = load_us_scan_data()
            scan_symbols = sorted(scan_prices["symbol"].dropna().astype(str).unique())
            scan_close, _scan_open, scan_volume = px_frame(scan_prices, scan_symbols)
            if len(scan_close) >= 252:
                scan_last = len(scan_close.index) - 1
                dynamic_pack = us_dynamic_formal_select_targets(scan_close, scan_volume, scan_universe, scan_last, dynamic_config)
                base_equity_tail = simulate_us_combo_base_equity(spec, close, volume, component_eq)
                dynamic_equity_tail = simulate_us_dynamic_formal_equity(scan_close, scan_volume, scan_universe, dynamic_config)
                dynamic_gate = us_dynamic_gate_decision(
                    base_equity_tail,
                    dynamic_equity_tail,
                    dynamic_config,
                    latest_date,
                    scan_close.index[-1],
                )
                if bool(dynamic_gate.get("active")) and dynamic_pack.get("targets"):
                    targets = combine_us_dynamic_sleeve_targets(
                        base_targets,
                        dynamic_pack["targets"],
                        float(dynamic_gate.get("dynamic_weight", 0.0)),
                    )
            else:
                dynamic_gate = {"active": False, "dynamic_weight": 0.0, "reason": "scan_history_too_short"}
        except Exception as exc:
            dynamic_gate = {
                "active": False,
                "dynamic_weight": 0.0,
                "reason": "dynamic_formal_error",
                "error": str(exc),
            }

    theme_rows: list[dict[str, Any]] = []
    stock_rows: list[dict[str, Any]] = []
    for component, pack in current_packs.items():
        for row in pack["theme_rows"]:
            rr = dict(row)
            rr["component_weight"] = component_weights.get(component, 0.0)
            theme_rows.append(rr)
        for row in pack["stock_rows"]:
            rr = dict(row)
            rr["component_weight"] = component_weights.get(component, 0.0)
            rr["selected_today"] = str(rr.get("symbol")) in {t["symbol"] for t in targets if t["symbol"] != "CASH"}
            stock_rows.append(rr)
    dynamic_live_weight = float(dynamic_gate.get("dynamic_weight", 0.0) or 0.0)
    for row in dynamic_pack.get("theme_rows", []):
        rr = dict(row)
        rr["component_weight"] = dynamic_live_weight
        rr["dynamic_gate_active"] = bool(dynamic_gate.get("active", False))
        theme_rows.append(rr)
    for row in dynamic_pack.get("stock_rows", []):
        rr = dict(row)
        rr["component_weight"] = dynamic_live_weight
        rr["dynamic_gate_active"] = bool(dynamic_gate.get("active", False))
        rr["selected_today"] = str(rr.get("symbol")) in {t["symbol"] for t in targets if t["symbol"] != "CASH"}
        stock_rows.append(rr)
    theme_rows = sorted(theme_rows, key=lambda x: -999.0 if not np.isfinite(x.get("theme_score", np.nan)) else -float(x["theme_score"]))
    for rank, row in enumerate(theme_rows, 1):
        row["rank"] = rank
    stock_rank = pd.DataFrame(stock_rows)
    if not stock_rank.empty:
        stock_rank["mom63_num"] = pd.to_numeric(stock_rank["mom63_skip5_pct"], errors="coerce")
        stock_rank = stock_rank.sort_values(
            ["selected_today", "component_weight", "eligible", "mom63_num"],
            ascending=[False, False, False, False],
        )

    us_state = state.setdefault("US", {})
    latest_step = update_step_counter(us_state, latest_date, dates, fallback_step=len(dates) - 1)
    min_rebalance = min(us_component_rebalance_n(spec, c) for c in US_COMPONENTS)
    rebalance_fields = rebalance_counter_fields(latest_step, last, dates, min_rebalance)
    regimes = [str(pack["regime"]) for pack in current_packs.values()]
    target_gross = float(sum(float(row["target_weight"]) for row in targets if row["symbol"] != "CASH"))
    regime = "risk_off" if target_gross <= 0 else "risk_on" if all(r == "risk_on" for r in regimes) else "mixed"
    smh_mom63 = ret_at(close["SMH"], last, 63, 0)
    xlk_mom21 = ret_at(close["XLK"], last, 21, 0)
    xlk_mom63 = ret_at(close["XLK"], last, 63, 0)
    us_state.update(
        {
            "latest_price_date": latest_date.strftime("%Y-%m-%d"),
            "latest_backtest_step": latest_step,
            "regime": regime,
            "target_position_if_rebalanced": targets,
            "raw_combo_target_position_if_rebalanced": raw_targets,
            "execution_overlay": "US_BASE_PLUS_DYNAMIC_FORMAL99" if dynamic_live_weight > 0 else "US_TOP6_CAP25_FROM_COMBO",
            "dynamic_formal_gate": dynamic_gate,
        }
    )

    component_weight_text = "|".join(f"{k}:{v:.4f}" for k, v in sorted(component_weights.items()))
    raw_target_gross = float(sum(float(row["target_weight"]) for row in raw_targets if row["symbol"] != "CASH"))
    market_state = {
        "market": "US",
        "latest_price_date": latest_date,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "frozen_strategy": spec["production_name"],
        "execution_overlay": "US_BASE_PLUS_DYNAMIC_FORMAL99" if dynamic_live_weight > 0 else "US_TOP6_CAP25_FROM_COMBO",
        "regime": regime,
        "component_regimes": "|".join(f"{k}:{v['regime']}" for k, v in current_packs.items()),
        "stable_component_weights": "|".join(f"{k}:{v:.4f}" for k, v in sorted(stable_component_weights.items())),
        "old_top1_component_weights": "|".join(f"{k}:{v:.4f}" for k, v in sorted(old_component_weights.items())),
        "combo_component_weights": component_weight_text,
        "dynamic_formal_strategy": str(dynamic_config.get("strategy_name", "")),
        "dynamic_formal_enabled": bool(dynamic_config.get("enabled", True)),
        "dynamic_formal_gate_active": bool(dynamic_gate.get("active", False)),
        "dynamic_formal_gate_reason": str(dynamic_gate.get("reason", "")),
        "dynamic_formal_weight": dynamic_live_weight,
        "dynamic_formal_latest_price_date": dynamic_pack.get("latest_price_date", ""),
        "dynamic_formal_scan_staleness_days": dynamic_gate.get("scan_staleness_days"),
        "dynamic_formal_pool_count": dynamic_pack.get("formal_pool_count", 0),
        "dynamic_formal_active_stock_count": dynamic_pack.get("active_stock_count", 0),
        "dynamic_formal_relative_value": dynamic_gate.get("relative_value"),
        "dynamic_formal_relative_ma": dynamic_gate.get("relative_ma"),
        "dynamic_formal_mom20_pct": dynamic_gate.get("dynamic_mom_pct"),
        "dynamic_formal_dd60_pct": dynamic_gate.get("dynamic_dd_pct"),
        "base_risk_on": regime != "risk_off",
        "weak_confirmed_3d": any(str(v["regime"]) == "weak" for v in current_packs.values()),
        "hard_confirmed_3d": any(str(v["regime"]) == "hard_weak" for v in current_packs.values()),
        "gross_cap": target_gross,
        "raw_combo_gross": raw_target_gross,
        "raw_combo_target_count": len(raw_targets),
        "execution_top_n": 6 if dynamic_live_weight <= 0 else "",
        "execution_single_symbol_cap": 0.25,
        "smh_mom63_pct": pct(smh_mom63),
        "xlk_mom21_pct": pct(xlk_mom21),
        "xlk_mom63_pct": pct(xlk_mom63),
        "accelerator_enabled_if_rebalanced": any(bool(v["accelerator_enabled"]) for v in current_packs.values()),
        "latest_backtest_step": latest_step,
        "rebalance_step_trading_days": min_rebalance,
        **rebalance_fields,
        "uses_2025_2026_for_tuning": False,
        "notes": (
            "Cloud package uses the frozen clean US core plus a dynamic Formal99 sleeve. "
            "Formal99 is rebuilt from the rolling Top800 scan pool and can enter official targets only when the clean gate is active. "
            "It does not retrain in the cloud package."
        ),
    }
    market_state = apply_live_cycle_baseline(market_state, "US", dates)

    pd.DataFrame([market_state]).to_csv(US_OUT / "latest_market_state.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(theme_rows).to_csv(US_OUT / "latest_theme_rank.csv", index=False, encoding="utf-8-sig")
    stock_rank.drop(columns=["mom63_num"], errors="ignore").to_csv(US_OUT / "latest_candidate_pool.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(targets).to_csv(US_OUT / "latest_target_position.csv", index=False, encoding="utf-8-sig")
    action_report = {
        "market_state": market_state,
        "target_position_if_rebalanced": targets,
        "raw_combo_target_position_if_rebalanced": raw_targets,
        "top_themes": theme_rows[:12],
        "top_candidates": stock_rank.drop(columns=["mom63_num"], errors="ignore").head(60).to_dict(orient="records"),
        "component_targets": {k: v["targets"] for k, v in current_packs.items()},
        "dynamic_formal": {
            "config": dynamic_config,
            "gate": dynamic_gate,
            "target_position_if_active": dynamic_pack.get("targets", []),
            "raw_targets": dynamic_pack.get("raw_targets", []),
            "top_themes": dynamic_pack.get("theme_rows", [])[:12],
            "top_candidates": pd.DataFrame(dynamic_pack.get("stock_rows", []))
            .sort_values(["selected_today", "scan_score"], ascending=[False, False], na_position="last")
            .head(60)
            .to_dict(orient="records")
            if dynamic_pack.get("stock_rows")
            else [],
        },
        "component_equity_tail": component_eq.tail(5).reset_index().to_dict(orient="records"),
        "base_equity_tail": base_equity_tail.tail(5).reset_index().to_dict(orient="records") if base_equity_tail is not None else [],
        "dynamic_formal_equity_tail": dynamic_equity_tail.tail(5).reset_index().to_dict(orient="records") if dynamic_equity_tail is not None else [],
    }
    write_json(US_OUT / "latest_action_report.json", action_report)
    return action_report


def run_us_daily(state: dict[str, Any]) -> dict[str, Any]:
    spec = json.loads((RULE_DIR / "us_rule.json").read_text(encoding="utf-8"))
    if spec.get("strategy_type") == "us_top1_dynamic_market_like_tw":
        return run_us_dynamic_market_daily(state, spec)
    if spec.get("strategy_type") == "us_combo_top1_top3":
        return run_us_combo_daily(state, spec)
    return run_us_v30_daily(state)


def load_us_scan_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    price_path = DATA_DIR / "us_scan_prices_tail.csv"
    universe_path = DATA_DIR / "us_scan_universe.csv"
    if not price_path.exists() or not universe_path.exists():
        raise RuntimeError("US scan files are missing; rebuild data_live with BUILD_LIVE_DATA.cmd")
    prices = pd.read_csv(price_path, dtype={"symbol": str}, parse_dates=["date"], low_memory=False)
    universe = pd.read_csv(universe_path, dtype={"symbol": str}, low_memory=False)
    for col in ["symbol", "name", "trade_group", "parent_group", "execution_theme", "in_execution_pool"]:
        if col not in universe.columns:
            universe[col] = ""
        universe[col] = universe[col].fillna("").astype(str)
    universe["in_execution_pool_bool"] = universe["in_execution_pool"].str.lower().isin(["true", "1", "yes"])
    return prices, universe.drop_duplicates("symbol", keep="first")


def frame_return(close: pd.DataFrame, end_offset: int, window: int) -> pd.Series:
    end = len(close) - 1 - end_offset
    start = end - window
    if start < 0 or end < 0:
        return pd.Series(np.nan, index=close.columns)
    return close.iloc[end] / close.iloc[start] - 1.0


def choose_scan_theme(row: pd.Series) -> str:
    symbol = str(row.get("symbol", "") or "")
    if symbol in US_SCAN_THEME_OVERRIDES:
        return US_SCAN_THEME_OVERRIDES[symbol]
    execution_theme = str(row.get("execution_theme", "") or "")
    if execution_theme:
        return execution_theme
    trade_group = str(row.get("trade_group", "") or "")
    parent_group = str(row.get("parent_group", "") or "")
    if trade_group and trade_group != "us_unclassified_stock":
        return trade_group
    if parent_group and parent_group != "us_unclassified":
        return parent_group
    return "us_unclassified"


def run_us_scan_daily(us_action: dict[str, Any]) -> dict[str, Any]:
    US_SCAN_OUT.mkdir(parents=True, exist_ok=True)
    prices, universe = load_us_scan_data()
    symbols = sorted(prices["symbol"].dropna().astype(str).unique())
    close, _open, volume = px_frame(prices, symbols)
    dates = close.index
    if len(dates) < 132:
        raise RuntimeError("US scan data needs at least 132 trading rows")
    latest_date = dates[-1]

    mom20 = frame_return(close, 0, 20)
    mom63_skip5 = frame_return(close, 5, 63)
    mom126_skip5 = frame_return(close, 5, 126)
    dollar = close * volume
    dollar_volume20 = dollar.tail(20).median()
    dollar_volume60 = dollar.tail(60).median()
    dollar_accel = dollar_volume20 / dollar_volume60.replace(0, np.nan) - 1.0
    obs = close.notna().sum()

    meta = universe.set_index("symbol").reindex(symbols)
    meta["symbol"] = meta.index.astype(str)
    meta["scan_theme"] = meta.apply(choose_scan_theme, axis=1)
    in_execution = meta["in_execution_pool_bool"].fillna(False).astype(bool)
    scan_rank_liquidity = pd.to_numeric(meta.get("scan_rank_60d_dollar_volume", pd.Series(index=meta.index)), errors="coerce")
    median_dollar_volume_60d = pd.to_numeric(meta.get("median_dollar_volume_60d", pd.Series(index=meta.index)), errors="coerce")
    score = (
        0.45 * mom63_skip5
        + 0.25 * mom126_skip5.fillna(0.0)
        + 0.20 * mom20.fillna(0.0)
        + 0.10 * dollar_accel.clip(lower=-1.0, upper=5.0).fillna(0.0)
    )
    eligible = obs.ge(126) & mom63_skip5.replace([np.inf, -np.inf], np.nan).notna() & dollar_volume20.gt(0)

    rows: list[dict[str, Any]] = []
    for sym in symbols:
        rows.append(
            {
                "symbol": sym,
                "name": str(meta.at[sym, "name"]) if sym in meta.index else "",
                "scan_theme": str(meta.at[sym, "scan_theme"]) if sym in meta.index else "us_unclassified",
                "trade_group": str(meta.at[sym, "trade_group"]) if sym in meta.index else "",
                "parent_group": str(meta.at[sym, "parent_group"]) if sym in meta.index else "",
                "in_execution_pool": bool(in_execution.get(sym, False)),
                "execution_theme": str(meta.at[sym, "execution_theme"]) if sym in meta.index else "",
                "scan_rank_60d_dollar_volume": float(scan_rank_liquidity.get(sym, np.nan))
                if np.isfinite(scan_rank_liquidity.get(sym, np.nan))
                else None,
                "median_dollar_volume_60d": float(median_dollar_volume_60d.get(sym, np.nan))
                if np.isfinite(median_dollar_volume_60d.get(sym, np.nan))
                else None,
                "eligible": bool(eligible.get(sym, False)),
                "scan_score": float(score.get(sym, np.nan)),
                "mom20_pct": pct(float(mom20.get(sym, np.nan))),
                "mom63_skip5_pct": pct(float(mom63_skip5.get(sym, np.nan))),
                "mom126_skip5_pct": pct(float(mom126_skip5.get(sym, np.nan))),
                "dollar_volume20": float(dollar_volume20.get(sym, np.nan)),
                "dollar_accel20_60_pct": pct(float(dollar_accel.get(sym, np.nan))),
                "obs": int(obs.get(sym, 0)),
            }
        )
    stock_rank = pd.DataFrame(rows)
    stock_rank["score_num"] = pd.to_numeric(stock_rank["scan_score"], errors="coerce")
    stock_rank = stock_rank.sort_values(["eligible", "score_num", "dollar_volume20"], ascending=[False, False, False])
    stock_rank.insert(0, "rank", range(1, len(stock_rank) + 1))

    theme_rows: list[dict[str, Any]] = []
    eligible_rank = stock_rank[stock_rank["eligible"].astype(bool)].copy()
    for theme_name, grp in eligible_rank.groupby("scan_theme", dropna=False):
        if len(grp) < 3:
            continue
        top = grp.sort_values("score_num", ascending=False).head(8)
        top5 = top.head(5)
        top_symbols = "|".join(top["symbol"].head(8).astype(str))
        val = (
            float(top5["score_num"].mean())
            + 0.20 * float(pd.to_numeric(grp["mom63_skip5_pct"], errors="coerce").median() / 100.0)
            + 0.10 * float((pd.to_numeric(grp["mom63_skip5_pct"], errors="coerce") > 0).mean())
        )
        theme_rows.append(
            {
                "scan_theme": str(theme_name),
                "theme_score": val,
                "member_count": int(len(grp)),
                "execution_pool_member_count": int(grp["in_execution_pool"].astype(bool).sum()),
                "top_symbols": top_symbols,
                "top_outside_execution_symbols": "|".join(top[~top["in_execution_pool"].astype(bool)]["symbol"].head(8).astype(str)),
            }
        )
    theme_rank = pd.DataFrame(theme_rows)
    if not theme_rank.empty:
        theme_rank = theme_rank.sort_values("theme_score", ascending=False)
        theme_rank.insert(0, "rank", range(1, len(theme_rank) + 1))

    official_themes = {
        str(row.get("theme", ""))
        for row in us_action.get("top_themes", [])
        if bool(row.get("selected", False))
    }
    target_symbols = {
        str(row.get("symbol", ""))
        for row in us_action.get("target_position_if_rebalanced", [])
        if str(row.get("symbol", "")) != "CASH"
    }
    top20 = stock_rank.head(20)
    outside_top20 = top20[~top20["in_execution_pool"].astype(bool)]
    execution_scores = stock_rank[stock_rank["in_execution_pool"].astype(bool)]["score_num"].dropna()
    outside_scores = stock_rank[~stock_rank["in_execution_pool"].astype(bool)]["score_num"].dropna()
    official_scores = stock_rank[stock_rank["symbol"].isin(target_symbols)]["score_num"].dropna()
    outside_theme_rank = theme_rank[
        ~theme_rank["scan_theme"].astype(str).isin(official_themes)
    ].head(5) if not theme_rank.empty else pd.DataFrame()
    us_strategy_mode = str(us_action.get("market_state", {}).get("strategy_mode", ""))
    feeds_dynamic_market = us_strategy_mode == "top1_dynamic_market_like_tw"
    scan_note = (
        "US_SCAN monitors the rolling Top800 market pool. The official US Top1 dynamic-market strategy can select from this live pool when frozen rules pass."
        if feeds_dynamic_market
        else "US_SCAN monitors the rolling Top800 market pool. Scan names are still not direct buys unless the official US strategy selects them."
    )

    alerts = {
        "latest_price_date": latest_date,
        "scan_universe_symbols": int(len(symbols)),
        "official_selected_themes": sorted(official_themes),
        "official_target_symbols": sorted(target_symbols),
        "outside_execution_count_in_top20": int(len(outside_top20)),
        "top_outside_execution_symbols": outside_top20[["rank", "symbol", "scan_theme", "scan_score", "mom63_skip5_pct"]].head(10).to_dict(orient="records"),
        "top_outside_scan_themes": outside_theme_rank.to_dict(orient="records") if not outside_theme_rank.empty else [],
        "best_execution_pool_score": float(execution_scores.max()) if len(execution_scores) else None,
        "best_official_target_score": float(official_scores.max()) if len(official_scores) else None,
        "best_outside_execution_score": float(outside_scores.max()) if len(outside_scores) else None,
        "watch_only": False,
        "feeds_dynamic_market_strategy": feeds_dynamic_market,
        "feeds_dynamic_formal_strategy": not feeds_dynamic_market,
        "notes": scan_note,
    }

    market_state = {
        "market": "US_SCAN",
        "latest_price_date": latest_date,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "scan_universe_symbols": int(len(symbols)),
        "top_stock": str(stock_rank.iloc[0]["symbol"]) if not stock_rank.empty else "",
        "top_theme": str(theme_rank.iloc[0]["scan_theme"]) if not theme_rank.empty else "",
        "outside_execution_count_in_top20": alerts["outside_execution_count_in_top20"],
        "watch_only": False,
        "feeds_dynamic_market_strategy": feeds_dynamic_market,
        "feeds_dynamic_formal_strategy": not feeds_dynamic_market,
    }

    pd.DataFrame([market_state]).to_csv(US_SCAN_OUT / "latest_market_state.csv", index=False, encoding="utf-8-sig")
    stock_rank.drop(columns=["score_num"], errors="ignore").to_csv(US_SCAN_OUT / "latest_scan_stock_rank.csv", index=False, encoding="utf-8-sig")
    theme_rank.to_csv(US_SCAN_OUT / "latest_scan_theme_rank.csv", index=False, encoding="utf-8-sig")
    write_json(US_SCAN_OUT / "latest_scan_alerts.json", alerts)
    action_report = {
        "market_state": market_state,
        "top_scan_stocks": stock_rank.drop(columns=["score_num"], errors="ignore").head(50).to_dict(orient="records"),
        "top_scan_themes": theme_rank.head(20).to_dict(orient="records") if not theme_rank.empty else [],
        "alerts": alerts,
    }
    write_json(US_SCAN_OUT / "latest_action_report.json", action_report)
    return action_report


def load_tw_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prices = pd.read_csv(DATA_DIR / "tw_strategy_prices_tail.csv", dtype={"symbol": str}, parse_dates=["date"], low_memory=False)
    index_prices = pd.read_csv(DATA_DIR / "tw_index_prices_tail.csv", dtype={"symbol": str}, parse_dates=["date"], low_memory=False)
    symbol_map = pd.read_csv(DATA_DIR / "tw_strategy_group_map.csv", dtype={"symbol": str}, low_memory=False)
    return prices, index_prices, symbol_map


def is_common_stock(symbol: str) -> bool:
    code = str(symbol).split(".")[0]
    return len(code) == 4 and code.isdigit() and int(code) >= 1101


def shifted_ret(close: pd.DataFrame, window: int) -> pd.DataFrame:
    return (close / close.shift(window) - 1.0).shift(1)


def shifted_above_ma(close: pd.DataFrame, window: int) -> pd.DataFrame:
    ma = close.rolling(window, min_periods=max(20, window // 2)).mean()
    return (close > ma).shift(1).fillna(False)


def tw_stock_score(r20: pd.DataFrame, r60: pd.DataFrame, r120: pd.DataFrame, score_type: str) -> pd.DataFrame:
    if score_type == "blend_long":
        return 0.25 * r20.fillna(-9.0) + 0.45 * r60.fillna(-9.0) + 0.30 * r120.fillna(-9.0)
    if score_type == "ret60":
        return r60.fillna(-9.0)
    if score_type == "ret20":
        return r20.fillna(-9.0)
    raise ValueError(f"unsupported TW stock_score={score_type}")


def tw_market_state_series(
    close: pd.DataFrame,
    amount_rank60: pd.DataFrame,
    r20: pd.DataFrame,
    r60: pd.DataFrame,
    above_ma60: pd.DataFrame,
    index_prices: pd.DataFrame,
    candidate: dict[str, Any],
) -> pd.Series:
    dates = close.index
    core_symbols = [s for s in ["0050.TW", "2330.TW"] if s in close.columns]
    core = close.reindex(columns=core_symbols)
    twii_df = index_prices.copy()
    for col in ["close", "adj_close"]:
        twii_df[col] = pd.to_numeric(twii_df[col], errors="coerce")
    twii_df["px"] = twii_df["adj_close"].where(twii_df["adj_close"].gt(0), twii_df["close"])
    twii = twii_df[twii_df["symbol"].eq("^TWII")].set_index("date")["px"].reindex(dates).ffill()
    twii_ma120 = (twii > twii.rolling(120, min_periods=60).mean()).shift(1).fillna(False)
    twii_ret60 = (twii / twii.shift(60) - 1.0).shift(1)

    score = pd.Series(0, index=dates, dtype=float)
    if "0050.TW" in core:
        score += (core["0050.TW"] > core["0050.TW"].rolling(120, min_periods=60).mean()).shift(1).fillna(False).astype(int)
        score += (core["0050.TW"] > core["0050.TW"].rolling(240, min_periods=120).mean()).shift(1).fillna(False).astype(int)
    if "2330.TW" in core:
        score += (core["2330.TW"] > core["2330.TW"].rolling(120, min_periods=60).mean()).shift(1).fillna(False).astype(int)
        score += ((core["2330.TW"] / core["2330.TW"].shift(60) - 1.0).shift(1) > 0).fillna(False).astype(int)
    score += twii_ma120.astype(int)
    score += (twii_ret60 > 0).fillna(False).astype(int)

    breadth_mask = amount_rank60.le(300) & amount_rank60.notna()
    breadth_ma60 = ((above_ma60 & breadth_mask).sum(axis=1) / breadth_mask.sum(axis=1).replace(0, np.nan)).fillna(0.0)
    breadth_ret20 = (((r20 > 0.0) & breadth_mask).sum(axis=1) / breadth_mask.sum(axis=1).replace(0, np.nan)).fillna(0.0)
    score += (breadth_ma60 >= float(candidate["breadth_ma60_min"])).astype(int)
    score += (breadth_ret20 >= float(candidate["breadth_ret20_min"])).astype(int)
    return score


def tw_regime_state(score: pd.Series, candidate: dict[str, Any], initial_state: int = 1) -> pd.Series:
    state = int(initial_state)
    out: list[int] = []
    strong_enter = int(candidate["strong_enter"])
    strong_exit = int(candidate["strong_exit"])
    normal_enter = int(candidate["normal_enter"])
    weak_score = int(candidate["weak_score"])
    for val in score.fillna(0).astype(int):
        if state == 2:
            if val <= weak_score:
                state = 0
            elif val < strong_exit:
                state = 1
        elif state == 1:
            if val >= strong_enter:
                state = 2
            elif val <= weak_score:
                state = 0
        else:
            if val >= strong_enter:
                state = 2
            elif val >= normal_enter:
                state = 1
        out.append(state)
    return pd.Series(out, index=score.index)


def tw_theme_scores(
    close: pd.DataFrame,
    groups: pd.Series,
    amount_rank60: pd.DataFrame,
    amount_ratio20_60: pd.DataFrame,
    above_ma20: pd.DataFrame,
    above_ma60: pd.DataFrame,
    r20: pd.DataFrame,
    r60: pd.DataFrame,
    r120: pd.DataFrame,
    candidate: dict[str, Any],
) -> pd.DataFrame:
    group_names = sorted(g for g in groups.dropna().unique() if str(g))
    scores = pd.DataFrame(np.nan, index=close.index, columns=group_names)
    base_liquid = amount_rank60.le(int(candidate["global_rank_cap"])) & amount_rank60.notna()
    require_ma = str(candidate["require_ma"])
    if require_ma == "ma20":
        eligible = base_liquid & above_ma20
    elif require_ma == "ma60":
        eligible = base_liquid & above_ma60
    else:
        eligible = base_liquid
    for group in group_names:
        syms = [s for s in groups[groups.eq(group)].index if s in close.columns]
        if len(syms) < 3:
            continue
        row_ok = eligible[syms].fillna(False)
        valid_count = row_ok.sum(axis=1)
        g20 = r20[syms].where(row_ok)
        g60 = r60[syms].where(row_ok)
        g120 = r120[syms].where(row_ok)
        positive = (g120 > 0.0).sum(axis=1) / g120.notna().sum(axis=1).replace(0, np.nan)
        val = 0.20 * g20.median(axis=1) + 0.35 * g60.median(axis=1) + 0.35 * g120.median(axis=1) + 0.10 * positive
        scores[group] = val.where(valid_count >= 3)
    return scores


def tw_select_for_date(
    date: pd.Timestamp,
    close: pd.DataFrame,
    open_: pd.DataFrame,
    prev_close: pd.DataFrame,
    groups: pd.Series,
    stock_score: pd.DataFrame,
    theme_scores: pd.DataFrame,
    amount_rank60: pd.DataFrame,
    amount_ratio20_60: pd.DataFrame,
    top10_mask: pd.DataFrame,
    above_ma20: pd.DataFrame,
    above_ma60: pd.DataFrame,
    r20: pd.DataFrame,
    r60: pd.DataFrame,
    candidate: dict[str, Any],
    state: int,
) -> tuple[list[str], list[str]]:
    if state == 0:
        return [], []
    theme_row = theme_scores.loc[date].dropna().sort_values(ascending=False)
    selected_themes = list(theme_row.head(int(candidate["category_top_n"])).index)
    if not selected_themes:
        return [], []
    if str(candidate["require_ma"]) == "ma20":
        ma_ok = above_ma20.loc[date]
    elif str(candidate["require_ma"]) == "ma60":
        ma_ok = above_ma60.loc[date]
    else:
        ma_ok = pd.Series(True, index=close.columns)
    gap = open_.loc[date] / prev_close.loc[date] - 1.0
    common = pd.Series([is_common_stock(s) for s in close.columns], index=close.columns)
    base_ok = (
        common
        & top10_mask.loc[date].fillna(False)
        & ma_ok.fillna(False)
        & stock_score.loc[date].replace([np.inf, -np.inf], np.nan).notna()
        & open_.loc[date].gt(0).fillna(False)
        & close.loc[date].gt(0).fillna(False)
        & prev_close.loc[date].gt(0).fillna(False)
        & amount_rank60.loc[date].le(int(candidate["global_rank_cap"])).fillna(False)
        & r20.loc[date].ge(float(candidate["min_ret20"])).fillna(False)
        & r60.loc[date].ge(float(candidate["min_ret60"])).fillna(False)
        & amount_ratio20_60.loc[date].ge(float(candidate["min_amount_ratio"])).fillna(False)
        & gap.le(float(candidate["gap_max"])).fillna(False)
    )
    selected: list[str] = []
    for theme in selected_themes:
        syms = [s for s in groups[groups.eq(theme)].index if s in close.columns]
        pool = [s for s in syms if bool(base_ok.get(s, False))]
        pool = sorted(pool, key=lambda s: float(stock_score.at[date, s]), reverse=True)
        selected.extend(pool[: int(candidate["buy_per_category"])])
    selected = sorted(set(selected), key=lambda s: float(stock_score.at[date, s]), reverse=True)
    return selected[: int(candidate["max_positions"])], selected_themes


def build_top10_mask(amount_rank60: pd.DataFrame, groups: pd.Series, top_n: int = 10) -> pd.DataFrame:
    out = pd.DataFrame(False, index=amount_rank60.index, columns=amount_rank60.columns)
    for date in amount_rank60.index:
        ranks = amount_rank60.loc[date]
        for group in sorted(g for g in groups.dropna().unique() if str(g)):
            syms = [s for s in groups[groups.eq(group)].index if s in ranks.index and is_common_stock(s)]
            ordered = ranks[syms].dropna().sort_values().head(top_n).index
            out.loc[date, ordered] = True
    return out


def weak_theme_cap(rule: str, p20: float, p60: float) -> float:
    if rule == "none":
        return 1.0
    if rule == "p60_neg_cap30":
        return 0.3 if np.isfinite(p60) and p60 < 0.0 else 1.0
    if rule == "p20_neg_cap30":
        return 0.3 if np.isfinite(p20) and p20 < 0.0 else 1.0
    if rule == "p20_p60_neg_cap0":
        return 0.0 if np.isfinite(p20) and np.isfinite(p60) and p20 < 0.0 and p60 < 0.0 else 1.0
    return 1.0


def simulate_tw_theme_switch(
    close: pd.DataFrame,
    selected_by_date: dict[pd.Timestamp, list[str]],
    px2330: pd.Series,
    rule: dict[str, Any],
    initial_mode: str = "2330",
) -> tuple[pd.Series, pd.Series, pd.Series]:
    theme_ret = pd.Series(0.0, index=close.index)
    stock_ret = close.pct_change().fillna(0.0)
    for date, syms in selected_by_date.items():
        usable = [s for s in syms if s in stock_ret.columns]
        if usable:
            theme_ret.at[date] = float(stock_ret.loc[date, usable].mean())
    theme_eq = (1.0 + theme_ret).cumprod()
    ret2330 = px2330.pct_change().fillna(0.0)
    mode = str(initial_mode or "2330")
    modes: list[str] = []
    allocs: list[float] = []
    for n, _date in enumerate(close.index):
        if n > 0:
            prev = n - 1
            lb = int(rule["lookback"])
            trend = int(rule["trend_window"])
            lb_idx = prev - lb
            tr_idx = prev - trend
            f20_idx = prev - 20
            f60_idx = prev - 60
            t_rel = theme_eq.iloc[prev] / theme_eq.iloc[lb_idx] - 1.0 if lb_idx >= 0 else np.nan
            p_rel = px2330.iloc[prev] / px2330.iloc[lb_idx] - 1.0 if lb_idx >= 0 else np.nan
            spread = t_rel - p_rel if np.isfinite(t_rel) and np.isfinite(p_rel) else np.nan
            t_trend = theme_eq.iloc[prev] / theme_eq.iloc[tr_idx] - 1.0 if tr_idx >= 0 else np.nan
            p_trend = px2330.iloc[prev] / px2330.iloc[tr_idx] - 1.0 if tr_idx >= 0 else np.nan
            p20 = px2330.iloc[prev] / px2330.iloc[f20_idx] - 1.0 if f20_idx >= 0 else np.nan
            p60 = px2330.iloc[prev] / px2330.iloc[f60_idx] - 1.0 if f60_idx >= 0 else np.nan
            weak_cap = weak_theme_cap(str(rule["weak_rule"]), p20, p60)
            if mode == "2330":
                can_enter = (
                    np.isfinite(spread)
                    and np.isfinite(t_trend)
                    and spread >= float(rule["enter_spread"])
                    and t_trend >= float(rule["theme_trend_min"])
                )
                if bool(rule["block_enter_when_weak"]) and weak_cap < 1.0:
                    can_enter = False
                if can_enter:
                    mode = "theme"
            else:
                reclaim = np.isfinite(p_trend) and np.isfinite(t_trend) and p_trend - t_trend >= float(rule["2330_reclaim"])
                spread_exit = np.isfinite(spread) and spread <= float(rule["exit_spread"])
                if spread_exit or reclaim:
                    mode = "2330"
        if mode == "theme":
            alloc = min(float(rule["theme_weight"]), 1.0)
        else:
            alloc = 0.0
        modes.append(mode)
        allocs.append(alloc)
    return theme_eq, pd.Series(modes, index=close.index), pd.Series(allocs, index=close.index)


TW_TOP1_POOL_GROUPS = {
    "all_tech": [
        "pcb_material",
        "pcb",
        "substrate",
        "thermal_cooling",
        "ai_server",
        "networking_switch",
        "ic_design_highspeed",
        "ic_design_mobile",
        "packaging_test",
        "semiconductor_equipment",
        "power_supply",
        "connector",
        "foundry",
        "passive_component",
        "dram_nand",
        "wafer_material",
        "semiconductor_general",
        "component_general",
        "equipment_service",
        "optical_communication",
        "server_board",
        "storage_device",
    ]
}

TW_TOP1_COMPONENT_RULES = {
    "DEFENSE_040838": {
        "candidate_id": "TWAGGCLEAN-040838",
        "pool": "all_tech",
        "score": "ultra20",
        "n": 2,
        "rebalance_n": 46,
        "min_ret20": -0.02,
        "min_ret60": 0.05,
        "min_ret120": 0.0,
        "min_high120_dd": -0.6,
        "min_amount_ratio": 0.7,
        "liquidity_rank_max": 100,
        "market_cap_rank_max": 250,
    },
    "VALMAX_046612": {
        "candidate_id": "TWAGGCLEAN-046612",
        "pool": "all_tech",
        "score": "accel60",
        "n": 2,
        "rebalance_n": 46,
        "min_ret20": -0.02,
        "min_ret60": 0.05,
        "min_ret120": 0.05,
        "min_high120_dd": -0.6,
        "min_amount_ratio": 0.0,
        "liquidity_rank_max": 250,
        "market_cap_rank_max": 250,
    },
    "TOP5_041880": {
        "candidate_id": "TWAGGCLEAN-041880",
        "pool": "all_tech",
        "score": "ultra20",
        "n": 2,
        "rebalance_n": 46,
        "min_ret20": -0.10,
        "min_ret60": 0.0,
        "min_ret120": 0.05,
        "min_high120_dd": -0.6,
        "min_amount_ratio": 0.7,
        "liquidity_rank_max": 250,
        "market_cap_rank_max": 250,
    },
    "BALANCED_058008": {
        "candidate_id": "TWAGGCLEAN-058008",
        "pool": "all_tech",
        "score": "combo",
        "n": 2,
        "rebalance_n": 46,
        "min_ret20": -0.10,
        "min_ret60": 0.0,
        "min_ret120": 0.05,
        "min_high120_dd": -0.6,
        "min_amount_ratio": 0.7,
        "liquidity_rank_max": 250,
        "market_cap_rank_max": 250,
    },
}

TW_PRODUCTION_STRATEGY_ID = "TW_ATTACK_RECOMMEND_FUSION_H46_EDGE005_CLEAN"
TW_PRODUCTION_REBALANCE_STEP = 46
TW_PRODUCTION_EXECUTION_TOP_N = 2
TW_PRODUCTION_FALLBACK_LEG = "DEFENSE_040838__TOP2"
TW_PRODUCTION_STRATEGY_MODE = "attack_recommend_fusion_h46_edge005"
TW_PRODUCTION_SWITCH_THRESHOLD = 0.20
TW_PRODUCTION_CURRENT_MODE_BONUS = 0.05
TW_ATTACK_COMPONENT_SCORE_WEIGHTS = (0.45, 0.35, 0.15, 0.05)
TW_RECOMMEND_COMPONENT_SCORE_WEIGHTS = (0.20, 0.20, 0.45, 0.15)
TW_FUSION_SLEEVE_SCORE_WEIGHTS = (0.45, 0.35, 0.15, 0.05)
TW_ATTACK_COMPONENT_THRESHOLD = 0.05
TW_ATTACK_CURRENT_MODE_BONUS = 0.10
TW_RECOMMEND_COMPONENT_THRESHOLD = 0.16
TW_RECOMMEND_CURRENT_MODE_BONUS = 0.00
TW_FUSION_ATTACK_EDGE_THRESHOLD = 0.05
TW_FUSION_ATTACK_MIN_DD60 = -0.60
TW_FUSION_ATTACK_MIN_R60 = 0.00


def tw_leg_base(component: str) -> str:
    return str(component).split("__TOP", 1)[0]


def tw_leg_top_n(component: str) -> int:
    raw = str(component)
    if "__TOP" not in raw:
        return int(TW_TOP1_COMPONENT_RULES.get(raw, {}).get("n", TW_PRODUCTION_EXECUTION_TOP_N))
    try:
        return max(1, int(raw.rsplit("__TOP", 1)[1]))
    except ValueError:
        return TW_PRODUCTION_EXECUTION_TOP_N


def tw_rule_for_leg(component: str) -> dict[str, Any]:
    base = tw_leg_base(component)
    rule = dict(TW_TOP1_COMPONENT_RULES[base])
    rule["n"] = tw_leg_top_n(component)
    return rule


def cap_weights_np(w: np.ndarray, cap: float) -> np.ndarray:
    w = w.astype(float, copy=True)
    total = float(w.sum())
    if total <= 0:
        return w
    w /= total
    for _ in range(10):
        over = w > cap
        if not over.any():
            break
        excess = float((w[over] - cap).sum())
        w[over] = cap
        under = ~over
        under_sum = float(w[under].sum())
        if under_sum <= 0:
            break
        w[under] += excess * w[under] / under_sum
    s = float(w.sum())
    return w / s if s > 0 else w


def tw_top1_feature_pack(close: pd.DataFrame, volume: pd.DataFrame) -> dict[str, pd.DataFrame]:
    amount = close * volume
    amount_ratio = amount.rolling(20, min_periods=10).mean() / amount.rolling(60, min_periods=30).mean()
    amount_bonus = amount_ratio.rank(axis=1, pct=True).fillna(0.0)
    r20 = close / close.shift(20) - 1.0
    r60 = close / close.shift(60) - 1.0
    r120 = close / close.shift(120) - 1.0
    high120_dd = close / close.rolling(120, min_periods=40).max() - 1.0
    return {
        "r20": r20,
        "r60": r60,
        "r120": r120,
        "high120_dd": high120_dd,
        "amount_ratio": amount_ratio,
        "ultra20": 0.65 * r20 + 0.25 * r60 + 0.10 * r120,
        "accel60": 0.35 * r20 + 0.45 * r60 + 0.20 * r120,
        "combo": 0.20 * r20 + 0.30 * r60 + 0.40 * r120 + 0.10 * amount_bonus,
    }


def tw_simulate_component(
    name: str,
    rule: dict[str, Any],
    close: pd.DataFrame,
    volume: pd.DataFrame,
    symbol_map: pd.DataFrame,
    features: dict[str, pd.DataFrame],
) -> tuple[pd.Series, pd.DataFrame, pd.Series]:
    dates = close.index
    symbols = list(close.columns)
    ret = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    mp = symbol_map.drop_duplicates("symbol").set_index("symbol").reindex(symbols)
    strategy_group = mp["strategy_group"].fillna("").astype(str)
    liquidity = pd.to_numeric(mp["liquidity_rank"], errors="coerce").fillna(9999.0)
    cap_rank = pd.to_numeric(mp["true_market_cap_rank"], errors="coerce").fillna(9999.0)
    pool_groups = TW_TOP1_POOL_GROUPS[str(rule["pool"])]
    pool_mask = (
        strategy_group.isin(pool_groups)
        & liquidity.le(float(rule["liquidity_rank_max"]))
        & cap_rank.le(float(rule["market_cap_rank_max"]))
    )
    pool_symbols = list(pool_mask[pool_mask].index)
    weights = pd.Series(0.0, index=symbols)
    equity = 1.0
    eq_vals: list[float] = []
    pos_rows: list[dict[str, Any]] = []
    score = features[str(rule["score"])]
    for j, date in enumerate(dates):
        if j > 0:
            daily = float((ret.iloc[j] * weights).sum())
            equity *= 1.0 + daily
        eq_vals.append(equity)
        if j == 0 or j % int(rule["rebalance_n"]) == 0:
            sig_i = max(0, j - 1)
            sig_date = dates[sig_i]
            sig = score.iloc[sig_i].reindex(pool_symbols)
            ok = (
                sig.replace([np.inf, -np.inf], np.nan).notna()
                & close.iloc[sig_i].reindex(pool_symbols).gt(0).fillna(False)
                & features["r20"].iloc[sig_i].reindex(pool_symbols).ge(float(rule["min_ret20"])).fillna(False)
                & features["r60"].iloc[sig_i].reindex(pool_symbols).ge(float(rule["min_ret60"])).fillna(False)
                & features["r120"].iloc[sig_i].reindex(pool_symbols).ge(float(rule["min_ret120"])).fillna(False)
                & features["high120_dd"].iloc[sig_i].reindex(pool_symbols).ge(float(rule["min_high120_dd"])).fillna(False)
                & features["amount_ratio"].iloc[sig_i].reindex(pool_symbols).ge(float(rule["min_amount_ratio"])).fillna(False)
            )
            chosen = sig[ok].sort_values(ascending=False).head(int(rule["n"])).index.tolist()
            weights[:] = 0.0
            if chosen:
                sleeve = 1.0 / np.arange(1, len(chosen) + 1, dtype=float)
                sleeve = cap_weights_np(sleeve, 0.30)
                for sym, weight in zip(chosen, sleeve):
                    weights.at[sym] = float(weight)
            active = weights[weights > 1e-12].sort_values(ascending=False)
            pos_rows.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "signal_date": sig_date.strftime("%Y-%m-%d"),
                    "component": name,
                    "symbols": "|".join(active.index.astype(str)),
                    "weights": "|".join(f"{float(x):.6f}" for x in active.values),
                }
            )
    return pd.Series(eq_vals, index=dates, name=name), pd.DataFrame(pos_rows), weights.copy()


def tw_top1_switch_score(component_equity: pd.DataFrame) -> pd.DataFrame:
    r20 = component_equity / component_equity.shift(20) - 1.0
    r60 = component_equity / component_equity.shift(60) - 1.0
    r120 = component_equity / component_equity.shift(120) - 1.0
    dd60 = component_equity / component_equity.rolling(60, min_periods=20).max() - 1.0
    return 0.25 * r20 + 0.35 * r60 + 0.30 * r120 + 0.10 * dd60


def tw_component_switch_score(component_equity: pd.DataFrame, weights: tuple[float, float, float, float]) -> pd.DataFrame:
    r20 = component_equity / component_equity.shift(20) - 1.0
    r60 = component_equity / component_equity.shift(60) - 1.0
    r120 = component_equity / component_equity.shift(120) - 1.0
    dd60 = component_equity / component_equity.rolling(60, min_periods=20).max() - 1.0
    w20, w60, w120, wdd = weights
    return w20 * r20 + w60 * r60 + w120 * r120 + wdd * dd60


def tw_sleeve_score(equity: pd.Series, weights: tuple[float, float, float, float]) -> pd.Series:
    r20 = equity / equity.shift(20) - 1.0
    r60 = equity / equity.shift(60) - 1.0
    r120 = equity / equity.shift(120) - 1.0
    dd60 = equity / equity.rolling(60, min_periods=20).max() - 1.0
    w20, w60, w120, wdd = weights
    return w20 * r20 + w60 * r60 + w120 * r120 + wdd * dd60


def tw_latest_component_position(component_positions: dict[str, Any], component: str, date_str: str) -> dict[str, Any] | None:
    pos = component_positions.get(component)
    if pos is None:
        return None
    if isinstance(pos, pd.DataFrame):
        if pos.empty or "date" not in pos.columns:
            return None
        frame = pos[pos["date"].astype(str) <= date_str]
        if frame.empty:
            return None
        return frame.iloc[-1].to_dict()
    if isinstance(pos, list):
        best = None
        for row in pos:
            if str(row.get("date", "")) <= date_str:
                best = row
            else:
                break
        return dict(best) if best is not None else None
    return None


def tw_parse_position_weights(row: dict[str, Any] | None) -> tuple[list[str], list[float]]:
    if row is None:
        return [], []
    syms = [s for s in str(row.get("symbols", "")).split("|") if s]
    weights: list[float] = []
    for raw in [s for s in str(row.get("weights", "")).split("|") if s]:
        try:
            weights.append(float(raw))
        except Exception:
            weights.append(0.0)
    return syms, weights


def tw_component_position_weights(
    component_positions: dict[str, Any],
    component: str,
    date: pd.Timestamp,
    symbols: list[str],
    equal_execution: bool = True,
) -> tuple[dict[str, float], list[str], list[float], list[float]]:
    row = tw_latest_component_position(component_positions, component, pd.Timestamp(date).strftime("%Y-%m-%d"))
    syms, raw_ws = tw_parse_position_weights(row)
    syms = [sym for sym in syms if sym in symbols]
    raw_ws = raw_ws[: len(syms)]
    gross = min(1.0, max(0.0, float(sum(raw_ws)))) if raw_ws else 0.0
    if not syms or gross <= 0:
        return {}, syms, raw_ws, []
    exec_ws = [gross / len(syms)] * len(syms) if equal_execution else raw_ws
    return {sym: float(weight) for sym, weight in zip(syms, exec_ws)}, syms, raw_ws, exec_ws


def tw_simulate_component_switch(
    component_equity: pd.DataFrame,
    score_weights: tuple[float, float, float, float],
    min_top_score: float,
    stay_bonus: float,
    fallback: str,
    sleeve_name: str,
    component_positions: dict[str, Any] | None = None,
    close: pd.DataFrame | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    dates = component_equity.index
    scores = tw_component_switch_score(component_equity, score_weights)
    component_ret = component_equity.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    stock_ret = None
    stock_weights = None
    stock_symbols: list[str] = []
    if close is not None:
        stock_symbols = list(close.columns)
        stock_ret = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        stock_weights = pd.Series(0.0, index=stock_symbols, dtype=float)
    equity = 1.0
    eq_vals: list[float] = []
    selected_rows: list[dict[str, Any]] = []
    mode = fallback
    for j, date in enumerate(dates):
        if j > 0:
            if stock_ret is not None and stock_weights is not None:
                daily = float((stock_ret.iloc[j] * stock_weights).sum())
            else:
                daily = float(component_ret.iloc[j].get(mode, 0.0)) if mode in component_ret.columns else 0.0
            equity *= 1.0 + daily
        eq_vals.append(float(equity))
        if j == 0 or j % TW_PRODUCTION_REBALANCE_STEP == 0:
            sig_i = max(0, j - 1)
            sig_date = dates[sig_i]
            row = scores.iloc[sig_i].copy()
            if mode in row.index and np.isfinite(row.get(mode, np.nan)):
                row.at[mode] = float(row.at[mode]) + float(stay_bonus)
            row = row.replace([np.inf, -np.inf], np.nan).dropna()
            top_score = float("nan")
            second_score = float("nan")
            selected = fallback
            if len(row):
                ranked = row.sort_values(ascending=False)
                top_score = float(ranked.iloc[0])
                second_score = float(ranked.iloc[1]) if len(ranked) > 1 else float("nan")
                selected = str(ranked.index[0]) if top_score >= float(min_top_score) else fallback
            mode = selected
            pos_weights: dict[str, float] = {}
            pos_syms: list[str] = []
            raw_ws: list[float] = []
            exec_ws: list[float] = []
            if stock_weights is not None:
                stock_weights[:] = 0.0
                if component_positions is not None:
                    pos_weights, pos_syms, raw_ws, exec_ws = tw_component_position_weights(
                        component_positions,
                        mode,
                        pd.Timestamp(date),
                        stock_symbols,
                        equal_execution=True,
                    )
                    for sym, weight in pos_weights.items():
                        stock_weights.at[sym] = float(weight)
            selected_rows.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "signal_date": sig_date.strftime("%Y-%m-%d"),
                    "score_preset": TW_PRODUCTION_STRATEGY_ID,
                    "mode": f"{TW_PRODUCTION_STRATEGY_MODE}_{sleeve_name}",
                    "selected_sleeve": sleeve_name,
                    "selected": mode,
                    "selected_component": mode,
                    "top_score": top_score,
                    "second_score": second_score,
                    "fallback": fallback,
                    "component_weights": f"{mode}:1.000000",
                    "symbols": "|".join(pos_syms),
                    "raw_weights": "|".join(f"{float(x):.6f}" for x in raw_ws),
                    "exec_weights": "|".join(f"{float(x):.6f}" for x in exec_ws),
                }
            )
    return pd.Series(eq_vals, index=dates, dtype=float, name=sleeve_name), pd.DataFrame(selected_rows)


def tw_fusion_sleeve_metrics(attack_eq: pd.Series, recommend_eq: pd.Series) -> dict[str, pd.Series]:
    attack_score = tw_sleeve_score(attack_eq, TW_FUSION_SLEEVE_SCORE_WEIGHTS)
    recommend_score = tw_sleeve_score(recommend_eq, TW_FUSION_SLEEVE_SCORE_WEIGHTS)
    return {
        "attack_score": attack_score,
        "recommend_score": recommend_score,
        "attack_edge": attack_score - recommend_score,
        "attack_dd60": attack_eq / attack_eq.rolling(60, min_periods=20).max() - 1.0,
        "attack_r60": attack_eq / attack_eq.shift(60) - 1.0,
    }


def tw_should_use_attack(metric_row: dict[str, float]) -> bool:
    edge = float(metric_row.get("attack_edge", np.nan))
    dd60 = float(metric_row.get("attack_dd60", np.nan))
    r60 = float(metric_row.get("attack_r60", np.nan))
    return (
        np.isfinite(edge)
        and np.isfinite(dd60)
        and np.isfinite(r60)
        and edge >= TW_FUSION_ATTACK_EDGE_THRESHOLD
        and dd60 >= TW_FUSION_ATTACK_MIN_DD60
        and r60 >= TW_FUSION_ATTACK_MIN_R60
    )


def tw_select_fusion_component(
    component_equity: pd.DataFrame,
    component_positions: dict[str, Any] | None = None,
    close: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    attack_eq, attack_modes = tw_simulate_component_switch(
        component_equity,
        TW_ATTACK_COMPONENT_SCORE_WEIGHTS,
        TW_ATTACK_COMPONENT_THRESHOLD,
        TW_ATTACK_CURRENT_MODE_BONUS,
        TW_PRODUCTION_FALLBACK_LEG,
        "attack",
        component_positions,
        close,
    )
    recommend_eq, recommend_modes = tw_simulate_component_switch(
        component_equity,
        TW_RECOMMEND_COMPONENT_SCORE_WEIGHTS,
        TW_RECOMMEND_COMPONENT_THRESHOLD,
        TW_RECOMMEND_CURRENT_MODE_BONUS,
        TW_PRODUCTION_FALLBACK_LEG,
        "recommend",
        component_positions,
        close,
    )
    metrics = tw_fusion_sleeve_metrics(attack_eq, recommend_eq)
    attack_by_date = attack_modes.set_index("date").to_dict("index") if not attack_modes.empty else {}
    recommend_by_date = recommend_modes.set_index("date").to_dict("index") if not recommend_modes.empty else {}
    selected_rows: list[dict[str, Any]] = []
    selected_component = TW_PRODUCTION_FALLBACK_LEG
    selected_sleeve = "recommend"
    dates = component_equity.index
    for j, date in enumerate(dates):
        if j == 0 or j % TW_PRODUCTION_REBALANCE_STEP == 0:
            sig_i = max(0, j - 1)
            sig_date = dates[sig_i]
            metric_row = {
                "attack_score": float(metrics["attack_score"].iloc[sig_i]),
                "recommend_score": float(metrics["recommend_score"].iloc[sig_i]),
                "attack_edge": float(metrics["attack_edge"].iloc[sig_i]),
                "attack_dd60": float(metrics["attack_dd60"].iloc[sig_i]),
                "attack_r60": float(metrics["attack_r60"].iloc[sig_i]),
            }
            selected_sleeve = "attack" if tw_should_use_attack(metric_row) else "recommend"
            sleeve_row = (attack_by_date if selected_sleeve == "attack" else recommend_by_date).get(date.strftime("%Y-%m-%d"), {})
            selected_component = str(sleeve_row.get("selected_component") or sleeve_row.get("selected") or TW_PRODUCTION_FALLBACK_LEG)
            top_score = float(sleeve_row.get("top_score", np.nan))
            selected_rows.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "signal_date": sig_date.strftime("%Y-%m-%d"),
                    "score_preset": TW_PRODUCTION_STRATEGY_ID,
                    "mode": TW_PRODUCTION_STRATEGY_MODE,
                    "selected_sleeve": selected_sleeve,
                    "selected": selected_component,
                    "selected_component": selected_component,
                    "top_score": top_score,
                    "second_score": sleeve_row.get("second_score", np.nan),
                    "attack_score": metric_row["attack_score"],
                    "recommend_score": metric_row["recommend_score"],
                    "attack_edge": metric_row["attack_edge"],
                    "attack_dd60": metric_row["attack_dd60"],
                    "attack_r60": metric_row["attack_r60"],
                    "fallback": TW_PRODUCTION_FALLBACK_LEG,
                    "component_weights": f"{selected_component}:1.000000",
                }
            )
    return pd.DataFrame(selected_rows), pd.Series([selected_component], index=[dates[-1]])


def tw_select_top1_component(
    component_equity: pd.DataFrame,
    component_positions: dict[str, Any] | None = None,
    close: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    return tw_select_fusion_component(component_equity, component_positions, close)


def apply_tw_equal_weight_execution_overlay(target_rows: list[dict[str, Any]], source_component: str) -> list[dict[str, Any]]:
    raw_rows = [dict(row) for row in target_rows]
    stock_rows = [
        row
        for row in raw_rows
        if str(row.get("symbol", "")) != "CASH" and float(row.get("target_weight", 0.0) or 0.0) > 1e-12
    ]
    if not stock_rows:
        return raw_rows
    stock_gross = min(1.0, max(0.0, sum(float(row.get("target_weight", 0.0) or 0.0) for row in stock_rows)))
    equal_weight = stock_gross / len(stock_rows)
    out: list[dict[str, Any]] = []
    for row in stock_rows:
        rr = dict(row)
        rr["raw_target_weight"] = float(row.get("target_weight", 0.0) or 0.0)
        rr["target_weight"] = equal_weight
        rr["role"] = f"{rr.get('role', '')}|equal_weight_exec"
        rr["source_sleeve"] = source_component
        rr["execution_group"] = "main"
        rr["execution_overlay"] = "TW_EQUAL_WEIGHT"
        out.append(rr)
    if stock_gross < 1.0 - 1e-12:
        out.append(
            {
                "symbol": "CASH",
                "name": "Cash",
                "theme": "cash",
                "role": "residual_cash",
                "target_weight": 1.0 - stock_gross,
                "source_sleeve": "execution_overlay",
                "execution_group": "main",
                "execution_overlay": "TW_EQUAL_WEIGHT",
            }
        )
    return out


def tw_component_stock_weights_at(
    component: str,
    rule: dict[str, Any],
    close: pd.DataFrame,
    symbol_map: pd.DataFrame,
    features: dict[str, pd.DataFrame],
    signal_i: int,
) -> dict[str, float]:
    symbols = list(close.columns)
    mp = symbol_map.drop_duplicates("symbol").set_index("symbol").reindex(symbols)
    strategy_group = mp["strategy_group"].fillna("").astype(str)
    liquidity = pd.to_numeric(mp["liquidity_rank"], errors="coerce").fillna(9999.0)
    cap_rank = pd.to_numeric(mp["true_market_cap_rank"], errors="coerce").fillna(9999.0)
    pool_groups = TW_TOP1_POOL_GROUPS[str(rule["pool"])]
    pool_mask = (
        strategy_group.isin(pool_groups)
        & liquidity.le(float(rule["liquidity_rank_max"]))
        & cap_rank.le(float(rule["market_cap_rank_max"]))
    )
    pool_symbols = list(pool_mask[pool_mask].index)
    sig = features[str(rule["score"])].iloc[signal_i].reindex(pool_symbols)
    ok = (
        sig.replace([np.inf, -np.inf], np.nan).notna()
        & close.iloc[signal_i].reindex(pool_symbols).gt(0).fillna(False)
        & features["r20"].iloc[signal_i].reindex(pool_symbols).ge(float(rule["min_ret20"])).fillna(False)
        & features["r60"].iloc[signal_i].reindex(pool_symbols).ge(float(rule["min_ret60"])).fillna(False)
        & features["r120"].iloc[signal_i].reindex(pool_symbols).ge(float(rule["min_ret120"])).fillna(False)
        & features["high120_dd"].iloc[signal_i].reindex(pool_symbols).ge(float(rule["min_high120_dd"])).fillna(False)
        & features["amount_ratio"].iloc[signal_i].reindex(pool_symbols).ge(float(rule["min_amount_ratio"])).fillna(False)
    )
    chosen = sig[ok].sort_values(ascending=False).head(int(rule["n"])).index.tolist()
    if not chosen:
        return {}
    sleeve = 1.0 / np.arange(1, len(chosen) + 1, dtype=float)
    sleeve = cap_weights_np(sleeve, 0.30)
    return {str(sym): float(weight) for sym, weight in zip(chosen, sleeve)}


def tw_live_go_live_selection(
    close: pd.DataFrame,
    symbol_map: pd.DataFrame,
    features: dict[str, pd.DataFrame],
    component_eq: pd.DataFrame,
    component_positions: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    rec = live_cycle_baseline("TW")
    if not rec or str(rec.get("baseline_type", "")) != "manual_go_live_close":
        return None
    cycle_date = str(rec.get("cycle_rebalance_date") or rec.get("baseline_date") or "").strip()
    if not cycle_date:
        return None
    try:
        ts = pd.Timestamp(cycle_date).normalize()
    except Exception:
        return None
    date_index = pd.DatetimeIndex(pd.to_datetime(close.index)).normalize()
    locs = np.where(date_index <= ts)[0]
    if len(locs) == 0:
        return None
    signal_i = int(locs[-1])
    signal_date = pd.Timestamp(close.index[signal_i]).strftime("%Y-%m-%d")
    attack_eq, _attack_modes = tw_simulate_component_switch(
        component_eq,
        TW_ATTACK_COMPONENT_SCORE_WEIGHTS,
        TW_ATTACK_COMPONENT_THRESHOLD,
        TW_ATTACK_CURRENT_MODE_BONUS,
        TW_PRODUCTION_FALLBACK_LEG,
        "attack",
        component_positions,
        close,
    )
    recommend_eq, _recommend_modes = tw_simulate_component_switch(
        component_eq,
        TW_RECOMMEND_COMPONENT_SCORE_WEIGHTS,
        TW_RECOMMEND_COMPONENT_THRESHOLD,
        TW_RECOMMEND_CURRENT_MODE_BONUS,
        TW_PRODUCTION_FALLBACK_LEG,
        "recommend",
        component_positions,
        close,
    )
    metrics = tw_fusion_sleeve_metrics(attack_eq, recommend_eq)
    metric_row = {
        "attack_score": float(metrics["attack_score"].iloc[signal_i]),
        "recommend_score": float(metrics["recommend_score"].iloc[signal_i]),
        "attack_edge": float(metrics["attack_edge"].iloc[signal_i]),
        "attack_dd60": float(metrics["attack_dd60"].iloc[signal_i]),
        "attack_r60": float(metrics["attack_r60"].iloc[signal_i]),
    }
    selected_sleeve = "attack" if tw_should_use_attack(metric_row) else "recommend"
    score_weights = TW_ATTACK_COMPONENT_SCORE_WEIGHTS if selected_sleeve == "attack" else TW_RECOMMEND_COMPONENT_SCORE_WEIGHTS
    min_score = TW_ATTACK_COMPONENT_THRESHOLD if selected_sleeve == "attack" else TW_RECOMMEND_COMPONENT_THRESHOLD
    scores = tw_component_switch_score(component_eq, score_weights).iloc[signal_i].replace([np.inf, -np.inf], np.nan).dropna()
    selected = TW_PRODUCTION_FALLBACK_LEG
    top_score = float("nan")
    second_score = float("nan")
    if len(scores):
        ranked = scores.sort_values(ascending=False)
        top_score = float(ranked.iloc[0])
        second_score = float(ranked.iloc[1]) if len(ranked) > 1 else float("nan")
        selected = str(ranked.index[0]) if top_score >= min_score else TW_PRODUCTION_FALLBACK_LEG
    weights = tw_component_stock_weights_at(
        selected,
        tw_rule_for_leg(selected),
        close,
        symbol_map,
        features,
        signal_i,
    )
    return {
        "selected_component": selected,
        "selected_sleeve": selected_sleeve,
        "signal_date": signal_date,
        "rebalance_date": signal_date,
        "top_score": top_score,
        "second_score": second_score,
        "attack_score": metric_row["attack_score"],
        "recommend_score": metric_row["recommend_score"],
        "attack_edge": metric_row["attack_edge"],
        "attack_dd60": metric_row["attack_dd60"],
        "attack_r60": metric_row["attack_r60"],
        "stock_weights": weights,
        "signal_i": signal_i,
        "cycle_date": cycle_date,
    }


def run_tw_top1_daily(state: dict[str, Any]) -> dict[str, Any]:
    TW_OUT.mkdir(parents=True, exist_ok=True)
    prices, _index_prices, symbol_map = load_tw_data()
    for col in ["symbol", "name", "strategy_group", "trade_group", "parent_group", "strategy_sector"]:
        if col not in symbol_map.columns:
            symbol_map[col] = ""
        symbol_map[col] = symbol_map[col].fillna("").astype(str)
    if os.getenv("TW_CAP_TO_MAP_LATEST", "1") != "0" and "latest_price_date" in symbol_map.columns:
        map_latest = pd.to_datetime(symbol_map["latest_price_date"], errors="coerce").max()
        if pd.notna(map_latest):
            prices = prices[prices["date"] <= map_latest].copy()
    symbols = sorted(set(symbol_map["symbol"].dropna().astype(str)))
    prices = prices[prices["symbol"].isin(symbols)].copy()
    close, _open, volume = px_frame(prices, symbols)
    if len(close.index) < 252:
        raise RuntimeError("TW Top1 live data should keep at least 252 trading rows")
    features = tw_top1_feature_pack(close, volume)
    component_equities: dict[str, pd.Series] = {}
    component_positions: dict[str, pd.DataFrame] = {}
    for name, rule in TW_TOP1_COMPONENT_RULES.items():
        for top_n in [1, 2]:
            leg = f"{name}__TOP{top_n}"
            leg_rule = dict(rule)
            leg_rule["n"] = top_n
            eq, pos, _weights = tw_simulate_component(leg, leg_rule, close, volume, symbol_map, features)
            component_equities[leg] = eq.rename(leg)
            component_positions[leg] = pos
    component_eq = pd.concat(component_equities, axis=1)
    switch_modes, latest_mode_series = tw_select_top1_component(component_eq, component_positions, close)
    latest_component = str(latest_mode_series.iloc[0])
    latest_date = close.index[-1]
    latest_mode_row = switch_modes.tail(1).iloc[0].to_dict()
    live_selection = tw_live_go_live_selection(close, symbol_map, features, component_eq, component_positions)
    if live_selection:
        latest_component = str(live_selection["selected_component"])
        latest_mode_row = {
            "date": live_selection["rebalance_date"],
            "signal_date": live_selection["signal_date"],
            "score_preset": TW_PRODUCTION_STRATEGY_ID,
            "mode": TW_PRODUCTION_STRATEGY_MODE,
            "selected_sleeve": live_selection.get("selected_sleeve"),
            "selected": latest_component,
            "top_score": live_selection["top_score"],
            "second_score": live_selection.get("second_score"),
            "attack_score": live_selection.get("attack_score"),
            "recommend_score": live_selection.get("recommend_score"),
            "attack_edge": live_selection.get("attack_edge"),
            "attack_dd60": live_selection.get("attack_dd60"),
            "attack_r60": live_selection.get("attack_r60"),
            "component_weights": f"{latest_component}:1.000000",
        }
    rebalance_step = TW_PRODUCTION_REBALANCE_STEP
    latest_rebalance_date = pd.Timestamp(latest_mode_row.get("date", latest_date))
    latest_signal_date = pd.Timestamp(latest_mode_row.get("signal_date", latest_date))
    latest_idx = len(close.index) - 1
    try:
        latest_rebalance_idx = int(close.index.get_loc(latest_rebalance_date))
    except KeyError:
        latest_rebalance_idx = latest_idx
    try:
        latest_signal_idx = int(close.index.get_loc(latest_signal_date))
    except KeyError:
        latest_signal_idx = latest_rebalance_idx
    trading_days_since_rebalance = max(0, latest_idx - latest_rebalance_idx)
    trading_days_since_signal = max(0, latest_idx - latest_signal_idx)
    rebalance_due_today = latest_rebalance_idx == latest_idx
    rebalance_due_next_session = (not rebalance_due_today) and (trading_days_since_rebalance + 1 >= rebalance_step)
    rebalance_days_remaining = max(0, rebalance_step - trading_days_since_rebalance)
    if rebalance_due_today:
        action_signal = "rebalance_fusion_target"
    elif rebalance_due_next_session:
        action_signal = "next_session_rebalance_pending"
    else:
        action_signal = "hold_fusion_target"

    mp2 = symbol_map.drop_duplicates("symbol").set_index("symbol")
    stock_weights: dict[str, float] = {}
    if live_selection:
        stock_weights = {
            str(sym): float(weight)
            for sym, weight in dict(live_selection.get("stock_weights", {})).items()
        }
    else:
        latest_pos = component_positions[latest_component].tail(1)
        if not latest_pos.empty:
            syms = [s for s in str(latest_pos.iloc[0]["symbols"]).split("|") if s]
            ws = [float(x) for x in str(latest_pos.iloc[0]["weights"]).split("|") if x]
            stock_weights = {sym: weight for sym, weight in zip(syms, ws)}
    target_rows: list[dict[str, Any]] = []
    for sym, weight in sorted(stock_weights.items(), key=lambda kv: kv[1], reverse=True):
        meta = mp2.loc[sym] if sym in mp2.index else {}
        target_rows.append(
            {
                "symbol": sym,
                "name": meta.get("name", "") if hasattr(meta, "get") else "",
                "theme": meta.get("strategy_group", "") if hasattr(meta, "get") else "",
                "role": f"tw_fusion_{latest_mode_row.get('selected_sleeve', 'recommend')}_{latest_component}",
                "target_weight": float(weight),
            }
        )
    residual = 1.0 - sum(float(row["target_weight"]) for row in target_rows)
    if residual > 1e-6:
        target_rows.append({"symbol": "CASH", "name": "Cash", "theme": "cash", "role": "rounding_cash", "target_weight": residual})
    raw_target_rows = [dict(row) for row in target_rows]
    target_rows = apply_tw_equal_weight_execution_overlay(target_rows, latest_component)
    final_stock_weights = {
        str(row["symbol"]): float(row["target_weight"])
        for row in target_rows
        if str(row.get("symbol", "")) != "CASH"
    }

    score_i = int(live_selection["signal_i"]) if live_selection else (-2 if len(component_eq) >= 2 else -1)
    score_sleeve = str(latest_mode_row.get("selected_sleeve") or "recommend")
    selected_score_weights = TW_ATTACK_COMPONENT_SCORE_WEIGHTS if score_sleeve == "attack" else TW_RECOMMEND_COMPONENT_SCORE_WEIGHTS
    latest_scores = tw_component_switch_score(component_eq, selected_score_weights).iloc[score_i].sort_values(ascending=False)
    theme_rank_rows = [
        {
            "rank": i,
            "theme": component,
            "theme_score": float(score),
            "selected": component == latest_component,
        }
        for i, (component, score) in enumerate(latest_scores.items(), 1)
        if np.isfinite(score)
    ]
    candidate_rows: list[dict[str, Any]] = []
    for component, pos in component_positions.items():
        if live_selection:
            comp_weights = tw_component_stock_weights_at(
                component,
                tw_rule_for_leg(component),
                close,
                symbol_map,
                features,
                int(live_selection["signal_i"]),
            )
            syms = list(comp_weights.keys())
            ws = list(comp_weights.values())
        else:
            if pos.empty:
                continue
            row = pos.tail(1).iloc[0]
            syms = [s for s in str(row["symbols"]).split("|") if s]
            ws = [float(x) for x in str(row["weights"]).split("|") if x]
        for rank, (sym, weight) in enumerate(zip(syms, ws), 1):
            meta = mp2.loc[sym] if sym in mp2.index else {}
            candidate_rows.append(
                {
                    "rank": rank,
                    "component": component,
                    "symbol": sym,
                    "name": meta.get("name", "") if hasattr(meta, "get") else "",
                    "theme": meta.get("strategy_group", "") if hasattr(meta, "get") else "",
                    "selected_today": component == latest_component,
                    "component_weight": float(weight),
                    "final_target_weight": float(final_stock_weights.get(sym, 0.0)) if component == latest_component else 0.0,
                }
            )

    tw_state = state.setdefault("TW", {})
    tw_state.update(
        {
            "latest_price_date": latest_date.strftime("%Y-%m-%d"),
            "outer_mode": latest_component,
            "target_position": target_rows,
            "raw_target_position": raw_target_rows,
            "top1_switch_rule": TW_PRODUCTION_STRATEGY_ID,
            "execution_overlay": "TW_EQUAL_WEIGHT",
            "selected_sleeve": latest_mode_row.get("selected_sleeve", "recommend"),
        }
    )
    market_state = {
        "market": "TW",
        "latest_price_date": latest_date,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "frozen_strategy": TW_PRODUCTION_STRATEGY_ID,
        "execution_overlay": "TW_EQUAL_WEIGHT",
        "strategy_mode": TW_PRODUCTION_STRATEGY_MODE,
        "outer_mode": latest_component,
        "theme_alloc": 1.0,
        "regime_state": 2,
        "regime_state_label": TW_PRODUCTION_STRATEGY_MODE,
        "selected_component": latest_component,
        "selected_sleeve": latest_mode_row.get("selected_sleeve", "recommend"),
        "component_signal_date": latest_mode_row.get("signal_date"),
        "component_rebalance_date": latest_rebalance_date.strftime("%Y-%m-%d"),
        "component_top_score": latest_mode_row.get("top_score"),
        "component_second_score": latest_mode_row.get("second_score"),
        "attack_score": latest_mode_row.get("attack_score"),
        "recommend_score": latest_mode_row.get("recommend_score"),
        "attack_edge": latest_mode_row.get("attack_edge"),
        "attack_dd60": latest_mode_row.get("attack_dd60"),
        "attack_r60": latest_mode_row.get("attack_r60"),
        "rebalance_step_trading_days": rebalance_step,
        "trading_days_since_rebalance": trading_days_since_rebalance,
        "trading_days_since_signal": trading_days_since_signal,
        "rebalance_days_remaining": rebalance_days_remaining,
        "rebalance_due_today_by_counter": rebalance_due_today,
        "rebalance_due_next_session_by_counter": rebalance_due_next_session,
        "rebalance_count_basis": "execution_date_not_signal_date",
        "action_signal": action_signal,
        "notes": (
            "TW production target uses clean Attack/Recommend Fusion H46. Attack sleeve is used only when its sleeve score beats recommend by 0.05, attack 60D return is non-negative, and attack 60D drawdown stays above -60%; otherwise recommend sleeve is used. "
            "Execution overlay remains equal weight across selected stocks. "
            "Manual go-live baseline recomputes the initial live target from that baseline close and then holds it until the next live cycle. "
            "No retraining in cloud package."
        ),
    }
    market_state = apply_live_cycle_baseline(market_state, "TW", close.index)
    pd.DataFrame([market_state]).to_csv(TW_OUT / "latest_market_state.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(theme_rank_rows).to_csv(TW_OUT / "latest_theme_rank.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(candidate_rows).to_csv(TW_OUT / "latest_candidate_pool.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(target_rows).to_csv(TW_OUT / "latest_target_position.csv", index=False, encoding="utf-8-sig")
    switch_modes.to_csv(TW_OUT / "latest_top1_switch_modes.csv", index=False, encoding="utf-8-sig")
    switch_modes.to_csv(TW_OUT / "latest_top2_h46_switch_modes.csv", index=False, encoding="utf-8-sig")
    switch_modes.to_csv(TW_OUT / "latest_top1_top2_switch_h46_th020_modes.csv", index=False, encoding="utf-8-sig")
    switch_modes.to_csv(TW_OUT / "latest_attack_recommend_fusion_h46_modes.csv", index=False, encoding="utf-8-sig")
    action_report = {
        "market_state": market_state,
        "target_position": target_rows,
        "raw_target_position": raw_target_rows,
        "component_scores": theme_rank_rows,
        "component_candidates": candidate_rows,
    }
    write_json(TW_OUT / "latest_action_report.json", action_report)
    return action_report


def run_tw_daily(state: dict[str, Any]) -> dict[str, Any]:
    return run_tw_top1_daily(state)
    TW_OUT.mkdir(parents=True, exist_ok=True)
    switch_rule = pd.read_csv(RULE_DIR / "tw_selected_rule.csv").iloc[0].to_dict()
    theme_rule = pd.read_csv(RULE_DIR / "tw_theme_rule.csv").iloc[0].to_dict()
    prices, index_prices, symbol_map = load_tw_data()
    map_cols = ["symbol", "name", "strategy_group", "trade_group", "parent_group", "strategy_sector"]
    for col in map_cols:
        if col not in symbol_map.columns:
            symbol_map[col] = ""
        symbol_map[col] = symbol_map[col].fillna("").astype(str)
    if os.getenv("TW_CAP_TO_MAP_LATEST", "1") != "0" and "latest_price_date" in symbol_map.columns:
        map_latest = pd.to_datetime(symbol_map["latest_price_date"], errors="coerce").max()
        if pd.notna(map_latest):
            prices = prices[prices["date"] <= map_latest].copy()
            index_prices = index_prices[index_prices["date"] <= map_latest].copy()
    symbols = sorted(set(symbol_map["symbol"].dropna().astype(str)) | {"0050.TW", "2330.TW"})
    prices = prices[prices["symbol"].isin(symbols)].copy()
    close, open_, volume = px_frame(prices, symbols)
    dates = close.index
    if len(dates) < 252:
        raise RuntimeError("TW live data should keep at least 252 trading rows")
    latest_date = dates[-1]
    prev_close = close.shift(1)
    amount = close * volume
    amount_med20 = amount.rolling(20, min_periods=10).median().shift(1)
    amount_med60 = amount.rolling(60, min_periods=30).median().shift(1)
    amount_ratio20_60 = amount_med20 / amount_med60
    amount_rank60 = amount_med60.rank(axis=1, method="min", ascending=False)
    r20 = shifted_ret(close, 20)
    r60 = shifted_ret(close, 60)
    r120 = shifted_ret(close, 120)
    above_ma20 = shifted_above_ma(close, 20)
    above_ma60 = shifted_above_ma(close, 60)
    stock_score = tw_stock_score(r20, r60, r120, str(theme_rule["stock_score"]))
    groups = symbol_map.drop_duplicates("symbol").set_index("symbol")[str(theme_rule["group_col"])].reindex(close.columns).fillna("")
    top10_mask = build_top10_mask(amount_rank60.loc[[latest_date]], groups, top_n=int(theme_rule.get("category_leader_n", 10)))
    theme_scores = tw_theme_scores(
        close.loc[[latest_date]],
        groups,
        amount_rank60.loc[[latest_date]],
        amount_ratio20_60.loc[[latest_date]],
        above_ma20.loc[[latest_date]],
        above_ma60.loc[[latest_date]],
        r20.loc[[latest_date]],
        r60.loc[[latest_date]],
        r120.loc[[latest_date]],
        theme_rule,
    )
    market_score = tw_market_state_series(close, amount_rank60, r20, r60, above_ma60, index_prices, theme_rule)
    tw_state = state.setdefault("TW", {})
    initial_state = int(tw_state.get("regime_state", 1))
    regimes = tw_regime_state(market_score, theme_rule, initial_state=initial_state)

    latest_selected, latest_themes = tw_select_for_date(
        latest_date,
        close,
        open_,
        prev_close,
        groups,
        stock_score,
        theme_scores,
        amount_rank60,
        amount_ratio20_60,
        top10_mask,
        above_ma20,
        above_ma60,
        r20,
        r60,
        theme_rule,
        int(regimes.loc[latest_date]),
    )
    selected_by_date = {date: latest_selected for date in dates}
    px2330 = close["2330.TW"].ffill()
    _theme_eq, modes, allocs = simulate_tw_theme_switch(
        close,
        selected_by_date,
        px2330,
        switch_rule,
        initial_mode=str(tw_state.get("outer_mode", "2330")),
    )
    latest_mode = str(modes.loc[latest_date])
    theme_alloc = float(allocs.loc[latest_date])
    regime_state = int(regimes.loc[latest_date])
    state_stock_exposure = (
        float(theme_rule["strong_stock_exposure"])
        if regime_state == 2
        else float(theme_rule["normal_stock_exposure"])
        if regime_state == 1
        else 0.0
    )
    stock_sleeve = theme_alloc * state_stock_exposure
    core_weight = max(0.0, 1.0 - theme_alloc)
    cash_weight = max(0.0, theme_alloc - stock_sleeve)

    mp2 = symbol_map.drop_duplicates("symbol").set_index("symbol")
    target_rows: list[dict[str, Any]] = []
    if core_weight > 1e-9:
        target_rows.append({"symbol": "2330.TW", "name": "台積電", "theme": "core_2330", "role": "core", "target_weight": core_weight})
    if latest_mode == "theme" and latest_selected and stock_sleeve > 0:
        per = stock_sleeve / len(latest_selected)
        for sym in latest_selected:
            meta = mp2.loc[sym] if sym in mp2.index else {}
            target_rows.append(
                {
                    "symbol": sym,
                    "name": meta.get("name", "") if hasattr(meta, "get") else "",
                    "theme": groups.get(sym, ""),
                    "role": "theme_leader",
                    "target_weight": per,
                }
            )
    if cash_weight > 1e-9:
        target_rows.append({"symbol": "CASH", "name": "Cash", "theme": "cash", "role": "weak_or_residual_cash", "target_weight": cash_weight})

    theme_rank_rows: list[dict[str, Any]] = []
    latest_theme_scores = theme_scores.loc[latest_date].dropna().sort_values(ascending=False)
    for rank, (theme_name, score_val) in enumerate(latest_theme_scores.head(20).items(), 1):
        theme_rank_rows.append(
            {
                "rank": rank,
                "theme": theme_name,
                "theme_score": float(score_val),
                "selected": theme_name in latest_themes,
            }
        )

    candidate_rows: list[dict[str, Any]] = []
    latest_score = stock_score.loc[latest_date].replace([np.inf, -np.inf], np.nan)
    selected_set = set(latest_selected)
    candidate_symbols = latest_score.dropna().sort_values(ascending=False).head(100).index
    for rank, sym in enumerate(candidate_symbols, 1):
        meta = mp2.loc[sym] if sym in mp2.index else {}
        candidate_rows.append(
            {
                "rank": rank,
                "symbol": sym,
                "name": meta.get("name", "") if hasattr(meta, "get") else "",
                "theme": groups.get(sym, ""),
                "selected_today": sym in selected_set,
                "stock_score": float(latest_score[sym]),
                "ret20_pct": pct(float(r20.at[latest_date, sym])) if sym in r20.columns else None,
                "ret60_pct": pct(float(r60.at[latest_date, sym])) if sym in r60.columns else None,
                "ret120_pct": pct(float(r120.at[latest_date, sym])) if sym in r120.columns else None,
                "amount_rank60": float(amount_rank60.at[latest_date, sym]) if sym in amount_rank60.columns and np.isfinite(amount_rank60.at[latest_date, sym]) else None,
                "amount_ratio20_60": float(amount_ratio20_60.at[latest_date, sym]) if sym in amount_ratio20_60.columns and np.isfinite(amount_ratio20_60.at[latest_date, sym]) else None,
            }
        )

    tw_state.update(
        {
            "latest_price_date": latest_date.strftime("%Y-%m-%d"),
            "regime_state": regime_state,
            "outer_mode": latest_mode,
            "target_position": target_rows,
        }
    )
    market_state = {
        "market": "TW",
        "latest_price_date": latest_date,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "frozen_strategy": str(switch_rule["candidate_id"]),
        "theme_candidate_id": str(theme_rule["candidate_id"]),
        "outer_mode": latest_mode,
        "theme_alloc": theme_alloc,
        "regime_state": regime_state,
        "regime_state_label": {0: "weak_cash", 1: "normal", 2: "strong"}.get(regime_state, str(regime_state)),
        "market_score": float(market_score.loc[latest_date]),
        "action_signal": "hold_target",
        "notes": "Cloud package uses compact tail data and frozen TW rules. It does not retrain.",
    }

    pd.DataFrame([market_state]).to_csv(TW_OUT / "latest_market_state.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(theme_rank_rows).to_csv(TW_OUT / "latest_theme_rank.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(candidate_rows).to_csv(TW_OUT / "latest_candidate_pool.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(target_rows).to_csv(TW_OUT / "latest_target_position.csv", index=False, encoding="utf-8-sig")
    action_report = {
        "market_state": market_state,
        "target_position": target_rows,
        "top_themes": theme_rank_rows[:10],
        "top_candidates": candidate_rows[:50],
    }
    write_json(TW_OUT / "latest_action_report.json", action_report)
    return action_report


def run_update() -> dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    state = read_json(STATE_PATH, {"version": 1})
    tw = run_tw_daily(state)
    us = run_us_daily(state)
    us_scan = run_us_scan_daily(us)
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "package": "CLOUD_DAILY_MARKET_POOL",
        "data_mode": "compact_tail",
        "TW": {
            "latest_price_date": tw["market_state"]["latest_price_date"],
            "outer_mode": tw["market_state"]["outer_mode"],
            "theme_alloc": tw["market_state"]["theme_alloc"],
            "regime_state": tw["market_state"]["regime_state"],
            "strategy_mode": tw["market_state"].get("strategy_mode"),
            "selected_sleeve": tw["market_state"].get("selected_sleeve"),
            "selected_component": tw["market_state"].get("selected_component"),
            "attack_edge": tw["market_state"].get("attack_edge"),
            "component_signal_date": tw["market_state"].get("component_signal_date"),
            "component_rebalance_date": tw["market_state"].get("component_rebalance_date"),
            "rebalance_step_trading_days": tw["market_state"].get("rebalance_step_trading_days"),
            "trading_days_since_rebalance": tw["market_state"].get("trading_days_since_rebalance"),
            "trading_days_since_signal": tw["market_state"].get("trading_days_since_signal"),
            "rebalance_days_remaining": tw["market_state"].get("rebalance_days_remaining"),
            "rebalance_due_today_by_counter": tw["market_state"].get("rebalance_due_today_by_counter"),
            "rebalance_due_next_session_by_counter": tw["market_state"].get("rebalance_due_next_session_by_counter"),
            "target_position": tw["target_position"],
        },
        "US": {
            "latest_price_date": us["market_state"]["latest_price_date"],
            "regime": us["market_state"]["regime"],
            "gross_cap": us["market_state"]["gross_cap"],
            "strategy_mode": us["market_state"].get("strategy_mode"),
            "execution_overlay": us["market_state"].get("execution_overlay"),
            "execution_top_n": us["market_state"].get("execution_top_n"),
            "selected_component": us["market_state"].get("selected_component"),
            "component_signal_date": us["market_state"].get("component_signal_date"),
            "dynamic_market_pool": us["market_state"].get("dynamic_market_pool"),
            "scan_universe_symbols": us["market_state"].get("scan_universe_symbols"),
            "dynamic_formal_gate_active": us["market_state"].get("dynamic_formal_gate_active"),
            "dynamic_formal_gate_reason": us["market_state"].get("dynamic_formal_gate_reason"),
            "dynamic_formal_weight": us["market_state"].get("dynamic_formal_weight"),
            "dynamic_formal_latest_price_date": us["market_state"].get("dynamic_formal_latest_price_date"),
            "dynamic_formal_pool_count": us["market_state"].get("dynamic_formal_pool_count"),
            "component_rebalance_date": us["market_state"].get("component_rebalance_date"),
            "rebalance_step_trading_days": us["market_state"].get("rebalance_step_trading_days"),
            "trading_days_since_rebalance": us["market_state"].get("trading_days_since_rebalance"),
            "rebalance_days_remaining": us["market_state"].get("rebalance_days_remaining"),
            "rebalance_due_today_by_counter": us["market_state"].get("rebalance_due_today_by_counter"),
            "rebalance_due_next_session_by_counter": us["market_state"]["rebalance_due_next_session_by_counter"],
            "target_position_if_rebalanced": us["target_position_if_rebalanced"],
        },
        "US_SCAN": {
            "latest_price_date": us_scan["market_state"]["latest_price_date"],
            "scan_universe_symbols": us_scan["market_state"]["scan_universe_symbols"],
            "top_stock": us_scan["market_state"]["top_stock"],
            "top_theme": us_scan["market_state"]["top_theme"],
            "outside_execution_count_in_top20": us_scan["market_state"]["outside_execution_count_in_top20"],
            "top_outside_execution_symbols": us_scan["alerts"]["top_outside_execution_symbols"][:10],
            "watch_only": False,
            "feeds_dynamic_market_strategy": us_scan["alerts"].get("feeds_dynamic_market_strategy"),
            "notes": us_scan["alerts"].get("notes"),
        },
        "files": {
            "summary": str(OUTPUT_DIR / "LATEST_DAILY_MARKET_POOL_SUMMARY.json"),
            "state": str(STATE_PATH),
        },
    }
    state["updated_at"] = summary["generated_at"]
    write_json(STATE_PATH, state)
    write_json(OUTPUT_DIR / "LATEST_DAILY_MARKET_POOL_SUMMARY.json", summary)
    return clean(summary)
