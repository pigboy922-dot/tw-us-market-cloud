from __future__ import annotations

import os
import shutil
import threading
import traceback
import gzip
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse

import build_dashboard
import engine
from tools.update_live_prices import update_live_prices
from tools.refresh_target_current_prices import refresh_target_current_prices
from tools.export_excel_record import export_excel_record
from tools.reset_entry_baseline_to_latest_close import reset_entry_baseline_to_latest_close


APP_VERSION = "1.0.3-entry-baseline-reset"

app = FastAPI(title="Daily Market Pool Cloud", version=APP_VERSION)

CLOUD_UPDATE_JOB_PATH = engine.OUTPUT_DIR / "LATEST_CLOUD_UPDATE_JOB.json"
CLOUD_UPDATE_PROGRESS_PATH = engine.OUTPUT_DIR / "LATEST_CLOUD_UPDATE_PROGRESS.txt"
CLOUD_UPDATE_LOCK = threading.Lock()
CLOUD_UPDATE_THREAD: threading.Thread | None = None
CLOUD_UPDATE_JOB_ID = ""


def seed_runtime_dir(source: Path, target: Path) -> None:
    if not source.exists():
        return
    overwrite = os.getenv("SEED_PERSISTENT_OVERWRITE", "0").strip().lower() in {"1", "true", "yes"}
    if source.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        if overwrite or not target.exists():
            shutil.copy2(source, target)
        return
    target.mkdir(parents=True, exist_ok=True)
    for src in source.rglob("*"):
        rel = src.relative_to(source)
        dst = target / rel
        if src.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
        elif overwrite or not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def seed_runtime_gzip_csv(source_dir: Path, target_dir: Path) -> None:
    required = [
        "tw_strategy_prices_tail.csv",
        "us_scan_prices_tail.csv",
        "us_execution_prices_tail.csv",
    ]
    target_dir.mkdir(parents=True, exist_ok=True)
    overwrite = os.getenv("SEED_PERSISTENT_OVERWRITE", "0").strip().lower() in {"1", "true", "yes"}
    for name in required:
        target = target_dir / name
        gz_source = source_dir / f"{name}.gz"
        if not gz_source.exists():
            continue
        if not overwrite and target.exists() and target.stat().st_size > 1024:
            continue
        with gzip.open(gz_source, "rb") as src, target.open("wb") as dst:
            shutil.copyfileobj(src, dst)


@app.on_event("startup")
def seed_render_persistent_disk() -> None:
    if os.getenv("SEED_PERSISTENT_DATA", "1").strip().lower() in {"0", "false", "no"}:
        return
    seed_runtime_dir(engine.BASE_DIR / "data_live", engine.DATA_DIR)
    seed_runtime_gzip_csv(engine.BASE_DIR / "data_live", engine.DATA_DIR)
    seed_runtime_dir(engine.BASE_DIR / "runtime_outputs", engine.OUTPUT_DIR)
    state_dir = Path(os.getenv("STATE_DIR", engine.BASE_DIR / "state"))
    seed_runtime_dir(engine.BASE_DIR / "state", state_dir)


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def write_cloud_job(data: dict[str, Any]) -> None:
    engine.write_json(CLOUD_UPDATE_JOB_PATH, data)


def append_cloud_progress(message: str, reset: bool = False) -> None:
    engine.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if reset:
        CLOUD_UPDATE_PROGRESS_PATH.write_text("", encoding="utf-8-sig")
    line = f"[{now_iso()}] {message}"
    print(line, flush=True)
    with CLOUD_UPDATE_PROGRESS_PATH.open("a", encoding="utf-8-sig") as fh:
        fh.write(line + "\n")


def read_text_tail(path: Path, max_lines: int = 80) -> list[str]:
    if not path.exists():
        return []
    try:
        return path.read_text(encoding="utf-8-sig", errors="ignore").splitlines()[-max_lines:]
    except Exception:
        return []


def update_mode() -> str:
    mode = os.getenv("CLOUD_UPDATE_MODE", "full").strip().lower()
    aliases = {
        "fast": "target_only",
        "targets": "target_only",
        "target": "target_only",
        "safe": "daily_safe",
    }
    return aliases.get(mode, mode)


def ensure_dashboard() -> Path:
    path = build_dashboard.DASHBOARD_PATH
    if not path.exists():
        summary_path = engine.OUTPUT_DIR / "LATEST_DAILY_MARKET_POOL_SUMMARY.json"
        if not summary_path.exists():
            engine.run_update()
        build_dashboard.main()
        safe_export_excel_record()
    return path


def safe_export_excel_record() -> dict[str, Any]:
    try:
        return export_excel_record()
    except Exception as exc:
        report = {
            "ok": False,
            "status": "excel_record_export_failed",
            "error": str(exc),
            "traceback": traceback.format_exc(limit=8),
        }
        engine.write_json(engine.OUTPUT_DIR / "LATEST_EXCEL_RECORD_REPORT.json", report)
        return report


@app.get("/")
def index() -> FileResponse:
    return FileResponse(ensure_dashboard())


@app.get("/api/version")
def version() -> JSONResponse:
    required = [
        "tw_strategy_prices_tail.csv",
        "us_scan_prices_tail.csv",
        "us_execution_prices_tail.csv",
    ]
    return JSONResponse(
        {
            "ok": True,
            "app_version": APP_VERSION,
            "data_dir": str(engine.DATA_DIR),
            "seed_files": {
                name: {
                    "exists": (engine.DATA_DIR / name).exists(),
                    "bytes": (engine.DATA_DIR / name).stat().st_size if (engine.DATA_DIR / name).exists() else 0,
                    "gz_exists": (engine.BASE_DIR / "data_live" / f"{name}.gz").exists(),
                }
                for name in required
            },
        }
    )


@app.get("/api/status")
def status() -> JSONResponse:
    summary = engine.read_json(engine.OUTPUT_DIR / "LATEST_DAILY_MARKET_POOL_SUMMARY.json", {})
    state = engine.read_json(engine.STATE_PATH, {})
    price_update = engine.read_json(engine.OUTPUT_DIR / "LATEST_PRICE_UPDATE_REPORT.json", {})
    target_price_refresh = engine.read_json(engine.OUTPUT_DIR / "LATEST_TARGET_PRICE_REFRESH_REPORT.json", {})
    excel_record = engine.read_json(engine.OUTPUT_DIR / "LATEST_EXCEL_RECORD_REPORT.json", {})
    holdings = build_dashboard.read_current_holdings()
    prices = build_dashboard.read_current_prices()
    return JSONResponse(
        {
            "ok": True,
            "summary": summary,
            "state": state,
            "price_update": price_update,
            "target_price_refresh": target_price_refresh,
            "excel_record": excel_record,
            "holdings": holdings,
            "prices": prices,
        }
    )


@app.get("/api/holdings")
def holdings() -> JSONResponse:
    return JSONResponse({"ok": True, "holdings": build_dashboard.read_current_holdings()})


@app.get("/api/current-prices")
def current_prices() -> JSONResponse:
    return JSONResponse({"ok": True, "prices": build_dashboard.read_current_prices()})


@app.post("/api/current-prices")
async def save_current_prices(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
        rows = payload.get("prices", []) if isinstance(payload, dict) else []
        prices = build_dashboard.write_current_prices(rows)
        dashboard = build_dashboard.main()
        excel_record = safe_export_excel_record()
        return JSONResponse({"ok": True, "prices": prices, "dashboard": str(dashboard), "excel_record": excel_record})
    except Exception as exc:
        return JSONResponse(
            {"ok": False, "error": str(exc), "traceback": traceback.format_exc(limit=8)},
            status_code=500,
        )


@app.post("/api/reset-entry-baseline")
def reset_entry_baseline() -> JSONResponse:
    try:
        engine.run_update()
        report = reset_entry_baseline_to_latest_close()
        dashboard = build_dashboard.main()
        excel_record = safe_export_excel_record()
        return JSONResponse({"ok": True, "report": report, "dashboard": str(dashboard), "excel_record": excel_record})
    except Exception as exc:
        return JSONResponse(
            {"ok": False, "error": str(exc), "traceback": traceback.format_exc(limit=8)},
            status_code=500,
        )


@app.post("/api/refresh-target-prices")
def refresh_target_prices() -> JSONResponse:
    try:
        report = refresh_target_current_prices()
        dashboard = build_dashboard.main()
        excel_record = safe_export_excel_record()
        return JSONResponse(
            {"ok": bool(report.get("ok", True)), "report": report, "dashboard": str(dashboard), "excel_record": excel_record}
        )
    except Exception as exc:
        return JSONResponse(
            {"ok": False, "error": str(exc), "traceback": traceback.format_exc(limit=8)},
            status_code=500,
        )


@app.post("/api/holdings")
async def save_holdings(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
        rows = payload.get("holdings", []) if isinstance(payload, dict) else []
        holdings = build_dashboard.write_current_holdings(rows)
        dashboard = build_dashboard.main()
        excel_record = safe_export_excel_record()
        return JSONResponse({"ok": True, "holdings": holdings, "dashboard": str(dashboard), "excel_record": excel_record})
    except Exception as exc:
        return JSONResponse(
            {"ok": False, "error": str(exc), "traceback": traceback.format_exc(limit=8)},
            status_code=500,
        )


def run_update_pipeline() -> dict[str, Any]:
    price_update = {"ok": True, "status": "disabled"}
    target_price_refresh: dict[str, Any] = {"ok": True, "status": "not_started"}
    dynamic_us_scan: dict[str, Any] = {
        "ok": True,
        "status": "removed_from_render_package",
        "reason": "Render daily app keeps compact live data only and does not rebuild from the raw research pack.",
    }
    promotion_candidates: dict[str, Any] = {"status": "kept_from_seed_data"}
    formal_dynamic_pool: dict[str, Any] = {"status": "kept_from_seed_data"}
    try:
        mode = update_mode()
        if os.getenv("ENABLE_LIVE_PRICE_UPDATE", "1") != "0":
            if mode == "target_only":
                price_update = {
                    "ok": True,
                    "status": "skipped_full_market_update_target_only_mode",
                    "mode": mode,
                    "note": "Full TW/US market pools were not fetched. The update only refreshes current quotes for existing target/holding symbols after signal recalculation.",
                }
                engine.write_json(engine.OUTPUT_DIR / "LATEST_PRICE_UPDATE_REPORT.json", price_update)
            elif mode == "none":
                price_update = {
                    "ok": True,
                    "status": "skipped_by_cloud_update_mode_none",
                    "mode": mode,
                }
                engine.write_json(engine.OUTPUT_DIR / "LATEST_PRICE_UPDATE_REPORT.json", price_update)
            else:
                try:
                    price_update = update_live_prices(
                        markets={"us", "tw"},
                        chunk_size=env_int("CLOUD_UPDATE_CHUNK_SIZE", 40),
                        sleep_seconds=env_float("CLOUD_UPDATE_SLEEP_SECONDS", 0.5),
                        timeout=env_int("CLOUD_UPDATE_TIMEOUT", 15),
                        daily_safe=(mode == "daily_safe"),
                    )
                    price_update["mode"] = mode
                except Exception as exc:
                    price_update = {
                        "ok": False,
                        "status": "failed_before_signal_recalc",
                        "mode": mode,
                        "error": str(exc),
                        "traceback": traceback.format_exc(limit=8),
                    }
                    engine.write_json(engine.OUTPUT_DIR / "LATEST_PRICE_UPDATE_REPORT.json", price_update)
        summary = engine.run_update()
        if os.getenv("ENABLE_TARGET_PRICE_REFRESH_ON_UPDATE", "1") != "0":
            try:
                target_price_refresh = refresh_target_current_prices()
            except Exception as exc:
                target_price_refresh = {
                    "ok": False,
                    "status": "target_price_refresh_failed",
                    "error": str(exc),
                    "traceback": traceback.format_exc(limit=8),
                }
        dashboard = build_dashboard.main()
        excel_record = safe_export_excel_record()
        return {
            "ok": True,
            "dynamic_us_scan": dynamic_us_scan,
            "promotion_candidates": promotion_candidates,
            "formal_dynamic_pool": formal_dynamic_pool,
            "price_update": price_update,
            "target_price_refresh": target_price_refresh,
            "summary": summary,
            "dashboard": str(dashboard),
            "excel_record": excel_record,
            "mode": mode,
        }
    except Exception as exc:
        return {
            "ok": False,
            "dynamic_us_scan": dynamic_us_scan,
            "promotion_candidates": promotion_candidates,
            "formal_dynamic_pool": formal_dynamic_pool,
            "price_update": price_update,
            "target_price_refresh": target_price_refresh,
            "error": str(exc),
            "traceback": traceback.format_exc(limit=8),
        }


@app.post("/api/update")
def update() -> JSONResponse:
    result = run_update_pipeline()
    return JSONResponse(result, status_code=200 if result.get("ok") else 500)


def cloud_update_running() -> bool:
    return CLOUD_UPDATE_THREAD is not None and CLOUD_UPDATE_THREAD.is_alive()


def cloud_update_worker(job_id: str) -> None:
    started_at = now_iso()
    job = {
        "ok": True,
        "job_id": job_id,
        "status": "running",
        "message": "cloud update running",
        "started_at": started_at,
        "finished_at": None,
    }
    write_cloud_job(job)
    append_cloud_progress(f"job {job_id} started", reset=True)
    try:
        result = run_update_pipeline()
        status = "finished" if result.get("ok") else "failed"
        message = "cloud update finished" if result.get("ok") else str(result.get("error") or "cloud update failed")
        job.update(
            {
                "ok": bool(result.get("ok")),
                "status": status,
                "message": message,
                "finished_at": now_iso(),
                "result": result,
            }
        )
        append_cloud_progress(f"job {job_id} {status}: {message}")
    except Exception as exc:
        job.update(
            {
                "ok": False,
                "status": "failed",
                "message": str(exc),
                "finished_at": now_iso(),
                "error": str(exc),
                "traceback": traceback.format_exc(limit=8),
            }
        )
        append_cloud_progress(f"job {job_id} failed: {exc}")
    finally:
        write_cloud_job(job)


@app.post("/api/cloud-update/start")
def start_cloud_update() -> JSONResponse:
    global CLOUD_UPDATE_JOB_ID, CLOUD_UPDATE_THREAD
    with CLOUD_UPDATE_LOCK:
        if cloud_update_running():
            job = engine.read_json(CLOUD_UPDATE_JOB_PATH, {})
            return JSONResponse(
                {
                    "ok": True,
                    "status": "already_running",
                    "job": job,
                    "job_id": CLOUD_UPDATE_JOB_ID or job.get("job_id"),
                },
                status_code=202,
            )
        CLOUD_UPDATE_JOB_ID = uuid4().hex
        initial_job = {
            "ok": True,
            "job_id": CLOUD_UPDATE_JOB_ID,
            "status": "queued",
            "message": "cloud update queued",
            "started_at": now_iso(),
            "finished_at": None,
        }
        write_cloud_job(initial_job)
        append_cloud_progress(f"job {CLOUD_UPDATE_JOB_ID} queued", reset=True)
        CLOUD_UPDATE_THREAD = threading.Thread(
            target=cloud_update_worker,
            args=(CLOUD_UPDATE_JOB_ID,),
            name=f"cloud-update-{CLOUD_UPDATE_JOB_ID[:8]}",
            daemon=True,
        )
        CLOUD_UPDATE_THREAD.start()
        return JSONResponse({"ok": True, "status": "started", "job": initial_job, "job_id": CLOUD_UPDATE_JOB_ID}, status_code=202)


@app.get("/api/cloud-update/status")
def cloud_update_status() -> JSONResponse:
    job = engine.read_json(CLOUD_UPDATE_JOB_PATH, {})
    running = cloud_update_running()
    if running and job:
        job["status"] = "running"
    market_progress = engine.read_json(engine.OUTPUT_DIR / "LATEST_MARKET_UPDATE_PROGRESS.json", {})
    price_update = engine.read_json(engine.OUTPUT_DIR / "LATEST_PRICE_UPDATE_REPORT.json", {})
    target_price_refresh = engine.read_json(engine.OUTPUT_DIR / "LATEST_TARGET_PRICE_REFRESH_REPORT.json", {})
    return JSONResponse(
        {
            "ok": True,
            "running": running,
            "job": job,
            "price_update": price_update,
            "target_price_refresh": target_price_refresh,
            "cloud_progress_tail": read_text_tail(CLOUD_UPDATE_PROGRESS_PATH, 30),
            "market_progress": market_progress,
            "market_progress_tail": read_text_tail(engine.OUTPUT_DIR / "LATEST_MARKET_UPDATE_PROGRESS.txt", 40),
            "target_progress_tail": read_text_tail(engine.OUTPUT_DIR / "LATEST_TARGET_PRICE_REFRESH_PROGRESS.txt", 30),
        }
    )


@app.get("/files/{path:path}")
def files(path: str) -> FileResponse:
    root = engine.OUTPUT_DIR.resolve()
    target = (root / path).resolve()
    if root not in target.parents and target != root:
        return JSONResponse({"ok": False, "error": "path outside runtime_outputs"}, status_code=400)
    if not target.exists() or not target.is_file():
        return JSONResponse({"ok": False, "error": "file not found"}, status_code=404)
    return FileResponse(target)
