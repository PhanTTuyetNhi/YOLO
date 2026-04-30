import os
import threading
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import FastAPI
from fastapi import Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from settings import load_env_file


load_env_file()


APP_TIMEZONE = os.getenv("APP_TIMEZONE", "Asia/Bangkok").strip() or "Asia/Bangkok"
DEFAULT_TIMEZONE = timezone(timedelta(hours=7), name="Asia/Bangkok")


def get_app_timezone():
    try:
        return ZoneInfo(APP_TIMEZONE)
    except ZoneInfoNotFoundError:
        return DEFAULT_TIMEZONE

# ====== STATE (API-only safe for Render) ======
people_count = 0
last_updated = ""
detections = []
last_received_at = ""
last_received_at_ts = 0.0
last_source = ""
camera_enabled = False
state_lock = threading.Lock()
CAMERA_OFFLINE_AFTER_SECONDS = float(os.getenv("CAMERA_OFFLINE_AFTER_SECONDS", "10"))


def set_state(
    count: int,
    dets: list[dict] | None = None,
    *,
    source_time: str | None = None,
    source: str = "local-camera",
    enabled: bool = True,
) -> None:
    global people_count, last_updated, detections, last_received_at, last_received_at_ts, last_source, camera_enabled
    received_at = datetime.now(get_app_timezone()).strftime("%Y-%m-%d %H:%M:%S")
    with state_lock:
        people_count = int(count)
        if dets is not None:
            detections = dets
        last_updated = source_time or received_at
        last_received_at = received_at
        last_received_at_ts = time.time()
        last_source = source
        camera_enabled = enabled


def is_camera_connected() -> bool:
    return camera_enabled and last_received_at_ts > 0 and (time.time() - last_received_at_ts) <= CAMERA_OFFLINE_AFTER_SECONDS


def parse_bool(value, default: bool = True) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def get_response_time() -> str:
    return datetime.now(get_app_timezone()).strftime("%Y-%m-%d %H:%M:%S")


def get_public_api_base_url() -> str:
    return os.getenv("RENDER_API_URL", "").strip().rstrip("/")


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def disable_cache(request, call_next):
    response: Response = await call_next(request)
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.get("/")
def root():
    base_url = get_public_api_base_url()
    connected = is_camera_connected()
    return {
        "message": "People counter API is running",
        "api_url": base_url,
        "endpoints": {
            "people": f"{base_url}/people" if base_url else "/people",
            "people_detail": f"{base_url}/people/detail" if base_url else "/people/detail",
            "health": f"{base_url}/health" if base_url else "/health",
            "update": f"{base_url}/update" if base_url else "/update",
        },
        "camera_enabled": connected,
    }


@app.get("/health")
def health():
    with state_lock:
        connected = is_camera_connected()
        return {
            "status": "ok",
            "camera_enabled": connected,
            "last_updated": last_updated,
            "last_received_at": last_received_at,
            "source": last_source,
        }


@app.get("/people")
def get_people():
    with state_lock:
        if not is_camera_connected():
            return {
                "camera_enabled": False,
                "message": "Khong ket noi voi camera",
            }

        return {
            "camera_enabled": True,
            "people_count": people_count,
            "time": get_response_time(),
        }


@app.get("/people/detail")
def get_people_detail():
    with state_lock:
        connected = is_camera_connected()
        if not connected:
            return {
                "api_url": get_public_api_base_url(),
                "camera_enabled": False,
                "message": "Khong ket noi voi camera",
                "last_received_at": last_received_at,
                "source": last_source,
            }

        return {
            "api_url": get_public_api_base_url(),
            "camera_enabled": True,
            "people_count": people_count,
            "time": get_response_time(),
            "last_updated": last_updated,
            "last_received_at": last_received_at,
            "source": last_source,
            "detections": detections,
        }


@app.post("/update")
def update_people(payload: dict, x_update_token: str | None = Header(default=None)):
    expected_token = os.getenv("UPDATE_TOKEN", "").strip()
    if expected_token and x_update_token != expected_token:
        raise HTTPException(status_code=401, detail="Invalid update token")

    count = int(payload.get("people_count", 0))
    dets = payload.get("detections", [])
    source_time = payload.get("time")
    source = str(payload.get("source", "local-camera")).strip() or "local-camera"
    enabled = parse_bool(payload.get("camera_enabled"), default=True)
    if not isinstance(dets, list):
        raise HTTPException(status_code=400, detail="detections must be a list")

    set_state(count, dets, source_time=source_time, source=source, enabled=enabled)
    connected = is_camera_connected()
    return {
        "ok": True,
        "api_url": get_public_api_base_url(),
        "camera_enabled": connected,
        "people_count": count,
        "time": get_response_time(),
        "last_updated": last_updated,
        "last_received_at": last_received_at,
        "source": last_source,
    }

