import os
import threading
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
last_source = ""
state_lock = threading.Lock()


def set_state(
    count: int,
    dets: list[dict] | None = None,
    *,
    source_time: str | None = None,
    source: str = "local-camera",
) -> None:
    global people_count, last_updated, detections, last_received_at, last_source
    received_at = datetime.now(get_app_timezone()).strftime("%Y-%m-%d %H:%M:%S")
    with state_lock:
        people_count = int(count)
        if dets is not None:
            detections = dets
        last_updated = source_time or received_at
        last_received_at = received_at
        last_source = source


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
    return {
        "message": "People counter API is running",
        "api_url": base_url,
        "endpoints": {
            "people": f"{base_url}/people" if base_url else "/people",
            "people_detail": f"{base_url}/people/detail" if base_url else "/people/detail",
            "health": f"{base_url}/health" if base_url else "/health",
            "update": f"{base_url}/update" if base_url else "/update",
        },
        "camera_enabled": os.getenv("ENABLE_CAMERA", "0") == "1",
    }


@app.get("/health")
def health():
    with state_lock:
        return {
            "status": "ok",
            "time": get_response_time(),
            "people_count": people_count,
            "last_updated": last_updated,
            "last_received_at": last_received_at,
            "source": last_source,
        }


@app.get("/people")
def get_people():
    with state_lock:
        return {
            "api_url": get_public_api_base_url(),
            "people_count": people_count,
            "time": get_response_time(),
            "last_updated": last_updated,
            "last_received_at": last_received_at,
            "source": last_source,
        }


@app.get("/people/detail")
def get_people_detail():
    with state_lock:
        return {
            "api_url": get_public_api_base_url(),
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
    if not isinstance(dets, list):
        raise HTTPException(status_code=400, detail="detections must be a list")

    set_state(count, dets, source_time=source_time, source=source)
    return {
        "ok": True,
        "api_url": get_public_api_base_url(),
        "people_count": count,
        "time": get_response_time(),
        "last_updated": last_updated,
        "last_received_at": last_received_at,
        "source": last_source,
    }

