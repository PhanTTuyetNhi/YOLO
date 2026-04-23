import os
import threading
from datetime import datetime

from fastapi import FastAPI
from fastapi import Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware


# ====== STATE (API-only safe for Render) ======
people_count = 0
last_updated = ""
detections = []
state_lock = threading.Lock()


def set_state(count: int, dets: list[dict] | None = None) -> None:
    global people_count, last_updated, detections
    with state_lock:
        people_count = int(count)
        if dets is not None:
            detections = dets
        last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


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
    return {"status": "ok"}


@app.get("/people")
def get_people():
    with state_lock:
        return {
            "api_url": get_public_api_base_url(),
            "people_count": people_count,
            "time": last_updated,
        }


@app.get("/people/detail")
def get_people_detail():
    with state_lock:
        return {
            "api_url": get_public_api_base_url(),
            "people_count": people_count,
            "time": last_updated,
            "detections": detections,
        }


@app.post("/update")
def update_people(payload: dict, x_update_token: str | None = Header(default=None)):
    expected_token = os.getenv("UPDATE_TOKEN", "").strip()
    if expected_token and x_update_token != expected_token:
        raise HTTPException(status_code=401, detail="Invalid update token")

    count = int(payload.get("people_count", 0))
    dets = payload.get("detections", [])
    if not isinstance(dets, list):
        raise HTTPException(status_code=400, detail="detections must be a list")

    set_state(count, dets)
    return {"ok": True, "api_url": get_public_api_base_url(), "people_count": count}

