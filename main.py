from datetime import datetime
import json
import os
import socket
import threading
import time
from urllib import error, request

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn


FLIP_CAMERA = True
ENABLE_CAMERA = os.getenv("ENABLE_CAMERA", "1") == "1"
MIN_TRACK_HISTORY = int(os.getenv("MIN_TRACK_HISTORY", "2"))

people_count = 0
last_updated = ""
detections = []
state_lock = threading.Lock()
last_sync_at = 0.0

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_public_api_base_url():
    render_api_url = os.getenv("RENDER_API_URL", "").strip().rstrip("/")
    if render_api_url and "ten-app-cua-ban.onrender.com" not in render_api_url:
        return render_api_url
    return "http://127.0.0.1:8000"


def sync_state_to_render(count, current_detections):
    global last_sync_at

    render_api_url = os.getenv("RENDER_API_URL", "").strip().rstrip("/")
    if not render_api_url:
        return

    sync_interval = float(os.getenv("RENDER_SYNC_INTERVAL", "1.0"))
    now = time.time()
    if now - last_sync_at < sync_interval:
        return

    payload = json.dumps(
        {
            "people_count": int(count),
            "detections": current_detections,
        }
    ).encode("utf-8")

    headers = {"Content-Type": "application/json"}
    update_token = os.getenv("UPDATE_TOKEN", "").strip()
    if update_token:
        headers["X-Update-Token"] = update_token

    update_url = f"{render_api_url}/update"
    req = request.Request(update_url, data=payload, headers=headers, method="POST")

    try:
        with request.urlopen(req, timeout=3) as response:
            if 200 <= response.status < 300:
                last_sync_at = now
    except (error.URLError, TimeoutError, OSError) as exc:
        print(f"[Render Sync] Failed to sync to {update_url}: {exc}")


@app.get("/")
def root():
    base_url = get_public_api_base_url()
    return {
        "message": "People counter API is running",
        "api_url": base_url,
        "camera_enabled": ENABLE_CAMERA,
        "endpoints": {
            "people": f"{base_url}/people",
            "people_detail": f"{base_url}/people/detail",
            "health": f"{base_url}/health",
        },
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/people")
def get_people():
    base_url = get_public_api_base_url()
    with state_lock:
        return {
            "api_url": base_url,
            "people_count": people_count,
            "time": last_updated,
        }


@app.get("/people/detail")
def get_people_detail():
    base_url = get_public_api_base_url()
    with state_lock:
        return {
            "api_url": base_url,
            "people_count": people_count,
            "time": last_updated,
            "detections": detections,
        }


prev_frame = None
track_history = {}


def has_motion(cv2, np, previous_frame, current_frame, box, threshold=1500):
    if previous_frame is None:
        return True

    x1, y1, x2, y2 = box
    prev_crop = previous_frame[y1:y2, x1:x2]
    curr_crop = current_frame[y1:y2, x1:x2]

    if prev_crop.shape != curr_crop.shape or prev_crop.size == 0:
        return True

    diff = cv2.absdiff(prev_crop, curr_crop)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
    return np.sum(thresh) > threshold


def is_mirror_pair(box1, box2, frame_width, tolerance=80):
    x1 = (box1[0] + box1[2]) // 2
    x2 = (box2[0] + box2[2]) // 2
    return abs((x1 + x2) - frame_width) < tolerance


def run_camera():
    global people_count, last_updated, prev_frame, track_history, detections

    try:
        import cv2
        import numpy as np
        from ultralytics import YOLO
    except ImportError as exc:
        print(f"[Camera] Camera mode disabled because dependencies are missing: {exc}")
        return

    model = YOLO("yolov8s.pt")

    def open_camera():
        candidates = [
            (0, cv2.CAP_DSHOW),
            (0, cv2.CAP_MSMF),
            (1, cv2.CAP_DSHOW),
            (1, cv2.CAP_MSMF),
            (0, cv2.CAP_ANY),
            (1, cv2.CAP_ANY),
            (2, cv2.CAP_ANY),
        ]

        for index, backend in candidates:
            cap = cv2.VideoCapture(index, backend)
            if not cap.isOpened():
                cap.release()
                continue

            ret, _ = cap.read()
            if ret:
                print(f"[Camera] Connected with index={index}, backend={backend}")
                return cap

            cap.release()

        return None

    cap = open_camera()
    if cap is None:
        print("[Camera] Cannot open any available camera.")
        with state_lock:
            last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Camera] Failed to read frame.")
            break

        if FLIP_CAMERA:
            frame = cv2.flip(frame, 1)

        _, w, _ = frame.shape
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        results = model.track(frame, persist=True, conf=0.4, iou=0.5, verbose=False)
        current_boxes = []

        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                if cls != 0 or box.conf[0] <= 0.4:
                    continue

                track_id = int(box.id[0]) if box.id is not None else -1
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                if x1 > w * 0.85:
                    continue

                if box.id is not None and not has_motion(cv2, np, prev_frame, frame, (x1, y1, x2, y2)):
                    continue

                if box.id is not None:
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    track_history.setdefault(track_id, []).append(center)
                    if len(track_history[track_id]) > 10:
                        track_history[track_id].pop(0)
                    if len(track_history[track_id]) < MIN_TRACK_HISTORY:
                        continue

                current_boxes.append((track_id, x1, y1, x2, y2))

        if not current_boxes:
            detect_results = model(frame, conf=0.4, verbose=False)
            fallback_id = 1
            for result in detect_results:
                for box in result.boxes:
                    cls = int(box.cls[0])
                    if cls != 0 or box.conf[0] <= 0.4:
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    if x1 > w * 0.85:
                        continue

                    current_boxes.append((fallback_id, x1, y1, x2, y2))
                    fallback_id += 1

        removed_ids = set()
        for i in range(len(current_boxes)):
            for j in range(i + 1, len(current_boxes)):
                _, x1a, y1a, x2a, y2a = current_boxes[i]
                id2, x1b, y1b, x2b, y2b = current_boxes[j]
                if is_mirror_pair((x1a, y1a, x2a, y2a), (x1b, y1b, x2b, y2b), w):
                    removed_ids.add(id2)

        count = 0
        current_detections = []

        for track_id, x1, y1, x2, y2 in current_boxes:
            if track_id in removed_ids:
                continue

            count += 1
            current_detections.append({"id": track_id, "bbox": [x1, y1, x2, y2]})

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"ID {track_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2,
            )

        with state_lock:
            people_count = count
            detections = current_detections
            last_updated = current_time

        sync_state_to_render(count, current_detections)

        cv2.putText(
            frame,
            f"People: {count}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )
        cv2.imshow("Camera", frame)

        prev_frame = frame.copy()
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


def get_local_ip():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        sock.close()


if __name__ == "__main__":
    if ENABLE_CAMERA:
        t = threading.Thread(target=run_camera, daemon=True)
        t.start()
    else:
        print("[Camera] ENABLE_CAMERA=0, starting API without camera thread.")

    local_ip = get_local_ip()
    public_api_url = get_public_api_base_url()
    print(f"API public: {public_api_url}/people")
    print(f"API local: http://127.0.0.1:8000/people")
    print(f"API LAN:   http://{local_ip}:8000/people")

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
