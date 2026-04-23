import threading
from datetime import datetime


def start_camera_thread(
    *,
    flip_camera: bool = True,
    mirror_roi_right_ratio: float = 0.85,
    motion_threshold: int = 1500,
) -> threading.Thread:
    """
    Starts the camera/YOLO loop in a daemon thread.
    Heavy imports (cv2/torch/ultralytics) live inside the thread so Render can import the API safely.
    """

    def _worker():
        import cv2
        import numpy as np
        from ultralytics import YOLO

        from app import set_state

        model = YOLO("yolov8s.pt")

        prev_frame = None
        track_history: dict[int, list[tuple[int, int]]] = {}

        def has_motion(prev_f, curr_f, box, threshold=motion_threshold):
            if prev_f is None:
                return True
            x1, y1, x2, y2 = box
            prev_crop = prev_f[y1:y2, x1:x2]
            curr_crop = curr_f[y1:y2, x1:x2]
            if prev_crop.shape != curr_crop.shape or prev_crop.size == 0:
                return True
            diff = cv2.absdiff(prev_crop, curr_crop)
            gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
            return int(np.sum(thresh)) > int(threshold)

        def is_mirror_pair(box1, box2, frame_width, tolerance=80):
            x1 = (box1[0] + box1[2]) // 2
            x2 = (box2[0] + box2[2]) // 2
            return abs((x1 + x2) - frame_width) < tolerance

        def open_camera():
            candidates = [
                (0, cv2.CAP_DSHOW),
                (0, cv2.CAP_MSMF),
                (1, cv2.CAP_DSHOW),
                (1, cv2.CAP_MSMF),
                (0, cv2.CAP_ANY),
                (1, cv2.CAP_ANY),
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
            print("[Camera] Cannot connect to camera.")
            print("[Camera] Check camera permission, close apps using camera, and try different index.")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[Camera] Lost frame from camera. Stopping camera thread.")
                break

            if flip_camera:
                frame = cv2.flip(frame, 1)

            h, w, _ = frame.shape

            results = model.track(frame, persist=True, conf=0.5, iou=0.5)
            current_boxes: list[tuple[int, int, int, int, int]] = []

            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    if cls == 0 and box.conf[0] > 0.5 and box.id is not None:
                        track_id = int(box.id[0])
                        x1, y1, x2, y2 = map(int, box.xyxy[0])

                        if x1 > w * mirror_roi_right_ratio:
                            continue

                        if not has_motion(prev_frame, frame, (x1, y1, x2, y2)):
                            continue

                        center = ((x1 + x2) // 2, (y1 + y2) // 2)
                        track_history.setdefault(track_id, []).append(center)
                        if len(track_history[track_id]) > 10:
                            track_history[track_id].pop(0)
                        if len(track_history[track_id]) < 5:
                            continue

                        current_boxes.append((track_id, x1, y1, x2, y2))

            removed_ids: set[int] = set()
            for i in range(len(current_boxes)):
                for j in range(i + 1, len(current_boxes)):
                    id1, x1a, y1a, x2a, y2a = current_boxes[i]
                    id2, x1b, y1b, x2b, y2b = current_boxes[j]
                    if is_mirror_pair((x1a, y1a, x2a, y2a), (x1b, y1b, x2b, y2b), w):
                        removed_ids.add(id2)

            count = 0
            current_detections: list[dict] = []
            for (track_id, x1, y1, x2, y2) in current_boxes:
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

            set_state(count, current_detections)

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

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    return t

