import logging
import os
import threading
import time
import cv2
from flask import Flask, jsonify, render_template, Response

from tracker import PersonTracker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

RTSP_URL = os.environ.get(
    "RTSP_URL",
    "rtsp://admin:L26DDDDF@10.40.90.225:554/cam/realmonitor?channel=1&subtype=1",
)

latest_frame: bytes | None = None
latest_dets: list[dict] = []
frame_idx: int = 0
linked_targets: dict[int, str] = {}
pending_targets: list[str] = []
lock = threading.Lock()

tracker_app: PersonTracker | None = None


def _zmq_listener():
    import zmq

    context = zmq.Context()
    pull = context.socket(zmq.PULL)
    pull.connect("tcp://127.0.0.1:5557")

    while True:
        try:
            msg = pull.recv_json()
            if msg["action"] == "track":
                name = msg["name"]
                with lock:
                    if name not in pending_targets:
                        pending_targets.append(name)
                        if tracker_app is not None:
                            tracker_app.pending_targets = pending_targets
                        logger.info(
                            "[FLASK] Target added via ZMQ: %s — queue size: %d",
                            name,
                            len(pending_targets),
                        )
                    else:
                        logger.info("[FLASK] Target skipped (already pending): %s", name)
        except Exception as e:
            logger.error("[FLASK] ZMQ error: %s", e)


def _track_worker():
    global latest_frame, latest_dets, frame_idx, linked_targets, pending_targets, tracker_app

    tracker_app = PersonTracker(
        model_path="yolo11s.pt",
        reid_weights="osnet_x1_0_msmt17.pth",
    )

    linked_targets = tracker_app._linked_targets
    pending_targets = tracker_app.pending_targets

    for frame, detections in tracker_app.track(RTSP_URL):
        ret, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ret:
            continue

        with lock:
            latest_frame = jpeg.tobytes()
            latest_dets = detections
            frame_idx = tracker_app.frame_idx
            linked_targets = tracker_app._linked_targets
            pending_targets = tracker_app.pending_targets


def _generate_frames():
    while True:
        with lock:
            frame_bytes = latest_frame

        if frame_bytes is not None:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
        else:
            time.sleep(0.05)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        _generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/status")
def status():
    with lock:
        return jsonify(
            frame_idx=frame_idx,
            linked_targets={str(gid): name for gid, name in linked_targets.items()},
            pending_targets=pending_targets,
            detection_count=len(latest_dets),
        )


if __name__ == "__main__":
    threading.Thread(target=_zmq_listener, daemon=True).start()

    threading.Thread(target=_track_worker, daemon=True).start()

    app.run(host="0.0.0.0", port=5150, threaded=True, debug=False)
