"""
AI Drone Crowd Monitor — backend server
=======================================
Runs YOLO11 + ByteTrack on the Tello stream (or a webcam) and exposes:

    GET /video   -> MJPEG live stream (annotated frames)
    GET /stats   -> JSON telemetry consumed by the dashboard
    POST /reset  -> reset counters

Run:
    pip install -r requirements.txt
    python server.py            # webcam  (SOURCE = "webcam")
    python server.py --tello    # Tello drone

Then open "Drone Crowd Dashboard Live.dc.html" in a browser.
"""

import argparse
import threading
import time

import cv2
from flask import Flask, Response, jsonify
from flask_cors import CORS
from ultralytics import YOLO

# ==========================================
# SETTINGS
# ==========================================

MAX_PEOPLE = 10               # capacity of the room
HIGH_RATIO = 0.7              # >= 70% of capacity -> HIGH
MODEL_PATH = "yolo11n.pt"
CONFIDENCE_THRESHOLD = 0.40
TELLO_VIDEO_URL = "udp://0.0.0.0:11111"
MIN_BATTERY = 40
PORT = 8000

# ==========================================
# SHARED STATE (written by worker, read by HTTP)
# ==========================================

state = {
    "people": 0,
    "peak": 0,
    "occupancy": 0,
    "confidence": 0.0,
    "status": "NORMAL",           # NORMAL | HIGH | CROWDED
    "max_people": MAX_PEOPLE,
    "battery": None,
    "altitude": None,
    "fps": 0.0,
    "elapsed": 0,
    "source": "webcam",
    "connected": False,
    "logs": [],                   # [{t, msg, kind}]
}

state_lock = threading.Lock()
latest_frame = [None]             # JPEG bytes
frame_lock = threading.Lock()
start_time = time.time()


def log(msg, kind="info"):
    entry = {"t": time.strftime("%H:%M:%S"), "msg": msg, "kind": kind}
    with state_lock:
        state["logs"].insert(0, entry)
        del state["logs"][12:]
    print(f"[{entry['t']}] {msg}")


# ==========================================
# TELLO
# ==========================================

def connect_tello():
    """Connect, run the pre-flight check, start the stream. Returns tello or None."""
    from djitellopy import Tello

    tello = Tello()
    print("Connecting to Tello...")
    tello.connect()

    battery = tello.get_battery()
    print(f"Battery: {battery}%")
    with state_lock:
        state["battery"] = battery

    if battery < MIN_BATTERY:
        log(f"Battery too low ({battery}%) — minimum is {MIN_BATTERY}%", "alert")
        return None

    tello.streamon()
    time.sleep(2)
    log("Tello connected — video stream started", "ok")
    return tello


def telemetry_loop(tello):
    """Poll battery / height every 2 s."""
    while True:
        try:
            with state_lock:
                state["battery"] = tello.get_battery()
                state["altitude"] = round(tello.get_height() / 100.0, 1)
        except Exception:
            pass
        time.sleep(2)


# ==========================================
# DETECTION WORKER
# ==========================================

def detection_loop(use_tello: bool):
    model = YOLO(MODEL_PATH)
    tello = None

    if use_tello:
        try:
            tello = connect_tello()
        except Exception as e:
            log(f"Tello connection failed: {e}", "alert")
            tello = None

        if tello is None:
            log("Falling back to webcam", "info")
            use_tello = False
        else:
            threading.Thread(target=telemetry_loop, args=(tello,), daemon=True).start()

    cap = (cv2.VideoCapture(TELLO_VIDEO_URL, cv2.CAP_FFMPEG)
           if use_tello else cv2.VideoCapture(0))
    time.sleep(2)

    if not cap.isOpened():
        log("Could not open the video source", "alert")
        return

    with state_lock:
        state["source"] = "tello" if use_tello else "webcam"
        state["connected"] = True
    log(f"Detection started on {'Tello' if use_tello else 'webcam'}", "ok")

    prev_people = 0
    last_t = time.time()

    while True:
        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            time.sleep(0.03)
            continue

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[0],                       # person
            conf=CONFIDENCE_THRESHOLD,
            verbose=False,
        )
        result = results[0]

        people = 0
        confidence = 0.0
        if result.boxes is not None and result.boxes.id is not None:
            people = len(result.boxes.id)
            confs = result.boxes.conf.tolist()
            if confs:
                confidence = sum(confs) / len(confs) * 100

        if people >= MAX_PEOPLE:
            status = "CROWDED"
        elif people >= max(1, round(MAX_PEOPLE * HIGH_RATIO)):
            status = "HIGH"
        else:
            status = "NORMAL"

        now = time.time()
        fps = 1.0 / max(now - last_t, 1e-6)
        last_t = now

        with state_lock:
            prev_status = state["status"]
            state["people"] = people
            state["peak"] = max(state["peak"], people)
            state["occupancy"] = min(100, round(people / MAX_PEOPLE * 100))
            state["confidence"] = round(confidence, 1)
            state["status"] = status
            state["fps"] = round(fps, 1)
            state["elapsed"] = int(now - start_time)

        if people != prev_people:
            log(f"{'Person entered' if people > prev_people else 'Person left'} · total {people}")
            prev_people = people
        if status == "CROWDED" and prev_status != "CROWDED":
            log("Capacity exceeded — crowd alert", "alert")
        if status == "NORMAL" and prev_status != "NORMAL":
            log("Crowd level back to normal", "ok")

        annotated = result.plot()
        ok, buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ok:
            with frame_lock:
                latest_frame[0] = buf.tobytes()

    cap.release()


# ==========================================
# HTTP API
# ==========================================

app = Flask(__name__)
CORS(app)


@app.route("/stats")
def stats():
    with state_lock:
        return jsonify(dict(state))


@app.route("/video")
def video():
    def gen():
        while True:
            with frame_lock:
                frame = latest_frame[0]
            if frame is None:
                time.sleep(0.05)
                continue
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            time.sleep(0.03)

    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/reset", methods=["POST", "GET"])
def reset():
    global start_time
    start_time = time.time()
    with state_lock:
        state["peak"] = 0
        state["logs"] = []
    log("Counters reset", "info")
    return jsonify({"ok": True})


# ==========================================
# ENTRY POINT
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tello", action="store_true", help="use the Tello drone camera")
    parser.add_argument("--max", type=int, default=MAX_PEOPLE, help="room capacity")
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()

    MAX_PEOPLE = args.max
    state["max_people"] = MAX_PEOPLE

    threading.Thread(target=detection_loop, args=(args.tello,), daemon=True).start()

    print(f"\nDashboard API running:  http://localhost:{args.port}/stats")
    print(f"Live video:             http://localhost:{args.port}/video\n")
    app.run(host="0.0.0.0", port=args.port, threaded=True)
