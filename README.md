# AI Drone Crowd Monitor — dashboard + backend

Two pieces:

| File | What it is |
|---|---|
| `server.py` | Python backend: Tello (or webcam) + YOLO11n + ByteTrack. Serves `/video` (MJPEG) and `/stats` (JSON). |
| `Drone Crowd Dashboard Live.dc.html` | The real dashboard — pulls live video and numbers from the server. |
| `Drone Crowd Dashboard.dc.html` | Demo version with simulated data (for presenting without the drone). |

## 1. Install

```bash
pip install -r requirements.txt
```

Keep `yolo11n.pt` in the same folder as `server.py` (copy it from `uploads/DroneCrowdMonitor-main/`).

## 2. Run the backend

Webcam (for testing indoors):

```bash
python server.py
```

Tello drone (connect to the Tello WiFi first):

```bash
python server.py --tello --max 10
```

`--max` is the room capacity used for the CROWDED threshold. HIGH is 70% of it.

You should see:

```
Dashboard API running:  http://localhost:8000/stats
Live video:             http://localhost:8000/video
```

## 3. Open the dashboard

Open `Drone Crowd Dashboard Live.dc.html` in a browser. It polls `/stats` every 500 ms and
shows `/video` in the camera panel.

If the backend runs on another machine, change `serverUrl` in the dashboard's Tweaks panel
(e.g. `http://192.168.1.20:8000`).

## What the dashboard shows

- Live annotated camera feed (YOLO boxes drawn server-side)
- People now / capacity, peak people, average confidence, battery
- Occupancy ring + room status: NORMAL → HIGH → CROWDED (card color changes)
- Crowd density chart for the last 60 samples, with the capacity threshold line
- Mission info (state, model, obstacle, monitoring time)
- Event log streamed from the backend (entries, exits, crowd alerts)
- Reconnect stream / reset counters buttons (reset also calls `POST /reset`)

## Endpoints

| Method | Path | Returns |
|---|---|---|
| GET | `/video` | `multipart/x-mixed-replace` MJPEG stream |
| GET | `/stats` | `{people, peak, occupancy, confidence, status, max_people, battery, altitude, fps, elapsed, source, connected, logs[]}` |
| POST | `/reset` | resets peak, timer and log |

## Notes

- `server.py` never commands takeoff. It only connects, checks battery (min 40%), starts the
  video stream and runs detection. Flight control stays in `main.py`.
- If the Tello fails to connect, the server falls back to the webcam automatically so the
  dashboard still has something to show during a demo.
