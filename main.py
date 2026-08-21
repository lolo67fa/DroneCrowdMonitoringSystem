from djitellopy import Tello
import cv2
import os
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser

from ultralytics import YOLO


# ==========================================
# SETTINGS
# ==========================================

MIN_BATTERY = 40
MAX_PEOPLE = 10

# Tello video stream
VIDEO_URL = "udp://0.0.0.0:11111"
MODEL_PATH = "yolo11n.pt"
CONFIDENCE_THRESHOLD = 0.40

# Dashboard launch (see launch_dashboard() below)
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
SERVER_SCRIPT = os.path.join(PROJECT_DIR, "server.py")
DASHBOARD_HTML = os.path.join(PROJECT_DIR, "Drone Crowd Dashboard Live.dc.html")
SERVER_PORT = 8000

# Key codes
KEY_SPACE = 32
KEY_L = ord("l")
KEY_Q = ord("q")
KEY_F = ord("f")


# ==========================================
# QUICK PRE-TAKEOFF CHECK (runs on 'f')
# ==========================================
# Full startup checks (connection, battery, camera) already ran once in
# main(). This one only re-checks what can change between startup and the
# moment you actually press 'f' to fly: is the link still alive, and is
# there still enough battery.

def preflight_quick_check(tello):

    try:
        battery = tello.get_battery()
    except Exception as e:
        return False, f"Connection check failed: {e}"

    if battery < MIN_BATTERY:
        return False, f"Battery too low ({battery}% < {MIN_BATTERY}% required)"

    return True, f"Preflight OK — battery {battery}%"


# ==========================================
# OPEN TELLO CAMERA WITH OPENCV
# ==========================================

def open_tello_camera():

    print("\n📷 Opening Tello video with OpenCV...")

    cap = cv2.VideoCapture(VIDEO_URL, cv2.CAP_FFMPEG)

    time.sleep(2)

    if not cap.isOpened():
        print("❌ OpenCV could not open Tello stream")
        return None

    print("OpenCV stream: OK ✅")

    return cap


# ==========================================
# READ FRAME
# ==========================================

def read_frame(cap):

    ret, frame = cap.read()

    if not ret or frame is None or frame.size == 0:
        return False, None

    return True, frame


# ==========================================
# SHOW CAMERA
# ==========================================

def show_camera(cap, text="READY", model=None):

    ret, frame = read_frame(cap)

    if not ret:
        return False, -1

    people = 0

    if model is not None:

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[0],       # person
            conf=CONFIDENCE_THRESHOLD,
            verbose=False
        )

        result = results[0]

        if result.boxes is not None and result.boxes.id is not None:
            people = len(result.boxes.id)

        frame = result.plot()

        if people >= MAX_PEOPLE:
            status, status_color = "CROWDED", (0, 0, 255)
        elif people >= round(MAX_PEOPLE * 0.7):
            status, status_color = "HIGH", (0, 165, 255)
        else:
            status, status_color = "NORMAL", (0, 255, 0)

        cv2.putText(frame, f"People: {people}/{MAX_PEOPLE}", (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Status: {status}", (20, 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

        if people >= MAX_PEOPLE:
            cv2.putText(frame, "!!! CROWD ALERT !!!", (20, 190),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.putText(frame, text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, "F=Takeoff+Dashboard  L=Land  SPACE=Emergency  Q=Quit", (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

    cv2.imshow("Tello Camera", frame)

    key = cv2.waitKey(1) & 0xFF

    return True, key


# ==========================================
# DASHBOARD LAUNCH  (this is the 'f'-key part you asked to see)
# ==========================================

def is_server_running():
    try:
        urllib.request.urlopen(f"http://localhost:{SERVER_PORT}/stats", timeout=0.5)
        return True
    except Exception:
        return False


def launch_dashboard():
    """Start server.py (if not already up) and open the dashboard HTML.

    Runs on its own thread so it can never stall the video/YOLO loop.
    NOTE: server.py is started WITHOUT --tello on purpose — main.py already
    owns the one live connection to the drone (command port 8889 + video
    port 11111). If server.py also opened a second Tello connection at the
    same time, the two would fight over those ports. server.py falls back
    to the PC webcam in that mode; if you want the dashboard fed by the
    drone instead, run `python server.py --tello` by itself, without
    flying from main.py at the same time.
    """
    try:
        if is_server_running():
            print("🌐 server.py already running — reusing it")
        else:
            print("🌐 Starting server.py ...")
            creationflags = subprocess.CREATE_NEW_CONSOLE if os.name == "nt" else 0
            subprocess.Popen(
                [sys.executable, SERVER_SCRIPT],
                cwd=PROJECT_DIR,
                creationflags=creationflags,
            )
            # Give Flask a moment to bind before the browser hits it.
            for _ in range(20):
                if is_server_running():
                    break
                time.sleep(0.25)

        print(f"🌐 Opening dashboard: {DASHBOARD_HTML}")
        webbrowser.open(f"file:///{DASHBOARD_HTML.replace(os.sep, '/')}")

    except Exception as e:
        print(f"⚠️ Could not launch dashboard: {e}")


# ==========================================
# FLIGHT COMMANDS (run on background threads so YOLO/video never blocks them)
# ==========================================

flight_lock = threading.Lock()   # serializes takeoff/land against each other


def do_takeoff(tello, flight_state):
    try:
        with flight_lock:
            print("\n🚁 Taking off...")
            tello.takeoff()
        flight_state["airborne"] = True
        print("Takeoff successful ✅")
    except Exception as e:
        print(f"❌ Takeoff failed: {e}")
    finally:
        flight_state["takeoff_pending"] = False


def do_land(tello, flight_state):
    try:
        with flight_lock:
            print("\n🛬 Landing...")
            tello.land()
        flight_state["airborne"] = False
        print("Landing successful ✅")
    except Exception as e:
        print(f"❌ Landing failed: {e}")


def do_emergency(tello, flight_state):
    # Deliberately does NOT wait on flight_lock: emergency must cut power
    # immediately even if a takeoff/land command is still in flight.
    try:
        print("\n🚨 EMERGENCY STOP!")
        tello.emergency()
        flight_state["airborne"] = False
    except Exception as e:
        print(f"❌ Emergency command failed: {e}")


# ==========================================
# MAIN
# ==========================================

def main():

    tello = Tello()
    model = YOLO(MODEL_PATH)

    cap = None

    flight_state = {
        "airborne": False,
        "takeoff_pending": False,
        "dashboard_launched": False,
    }

    try:
        # ======================================
        # 1. CONNECT  (fail gracefully, don't crash)
        # ======================================

        print("\n" + "=" * 55)
        print("              CONNECTING TO TELLO")
        print("=" * 55)

        try:
            tello.connect()
            battery = tello.get_battery()
            print(f"Connected ✅  Battery: {battery}%")
        except Exception as e:
            print("❌ Could not connect to the Tello.")
            print(f"   Reason: {e}")
            print("   Check Wi-Fi is joined to the drone's TELLO-XXXXXX network and retry.")
            return

        try:
            tello.streamon()
            time.sleep(2)
        except Exception as e:
            print(f"❌ Could not start video stream: {e}")
            return

        # ======================================
        # 2. OPEN CAMERA PREVIEW (no takeoff yet)
        # ======================================

        cap = open_tello_camera()

        if cap is None:
            print("❌ Camera preview failed to open.")
            return

        print("\n" + "=" * 55)
        print("   DRONE CONNECTED — camera live, NOT flying yet")
        print("=" * 55)
        print("\nPress F to run preflight + takeoff + open dashboard")
        print("Press L to land, SPACE for emergency stop, Q to quit\n")

        # ======================================
        # 3. MAIN LOOP — checked every frame
        # ======================================

        while True:

            label = "AIRBORNE" if flight_state["airborne"] else "READY - Press F to Takeoff"
            success, key = show_camera(cap, label, model)

            if not success:
                print("⚠️ Failed to read video frame")
                time.sleep(0.05)
                continue

            # ----------------------------------
            # SPACE = EMERGENCY STOP (highest priority)
            # ----------------------------------
            if key == KEY_SPACE:
                threading.Thread(
                    target=do_emergency, args=(tello, flight_state), daemon=True
                ).start()

            # ----------------------------------
            # F = PREFLIGHT -> TAKEOFF -> DASHBOARD
            # ----------------------------------
            elif key == KEY_F:
                if flight_state["airborne"] or flight_state["takeoff_pending"]:
                    pass  # already flying or already taking off — ignore repeat presses
                else:
                    flight_state["takeoff_pending"] = True
                    ok, reason = preflight_quick_check(tello)

                    if not ok:
                        print(f"\n❌ Takeoff blocked: {reason}")
                        flight_state["takeoff_pending"] = False
                    else:
                        print(f"\n✅ {reason}")
                        threading.Thread(
                            target=do_takeoff, args=(tello, flight_state), daemon=True
                        ).start()

                        # Launch dashboard immediately alongside takeoff.
                        # Flag is set here (main thread, before the takeoff
                        # thread finishes) so a double 'f' press can't race
                        # past this check and launch it twice.
                        if not flight_state["dashboard_launched"]:
                            flight_state["dashboard_launched"] = True
                            threading.Thread(
                                target=launch_dashboard, daemon=True
                            ).start()

            # ----------------------------------
            # L = LAND
            # ----------------------------------
            elif key == KEY_L:
                if flight_state["airborne"]:
                    threading.Thread(
                        target=do_land, args=(tello, flight_state), daemon=True
                    ).start()

            # ----------------------------------
            # Q = QUIT (land first if flying)
            # ----------------------------------
            elif key == KEY_Q:
                print("\nQuit requested.")
                if flight_state["airborne"]:
                    try:
                        with flight_lock:
                            print("🛬 Landing before quitting...")
                            tello.land()
                        flight_state["airborne"] = False
                        print("Landing successful ✅")
                    except Exception as e:
                        print(f"⚠️ Landing on quit failed: {e}")
                break

        print("\n" + "=" * 55)
        print("             SESSION ENDED ✅")
        print("=" * 55)

    except Exception as e:
        print("\n" + "=" * 55)
        print("               SYSTEM ERROR ❌")
        print("=" * 55)
        print("Error:", e)

        if flight_state.get("airborne"):
            try:
                print("\n⚠️ Attempting emergency landing...")
                tello.land()
                print("Emergency landing command sent.")
            except Exception as landing_error:
                print("Emergency landing failed:", landing_error)

    finally:
        print("\nCleaning up...")

        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        try:
            tello.streamoff()
        except Exception:
            pass

        try:
            tello.end()
        except Exception:
            pass

        print("Tello connection closed.")


# ==========================================
# PROGRAM ENTRY POINT
# ==========================================

if __name__ == "__main__":
    main()
