from djitellopy import Tello
from ultralytics import YOLO
import cv2
import time


# ==========================================
# SETTINGS
# ==========================================

MODEL_PATH = "yolo11n.pt"
CONFIDENCE_THRESHOLD = 0.40

VIDEO_URL = "udp://0.0.0.0:11111"


# ==========================================
# LOAD MODEL
# ==========================================

print("Loading YOLO model...")

model = YOLO(MODEL_PATH)

print("YOLO model loaded ✅")


# ==========================================
# CONNECT TELLO
# ==========================================

tello = Tello()

cap = None

try:

    print("\nConnecting to Tello...")

    tello.connect()

    print("Connected ✅")

    print(
        f"Battery: {tello.get_battery()}%"
    )


    # ======================================
    # START STREAM
    # ======================================

    print("\nStarting Tello stream...")

    tello.streamon()

    time.sleep(2)


    # ======================================
    # OPEN WITH OPENCV
    # ======================================

    print(
        "Opening Tello video with OpenCV..."
    )

    cap = cv2.VideoCapture(
        VIDEO_URL,
        cv2.CAP_FFMPEG
    )

    time.sleep(2)


    if not cap.isOpened():

        raise RuntimeError(
            "Could not open Tello video stream"
        )


    print("OpenCV stream: OK ✅")

    print("\nYOLO + ByteTrack is starting...")
    print("Press Q to quit.\n")


    # ======================================
    # MAIN LOOP
    # ======================================

    while True:

        ret, frame = cap.read()

        if not ret:

            continue


        # ==================================
        # YOLO + BYTETRACK
        # ==================================

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[0],
            conf=CONFIDENCE_THRESHOLD,
            verbose=False
        )


        result = results[0]


        # ==================================
        # DRAW RESULTS
        # ==================================

        annotated_frame = result.plot()


        # ==================================
        # PEOPLE COUNT
        # ==================================

        people_count = 0

        if result.boxes is not None:

            people_count = len(
                result.boxes
            )


        # ==================================
        # TRACK IDs + CONFIDENCE
        # ==================================

        if (
            result.boxes is not None
            and result.boxes.id is not None
        ):

            track_ids = (
                result.boxes.id
                .int()
                .cpu()
                .tolist()
            )

            confidences = (
                result.boxes.conf
                .cpu()
                .tolist()
            )


            for track_id, confidence in zip(
                track_ids,
                confidences
            ):

                print(
                    f"Person ID: {track_id} | "
                    f"Confidence: "
                    f"{confidence * 100:.1f}%"
                )


        # ==================================
        # DISPLAY PEOPLE COUNT
        # ==================================

        cv2.putText(
            annotated_frame,
            f"People: {people_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )


        cv2.putText(
            annotated_frame,
            "YOLO + ByteTrack",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )


        # ==================================
        # SHOW
        # ==================================

        cv2.imshow(
            "Tello - YOLO + ByteTrack",
            annotated_frame
        )


        # ==================================
        # QUIT
        # ==================================

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):

            break


finally:

    print("\nStopping...")

    if cap is not None:

        cap.release()


    cv2.destroyAllWindows()


    try:

        tello.streamoff()

    except:

        pass


    try:

        tello.end()

    except:

        pass


    print("Tello connection closed.")