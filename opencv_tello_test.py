from djitellopy import Tello
import cv2
import time


tello = Tello()

try:

    print("Connecting to Tello...")
    tello.connect()

    print("Battery:", tello.get_battery(), "%")

    tello.streamon()

    # Give the Tello time to start the video stream
    time.sleep(8)

    print("Opening Tello video with OpenCV...")

    cap = cv2.VideoCapture(
        "udp://0.0.0.0:11111",
        cv2.CAP_FFMPEG
    )

    if not cap.isOpened():

        print("❌ OpenCV could not open Tello stream.")

        raise RuntimeError("Video stream failed")

    print("OpenCV stream: OK ✅")
    print("Press Q to quit.")

    while True:

        # =================================
        # GET FRAME FROM TELLO
        # =================================

        ret, frame = cap.read()

        if not ret:
            continue

        # =================================
        # SHOW FRAME
        # =================================

        cv2.imshow(
            "Tello Camera - OpenCV",
            frame
        )

        # =================================
        # QUIT
        # =================================

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


finally:

    print("Stopping...")

    try:
        cap.release()
    except:
        pass

    cv2.destroyAllWindows()

    try:
        tello.streamoff()
    except:
        pass

    tello.end()