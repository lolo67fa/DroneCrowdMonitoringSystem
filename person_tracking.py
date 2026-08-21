from ultralytics import YOLO
import cv2
import time


# ==============================
# SETTINGS
# ==============================

MAX_PEOPLE = 5

# Load YOLO model
model = YOLO("yolo11n.pt")



# ==============================
# CAMERA
# ==============================



cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Could not open camera")
    exit()

print("📷 Camera started")
print("Press Q to quit")



# ==============================
# VARIABLES
# ==============================

peak_people = 0
start_time = time.time()


# ==============================
# MAIN LOOP
# ==============================

while True:

    ret, frame = cap.read()

    if not ret:
        print("❌ Failed to read frame")
        break

    # ==========================
    # YOLO + BYTE TRACK
    # ==========================

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[0],       # Person
        conf=0.4,
        verbose=False
    )

    result = results[0]

    # ==========================
    # CURRENT PEOPLE
    # ==========================

    current_people = 0

    if result.boxes.id is not None:
        current_people = len(result.boxes.id)

    # ==========================
    # PEAK PEOPLE
    # ==========================

    if current_people > peak_people:
        peak_people = current_people

    # ==========================
    # OCCUPANCY
    # ==========================

    occupancy = (
        current_people / MAX_PEOPLE
    ) * 100

    occupancy = min(occupancy, 100)

    # ==========================
    # CROWD STATUS
    # ==========================

    if current_people >= MAX_PEOPLE:
        status = "CROWDED"
        status_color = (0, 0, 255)

    elif current_people >= 7:
        status = "HIGH"
        status_color = (0, 165, 255)

    else:
        status = "NORMAL"
        status_color = (0, 255, 0)

    # ==========================
    # DRAW YOLO
    # ==========================

    annotated_frame = result.plot()

    # ==========================
    # DISPLAY INFORMATION
    # ==========================

    cv2.putText(
        annotated_frame,
        f"People: {current_people}/{MAX_PEOPLE}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )

    cv2.putText(
        annotated_frame,
        f"Occupancy: {occupancy:.0f}%",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        annotated_frame,
        f"Status: {status}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        status_color,
        2
    )

    cv2.putText(
        annotated_frame,
        f"Peak: {peak_people}",
        (20, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    # ==========================
    # CROWD ALERT
    # ==========================

    if current_people >= MAX_PEOPLE:

        cv2.putText(
            annotated_frame,
            "!!! CROWD ALERT !!!",
            (20, 210),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

    # ==========================
    # SHOW
    # ==========================

    cv2.imshow(
        "Drone Crowd Monitor",
        annotated_frame
    )

    # ==========================
    # QUIT
    # ==========================

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ==============================
# CLEANUP
# ==============================

cap.release()
cv2.destroyAllWindows()

elapsed_time = time.time() - start_time

print("\n==============================")
print("      MISSION SUMMARY")
print("==============================")
print(f"Peak People: {peak_people}")
print(f"Monitoring Time: {elapsed_time:.1f} seconds")
print("==============================")
print("Camera closed.")