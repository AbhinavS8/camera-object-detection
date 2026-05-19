# from ultralytics import YOLO
# import cv2

# model = YOLO("yolov8n.pt")

# cap = cv2.VideoCapture(0)

# while True:

#     ret, frame = cap.read()

#     if not ret:
#         break

#     # Run tracking
#     results = model.track(
#         frame,
#         persist=True,
#         tracker="bytetrack.yaml"
#     )

#     annotated = results[0].plot()

#     cv2.imshow("YOLO + ByteTrack", annotated)

#     if cv2.waitKey(1) == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

import cv2
import time
from ultralytics import YOLO

# ----------------------------
# Load YOLO Model
# ----------------------------
model = YOLO("yolov8n.pt")

# ----------------------------
# Video Source
# ----------------------------
# Webcam:
cap = cv2.VideoCapture(0)

# RTSPS Example:
# cap = cv2.VideoCapture(
#     "rtsps://user:password@ip:554/cam/realmonitor?channel=1&subtype=1"
# )

if not cap.isOpened():
    print("Failed to open stream")
    exit()

# ----------------------------
# Tracking Data
# ----------------------------
tracked_objects = {}
last_visible_objects_print_time = 0.0

# Structure:
# tracked_objects = {
#     track_id: {
#         "first_seen": timestamp,
#         "last_seen": timestamp,
#         "class_name": class_name,
#         "confidence": confidence,
#         "previous_y": cy,
#         "entered": False,
#         "exited": False
#     }
# }

# ----------------------------
# Counting
# ----------------------------
in_count = 0
out_count = 0

# Horizontal counting line
line_y = 350

# Timeout before object removal
max_disappeared_time = 2.0

# ----------------------------
# Main Loop
# ----------------------------
while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Resize for stable FPS
    frame = cv2.resize(frame, (1280, 720))

    # ----------------------------
    # YOLO + ByteTrack
    # ----------------------------
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        # classes=[0],      # Only detect people
        verbose=False
    )

    boxes = results[0].boxes

    current_ids = set()
    visible_objects = []
    current_time = time.time()

    # ----------------------------
    # Process Detections
    # ----------------------------
    if boxes.id is not None:

        for box in boxes:

            track_id = int(box.id[0])
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])

            current_ids.add(track_id)
            visible_objects.append(
                f"ID {track_id}: {class_name} ({confidence:.2f})"
            )

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            # Bottom-center centroid
            cx = (x1 + x2) // 2
            cy = y2

            # ----------------------------
            # New Object
            # ----------------------------
            if track_id not in tracked_objects:

                tracked_objects[track_id] = {
                    "first_seen": current_time,
                    "last_seen": current_time,
                    "class_name": class_name,
                    "confidence": confidence,
                    "previous_y": cy,
                    "entered": False,
                    "exited": False
                }

            # ----------------------------
            # Existing Object
            # ----------------------------
            else:

                previous_y = tracked_objects[track_id]["previous_y"]

                # Crossing DOWN → ENTER
                if (
                    previous_y < line_y and
                    cy >= line_y and
                    not tracked_objects[track_id]["entered"]
                ):

                    in_count += 1

                    tracked_objects[track_id]["entered"] = True

                    print(f"object {track_id} ENTERED")

                # Crossing UP → EXIT
                elif (
                    previous_y > line_y and
                    cy <= line_y and
                    not tracked_objects[track_id]["exited"]
                ):

                    out_count += 1

                    tracked_objects[track_id]["exited"] = True

                    print(f"object {track_id} EXITED")

                tracked_objects[track_id]["previous_y"] = cy

                tracked_objects[track_id]["last_seen"] = current_time
                tracked_objects[track_id]["class_name"] = class_name
                tracked_objects[track_id]["confidence"] = confidence

            # ----------------------------
            # Dwell Time
            # ----------------------------
            dwell_time = (
                current_time -
                tracked_objects[track_id]["first_seen"]
            )

            # ----------------------------
            # Draw Bounding Box
            # ----------------------------
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Draw centroid
            cv2.circle(
                frame,
                (cx, cy),
                5,
                (0, 0, 255),
                -1
            )

            # Draw ID
            cv2.putText(
                frame,
                f"ID {track_id}: {class_name}",
                (x1, y1 - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )

            # Draw dwell time
            cv2.putText(
                frame,
                f"Time: {dwell_time:.1f}s",
                (x1, y1 - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

    # ----------------------------
    # Print Currently Visible Objects
    # ----------------------------
    if current_time - last_visible_objects_print_time >= 1.0:
        if visible_objects:
            print("Visible objects: " + ", ".join(visible_objects))
        else:
            print("Visible objects: none")

        last_visible_objects_print_time = current_time

    # ----------------------------
    # Remove Lost Objects
    # ----------------------------
    current_time = time.time()

    remove_ids = []

    for track_id, data in tracked_objects.items():

        if (
            current_time - data["last_seen"]
            > max_disappeared_time
        ):

            total_time = (
                data["last_seen"] -
                data["first_seen"]
            )

            print(
                f"object {track_id} "
                f"was visible for "
                f"{total_time:.2f} seconds"
            )

            remove_ids.append(track_id)

    for track_id in remove_ids:
        del tracked_objects[track_id]

    # ----------------------------
    # Draw Counting Line
    # ----------------------------
    cv2.line(
        frame,
        (0, line_y),
        (frame.shape[1], line_y),
        (255, 0, 0),
        3
    )

    # ----------------------------
    # Display Counts
    # ----------------------------
    cv2.putText(
        frame,
        f"IN: {in_count}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"OUT: {out_count}",
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    # ----------------------------
    # Show Frame
    # ----------------------------
    cv2.imshow(
        "YOLO + ByteTrack",
        frame
    )

    # Quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ----------------------------
# Cleanup
# ----------------------------
cap.release()
cv2.destroyAllWindows()
