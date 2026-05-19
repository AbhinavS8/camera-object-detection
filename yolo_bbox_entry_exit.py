import cv2
import time
from ultralytics import YOLO


def get_box_edge_zones(x1, y1, x2, y2, frame_width, frame_height, margin):
    zones = []

    if x1 <= margin:
        zones.append("left")
    if x2 >= frame_width - margin:
        zones.append("right")
    if y1 <= margin:
        zones.append("top")
    if y2 >= frame_height - margin:
        zones.append("bottom")

    return zones


def is_box_inside_edge_margin(x1, y1, x2, y2, frame_width, frame_height, margin):
    return (
        x1 > margin and
        y1 > margin and
        x2 < frame_width - margin and
        y2 < frame_height - margin
    )


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
#         "first_edge_zones": ["left"],
#         "last_edge_zones": ["left"],
#         "inside_frames": 0,
#         "has_been_inside": False,
#         "entered": False,
#         "exited": False
#     }
# }

# ----------------------------
# Entry / Exit Settings
# ----------------------------
in_count = 0
out_count = 0

edge_margin_ratio = 0.01
min_track_time = 0.3
min_inside_frames = 2
min_exit_track_time = 0.8
max_disappeared_time = 1.5

# ----------------------------
# Main Loop
# ----------------------------
while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Resize for stable FPS
    frame = cv2.resize(frame, (1280, 720))
    frame_height, frame_width = frame.shape[:2]
    edge_margin = int(min(frame_width, frame_height) * edge_margin_ratio)

    # ----------------------------
    # YOLO + ByteTrack
    # ----------------------------
    results = model.track(
        frame,
        persist=True,
        tracker="botsort.yaml",
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

            edge_zones = get_box_edge_zones(
                x1,
                y1,
                x2,
                y2,
                frame_width,
                frame_height,
                edge_margin
            )
            is_inside = is_box_inside_edge_margin(
                x1,
                y1,
                x2,
                y2,
                frame_width,
                frame_height,
                edge_margin
            )

            # ----------------------------
            # New Object
            # ----------------------------
            if track_id not in tracked_objects:

                tracked_objects[track_id] = {
                    "first_seen": current_time,
                    "last_seen": current_time,
                    "class_name": class_name,
                    "confidence": confidence,
                    "first_edge_zones": edge_zones,
                    "last_edge_zones": edge_zones,
                    "inside_frames": 1 if is_inside else 0,
                    "has_been_inside": is_inside,
                    "entered": False,
                    "exited": False
                }

            # ----------------------------
            # Existing Object
            # ----------------------------
            else:

                tracked_object = tracked_objects[track_id]

                tracked_object["last_seen"] = current_time
                tracked_object["class_name"] = class_name
                tracked_object["confidence"] = confidence
                tracked_object["last_edge_zones"] = edge_zones

                if is_inside:
                    tracked_object["inside_frames"] += 1

                    if tracked_object["inside_frames"] >= min_inside_frames:
                        tracked_object["has_been_inside"] = True

            tracked_object = tracked_objects[track_id]
            track_time = current_time - tracked_object["first_seen"]

            # ----------------------------
            # Confirm Entry
            # ----------------------------
            if (
                not tracked_object["entered"] and
                tracked_object["first_edge_zones"] and
                tracked_object["has_been_inside"] and
                track_time >= min_track_time
            ):

                in_count += 1
                tracked_object["entered"] = True

                print(
                    f"object {track_id} ENTERED from "
                    f"{'/'.join(tracked_object['first_edge_zones'])}"
                )

            # ----------------------------
            # Draw Bounding Box
            # ----------------------------
            box_color = (0, 255, 0) if is_inside else (0, 165, 255)

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                box_color,
                2
            )

            # Draw centroid
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            cv2.circle(
                frame,
                (cx, cy),
                5,
                (0, 0, 255),
                -1
            )

            # Draw ID and class
            cv2.putText(
                frame,
                f"ID {track_id}: {class_name}",
                (x1, max(20, y1 - 40)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )

            # Draw bbox state
            state_text = "inside" if is_inside else "edge"
            cv2.putText(
                frame,
                f"{state_text} {track_time:.1f}s",
                (x1, max(45, y1 - 15)),
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
    # Remove Lost Objects / Confirm Exit
    # ----------------------------
    current_time = time.time()

    remove_ids = []

    for track_id, data in tracked_objects.items():

        if (
            current_time - data["last_seen"]
            > max_disappeared_time
        ):

            total_time = data["last_seen"] - data["first_seen"]

            if (
                data["has_been_inside"] and
                data["last_edge_zones"] and
                total_time >= min_exit_track_time and
                not data["exited"]
            ):

                out_count += 1
                data["exited"] = True

                print(
                    f"object {track_id} EXITED through "
                    f"{'/'.join(data['last_edge_zones'])}"
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
    # Draw Edge Zone
    # ----------------------------
    # cv2.rectangle(
    #     frame,
    #     (edge_margin, edge_margin),
    #     (frame_width - edge_margin, frame_height - edge_margin),
    #     (255, 0, 0),
    #     2
    # )

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

    cv2.putText(
        frame,
        f"EDGE: {edge_margin}px",
        (20, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2
    )

    # ----------------------------
    # Show Frame
    # ----------------------------
    cv2.imshow(
        "YOLO BBox Entry / Exit",
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
