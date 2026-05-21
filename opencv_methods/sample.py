import cv2
import numpy as np
import time
import math

# ----------------------------
# Camera / Stream
# ----------------------------
cap = cv2.VideoCapture(0)

# Example RTSPS stream:
# cap = cv2.VideoCapture(
#     "rtsps://user:password@ip:554/cam/realmonitor?channel=1&subtype=1"
# )

# ----------------------------
# Background Subtractor
# ----------------------------
fgbg = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=16,
    detectShadows=False
)

# ----------------------------
# Tracking Variables
# ----------------------------
next_object_id = 0
objects = {}

# Object structure:
# {
#     id: {
#         "centroid": (cx, cy),
#         "first_seen": time,
#         "last_seen": time,
#         "counted_in": False,
#         "counted_out": False
#     }
# }

# ----------------------------
# Counting Variables
# ----------------------------
in_count = 0
out_count = 0

# Horizontal line
line_y = 300

# Matching distance threshold
max_distance = 50

# Minimum contour area
min_area = 1500

# ----------------------------
# Utility Function
# ----------------------------
def distance(p1, p2):
    return math.sqrt(
        (p1[0] - p2[0])**2 +
        (p1[1] - p2[1])**2
    )

# ----------------------------
# Main Loop
# ----------------------------
while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.resize(frame, (1280, 720))

    height, width = frame.shape[:2]

    # ----------------------------
    # Preprocessing
    # ----------------------------
    blur = cv2.GaussianBlur(frame, (5, 5), 0)

    # Background subtraction
    mask = fgbg.apply(blur)

    # Remove shadows / weak pixels
    _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)

    # Morphology cleanup
    kernel = np.ones((3, 3), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    # ----------------------------
    # Contour Detection
    # ----------------------------
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    current_centroids = []

    for cnt in contours:

        area = cv2.contourArea(cnt)

        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        cx = x + w // 2
        cy = y + h // 2

        current_centroids.append((cx, cy, x, y, w, h))

    # ----------------------------
    # Tracking
    # ----------------------------
    updated_objects = {}

    for centroid_data in current_centroids:

        cx, cy, x, y, w, h = centroid_data

        matched_id = None

        # Find closest existing object
        for object_id, data in objects.items():

            prev_centroid = data["centroid"]

            dist = distance(
                (cx, cy),
                prev_centroid
            )

            if dist < max_distance:
                matched_id = object_id
                break

        # ----------------------------
        # New Object
        # ----------------------------
        if matched_id is None:

            matched_id = next_object_id

            objects[matched_id] = {
                "centroid": (cx, cy),
                "first_seen": time.time(),
                "last_seen": time.time(),
                "counted_in": False,
                "counted_out": False,
                "previous_y": cy
            }

            next_object_id += 1

        # ----------------------------
        # Existing Object
        # ----------------------------
        else:

            previous_y = objects[matched_id]["previous_y"]

            # Crossing DOWN → IN
            if (
                previous_y < line_y and
                cy >= line_y and
                not objects[matched_id]["counted_in"]
            ):

                in_count += 1
                objects[matched_id]["counted_in"] = True

                print(f"Object {matched_id} ENTERED")

            # Crossing UP → OUT
            elif (
                previous_y > line_y and
                cy <= line_y and
                not objects[matched_id]["counted_out"]
            ):

                out_count += 1
                objects[matched_id]["counted_out"] = True

                print(f"Object {matched_id} EXITED")

            objects[matched_id]["previous_y"] = cy

        # Update object info
        updated_objects[matched_id] = {
            **objects[matched_id],
            "centroid": (cx, cy),
            "last_seen": time.time()
        }

        # ----------------------------
        # Dwell Time
        # ----------------------------
        dwell_time = (
            time.time() -
            updated_objects[matched_id]["first_seen"]
        )

        # ----------------------------
        # Draw Bounding Box
        # ----------------------------
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
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

        # Draw object info
        cv2.putText(
            frame,
            f"ID {matched_id}",
            (x, y - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Dwell: {dwell_time:.1f}s",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

    # Replace old objects
    objects = updated_objects

    # ----------------------------
    # Draw Counting Line
    # ----------------------------
    cv2.line(
        frame,
        (0, line_y),
        (width, line_y),
        (255, 0, 0),
        2
    )

    # ----------------------------
    # Display Counts
    # ----------------------------
    cv2.putText(
        frame,
        f"IN: {in_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"OUT: {out_count}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    # ----------------------------
    # Show Windows
    # ----------------------------
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    # Exit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ----------------------------
# Cleanup
# ----------------------------
cap.release()
cv2.destroyAllWindows()