import cv2
import numpy as np
import time
import math

# ----------------------------
# Camera / Stream
# ----------------------------
cap = cv2.VideoCapture(0)

# RTSPS Example:
# cap = cv2.VideoCapture(
#     "rtsps://user:password@ip:554/cam/realmonitor?channel=1&subtype=1"
# )

if not cap.isOpened():
    print("Failed to open stream")
    exit()

# ----------------------------
# Capture Initial Background
# ----------------------------
ret, background = cap.read()

if not ret:
    print("Failed to capture background")
    exit()

background = cv2.resize(background, (1280, 720))

background = cv2.GaussianBlur(
    background,
    (5, 5),
    0
)

background = cv2.cvtColor(
    background,
    cv2.COLOR_BGR2GRAY
)

# ----------------------------
# Tracking Variables
# ----------------------------
next_object_id = 0

objects = {}

# Structure:
# objects = {
#     id: {
#         "centroid": (cx, cy),
#         "first_seen": timestamp,
#         "last_seen": timestamp,
#         "previous_y": cy,
#         "counted_in": False,
#         "counted_out": False
#     }
# }

# ----------------------------
# Counting Variables
# ----------------------------
in_count = 0
out_count = 0

# Horizontal counting line
line_y = 350

# Maximum centroid matching distance
max_distance = 50

# Minimum contour area
min_area = 2000

# Object timeout
max_disappeared_time = 2.0

# ----------------------------
# Distance Function
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

    # ----------------------------
    # Resize
    # ----------------------------
    frame = cv2.resize(
        frame,
        (1280, 720)
    )

    height, width = frame.shape[:2]

    # ----------------------------
    # Preprocessing
    # ----------------------------
    blur = cv2.GaussianBlur(
        frame,
        (5, 5),
        0
    )

    gray = cv2.cvtColor(
        blur,
        cv2.COLOR_BGR2GRAY
    )

    # ----------------------------
    # Frame Difference
    # ----------------------------
    diff = cv2.absdiff(
        background,
        gray
    )

    # ----------------------------
    # Threshold
    # ----------------------------
    _, mask = cv2.threshold(
        diff,
        25,
        255,
        cv2.THRESH_BINARY
    )

    # ----------------------------
    # Morphology
    # ----------------------------
    kernel = np.ones((5, 5), np.uint8)

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
    # Find Contours
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

        # Bottom-center point
        cy = y + h

        current_centroids.append(
            (cx, cy, x, y, w, h)
        )

    # ----------------------------
    # Tracking
    # ----------------------------
    updated_objects = {}

    for centroid_data in current_centroids:

        cx, cy, x, y, w, h = centroid_data

        matched_id = None

        # ----------------------------
        # Find Closest Existing Object
        # ----------------------------
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

                "previous_y": cy,

                "counted_in": False,

                "counted_out": False
            }

            next_object_id += 1

        # ----------------------------
        # Existing Object
        # ----------------------------
        else:

            previous_y = objects[matched_id]["previous_y"]

            # ENTER
            if (
                previous_y < line_y and
                cy >= line_y and
                not objects[matched_id]["counted_in"]
            ):

                in_count += 1

                objects[matched_id]["counted_in"] = True

                print(
                    f"Object {matched_id} ENTERED"
                )

            # EXIT
            elif (
                previous_y > line_y and
                cy <= line_y and
                not objects[matched_id]["counted_out"]
            ):

                out_count += 1

                objects[matched_id]["counted_out"] = True

                print(
                    f"Object {matched_id} EXITED"
                )

            objects[matched_id]["previous_y"] = cy

        # ----------------------------
        # Update Object
        # ----------------------------
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

        # Draw object ID
        cv2.putText(
            frame,
            f"ID {matched_id}",
            (x, y - 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2
        )

        # Draw dwell time
        cv2.putText(
            frame,
            f"Time: {dwell_time:.1f}s",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

    # ----------------------------
    # Replace Old Objects
    # ----------------------------
    objects = updated_objects

    # ----------------------------
    # Remove Lost Objects
    # ----------------------------
    current_time = time.time()

    remove_ids = []

    for object_id, data in objects.items():

        if (
            current_time - data["last_seen"]
            > max_disappeared_time
        ):

            total_time = (
                data["last_seen"] -
                data["first_seen"]
            )

            print(
                f"Object {object_id} visible for "
                f"{total_time:.2f} seconds"
            )

            remove_ids.append(object_id)

    for object_id in remove_ids:

        del objects[object_id]

    # ----------------------------
    # Draw Counting Line
    # ----------------------------
    cv2.line(
        frame,
        (0, line_y),
        (width, line_y),
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
    # Show Windows
    # ----------------------------
    cv2.imshow("Frame", frame)

    cv2.imshow("Mask", mask)

    cv2.imshow("Difference", diff)

    # ----------------------------
    # Quit Key
    # ----------------------------
    if cv2.waitKey(1) & 0xFF == ord('q'):

        break

# ----------------------------
# Cleanup
# ----------------------------
cap.release()

cv2.destroyAllWindows()