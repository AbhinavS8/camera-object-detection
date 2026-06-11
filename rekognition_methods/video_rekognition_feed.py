import time
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from ultralytics.trackers.byte_tracker import BYTETracker

from rekognition import detect_custom_labels_from_bytes


CAMERA_INDEX = 0
SAMPLE_INTERVAL_SECONDS = 0.2
JPEG_QUALITY = 85
WINDOW_NAME = "AWS Rekognition ByteTrack Feed"
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
TRACKER_FRAME_RATE = int(1 / SAMPLE_INTERVAL_SECONDS)

EDGE_MARGIN_RATIO = 0.2
MIN_TRACK_TIME = 0.3
MIN_INSIDE_FRAMES = 2
MIN_EXIT_TRACK_TIME = 0.8
MAX_DISAPPEARED_TIME = 1.5


class ArrayAdapter:
    def __init__(self, values):
        self.values = values

    def cpu(self):
        return self

    def numpy(self):
        return self.values

    def astype(self, dtype):
        return self.values.astype(dtype)

    def reshape(self, *shape):
        return self.values.reshape(*shape)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, item):
        return self.values[item]

    def __array__(self, dtype=None):
        if dtype is None:
            return self.values

        return self.values.astype(dtype)

    def __ge__(self, other):
        return self.values >= other

    def __gt__(self, other):
        return self.values > other

    def __le__(self, other):
        return self.values <= other

    def __lt__(self, other):
        return self.values < other


class TrackerDetections:
    def __init__(self, xyxy, xywh, conf, cls):
        self.xyxy = ArrayAdapter(xyxy)
        self.xywh = ArrayAdapter(xywh)
        self.conf = ArrayAdapter(conf)
        self.cls = ArrayAdapter(cls)

    def cpu(self):
        return self

    def numpy(self):
        return self

    def __len__(self):
        return len(self.conf)

    def __getitem__(self, item):
        if isinstance(item, int):
            return self

        return TrackerDetections(
            self.xyxy.values[item],
            self.xywh.values[item],
            self.conf.values[item],
            self.cls.values[item]
        )


def create_tracker():
    args = SimpleNamespace(
        track_high_thresh=0.05,
        track_low_thresh=0.001,
        new_track_thresh=0.05,
        track_buffer=80,
        match_thresh=0.3,
        fuse_score=True,
        mot20=False
    )

    try:
        return BYTETracker(args, frame_rate=TRACKER_FRAME_RATE)
    except TypeError:
        return BYTETracker(args)


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


def encode_frame_as_jpeg(frame):
    success, buffer = cv2.imencode(
        ".jpg",
        frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
    )

    if not success:
        raise RuntimeError("Could not encode frame as JPEG")

    return buffer.tobytes()


def detect_frame(frame):
    image_bytes = encode_frame_as_jpeg(frame)
    response = detect_custom_labels_from_bytes(image_bytes)
    return response.get("CustomLabels", [])


def labels_to_tracker_detections(custom_labels, frame_width, frame_height):
    xyxy = []
    xywh = []
    confidences = []
    class_ids = []

    for label in custom_labels:
        box = label.get("Geometry", {}).get("BoundingBox")

        if not box:
            continue

        left = box["Left"] * frame_width
        top = box["Top"] * frame_height
        width = box["Width"] * frame_width
        height = box["Height"] * frame_height
        right = left + width
        bottom = top + height

        center_x = left + width / 2
        center_y = top + height / 2

        xyxy.append([left, top, right, bottom])
        xywh.append([center_x, center_y, width, height])
        confidences.append(label["Confidence"] / 100.0)
        class_ids.append(0)

    return TrackerDetections(
        np.array(xyxy, dtype=np.float32).reshape(-1, 4),
        np.array(xywh, dtype=np.float32).reshape(-1, 4),
        np.array(confidences, dtype=np.float32),
        np.array(class_ids, dtype=np.float32)
    )


def normalize_track(track):
    if hasattr(track, "result"):
        track = track.result
    elif hasattr(track, "tlbr"):
        x1, y1, x2, y2 = track.tlbr
        track = [
            x1,
            y1,
            x2,
            y2,
            getattr(track, "track_id", -1),
            getattr(track, "score", 0.0),
            getattr(track, "cls", 0),
            getattr(track, "idx", -1)
        ]

    return np.array(track, dtype=np.float32)


def update_tracker(tracker, detections, frame):
    if len(detections) == 0:
        return np.empty((0, 8), dtype=np.float32)

    tracks = tracker.update(detections, frame)

    if len(tracks) == 0:
        return np.empty((0, 8), dtype=np.float32)

    return np.array(
        [normalize_track(track) for track in tracks],
        dtype=np.float32
    )


def update_entry_exit_state(
    tracks,
    tracked_objects,
    frame_width,
    frame_height,
    counts
):
    current_time = time.time()
    visible_objects = []
    edge_margin = int(min(frame_width, frame_height) * EDGE_MARGIN_RATIO)

    for track in tracks:
        x1, y1, x2, y2 = map(int, track[:4])
        track_id = int(track[4])
        confidence = float(track[5])

        visible_objects.append(f"ID {track_id}: package ({confidence:.2f})")

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

        if track_id not in tracked_objects:
            tracked_objects[track_id] = {
                "first_seen": current_time,
                "last_seen": current_time,
                "confidence": confidence,
                "first_edge_zones": edge_zones,
                "last_edge_zones": edge_zones,
                "inside_frames": 1 if is_inside else 0,
                "has_been_inside": is_inside,
                "entered": False,
                "exited": False
            }
        else:
            tracked_object = tracked_objects[track_id]
            tracked_object["last_seen"] = current_time
            tracked_object["confidence"] = confidence
            tracked_object["last_edge_zones"] = edge_zones

            if is_inside:
                tracked_object["inside_frames"] += 1

                if tracked_object["inside_frames"] >= MIN_INSIDE_FRAMES:
                    tracked_object["has_been_inside"] = True

        tracked_object = tracked_objects[track_id]
        track_time = current_time - tracked_object["first_seen"]

        if (
            not tracked_object["entered"] and
            tracked_object["first_edge_zones"] and
            tracked_object["has_been_inside"] and
            track_time >= MIN_TRACK_TIME
        ):
            counts["in"] += 1
            tracked_object["entered"] = True

            print(
                f"object {track_id} ENTERED from "
                f"{'/'.join(tracked_object['first_edge_zones'])}"
            )

    return visible_objects


def remove_lost_objects(tracked_objects, counts):
    current_time = time.time()
    remove_ids = []

    for track_id, data in tracked_objects.items():
        if current_time - data["last_seen"] <= MAX_DISAPPEARED_TIME:
            continue

        total_time = data["last_seen"] - data["first_seen"]

        if (
            data["has_been_inside"] and
            data["last_edge_zones"] and
            total_time >= MIN_EXIT_TRACK_TIME and
            not data["exited"]
        ):
            counts["out"] += 1
            data["exited"] = True

            print(
                f"object {track_id} EXITED through "
                f"{'/'.join(data['last_edge_zones'])}"
            )

        print(
            f"object {track_id} was visible for "
            f"{total_time:.2f} seconds"
        )

        remove_ids.append(track_id)

    for track_id in remove_ids:
        del tracked_objects[track_id]

    return remove_ids


def remove_stale_tracks(tracks, removed_track_ids):
    if len(tracks) == 0 or not removed_track_ids:
        return tracks

    removed_track_ids = set(removed_track_ids)

    return np.array(
        [track for track in tracks if int(track[4]) not in removed_track_ids],
        dtype=np.float32
    )


def draw_tracks(frame, tracks, tracked_objects):
    frame_height, frame_width = frame.shape[:2]
    edge_margin = int(min(frame_width, frame_height) * EDGE_MARGIN_RATIO)

    for track in tracks:
        x1, y1, x2, y2 = map(int, track[:4])
        track_id = int(track[4])
        confidence = float(track[5])
        tracked_object = tracked_objects.get(track_id, {})
        track_time = time.time() - tracked_object.get("first_seen", time.time())
        is_inside = is_box_inside_edge_margin(
            x1,
            y1,
            x2,
            y2,
            frame_width,
            frame_height,
            edge_margin
        )
        box_color = (0, 255, 0) if is_inside else (0, 165, 255)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            box_color,
            2
        )

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

        cv2.putText(
            frame,
            f"ID {track_id}: package {confidence:.2f}",
            (x1, max(20, y1 - 40)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

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


def draw_counts(frame, counts):
    frame_height, frame_width = frame.shape[:2]
    edge_margin = int(min(frame_width, frame_height) * EDGE_MARGIN_RATIO)

    cv2.putText(
        frame,
        f"IN: {counts['in']}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"OUT: {counts['out']}",
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

    cv2.rectangle(
        frame,
        (edge_margin, edge_margin),
        (frame_width - edge_margin, frame_height - edge_margin),
        (255, 0, 0),
        2
    )


def main():
    tracker = create_tracker()
    latest_tracks = np.empty((0, 8), dtype=np.float32)
    tracked_objects = {}
    counts = {
        "in": 0,
        "out": 0
    }
    last_visible_objects_print_time = 0.0

    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, FRAME_WIDTH, FRAME_HEIGHT)

    in_flight_detection = None
    last_sample_time = 0

    with ThreadPoolExecutor(max_workers=1) as executor:
        try:
            while True:
                ret, frame = cap.read()

                if not ret:
                    print("Error: Could not read frame")
                    break

                frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
                now = time.monotonic()

                if in_flight_detection and in_flight_detection.done():
                    try:
                        latest_labels = in_flight_detection.result()
                        detections = labels_to_tracker_detections(
                            latest_labels,
                            FRAME_WIDTH,
                            FRAME_HEIGHT
                        )
                        latest_tracks = update_tracker(
                            tracker,
                            detections,
                            frame
                        )
                        visible_objects = update_entry_exit_state(
                            latest_tracks,
                            tracked_objects,
                            FRAME_WIDTH,
                            FRAME_HEIGHT,
                            counts
                        )

                        if time.time() - last_visible_objects_print_time >= 1.0:
                            if visible_objects:
                                print("Visible objects: " + ", ".join(visible_objects))
                            else:
                                print("Visible objects: none")

                            last_visible_objects_print_time = time.time()
                    except Exception as exc:
                        print(f"Detection/tracking error: {exc}")
                        latest_tracks = np.empty((0, 8), dtype=np.float32)

                    in_flight_detection = None

                if (
                    in_flight_detection is None
                    and now - last_sample_time >= SAMPLE_INTERVAL_SECONDS
                ):
                    last_sample_time = now
                    detection_frame = frame.copy()
                    in_flight_detection = executor.submit(
                        detect_frame,
                        detection_frame
                    )

                output_frame = frame.copy()
                removed_track_ids = remove_lost_objects(tracked_objects, counts)
                latest_tracks = remove_stale_tracks(latest_tracks, removed_track_ids)
                draw_tracks(output_frame, latest_tracks, tracked_objects)
                draw_counts(output_frame, counts)

                cv2.imshow(WINDOW_NAME, output_frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
