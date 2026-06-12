import time
from types import SimpleNamespace

import cv2
import numpy as np
from ultralytics.trackers.byte_tracker import BYTETracker


CAMERA_INDEX = 0
SAMPLE_INTERVAL_SECONDS = 0.2
JPEG_QUALITY = 85
WINDOW_NAME = "AWS Rekognition ByteTrack Feed"
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
TRACKER_FRAME_RATE = int(1 / SAMPLE_INTERVAL_SECONDS)

EDGE_MARGIN_RATIO = 0.2
# MIN_TRACK_TIME = 0.3
# MIN_INSIDE_FRAMES = 2
# MIN_EXIT_TRACK_TIME = 0.8

SAMPLE_INTERVAL_SECONDS = 0.2

MIN_TRACK_FRAMES = 2      # 0.4 sec
MIN_EXIT_FRAMES = 4       # 0.8 sec
MAX_MISSING_FRAMES = 8    # 1.6 sec

MAX_DISAPPEARED_TIME = 4


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
    def __init__(self, xywh, conf, cls):
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
            self.xywh.values[item],
            self.conf.values[item],
            self.cls.values[item]
        )

def age_tracks(tracked_objects):
    for obj in tracked_objects.values():
        obj["frames_missing"] += 1
        
def create_tracker():
    args = SimpleNamespace(
        track_high_thresh=0.2,
        track_low_thresh=0.05,
        new_track_thresh=0.2,
        track_buffer=30,
        match_thresh=0.8,
        fuse_score=True,
        mot20=False,
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


def labels_to_tracker_detections(custom_labels, frame_width, frame_height):
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

        center_x = left + width / 2
        center_y = top + height / 2

        xywh.append([center_x, center_y, width, height])
        confidences.append(label["Confidence"] / 100.0)
        class_ids.append(0)

    return TrackerDetections(
        np.array(xywh, dtype=np.float32).reshape(-1, 4),
        np.array(confidences, dtype=np.float32),
        np.array(class_ids, dtype=np.float32),
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
            getattr(track, "idx", -1),
        ]

    return np.array(track, dtype=np.float32)


def update_tracker(tracker, detections, frame):
    if len(detections) == 0:
        return np.empty((0, 8), dtype=np.float32)

    tracks = tracker.update(detections, frame)

    if len(tracks) == 0:
        return np.empty((0, 8), dtype=np.float32)

    return np.array([normalize_track(track) for track in tracks], dtype=np.float32)


def update_entry_exit_state(tracks, tracked_objects, frame_width, frame_height, counts):
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
            edge_margin,
        )
        is_inside = is_box_inside_edge_margin(
            x1,
            y1,
            x2,
            y2,
            frame_width,
            frame_height,
            edge_margin,
        )

        if track_id not in tracked_objects:
            tracked_objects[track_id] = {
                "frames_seen": 1,
                "frames_missing": 0,
                "confidence": confidence,
                "first_edge_zones": edge_zones,
                "last_edge_zones": edge_zones,
                "inside_frames": 1 if is_inside else 0,
                "has_been_inside": is_inside,
                "entered": False,
                "exited": False,
            }
        else:
            tracked_object = tracked_objects[track_id]

            tracked_object["frames_seen"] += 1
            tracked_object["frames_missing"] = 0
            tracked_object["confidence"] = confidence
            tracked_object["last_edge_zones"] = edge_zones

            if is_inside:
                tracked_object["inside_frames"] += 1

                if tracked_object["inside_frames"] >= MIN_INSIDE_FRAMES:
                    tracked_object["has_been_inside"] = True

        tracked_object = tracked_objects[track_id]
        track_time = (
            tracked_object["frames_seen"]
            * SAMPLE_INTERVAL_SECONDS
        )
        if (
            not tracked_object["entered"] and
            tracked_object["first_edge_zones"] and
            tracked_object["has_been_inside"] and
            tracked_object["frames_seen"] >= MIN_TRACK_FRAMES
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

        total_time = (
            data["frames_seen"]
            * SAMPLE_INTERVAL_SECONDS
        )
        if (
            data["has_been_inside"] and
            data["last_edge_zones"] and
            data["frames_seen"] >= MIN_EXIT_FRAMES and
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
        dtype=np.float32,
    )