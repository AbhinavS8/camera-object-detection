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


def create_tracker():
    args = SimpleNamespace(
        track_high_thresh=0.2,
        track_low_thresh=0.05,
        new_track_thresh=0.2,
        track_buffer=30,
        match_thresh=0.8,
        fuse_score=True,
        mot20=False
    )

    try:
        return BYTETracker(args, frame_rate=TRACKER_FRAME_RATE)
    except TypeError:
        return BYTETracker(args)


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


def draw_tracks(frame, tracks):
    for track in tracks:
        x1, y1, x2, y2 = map(int, track[:4])
        track_id = int(track[4])
        confidence = float(track[5])

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"ID {track_id}: package {confidence:.2f}",
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


def main():
    tracker = create_tracker()
    latest_tracks = np.empty((0, 8), dtype=np.float32)

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
                draw_tracks(output_frame, latest_tracks)

                cv2.imshow(WINDOW_NAME, output_frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
