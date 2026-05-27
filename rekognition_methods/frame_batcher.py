import time
from dataclasses import dataclass
from datetime import datetime, timezone

import cv2


CAMERA_INDEX = 0
SAMPLE_INTERVAL_SECONDS = 0.2
BATCH_SIZE = 5
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720


@dataclass
class CapturedFrame:
    run_id: str
    sequence_id: int
    batch_id: int
    batch_index: int
    captured_at_epoch: float
    captured_at_utc: str
    frame: object


@dataclass
class FrameBatch:
    run_id: str
    batch_id: int
    created_at_epoch: float
    created_at_utc: str
    frames: list[CapturedFrame]


def utc_timestamp():
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def safe_timestamp(value):
    return value.replace(":", "").replace(".", "-")


class FrameBatcher:
    def __init__(
        self,
        camera_index=CAMERA_INDEX,
        batch_size=BATCH_SIZE,
        sample_interval_seconds=SAMPLE_INTERVAL_SECONDS,
        frame_width=FRAME_WIDTH,
        frame_height=FRAME_HEIGHT
    ):
        self.camera_index = camera_index
        self.batch_size = batch_size
        self.sample_interval_seconds = sample_interval_seconds
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.run_id = safe_timestamp(utc_timestamp())

    def capture_frames(self, stop_event):
        cap = cv2.VideoCapture(self.camera_index)

        if not cap.isOpened():
            raise RuntimeError("Could not open webcam")

        sequence_id = 0
        next_sample_time = time.monotonic()

        try:
            while not stop_event.is_set():
                now = time.monotonic()

                if now < next_sample_time:
                    time.sleep(min(0.01, next_sample_time - now))
                    continue

                ret, frame = cap.read()

                if not ret:
                    raise RuntimeError("Could not read frame")

                frame = cv2.resize(
                    frame,
                    (self.frame_width, self.frame_height)
                )
                yield CapturedFrame(
                    run_id=self.run_id,
                    sequence_id=sequence_id,
                    batch_id=sequence_id // self.batch_size,
                    batch_index=sequence_id % self.batch_size,
                    captured_at_epoch=time.time(),
                    captured_at_utc=utc_timestamp(),
                    frame=frame.copy()
                )
                sequence_id += 1
                next_sample_time += self.sample_interval_seconds
        finally:
            cap.release()

    def capture_batches(self, stop_event):
        frames = []
        batch_id = 0

        for captured_frame in self.capture_frames(stop_event):
            frames.append(captured_frame)

            if len(frames) < self.batch_size:
                continue

            yield FrameBatch(
                run_id=self.run_id,
                batch_id=batch_id,
                created_at_epoch=time.time(),
                created_at_utc=utc_timestamp(),
                frames=frames
            )
            frames = []
            batch_id += 1

        if frames:
            yield FrameBatch(
                run_id=self.run_id,
                batch_id=batch_id,
                created_at_epoch=time.time(),
                created_at_utc=utc_timestamp(),
                frames=frames
            )
