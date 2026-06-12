import time
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
import cv2

from core.config import config

logger = logging.getLogger(__name__)

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
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")

def safe_timestamp(value):
    return value.replace(":", "").replace(".", "-")

class FrameBatcher:
    def __init__(
        self,
        camera_index=config.CAMERA_INDEX,
        batch_size=config.BATCH_SIZE,
        sample_interval_seconds=config.SAMPLE_INTERVAL,
        frame_width=config.FRAME_WIDTH,
        frame_height=config.FRAME_HEIGHT
    ):
        self.camera_index = camera_index
        self.batch_size = batch_size
        self.sample_interval_seconds = sample_interval_seconds
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.run_id = safe_timestamp(utc_timestamp())

    def capture_frames(self, stop_event):
        """Yields continuous frames from the webcam at the specified interval."""
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            logger.error(f"Hardware Error: Could not open webcam at index {self.camera_index}")
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
                    logger.error("Hardware Error: Could not read frame from webcam")
                    raise RuntimeError("Could not read frame")

                frame = cv2.resize(frame, (self.frame_width, self.frame_height))
                
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
            logger.info("Camera released successfully.")

    def create_batch(self, frames: list[CapturedFrame]) -> FrameBatch:
        """Packages a list of CapturedFrames into a single FrameBatch."""
        if not frames:
            raise ValueError("Cannot create a batch from an empty frame list")
            
        return FrameBatch(
            run_id=frames[0].run_id,
            batch_id=frames[0].batch_id,
            created_at_epoch=time.time(),
            created_at_utc=utc_timestamp(),
            frames=frames
        )