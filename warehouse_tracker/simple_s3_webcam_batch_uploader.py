import json
import logging
import os
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import boto3
import cv2
import requests
from dotenv import load_dotenv


APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent

load_dotenv(PROJECT_DIR / ".env")
load_dotenv(APP_DIR / ".env", override=True)

DEFAULT_CONFIRMATION_URL = (
    "https://t4z7rn5u4l.execute-api.us-east-1.amazonaws.com/"
    "default/confirmationService"
)

CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
SAMPLE_INTERVAL_SECONDS = float(os.getenv("SAMPLE_INTERVAL_SECONDS", "0.2"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "5"))
FRAME_WIDTH = int(os.getenv("FRAME_WIDTH", "1280"))
FRAME_HEIGHT = int(os.getenv("FRAME_HEIGHT", "720"))
UPLOAD_WORKERS = int(os.getenv("S3_UPLOAD_WORKERS", "2"))
MAX_QUEUED_UPLOADS = int(os.getenv("S3_MAX_QUEUED_UPLOADS", "10"))

if UPLOAD_WORKERS < 1:
    raise ValueError("S3_UPLOAD_WORKERS must be at least 1")
if MAX_QUEUED_UPLOADS < UPLOAD_WORKERS:
    raise ValueError("S3_MAX_QUEUED_UPLOADS must be >= S3_UPLOAD_WORKERS")

S3_BUCKET = os.getenv("S3_FRAME_BUCKET")
AWS_REGION = (
    os.getenv("AWS_REGION") or
    os.getenv("AWS_DEFAULT_REGION") or
    "ap-south-1"
)
S3_PREFIX = os.getenv("S3_FRAME_PREFIX", "input").strip("/")
JPEG_QUALITY = int(os.getenv("S3_FRAME_JPEG_QUALITY", "85"))
CONFIRMATION_URL = os.getenv(
    "S3_UPLOAD_CONFIRMATION_URL",
    DEFAULT_CONFIRMATION_URL
)

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


@dataclass
class UploadedFrame:
    run_id: str
    sequence_id: int
    batch_id: int
    batch_index: int
    captured_at_epoch: float
    captured_at_utc: str
    bucket: str
    key: str


@dataclass
class UploadedBatch:
    run_id: str
    batch_id: int
    created_at_epoch: float
    created_at_utc: str
    bucket: str
    prefix: str
    manifest_key: str
    frames: list[UploadedFrame]


def utc_timestamp():
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def safe_timestamp(value):
    return value.replace(":", "").replace(".", "-")


def capture_frames():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam at index {CAMERA_INDEX}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    run_id = safe_timestamp(utc_timestamp())
    sequence_id = 0
    next_capture_at = time.monotonic()

    logger.info(
        "Capturing webcam frames every %.3fs from camera index %s",
        SAMPLE_INTERVAL_SECONDS,
        CAMERA_INDEX
    )

    try:
        while True:
            now = time.monotonic()
            if now < next_capture_at:
                time.sleep(min(0.01, next_capture_at - now))
                continue

            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Could not read frame from webcam")

            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            yield CapturedFrame(
                run_id=run_id,
                sequence_id=sequence_id,
                batch_id=sequence_id // BATCH_SIZE,
                batch_index=sequence_id % BATCH_SIZE,
                captured_at_epoch=time.time(),
                captured_at_utc=utc_timestamp(),
                frame=frame.copy()
            )

            sequence_id += 1
            next_capture_at += SAMPLE_INTERVAL_SECONDS
    finally:
        cap.release()
        logger.info("Camera released")


def create_batch(frames):
    if len(frames) != BATCH_SIZE:
        raise ValueError(f"Expected {BATCH_SIZE} frames, got {len(frames)}")

    return FrameBatch(
        run_id=frames[0].run_id,
        batch_id=frames[0].batch_id,
        created_at_epoch=time.time(),
        created_at_utc=utc_timestamp(),
        frames=frames
    )


def upload_batch_with_logging(uploader, batch):
    logger.info(
        "Uploading batch %s with %s frames",
        batch.batch_id,
        len(batch.frames)
    )
    uploaded_batch = uploader.upload_batch(batch)
    logger.info(
        "Batch %s complete: manifest=s3://%s/%s",
        uploaded_batch.batch_id,
        uploaded_batch.bucket,
        uploaded_batch.manifest_key
    )
    return uploaded_batch


class S3BatchUploader:
    def __init__(self):
        if not S3_BUCKET:
            raise RuntimeError("Set S3_FRAME_BUCKET to the destination S3 bucket.")

        self.bucket_name = S3_BUCKET
        self.prefix = S3_PREFIX
        self.jpeg_quality = JPEG_QUALITY
        self.confirmation_url = CONFIRMATION_URL
        self.client = boto3.client("s3", region_name=AWS_REGION)

        logger.info(
            "S3 upload target: s3://%s/%s/ region=%s",
            self.bucket_name,
            self.prefix,
            AWS_REGION
        )

    def upload_batch(self, batch):
        uploaded_frames = []

        for captured_frame in batch.frames:
            key = self._frame_key(batch, captured_frame)
            image_bytes = self._encode_frame(captured_frame.frame)

            self.client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=image_bytes,
                ContentType="image/jpeg",
                Metadata={
                    "run-id": captured_frame.run_id,
                    "batch-id": str(captured_frame.batch_id),
                    "batch-index": str(captured_frame.batch_index),
                    "sequence-id": str(captured_frame.sequence_id),
                    "captured-at-utc": captured_frame.captured_at_utc,
                    "captured-at-epoch": str(captured_frame.captured_at_epoch)
                }
            )

            uploaded_frames.append(
                UploadedFrame(
                    run_id=captured_frame.run_id,
                    sequence_id=captured_frame.sequence_id,
                    batch_id=captured_frame.batch_id,
                    batch_index=captured_frame.batch_index,
                    captured_at_epoch=captured_frame.captured_at_epoch,
                    captured_at_utc=captured_frame.captured_at_utc,
                    bucket=self.bucket_name,
                    key=key
                )
            )
            logger.info("Uploaded frame s3://%s/%s", self.bucket_name, key)

        manifest_key = self._manifest_key(batch)
        manifest = self._create_manifest(batch, uploaded_frames, manifest_key)
        self.client.put_object(
            Bucket=self.bucket_name,
            Key=manifest_key,
            Body=json.dumps(manifest, indent=2).encode("utf-8"),
            ContentType="application/json",
            Metadata={
                "run-id": batch.run_id,
                "batch-id": str(batch.batch_id),
                "created-at-utc": batch.created_at_utc
            }
        )
        logger.info("Uploaded manifest s3://%s/%s", self.bucket_name, manifest_key)

        self._send_confirmation(manifest)

        return UploadedBatch(
            run_id=batch.run_id,
            batch_id=batch.batch_id,
            created_at_epoch=batch.created_at_epoch,
            created_at_utc=batch.created_at_utc,
            bucket=self.bucket_name,
            prefix=self.prefix,
            manifest_key=manifest_key,
            frames=uploaded_frames
        )

    def _encode_frame(self, frame):
        success, buffer = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        )
        if not success:
            raise RuntimeError("Could not encode frame as JPEG")

        return buffer.tobytes()

    def _frame_key(self, batch, captured_frame):
        frame_time = captured_frame.captured_at_utc.replace(":", "").replace(".", "-")
        return (
            f"{self.prefix}/"
            f"seq_{captured_frame.sequence_id:012d}_"
            f"batch_{batch.batch_id:08d}_"
            f"idx_{captured_frame.batch_index:02d}_"
            f"{frame_time}.jpg"
        )

    def _manifest_key(self, batch):
        batch_time = batch.created_at_utc.replace(":", "").replace(".", "-")
        return (
            f"{self.prefix}/"
            f"manifest_batch_{batch.batch_id:08d}_{batch_time}.json"
        )

    def _create_manifest(self, batch, uploaded_frames, manifest_key):
        return {
            "run_id": batch.run_id,
            "batch_id": batch.batch_id,
            "created_at_epoch": batch.created_at_epoch,
            "created_at_utc": batch.created_at_utc,
            "bucket": self.bucket_name,
            "prefix": self.prefix,
            "manifest_key": manifest_key,
            "frames": [
                {
                    "run_id": frame.run_id,
                    "sequence_id": frame.sequence_id,
                    "batch_id": frame.batch_id,
                    "batch_index": frame.batch_index,
                    "captured_at_epoch": frame.captured_at_epoch,
                    "captured_at_utc": frame.captured_at_utc,
                    "bucket": frame.bucket,
                    "key": frame.key
                }
                for frame in uploaded_frames
            ]
        }

    def _send_confirmation(self, manifest):
        image_keys = [frame["key"] for frame in manifest["frames"]]
        frame_paths = [
            f"s3://{frame['bucket']}/{frame['key']}"
            for frame in manifest["frames"]
        ]
        payload = {
            "run_id": manifest["run_id"],
            "batch_id": manifest["batch_id"],
            "created_at_epoch": manifest["created_at_epoch"],
            "created_at_utc": manifest["created_at_utc"],
            "bucket": manifest["bucket"],
            "prefix": manifest["prefix"],
            "manifest_key": manifest["manifest_key"],
            "manifest_path": (
                f"s3://{manifest['bucket']}/{manifest['manifest_key']}"
            ),
            "images": image_keys,
            "image_keys": image_keys,
            "image_paths": frame_paths,
            "image_details": [
                {
                    "run_id": frame["run_id"],
                    "sequence_id": frame["sequence_id"],
                    "batch_id": frame["batch_id"],
                    "batch_index": frame["batch_index"],
                    "captured_at_epoch": frame["captured_at_epoch"],
                    "captured_at_utc": frame["captured_at_utc"],
                    "bucket": frame["bucket"],
                    "key": frame["key"],
                    "path": f"s3://{frame['bucket']}/{frame['key']}"
                }
                for frame in manifest["frames"]
            ]
        }

        logger.info(
            "Sending confirmation for batch %s to %s with %s images",
            payload["batch_id"],
            self.confirmation_url,
            len(payload["images"])
        )

        try:
            response = requests.post(
                self.confirmation_url,
                json=payload,
                timeout=10
            )
            logger.info(
                "Confirmation response status=%s body=%s",
                response.status_code,
                response.text[:1000]
            )
            response.raise_for_status()
        except requests.RequestException:
            logger.exception(
                "Confirmation failed for batch %s payload=%s",
                payload["batch_id"],
                json.dumps(payload)[:4000]
            )
            raise


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    )

    uploader = S3BatchUploader()
    frame_buffer = []
    in_flight_uploads = {}

    logger.info(
        "Starting webcam S3 batch uploader: batch_size=%s interval=%.3fs workers=%s",
        BATCH_SIZE,
        SAMPLE_INTERVAL_SECONDS,
        UPLOAD_WORKERS
    )

    with ThreadPoolExecutor(max_workers=UPLOAD_WORKERS) as executor:
        try:
            for captured_frame in capture_frames():
                frame_buffer.append(captured_frame)
                logger.info(
                    "Captured frame sequence=%s batch=%s index=%s buffered=%s/%s in_flight=%s",
                    captured_frame.sequence_id,
                    captured_frame.batch_id,
                    captured_frame.batch_index,
                    len(frame_buffer),
                    BATCH_SIZE,
                    len(in_flight_uploads)
                )

                completed = [
                    future
                    for future in in_flight_uploads
                    if future.done()
                ]
                for future in completed:
                    batch_id = in_flight_uploads.pop(future)
                    try:
                        future.result()
                    except Exception:
                        logger.exception("Batch %s failed", batch_id)

                if len(frame_buffer) < BATCH_SIZE:
                    continue

                if len(in_flight_uploads) >= MAX_QUEUED_UPLOADS:
                    logger.warning(
                        "Upload queue is full (%s); waiting for one upload to finish",
                        MAX_QUEUED_UPLOADS
                    )
                    completed, _ = wait(
                        in_flight_uploads,
                        return_when=FIRST_COMPLETED
                    )
                    for future in completed:
                        batch_id = in_flight_uploads.pop(future)
                        try:
                            future.result()
                        except Exception:
                            logger.exception("Batch %s failed", batch_id)

                batch = create_batch(frame_buffer[:BATCH_SIZE])
                del frame_buffer[:BATCH_SIZE]
                future = executor.submit(upload_batch_with_logging, uploader, batch)
                in_flight_uploads[future] = batch.batch_id
                logger.info(
                    "Queued batch %s for upload; in_flight=%s",
                    batch.batch_id,
                    len(in_flight_uploads)
                )
        except KeyboardInterrupt:
            logger.info("Stopped by user")
        finally:
            if in_flight_uploads:
                logger.info(
                    "Waiting for %s in-flight upload(s) to finish",
                    len(in_flight_uploads)
                )
            for future, batch_id in list(in_flight_uploads.items()):
                try:
                    future.result()
                except Exception:
                    logger.exception("Batch %s failed during shutdown", batch_id)


if __name__ == "__main__":
    main()
