import json
import os
from dataclasses import dataclass

import boto3
import cv2
from dotenv import load_dotenv


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"), override=True)

S3_BUCKET_ENV = "S3_FRAME_BUCKET"
AWS_REGION_ENV = "AWS_REGION"
AWS_DEFAULT_REGION_ENV = "AWS_DEFAULT_REGION"
S3_PREFIX_ENV = "S3_FRAME_PREFIX"
JPEG_QUALITY_ENV = "S3_FRAME_JPEG_QUALITY"


def get_env_value(name, default=None):
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default

    return value


def get_aws_region(region_name=None):
    return (
        region_name or
        get_env_value(AWS_REGION_ENV) or
        get_env_value(AWS_DEFAULT_REGION_ENV) or
        "ap-south-1"
    )


def get_s3_prefix(prefix=None):
    return (prefix or get_env_value(S3_PREFIX_ENV, "input")).strip("/")


def get_jpeg_quality(jpeg_quality=None):
    if jpeg_quality is not None:
        return jpeg_quality

    return int(get_env_value(JPEG_QUALITY_ENV, "85"))


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


class S3BatchUploader:
    def __init__(
        self,
        bucket_name=None,
        prefix=None,
        region_name=None,
        jpeg_quality=None
    ):
        self.bucket_name = bucket_name or get_env_value(S3_BUCKET_ENV)
        self.prefix = get_s3_prefix(prefix)
        self.jpeg_quality = get_jpeg_quality(jpeg_quality)
        self.region_name = get_aws_region(region_name)
        self.client = boto3.client("s3", region_name=self.region_name)

        if not self.bucket_name:
            raise RuntimeError(
                f"Set {S3_BUCKET_ENV} to the destination S3 bucket name."
            )

        print(
            "S3 upload target: "
            f"s3://{self.bucket_name}/{self.prefix}/ "
            f"region={self.region_name}"
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
            f"batch_{captured_frame.batch_id:08d}_"
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
