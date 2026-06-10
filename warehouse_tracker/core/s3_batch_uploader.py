import json
import logging
from dataclasses import dataclass
import boto3
import cv2
import requests

from core.config import config

logger = logging.getLogger(__name__)

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
    def __init__(self):
        self.bucket_name = config.S3_BUCKET
        self.prefix = config.S3_PREFIX
        self.jpeg_quality = config.JPEG_QUALITY
        self.confirmation_url = config.CONFIRMATION_URL
        self.region_name = config.AWS_REGION
        
        if not self.bucket_name:
            logger.critical("S3_FRAME_BUCKET is not set in environment variables!")
            raise RuntimeError("Set S3_FRAME_BUCKET to the destination S3 bucket name.")

        # Let boto3 automatically handle credentials via IAM roles or ~/.aws/credentials
        self.client = boto3.client("s3", region_name=self.region_name)
        logger.info(f"S3 Uploader Initialized. Target: s3://{self.bucket_name}/{self.prefix}/")

    def upload_batch(self, batch) -> UploadedBatch:
        """Uploads frames to S3 and triggers the API Gateway manifest signal."""
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
                    "sequence-id": str(captured_frame.sequence_id),
                    "captured-at-utc": captured_frame.captured_at_utc
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

        # Create and upload the backup manifest to S3 (optional but good for debugging)
        manifest_key = self._manifest_key(batch)
        manifest = self._create_manifest(batch, uploaded_frames, manifest_key)
        
        self.client.put_object(
            Bucket=self.bucket_name,
            Key=manifest_key,
            Body=json.dumps(manifest, indent=2).encode("utf-8"),
            ContentType="application/json"
        )

        # Trigger the main cloud pipeline
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
        success, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        if not success:
            raise RuntimeError("Could not encode frame as JPEG")
        return buffer.tobytes()

    def _frame_key(self, batch, captured_frame):
        frame_time = captured_frame.captured_at_utc.replace(":", "").replace(".", "-")
        return f"{self.prefix}/seq_{captured_frame.sequence_id:012d}_batch_{captured_frame.batch_id:08d}_idx_{captured_frame.batch_index:02d}_{frame_time}.jpg"

    def _manifest_key(self, batch):
        batch_time = batch.created_at_utc.replace(":", "").replace(".", "-")
        return f"{self.prefix}/manifest_batch_{batch.batch_id:08d}_{batch_time}.json"

    def _create_manifest(self, batch, uploaded_frames, manifest_key):
        """Creates the JSON payload structure expected by the API Gateway Lambda."""
        return {
            "run_id": batch.run_id,
            "batch_id": batch.batch_id,
            "created_at_epoch": batch.created_at_epoch,
            "created_at_utc": batch.created_at_utc,
            "bucket": self.bucket_name,
            "prefix": self.prefix,
            "manifest_key": manifest_key,
            "images": [frame.key for frame in uploaded_frames] # Simplified for API
        }

    def _send_confirmation(self, manifest):
        """Pushes the manifest via HTTP POST to the API Gateway."""
        logger.info(f"Sending upload confirmation to API Gateway for Batch {manifest['batch_id']}")
        
        headers = {
            "Content-Type": "application/json"
            # "x-api-key": config.API_KEY  <-- Uncomment this if you enabled API Keys
        }
        
        try:
            response = requests.post(
                self.confirmation_url,
                json=manifest,
                headers=headers,
                timeout=10
            )
            response.raise_for_status()
            logger.info(f"✅ Cloud confirmed batch {manifest['batch_id']}. Status: {response.status_code}")
            
        except requests.RequestException as exc:
            logger.error(f"❌ Cloud confirmation failed: {exc}", exc_info=True)
            raise