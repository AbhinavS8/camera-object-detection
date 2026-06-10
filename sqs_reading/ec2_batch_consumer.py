import argparse
import json
import logging
import os
import time
from pathlib import Path

import boto3
import cv2
import numpy as np
from botocore.exceptions import ClientError
from dotenv import load_dotenv

try:
    from sqs_reading.ec2_bytetrack_helpers import (
        create_tracker,
        labels_to_tracker_detections,
        update_tracker,
        update_entry_exit_state,
        remove_lost_objects,
        remove_stale_tracks,
    )
except ImportError:
    from ec2_bytetrack_helpers import (
        create_tracker,
        labels_to_tracker_detections,
        update_tracker,
        update_entry_exit_state,
        remove_lost_objects,
        remove_stale_tracks,
    )


PROJECT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_DIR / ".env", override=True)

DEFAULT_QUEUE_URL = (
    "https://sqs.us-east-1.amazonaws.com/794562053797/detectToTrackQueue"
)
DEFAULT_STATE_PATH = Path(__file__).resolve().parent / "ordered_batch_results.json"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output"
DEFAULT_RESULTS_KEY = "results/ordered_batch_results.json"

logger = logging.getLogger(__name__)


def get_aws_region(region_name=None):
    return (
        region_name
        or os.getenv("AWS_REGION")
        or os.getenv("AWS_DEFAULT_REGION")
        or "us-east-1"
    )


def get_queue_url(queue_url=None):
    return (
        queue_url
        or os.getenv("DETECT_TO_TRACK_QUEUE_URL")
        or os.getenv("SQS_QUEUE_URL")
        or DEFAULT_QUEUE_URL
    )


def create_sqs_client(region_name=None):
    return boto3.client("sqs", region_name=get_aws_region(region_name))


def create_s3_client(region_name=None):
    return boto3.client("s3", region_name=get_aws_region(region_name))


def load_state(state_path=DEFAULT_STATE_PATH):
    state_path = Path(state_path)

    if not state_path.exists():
        return {"next_batch_id": None, "batches": []}

    with state_path.open("r", encoding="utf-8") as state_file:
        state = json.load(state_file)

    return {
        "next_batch_id": state.get("next_batch_id"),
        "batches": state.get("batches", []),
    }


def save_state(state, state_path=DEFAULT_STATE_PATH):
    state_path = Path(state_path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def reset_batch_state(state_path=DEFAULT_STATE_PATH):
    state = load_state(state_path)
    state["next_batch_id"] = "00000000"
    save_state(state, state_path)
    return state


def batch_sort_key(batch):
    batch_id = str(batch["batch_id"])
    if batch_id.isdigit():
        return (0, int(batch_id))
    return (1, batch_id)


def increment_batch_id(batch_id):
    batch_id = str(batch_id)
    if not batch_id.isdigit():
        return None
    return str(int(batch_id) + 1).zfill(len(batch_id))


def consume_next_batch(state_path=DEFAULT_STATE_PATH):
    state = load_state(state_path)
    batches = sorted(state["batches"], key=batch_sort_key)

    if not batches:
        return None

    next_batch_id = state.get("next_batch_id")
    consume_index = None

    if next_batch_id is None:
        consume_index = 0
    else:
        for index, batch in enumerate(batches):
            if str(batch.get("batch_id")) == str(next_batch_id):
                consume_index = index
                break

    if consume_index is None:
        return None

    batch = batches.pop(consume_index)
    state["batches"] = batches
    state["next_batch_id"] = increment_batch_id(batch["batch_id"])
    save_state(state, state_path)
    return batch


def download_json_from_s3(bucket, key, s3_client=None, region_name=None):
    s3 = s3_client or create_s3_client(region_name)
    try:
        response = s3.get_object(Bucket=bucket, Key=key)
    except ClientError:
        logger.exception("Failed to get s3://%s/%s", bucket, key)
        raise

    body = response["Body"].read()
    return json.loads(body.decode("utf-8"))


def download_image_from_s3(bucket, key, s3_client=None, region_name=None):
    s3 = s3_client or create_s3_client(region_name)
    try:
        response = s3.get_object(Bucket=bucket, Key=key)
    except ClientError:
        logger.exception("Failed to get s3://%s/%s", bucket, key)
        raise

    body = response["Body"].read()
    arr = np.frombuffer(body, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"Could not decode image s3://{bucket}/{key}")
    return frame


def upload_json_to_s3(bucket, key, payload, s3_client=None, region_name=None):
    s3 = s3_client or create_s3_client(region_name)
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=(json.dumps(payload, indent=2) + "\n").encode("utf-8"),
        ContentType="application/json",
    )


def _default_output_path(batch_id):
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return DEFAULT_OUTPUT_DIR / f"batch_{batch_id}_counts.json"


def consume_and_track_batches(state_path=DEFAULT_STATE_PATH, region_name=None, stop_when_empty=False):
    s3 = create_s3_client(region_name)
    tracker = create_tracker()
    tracked_objects = {}
    counts = {"in": 0, "out": 0}
    state = reset_batch_state(state_path)

    while True:
        batch = consume_next_batch(state_path=state_path)
        if batch is None:
            if stop_when_empty:
                return
            time.sleep(1)
            continue

        batch_results = []
        for entry in batch.get("results", []):
            image_key = entry.get("image_key") or entry.get("imageKey")
            result_key = entry.get("result_key") or entry.get("resultKey") or entry.get("manifest_key")
            custom_labels = []

            if result_key:
                try:
                    payload = download_json_from_s3(batch["bucket"], result_key, s3_client=s3)
                    custom_labels = payload.get("CustomLabels", [])
                except Exception:
                    logger.exception("Could not load result JSON for batch %s key %s", batch.get("batch_id"), result_key)

            tracks = []
            if image_key:
                try:
                    frame = download_image_from_s3(batch["bucket"], image_key, s3_client=s3)
                    h, w = frame.shape[:2]
                    detections = labels_to_tracker_detections(custom_labels, w, h)
                    tracks = update_tracker(tracker, detections, frame)
                    removed_ids = remove_lost_objects(tracked_objects, counts)
                    tracks = remove_stale_tracks(tracks, removed_ids)
                    update_entry_exit_state(tracks, tracked_objects, w, h, counts)
                except Exception:
                    logger.exception("Could not process batch %s image %s", batch.get("batch_id"), image_key)

            batch_results.append({
                "image_key": image_key,
                "result_key": result_key,
                "tracks": tracks.tolist() if hasattr(tracks, "tolist") else [],
                "detections": custom_labels,
            })

        output_path = _default_output_path(batch["batch_id"])
        output_payload = {
            "batch_id": batch["batch_id"],
            "processed_at_epoch": time.time(),
            "counts": counts,
            "results": batch_results,
        }
        output_path.write_text(json.dumps(output_payload, indent=2) + "\n", encoding="utf-8")
        upload_json_to_s3(
            batch["bucket"],
            DEFAULT_RESULTS_KEY,
            output_payload,
            s3_client=s3,
        )
        print(f"Processed batch {batch['batch_id']}: IN={counts['in']} OUT={counts['out']}")


def parse_args():
    parser = argparse.ArgumentParser(description="Consume ordered batch results and run ByteTrack analysis on EC2.")
    parser.add_argument("--state-path", default=str(DEFAULT_STATE_PATH))
    parser.add_argument("--region", default=get_aws_region())
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
    )
    args = parse_args()
    consume_and_track_batches(state_path=args.state_path, region_name=args.region, stop_when_empty=args.once)


if __name__ == "__main__":
    main()