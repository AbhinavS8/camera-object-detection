import argparse
import json
import logging
import os
import time
from pathlib import Path
from tempfile import NamedTemporaryFile

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv


PROJECT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_DIR / ".env", override=True)

DEFAULT_QUEUE_URL = (
    "https://sqs.us-east-1.amazonaws.com/794562053797/detectToTrackQueue"
)
DEFAULT_STATE_PATH = Path(__file__).resolve().parent / "ordered_batch_results.json"
DEFAULT_FORCE_CONSUME_AFTER_BATCHES = 80

logger = logging.getLogger(__name__)

import cv2
import numpy as np

from video_rekognition_feed import (
    create_tracker,
    labels_to_tracker_detections,
    update_tracker,
    update_entry_exit_state,
    remove_lost_objects,
    remove_stale_tracks,
)


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
    """Create a boto3 S3 client using the configured AWS region."""
    return boto3.client("s3", region_name=get_aws_region(region_name))


def parse_message_body(body):
    return json.loads(body) if isinstance(body, str) else body


def batch_sort_key(batch):
    batch_id = str(batch["batch_id"])

    if batch_id.isdigit():
        return (0, int(batch_id))

    return (1, batch_id)


def normalize_batch_message(message):
    batch_id = str(message["batch_id"])
    bucket = message["bucket"]
    results = message["results"]

    ordered_results = sorted(
        results,
        key=lambda result: int(result.get("batch_index", 0))
    )

    return {
        "batch_id": batch_id,
        "bucket": bucket,
        "results": ordered_results
    }


def load_ordered_batches(state_path=DEFAULT_STATE_PATH):
    return load_state(state_path).get("batches", [])


def load_state(state_path=DEFAULT_STATE_PATH):
    state_path = Path(state_path)

    if not state_path.exists():
        return {
            "next_batch_id": None,
            "batches": []
        }

    with state_path.open("r", encoding="utf-8") as state_file:
        state = json.load(state_file)

    return {
        "next_batch_id": state.get("next_batch_id"),
        "batches": state.get("batches", [])
    }


def save_state(state, state_path=DEFAULT_STATE_PATH):
    state_path = Path(state_path)
    state_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "next_batch_id": state.get("next_batch_id"),
        "batches": sorted(state.get("batches", []), key=batch_sort_key)
    }

    with NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=state_path.parent,
        delete=False
    ) as temp_file:
        json.dump(payload, temp_file, indent=2)
        temp_file.write("\n")
        temp_path = Path(temp_file.name)

    temp_path.replace(state_path)


def save_ordered_batches(batches, state_path=DEFAULT_STATE_PATH):
    state = load_state(state_path)
    state["batches"] = batches
    save_state(state, state_path)


def upsert_batch(batch, state_path=DEFAULT_STATE_PATH):
    state = load_state(state_path)
    batches = state["batches"]
    existing_index = next(
        (
            index
            for index, existing in enumerate(batches)
            if str(existing.get("batch_id")) == batch["batch_id"]
        ),
        None
    )

    if existing_index is None:
        batches.append(batch)
    else:
        batches[existing_index] = batch

    state["batches"] = batches
    save_state(state, state_path)
    return sorted(batches, key=batch_sort_key)


def increment_batch_id(batch_id):
    batch_id = str(batch_id)

    if not batch_id.isdigit():
        return None

    return str(int(batch_id) + 1).zfill(len(batch_id))


def consume_next_batch(
    state_path=DEFAULT_STATE_PATH,
    force_after_batches=DEFAULT_FORCE_CONSUME_AFTER_BATCHES
):
    state = load_state(state_path)
    batches = sorted(state["batches"], key=batch_sort_key)

    if not batches:
        return None

    next_batch_id = state.get("next_batch_id")
    consume_index = None
    forced = False

    if next_batch_id is None:
        consume_index = 0
    else:
        for index, batch in enumerate(batches):
            if str(batch.get("batch_id")) == str(next_batch_id):
                consume_index = index
                break

    if consume_index is None and len(batches) > force_after_batches:
        consume_index = 0
        forced = True

    if consume_index is None:
        logger.info(
            "Waiting for batch %s. Local backlog is %s/%s batches.",
            next_batch_id,
            len(batches),
            force_after_batches
        )
        return None

    batch = batches.pop(consume_index)
    state["batches"] = batches
    state["next_batch_id"] = increment_batch_id(batch["batch_id"])
    save_state(state, state_path)

    if forced:
        logger.warning(
            "Force-consuming batch %s because local backlog exceeded %s batches.",
            batch["batch_id"],
            force_after_batches
        )
    else:
        logger.info("Consumed batch %s", batch["batch_id"])

    return batch


def receive_messages(client, queue_url, max_messages=10, wait_time=20):
    response = client.receive_message(
        QueueUrl=queue_url,
        MaxNumberOfMessages=max_messages,
        WaitTimeSeconds=wait_time,
        MessageAttributeNames=["All"],
        AttributeNames=["All"]
    )
    return response.get("Messages", [])


def download_json_from_s3(bucket, key, s3_client=None, region_name=None):
    """Download a JSON object from S3 and return the parsed payload.

    Raises ClientError on S3 failures and ValueError on JSON decode errors.
    """
    s3 = s3_client or create_s3_client(region_name)

    try:
        response = s3.get_object(Bucket=bucket, Key=key)
    except ClientError:
        logger.exception("Failed to get s3://%s/%s", bucket, key)
        raise

    body = response["Body"].read()

    try:
        return json.loads(body.decode("utf-8"))
    except Exception:
        logger.exception("Could not decode JSON from s3://%s/%s", bucket, key)
        raise


def download_image_from_s3(bucket, key, s3_client=None, region_name=None):
    """Download an image from S3 and return it as a BGR numpy array (cv2 image)."""
    s3 = s3_client or create_s3_client(region_name)

    try:
        response = s3.get_object(Bucket=bucket, Key=key)
    except ClientError:
        logger.exception("Failed to get s3://%s/%s", bucket, key)
        raise

    body = response["Body"].read()
    arr = np.frombuffer(body, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if img is None:
        logger.error("Could not decode image s3://%s/%s", bucket, key)
        raise RuntimeError(f"Could not decode image s3://{bucket}/{key}")

    return img


def consume_ordered_queue(
    state_path=DEFAULT_STATE_PATH,
    process_fn=None,
    s3_client=None,
    region_name=None,
    idle_sleep_seconds=1,
    stop_when_empty=False,
):
    """Consume ordered local batches and for each batch download result JSONs from S3.

    - `process_fn(batch, results)` is called for each consumed batch where `results`
      is a list of parsed JSON objects corresponding to each result entry for the batch.
    - The function uses `consume_next_batch` which updates `next_batch_id` in state
      to maintain ordering across restarts.
    - If a result entry contains one of `result_key`, `resultKey`, or `manifest_key`,
      that key is used to download a JSON object from the batch `bucket`.
    """
    if process_fn is None:
        raise ValueError("process_fn must be provided")

    s3 = s3_client or create_s3_client(region_name)

    while True:
        batch = consume_next_batch(state_path=state_path)

        if batch is None:
            if stop_when_empty:
                return
            time.sleep(idle_sleep_seconds)
            continue

        bucket = batch.get("bucket")
        downloaded = []

        for entry in batch.get("results", []):
            # support a few possible key names
            key = entry.get("result_key") or entry.get("resultKey") or entry.get("manifest_key")
            if not key:
                logger.warning("No result key found for batch %s entry %s", batch.get("batch_id"), entry)
                downloaded.append(None)
                continue

            try:
                payload = download_json_from_s3(bucket, key, s3_client=s3)
            except Exception:
                logger.exception("Failed to download/parse s3 object for batch %s key %s", batch.get("batch_id"), key)
                payload = None

            downloaded.append(payload)

        try:
            process_fn(batch, downloaded)
        except Exception:
            logger.exception("process_fn raised an exception for batch %s", batch.get("batch_id"))
            raise


def _default_output_path(batch_id):
    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"batch_{batch_id}_counts.json"


def consume_and_track_batches(
    state_path=DEFAULT_STATE_PATH,
    region_name=None,
    s3_client=None,
    stop_when_empty=False,
    output_path_fn=None
):
    """Consume ordered batches and process them with ByteTrack tracking.

    Writes a per-batch JSON containing updated counts and prints the counts.
    """
    s3 = s3_client or create_s3_client(region_name)

    tracker = create_tracker()
    tracked_objects = {}
    counts = {"in": 0, "out": 0}

    def process_fn(batch, downloaded_results):
        batch_id = batch.get("batch_id")
        batch_results = []

        for entry, payload in zip(batch.get("results", []), downloaded_results):
            image_key = entry.get("image_key") or entry.get("imageKey")
            result_key = entry.get("result_key") or entry.get("resultKey") or entry.get("manifest_key")

            frame = None
            if image_key:
                try:
                    frame = download_image_from_s3(batch.get("bucket"), image_key, s3_client=s3)
                except Exception:
                    logger.exception("Failed to download image for batch %s key %s", batch_id, image_key)
                    frame = None

            if payload is None:
                custom_labels = []
            else:
                custom_labels = payload.get("CustomLabels", [])

            if frame is None:
                tracks = []
            else:
                h, w = frame.shape[:2]
                detections = labels_to_tracker_detections(custom_labels, w, h)
                tracks = update_tracker(tracker, detections, frame)

                removed_ids = remove_lost_objects(tracked_objects, counts)
                tracks = remove_stale_tracks(tracks, removed_ids)
                update_entry_exit_state(tracks, tracked_objects, w, h, counts)

            batch_results.append({
                "image_key": image_key,
                "result_key": result_key,
                "detections": custom_labels,
                "tracks": tracks.tolist() if hasattr(tracks, "tolist") else []
            })

        out_path = (output_path_fn(batch_id) if output_path_fn else _default_output_path(batch_id))
        payload = {
            "batch_id": batch_id,
            "processed_at_epoch": time.time(),
            "counts": counts,
            "results": batch_results
        }

        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        print(f"Processed batch {batch_id}: IN={counts['in']} OUT={counts['out']}")

    consume_ordered_queue(
        state_path=state_path,
        process_fn=process_fn,
        s3_client=s3,
        region_name=region_name,
        stop_when_empty=stop_when_empty
    )


def delete_message(client, queue_url, receipt_handle):
    client.delete_message(
        QueueUrl=queue_url,
        ReceiptHandle=receipt_handle
    )


def process_sqs_message(client, queue_url, sqs_message, state_path):
    body = parse_message_body(sqs_message.get("Body", sqs_message.get("body", "{}")))
    batch = normalize_batch_message(body)
    batches = upsert_batch(batch, state_path)

    delete_message(
        client,
        queue_url,
        sqs_message["ReceiptHandle"]
    )

    logger.info(
        "Stored batch %s with %s result(s). Local ordered batch count: %s",
        batch["batch_id"],
        len(batch["results"]),
        len(batches)
    )

    return batch


def poll_queue(
    queue_url,
    state_path,
    region_name=None,
    once=False,
    max_messages=10,
    wait_time=20,
    idle_sleep_seconds=2
):
    client = create_sqs_client(region_name)

    while True:
        try:
            messages = receive_messages(
                client,
                queue_url,
                max_messages=max_messages,
                wait_time=wait_time
            )

            if not messages:
                logger.info("No messages received from %s", queue_url)
                if once:
                    return

                time.sleep(idle_sleep_seconds)
                continue

            for message in messages:
                process_sqs_message(client, queue_url, message, state_path)

            if once:
                return

        except ClientError:
            logger.exception("AWS error while reading from %s", queue_url)
            raise
        except Exception:
            logger.exception("Could not process SQS message")
            raise


def parse_args():
    parser = argparse.ArgumentParser(
        description="Receive batch Rekognition result messages and store them locally in order."
    )
    parser.add_argument(
        "--queue-url",
        default=get_queue_url(),
        help="SQS queue URL to consume from."
    )
    parser.add_argument(
        "--state-path",
        default=str(DEFAULT_STATE_PATH),
        help="Local JSON file used as the ordered batch buffer."
    )
    parser.add_argument(
        "--region",
        default=get_aws_region(),
        help="AWS region for the SQS client."
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Poll once and exit instead of running continuously."
    )
    parser.add_argument(
        "--consume-next",
        action="store_true",
        help="Pop and print the next locally buffered batch for the tracking service."
    )
    parser.add_argument(
        "--consume-and-track",
        action="store_true",
        help="Consume ordered batches and run ByteTrack analysis, writing per-batch JSONs."
    )
    parser.add_argument(
        "--force-after-batches",
        type=int,
        default=DEFAULT_FORCE_CONSUME_AFTER_BATCHES,
        help="Consume the oldest available batch when the local backlog exceeds this size."
    )
    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    )
    args = parse_args()

    if args.consume_next:
        batch = consume_next_batch(
            state_path=args.state_path,
            force_after_batches=args.force_after_batches
        )
        if batch is not None:
            print(json.dumps(batch, indent=2))
        return

    if args.consume_and_track:
        consume_and_track_batches(
            state_path=args.state_path,
            region_name=args.region,
            stop_when_empty=args.once
        )
        return

    poll_queue(
        queue_url=args.queue_url,
        state_path=args.state_path,
        region_name=args.region,
        once=args.once
    )


if __name__ == "__main__":
    main()
