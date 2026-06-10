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
    state_path = Path(state_path)

    if not state_path.exists():
        return []

    with state_path.open("r", encoding="utf-8") as state_file:
        state = json.load(state_file)

    return state.get("batches", [])


def save_ordered_batches(batches, state_path=DEFAULT_STATE_PATH):
    state_path = Path(state_path)
    state_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "batches": sorted(batches, key=batch_sort_key)
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


def upsert_batch(batch, state_path=DEFAULT_STATE_PATH):
    batches = load_ordered_batches(state_path)
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

    save_ordered_batches(batches, state_path)
    return sorted(batches, key=batch_sort_key)


def receive_messages(client, queue_url, max_messages=10, wait_time=20):
    response = client.receive_message(
        QueueUrl=queue_url,
        MaxNumberOfMessages=max_messages,
        WaitTimeSeconds=wait_time,
        MessageAttributeNames=["All"],
        AttributeNames=["All"]
    )
    return response.get("Messages", [])


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
    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    )
    args = parse_args()

    poll_queue(
        queue_url=args.queue_url,
        state_path=args.state_path,
        region_name=args.region,
        once=args.once
    )


if __name__ == "__main__":
    main()
