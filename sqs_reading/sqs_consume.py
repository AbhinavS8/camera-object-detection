import argparse
import json
import logging
import os
from typing import Any

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"), override=True)

DEFAULT_QUEUE_URL = (
    "https://sqs.us-east-1.amazonaws.com/794562053797/detectToTrackQueue"
)
DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "downloaded_results"
)

logger = logging.getLogger(__name__)


def get_env_value(name, default=None):
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default

    return value


def get_aws_region(region_name=None):
    return (
        region_name or
        get_env_value("AWS_REGION") or
        get_env_value("AWS_DEFAULT_REGION") or
        "us-east-1"
    )


def get_queue_url(queue_url=None):
    return (
        queue_url or
        get_env_value("SQS_QUEUE_URL") or
        DEFAULT_QUEUE_URL
    )


def create_sqs_client(region_name=None):
    return boto3.client("sqs", region_name=get_aws_region(region_name))


def create_s3_client(region_name=None):
    return boto3.client("s3", region_name=get_aws_region(region_name))


def parse_message_body(body):
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return body


def receive_sqs_messages(
    client,
    queue_url,
    max_number=10,
    wait_time=20,
    visibility_timeout=None
):
    """
    Receive a batch of messages from an SQS queue using long polling.
    """
    request = {
        "QueueUrl": queue_url,
        "MaxNumberOfMessages": max_number,
        "WaitTimeSeconds": wait_time,
        "MessageAttributeNames": ["All"],
        "AttributeNames": ["All"]
    }

    if visibility_timeout is not None:
        request["VisibilityTimeout"] = visibility_timeout

    try:
        response = client.receive_message(**request)
    except ClientError:
        logger.exception("Could not receive messages from %s", queue_url)
        raise

    return response.get("Messages", [])


def delete_sqs_message(client, queue_url, receipt_handle):
    try:
        client.delete_message(
            QueueUrl=queue_url,
            ReceiptHandle=receipt_handle
        )
    except ClientError:
        logger.exception("Could not delete message from %s", queue_url)
        raise


def local_result_path(result_key, output_dir):
    clean_key = result_key.lstrip("/")
    return os.path.join(output_dir, clean_key)


def download_result_json(s3_client, message_body, output_dir):
    if not isinstance(message_body, dict):
        raise ValueError("Message body must be a JSON object.")

    bucket = message_body.get("bucket")
    result_key = message_body.get("result_key")

    if not bucket:
        raise ValueError("Message body is missing 'bucket'.")

    if not result_key:
        raise ValueError("Message body is missing 'result_key'.")

    if not result_key.lower().endswith(".json"):
        raise ValueError(f"result_key is not a JSON file: {result_key}")

    destination = local_result_path(result_key, output_dir)
    os.makedirs(os.path.dirname(destination), exist_ok=True)

    try:
        s3_client.download_file(bucket, result_key, destination)
    except ClientError:
        logger.exception(
            "Could not download s3://%s/%s",
            bucket,
            result_key
        )
        raise

    return destination


def format_message(message) -> dict[str, Any]:
    return {
        "message_id": message.get("MessageId"),
        "body": parse_message_body(message.get("Body", "")),
        "attributes": message.get("Attributes", {}),
        "message_attributes": message.get("MessageAttributes", {})
    }


def consume_messages(
    sqs_client,
    s3_client,
    queue_url,
    output_dir,
    max_number=10,
    wait_time=20,
    visibility_timeout=None,
    delete_after_receive=False,
    once=False
):
    while True:
        messages = receive_sqs_messages(
            client=sqs_client,
            queue_url=queue_url,
            max_number=max_number,
            wait_time=wait_time,
            visibility_timeout=visibility_timeout
        )

        if not messages:
            logger.info("No messages available.")
            if once:
                return
            continue

        for message in messages:
            formatted = format_message(message)
            logger.info(
                "Received message %s",
                formatted["message_id"]
            )
            print(json.dumps(formatted, indent=2, default=str))

            try:
                download_path = download_result_json(
                    s3_client,
                    formatted["body"],
                    output_dir
                )
            except ValueError as exc:
                logger.warning(
                    "Skipping malformed message %s: %s",
                    formatted["message_id"],
                    exc
                )
                continue

            logger.info("Downloaded result JSON to %s", download_path)

            if delete_after_receive:
                delete_sqs_message(
                    sqs_client,
                    queue_url,
                    message["ReceiptHandle"]
                )
                logger.info(
                    "Deleted message %s",
                    formatted["message_id"]
                )

        if once:
            return


def parse_args():
    parser = argparse.ArgumentParser(
        description="Receive messages from the detect-to-track SQS queue."
    )
    parser.add_argument("--queue-url", default=get_queue_url())
    parser.add_argument("--region", default=get_aws_region())
    parser.add_argument("--max-number", type=int, default=10)
    parser.add_argument("--wait-time", type=int, default=20)
    parser.add_argument("--visibility-timeout", type=int)
    parser.add_argument(
        "--output-dir",
        default=get_env_value("SQS_RESULT_OUTPUT_DIR", DEFAULT_OUTPUT_DIR),
        help="Local directory where downloaded result JSON files are stored."
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete each message after its result JSON is downloaded."
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Receive one batch and exit."
    )

    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )
    args = parse_args()
    sqs_client = create_sqs_client(args.region)
    s3_client = create_s3_client(args.region)

    logger.info("Listening to SQS queue: %s", args.queue_url)
    consume_messages(
        sqs_client=sqs_client,
        s3_client=s3_client,
        queue_url=args.queue_url,
        output_dir=args.output_dir,
        max_number=args.max_number,
        wait_time=args.wait_time,
        visibility_timeout=args.visibility_timeout,
        delete_after_receive=args.delete,
        once=args.once
    )


if __name__ == "__main__":
    main()


# message format
# {"bucket": "image-manager1", "image_key": "input/8_jpeg.rf.KSmwWjHCMpntaxFrLSLB.jpeg", "result_key": "output/8_jpeg.rf.KSmwWjHCMpntaxFrLSLB.json"}
