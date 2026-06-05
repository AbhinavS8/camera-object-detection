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


def format_message(message) -> dict[str, Any]:
    return {
        "message_id": message.get("MessageId"),
        "body": parse_message_body(message.get("Body", "")),
        "attributes": message.get("Attributes", {}),
        "message_attributes": message.get("MessageAttributes", {})
    }


def consume_messages(
    client,
    queue_url,
    max_number=10,
    wait_time=20,
    visibility_timeout=None,
    delete_after_receive=False,
    once=False
):
    while True:
        messages = receive_sqs_messages(
            client=client,
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

            if delete_after_receive:
                delete_sqs_message(
                    client,
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
        "--delete",
        action="store_true",
        help="Delete each message after it is printed."
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
    client = create_sqs_client(args.region)

    logger.info("Listening to SQS queue: %s", args.queue_url)
    consume_messages(
        client=client,
        queue_url=args.queue_url,
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