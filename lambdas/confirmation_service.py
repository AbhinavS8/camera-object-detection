import json
import logging
import os
import boto3
from botocore.exceptions import ClientError
import uuid

def lambda_handler(event, context):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    body = event.get("body")
    if body is None:
        return {"statusCode": 400, "body": json.dumps({"error": "Missing request body"})}

    try:
        payload = json.loads(body) if isinstance(body, str) else body
    except json.JSONDecodeError:
        return {"statusCode": 400, "body": json.dumps({"error": "Invalid JSON body"})}

    bucket = payload.get("bucket")
    images = payload.get("images") or payload.get("image_keys")

    if not bucket or not isinstance(images, list):
        return {"statusCode": 400, "body": json.dumps({"error": "Invalid bucket or images list"})}

    queue_url = os.getenv("SQS_QUEUE_URL")
    if not queue_url:
        return {"statusCode": 500, "body": json.dumps({"error": "SQS queue URL not configured"})}

    s3 = boto3.client("s3")
    sqs = boto3.client("sqs")

    accepted = []
    missing = []

    # 1. ONLY VERIFY IN THE LOOP (Do not send to SQS yet)
    for key in images:
        try:
            s3.head_object(Bucket=bucket, Key=key)
            accepted.append(key)
        except ClientError as exc:
            code = getattr(exc, "response", {}).get("Error", {}).get("Code")
            if code in ("404", "NoSuchKey"):
                missing.append(key)
            else:
                logger.exception("Error checking S3 object %s/%s", bucket, key)
                return {"statusCode": 500, "body": json.dumps({"error": "Error accessing S3"})}

    # 2. SEND ONE MASTER MESSAGE TO SQS
    if accepted:
        try:
            # We bundle the entire verified array into ONE message
            manifest_message = {
                "batch_id": payload.get("batch_id"),
                "bucket": bucket,
                "ordered_frames": accepted 
            }
            
            # Note: If using a FIFO queue, you still need MessageGroupId!
            # If using a Standard Queue, this single message is perfectly safe 
            # because the list inside it is strictly ordered.
            sqs.send_message(
                QueueUrl=queue_url, 
                MessageBody=json.dumps(manifest_message)
            )
            logger.info(f"Successfully enqueued batch of {len(accepted)} frames.")
            
        except ClientError:
            logger.exception("Failed to enqueue manifest batch")
            return {"statusCode": 500, "body": json.dumps({"error": "Failed to send SQS message"})}

    return {"statusCode": 200, "body": json.dumps({"accepted": accepted, "missing": missing})}