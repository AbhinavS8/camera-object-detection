import json
import boto3
import os

print("Loading function...")

sqs_client = boto3.client("sqs")
s3_client = boto3.client("s3")
rekognition_client = boto3.client("rekognition")

QUEUE_URL = os.environ.get(
    "QUEUE_URL",
)

PROJECT_VERSION_ARN = os.environ.get(
    "MODEL_ARN"
)


def output_key_for_image(image_key):
    output_key = image_key.replace("input/", "output/", 1)
    return output_key.rsplit(".", 1)[0] + ".json"


def lambda_handler(event, context):
    results = []

    for record in event.get("Records", []):
        try:
            body = record.get("body", "{}")
            message = json.loads(body) if isinstance(body, str) else body

            batch_id = message["batch_id"]
            bucket = message["bucket"]
            ordered_frames = message["ordered_frames"]

            batch_results = []

            for batch_index, image_key in enumerate(ordered_frames):
                if not image_key.startswith("input/"):
                    print(f"Ignored: {image_key} is not in the input/ folder.")
                    batch_results.append({
                        "batch_index": batch_index,
                        "image_key": image_key,
                        "result_key": None,
                        "status": "ignored"
                    })
                    continue

                print(f"Analyzing batch={batch_id} index={batch_index} {bucket}/{image_key}...")

                response = rekognition_client.detect_custom_labels(
                    ProjectVersionArn=PROJECT_VERSION_ARN,
                    Image={
                        "S3Object": {
                            "Bucket": bucket,
                            "Name": image_key
                        }
                    },
                    MinConfidence=20
                )

                output_key = output_key_for_image(image_key)

                result_json = {
                    "batch_id": batch_id,
                    "batch_index": batch_index,
                    "bucket": bucket,
                    "image_key": image_key,
                    "result_key": output_key,
                    "rekognition_response": response
                }

                print(f"Saving results to {bucket}/{output_key}...")

                s3_client.put_object(
                    Bucket=bucket,
                    Key=output_key,
                    Body=json.dumps(result_json, indent=4),
                    ContentType="application/json"
                )

                batch_results.append({
                    "batch_index": batch_index,
                    "image_key": image_key,
                    "result_key": output_key,
                    "status": "success"
                })

            next_message = {
                "batch_id": batch_id,
                "bucket": bucket,
                "results": batch_results
            }

            print(f"Sending batch result to SQS for batch={batch_id}...")

            sqs_client.send_message(
                QueueUrl=QUEUE_URL,
                MessageBody=json.dumps(next_message)
            )

            results.append({
                "batch_id": batch_id,
                "bucket": bucket,
                "frame_count": len(ordered_frames),
                "results": batch_results,
                "status": "success"
            })

        except Exception as exc:
            print(f"Error processing record: {record}")
            print(f"Exception: {exc}")
            raise

    return {
        "statusCode": 200,
        "body": json.dumps(results)
    }