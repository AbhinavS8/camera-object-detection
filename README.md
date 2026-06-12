# Camera Object Detection & Tracking Pipeline

A distributed object detection and tracking pipeline built on AWS using S3, Lambda, SQS, and EC2.

The system uploads batches of frames from an RTSP camera to Amazon S3, performs object detection using AWS Lambda and Rekognition, then forwards detection results to a tracking service running on an EC2 instance for counting and line-crossing analysis.

---

## Architecture Overview

<img width="1613" height="975" alt="image" src="https://github.com/user-attachments/assets/3bf028a0-6c22-4f51-8420-da023f10c145" />


---

## Repository Structure

```text
.
├── core/
│   ├── __init__.py
│   ├── config.py
│   ├── frame_batcher.py
│   └── s3_batch_uploader.py
│
├── lambdas/
│   ├── confirmation_service.py
│   └── detection_service.py
│
├── sqs_reading/
│   ├── ec2_batch_consumer.py
│   ├── ec2_sqs_buffer.py
│   └── ordered_batch_results.json
│
├── .env.example
├── .gitignore
├── camera_test.py
└── main.py
```

---

## Components

### core/

Helper modules used by the uploader.

### main.py

Entry point for the uploader.

Responsibilities:

- collect frames
- create batches
- upload batches to S3
- notify backend services after upload

Used primarily for testing and development.

---

### camera_test.py

Standalone RTSP camera connectivity test.

Used to verify:
- RTSP stream access
- OpenCV capture
- camera availability

---

## Lambda Services

### confirmation_service.py

Triggered by API Gateway.

The uploader calls this endpoint whenever a batch upload completes successfully.

Responsibilities:

1. Validate upload notification
2. Create processing message
3. Push batch metadata into the Detection SQS Queue

```text
Uploader
    ↓
API Gateway
    ↓
confirmation_service
    ↓
Detection Queue
```

---

### detection_service.py

Triggered by messages arriving in the Detection SQS Queue.

Responsibilities:

1. Retrieve uploaded batch information
2. Perform object detection
3. Generate bounding box results
4. Forward detection results to the Tracking Queue

```text
Detection Queue
     ↓
detection_service
     ↓
Tracking Queue
```

---

## EC2 Tracking Services

The tracking subsystem runs as long-running processes on an EC2 instance. Uses EC2 instead of Lambda to persist tracking state over time.

### ec2_sqs_buffer.py

Consumes messages from the Tracking Queue.

Responsibilities:

- receive detection results
- maintain ordering
- buffer incoming batches
- store intermediate tracking data locally

This acts as the ingestion layer for the tracker.


### ec2_batch_consumer.py

Consumes ordered batches and performs tracking logic.

Responsibilities:

- process detection batches
- maintain tracking state
- perform object counting
- detect line-crossing events
- generate final tracking results
- upload result JSON files back to S3



---


## Configuration

Copy the example environment file:

```bash
cp .env.example .env
```

Configure:

```env

# AWS / S3 frame batching
AWS_REGION=...
S3_UPLOAD_CONFIRMATION_URL = ..
DEFAULT_QUEUE_URL=..


```

---

## Running the Uploader

```bash
python main.py
```

---

## Testing Camera Connectivity

```bash
python camera_test.py
```

---

## Deployment Notes

The repository assumes:

- S3 buckets already exist
- SQS queues are configured
- Lambda functions are deployed
- API Gateway endpoint is configured
- EC2 tracking services are installed and running
- permissions are given appropriately

EC2 tracking processes are intended to run as background services (e.g. systemd).

