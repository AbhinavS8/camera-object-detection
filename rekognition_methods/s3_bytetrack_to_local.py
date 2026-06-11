import argparse
import json
import os
import time
from pathlib import Path

import boto3
import cv2
import numpy as np
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from types import SimpleNamespace
from ultralytics.trackers.byte_tracker import BYTETracker


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
load_dotenv(PROJECT_DIR / ".env", override=True)

DEFAULT_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
S3_BUCKET_ENV = "S3_FRAME_BUCKET"
AWS_REGION_ENV = "AWS_REGION"


class ArrayAdapter:
    def __init__(self, values):
        self.values = values

    def cpu(self):
        return self

    def numpy(self):
        return self.values

    def astype(self, dtype):
        return self.values.astype(dtype)

    def reshape(self, *shape):
        return self.values.reshape(*shape)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, item):
        return self.values[item]

    def __array__(self, dtype=None):
        if dtype is None:
            return self.values

        return self.values.astype(dtype)

    def __ge__(self, other):
        return self.values >= other

    def __gt__(self, other):
        return self.values > other

    def __le__(self, other):
        return self.values <= other

    def __lt__(self, other):
        return self.values < other


class TrackerDetections:
    def __init__(self, xywh, conf, cls):
        self.xywh = ArrayAdapter(xywh)
        self.conf = ArrayAdapter(conf)
        self.cls = ArrayAdapter(cls)

    def cpu(self):
        return self

    def numpy(self):
        return self

    def __len__(self):
        return len(self.conf)

    def __getitem__(self, item):
        if isinstance(item, int):
            return self

        return TrackerDetections(
            self.xywh.values[item],
            self.conf.values[item],
            self.cls.values[item]
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Pull Rekognition JSON files and matching image frames from S3, "
            "run ByteTrack locally, and store local track JSON files."
        )
    )
    parser.add_argument(
        "--bucket",
        default=os.getenv(S3_BUCKET_ENV),
        help=(
            "S3 bucket containing input images and output Rekognition JSONs. "
            f"Defaults to ${S3_BUCKET_ENV}."
        )
    )
    parser.add_argument(
        "--input-prefix",
        default="input",
        help="S3 prefix containing original images."
    )
    parser.add_argument(
        "--detections-prefix",
        default="output",
        help="S3 prefix containing Rekognition JSON files."
    )
    parser.add_argument(
        "--tracks-dir",
        default="tracks",
        help="Local directory where track JSON files will be written."
    )
    parser.add_argument(
        "--region",
        default=os.getenv(AWS_REGION_ENV),
        help=f"AWS region. Defaults to ${AWS_REGION_ENV}."
    )
    parser.add_argument(
        "--frame-rate",
        type=int,
        default=5,
        help="Tracker frame rate. Match your sampled frame rate."
    )
    parser.add_argument(
        "--class-id",
        type=int,
        default=0,
        help="Class ID to assign to Rekognition package detections."
    )
    parser.add_argument(
        "--image-extensions",
        default=",".join(DEFAULT_IMAGE_EXTENSIONS),
        help="Comma-separated image extensions to try for each JSON stem."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of detection JSON files to process."
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Keep polling S3 for new detection JSON files."
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=5.0,
        help="Seconds to wait between S3 polls when --watch is enabled."
    )
    args = parser.parse_args()

    if not args.bucket:
        parser.error(f"Set {S3_BUCKET_ENV} in .env or pass --bucket.")

    return args


def list_detection_keys(s3_client, bucket, detections_prefix):
    keys = []
    prefix = normalize_prefix(detections_prefix)
    paginator = s3_client.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            key = item["Key"]

            if key.lower().endswith(".json"):
                keys.append(key)

    return sorted(keys, key=natural_sort_key)


def read_json_from_s3(s3_client, bucket, key):
    response = s3_client.get_object(Bucket=bucket, Key=key)
    return json.loads(response["Body"].read().decode("utf-8"))


def read_image_from_s3(s3_client, bucket, key):
    response = s3_client.get_object(Bucket=bucket, Key=key)
    image_bytes = response["Body"].read()
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if frame is None:
        raise ValueError(f"Could not decode image from s3://{bucket}/{key}")

    return frame


def create_tracker(frame_rate):
    args = SimpleNamespace(
        track_high_thresh=0.05,
        track_low_thresh=0.001,
        new_track_thresh=0.05,
        track_buffer=80,
        match_thresh=0.3,
        fuse_score=True,
        mot20=False
    )

    try:
        return BYTETracker(args, frame_rate=frame_rate)
    except TypeError:
        return BYTETracker(args)


def labels_to_tracker_detections(custom_labels, frame_width, frame_height, class_id):
    xywh = []
    confidences = []
    class_ids = []

    for label in custom_labels:
        box = label.get("Geometry", {}).get("BoundingBox")

        if not box:
            continue

        left = box["Left"] * frame_width
        top = box["Top"] * frame_height
        width = box["Width"] * frame_width
        height = box["Height"] * frame_height

        xywh.append([
            left + width / 2,
            top + height / 2,
            width,
            height
        ])
        confidences.append(label.get("Confidence", 0) / 100.0)
        class_ids.append(class_id)

    return TrackerDetections(
        np.array(xywh, dtype=np.float32).reshape(-1, 4),
        np.array(confidences, dtype=np.float32),
        np.array(class_ids, dtype=np.float32)
    )


def update_tracker(tracker, detections, frame):
    tracks = tracker.update(detections, frame)

    if len(tracks) == 0:
        return np.empty((0, 8), dtype=np.float32)

    return np.array([normalize_track(track) for track in tracks], dtype=np.float32)


def normalize_track(track):
    if hasattr(track, "result"):
        track = track.result
    elif hasattr(track, "tlbr"):
        x1, y1, x2, y2 = track.tlbr
        track = [
            x1,
            y1,
            x2,
            y2,
            getattr(track, "track_id", -1),
            getattr(track, "score", 0.0),
            getattr(track, "cls", 0),
            getattr(track, "idx", -1)
        ]

    return np.array(track, dtype=np.float32)


def detections_to_json(custom_labels, frame_width, frame_height):
    detections = []

    for label in custom_labels:
        box = label.get("Geometry", {}).get("BoundingBox")

        if not box:
            continue

        left = box["Left"] * frame_width
        top = box["Top"] * frame_height
        width = box["Width"] * frame_width
        height = box["Height"] * frame_height

        detections.append(
            {
                "name": label.get("Name", "package"),
                "confidence": label.get("Confidence", 0) / 100.0,
                "bbox_xyxy": [
                    left,
                    top,
                    left + width,
                    top + height
                ],
                "bounding_box": {
                    "left": box["Left"],
                    "top": box["Top"],
                    "width": box["Width"],
                    "height": box["Height"]
                }
            }
        )

    return detections


def tracks_to_json(tracks):
    output = []

    for track in tracks:
        x1, y1, x2, y2 = map(float, track[:4])
        output.append(
            {
                "track_id": int(track[4]),
                "class_id": int(track[6]),
                "confidence": float(track[5]),
                "bbox_xyxy": [x1, y1, x2, y2],
                "bbox_xywh": [x1, y1, x2 - x1, y2 - y1]
            }
        )

    return output


def find_matching_image_key(
    s3_client,
    bucket,
    detection_key,
    input_prefix,
    detections_prefix,
    image_extensions
):
    detection_prefix = normalize_prefix(detections_prefix)
    input_prefix = normalize_prefix(input_prefix)

    if not detection_key.startswith(detection_prefix):
        raise ValueError(
            f"{detection_key} is not under detection prefix {detection_prefix}"
        )

    relative_stem = strip_extension(detection_key[len(detection_prefix):])

    for extension in image_extensions:
        image_key = f"{input_prefix}{relative_stem}{extension}"

        try:
            s3_client.head_object(Bucket=bucket, Key=image_key)
            return image_key
        except ClientError as exc:
            status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")

            if status != 404:
                raise

    raise FileNotFoundError(
        f"No matching input image found for s3://{bucket}/{detection_key}"
    )


def local_track_path(tracks_dir, detection_key, detections_prefix):
    detection_prefix = normalize_prefix(detections_prefix)
    relative_key = detection_key[len(detection_prefix):]
    return tracks_dir / f"{strip_extension(relative_key)}.json"


def natural_sort_key(path):
    parts = []
    current = ""
    in_number = False

    for character in str(path):
        character_is_number = character.isdigit()

        if current and character_is_number != in_number:
            parts.append(int(current) if in_number else current.lower())
            current = ""

        current += character
        in_number = character_is_number

    if current:
        parts.append(int(current) if in_number else current.lower())

    return parts


def process_s3_frames(
    s3_client,
    tracker,
    bucket,
    input_prefix,
    detections_prefix,
    tracks_dir,
    image_extensions,
    class_id,
    limit
):
    detection_keys = list_detection_keys(s3_client, bucket, detections_prefix)

    if limit is not None:
        detection_keys = detection_keys[:limit]

    if not detection_keys:
        print(
            f"No JSON files found at "
            f"s3://{bucket}/{normalize_prefix(detections_prefix)}"
        )
        return 0

    tracks_dir.mkdir(parents=True, exist_ok=True)
    processed_count = 0

    for frame_index, detection_key in enumerate(detection_keys, start=1):
        output_path = local_track_path(
            tracks_dir,
            detection_key,
            detections_prefix
        )

        if output_path.exists():
            continue

        image_key = find_matching_image_key(
            s3_client,
            bucket,
            detection_key,
            input_prefix,
            detections_prefix,
            image_extensions
        )
        detection_response = read_json_from_s3(s3_client, bucket, detection_key)
        frame = read_image_from_s3(s3_client, bucket, image_key)
        frame_height, frame_width = frame.shape[:2]
        custom_labels = detection_response.get("CustomLabels", [])

        detections = labels_to_tracker_detections(
            custom_labels,
            frame_width,
            frame_height,
            class_id
        )
        tracks = update_tracker(tracker, detections, frame)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        result = {
            "source_image_s3_uri": f"s3://{bucket}/{image_key}",
            "source_detection_s3_uri": f"s3://{bucket}/{detection_key}",
            "frame_index": frame_index,
            "processed_at_epoch": time.time(),
            "frame_width": frame_width,
            "frame_height": frame_height,
            "detections": detections_to_json(
                custom_labels,
                frame_width,
                frame_height
            ),
            "tracks": tracks_to_json(tracks)
        }

        with output_path.open("w", encoding="utf-8") as output_file:
            json.dump(result, output_file, indent=2)

        print(
            f"{frame_index:05d} {image_key}: "
            f"{len(result['detections'])} detections, "
            f"{len(result['tracks'])} tracks -> {output_path}"
        )
        processed_count += 1

    if processed_count == 0:
        print("No new detection JSON files to process.")

    return processed_count


def watch_s3_frames(
    s3_client,
    tracker,
    bucket,
    input_prefix,
    detections_prefix,
    tracks_dir,
    image_extensions,
    class_id,
    limit,
    poll_seconds
):
    print(
        f"Watching s3://{bucket}/{normalize_prefix(detections_prefix)} "
        f"every {poll_seconds:g}s..."
    )

    while True:
        try:
            process_s3_frames(
                s3_client,
                tracker,
                bucket,
                input_prefix,
                detections_prefix,
                tracks_dir,
                image_extensions,
                class_id,
                limit
            )
        except Exception as exc:
            print(f"Polling error: {exc}")

        time.sleep(poll_seconds)


def normalize_prefix(prefix):
    return f"{prefix.strip('/')}/"


def strip_extension(key):
    return key.rsplit(".", 1)[0]


def main():
    args = parse_args()
    image_extensions = tuple(
        extension.strip().lower()
        for extension in args.image_extensions.split(",")
        if extension.strip()
    )
    s3_client = boto3.client("s3", region_name=args.region)
    tracker = create_tracker(args.frame_rate)

    if args.watch:
        watch_s3_frames(
            s3_client,
            tracker,
            args.bucket,
            args.input_prefix,
            args.detections_prefix,
            Path(args.tracks_dir),
            image_extensions,
            args.class_id,
            args.limit,
            args.poll_seconds
        )
    else:
        process_s3_frames(
            s3_client,
            tracker,
            args.bucket,
            args.input_prefix,
            args.detections_prefix,
            Path(args.tracks_dir),
            image_extensions,
            args.class_id,
            args.limit
        )


if __name__ == "__main__":
    main()
