import os
import boto3
import cv2
from pathlib import Path
from dotenv import load_dotenv

root_dir = Path(__file__).resolve().parent.parent
env_path = root_dir / '.env'

# 2. Explicitly load the .env file from the root path
load_dotenv(dotenv_path=env_path)

# ----------------------------
# AWS Rekognition Client
# ----------------------------
client = boto3.client(
    "rekognition",
    region_name="ap-south-1"
)

# ----------------------------
# Config
# ----------------------------
PROJECT_VERSION_ARN = (
    "arn:aws:rekognition:ap-south-1:364598914440:project/package-recognition/version/package-recognition.2026-05-21T12.01.54/1779345114953"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TEST_FOLDER = os.path.join(BASE_DIR, "test")

OUTPUT_FOLDER = os.path.join(BASE_DIR, "output")

MIN_CONFIDENCE = 20

# ----------------------------
# Supported Image Extensions
# ----------------------------
VALID_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png"
)

def detect_custom_labels_from_bytes(image_bytes):
    return client.detect_custom_labels(
        ProjectVersionArn=PROJECT_VERSION_ARN,
        Image={
            "Bytes": image_bytes
        },
        MinConfidence=MIN_CONFIDENCE
    )


def draw_custom_labels(frame, custom_labels):
    height, width = frame.shape[:2]

    for label in custom_labels:
        if "Geometry" not in label or "BoundingBox" not in label["Geometry"]:
            continue

        name = label["Name"]
        confidence = label["Confidence"]
        box = label["Geometry"]["BoundingBox"]

        # Convert normalized Rekognition coordinates to frame pixels.
        left = int(box["Left"] * width)
        top = int(box["Top"] * height)
        box_width = int(box["Width"] * width)
        box_height = int(box["Height"] * height)

        right = left + box_width
        bottom = top + box_height

        cv2.rectangle(
            frame,
            (left, top),
            (right, bottom),
            (0, 255, 0),
            2
        )

        text = f"{name}: {confidence:.1f}%"
        text_y = max(top - 10, 20)

        cv2.putText(
            frame,
            text,
            (left, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    return frame


def process_images():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    for filename in os.listdir(TEST_FOLDER):
        if not filename.lower().endswith(VALID_EXTENSIONS):
            continue

        image_path = os.path.join(
            TEST_FOLDER,
            filename
        )

        print(f"\nProcessing: {filename}")

        with open(image_path, "rb") as image:
            image_bytes = image.read()

        response = detect_custom_labels_from_bytes(image_bytes)

        frame = cv2.imread(image_path)
        draw_custom_labels(frame, response["CustomLabels"])

        for label in response["CustomLabels"]:
            print(
                f"Detected {label['Name']} "
                f"({label['Confidence']:.1f}%)"
            )

        output_path = os.path.join(
            OUTPUT_FOLDER,
            filename
        )

        cv2.imwrite(
            output_path,
            frame
        )

        print(f"Saved: {output_path}")

    print("\nDone.")


if __name__ == "__main__":
    process_images()
