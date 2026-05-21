import os
import boto3
import cv2

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

TEST_FOLDER = "test"

OUTPUT_FOLDER = "output"

MIN_CONFIDENCE = 20

# Create output folder
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ----------------------------
# Supported Image Extensions
# ----------------------------
VALID_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png"
)

# ----------------------------
# Process All Images
# ----------------------------
for filename in os.listdir(TEST_FOLDER):

    if not filename.lower().endswith(VALID_EXTENSIONS):
        continue

    image_path = os.path.join(
        TEST_FOLDER,
        filename
    )

    print(f"\nProcessing: {filename}")

    # ----------------------------
    # Read Image Bytes
    # ----------------------------
    with open(image_path, "rb") as image:

        image_bytes = image.read()

    # ----------------------------
    # Run Detection
    # ----------------------------
    response = client.detect_custom_labels(

        ProjectVersionArn=PROJECT_VERSION_ARN,

        Image={
            "Bytes": image_bytes
        },

        MinConfidence=MIN_CONFIDENCE
    )

    # ----------------------------
    # Load Image with OpenCV
    # ----------------------------
    frame = cv2.imread(image_path)

    height, width = frame.shape[:2]

    # ----------------------------
    # Draw Bounding Boxes
    # ----------------------------
    for label in response["CustomLabels"]:

        name = label["Name"]

        confidence = label["Confidence"]

        box = label["Geometry"]["BoundingBox"]

        # Convert normalized coordinates
        left = int(box["Left"] * width)

        top = int(box["Top"] * height)

        box_width = int(box["Width"] * width)

        box_height = int(box["Height"] * height)

        right = left + box_width

        bottom = top + box_height

        # ----------------------------
        # Draw Rectangle
        # ----------------------------
        cv2.rectangle(
            frame,
            (left, top),
            (right, bottom),
            (0, 255, 0),
            2
        )

        # ----------------------------
        # Draw Label Text
        # ----------------------------
        text = f"{name}: {confidence:.1f}%"

        cv2.putText(
            frame,
            text,
            (left, top - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        print(
            f"Detected {name} "
            f"({confidence:.1f}%)"
        )

    # ----------------------------
    # Save Output Image
    # ----------------------------
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