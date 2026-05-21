import time
from concurrent.futures import ThreadPoolExecutor

import cv2

from rekognition import detect_custom_labels_from_bytes, draw_custom_labels


CAMERA_INDEX = 0
SAMPLE_INTERVAL_SECONDS = 0.2
JPEG_QUALITY = 85
WINDOW_NAME = "AWS Rekognition Video Feed"


def encode_frame_as_jpeg(frame):
    success, buffer = cv2.imencode(
        ".jpg",
        frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
    )

    if not success:
        raise RuntimeError("Could not encode frame as JPEG")

    return buffer.tobytes()


def detect_frame(frame):
    image_bytes = encode_frame_as_jpeg(frame)
    response = detect_custom_labels_from_bytes(image_bytes)
    return response.get("CustomLabels", [])


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 720)

    latest_labels = []
    in_flight_detection = None
    last_sample_time = 0

    with ThreadPoolExecutor(max_workers=1) as executor:
        try:
            while True:
                ret, frame = cap.read()

                if not ret:
                    print("Error: Could not read frame")
                    break

                now = time.monotonic()

                if in_flight_detection and in_flight_detection.done():
                    try:
                        latest_labels = in_flight_detection.result()
                    except Exception as exc:
                        print(f"Rekognition error: {exc}")
                        latest_labels = []

                    in_flight_detection = None

                if (
                    in_flight_detection is None
                    and now - last_sample_time >= SAMPLE_INTERVAL_SECONDS
                ):
                    last_sample_time = now
                    detection_frame = frame.copy()
                    in_flight_detection = executor.submit(
                        detect_frame,
                        detection_frame
                    )

                output_frame = frame.copy()

                if latest_labels:
                    draw_custom_labels(output_frame, latest_labels)

                cv2.imshow(WINDOW_NAME, output_frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
