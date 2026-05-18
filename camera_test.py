import cv2, os
from dotenv import load_dotenv

load_dotenv()

username = os.getenv("CAMERA_USER")
password = os.getenv("CAMERA_PASSWORD")
ip = os.getenv("CAMERA_IP")

stream_url = f"rtsps://{username}:{password}@{ip}:554/video/live?channel=1&subtype=1"

# Open the video stream
cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print("Error: Could not open video stream")
    exit()


# Create resizable window
cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)

# Set window size
cv2.resizeWindow("Camera", 1280, 720)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()