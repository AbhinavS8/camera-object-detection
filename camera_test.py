import cv2, os
from dotenv import load_dotenv
import gradio as gr
import numpy as np
import matplotlib.pyplot as plt

load_dotenv()

username = os.getenv("CAMERA_USER")
password = os.getenv("CAMERA_PASSWORD")
ip = os.getenv("CAMERA_IP")

stream_url = f"rtsps://{username}:{password}@{ip}:554/video/live?channel=1&subtype=1"

# # Open the video stream
# cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)

# inbuilt camera -- 0
cap = cv2.VideoCapture(0)

fgbg =cv2.createBackgroundSubtractorMOG2() #- part of core OpenCV, no need for contrib version. can use bgsegm as well


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

    # frame = cv2.GaussianBlur(frame, (5,5), 0)

    mask = fgbg.apply(frame)

    # _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3,3), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    fgmask = fgbg.apply(frame)
    cv2.imshow("MOG2", mask)
    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

cap.release()
cv2.destroyAllWindows()