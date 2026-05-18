import cv2

# Replace with your camera stream URL
# Examples:
# RTSP: rtsp://username:password@192.168.1.100:554/stream
# HTTP/MJPEG: http://192.168.1.100:8080/video

stream_url = ""

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