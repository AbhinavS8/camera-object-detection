import os
import time
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
from stiching import create_batch_collage, decode_collage_labels

import cv2
import numpy as np
from ultralytics.trackers.byte_tracker import BYTETracker  # type: ignore

from rekognition import detect_custom_labels_from_bytes, client, PROJECT_VERSION_ARN
from aws_controller import start_aws_model, stop_aws_model

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
CAMERA_INDEX = 0
SAMPLE_INTERVAL_SECONDS = 0.2
BATCH_SIZE = 5  # Number of frames to stitch together
JPEG_QUALITY = 85
WINDOW_NAME = "AWS Rekognition Batch-Stitched Feed"
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
TRACKER_FRAME_RATE = int(1 / SAMPLE_INTERVAL_SECONDS)
MIN_MOTION_AREA = 5000

# ---------------------------------------------------------
# Tracker Helper Classes
# ---------------------------------------------------------
class ArrayAdapter:
    def __init__(self, values): self.values = values
    def cpu(self): return self
    def numpy(self): return self.values
    def astype(self, dtype): return self.values.astype(dtype)
    def reshape(self, *shape): return self.values.reshape(*shape)
    def __len__(self): return len(self.values)
    def __getitem__(self, item): return self.values[item]
    def __array__(self, dtype=None): 
        if dtype is None: return self.values
        return self.values.astype(dtype)
    def __ge__(self, other): return self.values >= other
    def __gt__(self, other): return self.values > other
    def __le__(self, other): return self.values <= other
    def __lt__(self, other): return self.values < other

class TrackerDetections:
    def __init__(self, xywh, conf, cls):
        self.xywh = ArrayAdapter(xywh)
        self.conf = ArrayAdapter(conf)
        self.cls = ArrayAdapter(cls)
    def cpu(self): return self
    def numpy(self): return self
    def __len__(self): return len(self.conf)
    def __getitem__(self, item):
        if isinstance(item, int): return self
        return TrackerDetections(
            self.xywh.values[item], self.conf.values[item], self.cls.values[item]
        )

# ---------------------------------------------------------
# Core Functions
# ---------------------------------------------------------
def create_tracker():
    args = SimpleNamespace(
        track_high_thresh=0.2, track_low_thresh=0.05, new_track_thresh=0.2, 
        track_buffer=30, match_thresh=0.8, fuse_score=True, mot20=False
    )
    try: return BYTETracker(args, frame_rate=TRACKER_FRAME_RATE)
    except TypeError: return BYTETracker(args)

def encode_frame_as_jpeg(frame):
    success, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if not success: raise RuntimeError("Could not encode frame as JPEG")
    return buffer.tobytes()

def detect_frame(frame):
    """Sends a frame (or collage) to AWS Rekognition via bytes."""
    image_bytes = encode_frame_as_jpeg(frame)
    response = detect_custom_labels_from_bytes(image_bytes)
    return response.get("CustomLabels", [])


def process_stitched_batch(frames_list, width, height):
    """BACKGROUND THREAD: Creates collage, pings AWS once, and decodes the math."""
    collage = create_batch_collage(frames_list)
    raw_labels = detect_frame(collage)
    return decode_collage_labels(raw_labels, width, height)

# ---------------------------------------------------------
# Tracking & Output Functions
# ---------------------------------------------------------
def labels_to_tracker_detections(custom_labels, frame_width, frame_height):
    xywh, confidences, class_ids = [], [], []
    for label in custom_labels:
        box = label.get("Geometry", {}).get("BoundingBox")
        if not box: continue
        left = box["Left"] * frame_width
        top = box["Top"] * frame_height
        width = box["Width"] * frame_width
        height = box["Height"] * frame_height
        center_x = left + width / 2
        center_y = top + height / 2
        xywh.append([center_x, center_y, width, height])
        confidences.append(label["Confidence"] / 100.0)
        class_ids.append(0)
    return TrackerDetections(
        np.array(xywh, dtype=np.float32).reshape(-1, 4),
        np.array(confidences, dtype=np.float32),
        np.array(class_ids, dtype=np.float32)
    )

def normalize_track(track):
    if hasattr(track, "result"): track = track.result
    elif hasattr(track, "tlbr"):
        x1, y1, x2, y2 = track.tlbr
        track = [x1, y1, x2, y2, getattr(track, "track_id", -1), getattr(track, "score", 0.0), getattr(track, "cls", 0), getattr(track, "idx", -1)]
    return np.array(track, dtype=np.float32)

def update_tracker(tracker, detections, frame):
    if len(detections) == 0: return np.empty((0, 8), dtype=np.float32)
    tracks = tracker.update(detections, frame)
    if len(tracks) == 0: return np.empty((0, 8), dtype=np.float32)
    return np.array([normalize_track(track) for track in tracks], dtype=np.float32)

def process_entry_exit(frame, tracks, seen_ids):
    base_output_dir = "output"
    os.makedirs(base_output_dir, exist_ok=True)
    for track in tracks:
        track_id = int(track[4])
        track_folder = os.path.join(base_output_dir, f"ID_{track_id}")
        os.makedirs(track_folder, exist_ok=True)
        if track_id not in seen_ids:
            seen_ids.add(track_id)
            cv2.imwrite(os.path.join(track_folder, "entry.jpg"), frame)
            print(f"📦 NEW PACKAGE: ID {track_id} ENTERED. Saving entry frame...")
        cv2.imwrite(os.path.join(track_folder, "exit.jpg"), frame)

def draw_tracks(frame, tracks):
    for track in tracks:
        x1, y1, x2, y2 = map(int, track[:4])
        track_id, confidence = int(track[4]), float(track[5])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {track_id}: {confidence:.2f}", (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# ---------------------------------------------------------
# Main Execution Loop
# ---------------------------------------------------------
def main():
    start_aws_model(client, PROJECT_VERSION_ARN)
    tracker = create_tracker()
    latest_tracks = np.empty((0, 8), dtype=np.float32)
    seen_ids = set()

    # Buffers to manage the batches
    frame_buffer = []
    processing_buffer = []

    back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("Error: Could not open webcam")
        stop_aws_model(client, PROJECT_VERSION_ARN)
        return

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, FRAME_WIDTH, FRAME_HEIGHT)

    in_flight_detection = None
    last_sample_time = 0

    with ThreadPoolExecutor(max_workers=1) as executor:
        try:
            while True:
                ret, frame = cap.read()
                if not ret: break

                frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
                now = time.monotonic()

                # 1. Motion Detection
                fg_mask = back_sub.apply(frame)
                _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                motion_detected = any(cv2.contourArea(c) > MIN_MOTION_AREA for c in contours)

                # 2. Fill the buffer during motion events
                if motion_detected and (now - last_sample_time >= SAMPLE_INTERVAL_SECONDS):
                    last_sample_time = now
                    frame_buffer.append(frame.copy())
                    print(f"Frame buffered. Size: {len(frame_buffer)}/{BATCH_SIZE}")

                # 3. Stitch and send to AWS when buffer is full
                if len(frame_buffer) >= BATCH_SIZE and in_flight_detection is None:
                    print("🚀 Stitching collage and calling AWS...")
                    processing_buffer = frame_buffer[:BATCH_SIZE]
                    del frame_buffer[:BATCH_SIZE]
                    
                    # Submit the background task (1 API Call)
                    in_flight_detection = executor.submit(
                        process_stitched_batch, processing_buffer, FRAME_WIDTH, FRAME_HEIGHT
                    )

                # 4. Receive decoded AWS results and update tracker
                if in_flight_detection and in_flight_detection.done():
                    try:
                        batch_results = in_flight_detection.result()
                        
                        # Loop through the 5 sets of decoded labels and the 5 original frames
                        for i in range(BATCH_SIZE):
                            batch_frame = processing_buffer[i]
                            labels = batch_results[i]
                            
                            detections = labels_to_tracker_detections(labels, FRAME_WIDTH, FRAME_HEIGHT)
                            latest_tracks = update_tracker(tracker, detections, batch_frame)
                            
                            # Log entry/exit with the clean, unstitched original frame
                            if len(latest_tracks) > 0:
                                process_entry_exit(batch_frame, latest_tracks, seen_ids)
                                
                    except Exception as exc:
                        print(f"Batch decoding error: {exc}")
                        latest_tracks = np.empty((0, 8), dtype=np.float32)

                    # Clear the flight flags so we can send the next batch
                    in_flight_detection = None
                    processing_buffer = []

                # 5. Visual Rendering (Live Camera Output)
                output_frame = frame.copy()
                if len(latest_tracks) > 0:
                    draw_tracks(output_frame, latest_tracks)

                status_color = (0, 0, 255) if motion_detected else (255, 0, 0)
                status_text = f"MOTION ACTIVE - BUFFER: {len(frame_buffer)}" if motion_detected else "STANDBY"
                cv2.putText(output_frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
                
                cv2.imshow(WINDOW_NAME, output_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"): break

        finally:
            cap.release()
            cv2.destroyAllWindows()
            # stop_aws_model(client, PROJECT_VERSION_ARN)

if __name__ == "__main__":
    main()