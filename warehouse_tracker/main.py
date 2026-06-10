import threading
import logging
import cv2
from concurrent.futures import ThreadPoolExecutor

from core.config import config
from core.frame_batcher import FrameBatcher
from core.s3_batch_uploader import S3BatchUploader


logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

WINDOW_NAME = "Warehouse Edge Node"

def draw_hud(frame, status_text, color=(0, 255, 0)):
    """Draws the transparent overlay text onto the camera feed."""
    output = frame.copy()
    cv2.putText(output, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return output

def main():
    logger.info("🚀 Booting Edge Tracking Node...")
    
    stop_event = threading.Event()
    batcher = FrameBatcher()
    uploader = S3BatchUploader()
    
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, config.FRAME_WIDTH, config.FRAME_HEIGHT)
    
    frame_buffer = []
    in_flight_upload = None
    status_text = "System Ready"
    status_color = (0, 255, 0) # Green

    with ThreadPoolExecutor(max_workers=1) as executor:
        try:
            for captured_frame in batcher.capture_frames(stop_event):
                frame_buffer.append(captured_frame)

                # 1. Check if the background thread finished an upload
                if in_flight_upload and in_flight_upload.done():
                    try:
                        uploaded_batch = in_flight_upload.result()
                        status_text = f"Uploaded Batch: {uploaded_batch.batch_id}"
                        status_color = (0, 255, 0) # Green
                        logger.info(f"✅ Successfully uploaded and signaled Batch {uploaded_batch.batch_id}")
                    except Exception as exc:
                        status_text = "Upload Failed (Check Logs)"
                        status_color = (0, 0, 255) # Red
                        logger.error("Background upload crashed", exc_info=exc)
                    
                    in_flight_upload = None

                # 2. If buffer is full, trigger background upload
                if len(frame_buffer) >= config.BATCH_SIZE and in_flight_upload is None:
                    batch_to_upload = batcher.create_batch(frame_buffer[:config.BATCH_SIZE])
                    del frame_buffer[:config.BATCH_SIZE]
                    
                    status_text = f"Uploading Batch {batch_to_upload.batch_id}..."
                    status_color = (0, 165, 255) # Orange
                    logger.info(status_text)
                    
                    in_flight_upload = executor.submit(uploader.upload_batch, batch_to_upload)

                # 3. Draw UI
                ui_text = f"{status_text} | Buffer: {len(frame_buffer)}/{config.BATCH_SIZE}"
                ui_frame = draw_hud(captured_frame.frame, ui_text, color=status_color)
                cv2.imshow(WINDOW_NAME, ui_frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("Shutdown signal received (Q pressed).")
                    break
                    
        finally:
            stop_event.set()
            if in_flight_upload:
                logger.info("Waiting for final batch to upload before closing...")
                try:
                    in_flight_upload.result() # Wait gracefully for the final upload to finish
                except Exception:
                    logger.error("Final upload failed during shutdown.")
            
            cv2.destroyAllWindows()
            logger.info("System Offline.")

if __name__ == "__main__":
    main()