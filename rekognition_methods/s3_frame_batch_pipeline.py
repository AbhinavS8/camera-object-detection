import threading
import time
from concurrent.futures import ThreadPoolExecutor

import cv2

try:
    from frame_batcher import (
        BATCH_SIZE,
        FrameBatch,
        FRAME_HEIGHT,
        FRAME_WIDTH,
        SAMPLE_INTERVAL_SECONDS,
        FrameBatcher,
        utc_timestamp
    )
    from s3_batch_uploader import S3BatchUploader
except ImportError:
    from .frame_batcher import (
        BATCH_SIZE,
        FrameBatch,
        FRAME_HEIGHT,
        FRAME_WIDTH,
        SAMPLE_INTERVAL_SECONDS,
        FrameBatcher,
        utc_timestamp
    )
    from .s3_batch_uploader import S3BatchUploader


WINDOW_NAME = "S3 Frame Batch Upload"


def create_batch(frames):
    batch_id = frames[0].batch_id

    return FrameBatch(
        run_id=frames[0].run_id,
        batch_id=batch_id,
        created_at_epoch=time.time(),
        created_at_utc=utc_timestamp(),
        frames=frames
    )


def draw_upload_status(
    frame,
    captured_frame,
    buffered_count,
    upload_status="Ready",
    uploaded_batch=None,
    error=None
):
    output_frame = frame.copy()
    status = (
        f"Frame {captured_frame.sequence_id} | "
        f"Run {captured_frame.run_id} | "
        f"Batch {captured_frame.batch_id} | "
        f"{buffered_count}/{BATCH_SIZE} buffered | "
        f"{SAMPLE_INTERVAL_SECONDS:.1f}s sample"
    )
    cv2.putText(
        output_frame,
        status,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2
    )

    if uploaded_batch:
        upload_text = f"Uploaded manifest: {uploaded_batch.manifest_key}"
        color = (0, 255, 0)
    elif error:
        upload_text = f"Upload error: {error}"
        color = (0, 0, 255)
    else:
        upload_text = upload_status
        color = (0, 165, 255)

    cv2.putText(
        output_frame,
        upload_text[:100],
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2
    )

    return output_frame


def main():
    stop_event = threading.Event()
    batcher = FrameBatcher()
    uploader = S3BatchUploader()
    frame_buffer = []
    in_flight_upload = None
    upload_status = "Ready"
    uploaded_batch = None
    upload_error = None

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, FRAME_WIDTH, FRAME_HEIGHT)

    with ThreadPoolExecutor(max_workers=1) as executor:
        try:
            for captured_frame in batcher.capture_frames(stop_event):
                frame_buffer.append(captured_frame)

                if in_flight_upload and in_flight_upload.done():
                    uploaded_batch = None
                    upload_error = None

                    try:
                        uploaded_batch = in_flight_upload.result()
                        upload_status = (
                            f"Uploaded batch {uploaded_batch.batch_id}"
                        )
                        print(
                            f"Uploaded batch {uploaded_batch.batch_id}: "
                            f"{uploaded_batch.manifest_key}"
                        )
                    except Exception as exc:
                        upload_error = exc
                        upload_status = "Upload failed"
                        print(f"Upload failed: {exc}")

                    in_flight_upload = None

                if (
                    len(frame_buffer) >= BATCH_SIZE and
                    in_flight_upload is None
                ):
                    batch_frames = frame_buffer[:BATCH_SIZE]
                    del frame_buffer[:BATCH_SIZE]
                    batch = create_batch(batch_frames)
                    upload_status = f"Uploading batch {batch.batch_id} to S3..."
                    uploaded_batch = None
                    upload_error = None

                    print(
                        f"Uploading batch {batch.batch_id} "
                        f"({len(batch.frames)} frames) created at "
                        f"{batch.created_at_utc}"
                    )
                    in_flight_upload = executor.submit(
                        uploader.upload_batch,
                        batch
                    )

                output_frame = draw_upload_status(
                    captured_frame.frame,
                    captured_frame,
                    len(frame_buffer),
                    upload_status=upload_status,
                    uploaded_batch=uploaded_batch,
                    error=upload_error
                )
                cv2.imshow(WINDOW_NAME, output_frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    stop_event.set()
                    break
        finally:
            if in_flight_upload:
                try:
                    in_flight_upload.result()
                except Exception as exc:
                    print(f"Upload failed while shutting down: {exc}")
            stop_event.set()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
