import threading

import cv2

try:
    from frame_batcher import (
        BATCH_SIZE,
        FRAME_HEIGHT,
        FRAME_WIDTH,
        SAMPLE_INTERVAL_SECONDS,
        FrameBatcher
    )
    from s3_batch_uploader import S3BatchUploader
except ImportError:
    from .frame_batcher import (
        BATCH_SIZE,
        FRAME_HEIGHT,
        FRAME_WIDTH,
        SAMPLE_INTERVAL_SECONDS,
        FrameBatcher
    )
    from .s3_batch_uploader import S3BatchUploader


WINDOW_NAME = "S3 Frame Batch Upload"


def draw_upload_status(frame, batch, uploaded_batch=None, error=None):
    output_frame = frame.copy()
    status = (
        f"Batch {batch.batch_id} | {len(batch.frames)}/{BATCH_SIZE} frames | "
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
        upload_text = "Uploading batch to S3..."
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

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, FRAME_WIDTH, FRAME_HEIGHT)

    try:
        for batch in batcher.capture_batches(stop_event):
            preview_frame = batch.frames[-1].frame
            cv2.imshow(WINDOW_NAME, draw_upload_status(preview_frame, batch))
            cv2.waitKey(1)

            print(
                f"Uploading batch {batch.batch_id} "
                f"({len(batch.frames)} frames) created at {batch.created_at_utc}"
            )

            try:
                uploaded_batch = uploader.upload_batch(batch)
                print(
                    f"Uploaded batch {uploaded_batch.batch_id}: "
                    f"{uploaded_batch.manifest_key}"
                )
                output_frame = draw_upload_status(
                    preview_frame,
                    batch,
                    uploaded_batch=uploaded_batch
                )
            except Exception as exc:
                print(f"Upload failed for batch {batch.batch_id}: {exc}")
                output_frame = draw_upload_status(
                    preview_frame,
                    batch,
                    error=exc
                )

            cv2.imshow(WINDOW_NAME, output_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                stop_event.set()
                break
    finally:
        stop_event.set()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
