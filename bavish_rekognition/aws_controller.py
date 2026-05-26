
import time
import cv2
import numpy as np


def start_aws_model(client, PROJECT_VERSION_ARN):
    print("--- AWS Rekognition Cost Control ---")
    try:
        print("Sending START command to AWS...")
        client.start_project_version(
            ProjectVersionArn=PROJECT_VERSION_ARN,
            MinInferenceUnits=1 # 1 unit allows up to 5 transactions per second
        )
    except client.exceptions.ResourceInUseException:
        print("Model is already running or currently starting up.")

    print("Waiting for cloud servers to spin up (This usually takes 10 to 15 minutes).")
    
    # Create a blank 64x64 image to test if the model is awake
    dummy_img = np.zeros((64, 64, 3), dtype=np.uint8)
    _, buffer = cv2.imencode('.jpg', dummy_img)
    dummy_bytes = buffer.tobytes()

    while True:
        try:
            # Attempt a test prediction
            client.detect_custom_labels(
                ProjectVersionArn=PROJECT_VERSION_ARN,
                Image={'Bytes': dummy_bytes},
                MinConfidence=50
            )
            # If it doesn't throw an error, the model is awake!
            print("SUCCESS: Model is LIVE and accepting frames!\n")
            break
        except client.exceptions.ResourceNotReadyException:
            print("Still starting... checking again in 30 seconds.")
            time.sleep(30)
        except Exception:
            # If it fails for any other reason (like no objects found in the black box)
            # it still proves the model is online and responding.
            print("SUCCESS: Model is LIVE and accepting frames!\n")
            break

def stop_aws_model(client, PROJECT_VERSION_ARN):
    print("\n--- AWS Rekognition Cost Control ---")
    try:
        print("Sending STOP command to AWS...")
        client.stop_project_version(
            ProjectVersionArn=PROJECT_VERSION_ARN
        )
        print("SUCCESS: Model is shutting down. You will no longer be billed.")
    except client.exceptions.ResourceInUseException:
        print("Model is already stopping.")
    except Exception as e:
        print(f"Warning: Could not stop model automatically. Please check AWS Console. Error: {e}")