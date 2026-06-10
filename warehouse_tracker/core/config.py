import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env explicitly from the root directory
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env", override=True)

class Settings:
    CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", 0))
    SAMPLE_INTERVAL = float(os.getenv("SAMPLE_INTERVAL_SECONDS", 0.2))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", 5))
    FRAME_WIDTH = int(os.getenv("FRAME_WIDTH", 1280))
    FRAME_HEIGHT = int(os.getenv("FRAME_HEIGHT", 720))
    
    # AWS Settings
    S3_BUCKET = os.getenv("S3_FRAME_BUCKET")
    AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")
    S3_PREFIX = os.getenv("S3_FRAME_PREFIX", "input").strip("/")
    JPEG_QUALITY = int(os.getenv("S3_FRAME_JPEG_QUALITY", 85))
    
    # API Gateway
    CONFIRMATION_URL = os.getenv("S3_UPLOAD_CONFIRMATION_URL")

config = Settings()