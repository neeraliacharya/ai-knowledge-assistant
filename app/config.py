import os
from dotenv import load_dotenv

load_dotenv()

def get_env_variable(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise EnvironmentError(f"{name} is not set")
    return value

GROQ_API_KEY   = get_env_variable("GROQ_API_KEY")
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH")

# ── AWS / S3 ──────────────────────────────────────────────────────────────────
AWS_ACCESS_KEY_ID     = get_env_variable("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = get_env_variable("AWS_SECRET_ACCESS_KEY")
AWS_REGION            = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET_NAME        = get_env_variable("S3_BUCKET_NAME")
S3_PREFIX             = os.getenv("S3_PREFIX", "documents/")