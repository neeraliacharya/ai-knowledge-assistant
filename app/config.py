import os
from dotenv import load_dotenv

load_dotenv()


def get_env_variable(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise EnvironmentError(f"Required environment variable '{name}' is not set")
    return value


# ── Core LLM ──────────────────────────────────────────────────────────────────
GROQ_API_KEY    = get_env_variable("GROQ_API_KEY")
LLM_MODEL       = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
LLM_MAX_TOKENS  = int(os.getenv("LLM_MAX_TOKENS", "800"))

# ── Vector store ───────────────────────────────────────────────────────────────
VECTOR_DB_PATH  = os.getenv("VECTOR_DB_PATH", "./vector_store")

# ── AWS / S3 ──────────────────────────────────────────────────────────────────
AWS_ACCESS_KEY_ID     = get_env_variable("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = get_env_variable("AWS_SECRET_ACCESS_KEY")
AWS_REGION            = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET_NAME        = get_env_variable("S3_BUCKET_NAME")
S3_PREFIX             = os.getenv("S3_PREFIX", "documents/")

# ── Security ───────────────────────────────────────────────────────────────────
# If API_KEY is set, every request must include the header: X-API-Key: <value>
# Leave unset (or empty) to disable auth (dev mode).
API_KEY         = os.getenv("API_KEY", "")

# Comma-separated list of allowed CORS origins. Restrict this in production.
ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv(
        "ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:3000"
    ).split(",")
    if o.strip()
]

# ── Upload limits ──────────────────────────────────────────────────────────────
UPLOAD_MAX_MB = int(os.getenv("UPLOAD_MAX_MB", "50"))
