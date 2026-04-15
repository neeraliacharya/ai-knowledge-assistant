import os
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
AWS_ACCESS_KEY_ID     = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION            = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET_NAME        = os.getenv("S3_BUCKET_NAME")
S3_PREFIX             = os.getenv("S3_PREFIX", "documents/")   # folder inside bucket

# ── Client ────────────────────────────────────────────────────────────────────
def get_s3_client():
    return boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION,
    )


# ── Operations ────────────────────────────────────────────────────────────────

def upload_to_s3(local_path: str, filename: str) -> bool:
    """Upload a local file to S3. Returns True on success."""
    s3 = get_s3_client()
    key = f"{S3_PREFIX}{filename}"
    try:
        s3.upload_file(local_path, S3_BUCKET_NAME, key)
        print(f"[S3] Uploaded: {key}")
        return True
    except (ClientError, NoCredentialsError) as e:
        print(f"[S3] Upload failed for {filename}: {e}")
        return False


def delete_from_s3(filename: str) -> bool:
    """Delete a file from S3. Returns True on success."""
    s3 = get_s3_client()
    key = f"{S3_PREFIX}{filename}"
    try:
        s3.delete_object(Bucket=S3_BUCKET_NAME, Key=key)
        print(f"[S3] Deleted: {key}")
        return True
    except (ClientError, NoCredentialsError) as e:
        print(f"[S3] Delete failed for {filename}: {e}")
        return False


def list_s3_documents() -> list[dict]:
    """Return list of PDF documents in the S3 prefix as [{name, size}]."""
    s3 = get_s3_client()
    try:
        response = s3.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix=S3_PREFIX)
        files = []
        for obj in response.get("Contents", []):
            key = obj["Key"]
            filename = key.removeprefix(S3_PREFIX)
            # skip the folder placeholder itself and non-PDFs
            if not filename or not filename.lower().endswith(".pdf"):
                continue
            files.append({
                "name": filename,
                "size": obj["Size"],
            })
        return files
    except (ClientError, NoCredentialsError) as e:
        print(f"[S3] List failed: {e}")
        return []


def sync_s3_to_local(local_folder: str) -> None:
    """
    Download all PDFs from S3 into local_folder.
    Called once at startup so the RAG pipeline can read from disk as usual.
    """
    s3 = get_s3_client()
    os.makedirs(local_folder, exist_ok=True)

    try:
        response = s3.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix=S3_PREFIX)
        for obj in response.get("Contents", []):
            key = obj["Key"]
            filename = key.removeprefix(S3_PREFIX)
            if not filename or not filename.lower().endswith(".pdf"):
                continue
            local_path = os.path.join(local_folder, filename)
            if not os.path.exists(local_path):
                s3.download_file(S3_BUCKET_NAME, key, local_path)
                print(f"[S3] Synced to local: {filename}")
    except (ClientError, NoCredentialsError) as e:
        print(f"[S3] Startup sync failed: {e}")
