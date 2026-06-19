"""
API tests for document management endpoints:
  GET  /documents
  POST /upload
  DELETE /documents/{filename}
"""
import io
import os
import pytest

pytestmark = pytest.mark.api


# ── List documents ─────────────────────────────────────────────────────────────

def test_list_documents_returns_200(api_client):
    r = api_client.get("/documents")
    assert r.status_code == 200


def test_list_documents_returns_list(api_client):
    r = api_client.get("/documents")
    assert isinstance(r.json(), list)


def test_list_documents_items_have_name_field(api_client):
    """The mocked S3 list returns one document — verify its structure."""
    r = api_client.get("/documents")
    docs = r.json()
    if docs:  # mocked returns one entry
        assert "name" in docs[0]
        assert "size" in docs[0]


# ── Upload validation ──────────────────────────────────────────────────────────

def test_upload_non_pdf_extension_rejected(api_client):
    """A file with .txt extension must be rejected with 400."""
    fake_content = b"this is not a pdf"
    r = api_client.post(
        "/upload",
        files={"file": ("document.txt", io.BytesIO(fake_content), "text/plain")},
    )
    assert r.status_code == 400
    assert "pdf" in r.json()["detail"].lower()


def test_upload_invalid_pdf_content_rejected(api_client):
    """
    A file with .pdf extension but fake content (not starting with %PDF-)
    must be rejected with 400.
    """
    fake_pdf = b"this is totally not a pdf file just pretending"
    r = api_client.post(
        "/upload",
        files={"file": ("fake.pdf", io.BytesIO(fake_pdf), "application/pdf")},
    )
    assert r.status_code == 400
    assert "valid PDF" in r.json()["detail"] or "pdf" in r.json()["detail"].lower()


def test_upload_oversized_file_rejected(api_client, monkeypatch):
    """
    Files larger than UPLOAD_MAX_MB must be rejected with 413.
    We temporarily set the limit to 1 byte so any file exceeds it.
    """
    import app.main as m
    original = m.UPLOAD_MAX_BYTES
    m.UPLOAD_MAX_BYTES = 1  # 1 byte — any real file will exceed this
    try:
        r = api_client.post(
            "/upload",
            files={"file": ("small.pdf", io.BytesIO(b"%PDF-1.0 tiny"), "application/pdf")},
        )
        assert r.status_code == 413
    finally:
        m.UPLOAD_MAX_BYTES = original


def test_upload_real_pdf_succeeds(api_client, pdf_bytes, tmp_path, monkeypatch):
    """
    Uploading a real valid PDF must return 200 and a success message.
    We redirect DOCUMENTS_FOLDER to a temp dir so the test doesn't write
    to the real documents/ folder.
    """
    import app.main as m
    import app.services.rag_pipeline as rp
    from unittest.mock import patch

    pdf_content, fname = pdf_bytes
    test_fname = f"_test_upload_{fname}"
    temp_docs = str(tmp_path / "documents")
    os.makedirs(temp_docs)

    original_folder = m.DOCUMENTS_FOLDER
    m.DOCUMENTS_FOLDER = temp_docs + "/"

    # Mock add_document_to_index so we don't modify the real index
    with patch.object(rp, "add_document_to_index", return_value=None):
        try:
            r = api_client.post(
                "/upload",
                files={"file": (test_fname, io.BytesIO(pdf_content), "application/pdf")},
            )
            assert r.status_code == 200
            assert "uploaded" in r.json()["message"].lower()
        finally:
            m.DOCUMENTS_FOLDER = original_folder
            # Clean up temp file if created in real folder
            real_path = os.path.join("documents", test_fname)
            if os.path.exists(real_path):
                os.remove(real_path)


# ── Delete ─────────────────────────────────────────────────────────────────────

def test_delete_nonexistent_file_returns_404(api_client, monkeypatch):
    """
    Deleting a file not in S3 must return 404. The mocked delete_from_s3
    returns True by default, so we override it to return False for this test.
    """
    from unittest.mock import patch
    with patch("app.services.s3_storage.delete_from_s3", return_value=False):
        r = api_client.delete("/documents/nonexistent_file.pdf")
    assert r.status_code == 404


def test_delete_existing_file_returns_200(api_client, tmp_path):
    """
    Deleting a file that exists in S3 (mocked True) and optionally locally
    must return 200 with a success message.
    """
    import app.services.rag_pipeline as rp
    from unittest.mock import patch

    with patch("app.services.s3_storage.delete_from_s3", return_value=True), \
         patch.object(rp, "remove_document_from_index", return_value=None):
        r = api_client.delete("/documents/some_existing_file.pdf")

    assert r.status_code == 200
    assert "deleted" in r.json()["message"].lower()
