import { useState, useEffect, useRef } from 'react'
import { Upload, Trash2, FileText, AlertCircle, CheckCircle, Loader2, FolderOpen } from 'lucide-react'

export default function DocumentsTab() {
  const [documents, setDocuments] = useState([])
  const [loading, setLoading] = useState(true)
  const [uploading, setUploading] = useState(false)
  const [deletingId, setDeletingId] = useState(null)
  const [toast, setToast] = useState(null)
  const [dragOver, setDragOver] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(null)
  const fileInputRef = useRef(null)

  const showToast = (message, type = 'success') => {
    setToast({ message, type })
    setTimeout(() => setToast(null), 3500)
  }

  const fetchDocuments = async () => {
    try {
      const res = await fetch('/documents')
      if (!res.ok) throw new Error('Failed to fetch')
      const data = await res.json()
      // Support both array of strings and array of objects
      const normalized = Array.isArray(data)
        ? data.map((d) => (typeof d === 'string' ? { name: d } : d))
        : []
      setDocuments(normalized)
    } catch (err) {
      showToast('Could not load documents. Is the backend running?', 'error')
      setDocuments([])
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchDocuments()
  }, [])

  const handleUpload = async (files) => {
    const pdfs = Array.from(files).filter((f) => f.type === 'application/pdf')
    if (pdfs.length === 0) {
      showToast('Only PDF files are supported.', 'error')
      return
    }

    setUploading(true)
    setUploadProgress({ current: 0, total: pdfs.length, fileName: pdfs[0].name })
    let successCount = 0

    for (let i = 0; i < pdfs.length; i++) {
      const file = pdfs[i]
      setUploadProgress({ current: i + 1, total: pdfs.length, fileName: file.name })
      
      const formData = new FormData()
      formData.append('file', file)
      try {
        const res = await fetch('/upload', { method: 'POST', body: formData })
        if (!res.ok) throw new Error()
        successCount++
      } catch {
        showToast(`Failed to upload "${file.name}"`, 'error')
      }
    }

    setUploading(false)
    setUploadProgress(null)
    if (successCount > 0) {
      showToast(
        successCount === 1
          ? `"${pdfs[0].name}" uploaded successfully.`
          : `${successCount} files uploaded successfully.`
      )
      fetchDocuments()
    }
    if (fileInputRef.current) fileInputRef.current.value = ''
  }

  const handleDelete = async (name) => {
    setDeletingId(name)
    try {
      const res = await fetch(`/documents/${encodeURIComponent(name)}`, { method: 'DELETE' })
      if (!res.ok) throw new Error()
      showToast(`"${name}" deleted.`)
      setDocuments((prev) => prev.filter((d) => d.name !== name))
    } catch {
      showToast(`Failed to delete "${name}"`, 'error')
    } finally {
      setDeletingId(null)
    }
  }

  const onDrop = (e) => {
    e.preventDefault()
    setDragOver(false)
    handleUpload(e.dataTransfer.files)
  }

  const formatSize = (bytes) => {
    if (!bytes) return ''
    if (bytes < 1024) return `${bytes} B`
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
  }

  return (
    <div className="tab-content">
      {/* Toast */}
      {toast && (
        <div className={`toast toast--${toast.type}`}>
          {toast.type === 'error' ? <AlertCircle size={16} /> : <CheckCircle size={16} />}
          <span>{toast.message}</span>
        </div>
      )}

      <div className="section-header">
        <div>
          <h2 className="section-title">Documents</h2>
          <p className="section-subtitle">Upload PDF files to train your knowledge base</p>
        </div>
        <button
          className="btn btn--primary"
          onClick={() => fileInputRef.current?.click()}
          disabled={uploading}
        >
          {uploading ? (
            <><Loader2 size={16} className="spin" /> Uploading…</>
          ) : (
            <><Upload size={16} /> Upload PDF</>
          )}
        </button>
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf,application/pdf"
          multiple
          hidden
          onChange={(e) => handleUpload(e.target.files)}
        />
      </div>

      {/* Drop Zone */}
      <div
        className={`drop-zone ${dragOver ? 'drop-zone--active' : ''}`}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
        onClick={() => fileInputRef.current?.click()}
      >
        <Upload size={24} className="drop-zone__icon" />
        <p className="drop-zone__text">
          Drag &amp; drop PDF files here, or <span>click to browse</span>
        </p>
        <p className="drop-zone__hint">Only .pdf files are accepted</p>
      </div>

      {/* Documents List */}
      <div className="documents-list">
        {loading ? (
          <div className="state-empty">
            <Loader2 size={32} className="spin state-empty__icon" />
            <p>Loading documents…</p>
          </div>
        ) : documents.length === 0 ? (
          <div className="state-empty">
            <FolderOpen size={40} className="state-empty__icon" />
            <p className="state-empty__title">No documents yet</p>
            <p className="state-empty__sub">Upload a PDF to get started</p>
          </div>
        ) : (
          <>
            <div className="list-meta">{documents.length} document{documents.length !== 1 ? 's' : ''}</div>
            {documents.map((doc) => (
              <div key={doc.name} className="doc-row">
                <div className="doc-row__icon">
                  <FileText size={18} />
                </div>
                <div className="doc-row__info">
                  <span className="doc-row__name">{doc.name}</span>
                  {doc.size && <span className="doc-row__size">{formatSize(doc.size)}</span>}
                </div>
                <button
                  className="btn btn--ghost btn--danger"
                  onClick={() => handleDelete(doc.name)}
                  disabled={deletingId === doc.name}
                  title="Delete document"
                >
                  {deletingId === doc.name ? (
                    <Loader2 size={15} className="spin" />
                  ) : (
                    <Trash2 size={15} />
                  )}
                </button>
              </div>
            ))}
          </>
        )}
      </div>
      {/* Upload Progress (Sticky Bottom) */}
      {uploadProgress && (
        <div className="upload-progress">
          <div className="upload-progress__header">
            <div className="upload-progress__info">
              <Loader2 size={16} className="spin" />
              <span className="upload-progress__text">
                Uploading <strong>{uploadProgress.fileName}</strong> ({uploadProgress.current} of {uploadProgress.total})
              </span>
            </div>
            <span className="upload-progress__percent">
              {Math.round((uploadProgress.current / uploadProgress.total) * 100)}%
            </span>
          </div>
          <div className="upload-progress__bar-bg">
            <div 
              className="upload-progress__bar-fill" 
              style={{ width: `${(uploadProgress.current / uploadProgress.total) * 100}%` }}
            />
          </div>
        </div>
      )}
    </div>
  )
}
