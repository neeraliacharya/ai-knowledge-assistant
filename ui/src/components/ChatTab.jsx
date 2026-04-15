import { useState, useRef, useEffect } from 'react'
import { Send, Bot, User, Loader2, AlertCircle, Sparkles, Trash2, FileText, X } from 'lucide-react'

export default function ChatTab() {
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: 'Hello! I\'m your AI knowledge assistant. Ask me anything about your uploaded documents.',
      id: 'welcome',
    },
  ])
  const [sessionId, setSessionId] = useState(() => Math.random().toString(36).substring(2, 11))
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  
  // File Suggestion State
  const [documents, setDocuments] = useState([])
  const [selectedFiles, setSelectedFiles] = useState([])
  const [showMentions, setShowMentions] = useState(false)
  const [mentionQuery, setMentionQuery] = useState('')
  const [mentionIndex, setMentionIndex] = useState(0)

  const bottomRef = useRef(null)
  const inputRef = useRef(null)
  const textareaRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  // Fetch available documents for suggestions
  useEffect(() => {
    const fetchDocuments = async () => {
      try {
        const res = await fetch('/documents')
        if (res.ok) {
          const data = await res.json()
          const normalized = Array.isArray(data) ? data.map(d => typeof d === 'string' ? {name: d} : d) : []
          setDocuments(normalized)
        }
      } catch (err) {
        console.error("Failed to load documents for mentions")
      }
    }
    fetchDocuments()
  }, [])

  const autoResize = (el) => {
    el.style.height = 'auto'
    el.style.height = Math.min(el.scrollHeight, 160) + 'px'
  }

  // Handle typing & mention detection
  const handleInputChange = (e) => {
    const val = e.target.value
    setInput(val)
    autoResize(e.target)

    const cursor = e.target.selectionStart
    const textBeforeCursor = val.slice(0, cursor)
    const words = textBeforeCursor.split(/\s+/)
    const lastWord = words[words.length - 1]

    if (lastWord.startsWith('/')) {
      setShowMentions(true)
      setMentionQuery(lastWord.slice(1).toLowerCase())
      setMentionIndex(0)
    } else {
      setShowMentions(false)
    }
  }

  const filteredDocs = documents.filter((d) => 
    d.name.toLowerCase().includes(mentionQuery) && !selectedFiles.includes(d.name)
  )

  const addFile = (filename) => {
    if (!selectedFiles.includes(filename)) {
      setSelectedFiles([...selectedFiles, filename])
    }
    
    // Remove the /query from input
    const cursor = textareaRef.current.selectionStart
    const textBefore = input.slice(0, cursor)
    const textAfter = input.slice(cursor)
    
    const words = textBefore.split(/\s+/)
    words.pop() // remove the token starting with /
    
    const newTextBefore = words.length > 0 ? words.join(' ') + ' ' : ''
    
    setInput(newTextBefore + textAfter)
    setShowMentions(false)
    
    setTimeout(() => {
      inputRef.current?.focus()
      if (textareaRef.current) autoResize(textareaRef.current)
    }, 0)
  }

  const removeFile = (filename) => {
    setSelectedFiles(prev => prev.filter(f => f !== filename))
    inputRef.current?.focus()
  }

  const handleKeyDown = (e) => {
    // Handle mention navigation
    if (showMentions && filteredDocs.length > 0) {
      if (e.key === 'ArrowDown') {
        e.preventDefault()
        setMentionIndex((prev) => (prev + 1) % filteredDocs.length)
        return
      }
      if (e.key === 'ArrowUp') {
        e.preventDefault()
        setMentionIndex((prev) => (prev - 1 + filteredDocs.length) % filteredDocs.length)
        return
      }
      if (e.key === 'Enter' || e.key === 'Tab') {
        e.preventDefault()
        addFile(filteredDocs[mentionIndex].name)
        return
      }
      if (e.key === 'Escape') {
        setShowMentions(false)
        return
      }
    }

    // Normal message submit
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      if (!showMentions) handleSubmit()
    }
  }

  const handleSubmit = async (e) => {
    e?.preventDefault()
    const question = input.trim()
    if (!question || loading) return

    const filesToSend = [...selectedFiles]
    const userMsg = { role: 'user', content: question, id: Date.now(), files: filesToSend }
    setMessages((prev) => [...prev, userMsg])
    
    setInput('')
    setSelectedFiles([])
    setError(null)
    setLoading(true)
    setShowMentions(false)

    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
    }

    try {
      const params = new URLSearchParams()
      params.append('question', question)
      params.append('session_id', sessionId)
      filesToSend.forEach(f => params.append('files', f))
      
      const res = await fetch(`/ask?${params.toString()}`)
      if (!res.ok) throw new Error(`Server responded with ${res.status}`)
      const data = await res.json()

      const answer = data.answer ?? data.response ?? data.result ?? data.text ?? (typeof data === 'string' ? data : JSON.stringify(data))

      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: answer, id: Date.now() + 1 },
      ])
    } catch (err) {
      setError(err.message || 'Something went wrong. Is the backend running?')
    } finally {
      setLoading(false)
      setTimeout(() => inputRef.current?.focus(), 0)
    }
  }

  const clearChat = () => {
    setMessages([
      {
        role: 'assistant',
        content: 'Chat cleared. Ask me anything about your uploaded documents.',
        id: 'reset-' + Date.now(),
      },
    ])
    setError(null)
    setSelectedFiles([])
    setSessionId(Math.random().toString(36).substring(2, 11))
  }

  return (
    <div className="chat-layout">
      {/* Header */}
      <div className="chat-header">
        <div className="chat-header__left">
          <div className="chat-header__avatar">
            <Sparkles size={16} />
          </div>
          <div>
            <p className="chat-header__title">AI Knowledge Assistant</p>
            <p className="chat-header__sub">Powered by your documents</p>
          </div>
        </div>
        <button className="btn btn--ghost" onClick={clearChat} title="Clear chat">
          <Trash2 size={15} />
          <span>Clear</span>
        </button>
      </div>

      {/* Messages */}
      <div className="chat-messages">
        <div className="chat-messages-inner">
          {messages.map((msg) => (
            <div key={msg.id} className={`message message--${msg.role}`}>
              <div className="message__avatar">
                {msg.role === 'assistant' ? <Bot size={16} /> : <User size={16} />}
              </div>
              <div className="message__bubble">
                <div className="message__text">{msg.content}</div>
                {msg.files && msg.files.length > 0 && (
                  <div className="message__files">
                    {msg.files.map(f => (
                      <span key={f} className="message-chip"><FileText size={12}/>{f}</span>
                    ))}
                  </div>
                )}
              </div>
            </div>
          ))}

          {loading && (
            <div className="message message--assistant">
              <div className="message__avatar">
                <Bot size={16} />
              </div>
              <div className="message__bubble message__bubble--typing">
                <span className="dot" />
                <span className="dot" />
                <span className="dot" />
              </div>
            </div>
          )}

          {error && (
            <div className="chat-error">
              <AlertCircle size={15} />
              <span>{error}</span>
            </div>
          )}

          <div ref={bottomRef} />
        </div>
      </div>

      {/* Input */}
      <form className="chat-input-bar" onSubmit={handleSubmit}>
        <div className="chat-input-bar-inner">
          <div className="chat-input-wrap-container">
            {/* Dropdown Mentions */}
            {showMentions && (
              <div className="mentions-dropdown">
                {filteredDocs.length > 0 ? (
                  filteredDocs.map((doc, idx) => (
                    <div 
                      key={doc.name} 
                      className={`mention-item ${idx === mentionIndex ? 'mention-item--active' : ''}`}
                      onClick={() => addFile(doc.name)}
                    >
                      <FileText size={14} className="mention-icon" />
                      <span className="mention-name">{doc.name}</span>
                    </div>
                  ))
                ) : (
                  <div className="mention-item mention-item--empty">No exact matches</div>
                )}
              </div>
            )}

            <div 
              className="chat-input-wrap"
              onClick={() => inputRef.current?.focus()}
            >
              <div className="chat-input-content">
                {selectedFiles.length > 0 && (
                  <div className="chat-selected-files">
                    {selectedFiles.map(f => (
                      <div key={f} className="chat-file-chip">
                        <FileText size={13} />
                        <span>{f}</span>
                        <button type="button" onClick={(e) => { e.stopPropagation(); removeFile(f); }}>
                          <X size={13} />
                        </button>
                      </div>
                    ))}
                  </div>
                )}
                
                <textarea
                  ref={(el) => {
                    textareaRef.current = el
                    inputRef.current = el
                  }}
                  className="chat-input"
                  placeholder="Ask a question about your documents… (Type '/' to filter by file)"
                  value={input}
                  onChange={handleInputChange}
                  onKeyDown={handleKeyDown}
                  rows={1}
                  autoFocus
                />
              </div>

              <button
                type="submit"
                className="chat-send-btn"
                disabled={!input.trim() || loading}
                title="Send"
              >
                {loading ? <Loader2 size={18} className="spin" /> : <Send size={18} />}
              </button>
            </div>
          </div>
          <p className="chat-input-hint">Enter to send · Shift+Enter for new line · Type '/' for files</p>
        </div>
      </form>
    </div>
  )
}
