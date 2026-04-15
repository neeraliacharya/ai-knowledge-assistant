import { useState, useEffect, useRef } from 'react'
import { FileText, MessageSquare, Brain, Menu, X } from 'lucide-react'
import DocumentsTab from './components/DocumentsTab'
import ChatTab from './components/ChatTab'

const TABS = [
  { id: 'chat', label: 'AI Chat', icon: MessageSquare },
  { id: 'documents', label: 'Documents', icon: FileText },
]

export default function App() {
  const [activeTab, setActiveTab] = useState('chat')
  const [menuOpen, setMenuOpen] = useState(false)
  const sidebarRef = useRef(null)

  // Close sidebar when clicking outside on mobile
  useEffect(() => {
    const handleClick = (e) => {
      if (menuOpen && sidebarRef.current && !sidebarRef.current.contains(e.target)) {
        setMenuOpen(false)
      }
    }
    document.addEventListener('mousedown', handleClick)
    return () => document.removeEventListener('mousedown', handleClick)
  }, [menuOpen])

  const switchTab = (id) => {
    setActiveTab(id)
    setMenuOpen(false)
  }

  return (
    <div className="app">
      {/* Mobile overlay backdrop */}
      {menuOpen && <div className="sidebar-backdrop" onClick={() => setMenuOpen(false)} />}

      {/* Sidebar */}
      <aside ref={sidebarRef} className={`sidebar ${menuOpen ? 'sidebar--open' : ''}`}>
        <div className="sidebar__brand">
          <div className="sidebar__logo">
            <Brain size={20} />
          </div>
          <div className="sidebar__brand-text">
            <p className="sidebar__name">Knowledge AI</p>
            <p className="sidebar__version">v1.0</p>
          </div>
          {/* Close button — mobile only */}
          <button className="sidebar__close" onClick={() => setMenuOpen(false)} aria-label="Close menu">
            <X size={18} />
          </button>
        </div>

        <nav className="sidebar__nav">
          <p className="sidebar__section-label">Navigation</p>
          {TABS.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              className={`sidebar__item ${activeTab === id ? 'sidebar__item--active' : ''}`}
              onClick={() => switchTab(id)}
            >
              <Icon size={17} />
              <span>{label}</span>
            </button>
          ))}
        </nav>

        <div className="sidebar__footer">
          <div className="status-dot status-dot--online" />
          <span>Backend connected</span>
        </div>
      </aside>

      {/* Main */}
      <main className="main">
        {/* Mobile topbar */}
        <div className="topbar">
          <button className="topbar__menu-btn" onClick={() => setMenuOpen(true)} aria-label="Open menu">
            <Menu size={20} />
          </button>
          <span className="topbar__title">
            {TABS.find(t => t.id === activeTab)?.label}
          </span>
          <div className="topbar__spacer" />
        </div>

        {activeTab === 'documents' && <DocumentsTab />}
        {activeTab === 'chat' && <ChatTab />}
      </main>
    </div>
  )
}
