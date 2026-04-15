# AI Knowledge Assistant — UI

React frontend for the AI Knowledge Assistant backend.

## Setup

```bash
cd ui
npm install
npm run dev
```

Open http://localhost:3000

## Backend expectations

The Vite dev server proxies these routes to `http://127.0.0.1:8000/`:

| Route | Method | Description |
|---|---|---|
| `/documents` | GET | Returns list of documents (array of strings or `{name, size}` objects) |
| `/upload` | POST | Multipart form upload with field `file` |
| `/documents/:name` | DELETE | Delete document by filename |
| `/ask?question=...` | GET | Returns `{ "answer": "..." }` |

## Build for production

```bash
npm run build
# Output goes to ui/dist/
```
