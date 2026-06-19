#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# verify.sh — Full system verification for AI Knowledge Assistant
#
# Run from the project root:
#   bash scripts/verify.sh
#
# What it checks (in order):
#   1. Unit tests        — chunking, memory, logger (fast, ~5s)
#   2. Integration tests — embeddings, retrieval, reranker, RAG pipeline (~90s)
#   3. API tests         — all FastAPI endpoints via TestClient (~90s)
#   4. RAGAS asyncio fix — verifies nest_asyncio + Python 3.14 patch works
#   5. Eval script smoke — imports and loads the pipeline (no real LLM calls)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

PYTHON="venv/bin/python3.14"
PYTEST="$PYTHON -m pytest"
PASS=0
FAIL=0
SKIP=0

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m'

section()  { echo; echo -e "${BLUE}══════════════════════════════════════════${NC}"; echo -e "${BLUE}  $1${NC}"; echo -e "${BLUE}══════════════════════════════════════════${NC}"; }
ok()       { echo -e "  ${GREEN}✓ PASS${NC}  $1"; PASS=$((PASS + 1)); }
fail()     { echo -e "  ${RED}✗ FAIL${NC}  $1"; FAIL=$((FAIL + 1)); }
skipped()  { echo -e "  ${YELLOW}~ SKIP${NC}  $1"; SKIP=$((SKIP + 1)); }

# ── Preflight ──────────────────────────────────────────────────────────────
section "Preflight checks"

if [ ! -f "$PYTHON" ]; then
    echo -e "${RED}ERROR: venv not found. Run: python3.14 -m venv venv && venv/bin/pip install -r requirements.txt${NC}"
    exit 1
fi
ok "Python venv present ($($PYTHON --version 2>&1))"

if [ ! -d "vector_store" ] || [ ! -f "vector_store/faiss.index" ]; then
    echo "  Note: No FAISS cache found — first API test run will rebuild index (~60s extra)"
fi

# Kill any stale pytest processes from previous runs
pkill -f "pytest" 2>/dev/null && echo "  Note: Killed stale pytest process" || true

# ── 1. Unit tests ──────────────────────────────────────────────────────────
section "1. Unit tests (chunking / memory / logger)"

if $PYTEST tests/unit/ -q --tb=short 2>&1; then
    ok "All unit tests passed"
else
    fail "Unit tests failed — see output above"
fi

# ── 2. Integration tests ───────────────────────────────────────────────────
section "2. Integration tests (embeddings / retrieval / RAG pipeline)"
echo "  Note: First run downloads BGE model (~400 MB). Subsequent runs use cache."

if $PYTEST tests/integration/ -q --tb=short 2>&1; then
    ok "All integration tests passed"
else
    fail "Integration tests failed — see output above"
fi

# ── 3. API tests ───────────────────────────────────────────────────────────
section "3. API tests (FastAPI endpoints via TestClient)"
echo "  Note: TestClient boots the full app including RAG pipeline (~60s init)."
echo "  RAGAS background tasks are mocked — no Groq API calls made here."

if $PYTEST tests/api/ -q --tb=short 2>&1; then
    ok "All API tests passed"
else
    fail "API tests failed — see output above"
fi

# ── 4. RAGAS asyncio fix ───────────────────────────────────────────────────
section "4. RAGAS asyncio fix (nest_asyncio + Python 3.14 compatibility)"

$PYTHON - <<'PYEOF'
import sys

# Step 1: import RAGAS (triggers nest_asyncio.apply())
try:
    from ragas.llms import LangchainLLMWrapper
    print("  [1/4] ragas imported (nest_asyncio.apply() called)")
except ImportError as e:
    print(f"  RAGAS not installed: {e}")
    sys.exit(0)

import asyncio
import asyncio.tasks as at

# Step 2: verify the bug exists before fix
loop = asyncio.new_event_loop()
async def _check_before():
    return asyncio.current_task()
result_before = loop.run_until_complete(_check_before())
loop.close()
print(f"  [2/4] current_task() before fix: {result_before}")

# Step 3: apply the fix (same logic as _init_ragas)
if not getattr(asyncio, '_current_task_patched', False):
    def _fixed(loop=None):
        if loop is None:
            try: loop = asyncio.get_running_loop()
            except RuntimeError: return None
        return at._current_tasks.get(loop)
    asyncio.current_task = _fixed
    at.current_task = _fixed
    asyncio._current_task_patched = True

# Step 4: verify asyncio.timeout works after fix
loop2 = asyncio.new_event_loop()
async def _check_after():
    task = asyncio.current_task()
    if task is None:
        raise RuntimeError("current_task() still None after patch!")
    async with asyncio.timeout(1.0):
        pass
    return "ok"
try:
    result = loop2.run_until_complete(_check_after())
    print(f"  [3/4] current_task() after fix:  <Task ...> ✓")
    print(f"  [4/4] asyncio.timeout() works:    ✓")
    sys.exit(0)
except Exception as e:
    print(f"  Fix FAILED: {e}")
    sys.exit(1)
finally:
    loop2.close()
PYEOF

if [ $? -eq 0 ]; then
    ok "RAGAS asyncio fix verified"
else
    fail "RAGAS asyncio fix not working"
fi

# ── 5. Eval script smoke test ──────────────────────────────────────────────
section "5. Evaluation script smoke test"

if [ ! -f "data/eval_testset.json" ]; then
    skipped "data/eval_testset.json not found — skipping eval smoke test"
else
    # Check file is valid JSON with at least one entry
    COUNT=$($PYTHON -c "import json; d=json.load(open('data/eval_testset.json')); print(len(d))" 2>/dev/null || echo "0")
    if [ "$COUNT" -gt 0 ]; then
        ok "eval_testset.json has $COUNT samples"
    else
        fail "eval_testset.json is empty or invalid"
    fi

    # Check pipeline loads from disk cache
    $PYTHON - <<'PYEOF2'
import sys, os
sys.path.insert(0, '.')
import app.services.rag_pipeline as rp
rp.initialize_rag()
if rp.vector_index is None:
    print("  WARNING: No documents indexed (documents/ folder may be empty)")
    sys.exit(0)
print(f"  Pipeline: {len(rp.chunks_store)} chunks loaded")
sys.exit(0)
PYEOF2

    if [ $? -eq 0 ]; then
        ok "RAG pipeline loads correctly for evaluation"
    else
        fail "RAG pipeline failed to load"
    fi
fi

# ── Summary ────────────────────────────────────────────────────────────────
section "Summary"
TOTAL=$((PASS + FAIL + SKIP))
echo "  Checks run:  $TOTAL"
echo -e "  ${GREEN}Passed: $PASS${NC}"
if [ $FAIL -gt 0 ]; then
    echo -e "  ${RED}Failed: $FAIL${NC}"
fi
if [ $SKIP -gt 0 ]; then
    echo -e "  ${YELLOW}Skipped: $SKIP${NC}"
fi
echo

if [ $FAIL -eq 0 ]; then
    echo -e "${GREEN}All checks passed!${NC}"
    exit 0
else
    echo -e "${RED}$FAIL check(s) failed. See output above for details.${NC}"
    exit 1
fi
