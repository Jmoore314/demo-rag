"""
Shared test fixtures and the sibling-import loader.

Why this file exists at all: every script in src/ and src_langchain/ uses
bare sibling imports (e.g. generate.py does `from retrieve import retrieve`,
not `from .retrieve import retrieve`) because they're written to be run
directly (`python3 src/generate.py`), not imported as a package -- Python
puts a script's own directory on sys.path automatically when you run it
that way. That's a fine, simple pattern for a repo meant to be read and run
stage-by-stage, but it means a plain `import src.generate` from a test at
the repo root won't work (src/ was never added to sys.path, so generate.py's
own `from retrieve import retrieve` fails), and worse: src/retrieve.py and
src_langchain/retrieve.py would collide under the same bare module name
"retrieve" if both directories ever ended up on sys.path at once, silently
returning whichever one got cached first.

`load_module()` below works around this the same way `python3 <script>`
does: put the target file's own directory at the front of sys.path just
long enough to import it (and, transitively, whatever it imports from a
sibling), then take it back off. It also evicts any previous copy of the
requested module name from sys.modules first, so importing "retrieve" from
src_langchain/ after already having imported "retrieve" from src/ (or vice
versa) gets the right one instead of a stale cached one.

A cleaner long-term fix would be converting src/ and src_langchain/ into
real packages with relative imports -- noted as a known limitation, not
done here since it would mean restructuring both directories.
"""

import importlib
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# Every basename that exists in BOTH src/ and src_langchain/ -- these are
# exactly the names load_module() must evict from sys.modules before each
# fresh import, or a second call could silently hand back the first
# directory's cached copy.
_SHARED_BASENAMES = {"ingest", "embed", "load_vectors", "retrieve", "generate"}


def load_module(subdir: str, modname: str):
    """Import <REPO_ROOT>/<subdir>/<modname>.py as top-level module `modname`,
    exactly the way running it as a script would see it -- including its own
    bare sibling imports resolving against its own directory."""
    target_dir = str(REPO_ROOT / subdir)

    for name in _SHARED_BASENAMES:
        sys.modules.pop(name, None)

    sys.path.insert(0, target_dir)
    try:
        module = importlib.import_module(modname)
    finally:
        sys.path.remove(target_dir)
    return module


@pytest.fixture
def fake_api_env(monkeypatch):
    """Fake-but-present values for every API/DB env var the pipelines check
    for via _require_env(). Needed before importing src_langchain/generate.py
    specifically, since it calls _require_env(...) AND constructs a real
    ChatAnthropic(...) client object at module import time (not deferred to
    a function) -- see that file's own comment on why. Client construction
    just stores the key locally; it doesn't make a network call, so a fake
    string is enough to get past both without hitting a real API."""
    monkeypatch.setenv("VOYAGE_API_KEY", "test-fake-voyage-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-fake-anthropic-key")
    monkeypatch.setenv("SUPABASE_DB_URL", "postgresql://fake:fake@localhost:5432/postgres")
