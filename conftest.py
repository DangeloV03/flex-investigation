"""Pytest bootstrap: make the coex/ and susceptibility/ source folders importable.

Source lives in two package folders (coex/, susceptibility/) rather than at the
repo root. Tests use bare imports (e.g. `import queue_manifest`,
`from susceptibility_runner import ...`); adding both folders (and the root) to
sys.path here mirrors the PYTHONPATH set by env.sh so `pytest` resolves them
without per-test path hacks.
"""

import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
for _sub in ("coex", "susceptibility", ""):
    _path = os.path.join(_ROOT, _sub) if _sub else _ROOT
    if _path not in sys.path:
        sys.path.insert(0, _path)
