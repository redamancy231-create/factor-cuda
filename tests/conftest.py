# factor-cuda -- pytest conftest (Phase 2 acceptance suite).
# Makes the repo root, the benchmark layer and the frozen fixtures importable
# from tests/. ASCII-only comments (Windows GBK-safe).
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "benchmarks", ROOT / "tests" / "fixtures"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)
