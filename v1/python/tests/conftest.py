import importlib.util
import sys
from pathlib import Path

PY_DIR = Path(__file__).resolve().parents[1]
V1 = PY_DIR.parent
ROOT = V1.parent
if str(PY_DIR) not in sys.path:
    sys.path.insert(0, str(PY_DIR))


def load_script(filename):
    """Import a numbered script such as '01_manifest.py' (not importable by name)."""
    path = PY_DIR / filename
    spec = importlib.util.spec_from_file_location(Path(filename).stem.replace("-", "_").lstrip("0123456789_") or "script", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
