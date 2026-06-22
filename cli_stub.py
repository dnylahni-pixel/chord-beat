import sys
from pathlib import Path

def _add(p):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

def bootstrap_cli(file_path=None):
    root = Path(file_path).resolve().parents[2] if file_path else Path.cwd()
    _add(root / "src")
    _add(root)
    return root

def ensure_src_on_path(file_path=None):
    return bootstrap_cli(file_path)

# export everything ChordMini might need
setup_path = bootstrap_cli
add_src_to_path = bootstrap_cli
