"""Put the geo-prewarp worktree root on sys.path so `import shape_library_3d`
and `import heatr3d` resolve (mirrors solve3d/tests/conftest.py)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # <worktree>/shape_library_3d/tests -> <worktree>
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
