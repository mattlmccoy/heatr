import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "heatr3d_d1_spike")):
    if p not in sys.path:
        sys.path.insert(0, p)
