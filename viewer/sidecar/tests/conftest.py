import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SIDECAR = os.path.dirname(HERE)  # viewer/sidecar
DASH_ROOT = os.environ.get("DASH_ROOT") or os.path.abspath(os.path.join(SIDECAR, "..", ".."))

for p in (SIDECAR, DASH_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)
