#!/usr/bin/env python3
"""Thin launcher invoked by the Rust ``dash-runtime`` (DASH_SIDECAR_SCRIPT).

It is spawned as ``python dash_sidecar.py --dash-root ... --model-dir ... \\
--iteration N --mask-mode M --host 127.0.0.1 --port 0`` and starts the TCP
server. The SOLID implementation lives in the ``dash_viewer_sidecar`` package
next to this file; humans can also use ``python -m dash_viewer_sidecar
{serve|info|bake}``.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dash_viewer_sidecar.cli import serve_main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(serve_main(sys.argv[1:]))
