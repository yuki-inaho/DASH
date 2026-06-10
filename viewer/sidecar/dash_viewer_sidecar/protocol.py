"""Length-prefixed TCP protocol (mirror of dash-runtime's wire format).

    request  = u32 LE json_len + json bytes
    response = u32 LE meta_len + meta json + u64 LE payload_len + payload bytes
"""

from __future__ import annotations

import json
import socket
import struct
import sys
from typing import Any, Dict

MAX_REQUEST_BYTES = 1_000_000


def log(message: str) -> None:
    """All diagnostics go to stderr; stdout is reserved for the readiness line."""
    print(f"[dash-sidecar] {message}", file=sys.stderr, flush=True)


def read_exact(sock: socket.socket, n: int) -> bytes:
    chunks = []
    remaining = n
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise EOFError("peer closed connection")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def read_request(sock: socket.socket) -> Dict[str, Any]:
    header = read_exact(sock, 4)
    (request_len,) = struct.unpack("<I", header)
    if request_len > MAX_REQUEST_BYTES:
        raise ValueError(f"request too large: {request_len}")
    return json.loads(read_exact(sock, request_len).decode("utf-8"))


def write_packet(sock: socket.socket, meta: Dict[str, Any], payload: bytes = b"") -> None:
    meta_bytes = json.dumps(meta, separators=(",", ":")).encode("utf-8")
    sock.sendall(struct.pack("<I", len(meta_bytes)))
    sock.sendall(meta_bytes)
    sock.sendall(struct.pack("<Q", len(payload)))
    if payload:
        sock.sendall(payload)
