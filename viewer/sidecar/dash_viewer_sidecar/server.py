"""Threading TCP server exposing a :class:`DashModelRuntime` over the protocol."""

from __future__ import annotations

import json
import socketserver
import threading

from .protocol import log, read_request, write_packet
from .runtime import DashModelRuntime


class DashRequestHandler(socketserver.BaseRequestHandler):
    runtime: DashModelRuntime
    shutdown_event: threading.Event

    def handle(self) -> None:
        sock = self.request
        try:
            while True:
                request = read_request(sock)
                cmd = request.get("cmd")

                if cmd == "info":
                    write_packet(sock, self.runtime.info(), b"")
                elif cmd == "frame":
                    t = float(request.get("time", 0.0))
                    payload = self.runtime.frame(t)
                    write_packet(
                        sock,
                        {
                            "ok": True,
                            "kind": "dash-frame",
                            "time": t % 1.0,
                            "gaussian_count": self.runtime.n,
                            "stride": self.runtime.info()["stride"],
                            "byte_len": len(payload),
                        },
                        payload,
                    )
                elif cmd == "shutdown":
                    write_packet(sock, {"ok": True, "kind": "shutdown"}, b"")
                    self.shutdown_event.set()
                    return
                else:
                    write_packet(sock, {"ok": False, "error": f"unknown cmd: {cmd}"}, b"")
        except EOFError:
            return
        except Exception as exc:  # noqa: BLE001
            log(f"request failed: {exc!r}")
            try:
                write_packet(sock, {"ok": False, "error": repr(exc)}, b"")
            except Exception:  # noqa: BLE001
                pass


class ThreadingDashServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def serve(runtime: DashModelRuntime, host: str, port: int) -> int:
    """Serve until a shutdown command arrives. Prints exactly one readiness line."""
    shutdown_event = threading.Event()

    class BoundHandler(DashRequestHandler):
        pass

    BoundHandler.runtime = runtime
    BoundHandler.shutdown_event = shutdown_event

    with ThreadingDashServer((host, port), BoundHandler) as server:
        bound_host, bound_port = server.server_address
        # The Rust dash-runtime waits for this exact stdout line.
        print(
            "DASH_SIDECAR_READY "
            + json.dumps({"host": bound_host, "port": bound_port}, separators=(",", ":")),
            flush=True,
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        shutdown_event.wait()
        server.shutdown()
        thread.join(timeout=2.0)

    return 0
