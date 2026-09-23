#!/usr/bin/env python3
"""Measure the largest ASGI http.request body message uvicorn delivers under
memora-server's settings (the /api/v1 body bound is "cap + one message").

Runs uvicorn with exactly the options memora-server passes (http="h11",
loop="asyncio", h11_max_incomplete_event_size=16384) around a tiny ASGI app
that records every body message size, then sends it one large request body
over a loopback socket as fast as possible (chunked, no Content-Length), and
prints the largest message seen. Run it with the interpreter the container
uses (python:3.12-slim): the result is a property of that interpreter's
asyncio transport, not of memora.

Usage: python scripts/measure_asgi_body_messages.py [--mib 4]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import socket
import sys
import threading
import time

import uvicorn

UVICORN_OPTIONS = {"http": "h11", "loop": "asyncio", "h11_max_incomplete_event_size": 16 * 1024}
sizes = []


async def app(scope, receive, send):
    if scope["type"] != "http":
        return
    while True:
        message = await receive()
        if message["type"] != "http.request":
            break
        sizes.append(len(message.get("body") or b""))
        if not message.get("more_body"):
            break
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": b"ok"})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mib", type=int, default=4)
    args = ap.parse_args(argv)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning",
                                           **UVICORN_OPTIONS))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.02)
    payload = b"x" * (args.mib * 1024 * 1024)
    with socket.create_connection(("127.0.0.1", port)) as c:
        c.sendall(b"POST / HTTP/1.1\r\nHost: t\r\nTransfer-Encoding: chunked\r\n\r\n")
        c.sendall(b"%x\r\n" % len(payload) + payload + b"\r\n0\r\n\r\n")
        c.recv(1024)
    server.should_exit = True
    thread.join(timeout=5)
    print(json.dumps({
        "python": sys.version.split()[0], "uvicorn": uvicorn.__version__, "options": UVICORN_OPTIONS,
        "body_bytes": len(payload), "messages": len(sizes), "largest_message": max(sizes or [0]),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
