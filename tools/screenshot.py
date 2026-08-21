"""Capture README screenshots from a running llmbench UI, stdlib only.

Drives a headless Chrome/Chromium over the DevTools protocol with a minimal
RFC 6455 WebSocket client, so regenerating the gallery needs no Node, no
Playwright, and no pip installs — the same zero-dependency rule as the app.

Usage:
    python bench.py serve &                       # the UI to photograph
    python tools/screenshot.py URL OUT.png [JS]   # JS runs before capture

The optional JS expression is awaited; use it to set the mode, open drawers,
or click into the state you want photographed, e.g.:
    python tools/screenshot.py http://127.0.0.1:8765 shot.png \
        "localStorage.setItem('llmbench.level','beginner'); applyLevel(); 'ok'"

Chrome is found via CHROME env var or common install paths. Capture is 1400 CSS
px wide at deviceScaleFactor 2, height fitted to the page.
"""
from __future__ import annotations

import base64
import json
import os
import socket
import struct
import subprocess
import sys
import tempfile
import time
import urllib.request

PORT = 9222
CHROME_CANDIDATES = [
    os.environ.get("CHROME", ""),
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Google Chrome Beta.app/Contents/MacOS/Google Chrome Beta",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/usr/bin/google-chrome", "/usr/bin/chromium", "/usr/bin/chromium-browser",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
]


class WS:
    """Just enough WebSocket to speak CDP: masked text frames out, frames in."""

    def __init__(self, url: str):
        _, rest = url.split("://", 1)
        hostport, path = rest.split("/", 1)
        host, port = hostport.split(":")
        self.sock = socket.create_connection((host, int(port)))
        key = base64.b64encode(os.urandom(16)).decode()
        self.sock.sendall((
            f"GET /{path} HTTP/1.1\r\nHost: {hostport}\r\nUpgrade: websocket\r\n"
            f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n\r\n").encode())
        buf = b""
        while b"\r\n\r\n" not in buf:
            buf += self.sock.recv(4096)
        self.buf = buf.split(b"\r\n\r\n", 1)[1]
        self.mid = 0

    def _send(self, payload: bytes) -> None:
        mask = os.urandom(4)
        n = len(payload)
        hdr = b"\x81"
        if n < 126:
            hdr += bytes([0x80 | n])
        elif n < 65536:
            hdr += bytes([0x80 | 126]) + struct.pack(">H", n)
        else:
            hdr += bytes([0x80 | 127]) + struct.pack(">Q", n)
        self.sock.sendall(hdr + mask +
                          bytes(b ^ mask[i % 4] for i, b in enumerate(payload)))

    def _read(self, n: int) -> bytes:
        while len(self.buf) < n:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise IOError("socket closed")
            self.buf += chunk
        out, self.buf = self.buf[:n], self.buf[n:]
        return out

    def call(self, method: str, **params):
        self.mid += 1
        self._send(json.dumps({"id": self.mid, "method": method,
                               "params": params}).encode())
        while True:
            _, b2 = self._read(2)
            ln = b2 & 0x7F
            if ln == 126:
                ln = struct.unpack(">H", self._read(2))[0]
            elif ln == 127:
                ln = struct.unpack(">Q", self._read(8))[0]
            msg = json.loads(self._read(ln))
            if msg.get("id") == self.mid:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})


def find_chrome() -> str:
    for c in CHROME_CANDIDATES:
        if c and os.path.exists(c):
            return c
    raise SystemExit("No Chrome/Chromium found; set CHROME=/path/to/chrome")


def start_chrome() -> "subprocess.Popen[bytes]":
    profile = tempfile.mkdtemp(prefix="llmbench-shot-")
    proc = subprocess.Popen(
        [find_chrome(), "--headless=new", f"--remote-debugging-port={PORT}",
         f"--user-data-dir={profile}", "--hide-scrollbars", "--disable-gpu",
         "--no-first-run", "--force-device-scale-factor=2",
         "--window-size=1400,1000", "about:blank"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(60):
        try:
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{PORT}/json/version", timeout=1):
                return proc
        except Exception:
            time.sleep(0.25)
    raise SystemExit("chrome did not expose the debug port")


def page_ws_url() -> str:
    # /json/new needs PUT on current Chrome; attach to the launch tab instead.
    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json/list",
                                timeout=5) as r:
        pages = [t for t in json.load(r) if t.get("type") == "page"]
    if not pages:
        raise SystemExit("no page target")
    return pages[0]["webSocketDebuggerUrl"]


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    url, out = sys.argv[1], sys.argv[2]
    setup_js = sys.argv[3] if len(sys.argv) > 3 else ""
    max_h = int(sys.argv[4]) if len(sys.argv) > 4 else 4000
    proc = start_chrome()
    try:
        ws = WS(page_ws_url())
        ws.call("Page.enable")
        ws.call("Runtime.enable")
        ws.call("Emulation.setDeviceMetricsOverride", width=1400, height=1000,
                deviceScaleFactor=2, mobile=False)
        ws.call("Page.navigate", url=url)
        time.sleep(3.0)
        if setup_js:
            r = ws.call("Runtime.evaluate", expression=setup_js,
                        awaitPromise=True, returnByValue=True)
            if r.get("exceptionDetails"):
                raise SystemExit(f"setup JS failed: {r['exceptionDetails']}")
            time.sleep(0.8)
        h = int(ws.call("Runtime.evaluate", returnByValue=True, expression=
                        f"Math.min(Math.ceil(document.body.scrollHeight),{max_h})"
                        )["result"]["value"])
        ws.call("Emulation.setDeviceMetricsOverride", width=1400, height=h,
                deviceScaleFactor=2, mobile=False)
        time.sleep(0.4)
        data = ws.call("Page.captureScreenshot", format="png",
                       captureBeyondViewport=False)["data"]
        with open(out, "wb") as f:
            f.write(base64.b64decode(data))
        print(f"wrote {out} (1400x{h} css px @2x)")
        return 0
    finally:
        proc.terminate()


if __name__ == "__main__":
    raise SystemExit(main())
