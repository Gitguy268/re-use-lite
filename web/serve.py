#!/usr/bin/env python3
"""Serve this folder on http://localhost:8737 with the COOP/COEP headers the page needs."""
import http.server
import os
import sys

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8737
os.chdir(os.path.dirname(os.path.abspath(__file__)))


class H(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map,
                      ".wasm": "application/wasm", ".js": "text/javascript", ".gguf": "application/octet-stream"}

    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()


print(f"RE-USE Lite: http://localhost:{PORT}/  (Ctrl+C to stop)")
http.server.ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
