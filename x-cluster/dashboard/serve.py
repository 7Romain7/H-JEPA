#!/usr/bin/env python3
"""
Tableau de bord local des salles info, avec bouton « Scanner ».

  python dashboard/serve.py            # puis ouvrir http://localhost:8765

Tourne sur le PC (là où marchent les clés SSH) et n'écoute que sur 127.0.0.1.
  GET  /               -> salles.html
  GET  /snapshot.json  -> dernier relevé enregistré
  POST /api/scan       -> sonde les 169 machines (~8 s) et renvoie le nouveau relevé
"""
import json, sys, threading, webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from snapshot import SNAPSHOT, take_snapshot  # noqa: E402

PORT = 8765
scan_lock = threading.Lock()   # un seul scan à la fois, même avec plusieurs onglets


class Handler(BaseHTTPRequestHandler):
    def send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/salles.html"):
            self.send(200, (HERE / "salles.html").read_bytes(), "text/html; charset=utf-8")
        elif path == "/snapshot.json" and SNAPSHOT.exists():
            self.send(200, SNAPSHOT.read_bytes(), "application/json")
        elif path == "/api/ping":
            self.send(200, b'{"ok":true}', "application/json")
        else:
            self.send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path != "/api/scan":
            return self.send(404, b"not found", "text/plain")
        with scan_lock:
            snap = take_snapshot()
        self.send(200, json.dumps(snap, ensure_ascii=False).encode(), "application/json")

    def log_message(self, fmt, *args):
        if "/api/scan" in (args[0] if args else ""):
            sys.stderr.write("scan %s\n" % args[1] if len(args) > 1 else "")


def main():
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    url = f"http://localhost:{PORT}"
    print(f"Tableau de bord sur {url}  (Ctrl+C pour arrêter)")
    if "--no-browser" not in sys.argv:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
