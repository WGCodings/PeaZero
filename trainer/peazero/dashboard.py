"""Live training dashboard: a tiny local web server that shows a run's
metrics.jsonl + status.json as statistics and charts.

train.py starts it automatically (see --dashboard-port). It can also be run
on its own to look at any run, finished or in progress:

    python -m peazero.dashboard --run ../runs/first
"""
import argparse
import json
import os
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HTML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard.html")
RUN_FILES = {"/metrics.jsonl": "metrics.jsonl", "/status.json": "status.json", "/config.json": "config.json",
             "/evals.jsonl": "evals.jsonl", "/eval_status.json": "eval_status.json"}
# Line logs that may not exist yet: served empty instead of 404.
OPTIONAL_LOGS = ("/metrics.jsonl", "/evals.jsonl")


class StatusWriter:
    """Writes <run>/status.json: what the trainer is doing right now. The
    dashboard polls it every couple of seconds; metrics.jsonl only gets a
    line at the end of each generation."""

    def __init__(self, run_dir, **static):
        self.path = os.path.join(run_dir, "status.json")
        self.state = {"session_start": time.time(), **static}
        self.last_write = 0.0

    def update(self, force=False, **fields):
        self.state.update(fields)
        now = time.time()
        if not force and now - self.last_write < 1.0:
            return
        self.last_write = now
        self.state["updated"] = now
        tmp = self.path + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(self.state, f)
            os.replace(tmp, self.path)
        except OSError:
            pass  # e.g. Windows: file briefly locked by the server; next update retries


def _handler(run):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                file, ctype = HTML_PATH, "text/html; charset=utf-8"
            elif path in RUN_FILES:
                file = os.path.join(run, RUN_FILES[path])
                ctype = "application/json" if path.endswith(".json") else "text/plain; charset=utf-8"
            else:
                self.send_error(404)
                return
            try:
                with open(file, "rb") as f:
                    body = f.read()
            except OSError:
                if path not in OPTIONAL_LOGS:
                    self.send_error(404)
                    return
                body = b""  # nothing logged yet
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass  # keep the training console clean

    return Handler


def start(run, port=8765, open_browser=True):
    """Serve the dashboard for `run` on http://localhost:<port> in a
    background thread. Returns the URL, or None if the port is taken."""
    try:
        server = ThreadingHTTPServer(("127.0.0.1", port), _handler(os.path.abspath(run)))
    except OSError as e:
        print(f"dashboard: could not listen on port {port} ({e}); use --dashboard-port to pick another")
        return None
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://localhost:{port}/"
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    return url


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    url = start(args.run, args.port, not args.no_browser)
    if url is None:
        return
    print(f"dashboard for {args.run}: {url}  (Ctrl+C to stop)")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
