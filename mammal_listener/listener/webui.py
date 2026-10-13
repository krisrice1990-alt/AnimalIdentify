"""Lightweight bug-report web UI for the Mammal Listener HA app.

Stdlib only (http.server). Serves a dark-themed form at / where the user
describes a bug; on submit it builds a pre-filled GitHub issue URL
(title + body URL-encoded, with auto-collected diagnostics) and shows a
"Submit to GitHub" button that opens the URL in a new tab.

Nothing is posted automatically and no credentials are embedded: the
user reviews and submits the issue themselves in their own browser.
"""

import html
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlencode

ISSUE_URL = "https://github.com/krisrice1990-alt/AnimalIdentify/issues/new"

# Must match `version:` in config.yaml. Bump with each release.
APP_VERSION = "1.0.0"

# Option keys safe to include in a public bug report. Anything not on
# this list (mqtt_host/user/password, rtsp_url, frigate_host) is dropped
# so LAN credentials never leak into a GitHub issue body.
SAFE_DIAG_KEYS = (
    "audio_source",
    "frigate_enabled",
    "frigate_camera",
    "frigate_cameras",
    "energy_threshold_db",
    "confidence_threshold",
    "cooldown_minutes",
    "base_topic",
    "model_path",
    "webui_port",
)


def collect_diagnostics(opt):
    """Build a safe diagnostics dict from the app options."""
    diag = {"app_version": APP_VERSION}
    diag["timestamp_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if isinstance(opt, dict):
        for key in SAFE_DIAG_KEYS:
            if key in opt:
                diag[key] = opt[key]
    return diag


def format_diagnostics(diag):
    lines = ["### Diagnostics (auto-collected)"]
    for key, value in diag.items():
        lines.append(f"- {key}: {value}")
    return "\n".join(lines)


def build_issue_url(title, description, steps, diag):
    """Return the pre-filled GitHub issue URL for the report."""
    body = (
        f"### Description\n{description}\n\n"
        f"### Steps to reproduce\n{steps}\n\n"
        f"{format_diagnostics(diag)}"
    )
    return ISSUE_URL + "?" + urlencode({"title": title, "body": body})


PAGE_CSS = """
body{background:#121212;color:#e0e0e0;font-family:system-ui,sans-serif;
max-width:640px;margin:2rem auto;padding:0 1rem}
h1{font-size:1.4rem}label{display:block;margin-top:1rem;font-weight:600}
input[type=text],textarea{width:100%;margin-top:.4rem;padding:.6rem;
background:#1e1e1e;border:1px solid #333;color:#e0e0e0;border-radius:6px;
box-sizing:border-box}textarea{min-height:90px}
button{margin-top:1.2rem;padding:.7rem 1.4rem;background:#2f81f7;border:0;
color:#fff;font-size:1rem;border-radius:6px;cursor:pointer}
button:hover{background:#1f6feb}.diag{margin-top:1.5rem;font-size:.85rem;
color:#9e9e9e;background:#1a1a1a;padding:.8rem;border-radius:6px}
.diag pre{white-space:pre-wrap;margin:.4rem 0 0}
a.btn{display:inline-block;margin-top:1.2rem;padding:.7rem 1.4rem;
background:#238636;color:#fff;text-decoration:none;border-radius:6px}
a.btn:hover{background:#2ea043}
"""


def render_form(diag):
    diag_html = html.escape(format_diagnostics(diag))
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport"
content="width=device-width,initial-scale=1">
<title>Mammal Listener — Bug Report</title>
<style>{PAGE_CSS}</style></head>
<body>
<h1>&#x1f41e; Mammal Listener — Report a bug</h1>
<p>Describe the problem below. Submitting opens a pre-filled GitHub issue
in a new tab — nothing is sent until you click "Submit new issue" there.</p>
<form method="post" action="/submit">
<label>Title<input type="text" name="title" required
placeholder="e.g. No detections from Frigate camera"></label>
<label>Description<textarea name="description" required
placeholder="What happened? What did you expect?"></textarea></label>
<label>Steps to reproduce<textarea name="steps"
placeholder="1. ...&#10;2. ..."></textarea></label>
<button type="submit">Generate GitHub issue</button>
</form>
<div class="diag"><strong>Diagnostics included automatically:</strong>
<pre>{diag_html}</pre></div>
</body></html>"""


def render_result(issue_url, title):
    safe_title = html.escape(title)
    safe_url = html.escape(issue_url, quote=True)
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport"
content="width=device-width,initial-scale=1">
<title>Mammal Listener — Bug Report ready</title>
<style>{PAGE_CSS}</style></head>
<body>
<h1>&#x1f41e; Report ready</h1>
<p>Your bug report <strong>"{safe_title}"</strong> is pre-filled.
Click below to open it on GitHub and submit:</p>
<a class="btn" href="{safe_url}" target="_blank" rel="noopener">
Open pre-filled GitHub issue</a>
<p style="margin-top:1rem"><a href="/" style="color:#58a6ff">
&larr; Back to form</a></p>
</body></html>"""


def make_handler(diag):
    class BugReportHandler(BaseHTTPRequestHandler):
        server_version = "MammalListener-BugUI/1.0"

        def log_message(self, fmt, *args):  # quieter than the default
            print(f"[webui] {self.address_string()} - {fmt % args}", flush=True)

        def _send(self, body, code=200):
            data = body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path.split("?")[0] == "/":
                self._send(render_form(diag))
            else:
                self._send("<h1>404</h1>", code=404)

        def do_POST(self):
            if self.path.split("?")[0] != "/submit":
                self._send("<h1>404</h1>", code=404)
                return
            length = int(self.headers.get("Content-Length", 0) or 0)
            fields = parse_qs(
                self.rfile.read(length).decode("utf-8", "replace"),
                keep_blank_values=True,
            )
            title = fields.get("title", [""])[0].strip() or "Bug report"
            description = fields.get("description", [""])[0].strip()
            steps = fields.get("steps", [""])[0].strip() or "(not provided)"
            url = build_issue_url(title, description, steps, diag)
            self._send(render_result(url, title))

    return BugReportHandler


def start(opt, version=None):
    """Start the bug-report web UI in a daemon thread.

    Returns the HTTPServer instance, or None if disabled / failed.
    Never raises: the detector must keep running even if the UI can't.
    """
    if version:
        global APP_VERSION
        APP_VERSION = version
    try:
        if not bool(opt.get("webui_enabled", True)):
            print("[webui] disabled via webui_enabled=false", flush=True)
            return None
        port = int(opt.get("webui_port", 8099) or 8099)
    except (TypeError, ValueError) as e:
        print(f"[webui] bad webui_port config ({e}); UI not started", flush=True)
        return None
    diag = collect_diagnostics(opt)
    try:
        server = HTTPServer(("", port), make_handler(diag))
    except OSError as e:
        print(f"[webui] could not bind port {port} ({e}); UI not started",
              flush=True)
        return None
    t = threading.Thread(target=server.serve_forever, name="mammal-webui",
                         daemon=True)
    t.start()
    print(f"[webui] bug-report UI listening on port {port}", flush=True)
    return server
