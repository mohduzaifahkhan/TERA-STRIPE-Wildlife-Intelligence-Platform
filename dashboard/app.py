"""
TERA-STRIPE Dashboard -- Tactical Wildlife Intelligence Console
=================================================================
Serves a single-page HTML5 dashboard with Leaflet GIS, live alerts,
HITL review queue, tiger dossier gallery, and NTCA export controls.

Launch:
  python dashboard/app.py
  -> opens http://localhost:8501

Uses Python's built-in http.server with mock data fallback so
the dashboard renders fully even without a database connection.
"""

from __future__ import annotations

import json
import os
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs

# Add project root for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class DashboardHandler(SimpleHTTPRequestHandler):
    """Serves the dashboard HTML and API endpoints."""

    DASHBOARD_DIR = Path(__file__).parent

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/" or parsed.path == "/index.html":
            self._serve_file(self.DASHBOARD_DIR / "index.html", "text/html")
        elif parsed.path == "/api/feed":
            self._serve_api_feed()
        elif parsed.path == "/api/kpi":
            self._serve_api_kpi()
        elif parsed.path == "/api/alerts":
            self._serve_api_alerts()
        else:
            self.send_error(404)

    def _serve_file(self, path: Path, content_type: str):
        if not path.exists():
            self.send_error(404)
            return
        content = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _serve_json(self, data: dict):
        body = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _serve_api_feed(self):
        try:
            from src.m10_reporting import ReportGenerator
            gen = ReportGenerator()
            feed = gen.generate_dashboard_feed()
            self._serve_json(feed.model_dump())
        except Exception:
            self._serve_json({"error": "Feed generation failed"})

    def _serve_api_kpi(self):
        try:
            from src.m10_reporting import ReportGenerator
            kpi = ReportGenerator().mock_kpis()
            self._serve_json(kpi.model_dump())
        except Exception:
            self._serve_json({})

    def _serve_api_alerts(self):
        try:
            from src.m10_reporting import ReportGenerator
            gen = ReportGenerator()
            feed = gen.generate_dashboard_feed()
            self._serve_json({"alerts": feed.alerts})
        except Exception:
            self._serve_json({"alerts": []})

    def log_message(self, format, *args):
        pass  # Suppress default logging


def main():
    port = int(os.environ.get("DASHBOARD_PORT", "8501"))
    server = HTTPServer(("0.0.0.0", port), DashboardHandler)
    print(f"\n  TERA-STRIPE Tactical Console")
    print(f"  http://localhost:{port}")
    print(f"  Press Ctrl+C to stop\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Dashboard stopped.")
        server.server_close()


if __name__ == "__main__":
    main()
