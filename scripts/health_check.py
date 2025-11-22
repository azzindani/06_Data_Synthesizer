#!/usr/bin/env python3
"""Simple health check HTTP server for monitoring."""

import json
import sys
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class HealthCheckHandler(BaseHTTPRequestHandler):
    """HTTP handler for health check requests."""

    def do_GET(self):
        """Handle GET requests."""
        if self.path == '/health' or self.path == '/':
            self.send_health_response()
        elif self.path == '/progress':
            self.send_progress_response()
        else:
            self.send_error(404, "Not Found")

    def send_health_response(self):
        """Send health check response."""
        health = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'service': 'data-synthesizer'
        }

        # Try to load progress for more details
        try:
            progress = self._load_progress()
            health['total_processed'] = progress.get('total_processed', 0)
            health['total_generated'] = progress.get('total_generated', 0)
            health['last_update'] = progress.get('last_update', 'unknown')
        except:
            pass

        self._send_json(200, health)

    def send_progress_response(self):
        """Send detailed progress response."""
        try:
            progress = self._load_progress()
            self._send_json(200, progress)
        except Exception as e:
            self._send_json(500, {'error': str(e)})

    def _load_progress(self):
        """Load progress from file."""
        progress_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'output',
            'synthesis_progress.json'
        )

        if os.path.exists(progress_file):
            with open(progress_file) as f:
                return json.load(f)

        return {'status': 'no progress file found'}

    def _send_json(self, status_code, data):
        """Send JSON response."""
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode())

    def log_message(self, format, *args):
        """Suppress default logging."""
        pass


def run_server(port: int = 8080):
    """Run health check server.

    Args:
        port: Port to listen on
    """
    server = HTTPServer(('0.0.0.0', port), HealthCheckHandler)
    print(f"Health check server running on port {port}")
    print(f"  - Health: http://localhost:{port}/health")
    print(f"  - Progress: http://localhost:{port}/progress")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped")
        server.shutdown()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Health check server")
    parser.add_argument('--port', type=int, default=8080, help='Port to listen on')
    args = parser.parse_args()

    run_server(args.port)
