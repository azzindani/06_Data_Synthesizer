"""Simple web dashboard for monitoring synthesis progress."""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from flask import Flask, render_template_string, jsonify

from ..core.logger import get_logger
from ..core.progress import ProgressManager
from ..utils.cost_tracker import CostTracker


app = Flask(__name__)
logger = get_logger(__name__)

# Dashboard HTML template
DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Data Synthesizer Dashboard</title>
    <meta http-equiv="refresh" content="30">
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        h1 {
            color: #333;
            border-bottom: 2px solid #007bff;
            padding-bottom: 10px;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .card {
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .card h2 {
            margin-top: 0;
            color: #555;
            font-size: 1.1em;
            text-transform: uppercase;
        }
        .stat {
            font-size: 2.5em;
            font-weight: bold;
            color: #007bff;
        }
        .stat-label {
            color: #666;
            font-size: 0.9em;
        }
        .progress-bar {
            background: #e9ecef;
            border-radius: 4px;
            height: 20px;
            overflow: hidden;
            margin: 10px 0;
        }
        .progress-fill {
            background: #28a745;
            height: 100%;
            transition: width 0.3s ease;
        }
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            padding: 8px 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background: #f8f9fa;
        }
        .status-running {
            color: #28a745;
        }
        .status-stopped {
            color: #dc3545;
        }
        .timestamp {
            color: #666;
            font-size: 0.8em;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔄 Data Synthesizer Dashboard</h1>
        <p class="timestamp">Last updated: {{ timestamp }}</p>

        <div class="grid">
            <div class="card">
                <h2>Progress</h2>
                <div class="stat">{{ progress.percentage }}%</div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {{ progress.percentage }}%"></div>
                </div>
                <div class="stat-label">{{ progress.processed }} / {{ progress.total }} items</div>
            </div>

            <div class="card">
                <h2>Generated</h2>
                <div class="stat">{{ stats.total_generated }}</div>
                <div class="stat-label">Total items generated</div>
            </div>

            <div class="card">
                <h2>API Requests</h2>
                <div class="stat">{{ stats.total_requests }}</div>
                <div class="stat-label">Total API calls</div>
            </div>

            <div class="card">
                <h2>Cost</h2>
                <div class="stat">${{ "%.4f"|format(cost.total_cost) }}</div>
                <div class="stat-label">Total spend</div>
            </div>
        </div>

        <div class="grid">
            <div class="card">
                <h2>Provider Usage</h2>
                <table>
                    <tr>
                        <th>Provider</th>
                        <th>Requests</th>
                        <th>Tokens</th>
                        <th>Cost</th>
                    </tr>
                    {% for provider, data in cost.by_provider.items() %}
                    <tr>
                        <td>{{ provider }}</td>
                        <td>{{ data.requests }}</td>
                        <td>{{ data.total_tokens }}</td>
                        <td>${{ "%.4f"|format(data.cost) }}</td>
                    </tr>
                    {% endfor %}
                </table>
            </div>

            <div class="card">
                <h2>Recent Activity</h2>
                <table>
                    <tr>
                        <th>Time</th>
                        <th>Event</th>
                    </tr>
                    {% for event in recent_events %}
                    <tr>
                        <td>{{ event.time }}</td>
                        <td>{{ event.message }}</td>
                    </tr>
                    {% endfor %}
                </table>
            </div>
        </div>

        <div class="card">
            <h2>System Status</h2>
            <table>
                <tr>
                    <td>Status</td>
                    <td class="status-{{ 'running' if status.running else 'stopped' }}">
                        {{ 'Running' if status.running else 'Stopped' }}
                    </td>
                </tr>
                <tr>
                    <td>Current Provider</td>
                    <td>{{ status.current_provider }}</td>
                </tr>
                <tr>
                    <td>Uptime</td>
                    <td>{{ status.uptime }}</td>
                </tr>
                <tr>
                    <td>Errors</td>
                    <td>{{ status.error_count }}</td>
                </tr>
            </table>
        </div>
    </div>
</body>
</html>
"""


class DashboardServer:
    """Web dashboard server for monitoring."""

    def __init__(self, config: Any = None, port: int = 8080):
        """Initialize dashboard server.

        Args:
            config: Configuration object
            port: Server port
        """
        self.config = config
        self.port = port
        self.logger = get_logger(__name__)

        # State tracking
        self._start_time = datetime.now()
        self._events = []
        self._running = False

        # Data sources
        self._progress_manager = None
        self._cost_tracker = None

        self._setup_routes()

    def _setup_routes(self):
        """Setup Flask routes."""

        @app.route('/')
        def index():
            return render_template_string(
                DASHBOARD_TEMPLATE,
                timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                progress=self._get_progress(),
                stats=self._get_stats(),
                cost=self._get_cost(),
                recent_events=self._get_recent_events(),
                status=self._get_status()
            )

        @app.route('/api/status')
        def api_status():
            return jsonify({
                'progress': self._get_progress(),
                'stats': self._get_stats(),
                'cost': self._get_cost(),
                'status': self._get_status()
            })

        @app.route('/api/health')
        def api_health():
            return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

    def _get_progress(self) -> Dict[str, Any]:
        """Get progress information."""
        if self._progress_manager:
            progress = self._progress_manager.get_progress()
            total = progress.get('total_items', 0)
            processed = progress.get('processed_count', 0)
            percentage = (processed / total * 100) if total > 0 else 0
            return {
                'processed': processed,
                'total': total,
                'percentage': round(percentage, 1)
            }
        return {'processed': 0, 'total': 0, 'percentage': 0}

    def _get_stats(self) -> Dict[str, Any]:
        """Get synthesis statistics."""
        if self._progress_manager:
            progress = self._progress_manager.get_progress()
            return {
                'total_generated': progress.get('total_generated', 0),
                'total_requests': progress.get('request_count', 0)
            }
        return {'total_generated': 0, 'total_requests': 0}

    def _get_cost(self) -> Dict[str, Any]:
        """Get cost information."""
        if self._cost_tracker:
            return self._cost_tracker.get_summary()
        return {
            'total_cost': 0,
            'by_provider': {}
        }

    def _get_recent_events(self) -> list:
        """Get recent events."""
        return self._events[-10:]

    def _get_status(self) -> Dict[str, Any]:
        """Get system status."""
        uptime = datetime.now() - self._start_time
        hours = int(uptime.total_seconds() // 3600)
        minutes = int((uptime.total_seconds() % 3600) // 60)

        return {
            'running': self._running,
            'current_provider': 'gemini',
            'uptime': f'{hours}h {minutes}m',
            'error_count': 0
        }

    def add_event(self, message: str):
        """Add event to recent activity."""
        self._events.append({
            'time': datetime.now().strftime('%H:%M:%S'),
            'message': message
        })
        # Keep only last 100 events
        if len(self._events) > 100:
            self._events = self._events[-100:]

    def set_progress_manager(self, manager: ProgressManager):
        """Set progress manager for data."""
        self._progress_manager = manager

    def set_cost_tracker(self, tracker: CostTracker):
        """Set cost tracker for data."""
        self._cost_tracker = tracker

    def run(self, debug: bool = False):
        """Run the dashboard server."""
        self._running = True
        self.logger.info(f"Starting dashboard on port {self.port}")
        app.run(host='0.0.0.0', port=self.port, debug=debug)


def create_dashboard(config: Any = None, port: int = 8080) -> DashboardServer:
    """Create dashboard server instance.

    Args:
        config: Configuration object
        port: Server port

    Returns:
        DashboardServer instance
    """
    return DashboardServer(config, port)


if __name__ == "__main__":
    print("=" * 60)
    print("DASHBOARD SERVER TEST")
    print("=" * 60)

    # Create and run dashboard
    dashboard = create_dashboard(port=8080)
    print(f"  Dashboard available at http://localhost:8080")
    print("  Press Ctrl+C to stop")

    try:
        dashboard.run(debug=True)
    except KeyboardInterrupt:
        print("\n  Dashboard stopped")

    print("=" * 60)
