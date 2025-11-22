"""Notification system for alerts and completion notices."""

import os
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional, Dict, Any
from datetime import datetime

import requests

from ..core.logger import get_logger


class NotificationManager:
    """Manages notifications via Slack and Email."""

    def __init__(self, config: Any = None):
        """Initialize notification manager.

        Args:
            config: Configuration object
        """
        self.logger = get_logger(__name__)

        # Load from config or environment
        if config and hasattr(config, 'notifications'):
            notif_config = config.notifications
            self.slack_webhook = getattr(notif_config, 'slack_webhook', '') or os.getenv('SLACK_WEBHOOK_URL', '')
            self.email_to = getattr(notif_config, 'email', '') or os.getenv('NOTIFICATION_EMAIL', '')
        else:
            self.slack_webhook = os.getenv('SLACK_WEBHOOK_URL', '')
            self.email_to = os.getenv('NOTIFICATION_EMAIL', '')

        # Email settings from environment
        self.smtp_host = os.getenv('SMTP_HOST', 'smtp.gmail.com')
        self.smtp_port = int(os.getenv('SMTP_PORT', '587'))
        self.smtp_user = os.getenv('SMTP_USER', '')
        self.smtp_password = os.getenv('SMTP_PASSWORD', '')

    def send_slack(self, message: str, title: str = None, color: str = None) -> bool:
        """Send notification to Slack.

        Args:
            message: Message text
            title: Optional title
            color: Optional color (good, warning, danger)

        Returns:
            True if sent successfully
        """
        if not self.slack_webhook:
            self.logger.debug("Slack webhook not configured")
            return False

        try:
            # Build payload
            if title or color:
                payload = {
                    "attachments": [{
                        "title": title or "Data Synthesizer",
                        "text": message,
                        "color": color or "good",
                        "ts": datetime.now().timestamp()
                    }]
                }
            else:
                payload = {"text": message}

            response = requests.post(
                self.slack_webhook,
                json=payload,
                timeout=10
            )

            if response.status_code == 200:
                self.logger.debug("Slack notification sent")
                return True
            else:
                self.logger.warning(f"Slack notification failed: {response.status_code}")
                return False

        except Exception as e:
            self.logger.error(f"Slack notification error: {e}")
            return False

    def send_email(self, subject: str, body: str, html: bool = False) -> bool:
        """Send notification via email.

        Args:
            subject: Email subject
            body: Email body
            html: Whether body is HTML

        Returns:
            True if sent successfully
        """
        if not all([self.email_to, self.smtp_user, self.smtp_password]):
            self.logger.debug("Email not configured")
            return False

        try:
            msg = MIMEMultipart()
            msg['From'] = self.smtp_user
            msg['To'] = self.email_to
            msg['Subject'] = subject

            content_type = 'html' if html else 'plain'
            msg.attach(MIMEText(body, content_type))

            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.send_message(msg)

            self.logger.debug("Email notification sent")
            return True

        except Exception as e:
            self.logger.error(f"Email notification error: {e}")
            return False

    def notify(self, message: str, title: str = None,
               level: str = "info", include_email: bool = True) -> Dict[str, bool]:
        """Send notification via all configured channels.

        Args:
            message: Notification message
            title: Optional title
            level: Notification level (info, warning, error, success)
            include_email: Whether to send email

        Returns:
            Dict with success status for each channel
        """
        results = {}

        # Map level to Slack color
        color_map = {
            'info': None,
            'success': 'good',
            'warning': 'warning',
            'error': 'danger'
        }
        color = color_map.get(level)

        # Send Slack
        results['slack'] = self.send_slack(message, title, color)

        # Send Email
        if include_email:
            subject = f"[Data Synthesizer] {title or level.upper()}"
            results['email'] = self.send_email(subject, message)

        return results

    def notify_completion(self, stats: Dict[str, Any]) -> Dict[str, bool]:
        """Send completion notification with statistics.

        Args:
            stats: Completion statistics

        Returns:
            Notification results
        """
        title = "Synthesis Complete"

        message = f"""
Data synthesis completed successfully!

📊 Statistics:
• Items Processed: {stats.get('total_processed', 0):,}
• Items Generated: {stats.get('total_generated', 0):,}
• API Requests: {stats.get('request_count', 0):,}
• Duration: {stats.get('duration', 'N/A')}

💰 Cost Summary:
• Total Cost: ${stats.get('total_cost', 0):.4f}

✅ Output: {stats.get('output_location', 'N/A')}
"""

        return self.notify(message, title, level="success")

    def notify_error(self, error: str, context: Dict[str, Any] = None) -> Dict[str, bool]:
        """Send error notification.

        Args:
            error: Error message
            context: Optional context information

        Returns:
            Notification results
        """
        title = "Synthesis Error"

        message = f"❌ Error occurred during synthesis:\n\n{error}"

        if context:
            message += f"\n\nContext:\n"
            for key, value in context.items():
                message += f"• {key}: {value}\n"

        return self.notify(message, title, level="error")

    def notify_progress(self, progress: Dict[str, Any]) -> Dict[str, bool]:
        """Send progress update notification.

        Args:
            progress: Progress information

        Returns:
            Notification results
        """
        title = "Progress Update"

        processed = progress.get('processed', 0)
        total = progress.get('total', 0)
        pct = (processed / total * 100) if total > 0 else 0

        message = f"""
📈 Synthesis Progress Update

Progress: {processed:,}/{total:,} ({pct:.1f}%)
Requests: {progress.get('requests', 0):,}
Current Topic: {progress.get('current_topic', 'N/A')}
"""

        return self.notify(message, title, level="info", include_email=False)


if __name__ == "__main__":
    print("=" * 60)
    print("NOTIFICATION MANAGER TEST")
    print("=" * 60)

    manager = NotificationManager()

    # Test 1: Check configuration
    print(f"  Slack configured: {'✓' if manager.slack_webhook else '✗'}")
    print(f"  Email configured: {'✓' if manager.email_to else '✗'}")

    # Test 2: Send test notification (if configured)
    if manager.slack_webhook:
        result = manager.send_slack("Test notification from Data Synthesizer", "Test")
        print(f"  {'✓' if result else '✗'} Slack test: {result}")
    else:
        print("  ⚠ Slack not configured, skipping test")

    # Test 3: Completion notification format
    stats = {
        'total_processed': 100,
        'total_generated': 500,
        'request_count': 100,
        'duration': '2h 30m',
        'total_cost': 0.0234,
        'output_location': 'username/dataset'
    }

    results = manager.notify_completion(stats)
    print(f"  ✓ Completion notification prepared")

    print("=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
