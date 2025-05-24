"""
Email notification system for trading bot alerts and reports.
"""

import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging
import os
from jinja2 import Template

from .decorators import retry_on_failure, rate_limit

logger = logging.getLogger(__name__)

class EmailNotifier:
    """Email notification system for trading alerts and reports."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize email notifier with configuration.
        
        Args:
            config: Email configuration dictionary
        """
        self.smtp_server = config.get('smtp_server', 'smtp.gmail.com')
        self.smtp_port = config.get('smtp_port', 587)
        self.email = config.get('email')
        self.password = config.get('password')
        self.recipients = config.get('recipients', [])
        self.enabled = config.get('enabled', False)
        
        if self.enabled and not all([self.email, self.password]):
            logger.warning("Email notifications enabled but credentials not provided")
            self.enabled = False
    
    @retry_on_failure(max_retries=3, delay=5)
    @rate_limit(max_calls=10, time_window=60)  # Limit to 10 emails per minute
    def send_email(self, 
                   subject: str, 
                   body: str, 
                   recipients: Optional[List[str]] = None,
                   html_body: Optional[str] = None,
                   attachments: Optional[List[str]] = None) -> bool:
        """
        Send email notification.
        
        Args:
            subject: Email subject
            body: Plain text body
            recipients: List of recipient emails (uses default if None)
            html_body: HTML formatted body (optional)
            attachments: List of file paths to attach
            
        Returns:
            bool: True if email sent successfully
        """
        if not self.enabled:
            logger.debug("Email notifications disabled")
            return False
        
        try:
            recipients = recipients or self.recipients
            if not recipients:
                logger.warning("No email recipients configured")
                return False
            
            # Create message
            msg = MIMEMultipart('alternative')
            msg['From'] = self.email
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = subject
            
            # Add plain text body
            msg.attach(MIMEText(body, 'plain'))
            
            # Add HTML body if provided
            if html_body:
                msg.attach(MIMEText(html_body, 'html'))
            
            # Add attachments if provided
            if attachments:
                for file_path in attachments:
                    if os.path.exists(file_path):
                        self._attach_file(msg, file_path)
                    else:
                        logger.warning(f"Attachment file not found: {file_path}")
            
            # Send email
            context = ssl.create_default_context()
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls(context=context)
                server.login(self.email, self.password)
                server.sendmail(self.email, recipients, msg.as_string())
            
            logger.info(f"Email sent successfully: {subject}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False
    
    def _attach_file(self, msg: MIMEMultipart, file_path: str):
        """Attach file to email message."""
        try:
            with open(file_path, 'rb') as attachment:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(attachment.read())
            
            encoders.encode_base64(part)
            filename = os.path.basename(file_path)
            part.add_header(
                'Content-Disposition',
                f'attachment; filename= {filename}'
            )
            msg.attach(part)
            
        except Exception as e:
            logger.error(f"Failed to attach file {file_path}: {e}")
    
    def send_trade_alert(self, 
                        trade_type: str, 
                        symbol: str, 
                        quantity: float, 
                        price: float, 
                        strategy: str,
                        reason: str = "") -> bool:
        """
        Send trade execution alert.
        
        Args:
            trade_type: 'BUY' or 'SELL'
            symbol: Trading symbol
            quantity: Trade quantity
            price: Execution price
            strategy: Strategy name
            reason: Optional reason for trade
            
        Returns:
            bool: True if sent successfully
        """
        subject = f"Trade Alert: {trade_type} {symbol}"
        
        body = f"""
Trading Bot Alert

Trade Executed:
- Action: {trade_type}
- Symbol: {symbol}
- Quantity: {quantity:,.2f}
- Price: ${price:.2f}
- Value: ${abs(quantity * price):,.2f}
- Strategy: {strategy}
- Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{f'Reason: {reason}' if reason else ''}

This is an automated message from your trading bot.
        """
        
        return self.send_email(subject, body)
    
    def send_risk_alert(self, 
                       alert_type: str, 
                       message: str, 
                       severity: str = "WARNING",
                       metrics: Optional[Dict] = None) -> bool:
        """
        Send risk management alert.
        
        Args:
            alert_type: Type of risk alert
            message: Alert message
            severity: Alert severity level
            metrics: Optional risk metrics dictionary
            
        Returns:
            bool: True if sent successfully
        """
        subject = f"Risk Alert ({severity}): {alert_type}"
        
        body = f"""
Risk Management Alert

Alert Type: {alert_type}
Severity: {severity}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Message:
{message}
"""
        
        if metrics:
            body += "\nRisk Metrics:\n"
            for key, value in metrics.items():
                if isinstance(value, float):
                    body += f"- {key}: {value:.4f}\n"
                else:
                    body += f"- {key}: {value}\n"
        
        body += "\nPlease review your trading positions and risk exposure."
        
        return self.send_email(subject, body)
    
    def send_performance_report(self, 
                              report_data: Dict,
                              period: str = "Daily",
                              html_report_path: Optional[str] = None) -> bool:
        """
        Send performance report.
        
        Args:
            report_data: Performance metrics dictionary
            period: Report period (Daily, Weekly, Monthly)
            html_report_path: Path to HTML report file
            
        Returns:
            bool: True if sent successfully
        """
        subject = f"{period} Trading Performance Report"
        
        # Create plain text report
        body = self._create_text_report(report_data, period)
        
        # Create HTML report
        html_body = self._create_html_report(report_data, period)
        
        # Prepare attachments
        attachments = [html_report_path] if html_report_path and os.path.exists(html_report_path) else None
        
        return self.send_email(subject, body, html_body=html_body, attachments=attachments)
    
    def _create_text_report(self, data: Dict, period: str) -> str:
        """Create plain text performance report."""
        template_str = """
{{ period }} Trading Performance Report
Generated: {{ timestamp }}

Portfolio Performance:
- Total Return: {{ total_return }}%
- Sharpe Ratio: {{ sharpe_ratio }}
- Max Drawdown: {{ max_drawdown }}%
- Win Rate: {{ win_rate }}%
- Total Trades: {{ total_trades }}

Risk Metrics:
- Value at Risk (95%): {{ var_95 }}%
- Volatility: {{ volatility }}%
- Beta: {{ beta }}

Top Performing Strategies:
{% for strategy in top_strategies %}
- {{ strategy.name }}: {{ strategy.return }}%
{% endfor %}

Recent Trades:
{% for trade in recent_trades %}
- {{ trade.timestamp }}: {{ trade.action }} {{ trade.symbol }} @ ${{ trade.price }}
{% endfor %}

This is an automated report from your trading bot.
        """
        
        template = Template(template_str)
        return template.render(
            period=period,
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            **data
        )
    
    def _create_html_report(self, data: Dict, period: str) -> str:
        """Create HTML performance report."""
        html_template = """
<html>
<head>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
        .metric { background-color: #f8f9fa; padding: 10px; margin: 5px 0; border-radius: 5px; }
        .positive { color: #27ae60; }
        .negative { color: #e74c3c; }
        .table { border-collapse: collapse; width: 100%; margin: 10px 0; }
        .table th, .table td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        .table th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <div class="header">
        <h2>{{ period }} Trading Performance Report</h2>
        <p>Generated: {{ timestamp }}</p>
    </div>
    
    <h3>Portfolio Performance</h3>
    <div class="metric">Total Return: <span class="{{ 'positive' if total_return > 0 else 'negative' }}">{{ total_return }}%</span></div>
    <div class="metric">Sharpe Ratio: {{ sharpe_ratio }}</div>
    <div class="metric">Max Drawdown: <span class="negative">{{ max_drawdown }}%</span></div>
    <div class="metric">Win Rate: {{ win_rate }}%</div>
    
    <h3>Risk Metrics</h3>
    <div class="metric">Value at Risk (95%): {{ var_95 }}%</div>
    <div class="metric">Volatility: {{ volatility }}%</div>
    <div class="metric">Beta: {{ beta }}</div>
    
    <h3>Strategy Performance</h3>
    <table class="table">
        <tr><th>Strategy</th><th>Return (%)</th><th>Trades</th><th>Win Rate (%)</th></tr>
        {% for strategy in strategies %}
        <tr>
            <td>{{ strategy.name }}</td>
            <td class="{{ 'positive' if strategy.return > 0 else 'negative' }}">{{ strategy.return }}</td>
            <td>{{ strategy.trades }}</td>
            <td>{{ strategy.win_rate }}</td>
        </tr>
        {% endfor %}
    </table>
    
    <p><em>This is an automated report from your trading bot.</em></p>
</body>
</html>
        """
        
        template = Template(html_template)
        return template.render(
            period=period,
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            **data
        )
    
    def send_system_alert(self, 
                         alert_type: str, 
                         message: str, 
                         error_details: Optional[str] = None) -> bool:
        """
        Send system alert for errors or important events.
        
        Args:
            alert_type: Type of system alert
            message: Alert message
            error_details: Optional error details
            
        Returns:
            bool: True if sent successfully
        """
        subject = f"System Alert: {alert_type}"
        
        body = f"""
Trading Bot System Alert

Alert Type: {alert_type}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Message:
{message}
"""
        
        if error_details:
            body += f"\nError Details:\n{error_details}"
        
        body += "\nPlease check your trading bot status and logs."
        
        return self.send_email(subject, body)


class NotificationManager:
    """Centralized notification management."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize notification manager."""
        self.email_notifier = EmailNotifier(config.get('email', {}))
        self.config = config
    
    def notify_trade(self, **kwargs) -> bool:
        """Send trade notification."""
        return self.email_notifier.send_trade_alert(**kwargs)
    
    def notify_risk(self, **kwargs) -> bool:
        """Send risk notification."""
        return self.email_notifier.send_risk_alert(**kwargs)
    
    def notify_performance(self, **kwargs) -> bool:
        """Send performance notification."""
        return self.email_notifier.send_performance_report(**kwargs)
    
    def notify_system(self, **kwargs) -> bool:
        """Send system notification."""
        return self.email_notifier.send_system_alert(**kwargs)
