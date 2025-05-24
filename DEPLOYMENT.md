# Deployment and Monitoring Guide

This guide covers deployment options, monitoring strategies, and operational best practices for the Algorithmic Trading Bot.

## 🚀 Deployment Options

### 1. Local Development Deployment

For development and testing:

```bash
# Setup environment
python setup.py

# Validate installation
python validate.py

# Run paper trading
python main.py --paper-trading
```

### 2. VPS/Cloud Server Deployment

For production use, deploy on a reliable server:

#### Recommended Providers:
- **DigitalOcean**: $5-20/month droplets
- **AWS EC2**: t3.micro to t3.small instances
- **Google Cloud**: e2-micro to e2-small instances
- **Linode**: $5-10/month instances

#### Server Requirements:
```bash
# Minimum specifications
CPU: 1-2 cores
RAM: 2-4 GB
Storage: 20-50 GB SSD
OS: Ubuntu 20.04+ or CentOS 8+
Network: Stable internet connection
```

#### Deployment Steps:

1. **Server Setup**:
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.8+
sudo apt install python3.8 python3.8-pip python3.8-venv

# Install system dependencies
sudo apt install build-essential git screen htop
```

2. **Application Deployment**:
```bash
# Clone repository
git clone <repository-url>
cd algotrade

# Create virtual environment
python3.8 -m venv venv
source venv/bin/activate

# Run setup
python setup.py

# Configure environment
cp .env.template .env
nano .env  # Edit with your settings
```

3. **Process Management**:
```bash
# Using screen (simple)
screen -S trading_bot
python main.py
# Ctrl+A, D to detach

# Using systemd (recommended)
sudo cp deployment/trading-bot.service /etc/systemd/system/
sudo systemctl enable trading-bot
sudo systemctl start trading-bot
```

### 3. Docker Deployment

Create a containerized deployment:

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

# Install TA-Lib
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    && wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz \
    && tar -xzf ta-lib-0.4.0-src.tar.gz \
    && cd ta-lib \
    && ./configure --prefix=/usr \
    && make && make install \
    && cd .. && rm -rf ta-lib*

COPY . .
CMD ["python", "main.py"]
```

```bash
# Build and run
docker build -t trading-bot .
docker run -d --name trading-bot \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  --env-file .env \
  trading-bot
```

## 📊 Monitoring and Alerting

### 1. System Monitoring

Monitor system resources and application health:

#### Resource Monitoring Script:
```bash
#!/bin/bash
# monitoring/system_monitor.sh

# Check CPU usage
cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | awk -F'%' '{print $1}')

# Check memory usage  
mem_usage=$(free | grep Mem | awk '{printf("%.2f", $3/$2 * 100.0)}')

# Check disk usage
disk_usage=$(df -h / | awk 'NR==2{printf "%s", $5}' | sed 's/%//')

# Check if trading bot is running
if pgrep -f "main.py" > /dev/null; then
    bot_status="RUNNING"
else
    bot_status="STOPPED"
fi

echo "$(date): CPU: ${cpu_usage}%, Memory: ${mem_usage}%, Disk: ${disk_usage}%, Bot: ${bot_status}"

# Alert if thresholds exceeded
if (( $(echo "$cpu_usage > 80" | bc -l) )); then
    echo "HIGH CPU USAGE ALERT: ${cpu_usage}%"
fi

if (( $(echo "$mem_usage > 80" | bc -l) )); then
    echo "HIGH MEMORY USAGE ALERT: ${mem_usage}%"
fi
```

#### Cron Job Setup:
```bash
# Add to crontab (crontab -e)
*/5 * * * * /path/to/monitoring/system_monitor.sh >> /var/log/trading_bot_monitor.log 2>&1
```

### 2. Application Monitoring

Monitor trading bot performance and errors:

#### Health Check Endpoint:
```python
# monitoring/health_check.py
import requests
import json
from datetime import datetime

def check_bot_health():
    """Check if trading bot is healthy."""
    try:
        # Check log files for recent activity
        log_file = 'logs/trading.log'
        with open(log_file, 'r') as f:
            lines = f.readlines()
            recent_lines = lines[-10:]  # Last 10 lines
            
        # Check for errors
        error_count = sum(1 for line in recent_lines if 'ERROR' in line)
        
        # Check last activity timestamp
        last_line = recent_lines[-1] if recent_lines else ""
        
        return {
            'status': 'healthy' if error_count == 0 else 'degraded',
            'error_count': error_count,
            'last_activity': last_line.split()[0:2] if last_line else None,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

if __name__ == '__main__':
    health = check_bot_health()
    print(json.dumps(health, indent=2))
```

### 3. Performance Monitoring

Track trading performance and metrics:

```python
# monitoring/performance_monitor.py
import pandas as pd
import sqlite3
from datetime import datetime, timedelta

class PerformanceMonitor:
    def __init__(self, db_path='data/trading_data.db'):
        self.db_path = db_path
    
    def get_daily_pnl(self):
        """Get daily P&L summary."""
        conn = sqlite3.connect(self.db_path)
        
        query = """
        SELECT DATE(timestamp) as date,
               SUM(CASE WHEN side = 'sell' THEN quantity * price ELSE -quantity * price END) as daily_pnl,
               COUNT(*) as trade_count
        FROM trades 
        WHERE timestamp >= datetime('now', '-30 days')
        GROUP BY DATE(timestamp)
        ORDER BY date DESC
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        return df
    
    def get_portfolio_metrics(self):
        """Get current portfolio metrics."""
        conn = sqlite3.connect(self.db_path)
        
        # Get current positions
        positions_query = "SELECT * FROM positions WHERE quantity > 0"
        positions = pd.read_sql_query(positions_query, conn)
        
        # Get portfolio value history
        portfolio_query = """
        SELECT timestamp, total_value 
        FROM portfolio_snapshots 
        WHERE timestamp >= datetime('now', '-7 days')
        ORDER BY timestamp DESC
        """
        portfolio_history = pd.read_sql_query(portfolio_query, conn)
        
        conn.close()
        
        return {
            'positions': positions,
            'portfolio_history': portfolio_history,
            'position_count': len(positions),
            'total_exposure': positions['quantity'].sum() if not positions.empty else 0
        }
```

### 4. Alert System

Set up automated alerts for important events:

```python
# monitoring/alert_system.py
import smtplib
from email.mime.text import MIMEText
from datetime import datetime

class AlertSystem:
    def __init__(self, config):
        self.config = config
        
    def send_alert(self, subject, message, level='INFO'):
        """Send alert via email."""
        try:
            msg = MIMEText(f"""
Alert Level: {level}
Time: {datetime.now()}
Subject: {subject}

Message:
{message}

Trading Bot Alert System
            """)
            
            msg['Subject'] = f"[Trading Bot {level}] {subject}"
            msg['From'] = self.config['email']['sender']
            msg['To'] = self.config['email']['recipients']
            
            # Send email
            server = smtplib.SMTP(self.config['email']['smtp_server'])
            server.starttls()
            server.login(self.config['email']['sender'], self.config['email']['password'])
            server.send_message(msg)
            server.quit()
            
        except Exception as e:
            print(f"Failed to send alert: {e}")
    
    def check_drawdown_alert(self, current_value, peak_value):
        """Check if drawdown exceeds threshold."""
        drawdown = (peak_value - current_value) / peak_value
        threshold = self.config.get('max_drawdown_alert', 0.10)
        
        if drawdown > threshold:
            self.send_alert(
                "High Drawdown Alert",
                f"Portfolio drawdown: {drawdown:.2%}\nThreshold: {threshold:.2%}",
                "WARNING"
            )
    
    def check_system_errors(self, log_file):
        """Check for system errors in logs."""
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
                
            # Check last 100 lines for errors
            recent_lines = lines[-100:]
            error_lines = [line for line in recent_lines if 'ERROR' in line]
            
            if len(error_lines) > 5:  # More than 5 errors recently
                self.send_alert(
                    "Multiple System Errors",
                    f"Found {len(error_lines)} errors in recent logs:\n\n" + 
                    "\n".join(error_lines[-5:]),
                    "ERROR"
                )
                
        except Exception as e:
            self.send_alert(
                "Log Monitoring Failed",
                f"Could not read log file: {e}",
                "ERROR"
            )
```

## 🔧 Operational Best Practices

### 1. Backup Strategy

```bash
#!/bin/bash
# backup/backup_data.sh

DATE=$(date +"%Y%m%d_%H%M%S")
BACKUP_DIR="/backup/trading_bot"

# Create backup directory
mkdir -p $BACKUP_DIR

# Backup database
cp data/trading_data.db $BACKUP_DIR/trading_data_$DATE.db

# Backup configuration
cp .env $BACKUP_DIR/env_$DATE.bak

# Backup logs (last 7 days)
tar -czf $BACKUP_DIR/logs_$DATE.tar.gz logs/

# Backup models
tar -czf $BACKUP_DIR/models_$DATE.tar.gz models/

# Clean old backups (keep 30 days)
find $BACKUP_DIR -name "*.db" -mtime +30 -delete
find $BACKUP_DIR -name "*.tar.gz" -mtime +30 -delete

echo "Backup completed: $DATE"
```

### 2. Log Rotation

```bash
# /etc/logrotate.d/trading-bot
/path/to/algotrade/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 ubuntu ubuntu
    postrotate
        systemctl reload trading-bot
    endscript
}
```

### 3. Security Checklist

- [ ] API keys stored securely (environment variables)
- [ ] Database access restricted
- [ ] Server firewall configured
- [ ] SSH key-based authentication
- [ ] Regular security updates
- [ ] Log monitoring for suspicious activity
- [ ] Backup encryption
- [ ] Access logs reviewed regularly

### 4. Disaster Recovery Plan

1. **Backup Verification**:
   - Test restore procedures monthly
   - Verify backup integrity
   - Document recovery steps

2. **Failover Strategy**:
   - Secondary server ready
   - Automated failover scripts
   - Contact information updated

3. **Communication Plan**:
   - Incident response team
   - Escalation procedures
   - Stakeholder notifications

## 📈 Performance Optimization

### 1. Database Optimization

```sql
-- Add indexes for common queries
CREATE INDEX idx_trades_timestamp ON trades(timestamp);
CREATE INDEX idx_trades_symbol ON trades(symbol);
CREATE INDEX idx_positions_symbol ON positions(symbol);

-- Archive old data
CREATE TABLE trades_archive AS SELECT * FROM trades WHERE timestamp < datetime('now', '-1 year');
DELETE FROM trades WHERE timestamp < datetime('now', '-1 year');
```

### 2. Memory Management

```python
# config/performance_config.py
PERFORMANCE_CONFIG = {
    'max_memory_usage': 0.8,  # 80% of available memory
    'data_cache_size': 1000,  # Maximum cached data points
    'model_cache_size': 5,    # Maximum cached models
    'cleanup_interval': 3600, # Cleanup every hour
}
```

### 3. API Rate Limiting

```python
# utils/rate_limiter.py
import time
from collections import deque

class RateLimiter:
    def __init__(self, max_calls, time_window):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = deque()
    
    def can_make_call(self):
        now = time.time()
        
        # Remove old calls outside time window
        while self.calls and self.calls[0] < now - self.time_window:
            self.calls.popleft()
        
        # Check if we can make a new call
        if len(self.calls) < self.max_calls:
            self.calls.append(now)
            return True
        
        return False
```

## 🚨 Troubleshooting Guide

### Common Issues and Solutions

1. **Bot Stops Responding**:
   - Check system resources
   - Review error logs
   - Restart bot service
   - Verify API connectivity

2. **Data Feed Issues**:
   - Check API limits
   - Verify network connectivity
   - Switch to backup data provider
   - Review data quality

3. **Performance Degradation**:
   - Monitor memory usage
   - Check database size
   - Review algorithm complexity
   - Optimize data processing

4. **Risk Management Alerts**:
   - Review position sizes
   - Check correlation exposure
   - Verify risk calculations
   - Adjust risk parameters

Remember: Always prioritize risk management and never risk more than you can afford to lose. Monitor your trading bot continuously and be prepared to intervene manually when necessary.
