"""
System Monitor and Auto-Recovery
Monitors trading bot health and provides auto-restart capabilities
"""
import psutil
import time
import os
import sys
import subprocess
import logging
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import json
import socket
import threading
from dataclasses import dataclass, asdict

@dataclass
class SystemHealth:
    """System health status"""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    bot_running: bool
    api_connectivity: bool
    database_accessible: bool
    last_trade_time: Optional[datetime]
    portfolio_value: float
    error_count: int
    warning_count: int

class SystemMonitor:
    """Comprehensive system monitoring and health checking"""
    
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        self.health_history = []
        self.alert_thresholds = {
            'cpu_usage': 80.0,        # %
            'memory_usage': 85.0,     # %
            'disk_usage': 90.0,       # %
            'max_downtime': 300,      # seconds
            'min_portfolio_value': 50000,  # $
            'max_error_rate': 10      # errors per hour
        }
        
        # Initialize monitoring database
        self.db_path = os.path.join(os.path.dirname(__file__), 'monitoring.db')
        self._init_monitoring_db()
        
        # Monitoring state
        self.monitoring_active = False
        self.bot_process = None
        self.restart_count = 0
        self.last_restart_time = None
    
    def _load_config(self, config_path: str) -> Dict:
        """Load monitoring configuration"""
        default_config = {
            'monitor_interval': 60,     # seconds
            'max_restarts': 3,         # per hour
            'restart_delay': 30,       # seconds
            'alert_email': None,
            'trading_bot_script': 'main.py',
            'working_directory': os.path.dirname(os.path.dirname(__file__)),
            'python_executable': sys.executable
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
            except Exception as e:
                print(f"Error loading config: {e}")
        
        return default_config
    
    def _setup_logging(self) -> logging.Logger:
        """Setup monitoring logging"""
        logger = logging.getLogger('SystemMonitor')
        logger.setLevel(logging.INFO)
        
        # Create logs directory if it doesn't exist
        logs_dir = os.path.join(os.path.dirname(__file__), 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        
        # File handler
        file_handler = logging.FileHandler(
            os.path.join(logs_dir, f'monitor_{datetime.now().strftime("%Y%m%d")}.log')
        )
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def _init_monitoring_db(self):
        """Initialize monitoring database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_health (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                cpu_usage REAL,
                memory_usage REAL,
                disk_usage REAL,
                bot_running INTEGER,
                api_connectivity INTEGER,
                database_accessible INTEGER,
                last_trade_time TEXT,
                portfolio_value REAL,
                error_count INTEGER,
                warning_count INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                alert_type TEXT NOT NULL,
                message TEXT NOT NULL,
                severity TEXT NOT NULL,
                resolved INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS restart_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                reason TEXT NOT NULL,
                success INTEGER,
                error_message TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def check_system_health(self) -> SystemHealth:
        """Perform comprehensive system health check"""
        timestamp = datetime.now()
        
        # System resource usage
        cpu_usage = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        memory_usage = memory.percent
        disk_usage = disk.percent
        
        # Check if trading bot is running
        bot_running = self._is_bot_running()
        
        # Check API connectivity
        api_connectivity = self._check_api_connectivity()
        
        # Check database accessibility
        database_accessible = self._check_database_access()
        
        # Get last trade time and portfolio value
        last_trade_time, portfolio_value = self._get_trading_status()
        
        # Count recent errors and warnings
        error_count, warning_count = self._count_recent_issues()
        
        health = SystemHealth(
            timestamp=timestamp,
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            disk_usage=disk_usage,
            bot_running=bot_running,
            api_connectivity=api_connectivity,
            database_accessible=database_accessible,
            last_trade_time=last_trade_time,
            portfolio_value=portfolio_value,
            error_count=error_count,
            warning_count=warning_count
        )
        
        # Store health data
        self._store_health_data(health)
        
        return health
    
    def _is_bot_running(self) -> bool:
        """Check if trading bot process is running"""
        try:
            if self.bot_process:
                return self.bot_process.poll() is None
            
            # Check for existing bot process
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = proc.info['cmdline']
                    if cmdline and any('main.py' in cmd for cmd in cmdline):
                        if any('algotrade' in cmd for cmd in cmdline):
                            return True
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            return False
        except Exception as e:
            self.logger.error(f"Error checking bot status: {e}")
            return False
    
    def _check_api_connectivity(self) -> bool:
        """Check API connectivity"""
        try:
            # Try to connect to a known endpoint (e.g., check internet connectivity)
            socket.create_connection(("8.8.8.8", 53), timeout=5)
            return True
        except Exception:
            return False
    
    def _check_database_access(self) -> bool:
        """Check database accessibility"""
        try:
            db_path = os.path.join(
                self.config['working_directory'], 'data', 'trading_data.db'
            )
            conn = sqlite3.connect(db_path, timeout=5)
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            conn.close()
            return True
        except Exception:
            return False
    
    def _get_trading_status(self) -> tuple:
        """Get last trade time and current portfolio value"""
        try:
            db_path = os.path.join(
                self.config['working_directory'], 'data', 'trading_data.db'
            )
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Get last trade time
            cursor.execute('''
                SELECT MAX(entry_date) FROM trade_history
            ''')
            result = cursor.fetchone()
            last_trade_time = result[0] if result and result[0] else None
            if last_trade_time:
                last_trade_time = datetime.fromisoformat(last_trade_time)
            
            # Get latest portfolio value
            cursor.execute('''
                SELECT portfolio_value FROM portfolio_history 
                ORDER BY date DESC LIMIT 1
            ''')
            result = cursor.fetchone()
            portfolio_value = result[0] if result else 0.0
            
            conn.close()
            return last_trade_time, portfolio_value
            
        except Exception as e:
            self.logger.error(f"Error getting trading status: {e}")
            return None, 0.0
    
    def _count_recent_issues(self) -> tuple:
        """Count recent errors and warnings from logs"""
        try:
            log_path = os.path.join(
                self.config['working_directory'], 'logs', 
                f'trading_bot_{datetime.now().strftime("%Y%m%d")}.log'
            )
            
            if not os.path.exists(log_path):
                return 0, 0
            
            error_count = 0
            warning_count = 0
            cutoff_time = datetime.now() - timedelta(hours=1)
            
            with open(log_path, 'r') as f:
                for line in f:
                    try:
                        # Parse timestamp from log line
                        if len(line) > 19:
                            timestamp_str = line[:19]
                            log_time = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                            
                            if log_time > cutoff_time:
                                if 'ERROR' in line:
                                    error_count += 1
                                elif 'WARNING' in line:
                                    warning_count += 1
                    except:
                        continue
            
            return error_count, warning_count
            
        except Exception as e:
            self.logger.error(f"Error counting log issues: {e}")
            return 0, 0
    
    def _store_health_data(self, health: SystemHealth):
        """Store health data in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO system_health 
                (timestamp, cpu_usage, memory_usage, disk_usage, bot_running,
                 api_connectivity, database_accessible, last_trade_time,
                 portfolio_value, error_count, warning_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                health.timestamp.isoformat(),
                health.cpu_usage,
                health.memory_usage,
                health.disk_usage,
                int(health.bot_running),
                int(health.api_connectivity),
                int(health.database_accessible),
                health.last_trade_time.isoformat() if health.last_trade_time else None,
                health.portfolio_value,
                health.error_count,
                health.warning_count
            ))
            
            conn.commit()
            conn.close()
            
            # Keep only last 30 days of data
            self._cleanup_old_data()
            
        except Exception as e:
            self.logger.error(f"Error storing health data: {e}")
    
    def _cleanup_old_data(self):
        """Clean up old monitoring data"""
        try:
            cutoff_date = (datetime.now() - timedelta(days=30)).isoformat()
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('DELETE FROM system_health WHERE timestamp < ?', (cutoff_date,))
            cursor.execute('DELETE FROM alerts WHERE timestamp < ?', (cutoff_date,))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error cleaning up old data: {e}")
    
    def analyze_health(self, health: SystemHealth) -> List[str]:
        """Analyze health status and return list of issues"""
        issues = []
        
        # Check resource usage
        if health.cpu_usage > self.alert_thresholds['cpu_usage']:
            issues.append(f"High CPU usage: {health.cpu_usage:.1f}%")
        
        if health.memory_usage > self.alert_thresholds['memory_usage']:
            issues.append(f"High memory usage: {health.memory_usage:.1f}%")
        
        if health.disk_usage > self.alert_thresholds['disk_usage']:
            issues.append(f"High disk usage: {health.disk_usage:.1f}%")
        
        # Check bot status
        if not health.bot_running:
            issues.append("Trading bot is not running")
        
        if not health.api_connectivity:
            issues.append("API connectivity issues")
        
        if not health.database_accessible:
            issues.append("Database access issues")
        
        # Check trading activity
        if health.last_trade_time:
            time_since_last_trade = datetime.now() - health.last_trade_time
            if time_since_last_trade.total_seconds() > self.alert_thresholds['max_downtime']:
                issues.append(f"No trading activity for {time_since_last_trade}")
        
        # Check portfolio value
        if health.portfolio_value < self.alert_thresholds['min_portfolio_value']:
            issues.append(f"Low portfolio value: ${health.portfolio_value:,.2f}")
        
        # Check error rate
        if health.error_count > self.alert_thresholds['max_error_rate']:
            issues.append(f"High error rate: {health.error_count} errors in last hour")
        
        return issues
    
    def restart_bot(self, reason: str = "System monitor restart") -> bool:
        """Restart the trading bot"""
        try:
            self.logger.info(f"Attempting to restart bot: {reason}")
            
            # Check restart limits
            if self._check_restart_limits():
                self.logger.warning("Restart limit exceeded, skipping restart")
                return False
            
            # Stop existing bot process
            if self.bot_process:
                self.bot_process.terminate()
                time.sleep(5)
                if self.bot_process.poll() is None:
                    self.bot_process.kill()
            
            # Start new bot process
            script_path = os.path.join(
                self.config['working_directory'], 
                self.config['trading_bot_script']
            )
            
            self.bot_process = subprocess.Popen(
                [self.config['python_executable'], script_path],
                cwd=self.config['working_directory'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait a bit and check if it started successfully
            time.sleep(self.config['restart_delay'])
            
            if self.bot_process.poll() is None:
                self.restart_count += 1
                self.last_restart_time = datetime.now()
                self._log_restart(reason, True, None)
                self.logger.info("Bot restarted successfully")
                return True
            else:
                error_msg = "Bot process failed to start"
                self._log_restart(reason, False, error_msg)
                self.logger.error(error_msg)
                return False
                
        except Exception as e:
            error_msg = f"Error restarting bot: {e}"
            self._log_restart(reason, False, error_msg)
            self.logger.error(error_msg)
            return False
    
    def _check_restart_limits(self) -> bool:
        """Check if restart limits have been exceeded"""
        if not self.last_restart_time:
            return False
        
        # Reset count if more than an hour has passed
        if datetime.now() - self.last_restart_time > timedelta(hours=1):
            self.restart_count = 0
            return False
        
        return self.restart_count >= self.config['max_restarts']
    
    def _log_restart(self, reason: str, success: bool, error_message: str = None):
        """Log restart attempt"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO restart_log (timestamp, reason, success, error_message)
                VALUES (?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                reason,
                int(success),
                error_message
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error logging restart: {e}")
    
    def send_alert(self, message: str, severity: str = "WARNING"):
        """Send alert notification"""
        try:
            self.logger.log(
                logging.ERROR if severity == "CRITICAL" else logging.WARNING,
                f"ALERT [{severity}]: {message}"
            )
            
            # Store alert in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO alerts (timestamp, alert_type, message, severity)
                VALUES (?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                "system_health",
                message,
                severity
            ))
            
            conn.commit()
            conn.close()
            
            # Send email alert if configured
            if self.config.get('alert_email'):
                self._send_email_alert(message, severity)
                
        except Exception as e:
            self.logger.error(f"Error sending alert: {e}")
    
    def _send_email_alert(self, message: str, severity: str):
        """Send email alert notification"""
        # This would be implemented with actual email configuration
        # For now, just log the email intent
        self.logger.info(f"Email alert would be sent: [{severity}] {message}")
    
    def start_monitoring(self):
        """Start system monitoring"""
        self.monitoring_active = True
        self.logger.info("System monitoring started")
        
        while self.monitoring_active:
            try:
                # Check system health
                health = self.check_system_health()
                
                # Analyze for issues
                issues = self.analyze_health(health)
                
                if issues:
                    for issue in issues:
                        self.send_alert(issue, "WARNING")
                    
                    # Auto-restart if bot is not running
                    if not health.bot_running:
                        self.restart_bot("Bot not running")
                
                # Sleep until next check
                time.sleep(self.config['monitor_interval'])
                
            except KeyboardInterrupt:
                self.logger.info("Monitoring stopped by user")
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(30)  # Wait before retrying
        
        self.monitoring_active = False
    
    def stop_monitoring(self):
        """Stop system monitoring"""
        self.monitoring_active = False
        self.logger.info("System monitoring stopped")
    
    def get_health_summary(self, hours: int = 24) -> Dict:
        """Get health summary for specified time period"""
        try:
            cutoff_time = (datetime.now() - timedelta(hours=hours)).isoformat()
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get recent health data
            cursor.execute('''
                SELECT * FROM system_health 
                WHERE timestamp > ? 
                ORDER BY timestamp DESC
            ''', (cutoff_time,))
            
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            
            # Get recent alerts
            cursor.execute('''
                SELECT COUNT(*) as alert_count, severity 
                FROM alerts 
                WHERE timestamp > ? 
                GROUP BY severity
            ''', (cutoff_time,))
            
            alerts = dict(cursor.fetchall())
            
            # Get restart count
            cursor.execute('''
                SELECT COUNT(*) FROM restart_log 
                WHERE timestamp > ?
            ''', (cutoff_time,))
            
            restart_count = cursor.fetchone()[0]
            
            conn.close()
            
            if not rows:
                return {"error": "No health data available"}
            
            # Calculate averages
            health_data = [dict(zip(columns, row)) for row in rows]
            
            avg_cpu = sum(h['cpu_usage'] for h in health_data) / len(health_data)
            avg_memory = sum(h['memory_usage'] for h in health_data) / len(health_data)
            uptime_percentage = sum(h['bot_running'] for h in health_data) / len(health_data) * 100
            
            return {
                "period_hours": hours,
                "data_points": len(health_data),
                "avg_cpu_usage": avg_cpu,
                "avg_memory_usage": avg_memory,
                "uptime_percentage": uptime_percentage,
                "alerts": alerts,
                "restart_count": restart_count,
                "latest_health": health_data[0] if health_data else None
            }
            
        except Exception as e:
            self.logger.error(f"Error getting health summary: {e}")
            return {"error": str(e)}

def main():
    """Main monitoring function"""
    monitor = SystemMonitor()
    
    try:
        monitor.start_monitoring()
    except KeyboardInterrupt:
        monitor.stop_monitoring()

if __name__ == "__main__":
    main()
