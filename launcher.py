#!/usr/bin/env python3
"""
Trading Bot Launcher
Advanced startup script with monitoring, health checks, and auto-recovery
"""
import os
import sys
import argparse
import subprocess
import time
from datetime import datetime
import signal
import atexit

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

try:
    from monitoring.system_monitor import SystemMonitor
    from analytics.portfolio_analytics import analytics
    from config.config import CONFIG
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all dependencies are installed: pip install -r requirements.txt")
    sys.exit(1)

class TradingBotLauncher:
    """Advanced trading bot launcher with monitoring capabilities"""
    
    def __init__(self):
        self.bot_process = None
        self.monitor = None
        self.dashboard_process = None
        self.running = False
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        atexit.register(self._cleanup)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print(f"\nReceived signal {signum}, shutting down...")
        self.shutdown()
    
    def _cleanup(self):
        """Cleanup function called on exit"""
        self.shutdown()
    
    def pre_launch_checks(self) -> bool:
        """Perform pre-launch system checks"""
        print("🔍 Performing pre-launch checks...")
        
        # Check Python version
        if sys.version_info < (3, 8):
            print("❌ Python 3.8+ required")
            return False
        print("✅ Python version check passed")
        
        # Check dependencies
        try:
            import pandas, numpy, yfinance, alpaca_trade_api
            print("✅ Core dependencies available")
        except ImportError as e:
            print(f"❌ Missing dependency: {e}")
            return False
        
        # Check configuration
        if not hasattr(CONFIG, 'SYMBOLS') or not CONFIG.SYMBOLS:
            print("❌ No trading symbols configured")
            return False
        print("✅ Configuration check passed")
        
        # Check data directory
        data_dir = os.path.join(project_root, 'data')
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            print("✅ Created data directory")
        else:
            print("✅ Data directory exists")
        
        # Check logs directory
        logs_dir = os.path.join(project_root, 'logs')
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)
            print("✅ Created logs directory")
        else:
            print("✅ Logs directory exists")
        
        # Test database connection
        try:
            analytics.get_portfolio_history()
            print("✅ Database connection test passed")
        except Exception as e:
            print(f"⚠️  Database test warning: {e}")
        
        print("✅ All pre-launch checks completed\n")
        return True
    
    def launch_dashboard(self) -> bool:
        """Launch the web dashboard"""
        try:
            dashboard_script = os.path.join(project_root, 'dashboard', 'app.py')
            
            if not os.path.exists(dashboard_script):
                print("⚠️  Dashboard not available")
                return False
            
            print("🚀 Starting web dashboard...")
            self.dashboard_process = subprocess.Popen(
                [sys.executable, dashboard_script],
                cwd=os.path.join(project_root, 'dashboard'),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            time.sleep(2)  # Give it time to start
            
            if self.dashboard_process.poll() is None:
                print("✅ Dashboard started at http://localhost:5000")
                return True
            else:
                print("❌ Dashboard failed to start")
                return False
                
        except Exception as e:
            print(f"❌ Error starting dashboard: {e}")
            return False
    
    def launch_bot(self, mode: str = 'paper') -> bool:
        """Launch the trading bot"""
        try:
            print(f"🚀 Starting trading bot in {mode} mode...")
            
            bot_script = os.path.join(project_root, 'main.py')
            cmd = [sys.executable, bot_script, '--mode', mode]
            
            self.bot_process = subprocess.Popen(
                cmd,
                cwd=project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            time.sleep(3)  # Give it time to start
            
            if self.bot_process.poll() is None:
                print(f"✅ Trading bot started successfully (PID: {self.bot_process.pid})")
                return True
            else:
                stdout, stderr = self.bot_process.communicate()
                print(f"❌ Trading bot failed to start")
                print(f"Error: {stderr.decode()}")
                return False
                
        except Exception as e:
            print(f"❌ Error starting trading bot: {e}")
            return False
    
    def launch_monitoring(self) -> bool:
        """Launch system monitoring"""
        try:
            print("🔍 Starting system monitoring...")
            
            self.monitor = SystemMonitor()
            
            # Start monitoring in a separate thread
            import threading
            monitor_thread = threading.Thread(
                target=self.monitor.start_monitoring,
                daemon=True
            )
            monitor_thread.start()
            
            print("✅ System monitoring started")
            return True
            
        except Exception as e:
            print(f"❌ Error starting monitoring: {e}")
            return False
    
    def run_interactive_mode(self):
        """Run in interactive mode with user commands"""
        print("\n" + "="*60)
        print("🤖 ALGORITHMIC TRADING BOT - INTERACTIVE MODE")
        print("="*60)
        print("\nAvailable commands:")
        print("  status    - Show system status")
        print("  portfolio - Show portfolio summary")
        print("  logs      - Show recent logs")
        print("  restart   - Restart trading bot")
        print("  stop      - Stop all processes")
        print("  help      - Show this help")
        print("  quit      - Exit launcher")
        print("\n" + "-"*60)
        
        while self.running:
            try:
                command = input("\n🤖 > ").strip().lower()
                
                if command == 'quit' or command == 'exit':
                    break
                elif command == 'status':
                    self._show_status()
                elif command == 'portfolio':
                    self._show_portfolio()
                elif command == 'logs':
                    self._show_logs()
                elif command == 'restart':
                    self._restart_bot()
                elif command == 'stop':
                    self._stop_processes()
                elif command == 'help':
                    self._show_help()
                elif command == '':
                    continue
                else:
                    print(f"Unknown command: {command}. Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                break
            except EOFError:
                break
        
        print("\nExiting interactive mode...")
    
    def _show_status(self):
        """Show system status"""
        print("\n📊 SYSTEM STATUS")
        print("-" * 40)
        
        # Bot status
        if self.bot_process:
            if self.bot_process.poll() is None:
                print("🟢 Trading Bot: RUNNING")
            else:
                print("🔴 Trading Bot: STOPPED")
        else:
            print("🔴 Trading Bot: NOT STARTED")
        
        # Dashboard status
        if self.dashboard_process:
            if self.dashboard_process.poll() is None:
                print("🟢 Dashboard: RUNNING (http://localhost:5000)")
            else:
                print("🔴 Dashboard: STOPPED")
        else:
            print("🔴 Dashboard: NOT STARTED")
        
        # Monitoring status
        if self.monitor and self.monitor.monitoring_active:
            print("🟢 Monitoring: ACTIVE")
        else:
            print("🔴 Monitoring: INACTIVE")
        
        # Health check
        if self.monitor:
            try:
                health = self.monitor.check_system_health()
                print(f"\n💻 System Resources:")
                print(f"   CPU Usage: {health.cpu_usage:.1f}%")
                print(f"   Memory Usage: {health.memory_usage:.1f}%")
                print(f"   Disk Usage: {health.disk_usage:.1f}%")
                
                if health.portfolio_value > 0:
                    print(f"\n💰 Portfolio Value: ${health.portfolio_value:,.2f}")
                
            except Exception as e:
                print(f"⚠️  Health check error: {e}")
    
    def _show_portfolio(self):
        """Show portfolio summary"""
        print("\n💰 PORTFOLIO SUMMARY")
        print("-" * 40)
        
        try:
            # This would connect to the actual execution engine
            # For now, show mock data
            print("Portfolio Value: $125,430.50")
            print("Daily P&L: +$1,250.30 (+1.01%)")
            print("Unrealized P&L: +$2,430.50")
            print("Active Positions: 5")
            print("\nTop Positions:")
            print("  AAPL: $25,000 (19.8%)")
            print("  MSFT: $22,500 (17.9%)")
            print("  GOOGL: $20,000 (15.9%)")
            
        except Exception as e:
            print(f"⚠️  Error fetching portfolio data: {e}")
    
    def _show_logs(self):
        """Show recent logs"""
        print("\n📋 RECENT LOGS")
        print("-" * 40)
        
        try:
            log_file = os.path.join(
                project_root, 'logs', 
                f'trading_bot_{datetime.now().strftime("%Y%m%d")}.log'
            )
            
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    # Show last 10 lines
                    for line in lines[-10:]:
                        print(line.strip())
            else:
                print("No log file found for today")
                
        except Exception as e:
            print(f"⚠️  Error reading logs: {e}")
    
    def _restart_bot(self):
        """Restart the trading bot"""
        print("\n🔄 Restarting trading bot...")
        
        if self.bot_process:
            self.bot_process.terminate()
            time.sleep(2)
            if self.bot_process.poll() is None:
                self.bot_process.kill()
        
        if self.launch_bot():
            print("✅ Trading bot restarted successfully")
        else:
            print("❌ Failed to restart trading bot")
    
    def _stop_processes(self):
        """Stop all processes"""
        print("\n🛑 Stopping all processes...")
        self.shutdown()
        print("✅ All processes stopped")
    
    def _show_help(self):
        """Show help information"""
        print("\n📚 HELP")
        print("-" * 40)
        print("Commands:")
        print("  status    - Show current status of all components")
        print("  portfolio - Display portfolio summary and positions")
        print("  logs      - Show recent trading bot logs")
        print("  restart   - Restart the trading bot process")
        print("  stop      - Stop all running processes")
        print("  quit      - Exit the launcher")
        print("\nFor more information, visit: http://localhost:5000 (if dashboard is running)")
    
    def launch_full_system(self, mode: str = 'paper', enable_dashboard: bool = True,
                          enable_monitoring: bool = True) -> bool:
        """Launch the complete trading system"""
        print("🚀 LAUNCHING ALGORITHMIC TRADING SYSTEM")
        print("=" * 50)
        
        # Pre-launch checks
        if not self.pre_launch_checks():
            return False
        
        success_count = 0
        total_components = 1  # Bot is mandatory
        
        # Launch trading bot (mandatory)
        if self.launch_bot(mode):
            success_count += 1
        
        # Launch dashboard (optional)
        if enable_dashboard:
            total_components += 1
            if self.launch_dashboard():
                success_count += 1
        
        # Launch monitoring (optional)
        if enable_monitoring:
            total_components += 1
            if self.launch_monitoring():
                success_count += 1
        
        self.running = True
        
        print(f"\n✅ System launched: {success_count}/{total_components} components started")
        
        if success_count == total_components:
            print("🎉 All components started successfully!")
        elif success_count > 0:
            print("⚠️  System partially started")
        else:
            print("❌ System launch failed")
            return False
        
        return True
    
    def shutdown(self):
        """Shutdown all processes"""
        self.running = False
        
        if self.monitor:
            self.monitor.stop_monitoring()
        
        if self.bot_process:
            self.bot_process.terminate()
            time.sleep(2)
            if self.bot_process.poll() is None:
                self.bot_process.kill()
        
        if self.dashboard_process:
            self.dashboard_process.terminate()
            time.sleep(1)
            if self.dashboard_process.poll() is None:
                self.dashboard_process.kill()

def main():
    """Main launcher function"""
    parser = argparse.ArgumentParser(description='Algorithmic Trading Bot Launcher')
    parser.add_argument('--mode', choices=['live', 'paper', 'backtest'], 
                       default='paper', help='Trading mode')
    parser.add_argument('--no-dashboard', action='store_true', 
                       help='Disable web dashboard')
    parser.add_argument('--no-monitoring', action='store_true', 
                       help='Disable system monitoring')
    parser.add_argument('--interactive', action='store_true', 
                       help='Run in interactive mode')
    parser.add_argument('--daemon', action='store_true', 
                       help='Run as daemon (background)')
    
    args = parser.parse_args()
    
    launcher = TradingBotLauncher()
    
    try:
        if args.mode == 'backtest':
            # Just run backtest and exit
            print("🔬 Running backtesting mode...")
            bot_script = os.path.join(project_root, 'main.py')
            subprocess.run([sys.executable, bot_script, '--mode', 'backtest'])
            return
        
        # Launch full system
        success = launcher.launch_full_system(
            mode=args.mode,
            enable_dashboard=not args.no_dashboard,
            enable_monitoring=not args.no_monitoring
        )
        
        if not success:
            print("❌ System launch failed")
            sys.exit(1)
        
        if args.interactive:
            launcher.run_interactive_mode()
        elif args.daemon:
            print("🏃 Running in daemon mode. Press Ctrl+C to stop.")
            try:
                while launcher.running:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
        else:
            print("\n🎯 System running. Available interfaces:")
            if not args.no_dashboard:
                print("   📊 Dashboard: http://localhost:5000")
            print("   📋 Logs: logs/trading_bot_YYYYMMDD.log")
            print("\nPress Ctrl+C to stop all processes.")
            
            try:
                while launcher.running:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
    
    finally:
        launcher.shutdown()
        print("\n👋 Goodbye!")

if __name__ == "__main__":
    main()
