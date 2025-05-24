#!/usr/bin/env python3
"""
Setup script for the Algorithmic Trading Bot.
This script helps set up the environment and configuration.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import urllib.request


def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version}")
    return True


def create_directories():
    """Create necessary directories."""
    directories = [
        'data',
        'logs',
        'reports',
        'models',
        'temp'
    ]
    
    print("\n📁 Creating directories...")
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"  ✅ {directory}/")


def install_dependencies():
    """Install Python dependencies."""
    print("\n📦 Installing dependencies...")
    
    try:
        # Upgrade pip first
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'])
        
        # Install requirements
        if Path('requirements.txt').exists():
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
            print("  ✅ Main dependencies installed")
        else:
            print("  ❌ requirements.txt not found")
            return False
            
        # Install development dependencies if requested
        response = input("\n🔧 Install development dependencies? (y/n): ").lower()
        if response == 'y' and Path('requirements-dev.txt').exists():
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements-dev.txt'])
            print("  ✅ Development dependencies installed")
            
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"  ❌ Failed to install dependencies: {e}")
        return False


def install_talib():
    """Install TA-Lib with platform-specific instructions."""
    print("\n🔧 Installing TA-Lib...")
    
    system = sys.platform.lower()
    
    if system == 'darwin':  # macOS
        print("  📋 macOS detected")
        print("  💡 Installing TA-Lib via Homebrew...")
        try:
            # Check if brew is installed
            subprocess.check_call(['which', 'brew'], stdout=subprocess.DEVNULL)
            subprocess.check_call(['brew', 'install', 'ta-lib'])
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'TA-Lib'])
            print("  ✅ TA-Lib installed successfully")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("  ❌ Homebrew not found or TA-Lib installation failed")
            print("  💡 Please install Homebrew first: /bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"")
            return False
            
    elif system.startswith('linux'):  # Linux
        print("  📋 Linux detected")
        print("  💡 Installing TA-Lib build dependencies...")
        print("  ⚠️  You may need to run with sudo or install manually")
        print("  💡 Manual installation:")
        print("      sudo apt-get update")
        print("      sudo apt-get install build-essential")
        print("      wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz")
        print("      tar -xzf ta-lib-0.4.0-src.tar.gz")
        print("      cd ta-lib/")
        print("      ./configure --prefix=/usr")
        print("      make")
        print("      sudo make install")
        print("      pip install TA-Lib")
        return False
        
    elif system.startswith('win'):  # Windows
        print("  📋 Windows detected")
        print("  💡 Please download TA-Lib wheel from:")
        print("      https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib")
        print("  💡 Then install with: pip install TA_Lib-0.4.XX-cpXX-cpXXm-win_amd64.whl")
        return False
    
    else:
        print(f"  ❌ Unsupported platform: {system}")
        return False


def setup_environment():
    """Set up environment configuration."""
    print("\n⚙️  Setting up environment configuration...")
    
    env_template = Path('.env.template')
    env_file = Path('.env')
    
    if env_template.exists() and not env_file.exists():
        shutil.copy(env_template, env_file)
        print("  ✅ Created .env file from template")
        print("  ⚠️  Please edit .env file with your API keys and configuration")
        return True
    elif env_file.exists():
        print("  ✅ .env file already exists")
        return True
    else:
        print("  ❌ .env.template not found")
        return False


def run_tests():
    """Run basic tests to verify setup."""
    print("\n🧪 Running setup tests...")
    
    try:
        # Test imports
        print("  🔍 Testing imports...")
        test_imports = [
            'pandas',
            'numpy',
            'yfinance',
            'plotly',
            'sklearn'
        ]
        
        for module in test_imports:
            try:
                __import__(module)
                print(f"    ✅ {module}")
            except ImportError:
                print(f"    ❌ {module} - not installed")
        
        # Test TA-Lib specifically
        try:
            import talib
            print("    ✅ talib")
        except ImportError:
            print("    ❌ talib - manual installation required")
        
        # Run basic configuration test
        try:
            from config.config import TRADING_CONFIG
            print("  ✅ Configuration loaded successfully")
        except Exception as e:
            print(f"  ❌ Configuration error: {e}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Test failed: {e}")
        return False


def print_next_steps():
    """Print next steps for the user."""
    print("\n🎉 Setup completed!")
    print("\n📋 Next steps:")
    print("  1. 🔑 Edit .env file with your API keys:")
    print("     - Get Alpaca API keys from https://alpaca.markets/")
    print("     - Configure email settings if desired")
    print("     - Set trading mode to 'paper' for testing")
    print("")
    print("  2. 🧪 Run tests to verify everything works:")
    print("     python tests/run_tests.py")
    print("")
    print("  3. 📊 Run backtesting to test strategies:")
    print("     python main.py --backtest-only")
    print("")
    print("  4. 📈 Start paper trading:")
    print("     python main.py --paper-trading")
    print("")
    print("  5. 📚 Read README.md for detailed instructions")
    print("")
    print("⚠️  IMPORTANT WARNINGS:")
    print("   - Always start with paper trading")
    print("   - Never risk money you can't afford to lose")
    print("   - Understand all strategies before using them")
    print("   - Monitor your bot regularly")


def main():
    """Main setup function."""
    print("🚀 Algorithmic Trading Bot Setup")
    print("=" * 50)
    
    success = True
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Create directories
    create_directories()
    
    # Install dependencies
    if not install_dependencies():
        success = False
    
    # Install TA-Lib
    if not install_talib():
        print("  ⚠️  TA-Lib installation may require manual steps")
    
    # Setup environment
    if not setup_environment():
        success = False
    
    # Run tests
    if not run_tests():
        print("  ⚠️  Some tests failed - check dependencies")
    
    # Print next steps
    if success:
        print_next_steps()
    else:
        print("\n❌ Setup completed with some issues")
        print("   Please review the errors above and fix them before proceeding")
    
    return success


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Setup Algorithmic Trading Bot')
    parser.add_argument('--skip-deps', action='store_true',
                       help='Skip dependency installation')
    parser.add_argument('--skip-talib', action='store_true',
                       help='Skip TA-Lib installation')
    parser.add_argument('--skip-tests', action='store_true',
                       help='Skip running tests')
    
    args = parser.parse_args()
    
    success = main()
    sys.exit(0 if success else 1)
