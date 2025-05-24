"""
Real-time Trading Dashboard
Web-based interface for monitoring trading bot performance
"""
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import json
import os
import sys
import sqlite3
from datetime import datetime, timedelta
import pandas as pd
import plotly
import plotly.graph_objs as go
import plotly.express as px

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from execution.execution_engine import execution_engine
from config.config import CONFIG

app = Flask(__name__)
app.config['SECRET_KEY'] = 'trading_bot_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*")

class DashboardData:
    """Data provider for dashboard"""
    
    def __init__(self):
        self.db_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'trading_data.db')
    
    def get_portfolio_summary(self):
        """Get current portfolio summary"""
        try:
            summary = execution_engine.get_portfolio_summary()
            return summary
        except:
            return {
                'total_value': 100000,
                'positions': 0,
                'unrealized_pnl': 0,
                'daily_pnl': 0,
                'position_details': {}
            }
    
    def get_performance_history(self, days=30):
        """Get portfolio performance history"""
        try:
            conn = sqlite3.connect(self.db_path)
            query = """
            SELECT date, portfolio_value, daily_return 
            FROM portfolio_history 
            WHERE date >= date('now', '-{} days')
            ORDER BY date
            """.format(days)
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            return df.to_dict('records')
        except:
            # Mock data for demonstration
            dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
            mock_data = []
            base_value = 100000
            
            for i, date in enumerate(dates):
                daily_return = (i * 0.001) + (i % 7 - 3) * 0.002  # Slight upward trend with noise
                portfolio_value = base_value * (1 + daily_return)
                
                mock_data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'portfolio_value': portfolio_value,
                    'daily_return': daily_return
                })
            
            return mock_data
    
    def get_recent_trades(self, limit=20):
        """Get recent trades"""
        try:
            return execution_engine.executed_orders[-limit:]
        except:
            return []
    
    def get_strategy_performance(self):
        """Get strategy performance breakdown"""
        # Mock data for demonstration
        return {
            'mean_reversion': {'return': 0.05, 'win_rate': 0.65, 'trades': 45},
            'momentum': {'return': 0.08, 'win_rate': 0.58, 'trades': 32},
            'breakout': {'return': 0.03, 'win_rate': 0.72, 'trades': 28},
            'ml_ensemble': {'return': 0.12, 'win_rate': 0.68, 'trades': 38}
        }

dashboard_data = DashboardData()

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/portfolio')
def api_portfolio():
    """API endpoint for portfolio data"""
    return jsonify(dashboard_data.get_portfolio_summary())

@app.route('/api/performance')
def api_performance():
    """API endpoint for performance data"""
    days = request.args.get('days', 30, type=int)
    return jsonify(dashboard_data.get_performance_history(days))

@app.route('/api/trades')
def api_trades():
    """API endpoint for recent trades"""
    limit = request.args.get('limit', 20, type=int)
    return jsonify(dashboard_data.get_recent_trades(limit))

@app.route('/api/strategies')
def api_strategies():
    """API endpoint for strategy performance"""
    return jsonify(dashboard_data.get_strategy_performance())

@app.route('/api/performance_chart')
def api_performance_chart():
    """Generate performance chart"""
    data = dashboard_data.get_performance_history(90)
    
    df = pd.DataFrame(data)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['portfolio_value'],
        mode='lines',
        name='Portfolio Value',
        line=dict(color='#2E86AB', width=2)
    ))
    
    fig.update_layout(
        title='Portfolio Performance',
        xaxis_title='Date',
        yaxis_title='Portfolio Value ($)',
        template='plotly_white',
        height=400
    )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

@app.route('/api/strategy_chart')
def api_strategy_chart():
    """Generate strategy performance chart"""
    strategies = dashboard_data.get_strategy_performance()
    
    strategy_names = list(strategies.keys())
    returns = [strategies[s]['return'] * 100 for s in strategy_names]
    
    fig = go.Figure(data=[
        go.Bar(x=strategy_names, y=returns, 
               marker_color=['#A23B72', '#F18F01', '#C73E1D', '#2E86AB'])
    ])
    
    fig.update_layout(
        title='Strategy Returns (%)',
        xaxis_title='Strategy',
        yaxis_title='Return (%)',
        template='plotly_white',
        height=400
    )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print('Client connected')
    # Send initial data
    emit('portfolio_update', dashboard_data.get_portfolio_summary())

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print('Client disconnected')

def broadcast_portfolio_update():
    """Broadcast portfolio updates to all connected clients"""
    portfolio_data = dashboard_data.get_portfolio_summary()
    socketio.emit('portfolio_update', portfolio_data)

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Run the dashboard
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
