"""
Machine Learning Ensemble Strategy
Uses multiple ML models to predict price movements with YDF (Yggdrasil Decision Forests)
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from strategies.base_strategy import BaseStrategy, Signal, SignalType
import ydf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

class MLEnsembleStrategy(BaseStrategy):
    """
    Machine Learning ensemble strategy combining multiple ML models
    """
    
    def __init__(self, parameters: Dict = None):
        default_params = {
            'lookback_period': 60,
            'prediction_horizon': 5,  # Predict 5 days ahead
            'min_prediction_confidence': 0.6,
            'retrain_frequency': 30,  # Retrain every 30 days
            'feature_importance_threshold': 0.01,
            'ensemble_threshold': 0.7,  # Minimum ensemble agreement
            'vol_adjustment': True,
            'fundamental_weight': 0.3
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__("ML Ensemble", default_params)
        
        # Initialize YDF models
        self.models = {
            'random_forest': ydf.RandomForestLearner(
                label="target",
                num_trees=100,
                max_depth=10,
                random_seed=42
            ),
            'gradient_boosted_trees': ydf.GradientBoostedTreesLearner(
                label="target",
                num_trees=100,
                max_depth=6,
                random_seed=42
            )
        }
        
        # Remove scalers as YDF doesn't need feature scaling
        self.trained_models = {}  # Store trained models separately
        self.last_training_date = {}
        self.feature_importance = {}
    
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> List[Signal]:
        """
        Generate ML-based trading signals
        """
        signals = []
        
        for symbol, df in data.items():
            if len(df) < self.parameters['lookback_period'] * 2:
                continue
            
            try:
                # Prepare features
                features_df = self._create_features(df)
                
                # Check if we need to retrain models
                if self._should_retrain(symbol, df):
                    self._train_models(symbol, features_df)
                
                # Generate prediction
                prediction, confidence = self._predict(symbol, features_df)
                
                if prediction is not None and confidence >= self.parameters['min_prediction_confidence']:
                    signal = self._create_signal(symbol, df, prediction, confidence)
                    if signal:
                        signals.append(signal)
                        
            except Exception as e:
                print(f"Error generating ML signal for {symbol}: {e}")
                continue
        
        return signals
    
    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive feature set for ML models
        """
        features = pd.DataFrame(index=df.index)
        
        # Price-based features
        features['returns_1d'] = df['Close'].pct_change()
        features['returns_5d'] = df['Close'].pct_change(5)
        features['returns_10d'] = df['Close'].pct_change(10)
        features['returns_20d'] = df['Close'].pct_change(20)
        
        # Moving average features
        for window in [5, 10, 20, 50]:
            features[f'sma_{window}'] = df['Close'].rolling(window).mean()
            features[f'price_to_sma_{window}'] = df['Close'] / features[f'sma_{window}']
            features[f'sma_{window}_slope'] = features[f'sma_{window}'].diff(5)
        
        # Volatility features
        features['volatility_5d'] = df['Close'].pct_change().rolling(5).std()
        features['volatility_20d'] = df['Close'].pct_change().rolling(20).std()
        features['atr'] = self._calculate_atr(df, 14)
        features['atr_ratio'] = features['atr'] / df['Close']
        
        # Volume features
        features['volume_ma'] = df['Volume'].rolling(20).mean()
        features['volume_ratio'] = df['Volume'] / features['volume_ma']
        features['volume_momentum'] = df['Volume'].pct_change(5)
        
        # Technical indicators
        features['rsi'] = self._calculate_rsi(df['Close'], 14)
        features['macd'], features['macd_signal'] = self._calculate_macd(df['Close'])
        features['macd_histogram'] = features['macd'] - features['macd_signal']
        
        # Bollinger Bands
        bb_middle = df['Close'].rolling(20).mean()
        bb_std = df['Close'].rolling(20).std()
        features['bb_position'] = (df['Close'] - bb_middle) / (2 * bb_std)
        features['bb_width'] = (4 * bb_std) / bb_middle
        
        # Price patterns
        features['higher_high'] = (df['High'] > df['High'].shift(1)).astype(int)
        features['lower_low'] = (df['Low'] < df['Low'].shift(1)).astype(int)
        features['doji'] = (abs(df['Open'] - df['Close']) / (df['High'] - df['Low'])).fillna(0)
        
        # Market structure
        features['support'] = df['Low'].rolling(20).min()
        features['resistance'] = df['High'].rolling(20).max()
        features['price_position'] = (df['Close'] - features['support']) / (features['resistance'] - features['support'])
        
        # Momentum oscillators
        features['stoch_k'] = self._calculate_stochastic(df, 14)
        features['williams_r'] = self._calculate_williams_r(df, 14)
        
        # Gap analysis
        features['gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
        features['gap_filled'] = (features['gap'] > 0) & (df['Low'] <= df['Close'].shift(1))
        
        # Lagged features
        for lag in [1, 2, 3, 5]:
            features[f'returns_lag_{lag}'] = features['returns_1d'].shift(lag)
            features[f'volume_ratio_lag_{lag}'] = features['volume_ratio'].shift(lag)
            features[f'rsi_lag_{lag}'] = features['rsi'].shift(lag)
        
        # Drop rows with NaN values
        features = features.dropna()
        
        return features
    
    def _calculate_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        return true_range.rolling(window=period).mean()
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        return macd, macd_signal
    
    def _calculate_stochastic(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Stochastic %K"""
        low_min = df['Low'].rolling(window=period).min()
        high_max = df['High'].rolling(window=period).max()
        return 100 * (df['Close'] - low_min) / (high_max - low_min)
    
    def _calculate_williams_r(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Williams %R"""
        high_max = df['High'].rolling(window=period).max()
        low_min = df['Low'].rolling(window=period).min()
        return -100 * (high_max - df['Close']) / (high_max - low_min)
    
    def _create_target(self, df: pd.DataFrame) -> pd.Series:
        """
        Create target variable for ML training
        """
        horizon = self.parameters['prediction_horizon']
        future_returns = df['Close'].pct_change(horizon).shift(-horizon)
        
        # Create categorical target: 0 = sell, 1 = hold, 2 = buy
        target = pd.Series(1, index=df.index)  # Default to hold
        target[future_returns > 0.02] = 2  # Buy if return > 2%
        target[future_returns < -0.02] = 0  # Sell if return < -2%
        
        return target
    
    def _should_retrain(self, symbol: str, df: pd.DataFrame) -> bool:
        """
        Check if models should be retrained
        """
        if symbol not in self.last_training_date:
            return True
        
        last_date = self.last_training_date[symbol]
        current_date = df.index[-1]
        days_since_training = (current_date - last_date).days
        
        return days_since_training >= self.parameters['retrain_frequency']
    
    def _train_models(self, symbol: str, features_df: pd.DataFrame):
        """
        Train ML models for the symbol
        """
        try:
            # Prepare training data
            target = self._create_target(features_df)
            
            # Align features and target
            common_index = features_df.index.intersection(target.index)
            X = features_df.loc[common_index]
            y = target.loc[common_index]
            
            # Remove NaN values
            mask = ~(X.isna().any(axis=1) | y.isna())
            X = X[mask]
            y = y[mask]
            
            if len(X) < 100:  # Need sufficient training data
                return
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Prepare training data for YDF (no scaling needed)
            train_df = X_train.copy()
            train_df['target'] = y_train
            
            test_df = X_test.copy()
            test_df['target'] = y_test
            
            # Train YDF models (no scaling needed for tree-based models)
            for name, learner in self.models.items():
                try:
                    # Convert to YDF dataset and train
                    train_dataset = ydf.create_vertical_dataset(train_df)
                    model = learner.train(train_dataset)
                    
                    # Store the trained model
                    if symbol not in self.trained_models:
                        self.trained_models[symbol] = {}
                    self.trained_models[symbol][name] = model
                    
                    # Evaluate model
                    test_dataset = ydf.create_vertical_dataset(test_df)
                    evaluation = model.evaluate(test_dataset)
                    accuracy = evaluation.accuracy
                    print(f"{name} accuracy for {symbol}: {accuracy:.3f}")
                    
                except Exception as e:
                    print(f"Error training {name} for {symbol}: {e}")
            
            # Store feature importance
            if symbol in self.trained_models and 'random_forest' in self.trained_models[symbol]:
                try:
                    rf_model = self.trained_models[symbol]['random_forest']
                    if hasattr(rf_model, 'variable_importances'):
                        importances = rf_model.variable_importances()
                        self.feature_importance[symbol] = {
                            var.name: var.importance for var in importances
                        }
                except:
                    pass
            
            self.last_training_date[symbol] = features_df.index[-1]
            
        except Exception as e:
            print(f"Error training models for {symbol}: {e}")
    
    def _predict(self, symbol: str, features_df: pd.DataFrame) -> Tuple[int, float]:
        """
        Generate ensemble prediction using YDF models
        """
        try:
            if symbol not in self.trained_models:
                return None, 0
                
            # Get latest features
            latest_features = features_df.iloc[-1:].copy()
            
            predictions = []
            confidences = []
            
            # Get predictions from all trained models
            for name, model in self.trained_models[symbol].items():
                if hasattr(model, 'predict'):  # Check if model is trained
                    try:
                        # Create dataset for prediction
                        pred_dataset = ydf.create_vertical_dataset(latest_features)
                        pred = model.predict(pred_dataset)[0]
                        
                        # Get prediction probabilities for confidence
                        pred_proba = model.predict_proba(pred_dataset)[0]
                        confidence = max(pred_proba) if len(pred_proba) > 0 else 0.5
                        
                        predictions.append(pred)
                        confidences.append(confidence)
                    except Exception as e:
                        print(f"Error in prediction with {name}: {e}")
                        continue
            
            # Ensemble decision
            avg_prediction = np.mean(predictions)
            avg_confidence = np.mean(confidences)
            
            # Require agreement between models
            if len(set(predictions)) == 1:  # All models agree
                final_prediction = predictions[0]
                final_confidence = avg_confidence
            else:
                # Models disagree, require higher confidence
                if avg_confidence > self.parameters['ensemble_threshold']:
                    final_prediction = int(round(avg_prediction))
                    final_confidence = avg_confidence * 0.8  # Reduce confidence for disagreement
                else:
                    return None, 0
            
            return final_prediction, final_confidence
            
        except Exception as e:
            print(f"Error making prediction for {symbol}: {e}")
            return None, 0
    
    def _create_signal(self, symbol: str, df: pd.DataFrame, prediction: int, 
                      confidence: float) -> Signal:
        """
        Create trading signal from ML prediction
        """
        latest = df.iloc[-1]
        
        if prediction == 2:  # Buy signal
            signal_type = SignalType.BUY
        elif prediction == 0:  # Sell signal
            signal_type = SignalType.SELL
        else:  # Hold signal
            return None
        
        # Adjust confidence based on market conditions
        volatility = df['Close'].pct_change().rolling(20).std().iloc[-1]
        vol_adjustment = 1.0
        if self.parameters['vol_adjustment']:
            vol_adjustment = max(0.7, min(1.3, 0.02 / volatility))
        
        final_strength = confidence * vol_adjustment
        
        return Signal(
            symbol=symbol,
            signal_type=signal_type,
            strength=min(final_strength, 1.0),
            price=latest['Close'],
            timestamp=latest.name,
            metadata={
                'strategy': 'ml_ensemble',
                'prediction': prediction,
                'confidence': confidence,
                'volatility_adjustment': vol_adjustment,
                'feature_importance': self.feature_importance.get(symbol, {})
            }
        )
    
    def calculate_position_size(self, signal: Signal, portfolio_value: float, 
                              volatility: float) -> float:
        """
        Calculate position size for ML-based trades
        """
        # Base position size
        base_size = 0.06  # 6% base allocation
        
        # Adjust based on prediction confidence
        confidence = signal.metadata.get('confidence', 0.5)
        confidence_multiplier = 1 + (confidence - 0.5) * 2  # Range: 0-2
        
        # Volatility adjustment
        vol_adjustment = max(0.5, min(2.0, 0.15 / volatility))
        
        # Calculate final position size
        position_size = base_size * confidence_multiplier * vol_adjustment
        
        # Cap at maximum position size
        max_position = 0.12  # Maximum 12% for ML trades
        position_size = min(position_size, max_position)
        
        return position_size * portfolio_value / signal.price
    
    def generate_signal(self, data: pd.DataFrame) -> Dict:
        """
        Generate a single signal for demo trading (adapter for generate_signals)
        """
        try:
            # Create a temporary data dict for compatibility
            symbol = 'TEMP'
            temp_data = {symbol: data}
            
            # Get signals using the main method
            signals = self.generate_signals(temp_data)
            
            if signals:
                signal = signals[0]
                # Extract reason from metadata or create a descriptive reason
                metadata = signal.metadata or {}
                prediction = metadata.get('prediction', 'unknown')
                confidence = metadata.get('confidence', signal.strength)
                reason = f"ML prediction: {prediction}, confidence: {confidence:.2f}"
                
                return {
                    'action': 'buy' if signal.signal_type == SignalType.BUY else 'sell',
                    'confidence': signal.strength,
                    'reason': reason
                }
            else:
                return {'action': 'hold', 'confidence': 0.0, 'reason': 'No signal'}
                
        except Exception as e:
            return {'action': 'hold', 'confidence': 0.0, 'reason': f'Error: {e}'}
