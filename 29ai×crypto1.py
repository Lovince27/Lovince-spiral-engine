#!/usr/bin/env python3
"""
ULTIMATE AI CRYPTOCURRENCY TRADING SYSTEM
-----------------------------------------
Features:
1. Real-time data integration
2. Advanced technical indicators
3. Multiple AI/ML models
4. Sentiment analysis
5. Portfolio management
6. Risk management
7. Backtesting framework
8. Trading execution
"""

import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from ta import add_all_ta_features
from ta.momentum import RSIIndicator, StochasticOscillator, MACD
from ta.volatility import BollingerBands, AverageTrueRange
from ta.trend import EMAIndicator, IchimokuIndicator
from transformers import pipeline
import alpaca_trade_api as tradeapi
import backtrader as bt
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Configuration
class Config:
    API_KEY = 'your_api_key'
    API_SECRET = 'your_api_secret'
    BASE_URL = 'https://paper-api.alpaca.markets'  # Use paper trading
    ASSET = 'BTC/USD'
    TIMEFRAME = '1D'
    RISK_PER_TRADE = 0.02  # 2% of portfolio
    MAX_POSITION = 0.2  # 20% of portfolio max

# 1. Data Module
class CryptoData:
    def __init__(self):
        self.api = tradeapi.REST(Config.API_KEY, Config.API_SECRET, Config.BASE_URL)
        
    def get_real_time_data(self, days=365):
        """Fetch real-time market data"""
        end = datetime.now()
        start = end - timedelta(days=days)
        bars = self.api.get_crypto_bars(Config.ASSET, Config.TIMEFRAME, start.isoformat(), end.isoformat()).df
        return bars
    
    def add_technical_indicators(self, df):
        """Add comprehensive technical indicators"""
        # Trend indicators
        df['EMA_20'] = EMAIndicator(df['close'], window=20).ema_indicator()
        df['EMA_50'] = EMAIndicator(df['close'], window=50).ema_indicator()
        
        # Momentum indicators
        df['RSI_14'] = RSIIndicator(df['close'], window=14).rsi()
        df['MACD'] = MACD(df['close']).macd()
        df['Stoch_%K'] = StochasticOscillator(df['high'], df['low'], df['close']).stoch()
        
        # Volatility indicators
        bb = BollingerBands(df['close'])
        df['BB_upper'] = bb.bollinger_hband()
        df['BB_middle'] = bb.bollinger_mavg()
        df['BB_lower'] = bb.bollinger_lband()
        df['ATR'] = AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
        
        # Ichimoku Cloud
        ichimoku = IchimokuIndicator(df['high'], df['low'])
        df['Ichimoku_a'] = ichimoku.ichimoku_a()
        df['Ichimoku_b'] = ichimoku.ichimoku_b()
        df['Ichimoku_base'] = ichimoku.ichimoku_base_line()
        df['Ichimoku_conversion'] = ichimoku.ichimoku_conversion_line()
        
        # Volume indicators
        df['VWAP'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
        
        return df
    
    def get_sentiment(self):
        """Fetch crypto sentiment from news and social media"""
        sentiment_analyzer = pipeline("sentiment-analysis", model="finiteautomata/bertweet-base-sentiment-analysis")
        
        # Get recent crypto news (mock - in production use NewsAPI/Twitter API)
        news = [
            "Bitcoin reaches new all-time high amid institutional adoption",
            "Regulatory concerns cause market pullback",
            "Ethereum upgrade scheduled for next month"
        ]
        
        sentiments = [sentiment_analyzer(headline)[0]['label'] for headline in news]
        return np.mean([1 if s == 'POS' else (-1 if s == 'NEG' else 0) for s in sentiments])

# 2. AI Engine
class AIEngine:
    def __init__(self):
        self.models = {
            'RandomForest': RandomForestRegressor(n_estimators=200, random_state=42),
            'GBoost': GradientBoostingRegressor(n_estimators=150, random_state=42),
            'NeuralNet': MLPRegressor(hidden_layer_sizes=(50, 25), max_iter=1000)
        }
        self.scaler = MinMaxScaler()
        
    def prepare_features(self, df):
        """Prepare features for ML models"""
        features = df.drop(columns=['open', 'high', 'low', 'volume', 'trade_count', 'vwap'])
        features['target'] = features['close'].shift(-1)
        features.dropna(inplace=True)
        
        X = features.drop(columns=['target'])
        y = features['target']
        
        # Time-series cross validation
        tscv = TimeSeriesSplit(n_splits=5)
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_models(self, X_train, y_train):
        """Train ensemble of models"""
        trained_models = {}
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            trained_models[name] = model
        return trained_models
    
    def predict(self, models, X):
        """Get predictions from all models"""
        predictions = {}
        for name, model in models.items():
            predictions[name] = model.predict(X)[0]
        return predictions

# 3. Trading Engine
class TradingEngine:
    def __init__(self):
        self.data = CryptoData()
        self.ai = AIEngine()
        self.api = tradeapi.REST(Config.API_KEY, Config.API_SECRET, Config.BASE_URL)
        
    def calculate_position_size(self, current_price, stop_loss, portfolio_value):
        """Calculate position size based on risk management"""
        risk_amount = portfolio_value * Config.RISK_PER_TRADE
        risk_per_unit = abs(current_price - stop_loss)
        position_size = min(risk_amount / risk_per_unit, 
                           (portfolio_value * Config.MAX_POSITION) / current_price)
        return position_size
    
    def generate_signals(self, df, models, portfolio_value):
        """Generate comprehensive trading signals"""
        # Prepare latest features
        latest = df.iloc[-1:].drop(columns=['open', 'high', 'low', 'volume', 'trade_count', 'vwap'])
        latest_scaled = self.ai.scaler.transform(latest)
        
        # Get predictions
        predictions = self.ai.predict(models, latest_scaled)
        consensus = np.mean(list(predictions.values()))
        
        # Current market state
        current_price = df['close'].iloc[-1]
        rsi = df['RSI_14'].iloc[-1]
        atr = df['ATR'].iloc[-1]
        
        # Sentiment analysis
        sentiment_score = self.data.get_sentiment()
        
        # Determine stop loss
        stop_loss_long = current_price - 2 * atr
        stop_loss_short = current_price + 2 * atr
        
        # Generate signals
        signal = "HOLD"
        confidence = 0
        position_size = 0
        
        # Bullish conditions
        bullish_conditions = (
            consensus > current_price * 1.01 and
            rsi < 70 and
            sentiment_score > 0 and
            df['EMA_20'].iloc[-1] > df['EMA_50'].iloc[-1] and
            current_price > df['Ichimoku_a'].iloc[-1]
        )
        
        # Bearish conditions
        bearish_conditions = (
            consensus < current_price * 0.99 and
            rsi > 30 and
            sentiment_score < 0 and
            df['EMA_20'].iloc[-1] < df['EMA_50'].iloc[-1] and
            current_price < df['Ichimoku_b'].iloc[-1]
        )
        
        if bullish_conditions:
            signal = "BUY"
            confidence = (consensus - current_price) / current_price
            position_size = self.calculate_position_size(current_price, stop_loss_long, portfolio_value)
        elif bearish_conditions:
            signal = "SELL"
            confidence = (current_price - consensus) / current_price
            position_size = self.calculate_position_size(current_price, stop_loss_short, portfolio_value)
            
        return signal, confidence, position_size, predictions
    
    def backtest(self, df):
        """Backtest strategy using Backtrader"""
        class CryptoStrategy(bt.Strategy):
            params = (
                ('rsi_period', 14),
                ('rsi_overbought', 70),
                ('rsi_oversold', 30)
            )
            
            def __init__(self):
                self.rsi = bt.indicators.RSI(self.data.close, period=self.p.rsi_period)
                self.ema20 = bt.indicators.EMA(self.data.close, period=20)
                self.ema50 = bt.indicators.EMA(self.data.close, period=50)
                
            def next(self):
                if not self.position:
                    if (self.rsi < self.p.rsi_oversold and 
                        self.ema20 > self.ema50):
                        self.buy()
                else:
                    if (self.rsi > self.p.rsi_overbought or 
                        self.ema20 < self.ema50):
                        self.sell()
        
        cerebro = bt.Cerebro()
        data = bt.feeds.PandasData(dataname=df)
        cerebro.adddata(data)
        cerebro.addstrategy(CryptoStrategy)
        cerebro.broker.setcash(100000)
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        
        print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
        results = cerebro.run()
        print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
        
        strat = results[0]
        print('Sharpe Ratio:', strat.analyzers.sharpe.get_analysis()['sharperatio'])
        print('Max Drawdown:', strat.analyzers.drawdown.get_analysis()['max']['drawdown'])
        
        cerebro.plot(style='candlestick')

# 4. Execution Module
class ExecutionEngine:
    def __init__(self):
        self.api = tradeapi.REST(Config.API_KEY, Config.API_SECRET, Config.BASE_URL)
        
    def execute_trade(self, signal, symbol, qty):
        """Execute trade through broker API"""
        try:
            if signal == "BUY":
                self.api.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side='buy',
                    type='market',
                    time_in_force='gtc'
                )
            elif signal == "SELL":
                self.api.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side='sell',
                    type='market',
                    time_in_force='gtc'
                )
            return True
        except Exception as e:
            print(f"Trade execution failed: {e}")
            return False

# Main Trading Bot
class CryptoTradingBot:
    def __init__(self):
        self.data = CryptoData()
        self.ai = AIEngine()
        self.trading = TradingEngine()
        self.execution = ExecutionEngine()
        
    def run(self):
        """Main execution loop"""
        print("=== CRYPTO AI TRADING BOT ===")
        
        # 1. Get and prepare data
        df = self.data.get_real_time_data()
        df = self.data.add_technical_indicators(df)
        
        # 2. Train AI models
        X_train, X_test, y_train, y_test = self.ai.prepare_features(df)
        models = self.ai.train_models(X_train, y_train)
        
        # 3. Get portfolio value
        portfolio = self.execution.api.get_account()
        portfolio_value = float(portfolio.equity)
        
        # 4. Generate signals
        signal, confidence, position_size, predictions = self.trading.generate_signals(df, models, portfolio_value)
        
        # 5. Display dashboard
        self.display_dashboard(df, signal, confidence, position_size, predictions)
        
        # 6. Execute trade if signal
        if signal != "HOLD":
            print(f"\nExecuting {signal} order for {position_size:.4f} {Config.ASSET}")
            self.execution.execute_trade(signal, Config.ASSET.split('/')[0], position_size)
        
        # 7. Run backtest
        print("\nRunning Backtest...")
        self.trading.backtest(df)
        
    def display_dashboard(self, df, signal, confidence, position_size, predictions):
        """Display comprehensive trading dashboard"""
        current_price = df['close'].iloc[-1]
        
        print("\n" + "="*70)
        print(f"{'AI CRYPTO TRADING DASHBOARD':^70}")
        print("="*70)
        
        print(f"\n{'Market Data':^70}")
        print("-"*70)
        print(f"Current Price: ${current_price:.2f} | RSI(14): {df['RSI_14'].iloc[-1]:.2f}")
        print(f"EMA(20): ${df['EMA_20'].iloc[-1]:.2f} | EMA(50): ${df['EMA_50'].iloc[-1]:.2f}")
        print(f"MACD: {df['MACD'].iloc[-1]:.2f} | ATR: {df['ATR'].iloc[-1]:.2f}")
        
        print(f"\n{'AI Predictions':^70}")
        print("-"*70)
        for model, pred in predictions.items():
            print(f"{model+':':<15} ${pred:.2f} ({((pred-current_price)/current_price*100):+.2f}%)")
        print(f"\nConsensus Prediction: ${np.mean(list(predictions.values())):.2f}")
        
        print(f"\n{'Trading Signal':^70}")
        print("-"*70)
        print(f"SIGNAL: {signal} | Confidence: {confidence*100:.1f}%")
        print(f"Position Size: {position_size:.4f} {Config.ASSET.split('/')[0]}")
        
        print("\n" + "="*70)

if __name__ == "__main__":
    bot = CryptoTradingBot()
    bot.run()