"""
Feature engineering for Bitcoin trading
"""
import numpy as np
import pandas as pd
import ta
from typing import List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Creates technical indicators and features for trading
    """
    
    def __init__(self):
        self.feature_names = []
    
    def add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features"""
        logger.info("Adding price features...")
        
        # Price changes
        df['price_change'] = df['close'].pct_change()
        df['price_change_2h'] = df['close'].pct_change(periods=2)
        df['price_change_4h'] = df['close'].pct_change(periods=4)
        df['price_change_24h'] = df['close'].pct_change(periods=24)
        
        # High-Low spread
        df['hl_spread'] = (df['high'] - df['low']) / df['close']
        
        # Close position in range
        df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)
        
        return df
    
    def add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features"""
        logger.info("Adding volume features...")
        
        # Nem todos os datasets possuem coluna de volume (ex.: XAUUSD diário do Yahoo).
        # Para evitar falhas, sintetizamos um volume dummy quando ausente.
        if 'volume' not in df.columns:
            logger.warning("Coluna 'volume' não encontrada; sintetizando volume dummy = 1.0 para criação de *features* de volume.")
            df['volume'] = 1.0
        
        # Volume changes
        df['volume_change'] = df['volume'].pct_change()
        
        # Volume moving averages
        df['volume_ma_7'] = df['volume'].rolling(window=7).mean()
        df['volume_ma_30'] = df['volume'].rolling(window=30).mean()
        
        # Volume ratio
        df['volume_ratio'] = df['volume'] / (df['volume_ma_30'] + 1e-10)
        
        return df
    
    def add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators"""
        logger.info("Adding momentum indicators...")
        
        # RSI (Relative Strength Index)
        df['rsi_14'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        df['rsi_7'] = ta.momentum.RSIIndicator(df['close'], window=7).rsi()
        
        # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(
            df['high'], df['low'], df['close'], window=14, smooth_window=3
        )
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        # MACD (Moving Average Convergence Divergence)
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # Rate of Change
        df['roc'] = ta.momentum.ROCIndicator(df['close'], window=12).roc()
        
        return df
    
    def add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend indicators"""
        logger.info("Adding trend indicators...")
        
        # Moving Averages
        df['sma_7'] = ta.trend.SMAIndicator(df['close'], window=7).sma_indicator()
        df['sma_25'] = ta.trend.SMAIndicator(df['close'], window=25).sma_indicator()
        df['sma_99'] = ta.trend.SMAIndicator(df['close'], window=99).sma_indicator()
        
        # Exponential Moving Averages
        df['ema_12'] = ta.trend.EMAIndicator(df['close'], window=12).ema_indicator()
        df['ema_26'] = ta.trend.EMAIndicator(df['close'], window=26).ema_indicator()
        
        # MA crossovers
        df['sma_7_25_cross'] = (df['sma_7'] - df['sma_25']) / df['close']
        df['ema_12_26_cross'] = (df['ema_12'] - df['ema_26']) / df['close']
        
        # ADX (Average Directional Index)
        adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14)
        df['adx'] = adx.adx()
        df['adx_pos'] = adx.adx_pos()
        df['adx_neg'] = adx.adx_neg()
        
        # Ichimoku
        ichimoku = ta.trend.IchimokuIndicator(df['high'], df['low'])
        df['ichimoku_a'] = ichimoku.ichimoku_a()
        df['ichimoku_b'] = ichimoku.ichimoku_b()
        
        return df
    
    def add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility indicators"""
        logger.info("Adding volatility indicators...")
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_high'] = bollinger.bollinger_hband()
        df['bb_mid'] = bollinger.bollinger_mavg()
        df['bb_low'] = bollinger.bollinger_lband()
        df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['bb_mid']
        df['bb_position'] = (df['close'] - df['bb_low']) / (df['bb_high'] - df['bb_low'] + 1e-10)
        
        # ATR (Average True Range)
        df['atr'] = ta.volatility.AverageTrueRange(
            df['high'], df['low'], df['close'], window=14
        ).average_true_range()
        df['atr_ratio'] = df['atr'] / df['close']
        
        # Keltner Channel
        keltner = ta.volatility.KeltnerChannel(df['high'], df['low'], df['close'])
        df['keltner_high'] = keltner.keltner_channel_hband()
        df['keltner_low'] = keltner.keltner_channel_lband()
        
        return df
    
    def add_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add pattern recognition features"""
        logger.info("Adding pattern features...")
        
        # Candle patterns
        df['body_size'] = abs(df['close'] - df['open']) / df['close']
        df['upper_shadow'] = (df['high'] - df[['close', 'open']].max(axis=1)) / df['close']
        df['lower_shadow'] = (df[['close', 'open']].min(axis=1) - df['low']) / df['close']
        
        # Bullish/Bearish
        df['is_bullish'] = (df['close'] > df['open']).astype(int)
        
        return df
    
    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        logger.info("Adding time features...")
        
        if 'timestamp' in df.columns:
            df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
            df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
            df['day_of_month'] = pd.to_datetime(df['timestamp']).dt.day
            
            # Cyclical encoding
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        return df
    
    def engineer_features(self, df: pd.DataFrame, include_all: bool = True) -> pd.DataFrame:
        """
        Apply all feature engineering steps
        
        Args:
            df: DataFrame with OHLCV data
            include_all: If True, include all feature types
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("Starting feature engineering pipeline...")
        
        df = df.copy()
        
        # Add all feature types
        df = self.add_price_features(df)
        df = self.add_volume_features(df)
        df = self.add_momentum_indicators(df)
        df = self.add_trend_indicators(df)
        df = self.add_volatility_indicators(df)
        df = self.add_pattern_features(df)
        df = self.add_time_features(df)
        
        # Drop rows with NaN values created by indicators
        initial_rows = len(df)
        df = df.dropna()
        logger.info(f"Dropped {initial_rows - len(df)} rows with NaN values")
        
        # Store feature names (excluding original OHLCV and timestamp)
        self.feature_names = [col for col in df.columns 
                             if col not in ['open', 'high', 'low', 'close', 'volume', 'timestamp']]
        
        logger.info(f"Feature engineering complete. Total features: {len(self.feature_names)}")
        
        return df
    
    def get_feature_importance_names(self) -> List[str]:
        """Get list of engineered feature names"""
        return self.feature_names


if __name__ == "__main__":
    # Example usage
    engineer = FeatureEngineer()
    
    # This would process your data
    # df = pd.read_csv('data/bitcoin_ohlcv.csv')
    # df_features = engineer.engineer_features(df)
