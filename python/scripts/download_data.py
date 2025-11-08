"""
Script para download de dados históricos de Bitcoin
"""
import ccxt
import pandas as pd
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_bitcoin_data(
    symbol: str = 'BTC/USDT',
    timeframe: str = '1h',
    days_back: int = 730,
    output_file: str = 'data/bitcoin_historical.csv'
):
    """
    Download historical Bitcoin data from Binance
    
    Args:
        symbol: Trading pair
        timeframe: Timeframe (1h, 4h, 1d, etc)
        days_back: Days of historical data
        output_file: Output CSV file path
    """
    logger.info(f"Downloading {days_back} days of {symbol} data...")
    
    try:
        # Initialize exchange
        exchange = ccxt.binance({
            'enableRateLimit': True,
        })
        
        # Calculate timestamp
        since = exchange.parse8601(
            (datetime.now() - timedelta(days=days_back)).isoformat()
        )
        
        # Download data
        all_ohlcv = []
        while True:
            logger.info(f"Fetching data from {datetime.fromtimestamp(since/1000)}")
            
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since)
            
            if len(ohlcv) == 0:
                break
            
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1
            
            if len(ohlcv) < 1000:  # No more data
                break
        
        # Convert to DataFrame
        df = pd.DataFrame(
            all_ohlcv,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        
        # Convert timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Save
        df.to_csv(output_file, index=False)
        
        logger.info(f"✓ Downloaded {len(df)} candles")
        logger.info(f"✓ Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        logger.info(f"✓ Saved to {output_file}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error downloading data: {e}")
        raise


if __name__ == "__main__":
    download_bitcoin_data()
