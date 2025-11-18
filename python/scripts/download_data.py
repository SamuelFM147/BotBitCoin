"""
Script para download de dados históricos (BTC via CCXT, XAUUSD via Yahoo Finance)
"""
import ccxt
import pandas as pd
import os
from datetime import datetime, timedelta
import logging
import yfinance as yf
import requests
try:
    import MetaTrader5 as MT5
except Exception:
    MT5 = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_bitcoin_data(
    symbol: str = 'BTC/USDT',
    timeframe: str = '1h',
    days_back: int = 730,
    output_file: str = 'python/data/bitcoin_historical.csv'
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
            
            # Fetch up to 1000 candles per request for faster backfill
            limit = 1000
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit=limit)
            
            if len(ohlcv) == 0:
                break
            
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1
            
            if len(ohlcv) < limit:  # No more data
                break
        
        # Convert to DataFrame
        df = pd.DataFrame(
            all_ohlcv,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        
        # Convert timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Ensure output directory exists and save
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df.to_csv(output_file, index=False)
        
        logger.info(f"✓ Downloaded {len(df)} candles")
        logger.info(f"✓ Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        logger.info(f"✓ Saved to {output_file}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error downloading data: {e}")
        raise


def download_yfinance_data(
    symbol: str,
    interval: str,
    start: str,
    end: str,
    output_file: str
) -> pd.DataFrame:
    """
    Download de dados via Yahoo Finance (yfinance), salvando em CSV no formato OHLCV
    """
    logger.info(f"Downloading {symbol} from {start} to {end} at interval {interval}...")
    try:
        candidates = [symbol]
        if symbol.upper() in {"XAUUSD=X", "XAUUSD"}:
            candidates += ["GC=F"]
        df = pd.DataFrame()
        last_err: Exception | None = None
        for sym in candidates:
            logger.info(f"Trying symbol {sym}")
            try:
                df = yf.download(
                    tickers=sym,
                    interval=interval,
                    start=start,
                    end=end,
                    auto_adjust=False,
                    progress=False,
                )
                if not df.empty:
                    symbol = sym
                    break
            except Exception as e:
                last_err = e
                logger.warning(f"yfinance error for {sym}: {e}")
        if df.empty:
            raise RuntimeError("No data returned from yfinance")

        df = df.reset_index()
        # Padroniza colunas
        rename_map = {
            'Datetime': 'timestamp',
            'Date': 'timestamp',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
        }
        df = df.rename(columns=rename_map)

        # Garante tipos consistentes
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df.to_csv(output_file, index=False)

        logger.info(f"✓ Downloaded {len(df)} rows")
        logger.info(f"✓ Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        logger.info(f"✓ Saved to {output_file}")

        return df
    except Exception as e:
        logger.error(f"Error downloading yfinance data: {e}")
        # Fallback para Stooq (diário)
        try:
            base_symbol = 'xauusd' if 'XAU' in symbol.upper() else symbol.lower()
            url = f"https://stooq.com/q/d/l/?s={base_symbol}&i=d"
            logger.info(f"Trying Stooq daily CSV: {url}")
            df = pd.read_csv(url)
            if df.empty:
                raise RuntimeError("No data returned from Stooq")
            df = df.rename(columns={
                'Date': 'timestamp',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume',
            })
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            mask = (df['timestamp'] >= pd.to_datetime(start)) & (df['timestamp'] <= pd.to_datetime(end))
            df = df.loc[mask]
            if df.empty:
                raise RuntimeError("Stooq returned data but none in date range")
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            df.to_csv(output_file, index=False)
            logger.info(f"✓ Downloaded {len(df)} rows (Stooq)")
            logger.info(f"✓ Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            logger.info(f"✓ Saved to {output_file}")
            return df
        except Exception as e2:
            logger.error(f"Fallback failed: {e2}")
            raise


def download_xauusd_datasets(interval: str = '1d') -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Baixa conjuntos de teste (2020–2023) e treino (2025) para XAUUSD
    """
    test_df = download_yfinance_data(
        symbol='XAUUSD=X',
        interval=interval,
        start='2020-01-01',
        end='2023-12-31',
        output_file='python/data/xauusd_test.csv',
    )
    train_df = download_yfinance_data(
        symbol='XAUUSD=X',
        interval=interval,
        start='2025-01-01',
        end='2025-12-31',
        output_file='python/data/xauusd_train_2025.csv',
    )
    return test_df, train_df


def _mt5_init(mt5_path: str | None = None) -> None:
    if MT5 is None:
        raise RuntimeError("MetaTrader5 package not installed. Use 'poetry add MetaTrader5'.")
    ok = MT5.initialize(path=mt5_path) if mt5_path else MT5.initialize()
    if not ok:
        raise RuntimeError(f"MT5 initialize failed: {MT5.last_error()}")


def _mt5_resolve_symbol(base: str = "XAUUSD") -> str:
    symbols = MT5.symbols_get()
    candidates = [s.name for s in symbols if ("XAU" in s.name.upper() and "USD" in s.name.upper())]
    if base in candidates:
        sym = base
    elif candidates:
        sym = candidates[0]
    else:
        sym = base
    MT5.symbol_select(sym, True)
    return sym


def _mt5_timeframe(s: str) -> int:
    key = s.strip().lower()
    if key in {"15m", "m15", "15min"}:
        return MT5.TIMEFRAME_M15
    if key in {"1d", "d1", "daily"}:
        return MT5.TIMEFRAME_D1
    if key in {"1h", "h1"}:
        return MT5.TIMEFRAME_H1
    raise ValueError("Unsupported MT5 timeframe")


def download_mt5_data(symbol: str, timeframe: str, start: str, end: str, output_file: str, mt5_path: str | None = None) -> pd.DataFrame:
    _mt5_init(mt5_path)
    tf = _mt5_timeframe(timeframe)
    sym = _mt5_resolve_symbol(symbol)
    t0 = datetime.fromisoformat(start)
    t1 = datetime.fromisoformat(end)
    rates = MT5.copy_rates_range(sym, tf, t0, t1)
    if rates is None or len(rates) == 0:
        raise RuntimeError("No MT5 data returned")
    df = pd.DataFrame(rates)
    df['timestamp'] = pd.to_datetime(df['time'], unit='s')
    df = df.rename(columns={
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'close': 'close',
        'tick_volume': 'volume',
    })[["timestamp", "open", "high", "low", "close", "volume"]]
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    logger.info(f"✓ Downloaded {len(df)} rows (MT5 {timeframe})")
    logger.info(f"✓ Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    logger.info(f"✓ Saved to {output_file}")
    MT5.shutdown()
    return df


def download_xauusd_m15_mt5_datasets(mt5_path: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    test_df = download_mt5_data(
        symbol="XAUUSD",
        timeframe="15m",
        start="2020-01-01",
        end="2023-12-31",
        output_file="python/data/xauusd_test_m15.csv",
        mt5_path=mt5_path,
    )
    train_df = download_mt5_data(
        symbol="XAUUSD",
        timeframe="15m",
        start="2025-01-01",
        end="2025-12-31",
        output_file="python/data/xauusd_train_2025_m15.csv",
        mt5_path=mt5_path,
    )
    return test_df, train_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog="download_data",
        description="Coleta de dados: BTC via CCXT ou XAUUSD via yfinance",
    )
    parser.add_argument("--asset", choices=["btc", "xauusd"], default="btc")
    parser.add_argument("--timeframe", default="1d")
    parser.add_argument("--days_back", type=int, default=730)
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--output_file")
    parser.add_argument("--source", choices=["yfinance", "mt5"], default="yfinance")
    parser.add_argument("--mt5_path")

    args = parser.parse_args()

    if args.asset == "xauusd":
        if args.source == "mt5":
            if args.start and args.end:
                out = args.output_file or "python/data/xauusd_custom_m15.csv"
                download_mt5_data(
                    symbol="XAUUSD",
                    timeframe=args.timeframe,
                    start=args.start,
                    end=args.end,
                    output_file=out,
                    mt5_path=args.mt5_path,
                )
            else:
                download_xauusd_m15_mt5_datasets(mt5_path=args.mt5_path)
        else:
            if args.start and args.end:
                out = args.output_file or "python/data/xauusd_custom.csv"
                download_yfinance_data(
                    symbol="XAUUSD=X",
                    interval=args.timeframe,
                    start=args.start,
                    end=args.end,
                    output_file=out,
                )
            else:
                download_xauusd_datasets(interval=args.timeframe)
    else:
        out = args.output_file or "python/data/bitcoin_historical.csv"
        download_bitcoin_data(
            symbol="BTC/USDT",
            timeframe=args.timeframe,
            days_back=args.days_back,
            output_file=out,
        )
