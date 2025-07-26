import os
from pathlib import Path
import sys
import pandas as pd
# -----------------------------------------------------------------------------
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(''))))
sys.path.append(root + '/python')
import ccxt
# -----------------------------------------------------------------------------
def retry_fetch_ohlcv(exchange, max_retries, symbol, timeframe, since, limit):
    num_retries = 0
    try:
        num_retries += 1
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
        # print('Fetched', len(ohlcv), symbol, 'candles from', exchange.iso8601 (ohlcv[0][0]), 'to', exchange.iso8601 (ohlcv[-1][0]))
        return ohlcv
    except Exception:
        if num_retries > max_retries:
            raise  # Exception('Failed to fetch', timeframe, symbol, 'OHLCV in', max_retries, 'attempts')

def scrape_ohlcv(exchange, max_retries, symbol, timeframe, since, limit):
    earliest_timestamp = exchange.milliseconds()
    timeframe_duration_in_seconds = exchange.parse_timeframe(timeframe)
    timeframe_duration_in_ms = timeframe_duration_in_seconds * 1000
    timedelta = limit * timeframe_duration_in_ms
    all_ohlcv = []
    while True:
        fetch_since = earliest_timestamp - timedelta
        ohlcv = retry_fetch_ohlcv(exchange, max_retries, symbol, timeframe, fetch_since, limit)
        # if we have reached the beginning of history
        if ohlcv[0][0] >= earliest_timestamp:
            break
        earliest_timestamp = ohlcv[0][0]
        all_ohlcv = ohlcv + all_ohlcv
        #print(len(all_ohlcv), symbol, 'candles in total from', exchange.iso8601(all_ohlcv[0][0]), 'to', exchange.iso8601(all_ohlcv[-1][0]))
        # if we have reached the checkpoint
        if fetch_since < since:
            break
    return all_ohlcv

def ohlcv_to_dataframe(ohlcv_data):
    """
    Convierte los datos OHLCV a un DataFrame de pandas con columnas nombradas correctamente
    """
    df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    
    # Convertir timestamp a datetime
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Reordenar columnas para tener datetime primero
    df = df[['datetime', 'timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']]
    
    return df

def scrape_candles_to_dataframe(exchange_id, max_retries, symbol, timeframe, since, limit):
    """
    Scraping de velas y retorno directo como DataFrame de pandas
    """
    # instantiate the exchange by id
    exchange = getattr(ccxt, exchange_id)({
        'enableRateLimit': True,  # required by the Manual
    })
    # convert since from string to milliseconds integer if needed
    if isinstance(since, str):
        since = exchange.parse8601(since)
    # preload all markets from the exchange
    exchange.load_markets()
    # fetch all candles
    ohlcv = scrape_ohlcv(exchange, max_retries, symbol, timeframe, since, limit)
    # convert to DataFrame
    df = ohlcv_to_dataframe(ohlcv)
    
    print(f'Obtenidos {len(df)} registros desde {df.iloc[0]["datetime"]} hasta {df.iloc[-1]["datetime"]}')
    
    return df