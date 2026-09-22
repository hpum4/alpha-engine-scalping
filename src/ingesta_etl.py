import requests
import pandas as pd
import numpy as np

def obtener_datos_binance(symbol="BTCUSDT", interval="1m", limit=100):
    """
    Ingesta de precios en tiempo real de Bitcoin desde la API pública de Binance.
    """
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    response = requests.get(url, timeout=10)
    data = response.json()
    
    df = pd.DataFrame(data, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'qav', 'num_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore'
    ])
    
    df['fecha_hora'] = pd.to_datetime(df['open_time'], unit='ms')
    df['close'] = df['close'].astype(float)
    df['volume'] = df['volume'].astype(float)
    return df[['fecha_hora', 'close', 'volume']]

def calcular_indicadores_etl(df):
    """
    Proceso ETL: Cálculo de medias móviles (SMA_7, SMA_25) y volatilidad (VOL_15).
    """
    df = df.copy()
    df['SMA_7'] = df['close'].rolling(window=7).mean()
    df['SMA_25'] = df['close'].rolling(window=25).mean()
    df['retorno'] = df['close'].pct_change()
    df['VOL_15'] = df['retorno'].rolling(window=15).std() * 100
    df = df.dropna().reset_index(drop=True)
    return df