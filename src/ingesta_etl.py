import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def obtener_datos_binance(symbol="BTCUSDT", interval="1m", limit=100):
    """
    Ingesta con redundancia de endpoints y Fallback automático para evitar
    bloqueos de IP en servidores cloud (AWS / Streamlit Cloud).
    """
    endpoints = [
        f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}",
        f"https://api1.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}",
        f"https://api2.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}",
        f"https://api3.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    ]
    
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    
    # 1. Intentar consultar Binance API
    for url in endpoints:
        try:
            response = requests.get(url, headers=headers, timeout=4)
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list) and len(data) >= 25:
                    df = pd.DataFrame(data, columns=[
                        'open_time', 'open', 'high', 'low', 'close', 'volume',
                        'close_time', 'qav', 'num_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore'
                    ])
                    df['fecha_hora'] = pd.to_datetime(df['open_time'], unit='ms')
                    df['close'] = df['close'].astype(float)
                    df['volume'] = df['volume'].astype(float)
                    df['es_simulado'] = False
                    return df[['fecha_hora', 'close', 'volume', 'es_simulado']]
        except Exception:
            continue

    # 2. FALLBACK RESILIENTE: Generación de serie temporal en vivo si la IP es bloqueada
    now = datetime.now()
    fechas = [now - timedelta(minutes=i) for i in range(limit - 1, -1, -1)]
    
    np.random.seed(int(now.timestamp()) % 1000)
    precio_base = 68500.0
    retornos = np.random.normal(0.0001, 0.0015, limit)
    precios = precio_base * np.exp(np.cumsum(retornos))
    volumenes = np.random.uniform(15.0, 85.0, limit)
    
    df_fallback = pd.DataFrame({
        'fecha_hora': fechas,
        'close': precios,
        'volume': volumenes,
        'es_simulado': True
    })
    return df_fallback

def calcular_indicadores_etl(df):
    """
    Proceso ETL: Cálculo de SMA_7, SMA_25 y VOL_15.
    """
    if df.empty or len(df) < 25:
        return pd.DataFrame()
        
    df = df.copy()
    df['SMA_7'] = df['close'].rolling(window=7).mean()
    df['SMA_25'] = df['close'].rolling(window=25).mean()
    df['retorno'] = df['close'].pct_change()
    df['VOL_15'] = df['retorno'].rolling(window=15).std() * 100
    df = df.dropna().reset_index(drop=True)
    return df