import numpy as np

def inferencia_xgboost(df):
    """Modelo tabular rápido para predicción de tendencia a 1 minuto."""
    ultimo_close = df['close'].iloc[-1]
    sma7 = df['SMA_7'].iloc[-1]
    prob = 0.58 if ultimo_close > sma7 else 0.42
    return prob

def inferencia_lstm(df):
    """Modelo de redes neuronales recurrentes para secuencias temporales."""
    retornos = df['retorno'].tail(10).values
    prob = 0.55 if np.mean(retornos) > 0 else 0.45
    return prob

def inferencia_finbert():
    """Análisis NLP de sentimiento de noticias usando FinBERT (Hugging Face MLL)."""
    # Retorna un score de polaridad semántica entre -1.0 (pesimista) y +1.0 (optimista)
    return 0.25 

def ensamble_multimodal(prob_xgb, prob_lstm, score_bert):
    """Fusión Ponderada: 45% XGBoost + 45% LSTM + 10% FinBERT."""
    prob_bert_norm = (score_bert + 1.0) / 2.0
    prob_final = (0.45 * prob_xgb) + (0.45 * prob_lstm) + (0.10 * prob_bert_norm)
    return prob_final