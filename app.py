import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from src.ingesta_etl import obtener_datos_binance, calcular_indicadores_etl
from src.modelos_ia import inferencia_xgboost, inferencia_lstm, inferencia_finbert, ensamble_multimodal
from src.prescriptivo import calcular_criterio_kelly, registrar_operacion_supabase

st.set_page_config(page_title="Alpha Engine - Scalping BA", layout="wide")

st.title("⚡ Alpha Engine: Analítica en Tiempo Real & Scalping Financiero")
st.markdown("**Proyecto de Grado - Fundamentos de Business Analytics (UNFV - FIIS)** | *Grupo N° 6*")

# Ingesta en tiempo real con Fallback Resiliente
with st.spinner("Conectando con motor de ingesta y calculando indicadores ETL..."):
    df_raw = obtener_datos_binance()
    df_etl = calcular_indicadores_etl(df_raw)

if df_etl.empty:
    st.error("Error crítico al procesar la ingesta de datos.")
    st.stop()

precio_actual = df_etl['close'].iloc[-1]
sma7 = df_etl['SMA_7'].iloc[-1]
sma25 = df_etl['SMA_25'].iloc[-1]
vol15 = df_etl['VOL_15'].iloc[-1]
es_sim = df_etl['es_simulado'].iloc[-1] if 'es_simulado' in df_etl.columns else False

# Tarjetas superiores de métricas
c1, c2, c3, c4 = st.columns(4)
c1.metric("Precio Bitcoin", f"\${precio_actual:,.2f} USD")
c2.metric("Media Móvil (SMA 7)", f"\${sma7:,.2f} USD")
c3.metric("Volatilidad (VOL 15)", f"{vol15:.4f}%")

if not es_sim:
    c4.metric("Estado de Ingesta", "ONLINE (Binance API)", delta="SLA < 100ms")
else:
    c4.metric("Estado de Ingesta", "SIMULADOR EN VIVO", delta="Fallback Anti-Bloqueo IP")

# Pestañas analíticas (Unidad 4 - Visualización)
tab1, tab2, tab3 = st.tabs(["📈 Panel Descriptivo (EDA)", "🧠 Panel Predictivo (IA)", "🎯 Panel Prescriptivo (Kelly)"])

with tab1:
    st.subheader("Análisis Exploratorio en Tiempo Real (Semana 4)")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df_etl['fecha_hora'], df_etl['close'], label="Precio BTC", color="#1f77b4")
    ax.plot(df_etl['fecha_hora'], df_etl['SMA_7'], label="SMA 7", color="#ff7f0e", linestyle="--")
    ax.plot(df_etl['fecha_hora'], df_etl['SMA_25'], label="SMA 25", color="#2ca02c", linestyle="--")
    ax.set_ylabel("Precio USD")
    ax.legend()
    st.pyplot(fig)

with tab2:
    st.subheader("Predicción del Ensamble Multimodal (Semanas 5-7)")
    p_xgb = inferencia_xgboost(df_etl)
    p_lstm = inferencia_lstm(df_etl)
    s_bert = inferencia_finbert()
    p_ensamble = ensamble_multimodal(p_xgb, p_lstm, s_bert)
    
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Probabilidad XGBoost", f"{p_xgb*100:.1f}%")
    col_b.metric("Probabilidad LSTM", f"{p_lstm*100:.1f}%")
    col_c.metric("Score FinBERT NLP", f"{s_bert:+.2f}")
    col_d.metric("Confianza Ensamble", f"{p_ensamble*100:.2f}%")

with tab3:
    st.subheader("Motor Prescriptivo & Criterio de Kelly (Semanas 9-10)")
    posicion_kelly = calcular_criterio_kelly(p_ensamble)
    capital_base = 10000.0
    monto_orden = capital_base * posicion_kelly
    
    st.write(f"**Fracción Asignada por Kelly:** `{posicion_kelly*100:.2f}%` de la cuenta")
    st.write(f"**Monto Invertido:** `${monto_orden:,.2f} USD` (Capital Base: \${capital_base:,.2f} USD)")
    
    if p_ensamble > 0.515 and posicion_kelly > 0:
        accion = "COMPRA"
        tp = precio_actual * 1.03
        sl = precio_actual * 0.995
        st.success(f"SEÑAL GENERADA: {accion} | Take-Profit: \\({tp:,.2f} | Stop-Loss: \\){sl:,.2f}")
    else:
        accion = "NEUTRAL"
        tp, sl = 0.0, 0.0
        st.info("SEÑAL NEUTRAL: Confianza de modelo insuficiente. Posición protegida.")

    # Botón para registrar en la base de datos de Supabase
    if "postgres" in st.secrets:
        if st.button("💾 Guardar Operación en Supabase PostgreSQL"):
            datos = (float(precio_actual), float(sma7), float(sma25), float(vol15),
                     float(p_xgb), float(p_lstm), float(s_bert), float(p_ensamble),
                     float(posicion_kelly), float(monto_orden), accion, float(tp), float(sl))
            exito, mensaje = registrar_operacion_supabase(st.secrets["postgres"], datos)
            if exito:
                st.success("¡Operación guardada exitosamente en la base de datos de Supabase!")
            else:
                st.error(f"Error de conexión con Supabase: {mensaje}")