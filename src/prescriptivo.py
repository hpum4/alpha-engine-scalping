import psycopg2

def calcular_criterio_kelly(prob_ganar, ratio_b=2.0, multiplicador=0.36):
    """Calcula la fracción óptima de capital a arriesgar (Kelly Fraccionario)."""
    p = prob_ganar
    q = 1.0 - p
    b = ratio_b
    
    f_star = (b * p - q) / b
    if f_star <= 0:
        return 0.0
    
    posicion_final = min(f_star * multiplicador, 0.10) # Límite máximo 10%
    return posicion_final

def registrar_operacion_supabase(config_db, datos_operacion):
    """Persistencia transaccional en PostgreSQL de Supabase con SSL obligatorio."""
    try:
        conn = psycopg2.connect(
            host=config_db['host'],
            database=config_db['database'],
            user=config_db['user'],
            password=config_db['password'],
            port=config_db['port'],
            sslmode='require',          # REQUERIDO POR SUPABASE CLOUD
            connect_timeout=10
        )
        cursor = conn.cursor()
        query = """
        INSERT INTO operaciones_bot 
        (precio_btc, sma_7, sma_25, volatilidad_15, prob_xgboost, prob_lstm, score_finbert, 
         prob_ensamble, posicion_kelly, monto_invertido, accion, take_profit, stop_loss)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
        """
        cursor.execute(query, datos_operacion)
        conn.commit()
        cursor.close()
        conn.close()
        return True, "Operación guardada exitosamente"
    except Exception as e:
        return False, str(e)