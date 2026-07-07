
import joblib
# ... (código inicial de streamlit)

# Cargar el modelo pre-entrenado de forma instantánea
modelo_ia = joblib.load('best_model.pkl')
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

# Configuración de la página web
st.set_page_config(page_title="BioAI - Ablación Tumoral", layout="wide")

st.title("🔬 Plataforma Predictiva de Biotransporte: Daño Tisular mediante IA")
st.markdown("""
Esta aplicación web interactiva funciona como un **Modelo Subrogado de Inteligencia Artificial**. 
Utiliza un regresor polinomial avanzado entrenado con las soluciones numéricas de la ecuación de Bioheat extraídas de COMSOL.
""")

# --- BASE DE DATOS HISTÓRICA COMSOL ---
tiempos = np.array([0, 0.01, 0.02, 0.04, 0.08, 0.12, 0.2, 0.28, 0.44, 0.6, 0.92, 1.24, 1.88, 2.52, 3.52, 4.52, 5.52, 6.52, 7.52, 8.52, 9.52, 10.52])
dano_4mm = np.array([3.52e-7, 1.78e-4, 3.57e-4, 7.18e-4, 0.0014, 0.0022, 0.0038, 0.0056, 0.0096, 0.0141, 0.0256, 0.0400, 0.0799, 0.1311, 0.2318, 0.3418, 0.4497, 0.5479, 0.6329, 0.7041, 0.7626, 0.8101])
dano_12mm = np.array([3.52e-7, 1.77e-4, 3.56e-4, 7.15e-4, 0.0014, 0.0022, 0.0037, 0.0054, 0.0091, 0.0133, 0.0241, 0.0377, 0.0770, 0.1284, 0.2299, 0.3395, 0.4457, 0.5416, 0.6245, 0.6942, 0.7519, 0.7992])
dano_20mm = np.array([3.52e-7, 1.77e-4, 3.55e-4, 7.11e-4, 0.0014, 0.0021, 0.0036, 0.0052, 0.0086, 0.0123, 0.0211, 0.0313, 0.0574, 0.0884, 0.1449, 0.2049, 0.2650, 0.3232, 0.3783, 0.4298, 0.4775, 0.5216])

# Reestructuración matemática para entrenamiento multivariable
T_all = np.concatenate([tiempos, tiempos, tiempos])
Dist_all = np.concatenate([np.full_like(tiempos, 4), np.full_like(tiempos, 12), np.full_like(tiempos, 20)])
Dano_all = np.concatenate([dano_4mm, dano_12mm, dano_20mm])

X = np.column_stack((T_all, Dist_all))
y = Dano_all

# --- ENTRENAMIENTO DEL MODELO SOFISTICADO CONTINUO ---
# Usamos un pipeline polinomial de grado 3 para evitar el efecto escalón y asegurar suavidad física
modelo_ia = make_pipeline(PolynomialFeatures(degree=3), LinearRegression())
modelo_ia.fit(X, y)

# --- INTERFAZ DE USUARIO EN VIVO (SIDEBAR) ---
st.sidebar.header("🕹️ Parámetros de Predicción en Vivo")
st.sidebar.markdown("Modifica las condiciones físicas para evaluar la respuesta de la IA de forma instantánea.")

tiempo_input = st.sidebar.slider("Tiempo de tratamiento (minificados):", 0.0, 10.52, 5.0, step=0.1)
distancia_input = st.sidebar.slider("Distancia analizada desde el electrodo (mm):", 4.0, 20.0, 8.0, step=0.5)

# Predicción puntual en vivo
prediccion_viva = modelo_ia.predict([[tiempo_input, distancia_input]])[0]
prediccion_viva = np.clip(prediccion_viva, 0.0, 1.0) # Asegurar límites físicos de 0 a 1

# Mostrar resultados numéricos destacados
col1, col2 = st.columns(2)
with col1:
    st.metric(label="📍 Fracción de Daño Tisular Predicho", value=f"{prediccion_viva:.4f} ({prediccion_viva*100:.2f}%)")
with col2:
    status = "🔴 Necrosis Crítica (>70%)" if prediccion_viva >= 0.7 else "🟡 Lesión Parcial / Tejido Viable"
    st.metric(label="⚠️ Estado Celular Estimado", value=status)

# --- GENERACIÓN DE GRÁFICAS DE ANÁLISIS AVANZADO ---
st.subheader("📊 Análisis Gráfico de Curvas de Daño Continuas")

tiempos_continuos = np.linspace(0, 10.52, 200)

fig, ax = plt.subplots(figsize=(10, 5))

# Graficar datos experimentales/COMSOL históricos
ax.scatter(tiempos, dano_4mm, color='blue', alpha=0.6, label='COMSOL Histórico (4 mm)')
ax.scatter(tiempos, dano_12mm, color='orange', alpha=0.6, label='COMSOL Histórico (12 mm)')
ax.scatter(tiempos, dano_20mm, color='green', alpha=0.6, label='COMSOL Histórico (20 mm)')

# Predicciones dinámicas de la IA de forma suave
X_dinamico = np.column_stack((tiempos_continuos, np.full_like(tiempos_continuos, distancia_input)))
pred_dinamica = np.clip(modelo_ia.predict(X_dinamico), 0.0, 1.0)

ax.plot(tiempos_continuos, pred_dinamica, color='red', linestyle='--', linewidth=2.5,
        label=f'Predicción IA Continua (Ajustada a {distancia_input} mm)')

# Marcador del punto exacto seleccionado por el usuario
ax.plot(tiempo_input, prediccion_viva, marker='X', color='black', markersuffix='', metrics='', markersize=12, label='Punto en Vivo Seleccionado')

ax.set_title('Evolución del Daño Tisular: Simulación Física vs. Regresión Matemática de IA', fontsize=12)
ax.set_xlabel('Tiempo de Exposición (min)', fontsize=10)
ax.set_ylabel('Fracción de Daño Celular (0 a 1)', fontsize=10)
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend(loc='upper left')

# Renderizar gráfico en el dashboard web
st.pyplot(fig)

st.markdown("""
---
### 🛠️ Diagrama de Flujo del Proceso
`COMSOL Multiphysics (Datos Numéricos)` ➡️ `Extracción de Sondas de Dominio` ➡️ `Pipeline Polinomial en Python` ➡️ `Despliegue en Streamlit Web App`
""")
