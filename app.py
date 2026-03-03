import streamlit as st
import pandas as pd
import numpy as np
import unicodedata
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------
# CONFIGURACIÓN Y ESTILO
# ---------------------------------------------------
st.set_page_config(
    page_title="Sistema Inteligencia Deportiva Fútbol Chileno 2025",
    layout="wide"
)

st.markdown("""
<style>
    .main { background-color: #F4F6FB; }
    h1 { color: #1E2A38; font-weight: 800; }
    .stMetric {
        background-color: #FFFFFF;
        padding: 20px;
        border-radius: 18px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.06);
    }
    div.stButton > button {
        background-color: #3B82F6; color: white; border-radius: 12px;
        width: 100%; font-weight: 600; height: 3em;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# FUNCIONES DE DATOS
# ---------------------------------------------------
def limpiar_texto(texto):
    texto = str(texto).lower()
    texto = unicodedata.normalize('NFKD', texto).encode('ascii', 'ignore').decode('utf-8')
    return texto.strip()

@st.cache_data
def cargar_datos():
    df = pd.read_csv("df_merge.csv")
    df["player_name_clean"] = df["player_name"].apply(limpiar_texto)
    return df

df = cargar_datos()

no_metricas = [
    'player_name', 'team_name', 'competition_name', 'unique_tournament_name', 
    'player_short_name', 'player_slug', 'position', 'Valor', 'Edad',
    'unique_tournament_id', 'season_id', 'team_id', 'player_id', 'shirt_number',
    'Equipo', 'name_clean', 'player_name_clean'
]

metricas_cols = [c for c in df.columns if c not in no_metricas and np.issubdtype(df[c].dtype, np.number)]
df_model = df.copy()
df_model[metricas_cols] = df_model[metricas_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

scaler = StandardScaler()
scaled_features = scaler.fit_transform(df_model[metricas_cols])

# ---------------------------------------------------
# LÓGICA DE ANÁLISIS
# ---------------------------------------------------
def analizar_jugador(nombre_input):
    nombre_input = limpiar_texto(nombre_input)
    coincidencias = df_model[df_model["player_name_clean"].str.contains(nombre_input, na=False)]
    
    if coincidencias.empty: return None

    idx = coincidencias.index[0]
    jugador_foco = df_model.loc[idx]

    df_pares = df_model[
        (df_model["position"] == jugador_foco["position"]) &
        (df_model["Edad"].between(jugador_foco["Edad"] - 3, jugador_foco["Edad"] + 3)) &
        (df_model["minutes_played_total"] >= 350)
    ].copy()

    if idx not in df_pares.index: return None

    indices = df_pares.index.tolist()
    idx_relativo = indices.index(idx)
    posiciones_esc = [df_model.index.get_loc(i) for i in indices]
    
    # 1. Cálculo de similitud base (-1 a 1)
    sim_raw = cosine_similarity([scaled_features[df_model.index.get_loc(idx)]], scaled_features[posiciones_esc])[0]
    
    # 2. TRANSFORMACIÓN: Escalar de [-1, 1] a [0, 100]
    # Esto asegura que "opuesto" sea 0% y "idéntico" sea 100%
    df_pares["Similitud_%"] = ((sim_raw + 1) / 2) * 100

    gemelos = df_pares.sort_values("Similitud_%", ascending=False).iloc[1:6].copy()
    gemelos_validos = gemelos[gemelos["Valor"] > 0]

    if not gemelos_validos.empty:
        # Ahora los pesos siempre son positivos, lo que hace el Fair Value estable
        pesos = gemelos_validos["Similitud_%"] * gemelos_validos["minutes_played_total"]
        fair_value = np.average(gemelos_validos["Valor"], weights=pesos)
        diferencia = fair_value - jugador_foco["Valor"]
    else:
        fair_value, diferencia = None, None

    return jugador_foco, fair_value, diferencia, gemelos, df_pares

# ---------------------------------------------------
# INTERFAZ DE USUARIO (Mismo diseño)
# ---------------------------------------------------
col_logo, col_title = st.columns([1, 6])
with col_logo: st.image("Liga de primera.png", width=150)
with col_title:
    st.title("⚽ Scouting System Intelligence Chile 2025")
    st.caption("📊 Análisis avanzado de Fair Value y Gemelos Estadísticos.")

st.divider()
nombre = st.text_input("🔎 Buscar jugador", placeholder="Ej: Cecilio Waterman")

if nombre:
    res = analizar_jugador(nombre)
    if res:
        jugador, fv, diff, gemelos, df_pares = res
        
        st.markdown(f"## 👤 {jugador['player_name']}")
        col1, col2, col3 = st.columns(3)
        col1.metric("🎂 Edad", f"{int(jugador['Edad'])} años")
        col2.metric("💰 Valor Actual", f"€ {jugador['Valor']:,.0f}")
        if fv:
            col3.metric("📈 Fair Value Estimado", f"€ {fv:,.0f}", delta=f"{diff:,.0f}")

        st.markdown("### 📌 Diagnóstico de mercado")
        if jugador["Edad"] <= 24:
            st.info("🧒 **POTENCIAL DE EXPORTACIÓN:** Perfil joven con proyección internacional.")
        elif diff > 150000:
            st.success("🟢💎 **INFRAVALORADO:** Oportunidad de mercado clara.")
        elif diff < -150000:
            st.error("🔴⚠️ **SOBREVALORADO:** Costo elevado respecto a su rendimiento comparado.")
        else:
            st.warning("🟡📊 **PRECIO JUSTO:** Acorde a sus pares estadísticos.")

        st.divider()
        st.markdown("## 📍 Posicionamiento en el Mercado")
        
        df_pares['Etiqueta'] = df_pares.apply(lambda x: 'JUGADOR ANALIZADO' if x['player_name'] == jugador['player_name'] else 'Pares Comparables', axis=1)
        
        fig = px.scatter(
            df_pares, x="Similitud_%", y="Valor",
            color="Etiqueta", size="minutes_played_total",
            hover_name="player_name", text="player_name",
            color_discrete_map={'JUGADOR ANALIZADO': '#EF4444', 'Pares Comparables': '#3B82F6'},
            labels={"Similitud_%": "Similitud Estadística (%)", "Valor": "Valor de Mercado (€)"},
            template="plotly_white", height=600
        )
        
        fig.add_hline(y=df_pares["Valor"].mean(), line_dash="dash", line_color="gray", annotation_text="Valor Medio")
        fig.add_vline(x=df_pares["Similitud_%"].mean(), line_dash="dash", line_color="gray", annotation_text="Similitud Media")
        
        fig.update_traces(textposition='top center')
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("## 🤝 Gemelos estadísticos IA")
        tabla = gemelos[["player_name", "team_name", "Valor", "Similitud_%", "Edad"]].copy()
        tabla["Similitud_%"] = tabla["Similitud_%"].round(1)
        
        st.dataframe(
            tabla, 
            column_config={
                "Valor": st.column_config.NumberColumn("Valor", format="€ %,d"),
                "Similitud_%": st.column_config.ProgressColumn("Similitud", min_value=0, max_value=100, format="%.1f%%")
            },
            hide_index=True, use_container_width=True
        )
    else:
        st.error("Jugador no encontrado o datos insuficientes.")
