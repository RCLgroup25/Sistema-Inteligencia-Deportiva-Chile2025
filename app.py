import streamlit as st
import pandas as pd
import numpy as np
import unicodedata
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------
# CONFIGURACIÓN
# ---------------------------------------------------

st.set_page_config(
    page_title="Sistema Inteligencia Deportiva Fútbol Chileno 2025",
    layout="wide"
)

# 🎨 ESTILO CLARO Y MODERNO
st.markdown("""
<style>

/* Fondo general */
.main {
    background-color: #F4F6FB;
}

/* Título principal */
h1 {
    color: #1E2A38;
    font-weight: 800;
}

/* Subtítulos */
h2, h3 {
    color: #2C3E50;
}

/* Tarjetas métricas */
.stMetric {
    background-color: #FFFFFF;
    padding: 20px;
    border-radius: 18px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.06);
    text-align: center;
}

/* Botón principal */
div.stButton > button {
    background-color: #3B82F6;
    color: white;
    border-radius: 12px;
    border: none;
    padding: 10px 20px;
    font-weight: 600;
}

div.stButton > button:hover {
    background-color: #2563EB;
}

/* Dataframe */
.stDataFrame {
    border-radius: 12px;
    background-color: white;
}

</style>
""", unsafe_allow_html=True)

col_logo, col_title = st.columns([1, 6])

with col_logo:
    st.image("Liga de primera.png", width=150)

with col_title:
    st.title("⚽ Scouting System Intelligence Chile 2025")
st.caption("📊 Sistema interactivo que analiza métricas avanzadas para estimar fair value, identificar ineficiencias de mercado y detectar talento exportable.")

# ---------------------------------------------------
# FUNCIONES
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

# -----------------------
# DEFINIR FEATURES MODELO
# -----------------------

no_metricas = [
    'player_name', 'team_name', 'competition_name',
    'unique_tournament_name', 'player_short_name',
    'player_slug', 'position', 'Valor', 'Edad',
    'unique_tournament_id', 'season_id',
    'team_id', 'player_id', 'shirt_number',
    'Equipo', 'name_clean'
]

metricas_cols = [
    c for c in df.columns
    if c not in no_metricas
    and np.issubdtype(df[c].dtype, np.number)
]

df_model = df.copy()

df_model[metricas_cols] = df_model[metricas_cols].apply(pd.to_numeric, errors="coerce")
df_model[metricas_cols] = df_model[metricas_cols].fillna(0)

if len(df_model) == 0:
    st.error("No hay datos disponibles en el modelo.")
    st.stop()

scaler = StandardScaler()
scaled_features = scaler.fit_transform(df_model[metricas_cols])

# ---------------------------------------------------
# FUNCIÓN PRINCIPAL
# ---------------------------------------------------

def analizar_jugador(nombre_input):

    nombre_input = limpiar_texto(nombre_input)

    coincidencias = df_model[
        df_model["player_name_clean"].str.contains(nombre_input, na=False)
    ]

    if coincidencias.empty:
        return None

    idx = coincidencias.index[0]
    jugador_foco = df_model.loc[idx]

    # Filtrar comparables
    df_pares = df_model[
        (df_model["position"] == jugador_foco["position"]) &
        (df_model["Edad"].between(jugador_foco["Edad"] - 3, jugador_foco["Edad"] + 3)) &
        (df_model["minutes_played_total"] >= 350)
    ].copy()

    if idx not in df_pares.index:
        return None

    indices = df_pares.index.tolist()
    idx_relativo = indices.index(idx)

    posiciones = [df_model.index.get_loc(i) for i in indices]
    scaled_pares = scaled_features[posiciones]

    similitudes = cosine_similarity(
        [scaled_pares[idx_relativo]],
        scaled_pares
    )[0]

    df_pares["Similitud_%"] = similitudes * 100

    gemelos = (
        df_pares
        .sort_values("Similitud_%", ascending=False)
        .iloc[1:6]
        .copy()
    )

    gemelos_validos = gemelos[gemelos["Valor"] > 0]

    if len(gemelos_validos) > 0:

        pesos = (
            gemelos_validos["Similitud_%"] *
            gemelos_validos["minutes_played_total"]
        )

        fair_value = np.average(
            gemelos_validos["Valor"],
            weights=pesos
        )

        diferencia = fair_value - jugador_foco["Valor"]

    else:
        fair_value = None
        diferencia = None

    return jugador_foco, fair_value, diferencia, gemelos


# ---------------------------------------------------
# INTERFAZ
# ---------------------------------------------------

st.divider()

st.markdown("## 🔎 Buscar jugador")

nombre = st.text_input(
    "",
    placeholder="Ej: Cecilio Waterman"
)

if nombre:
    if st.button("🚀 Analizar jugador"):

        resultado = analizar_jugador(nombre)

        if resultado is None:
            st.error("Jugador no encontrado o no cumple filtros.")
        else:
            jugador, fv, diff, gemelos = resultado

            st.markdown(f"## 👤 {jugador['player_name']}")

            if jugador["Edad"] <= 24:
                st.info("🧒 Perfil joven con proyección internacional")

            col1, col2, col3 = st.columns(3)

            col1.metric("🎂 Edad", f"{int(jugador['Edad'])} años")
            col2.metric("💰 Valor actual", f"€ {jugador['Valor']:,.0f}")

            if fv is not None:
                col3.metric("📈 Fair Value estimado", f"€ {fv:,.0f}")

                edad = jugador["Edad"]

                st.markdown("### 📌 Diagnóstico de mercado")

                # 🔹 JÓVENES → SOLO EXPORTACIÓN
                if edad <= 24:

                    if diff is not None and diff > 0:
                        st.markdown("### ✈️🔥 POTENCIAL DE EXPORTACIÓN (Proyección + margen económico)")
                    else:
                        st.markdown("### ✈️🧠 POTENCIAL DE EXPORTACIÓN (Perfil joven con proyección internacional)")

                # 🔹 25+
                else:

                    if diff > 150000:
                        st.markdown("### 🟢💎 INFRAVALORADO (Oportunidad de mercado)")

                    elif diff < -150000:
                        st.markdown("### 🔴⚠️ SOBREVALORADO (Costo elevado vs rendimiento)")

                    else:
                        st.markdown("### 🟡📊 PRECIO JUSTO (Acorde a sus pares)")

            st.divider()

            st.markdown("## 🤝 Gemelos estadísticos IA")

            tabla = gemelos[[
                "player_name",
                "team_name",
                "Valor",
                "Similitud_%",
                "Edad"
            ]].copy()

            tabla["Valor"] = tabla["Valor"].apply(lambda x: f"€ {x:,.0f}")
            tabla["Similitud_%"] = tabla["Similitud_%"].round(1)


            st.dataframe(tabla, use_container_width=True)


