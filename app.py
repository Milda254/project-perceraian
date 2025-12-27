import os
import json
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.express as px
import tensorflow as tf

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Perceraian Jabar - Final", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
FACTOR_MODELS_DIR = os.path.join(BASE_DIR, "factor_models")

CSV_PATH = os.path.join(DATA_DIR, "Dataset Jumlah Perceraian Kabupaten Kota Jawa Barat.csv")
GEOJSON_PATH = os.path.join(DATA_DIR, "Jawa Barat.geojson")

MLP_PATH = os.path.join(MODELS_DIR, "model_mlp.h5")
RF_PATH  = os.path.join(MODELS_DIR, "model_rf.joblib")

YEAR_COL = "Tahun"
CAT_COL = "Kabupaten/Kota"
TARGET_COL = "Jumlah"

FACTOR_COLS = [
    "Fakor Perceraian - Zina",
    "Fakor Perceraian - Mabuk",
    "Fakor Perceraian - Madat",
    "Fakor Perceraian - Judi",
    "Fakor Perceraian - Meninggalkan Salah satu Pihak",
    "Fakor Perceraian - Dihukum Penjara",
    "Fakor Perceraian - Poligami",
    "Fakor Perceraian - Kekerasan Dalam Rumah Tangga",
    "Fakor Perceraian - Cacat Badan",
    "Fakor Perceraian - Perselisihan dan Pertengkaran Terus Menerus",
    "Fakor Perceraian - Kawin Paksa",
    "Fakor Perceraian - Murtad",
    "Fakor Perceraian - Ekonomi",
    "Fakor Perceraian - Lain-lain",
]

FEATURE_COLS = [YEAR_COL, CAT_COL] + FACTOR_COLS

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_dataset():
    df = pd.read_csv(CSV_PATH)
    return df

@st.cache_resource
def load_models():
    mlp = tf.keras.models.load_model(MLP_PATH)
    rf = joblib.load(RF_PATH)
    return mlp, rf

@st.cache_resource
def build_preprocessor(df):
    categorical_cols = [CAT_COL]
    numeric_cols = [c for c in FEATURE_COLS if c not in categorical_cols]

    train_df = df[df[YEAR_COL] <= 2022].copy()
    X_train = train_df[FEATURE_COLS]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )
    preprocessor.fit(X_train)
    return preprocessor

@st.cache_resource
def load_factor_models():
    cols_path = os.path.join(FACTOR_MODELS_DIR, "factor_feature_columns.joblib")
    if not os.path.exists(cols_path):
        return None, None

    feature_cols = joblib.load(cols_path)
    models = {}

    for f in FACTOR_COLS:
        safe = f.replace("/", "_").replace(" ", "_")
        path = os.path.join(FACTOR_MODELS_DIR, f"{safe}.joblib")
        if not os.path.exists(path):
            return None, None
        models[f] = joblib.load(path)

    return models, feature_cols

@st.cache_data
def load_geojson():
    if not os.path.exists(GEOJSON_PATH):
        return None
    with open(GEOJSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

# =========================
# HELPERS
# =========================
def predict_factors(models, feature_cols, kabkota, tahun):
    X = pd.get_dummies(
        pd.DataFrame([{YEAR_COL: tahun, CAT_COL: kabkota}]),
        columns=[CAT_COL]
    )
    X = X.reindex(columns=feature_cols, fill_value=0)

    out = {}
    for f, m in models.items():
        out[f] = max(0.0, float(m.predict(X)[0]))
    return out

def predict_jumlah(preprocessor, model_name, mlp, rf, input_df):
    Xp = preprocessor.transform(input_df)
    if hasattr(Xp, "toarray"):
        Xp = Xp.toarray()

    if model_name == "MLP":
        pred = mlp.predict(Xp, verbose=0).flatten()[0]
    else:
        pred = rf.predict(Xp)[0]
    return max(0.0, float(pred))

# =========================
# APP START
# =========================
df = load_dataset()
mlp, rf = load_models()
preprocessor = build_preprocessor(df)
geojson = load_geojson()
factor_models, factor_feature_cols = load_factor_models()

kab_list = sorted(df[CAT_COL].unique().tolist())
year_list = sorted(df[YEAR_COL].unique().tolist())

st.sidebar.title("Menu")
page = st.sidebar.radio(
    "Pilih Halaman",
    ["Dashboard", "Prediksi Masa Depan (2 Tahap)", "Prediksi Manual (1 Tahap)", "Peta"]
)

# =========================
# DASHBOARD
# =========================
if page == "Dashboard":
    st.title("Dashboard Perceraian Jawa Barat")

    year = st.selectbox("Tahun", year_list, index=len(year_list)-1)
    kab = st.selectbox("Kabupaten/Kota", kab_list)

    d = df[(df[YEAR_COL] == year) & (df[CAT_COL] == kab)]
    st.dataframe(d, use_container_width=True)

    trend = df[df[CAT_COL] == kab].sort_values(YEAR_COL)
    fig = px.line(trend, x=YEAR_COL, y=TARGET_COL, markers=True)
    st.plotly_chart(fig, use_container_width=True)

# =========================
# PREDIKSI MASA DEPAN (2 TAHAP)
# =========================
elif page == "Prediksi Masa Depan (2 Tahap)":
    st.title("Prediksi Masa Depan (2 Tahap)")

    if factor_models is None:
        st.error("Folder factor_models tidak lengkap.")
        st.stop()

    kab = st.selectbox("Kabupaten/Kota", kab_list)
    tahun = st.number_input("Tahun Prediksi", min_value=2019, max_value=2100, value=2026)
    model2 = st.selectbox("Model Jumlah", ["MLP", "RandomForest"], index=0)

    if st.button("Prediksi"):
        factors = predict_factors(factor_models, factor_feature_cols, kab, tahun)
        df_f = pd.DataFrame(list(factors.items()), columns=["Faktor", "Prediksi"])
        st.subheader("Prediksi Faktor")
        st.dataframe(df_f, use_container_width=True)

        input_row = {YEAR_COL: tahun, CAT_COL: kab}
        input_row.update(factors)
        input_df = pd.DataFrame([input_row])

        jumlah = predict_jumlah(preprocessor, model2, mlp, rf, input_df)
        st.success(f"Prediksi Jumlah Perceraian: **{jumlah:,.0f}** kasus")

# =========================
# PREDIKSI MANUAL
# =========================
elif page == "Prediksi Manual (1 Tahap)":
    st.title("Prediksi Manual")

    model2 = st.selectbox("Model", ["MLP", "RandomForest"], index=0)
    kab = st.selectbox("Kabupaten/Kota", kab_list)
    tahun = st.number_input("Tahun", min_value=2019, max_value=2100, value=2024)

    values = {}
    for f in FACTOR_COLS:
        values[f] = st.number_input(f, min_value=0, value=0)

    input_row = {YEAR_COL: tahun, CAT_COL: kab}
    input_row.update(values)
    input_df = pd.DataFrame([input_row])

    if st.button("Prediksi"):
        jumlah = predict_jumlah(preprocessor, model2, mlp, rf, input_df)
        st.success(f"Prediksi Jumlah Perceraian: **{jumlah:,.0f}** kasus")

# =========================
# PETA (AGREGAT PROVINSI)
# =========================
else:
    st.title("Peta Provinsi Jawa Barat")

    if geojson is None:
        st.info("GeoJSON tidak ditemukan.")
        st.stop()

    year = st.selectbox("Tahun", year_list, index=len(year_list)-1)
    total = df[df[YEAR_COL] == year][TARGET_COL].sum()

    map_df = pd.DataFrame([{"id": 0, "Total": total}])
    geojson["features"][0]["properties"]["_id"] = 0

    fig = px.choropleth(
        map_df,
        geojson=geojson,
        locations="id",
        featureidkey="properties._id",
        color="Total",
        title=f"Total Perceraian Jawa Barat Tahun {year}"
    )
    fig.update_geos(fitbounds="locations", visible=False)
    st.plotly_chart(fig, use_container_width=True)
