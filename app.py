import os
import json
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.express as px
import tensorflow as tf

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

PREP_PATH = os.path.join(MODELS_DIR, "preprocessor.joblib")
MLP_PATH  = os.path.join(MODELS_DIR, "model_mlp.h5")
RF_PATH   = os.path.join(MODELS_DIR, "model_rf.joblib")

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
# LOADERS
# =========================
@st.cache_data
def load_dataset():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError("CSV dataset tidak ditemukan di folder /data")
    df = pd.read_csv(CSV_PATH)

    missing = [c for c in (FEATURE_COLS + [TARGET_COL]) if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom berikut tidak ada di dataset: {missing}")
    return df

@st.cache_resource
def load_stage2_models():
    preprocessor = joblib.load(PREP_PATH)
    mlp = tf.keras.models.load_model(MLP_PATH)
    rf = joblib.load(RF_PATH)
    return preprocessor, mlp, rf

@st.cache_resource
def load_stage1_factor_models():
    """
    Load 13 model faktor + list kolom dummy.
    Jika folder/berkas tidak lengkap, return (None, None, None).
    """
    cols_path = os.path.join(FACTOR_MODELS_DIR, "factor_feature_columns.joblib")
    if not os.path.exists(cols_path):
        return None, None, None

    feature_cols = joblib.load(cols_path)

    models = {}
    for fcol in FACTOR_COLS:
        safe = fcol.replace("/", "_").replace(" ", "_")
        model_path = os.path.join(FACTOR_MODELS_DIR, f"{safe}.joblib")
        if not os.path.exists(model_path):
            return None, None, None
        models[fcol] = joblib.load(model_path)

    return models, feature_cols, cols_path

@st.cache_data
def load_geojson():
    if not os.path.exists(GEOJSON_PATH):
        return None
    with open(GEOJSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================
# HELPERS
# =========================
def predict_stage1_factors(factor_models, factor_feature_cols, kabkota: str, tahun: int) -> dict:
    """
    Input: kabkota, tahun
    Output: dict 13 faktor prediksi (>=0)
    """
    X_new = pd.get_dummies(pd.DataFrame([{YEAR_COL: int(tahun), CAT_COL: kabkota}]), columns=[CAT_COL])
    X_new = X_new.reindex(columns=factor_feature_cols, fill_value=0)

    out = {}
    for fcol, model in factor_models.items():
        val = float(model.predict(X_new)[0])
        out[fcol] = max(0.0, val)
    return out

def predict_stage2_jumlah(preprocessor, model_name: str, mlp, rf, input_df: pd.DataFrame) -> float:
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
preprocessor, mlp, rf = load_stage2_models()
geojson = load_geojson()
factor_models, factor_feature_cols, _ = load_stage1_factor_models()

kab_list = sorted(df[CAT_COL].dropna().unique().tolist())
year_list = sorted(df[YEAR_COL].dropna().unique().tolist())

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

    colA, colB = st.columns(2)
    with colA:
        year_view = st.selectbox("Pilih Tahun (data historis)", year_list, index=len(year_list)-1)
    with colB:
        kab_view = st.selectbox("Pilih Kabupaten/Kota", kab_list)

    st.subheader("Tabel Data (filter)")
    d = df[(df[YEAR_COL] == year_view) & (df[CAT_COL] == kab_view)].copy()
    st.dataframe(d, use_container_width=True)

    st.subheader("Tren Jumlah Perceraian per Tahun (Kab/Kota terpilih)")
    d_trend = df[df[CAT_COL] == kab_view].sort_values(YEAR_COL)
    fig_trend = px.line(d_trend, x=YEAR_COL, y=TARGET_COL, markers=True)
    st.plotly_chart(fig_trend, use_container_width=True)

    st.subheader("Komposisi Faktor (Kab/Kota & Tahun terpilih)")
    if d.empty:
        st.warning("Data untuk kab/kota & tahun ini tidak ditemukan.")
    else:
        factor_vals = d.iloc[0][FACTOR_COLS].sort_values(ascending=False)
        fig_bar = px.bar(
            factor_vals.reset_index(),
            x="index",
            y=factor_vals.name,
            labels={"index": "Faktor", factor_vals.name: "Jumlah"},
        )
        fig_bar.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Top 10 Wilayah dengan Jumlah Perceraian Terbesar (Tahun terpilih)")
    top10 = (
        df[df[YEAR_COL] == year_view]
        .sort_values(TARGET_COL, ascending=False)
        [[CAT_COL, TARGET_COL]]
        .head(10)
    )
    fig_top = px.bar(top10, x=CAT_COL, y=TARGET_COL)
    fig_top.update_layout(xaxis_tickangle=-30)
    st.plotly_chart(fig_top, use_container_width=True)


# =========================
# PREDIKSI MASA DEPAN (2 TAHAP): Tahun+KabKota -> 13 faktor -> Jumlah
# =========================
elif page == "Prediksi Masa Depan (2 Tahap)":
    st.title("Prediksi Masa Depan (2 Tahap)")
    st.caption("Tahap 1: Prediksi 13 faktor dari (Tahun + Kab/Kota). Tahap 2: Prediksi Jumlah dari (Tahun + Kab/Kota + 13 faktor).")

    if factor_models is None:
        st.error(
            "Folder factor_models belum lengkap.\n\n"
            "Pastikan kamu sudah menyimpan:\n"
            "- factor_models/factor_feature_columns.joblib\n"
            "- 14 file model faktor (.joblib) sesuai nama kolom faktor\n"
        )
        st.stop()

    col1, col2, col3 = st.columns(3)
    with col1:
        kabkota = st.selectbox("Kabupaten/Kota", kab_list)
    with col2:
        tahun = st.number_input("Tahun prediksi", min_value=2019, max_value=2100, value=2026, step=1)
    with col3:
        model_stage2 = st.selectbox("Model Tahap 2 (Jumlah)", ["MLP", "RandomForest"], index=0)

    if st.button("Jalankan Prediksi 2 Tahap"):
        # Tahap 1
        factors_pred = predict_stage1_factors(factor_models, factor_feature_cols, kabkota, int(tahun))
        factors_df = pd.DataFrame([{"Faktor": k, "Prediksi": v} for k, v in factors_pred.items()]).sort_values("Prediksi", ascending=False)

        st.subheader("Hasil Prediksi 13 Faktor (Tahap 1)")
        st.dataframe(factors_df, use_container_width=True)

        fig_f = px.bar(factors_df, x="Faktor", y="Prediksi", title="Prediksi Faktor (Tahap 1)")
        fig_f.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_f, use_container_width=True)

        # Bangun input untuk Tahap 2
        input_row = {YEAR_COL: int(tahun), CAT_COL: kabkota}
        input_row.update({k: float(v) for k, v in factors_pred.items()})
        input_df = pd.DataFrame([input_row])

        # Tahap 2
        jumlah_pred = predict_stage2_jumlah(preprocessor, model_stage2, mlp, rf, input_df)

        st.subheader("Prediksi Jumlah Perceraian (Tahap 2)")
        st.success(f"Prediksi Jumlah Perceraian Tahun {int(tahun)} di **{kabkota}**: **{jumlah_pred:,.0f}** kasus")

        with st.expander("Lihat input lengkap yang dikirim ke Model Tahap 2"):
            st.dataframe(input_df, use_container_width=True)


# =========================
# PREDIKSI MANUAL (1 TAHAP): Tahun+KabKota+13 faktor -> Jumlah
# (berguna untuk demo & validasi terhadap data historis)
# =========================
elif page == "Prediksi Manual (1 Tahap)":
    st.title("Prediksi Manual (1 Tahap)")
    st.caption("Mode ini untuk input manual 13 faktor (sesuai data historis). Cocok untuk validasi dan demo perbandingan MLP vs RF.")

    colA, colB = st.columns([1, 2])
    with colA:
        model_stage2 = st.selectbox("Model (Jumlah)", ["MLP", "RandomForest"], index=0)
        tahun = st.number_input("Tahun", min_value=2019, max_value=2100, value=2024, step=1)
        kabkota = st.selectbox("Kabupaten/Kota", kab_list)

    with colB:
        st.subheader("Input Faktor")
        c1, c2 = st.columns(2)
        values = {}
        first_half = FACTOR_COLS[:7]
        second_half = FACTOR_COLS[7:]

        with c1:
            for f in first_half:
                values[f] = st.number_input(f, min_value=0, value=0, step=1)
        with c2:
            for f in second_half:
                values[f] = st.number_input(f, min_value=0, value=0, step=1)

    input_row = {YEAR_COL: int(tahun), CAT_COL: kabkota}
    input_row.update({k: int(v) for k, v in values.items()})
    input_df = pd.DataFrame([input_row])

    if st.button("Prediksi (1 Tahap)"):
        pred = predict_stage2_jumlah(preprocessor, model_stage2, mlp, rf, input_df)
        st.success(f"Prediksi Jumlah Perceraian: **{pred:,.0f}** kasus")

        # Jika user memasukkan tahun yang ada di data historis, tampilkan aktual (kalau tersedia)
        actual = df[(df[YEAR_COL] == int(tahun)) & (df[CAT_COL] == kabkota)]
        if not actual.empty:
            actual_val = float(actual.iloc[0][TARGET_COL])
            st.info(f"Nilai aktual di dataset (jika ada): **{actual_val:,.0f}** kasus")


# =========================
# PETA
# GeoJSON kamu hanya 1 polygon (Provinsi Jawa Barat), jadi peta ini agregat provinsi per tahun.
# =========================
else:
    st.title("Peta (Agregat Provinsi Jawa Barat)")
    st.caption("GeoJSON yang digunakan berisi batas Provinsi Jawa Barat (1 polygon), sehingga peta ditampilkan sebagai agregat total per tahun.")

    if geojson is None:
        st.info("File geojson tidak ditemukan di /data.")
        st.stop()

    year_map = st.selectbox("Pilih Tahun", year_list, index=len(year_list)-1)
    total = float(df[df[YEAR_COL] == year_map][TARGET_COL].sum())

    # Buat data 1 baris untuk 1 polygon
    map_df = pd.DataFrame([{"id": 0, "Total_Jumlah": total}])

    # Geojson hanya 1 feature -> kita pakai feature index as id=0
    # Plotly butuh locations match featureidkey. Kita set featureidkey ke 'id' buatan:
    # Cara paling stabil: sisipkan 'id' ke properties secara runtime.
    g = geojson.copy()
    try:
        if "features" in g and len(g["features"]) > 0:
            g["features"][0].setdefault("properties", {})
            g["features"][0]["properties"]["_id"] = 0
    except Exception:
        pass

    fig_map = px.choropleth(
        map_df,
        geojson=g,
        locations="id",
        featureidkey="properties._id",
        color="Total_Jumlah",
        labels={"Total_Jumlah": "Total Jumlah"},
        title=f"Total Perceraian Jawa Barat Tahun {year_map}"
    )
    fig_map.update_geos(fitbounds="locations", visible=False)
    st.plotly_chart(fig_map, use_container_width=True)

    st.metric("Total Jumlah (Agregat Provinsi)", f"{total:,.0f}")
    st.info("Jika kamu punya GeoJSON batas kab/kota, ganti file geojson agar peta bisa per wilayah.")
