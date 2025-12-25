import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from tensorflow.keras.models import load_model
import plotly.express as px

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# =========================
# KONFIGURASI PATH
# =========================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"  # opsional (untuk tabel perbandingan)

DATA_FILE = DATA_DIR / "Dataset Jumlah Perceraian Kabupaten Kota Jawa Barat.csv"
GEOJSON_FILE = DATA_DIR / "Kabupaten-Kota (Provinsi Jawa Barat).geojson"

TARGET_COL = "Jumlah"
YEAR_COL = "Tahun"
REGION_COL = "Kabupaten/Kota"

MLP_MODEL_FILE = "model_mlp.h5"  # sesuai info kamu


# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_FILE)


@st.cache_data
def load_geojson():
    with open(GEOJSON_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================
# BUILD PREPROCESSOR (NO JOBLIB)
# =========================
@st.cache_resource
def build_preprocessor(df: pd.DataFrame):
    """
    FIX untuk Streamlit Cloud:
    - Preprocessor dibangun ulang dari dataset (tidak load joblib)
    - Menghindari error pickle/joblib pada Python 3.13
    """
    feature_cols = [c for c in df.columns if c != TARGET_COL]

    categorical_cols = [REGION_COL]
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", StandardScaler(), numeric_cols),
        ]
    )

    preprocessor.fit(df[feature_cols])
    return preprocessor, feature_cols, numeric_cols


# =========================
# LOAD MODEL MLP
# =========================
@st.cache_resource
def load_mlp_model():
    model_path = MODELS_DIR / MLP_MODEL_FILE
    return load_model(model_path, compile=False)


# =========================
# HELPER: Nama faktor lebih cantik
# =========================
def pretty_factor_name(col: str) -> str:
    name = col.replace("Fakor Perceraian - ", "").strip()
    if name == "Perselisihan dan Pertengkaran Terus Menerus":
        name = "Perselisihan / Pertengkaran"
    elif name == "Kekerasan Dalam Rumah Tangga":
        name = "KDRT"
    return name


# =========================
# UI: PAGE
# =========================
st.set_page_config(page_title="Prediksi Perceraian Provinsi Jawa Barat", layout="wide")
st.title("📊 Prediksi Perceraian Provinsi Jawa Barat")
st.caption("Prediksi jumlah perceraian per kabupaten/kota di Provinsi Jawa Barat")

# =========================
# INIT
# =========================
df = load_data()
preprocessor, feature_cols, numeric_cols = build_preprocessor(df)
mlp_model = load_mlp_model()

years = sorted(df[YEAR_COL].unique())
regions = sorted(df[REGION_COL].unique())

# faktor = semua numeric kecuali Tahun
factor_cols = [c for c in numeric_cols if c != YEAR_COL]

# =========================
# SIDEBAR
# =========================
st.sidebar.header("⚙️ Filter Global")
selected_year = st.sidebar.selectbox("Pilih Tahun Analisis", options=years, index=len(years) - 1)
st.sidebar.markdown("---")
st.sidebar.write("Filter ini mempengaruhi grafik di tab *Eksplorasi* dan *Peta*.")

# =========================
# TABS
# =========================
tab1, tab2, tab3, tab4 = st.tabs(
    ["📈 Eksplorasi Daerah & Faktor", "🗺️ Peta Jawa Barat", "🤖 Prediksi Jumlah Perceraian", "📑 Tabel Data"]
)

# =========================
# TAB 1: EKSPLORASI
# =========================
with tab1:
    st.subheader(f"📈 Analisis Tahun {selected_year}")

    st.markdown("#### 🔥 Daerah dengan Angka Perceraian Tertinggi")
    df_year = df[df[YEAR_COL] == selected_year].copy()
    df_year_sorted = df_year.sort_values(TARGET_COL, ascending=True)

    fig_region = px.bar(
        df_year_sorted,
        x=TARGET_COL,
        y=REGION_COL,
        orientation="h",
        title=f"Jumlah Perceraian per Kabupaten/Kota ({selected_year})",
        labels={REGION_COL: "Kabupaten/Kota", TARGET_COL: "Jumlah Perceraian"},
    )
    fig_region.update_layout(
        yaxis=dict(categoryorder="total ascending"),
        height=700,
        margin=dict(l=120, r=40, t=60, b=40),
    )
    st.plotly_chart(fig_region, use_container_width=True)

    if not df_year_sorted.empty:
        top_row = df_year_sorted.iloc[-1]
        st.info(
            f"📌 **Tertinggi**: {top_row[REGION_COL]} dengan **{int(top_row[TARGET_COL]):,} kasus** di {selected_year}."
        )

    st.markdown("---")
    st.markdown("#### 🧩 Faktor-faktor Tertinggi")

    if factor_cols:
        factor_sum = df_year[factor_cols].sum().sort_values(ascending=False)
        factor_df = factor_sum.reset_index()
        factor_df.columns = ["Faktor", "Nilai"]

        factor_df["Faktor"] = factor_df["Faktor"].astype(str).apply(pretty_factor_name)
        factor_df = factor_df.sort_values("Nilai", ascending=True)

        fig_factor = px.bar(
            factor_df,
            x="Nilai",
            y="Faktor",
            orientation="h",
            title=f"Kontribusi Faktor-faktor Perceraian di Tahun {selected_year}",
            labels={"Nilai": "Total Nilai Faktor", "Faktor": "Faktor"},
        )
        fig_factor.update_layout(
            yaxis=dict(categoryorder="total ascending"),
            height=600,
            margin=dict(l=150, r=40, t=60, b=40),
        )
        st.plotly_chart(fig_factor, use_container_width=True)

        if len(factor_df) > 0:
            top_factor = factor_df.iloc[-1]
            st.info(f"📌 **Faktor paling dominan** di {selected_year}: **{top_factor['Faktor']}**.")
    else:
        st.warning("Tidak ada kolom faktor yang terdeteksi di dataset.")

# =========================
# TAB 2: PETA
# =========================
with tab2:
    st.subheader(f"🗺️ Peta Persebaran Perceraian Jawa Barat – {selected_year}")

    try:
        geojson = load_geojson()
        df_year = df[df[YEAR_COL] == selected_year].copy()

        fig_map = px.choropleth(
            df_year,
            geojson=geojson,
            locations=REGION_COL,
            featureidkey="properties.NAME_2",
            color=TARGET_COL,
            color_continuous_scale="Reds",
            hover_name=REGION_COL,
            hover_data={YEAR_COL: True, TARGET_COL: True},
            labels={TARGET_COL: "Jumlah Perceraian"},
            title=f"Peta Sebaran Perceraian per Kabupaten/Kota di Jawa Barat ({selected_year})",
        )
        fig_map.update_geos(fitbounds="locations", visible=False)
        fig_map.update_layout(margin={"r": 0, "t": 50, "l": 0, "b": 0})
        st.plotly_chart(fig_map, use_container_width=True)

        if not df_year.empty:
            max_row = df_year.loc[df_year[TARGET_COL].idxmax()]
            min_row = df_year.loc[df_year[TARGET_COL].idxmin()]
            st.success(
                f"✅ **Ringkasan {selected_year}**  \n"
                f"- Tertinggi: **{max_row[REGION_COL]}** – {int(max_row[TARGET_COL]):,} kasus  \n"
                f"- Terendah: **{min_row[REGION_COL]}** – {int(min_row[TARGET_COL]):,} kasus"
            )
    except FileNotFoundError:
        st.error("File GeoJSON belum ditemukan. Pastikan ada di folder `data/` dan `featureidkey` sesuai.")

# =========================
# TAB 3: PREDIKSI (LIVE MLP)
# =========================
with tab3:
    st.subheader("🤖 Prediksi Jumlah Perceraian")

    st.markdown(
        "Pilih **kabupaten/kota**, **tahun prediksi**, dan **alasan perceraian**. "
        "Aplikasi akan menghitung prediksi menggunakan model **MLP**."
    )

    # (Perbandingan algoritma tetap ditampilkan dari file metrik)
    st.info("Model live di aplikasi: **MLP**. Perbandingan MLP vs RandomForest ditampilkan dari hasil evaluasi (CSV).")

    # Tampilkan perbandingan performa (kalau file ada)
    metrics_path = REPORTS_DIR / "metrics_mlp_vs_rf.csv"
    if metrics_path.exists():
        st.subheader("📊 Perbandingan Performa (Test 2024)")
        st.dataframe(pd.read_csv(metrics_path), use_container_width=True)

        mdf = pd.read_csv(metrics_path)
        if {"Model", "MAE"}.issubset(mdf.columns):
            fig_mae = px.bar(mdf, x="Model", y="MAE", title="Perbandingan MAE (lebih kecil lebih baik)")
            st.plotly_chart(fig_mae, use_container_width=True)
        if {"Model", "RMSE"}.issubset(mdf.columns):
            fig_rmse = px.bar(mdf, x="Model", y="RMSE", title="Perbandingan RMSE (lebih kecil lebih baik)")
            st.plotly_chart(fig_rmse, use_container_width=True)
    else:
        st.warning("File `reports/metrics_mlp_vs_rf.csv` tidak ditemukan. Upload folder `reports/` ke repo.")

    # mapping nama faktor cantik
    factor_display_map = {col: pretty_factor_name(col) for col in factor_cols}
    display_to_col = {v: k for k, v in factor_display_map.items()}
    factor_options_display = [factor_display_map[c] for c in factor_cols]

    with st.form("prediction_form"):
        col_left, col_right = st.columns([1.4, 1])

        with col_left:
            st.markdown("##### Input Kondisi yang Ingin Diprediksi")
            regions_input = st.multiselect("Pilih Kabupaten/Kota", options=regions, default=[regions[0]] if regions else [])

            min_year = int(df[YEAR_COL].min())
            default_year = int(df[YEAR_COL].max()) + 1

            years_input = st.multiselect("Pilih Tahun Prediksi", options=list(range(min_year, 2101)), default=[default_year])

            st.markdown("###### Alasan Perceraian")
            st.caption(
                "Faktor yang dipilih dianggap aktif dengan intensitas tipikal (median dataset), "
                "yang tidak dipilih dianggap 0."
            )
            selected_factor_labels = st.multiselect("Pilih Alasan Perceraian (bisa lebih dari satu)", options=factor_options_display)

        with col_right:
            st.markdown("##### Hasil Prediksi")
            submit = st.form_submit_button("🔮 Prediksi Jumlah Perceraian")

    if submit:
        if not regions_input or not years_input:
            st.warning("Pilih minimal satu kabupaten/kota dan minimal satu tahun.")
        else:
            selected_factor_cols = [display_to_col[lbl] for lbl in selected_factor_labels]

            rows = []
            for region in regions_input:
                for year in years_input:
                    row = {REGION_COL: region, YEAR_COL: year}
                    for col in factor_cols:
                        row[col] = float(df[col].median()) if col in selected_factor_cols else 0.0
                    rows.append(row)

            input_df = pd.DataFrame(rows)[feature_cols]

            X_p = preprocessor.transform(input_df)
            if hasattr(X_p, "toarray"):
                X_p = X_p.toarray()

            y_pred = mlp_model.predict(X_p).flatten()

            result_df = input_df[[REGION_COL, YEAR_COL]].copy()
            result_df["Prediksi Jumlah Perceraian"] = [float(v) for v in y_pred]

            st.dataframe(result_df, use_container_width=True)

# =========================
# TAB 4: TABEL DATA
# =========================
with tab4:
    st.subheader("📑 Tabel Data Perceraian")

    col_a, col_b = st.columns(2)
    with col_a:
        region_filter = st.selectbox("Filter Kabupaten/Kota (opsional)", options=["(Semua)"] + regions)
    with col_b:
        year_filter = st.selectbox("Filter Tahun (opsional)", options=["(Semua)"] + [str(y) for y in years])

    df_table = df.copy()
    if region_filter != "(Semua)":
        df_table = df_table[df_table[REGION_COL] == region_filter]
    if year_filter != "(Semua)":
        df_table = df_table[df_table[YEAR_COL] == int(year_filter)]

    df_display = df_table.sort_values([YEAR_COL, REGION_COL]).reset_index(drop=True)
    df_display.index = df_display.index + 1
    df_display.index.name = "No"
    st.dataframe(df_display, use_container_width=True)
