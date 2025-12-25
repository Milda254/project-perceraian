import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import joblib
from tensorflow.keras.models import load_model

import plotly.express as px

# ====== KONFIGURASI PATH ======
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"  # opsional

DATA_FILE = DATA_DIR / "Dataset Jumlah Perceraian Kabupaten Kota Jawa Barat.csv"
GEOJSON_FILE = DATA_DIR / "Kabupaten-Kota (Provinsi Jawa Barat).geojson"

TARGET_COL = "Jumlah"
YEAR_COL = "Tahun"
REGION_COL = "Kabupaten/Kota"

# Nama file model (sesuai info kamu)
MLP_MODEL_FILE = "model_mlp.h5"
RF_MODEL_FILE = "model_rf.joblib"
PREPROCESSOR_FILE = "preprocessor.joblib"


# ====== FUNGSI LOAD DATA & ARTIFACT ======
@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_FILE)


@st.cache_resource
def load_artifacts():
    """
    Load artifact HASIL TRAINING (disarankan):
    - preprocessor.joblib (fit pada TRAIN saat training, bukan fit ulang di app)
    - model_mlp.h5
    - model_rf.joblib
    """
    preprocessor = joblib.load(MODELS_DIR / PREPROCESSOR_FILE)
    mlp_model = load_model(MODELS_DIR / MLP_MODEL_FILE, compile=False)
    rf_model = joblib.load(MODELS_DIR / RF_MODEL_FILE)
    return preprocessor, mlp_model, rf_model


@st.cache_data
def load_geojson():
    with open(GEOJSON_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def pretty_factor_name(col: str) -> str:
    name = col.replace("Fakor Perceraian - ", "").strip()
    if name == "Perselisihan dan Pertengkaran Terus Menerus":
        name = "Perselisihan / Pertengkaran"
    elif name == "Kekerasan Dalam Rumah Tangga":
        name = "KDRT"
    return name


def safe_int(x):
    try:
        return int(x)
    except Exception:
        return x


# ====== KONFIGURASI HALAMAN ======
st.set_page_config(page_title="Prediksi Perceraian Provinsi Jawa Barat", layout="wide")

st.title("📊 Prediksi Perceraian Provinsi Jawa Barat")
st.caption("Prediksi jumlah perceraian per kabupaten/kota di Provinsi Jawa Barat")

# ====== LOAD DATA & MODEL ======
df = load_data()
preprocessor, mlp_model, rf_model = load_artifacts()

# ====== DEFINISI KOLOM & FAKTOR ======
all_cols = df.columns.tolist()
feature_cols = [c for c in all_cols if c != TARGET_COL]

categorical_cols = [REGION_COL]
numeric_cols = [c for c in feature_cols if c not in categorical_cols]

# faktor = semua numeric kecuali Tahun
factor_cols = [c for c in numeric_cols if c != YEAR_COL]

years = sorted(df[YEAR_COL].unique())
regions = sorted(df[REGION_COL].unique())

# ====== SIDEBAR FILTER GLOBAL ======
st.sidebar.header("⚙️ Filter Global")

selected_year = st.sidebar.selectbox(
    "Pilih Tahun Analisis",
    options=years,
    index=len(years) - 1,  # default: tahun terakhir
)

st.sidebar.markdown("---")
st.sidebar.write("Filter ini mempengaruhi grafik di tab *Eksplorasi* dan *Peta*.")

# ====== TAB ======
tab1, tab2, tab3, tab4 = st.tabs(
    ["📈 Eksplorasi Daerah & Faktor", "🗺️ Peta Jawa Barat", "🤖 Prediksi Jumlah Perceraian", "📑 Tabel Data"]
)

# ====== TAB 1: GRAFIK DAERAH & FAKTOR ======
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
            f"📌 **Tertinggi**: {top_row[REGION_COL]} "
            f"dengan **{int(top_row[TARGET_COL]):,} kasus** di {selected_year}."
        )

    st.markdown("---")
    st.markdown("#### 🧩 Faktor-faktor Tertinggi")

    if factor_cols:
        factor_sum = df_year[factor_cols].sum().sort_values(ascending=False)
        factor_df = factor_sum.reset_index()
        factor_df.columns = ["Faktor", "Nilai"]

        factor_df["Faktor"] = (
            factor_df["Faktor"].astype(str).str.replace("Fakor Perceraian - ", "", regex=False).str.strip()
        )

        short_map = {
            "Perselisihan dan Pertengkaran Terus Menerus": "Perselisihan / Pertengkaran",
            "Kekerasan Dalam Rumah Tangga": "KDRT",
        }
        factor_df["Faktor"] = factor_df["Faktor"].replace(short_map)

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
            st.info(
                f"📌 **Faktor paling dominan** di {selected_year}: "
                f"**{top_factor['Faktor']}** (total {top_factor['Nilai']:.0f})."
            )
    else:
        st.warning("Tidak ada kolom faktor yang terdeteksi di dataset.")

# ====== TAB 2: PETA JAWA BARAT ======
with tab2:
    st.subheader(f"🗺️ Peta Persebaran Perceraian Jawa Barat – {selected_year}")

    try:
        geojson = load_geojson()
        df_year = df[df[YEAR_COL] == selected_year].copy()

        fig_map = px.choropleth(
            df_year,
            geojson=geojson,
            locations=REGION_COL,
            featureidkey="properties.NAME_2",  # sesuaikan kalau field geojson berbeda
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
        st.error(
            "File GeoJSON belum ditemukan.\n\n"
            "Tambahkan file GeoJSON ke folder `data/` dan sesuaikan `featureidkey`."
        )

# ====== TAB 3: PREDIKSI ======
with tab3:
    st.subheader("🤖 Prediksi Jumlah Perceraian")

    st.markdown(
        "Pilih **kabupaten/kota**, **tahun prediksi**, dan **alasan perceraian** "
        "yang ingin dianalisis. Model akan membuat prediksi untuk semua kombinasi "
        "kabupaten × tahun yang kamu pilih."
    )

    # Pilih algoritma (tambahan dari versi kamu)
    model_name = st.selectbox("Pilih Algoritma", ["RandomForest", "MLP"])

    # Mapping nama faktor cantik
    factor_display_map = {col: pretty_factor_name(col) for col in factor_cols}
    display_to_col = {v: k for k, v in factor_display_map.items()}
    factor_options_display = [factor_display_map[c] for c in factor_cols]

    with st.form("prediction_form"):
        col_left, col_right = st.columns([1.4, 1])

        with col_left:
            st.markdown("##### Input Kondisi yang Ingin Diprediksi")

            regions_input = st.multiselect(
                "Pilih Kabupaten/Kota",
                options=regions,
                default=[regions[0]] if len(regions) > 0 else [],
            )

            min_year = int(df[YEAR_COL].min())
            default_year = int(df[YEAR_COL].max()) + 1

            years_input = st.multiselect(
                "Pilih Tahun Prediksi",
                options=list(range(min_year, 2101)),
                default=[default_year],
            )

            st.markdown("###### Alasan Perceraian")
            st.caption(
                "Pilih satu atau beberapa alasan perceraian. "
                "Alasan yang dipilih akan dianggap **aktif** dengan intensitas tipikal, "
                "sedangkan yang tidak dipilih dianggap **tidak terjadi (0)**."
            )

            selected_factor_labels = st.multiselect(
                "Pilih Alasan Perceraian (bisa lebih dari satu)",
                options=factor_options_display,
            )

        with col_right:
            st.markdown("##### Hasil Prediksi")
            st.caption("Klik tombol di bawah untuk menghitung prediksi berdasarkan input di sebelah kiri.")
            submit = st.form_submit_button("🔮 Prediksi Jumlah Perceraian")

    if submit:
        if not regions_input or not years_input:
            st.warning("Pilih **minimal satu** kabupaten/kota dan **minimal satu** tahun terlebih dahulu.")
        else:
            selected_factor_cols = [display_to_col[lbl] for lbl in selected_factor_labels]

            rows = []
            for region in regions_input:
                for year in years_input:
                    row = {REGION_COL: region, YEAR_COL: year}

                    # Jika faktor dipilih -> median dataset, jika tidak -> 0
                    for col in factor_cols:
                        row[col] = float(df[col].median()) if col in selected_factor_cols else 0.0

                    rows.append(row)

            input_df = pd.DataFrame(rows)[feature_cols]

            # Transform pakai preprocessor hasil training
            X_p = preprocessor.transform(input_df)
            if hasattr(X_p, "toarray"):
                X_p = X_p.toarray()

            # Prediksi sesuai model terpilih
            if model_name == "MLP":
                y_pred = mlp_model.predict(X_p).flatten()
            else:
                y_pred = rf_model.predict(X_p).flatten()

            result_df = input_df[[REGION_COL, YEAR_COL]].copy()
            result_df["Prediksi Jumlah Perceraian"] = [float(v) for v in y_pred]

            st.dataframe(result_df, use_container_width=True)

            st.caption(
                "Catatan: faktor yang dipilih di-set ke **nilai median** dari dataset, "
                "sedangkan faktor yang tidak dipilih di-set ke **0**."
            )

            # (Opsional) Tampilkan metrik perbandingan jika file ada
            metrics_path = REPORTS_DIR / "metrics_mlp_vs_rf.csv"
            if metrics_path.exists():
                st.subheader("📊 Perbandingan Performa (Test 2024)")
                st.dataframe(pd.read_csv(metrics_path), use_container_width=True)
    else:
        st.info("Atur input di form, lalu klik tombol **Prediksi Jumlah Perceraian** untuk melihat hasil.")

# ====== TAB 4: TABEL DATA ======
with tab4:
    st.subheader("📑 Tabel Data Perceraian")

    st.markdown(
        "Tabel ini berisi data perceraian per faktor per wilayah seperti dataset asli, "
        "tapi bisa difilter per kabupaten/kota dan tahun."
    )

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

    st.caption("Kamu bisa scroll, sort kolom, dan download data dari menu di pojok kanan atas tabel.")
