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

DATA_FILE = DATA_DIR / "Dataset Jumlah Perceraian Kabupaten Kota Jawa Barat.csv"
GEOJSON_FILE = DATA_DIR / "Kabupaten-Kota (Provinsi Jawa Barat).geojson"   # GANTI sesuai nama file peta kamu

TARGET_COL = "Jumlah"
YEAR_COL = "Tahun"
REGION_COL = "Kabupaten/Kota"


# ====== FUNGSI LOAD DATA & MODEL ======
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_FILE)
    return df


@st.cache_resource
def load_artifacts():
    preprocessor = joblib.load(MODELS_DIR / "preprocessor.joblib")
    model = load_model(MODELS_DIR / "model_perceraian.h5")
    return preprocessor, model


@st.cache_data
def load_geojson():
    with open(GEOJSON_FILE, "r", encoding="utf-8") as f:
        geojson = json.load(f)
    return geojson


# ====== KONFIGURASI HALAMAN ======
st.set_page_config(
    page_title="Prediksi Perceraian Provinsi Jawa Barat",
    layout="wide",
)

st.title("📊 Prediksi Perceraian Provinsi Jawa Barat")
st.caption("Prediksi jumlah perceraian per kabupaten/kota di Provinsi Jawa Barat")

df = load_data()
preprocessor, model = load_artifacts()

# ====== DEFINISI KOLOM & FAKTOR ======
all_cols = df.columns.tolist()
feature_cols = [c for c in all_cols if c != TARGET_COL]

# di training kamu: categorical = ["Kabupaten/Kota"], sisanya numeric
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
    [
        "📈 Eksplorasi Daerah & Faktor",
        "🗺️ Peta Jawa Barat",
        "🤖 Prediksi Jumlah Perceraian",
        "📑 Tabel Data",
    ]
)

# ====== TAB 1: GRAFIK DAERAH & FAKTOR ======
with tab1:
    st.subheader(f"📈 Analisis Tahun {selected_year}")

    col1, col2 = st.columns(2)

    # --- Grafik daerah dengan angka perceraian tertinggi ---
    with col1:
        st.markdown("#### 🔥 Daerah dengan Angka Perceraian Tertinggi")

        df_year = df[df[YEAR_COL] == selected_year].copy()
        df_year_sorted = df_year.sort_values(TARGET_COL, ascending=False)

        fig_region = px.bar(
            df_year_sorted,
            x=REGION_COL,
            y=TARGET_COL,
            title=f"Jumlah Perceraian per Kabupaten/Kota ({selected_year})",
            labels={REGION_COL: "Kabupaten/Kota", TARGET_COL: "Jumlah Perceraian"},
        )
        fig_region.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_region, use_container_width=True)

        if not df_year_sorted.empty:
            top_row = df_year_sorted.iloc[0]
            st.info(
                f"📌 **Tertinggi**: {top_row[REGION_COL]} "
                f"dengan **{int(top_row[TARGET_COL]):,} kasus** di {selected_year}."
            )

    # --- Grafik faktor tertinggi ---
    with col2:
        st.markdown("#### 🧩 Faktor-faktor Tertinggi")

        if factor_cols:
            factor_sum = df_year[factor_cols].sum().sort_values(ascending=False)
            factor_df = factor_sum.reset_index()
            factor_df.columns = ["Faktor", "Nilai"]

            fig_factor = px.bar(
                factor_df,
                x="Faktor",
                y="Nilai",
                title=f"Kontribusi Faktor-faktor Perceraian di Tahun {selected_year}",
                labels={"Nilai": "Total Nilai Faktor"},
            )
            fig_factor.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_factor, use_container_width=True)

            if len(factor_df) > 0:
                top_factor = factor_df.iloc[0]
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

        # PENTING: sesuaikan 'properties.NAMA_KAB' dengan nama field kab/kota di geojson kamu
        fig_map = px.choropleth(
            df_year,
            geojson=geojson,
            locations=REGION_COL,
            featureidkey="properties.NAMA_KAB",  # GANTI sesuai nama field di geojson kamu
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
            "File GeoJSON untuk peta belum ditemukan.\n\n"
            "Tambahkan file `jawa_barat_kabkota.geojson` ke folder `data/` "
            "dan sesuaikan nama field di `featureidkey`."
        )


# ====== TAB 3: PREDIKSI ======
with tab3:
    st.subheader("🤖 Prediksi Jumlah Perceraian")

    st.markdown(
        "Pilih kabupaten/kota, tahun, dan nilai faktor untuk melihat prediksi jumlah perceraian "
        "berdasarkan model deep learning yang sudah kamu latih."
    )

    col_left, col_right = st.columns([1.4, 1])

    with col_left:
        st.markdown("##### Input Kondisi yang Ingin Diprediksi")

        region_input = st.selectbox(
            "Pilih Kabupaten/Kota",
            options=regions,
        )

        default_year = int(df[YEAR_COL].max()) + 1
        year_input = st.number_input(
            "Pilih Tahun Prediksi",
            min_value=int(df[YEAR_COL].min()),
            max_value=2100,
            value=default_year,
            step=1,
        )

        st.markdown("###### Nilai Faktor-faktor Perceraian")
        st.caption("Default slider di-set ke **median** dari dataset.")

        factor_values = {}
        for col in factor_cols:
            col_min = float(df[col].min())
            col_max = float(df[col].max())
            col_med = float(df[col].median())

            factor_values[col] = st.slider(
                col,
                min_value=col_min,
                max_value=col_max,
                value=col_med,
            )

        pred_button = st.button("🔮 Prediksi Jumlah Perceraian")

    with col_right:
        st.markdown("##### Hasil Prediksi")

        if pred_button:
            # susun input sesuai feature_cols
            input_dict = {REGION_COL: region_input, YEAR_COL: year_input}
            input_dict.update(factor_values)

            input_df = pd.DataFrame([input_dict])[feature_cols]

            X_p = preprocessor.transform(input_df)
            if hasattr(X_p, "toarray"):
                X_p = X_p.toarray()

            y_pred = model.predict(X_p)
            y_pred_value = float(y_pred.flatten()[0])

            st.metric(
                "Perkiraan Jumlah Perceraian",
                f"{y_pred_value:,.0f} kasus",
            )
            st.caption(
                f"Prediksi untuk **{region_input}** pada tahun **{int(year_input)}** "
                "berdasarkan nilai faktor yang kamu atur di sebelah kiri."
            )
        else:
            st.info("Isi input di sebelah kiri lalu klik tombol **Prediksi** untuk melihat hasil.")


# ====== TAB 4: TABEL DATA ======
with tab4:
    st.subheader("📑 Tabel Data Perceraian")

    st.markdown(
        "Tabel ini berisi data perceraian per faktor per wilayah seperti dataset asli, "
        "tapi bisa difilter per kabupaten/kota dan tahun."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        region_filter = st.selectbox(
            "Filter Kabupaten/Kota (opsional)",
            options=["(Semua)"] + regions,
        )
    with col_b:
        year_filter = st.selectbox(
            "Filter Tahun (opsional)",
            options=["(Semua)"] + [str(y) for y in years],
        )

    df_table = df.copy()
    if region_filter != "(Semua)":
        df_table = df_table[df_table[REGION_COL] == region_filter]
    if year_filter != "(Semua)":
        df_table = df_table[df_table[YEAR_COL] == int(year_filter)]

    st.dataframe(
        df_table.sort_values([YEAR_COL, REGION_COL]),
        use_container_width=True,
    )

    st.caption("Kamu bisa scroll, sort kolom, dan download data dari menu di pojok kanan atas tabel.")
