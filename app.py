import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor


# =========================
# 1) BUSINESS UNDERSTANDING (CRISP-DM)
# =========================
# Tujuan: memprediksi jumlah perceraian di kabupaten/kota Jawa Barat
# dan membandingkan performa 2 algoritma: MLP vs RandomForest.


# =========================
# 2) DATA UNDERSTANDING (CRISP-DM)
# =========================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_FILE = DATA_DIR / "Dataset Jumlah Perceraian Kabupaten Kota Jawa Barat.csv"

GEOJSON_FILE = DATA_DIR / "Kabupaten-Kota (Provinsi Jawa Barat).geojson"  # jika ada peta

TARGET_COL = "Jumlah"
YEAR_COL = "Tahun"
REGION_COL = "Kabupaten/Kota"


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE)
    return df


@st.cache_data
def load_geojson():
    if not GEOJSON_FILE.exists():
        return None
    with open(GEOJSON_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def detect_columns(df: pd.DataFrame):
    all_cols = df.columns.tolist()
    feature_cols = [c for c in all_cols if c != TARGET_COL]

    categorical_cols = [REGION_COL]
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]

    factor_cols = [c for c in numeric_cols if c != YEAR_COL]  # semua numeric selain tahun
    return feature_cols, categorical_cols, numeric_cols, factor_cols


# =========================
# 3) DATA PREPARATION (CRISP-DM)
# =========================
def make_preprocessor(categorical_cols, numeric_cols):
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", StandardScaler(), numeric_cols),
        ],
        remainder="drop"
    )
    return preprocessor


# =========================
# 4) MODELING (CRISP-DM)
# =========================
@st.cache_resource
def train_models(df: pd.DataFrame):
    """
    Train 2 model pipeline (MLPRegressor & RandomForestRegressor).
    Dibuat sebagai Pipeline agar preprocessor nempel di model dan aman di Streamlit Cloud.
    """
    feature_cols, categorical_cols, numeric_cols, _ = detect_columns(df)

    X = df[feature_cols].copy()
    y = df[TARGET_COL].astype(float).copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = make_preprocessor(categorical_cols, numeric_cols)

    # Model A: MLP (sklearn)
    mlp = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        max_iter=1000,
        random_state=42,
    )
    mlp_pipe = Pipeline([
        ("prep", preprocessor),
        ("model", mlp),
    ])

    # Model B: RandomForest
    rf = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )
    rf_pipe = Pipeline([
        ("prep", preprocessor),
        ("model", rf),
    ])

    # Fit
    mlp_pipe.fit(X_train, y_train)
    rf_pipe.fit(X_train, y_train)

    # Evaluate
    pred_mlp = mlp_pipe.predict(X_test)
    pred_rf = rf_pipe.predict(X_test)

    metrics = {
        "MLP": {
            "MAE": float(mean_absolute_error(y_test, pred_mlp)),
            "RMSE": float(mean_squared_error(y_test, pred_mlp, squared=False)),
        },
        "RandomForest": {
            "MAE": float(mean_absolute_error(y_test, pred_rf)),
            "RMSE": float(mean_squared_error(y_test, pred_rf, squared=False)),
        },
    }

    # return semua yang dibutuhkan app
    return {
        "feature_cols": feature_cols,
        "metrics": metrics,
        "mlp_pipe": mlp_pipe,
        "rf_pipe": rf_pipe,
        "test_set": (X_test, y_test, pred_mlp, pred_rf),
    }


# =========================
# 5) EVALUATION (CRISP-DM) + DEPLOYMENT UI
# =========================
st.set_page_config(page_title="Prediksi Perceraian Jawa Barat (MLP vs RF)", layout="wide")
st.title("📊 Prediksi Perceraian Provinsi Jawa Barat")
st.caption("Prediksi jumlah perceraian per kabupaten/kota di Provinsi Jawa Barat (MLP vs RandomForest)")

df = load_data()
geojson = load_geojson()

feature_cols, categorical_cols, numeric_cols, factor_cols = detect_columns(df)
bundle = train_models(df)

mlp_pipe = bundle["mlp_pipe"]
rf_pipe = bundle["rf_pipe"]
metrics = bundle["metrics"]

years = sorted(df[YEAR_COL].unique())
regions = sorted(df[REGION_COL].unique())

# Sidebar
st.sidebar.header("⚙️ Pengaturan")
selected_year = st.sidebar.selectbox("Pilih Tahun Analisis", options=years, index=len(years) - 1)
st.sidebar.markdown("---")
model_choice = st.sidebar.radio(
    "Model untuk Prediksi",
    options=["MLP", "RandomForest", "Rata-rata (MLP + RF)"],
    index=0
)

tab1, tab2, tab3, tab4 = st.tabs(
    ["📈 Eksplorasi", "🗺️ Peta", "🤖 Prediksi", "📊 Perbandingan Model"]
)

# -------------------------
# TAB 1: Eksplorasi
# -------------------------
with tab1:
    st.subheader(f"📈 Analisis Tahun {selected_year}")

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
        factor_df = factor_df.sort_values("Nilai", ascending=True)

        fig_factor = px.bar(
            factor_df,
            x="Nilai",
            y="Faktor",
            orientation="h",
            title=f"Total Faktor Penyebab Perceraian di Tahun {selected_year}",
            labels={"Nilai": "Total Nilai Faktor", "Faktor": "Faktor"},
        )
        fig_factor.update_layout(height=600, margin=dict(l=150, r=40, t=60, b=40))
        st.plotly_chart(fig_factor, use_container_width=True)
    else:
        st.warning("Tidak ada kolom faktor yang terdeteksi di dataset.")

# -------------------------
# TAB 2: Peta
# -------------------------
with tab2:
    st.subheader(f"🗺️ Peta Persebaran Perceraian Jawa Barat – {selected_year}")

    if geojson is None:
        st.info("GeoJSON belum ada. (Kalau mau peta, taruh file geojson di folder data/)")
    else:
        df_year = df[df[YEAR_COL] == selected_year].copy()

        fig_map = px.choropleth(
            df_year,
            geojson=geojson,
            locations=REGION_COL,
            featureidkey="properties.NAME_2",  # sesuaikan jika beda
            color=TARGET_COL,
            hover_name=REGION_COL,
            hover_data={YEAR_COL: True, TARGET_COL: True},
            title=f"Peta Sebaran Perceraian Jawa Barat ({selected_year})",
        )
        fig_map.update_geos(fitbounds="locations", visible=False)
        fig_map.update_layout(margin={"r": 0, "t": 50, "l": 0, "b": 0})
        st.plotly_chart(fig_map, use_container_width=True)

# -------------------------
# TAB 3: Prediksi
# -------------------------
with tab3:
    st.subheader("🤖 Prediksi Jumlah Perceraian")

    st.markdown(
        "Pilih **kabupaten/kota**, **tahun prediksi**, dan (opsional) atur nilai faktor. "
        "Aplikasi akan memberi prediksi berdasarkan model yang kamu pilih di sidebar."
    )

    with st.form("pred_form"):
        col1, col2 = st.columns(2)

        with col1:
            region_in = st.multiselect("Kabupaten/Kota", options=regions, default=[regions[0]] if regions else [])
            year_in = st.multiselect("Tahun Prediksi", options=list(range(int(min(years)), 2101)), default=[int(max(years)) + 1])

        with col2:
            st.markdown("##### Atur Nilai Faktor (opsional)")
            st.caption("Kalau tidak diubah, default = median dataset untuk masing-masing faktor.")
            factor_values = {}
            for fc in factor_cols[:8]:
                factor_values[fc] = st.number_input(fc, value=float(df[fc].median()))

            use_zero_for_unset = st.checkbox("Faktor lain yang tidak tampil → 0", value=False)

        submit = st.form_submit_button("🔮 Prediksi")

    if submit:
        if not region_in or not year_in:
            st.warning("Pilih minimal 1 kabupaten/kota dan 1 tahun.")
        else:
            rows = []
            for r in region_in:
                for y in year_in:
                    row = {REGION_COL: r, YEAR_COL: y}
                    for fc in factor_cols:
                        if fc in factor_values:
                            row[fc] = float(factor_values[fc])
                        else:
                            row[fc] = 0.0 if use_zero_for_unset else float(df[fc].median())
                    rows.append(row)

            input_df = pd.DataFrame(rows)[feature_cols]

            pred_mlp = mlp_pipe.predict(input_df)
            pred_rf = rf_pipe.predict(input_df)

            if model_choice == "MLP":
                pred_final = pred_mlp
            elif model_choice == "RandomForest":
                pred_final = pred_rf
            else:
                pred_final = (pred_mlp + pred_rf) / 2.0

            out = input_df[[REGION_COL, YEAR_COL]].copy()
            out["Prediksi (MLP)"] = pred_mlp.astype(float)
            out["Prediksi (RF)"] = pred_rf.astype(float)
            out["Prediksi (Dipakai)"] = pred_final.astype(float)

            st.dataframe(out, use_container_width=True)

# -------------------------
# TAB 4: Perbandingan Model
# -------------------------
with tab4:
    st.subheader("📊 Perbandingan Kinerja Model (Test Split 80/20)")

    met_df = pd.DataFrame([
        {"Model": "MLP", "MAE": metrics["MLP"]["MAE"], "RMSE": metrics["MLP"]["RMSE"]},
        {"Model": "RandomForest", "MAE": metrics["RandomForest"]["MAE"], "RMSE": metrics["RandomForest"]["RMSE"]},
    ])

    c1, c2 = st.columns(2)
    with c1:
        fig_mae = px.bar(met_df, x="Model", y="MAE", title="Perbandingan MAE (lebih kecil lebih baik)")
        st.plotly_chart(fig_mae, use_container_width=True)

    with c2:
        fig_rmse = px.bar(met_df, x="Model", y="RMSE", title="Perbandingan RMSE (lebih kecil lebih baik)")
        st.plotly_chart(fig_rmse, use_container_width=True)

    X_test, y_test, pred_mlp, pred_rf = bundle["test_set"]
    comp = pd.DataFrame({
        "Aktual": y_test.values,
        "Prediksi MLP": pred_mlp,
        "Prediksi RF": pred_rf,
    })

    st.markdown("#### 📌 Scatter Aktual vs Prediksi")
    sc1, sc2 = st.columns(2)
    with sc1:
        fig_sc_mlp = px.scatter(comp, x="Aktual", y="Prediksi MLP", title="Aktual vs Prediksi (MLP)")
        st.plotly_chart(fig_sc_mlp, use_container_width=True)

    with sc2:
        fig_sc_rf = px.scatter(comp, x="Aktual", y="Prediksi RF", title="Aktual vs Prediksi (RandomForest)")
        st.plotly_chart(fig_sc_rf, use_container_width=True)

    # Kesimpulan otomatis
    best_by_mae = met_df.sort_values("MAE").iloc[0]["Model"]
    best_by_rmse = met_df.sort_values("RMSE").iloc[0]["Model"]
    st.success(f"✅ Model terbaik berdasarkan MAE: **{best_by_mae}** | berdasarkan RMSE: **{best_by_rmse}**")
