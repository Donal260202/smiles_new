import streamlit as st
import pandas as pd
import numpy as np
import joblib
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import plotly.express as px
from io import BytesIO

# --- Streamlit Page Configuration ---
st.set_page_config(layout="wide")

# --- Configuration & Constants ---
DS_COL = ["mol_wt", "NumHdonors", "NumHAcceptors", "TPSA", "NumRotatableBonds", "MolLogP",
          "FpDensityMorgan1", "NumAromaticRings", "FractionCSP3", "NumAliphaticRings",
          "FpDensityMorgan2", "HeavyAtomMolWt"]

# --- Model Loading (Cached) ---
@st.cache_resource
def load_models_cached():
    return {
        "model": joblib.load("hybrid_model.pkl"),
        "absorption_model": joblib.load("hybrid_model_Absorption.pkl"),
        "metabolism_model": joblib.load("hybrid_model_Metabolism.pkl"),
    }

MODELS = load_models_cached()

# --- Helper Functions ---
def calculate_molecular_properties_series(smiles):
    try:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is not None:
            return pd.Series([
                Descriptors.MolWt(molecule), Descriptors.NumHDonors(molecule),
                Descriptors.NumHAcceptors(molecule), Descriptors.TPSA(molecule),
                rdMolDescriptors.CalcNumRotatableBonds(molecule), Descriptors.MolLogP(molecule),
                Descriptors.FpDensityMorgan1(molecule), Descriptors.NumAromaticRings(molecule),
                Descriptors.FractionCSP3(molecule), Descriptors.NumAliphaticRings(molecule),
                Descriptors.FpDensityMorgan2(molecule), Descriptors.HeavyAtomMolWt(molecule)
            ], index=DS_COL)
    except Exception:
        pass
    return pd.Series([np.nan] * len(DS_COL), index=DS_COL)


def perform_full_processing(df_input):
    df = df_input.copy()

    with st.spinner("🔬 Calculating molecular descriptors..."):
        descriptor_data = df["smiles"].apply(calculate_molecular_properties_series)
        for col in DS_COL:
            df[col] = descriptor_data[col]

        original_len = len(df)
        df.dropna(subset=DS_COL, inplace=True)
        if len(df) < original_len:
            st.warning(f"⚠️ {original_len - len(df)} molecules removed due to invalid SMILES.")

        if df.empty:
            st.warning("⚠️ No valid molecules found.")
            return None

    with st.spinner("🤖 Making predictions..."):
        X_pred = df[DS_COL].astype(float)

        df["Activity_Prediction"] = MODELS["model"].predict(X_pred)
        y_proba = MODELS["model"].predict_proba(X_pred)
        df["Probability"] = np.max(y_proba, axis=1)

        df["Absorption"] = MODELS["absorption_model"].predict(X_pred)
        df["Metabolism"] = MODELS["metabolism_model"].predict(X_pred)

    df["Concentration"] = df["Activity_Prediction"].apply(
        lambda x: "1 and 10 μM" if x == 2 else ("10 μM" if x == 1 else "Inactive")
    )

    df["Chances"] = df.apply(
        lambda row: 
        "Inactive" if row["Activity_Prediction"] == 0 else
        "High Chances" if row["Probability"] >= 0.75 else
        "Low Chances",
        axis=1
    )

    df["Metabolism_label"] = df["Metabolism"].apply(
        lambda x: "Metabolic" if x == 1 else "Non-Metabolic"
    )

    results = {"main_df": df}

    results["df_high"] = df[df["Chances"] == "High Chances"]
    results["df_active"] = df[df["Activity_Prediction"] != 0]

    # --- Preserve all input columns + predictions ---
    prediction_cols = [
        "Activity_Prediction", "Probability", "Concentration",
        "Chances", "Absorption", "Metabolism_label"
    ]

    original_input_cols = [col for col in df_input.columns if col != "smiles"]
    ordered_cols = ["smiles"] + original_input_cols + prediction_cols
    existing_display_cols = [col for col in ordered_cols if col in df.columns]
    results["df_display"] = df[existing_display_cols]

    st.toast("✅ Predictions complete!")
    return results


def download_excel_button(df_to_download, label, filename, key_suffix):
    if df_to_download is None or df_to_download.empty:
        st.button(label, disabled=True, key=f"download_{key_suffix}_disabled")
        return

    output = BytesIO()
    df_to_download.to_excel(output, index=False, sheet_name='Sheet1')
    st.download_button(
        label=label,
        data=output.getvalue(),
        file_name=filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key=f"download_{key_suffix}"
    )


# --- Streamlit App UI ---
st.title("🧪 SMILES-Based Activity, Absorption & Metabolism Predictor")
st.markdown("Upload an Excel file with **smiles** column.")

if "processed_file_id" not in st.session_state:
    st.session_state.processed_file_id = None
if "all_results" not in st.session_state:
    st.session_state.all_results = None

uploaded_file = st.file_uploader("📤 Upload Excel file", type=["xlsx"])
NEEDS_PROCESSING = False
current_file_id = None

if uploaded_file is not None:
    current_file_id = uploaded_file.file_id
    if current_file_id != st.session_state.processed_file_id:
        NEEDS_PROCESSING = True

if NEEDS_PROCESSING:
    try:
        raw_df = pd.read_excel(uploaded_file)

        if "smiles" not in raw_df.columns:
            st.error("❌ Column 'smiles' not found.")
            st.stop()

        results = perform_full_processing(raw_df)
        if results:
            st.session_state.all_results = results
            st.session_state.processed_file_id = current_file_id
            st.rerun()

    except Exception as e:
        st.error(f"❌ Error: {e}")
        st.stop()


# --- Display Section ---
if st.session_state.all_results:
    results = st.session_state.all_results
    df = results["main_df"]

    st.subheader("📊 Prediction Summary")
    summary = df["Activity_Prediction"].value_counts().rename_axis("Class").reset_index(name="Count")
    label_map = {0: "Inactive", 1: "10 μM", 2: "1 & 10 μM"}
    summary["Class"] = summary["Class"].map(label_map)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.table(summary)
    with col2:
        fig = px.pie(summary, names="Class", values="Count", title="Activity Distribution")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("🧾 Detailed Results (Top 20)")
    st.dataframe(results["df_display"].head(20))

    st.subheader("⬇️ Download Results")
    download_excel_button(results["df_display"], "📥 Download All Predictions", "predictions.xlsx", "all")
    download_excel_button(results["df_high"], "📥 High Chance Molecules", "high_chances.xlsx", "high")
    download_excel_button(results["df_active"], "📥 Active SMILES Only", "active_smiles.xlsx", "active")
