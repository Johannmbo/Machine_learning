import json
import os
 
import pandas as pd
import requests
import streamlit as st
 
# --- Config ---
API_URL = os.environ.get("API_URL", "http://localhost:8500").rstrip("/")
DEFAULT_DATA = os.environ.get("DEFAULT_DATA", "../data/diabetes.csv")
FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]
TARGET = "Outcome"
 
st.set_page_config(page_title="Diabetes Prediction", layout="wide")
st.title("Diabetes Prediction App")
 
# -- 1. Vérification de la disponibilité de l'API ---
try:
    health = requests.get(f"{API_URL}/health", timeout=5)
    health.raise_for_status()
    st.success(f"API connectée ({API_URL})")
except requests.RequestException as e:
    st.error(f"Impossible de joindre l'API sur {API_URL} : {e}")
    st.stop()
 
# --- 2. Chargement des données ---
uploaded = st.file_uploader("Charge un fichier CSV de patients", type="csv")
 
if uploaded is not None:
    df = pd.read_csv(uploaded)
elif os.path.exists(DEFAULT_DATA):
    df = pd.read_csv(DEFAULT_DATA)
    st.info(f"Aucun fichier chargé : utilisation de {DEFAULT_DATA}")
else:
    st.info("Charge un CSV pour commencer.")
    st.stop()
 
st.subheader("Aperçu des données")
st.dataframe(df.head())
 
# --- 3. Validation ---
missing = [c for c in FEATURES if c not in df.columns]
if missing:
    st.error(f"Colonnes manquantes : {', '.join(missing)}")
    st.stop()
 
y = df[TARGET] if TARGET in df.columns else None
X = df[FEATURES]
 
# --- 4. Prédictions ---
if st.button("Launch Predictions", type="primary"):
    payload = json.loads(X.to_json(orient="records"))  # types Python natifs
    try:
        with st.spinner(f"Prédiction de {len(X)} patients..."):
            r = requests.post(f"{API_URL}/predict/batch", json=payload, timeout=60)
        r.raise_for_status()
    except requests.RequestException as e:
        st.error(f"La prédiction a échoué : {e}")
        st.stop()
 
    preds = pd.DataFrame(r.json())
    results = X.copy()
    results["prediction"] = preds["prediction"].values
    results["probability"] = preds["probability"].values
    results["Diagnostic"] = results["prediction"].map({1: "Diabétique", 0: "Non diabétique"})
    if y is not None:
        results[TARGET] = y.values
    st.session_state["results"] = results  # survit aux relances du script
 
# --- 5. Affichage ---
if "results" not in st.session_state:
    st.stop()
 
results = st.session_state["results"]
n = len(results)
n_risk = int(results["prediction"].sum())
 
c1, c2, c3 = st.columns(3)
c1.metric("Patients", n)
c2.metric("Diabétiques prédits", n_risk)
c3.metric("Taux de risque", f"{n_risk / n:.1%}")
 
st.subheader("Résultats")
only_risk = st.checkbox("Afficher uniquement les patients à risque")
view = results[results["prediction"] == 1] if only_risk else results
st.dataframe(
    view,
    column_config={
        "probability": st.column_config.ProgressColumn(
            "Probabilité", min_value=0.0, max_value=1.0, format="%.2f"
        )
    },
    width="stretch",
)
 
st.subheader("Graphiques")
g1, g2 = st.columns(2)
with g1:
    st.caption("Répartition des diagnostics")
    st.bar_chart(results["Diagnostic"].value_counts())
with g2:
    st.caption("Probabilité de diabète selon le glucose")
    st.scatter_chart(results, x="Glucose", y="probability", color="Diagnostic")
 
# --- 6. Comparaison avec la vérité terrain ---
if TARGET in results.columns:
    st.subheader("Comparaison avec les vraies valeurs (Outcome)")
    accuracy = (results["prediction"] == results[TARGET]).mean()
    st.metric("Accuracy", f"{accuracy:.1%}")
    confusion = pd.crosstab(
        results[TARGET].map({1: "Réel : diabétique", 0: "Réel : non diabétique"}),
        results["Diagnostic"].map(lambda d: f"Prédit : {d.lower()}"),
    )
    st.dataframe(confusion)
 
# --- 7. Export ---
st.download_button(
    "Télécharger les résultats (CSV)",
    results.to_csv(index=False).encode("utf-8"),
    file_name="predictions.csv",
    mime="text/csv",
)
 