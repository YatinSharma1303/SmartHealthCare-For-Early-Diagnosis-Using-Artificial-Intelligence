import ast
import pickle
import re
from datetime import datetime as _dt
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from thefuzz import process

from theme_config import init_theme, get_theme_styles, render_theme_toggle


APP_TITLE = "Early Diagnosis AI"
CREATOR_NAME = "Yatin Sharma"
DATA_DIR = Path("data/Disease-Prediction-and-Medical dataset")
MODEL_PATH = Path("models/first_feature_models/RandomForest.pkl")

MODEL_DISEASE_MAP = {
    15: "Fungal infection", 4: "Allergy", 16: "GERD", 9: "Chronic cholestasis",
    14: "Drug Reaction", 33: "Peptic ulcer disease", 1: "AIDS", 12: "Diabetes",
    17: "Gastroenteritis", 6: "Bronchial Asthma", 23: "Hypertension", 30: "Migraine",
    7: "Cervical spondylosis", 32: "Paralysis (brain hemorrhage)", 28: "Jaundice",
    29: "Malaria", 8: "Chicken pox", 11: "Dengue", 37: "Typhoid", 40: "hepatitis A",
    19: "Hepatitis B", 20: "Hepatitis C", 21: "Hepatitis D", 22: "Hepatitis E",
    3: "Alcoholic hepatitis", 36: "Tuberculosis", 10: "Common Cold", 34: "Pneumonia",
    13: "Dimorphic hemmorhoids(piles)", 18: "Heart attack", 39: "Varicose veins",
    26: "Hypothyroidism", 24: "Hyperthyroidism", 25: "Hypoglycemia",
    31: "Osteoarthristis", 5: "Arthritis",
    0: "(vertigo) Paroymsal Positional Vertigo", 2: "Acne",
    38: "Urinary tract infection", 35: "Psoriasis", 27: "Impetigo",
}

DISEASE_ALIASES = {
    "Diabetes ": "Diabetes", "Hypertension ": "Hypertension",
    "Peptic ulcer diseae": "Peptic ulcer disease",
    "(vertigo) Paroymsal  Positional Vertigo": "(vertigo) Paroymsal Positional Vertigo",
}

SEVERE_INDICATORS = {
    "chest pain", "breathlessness", "high fever", "vomiting", "blood in sputum",
    "stomach bleeding", "coma", "fast heart rate", "altered sensorium",
    "weakness of one body side", "yellowing of eyes", "acute liver failure",
}

# Disease category tags for the Disease Explorer
DISEASE_CATEGORIES = {
    "Infectious": ["Malaria", "Dengue", "Typhoid", "Tuberculosis", "Chicken pox",
                   "Common Cold", "Pneumonia", "Fungal infection", "AIDS", "Impetigo"],
    "Liver": ["Jaundice", "hepatitis A", "Hepatitis B", "Hepatitis C",
              "Hepatitis D", "Hepatitis E", "Alcoholic hepatitis", "Chronic cholestasis"],
    "Digestive": ["Gastroenteritis", "GERD", "Peptic ulcer disease",
                  "Dimorphic hemmorhoids(piles)"],
    "Metabolic": ["Diabetes", "Hypoglycemia", "Hypothyroidism", "Hyperthyroidism"],
    "Cardiac": ["Heart attack", "Hypertension", "Varicose veins"],
    "Skin": ["Acne", "Psoriasis", "Drug Reaction", "Allergy"],
    "Neurological": ["Migraine", "Paralysis (brain hemorrhage)",
                     "(vertigo) Paroymsal Positional Vertigo", "Cervical spondylosis"],
    "Musculoskeletal": ["Arthritis", "Osteoarthristis"],
    "Urinary": ["Urinary tract infection"],
    "Respiratory": ["Bronchial Asthma"],
}


st.set_page_config(page_title=APP_TITLE, page_icon="🩺", layout="wide")
init_theme()
st.markdown(get_theme_styles(), unsafe_allow_html=True)
render_theme_toggle()


# ─────────────────────────── Utility functions (unchanged) ──────────────────

def normalize_spaces(value):
    return re.sub(r"\s+", " ", str(value).replace("_", " ")).strip()

def normalize_symptom_name(value):
    return normalize_spaces(value).lower()

def normalize_disease_key(value):
    value = DISEASE_ALIASES.get(str(value), str(value))
    value = normalize_spaces(value).lower()
    return re.sub(r"[^a-z0-9]+", "", value)

def pretty_value(value):
    return normalize_spaces(value).title()

def canonical_disease_name(value, disease_lookup=None):
    value = DISEASE_ALIASES.get(str(value), str(value))
    value = normalize_spaces(value)
    if disease_lookup:
        return disease_lookup.get(normalize_disease_key(value), value)
    return value


def clean_list(items):
    return [str(i).strip() for i in items if str(i).strip() and str(i).strip().lower() != "nan"]

def safe_parse_list(value):
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    try:
        parsed = ast.literal_eval(str(value))
        return parsed if isinstance(parsed, list) else [str(parsed)]
    except Exception:
        return [str(value)]

def build_lookup_from_description(description_df):
    disease_lookup = {}
    for disease in description_df["Disease"].dropna().unique():
        canonical = normalize_spaces(disease)
        disease_lookup[normalize_disease_key(canonical)] = canonical
    return disease_lookup

def build_recommendation_lookups(description_df, precautions_df, medications_df, diets_df, workout_df, disease_lookup):
    description_lookup = {}
    for _, row in description_df.iterrows():
        disease = canonical_disease_name(row.get("Disease"), disease_lookup)
        description_lookup[disease] = str(row.get("Description", "Description not available."))

    precautions_lookup = {}
    for _, row in precautions_df.iterrows():
        disease = canonical_disease_name(row.get("Disease"), disease_lookup)
        precautions_lookup[disease] = clean_list([
            row.get("Precaution_1"), row.get("Precaution_2"),
            row.get("Precaution_3"), row.get("Precaution_4"),
        ])

    medication_lookup = {}
    for _, row in medications_df.iterrows():
        disease = canonical_disease_name(row.get("Disease"), disease_lookup)
        medication_lookup[disease] = clean_list(safe_parse_list(row.get("Medication")))

    diet_lookup = {}
    for _, row in diets_df.iterrows():
        disease = canonical_disease_name(row.get("Disease"), disease_lookup)
        diet_lookup[disease] = clean_list(safe_parse_list(row.get("Diet")))

    workout_lookup = {}
    if "disease" in workout_df.columns and "workout" in workout_df.columns:
        for disease, group in workout_df.groupby("disease"):
            disease = canonical_disease_name(disease, disease_lookup)
            workout_lookup[disease] = clean_list(group["workout"].dropna().tolist())

    return description_lookup, precautions_lookup, medication_lookup, diet_lookup, workout_lookup

def build_disease_profiles(training_df, feature_columns, disease_lookup):
    disease_profiles = {}
    for disease, group in training_df.groupby("prognosis"):
        canonical = canonical_disease_name(disease, disease_lookup)
        feature_sums = group[feature_columns].sum(axis=0)
        disease_profiles[canonical] = set(feature_sums[feature_sums > 0].index)
    return disease_profiles

def build_severity_weights(severity_df, symptom_to_column):
    weights = {}
    if "Symptom" not in severity_df.columns or "weight" not in severity_df.columns:
        return weights
    for _, row in severity_df.iterrows():
        symptom = normalize_symptom_name(row.get("Symptom"))
        column = symptom_to_column.get(symptom)
        if column:
            try:
                weights[column] = float(row.get("weight", 1))
            except Exception:
                weights[column] = 1.0
    return weights


@st.cache_resource(show_spinner=False)
def load_data():
    try:
        training_df   = pd.read_csv(DATA_DIR / "Training.csv")
        severity_df   = pd.read_csv(DATA_DIR / "Symptom-severity.csv")
        symptoms_df   = pd.read_csv(DATA_DIR / "symptoms_df.csv")
        precautions_df = pd.read_csv(DATA_DIR / "precautions_df.csv")
        workout_df    = pd.read_csv(DATA_DIR / "workout_df.csv")
        description_df = pd.read_csv(DATA_DIR / "description.csv")
        medications_df = pd.read_csv(DATA_DIR / "medications.csv")
        diets_df      = pd.read_csv(DATA_DIR / "diets.csv")

        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)

        if "prognosis" not in training_df.columns:
            st.error("Training.csv must contain a prognosis column.")
            st.stop()

        feature_columns = [c for c in training_df.columns if c != "prognosis"]
        symptom_to_column = {normalize_symptom_name(c): c for c in feature_columns}
        column_to_symptom = {c: normalize_symptom_name(c) for c in feature_columns}

        disease_lookup = build_lookup_from_description(description_df)
        disease_names = sorted({
            canonical_disease_name(d, disease_lookup)
            for d in description_df["Disease"].dropna().unique()
        })

        (description_lookup, precautions_lookup, medication_lookup,
         diet_lookup, workout_lookup) = build_recommendation_lookups(
            description_df, precautions_df, medications_df, diets_df,
            workout_df, disease_lookup)

        disease_profiles = build_disease_profiles(training_df, feature_columns, disease_lookup)
        severity_weights = build_severity_weights(severity_df, symptom_to_column)

        return {
            "training": training_df, "symptoms": symptoms_df, "model": model,
            "feature_columns": feature_columns,
            "symptom_to_column": symptom_to_column,
            "column_to_symptom": column_to_symptom,
            "symptom_options": sorted(symptom_to_column.keys()),
            "disease_names": disease_names, "disease_lookup": disease_lookup,
            "disease_profiles": disease_profiles, "severity_weights": severity_weights,
            "description_lookup": description_lookup,
            "precautions_lookup": precautions_lookup,
            "medication_lookup": medication_lookup,
            "diet_lookup": diet_lookup, "workout_lookup": workout_lookup,
            "severity_df": severity_df,
        }
    except Exception as error:
        st.error(f"Error loading data: {error}")
        st.stop()


data = load_data()


# ─────────────────────────── Prediction logic (unchanged) ───────────────────

def correct_spelling(symptom):
    symptom = symptom.strip().lower()
    if not symptom:
        return None
    if symptom in data["symptom_to_column"]:
        return symptom
    match = process.extractOne(symptom, data["symptom_to_column"].keys())
    if not match:
        return None
    closest_match, score = match[0], match[1]
    return closest_match if score >= 80 else None

def normalize_symptoms(raw_text, selected_symptoms):
    symptoms = set(selected_symptoms)
    typed = [normalize_symptom_name(i) for i in raw_text.split(",") if i.strip()]
    for s in typed:
        corrected = correct_spelling(s)
        if corrected:
            symptoms.add(corrected)
    return sorted(symptoms)

def build_input_frame(patient_symptoms):
    input_data = {f: 0 for f in data["feature_columns"]}
    for s in patient_symptoms:
        col = data["symptom_to_column"].get(s)
        if col:
            input_data[col] = 1
    return pd.DataFrame([input_data], columns=data["feature_columns"])

def patient_symptoms_to_columns(patient_symptoms):
    return {data["symptom_to_column"][s] for s in patient_symptoms if s in data["symptom_to_column"]}

def map_model_label_to_disease(label):
    if isinstance(label, (str, np.str_)):
        return canonical_disease_name(label, data["disease_lookup"])
    try:
        label_index = int(label)
        disease = MODEL_DISEASE_MAP.get(label_index)
        if disease:
            return canonical_disease_name(disease, data["disease_lookup"])
    except Exception:
        pass
    return canonical_disease_name(str(label), data["disease_lookup"])

def get_model_rankings(input_frame):
    model = data["model"]
    try:
        probabilities = model.predict_proba(input_frame)[0]
        classes = list(getattr(model, "classes_", range(len(probabilities))))
        rows = []
        for label, prob in zip(classes, probabilities):
            disease = map_model_label_to_disease(label)
            rows.append((disease, float(prob) * 100))
        best_by_disease = {}
        for disease, prob in rows:
            best_by_disease[disease] = max(best_by_disease.get(disease, 0), prob)
        return sorted(best_by_disease.items(), key=lambda x: x[1], reverse=True)
    except Exception:
        prediction = model.predict(input_frame)[0]
        return [(map_model_label_to_disease(prediction), 100.0)]

def score_disease_by_symptoms(patient_columns, disease):
    profile = data["disease_profiles"].get(disease, set())
    if not patient_columns or not profile:
        return 0.0, []
    matched = patient_columns.intersection(profile)
    if not matched:
        return 0.0, []
    sw = data["severity_weights"]
    input_weight  = sum(sw.get(c, 1.0) for c in patient_columns)
    matched_weight = sum(sw.get(c, 1.0) for c in matched)
    coverage  = matched_weight / max(input_weight, 1.0)
    precision = len(matched) / max(len(patient_columns), 1)
    profile_s = len(matched) / max(len(profile), 1)
    final     = (coverage * 72) + (precision * 20) + (profile_s * 8)
    matched_names = sorted(data["column_to_symptom"].get(c, c) for c in matched)
    return round(final, 2), matched_names

def rank_diseases(patient_symptoms):
    input_frame    = build_input_frame(patient_symptoms)
    patient_cols   = patient_symptoms_to_columns(patient_symptoms)
    model_rankings = get_model_rankings(input_frame)
    model_scores   = dict(model_rankings)
    ranked_rows    = []
    all_diseases   = set(data["disease_profiles"].keys()).union(model_scores.keys())
    for disease in all_diseases:
        symp_score, matched = score_disease_by_symptoms(patient_cols, disease)
        model_score  = model_scores.get(disease, 0.0)
        final_score  = (symp_score * 0.78) + (model_score * 0.22)
        ranked_rows.append({
            "Disease": disease, "Match Score": round(final_score, 2),
            "Symptom Match": symp_score, "Model Confidence": round(model_score, 2),
            "Matched Symptoms": matched,
        })
    ranked_df = pd.DataFrame(ranked_rows).sort_values(
        by=["Match Score", "Symptom Match", "Model Confidence"], ascending=False)
    model_pred  = model_rankings[0][0] if model_rankings else "Unknown Disease"
    final_pred  = ranked_df.iloc[0]["Disease"] if not ranked_df.empty else model_pred
    return final_pred, ranked_df, model_pred

def disease_information(predicted_dis):
    return (
        data["description_lookup"].get(predicted_dis, "Description not available."),
        data["precautions_lookup"].get(predicted_dis, []),
        data["medication_lookup"].get(predicted_dis, []),
        data["diet_lookup"].get(predicted_dis, []),
        data["workout_lookup"].get(predicted_dis, []),
    )

def get_symptom_severity_level(patient_symptoms):
    total   = len(patient_symptoms)
    severe  = sum(1 for s in patient_symptoms if s in SEVERE_INDICATORS)
    if severe >= 2 or total >= 6:
        return "Severe", "#ba1a1a", "High attention recommended"
    if severe == 1 or total >= 4:
        return "Moderate", "#b45309", "Monitor symptoms carefully"
    return "Mild", "#0f7b55", "Basic care guidance"


# ─────────────────────────────────── CSS ────────────────────────────────────

@st.cache_data(show_spinner=False)
def _get_render_styles_html():
    return """
<link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600;700&display=swap" rel="stylesheet" media="print" onload="this.media='all'">
<noscript><link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600;700&display=swap" rel="stylesheet"></noscript>

<style>
/* ── Tokens ──────────────────────────────────────────────────────── */
:root {
    --p:       #6750a4;
    --p-rgb:   103,80,164;
    --s:       #006a6a;
    --s-rgb:   0,106,106;
    --warn:    #b45309;
    --err:     #ba1a1a;
    --ok:      #0f7b55;

    --sur:     rgba(255,255,255,.048);
    --sur2:    rgba(255,255,255,.075);
    --sur3:    rgba(255,255,255,.108);
    --out:     rgba(148,163,184,.26);
    --out2:    rgba(148,163,184,.15);
    --muted:   rgba(200,192,230,.60);

    --r-sm:  16px;
    --r-md:  24px;
    --r-lg:  32px;
    --r-xl:  42px;

    --sh1: 0 2px 10px rgba(0,0,0,.22);
    --sh2: 0 8px 28px rgba(0,0,0,.28);
    --sh3: 0 20px 56px rgba(0,0,0,.36);

    --font-head: 'DM Serif Display', serif;
    --font-body: 'DM Sans', sans-serif;
    --ease: 170ms cubic-bezier(.4,0,.2,1);
}

html { scroll-behavior: smooth; }
*, *::before, *::after { box-sizing: border-box; }

body, .stApp { font-family: var(--font-body); }

.block-container {
    max-width: 1240px;
    padding-top: .9rem;
    padding-bottom: 2.5rem;
}

::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-thumb { background: rgba(var(--p-rgb),.35); border-radius: 99px; }

/* ── Sidebar ─────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    border-right: 1px solid var(--out);
    background:
        radial-gradient(ellipse at 20% 0%,   rgba(var(--p-rgb),.22), transparent 45%),
        radial-gradient(ellipse at 90% 55%,  rgba(var(--s-rgb),.14), transparent 42%),
        radial-gradient(ellipse at 10% 100%, rgba(103,58,183,.10),   transparent 40%),
        linear-gradient(175deg, rgba(var(--p-rgb),.12) 0%, rgba(var(--s-rgb),.06) 55%, transparent 100%);
}

@media (max-width: 768px) {
    [data-testid="stSidebar"] { background: #1a1730 !important; }
}

[data-testid="stSidebar"] > div:first-child { padding: 0 !important; }
[data-testid="stSidebar"] [data-testid="stSidebarContent"] { padding: 0 !important; }

/* ── Sidebar inner scroll wrapper ───────────────────────────────── */
.sb-wrap {
    padding: 14px 14px 20px;
    display: flex; flex-direction: column; gap: 0;
}

/* ── Brand hero card ─────────────────────────────────────────────── */
.sb-hero {
    position: relative; overflow: hidden;
    border: 1px solid var(--out); border-radius: 20px;
    padding: 16px 16px 14px; background: var(--sur2);
    margin-bottom: 16px;
    box-shadow: 0 4px 24px rgba(var(--p-rgb),.13);
}
.sb-hero::before {
    content:''; position:absolute; top:-40px; right:-40px;
    width:120px; height:120px; border-radius:50%;
    background: radial-gradient(circle, rgba(var(--p-rgb),.22), transparent 70%);
    pointer-events: none;
}
@keyframes sb-hero-dna-scan {
    0%   { transform: rotateY(0deg) scale(1);    filter: brightness(1); }
    25%  { transform: rotateY(90deg) scale(1.06); filter: brightness(1.18); }
    50%  { transform: rotateY(180deg) scale(1);  filter: brightness(1); }
    75%  { transform: rotateY(270deg) scale(1.06); filter: brightness(1.18); }
    100% { transform: rotateY(360deg) scale(1);  filter: brightness(1); }
}
@keyframes sb-online-dp { 0%,100%{box-shadow:0 0 6px rgba(0,200,83,0.55);} 50%{box-shadow:0 0 10px rgba(0,200,83,0.28);} }
.sb-hero-row {
    display: flex; align-items: center; gap: 11px; margin-bottom: 10px;
}
.sb-logo {
    width: 54px; height: 54px; min-width: 54px; border-radius: 18px;
    background: linear-gradient(135deg, #6750a4, #006a6a);
    display: flex; align-items: center; justify-content: center;
    font-size: 26px; box-shadow: 0 4px 14px rgba(var(--p-rgb),.38);
    transform-origin: center;
    animation: sb-hero-dna-scan 4s linear infinite;
    position: relative;
}
.sb-logo::after {
    content: '';
    position: absolute; bottom: -3px; right: -3px;
    width: 13px; height: 13px; border-radius: 50%;
    background: #00c853;
    border: 2px solid var(--bg, #0d0d14);
    box-shadow: 0 0 6px rgba(0,200,83,0.55);
    animation: sb-online-dp 2.4s ease-in-out infinite;
}
.sb-brand { flex: 1; min-width: 0; }
.sb-badge {
    display: inline-flex; padding: 3px 9px; border-radius: 999px;
    background: rgba(var(--p-rgb),.18); border: 1px solid rgba(var(--p-rgb),.38);
    color: #d7c7ff; font-size: 10px; font-weight: 700; margin-bottom: 3px;
    font-family: var(--font-body); letter-spacing: .04em;
}
.sb-title {
    font-size: 15px; font-weight: 800; line-height: 1.2; font-family: var(--font-body);
    background: linear-gradient(135deg, #d7c7ff 0%, #a78bfa 40%, #6750a4 70%, #006a6a 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.sb-text  { color: var(--muted); font-size: 11.5px; line-height: 1.55; margin-top: 2px; }

/* mini stat chips inside hero */
.sb-stats-row {
    display: grid; grid-template-columns: 1fr 1fr; gap: 7px; margin-top: 10px;
}
.sb-stat-chip {
    border: 1px solid var(--out2); border-radius: 12px;
    padding: 8px 10px; background: rgba(255,255,255,.04);
    display: flex; flex-direction: column; gap: 1px;
}
.sb-stat-val  { font-size: 15px; font-weight: 800; line-height: 1.1; }
.sb-stat-lbl  { color: var(--muted); font-size: 10px; font-weight: 600; letter-spacing: .04em; }

/* ── Section header ──────────────────────────────────────────────── */
.sb-section {
    font-size: 10px; font-weight: 700; color: var(--muted);
    text-transform: uppercase; letter-spacing: .10em;
    margin: 14px 4px 7px; display: flex; align-items: center; gap: 7px;
}
.sb-section::after {
    content: ''; flex: 1; height: 1px;
    background: linear-gradient(90deg, var(--out2), transparent);
}

/* ── Nav item ────────────────────────────────────────────────────── */
.sb-nav-item {
    display: flex; align-items: center; gap: 11px;
    padding: 10px 12px; border-radius: 14px;
    border: 1px solid transparent; background: transparent;
    margin-bottom: 4px;
    transition: background var(--ease), border-color var(--ease), transform var(--ease), box-shadow var(--ease);
    cursor: default;
}
.sb-nav-item:hover {
    background: var(--sur2); border-color: var(--out2);
    transform: translateX(3px);
}
.sb-nav-item.active {
    background: linear-gradient(135deg, rgba(var(--p-rgb),.22), rgba(var(--s-rgb),.14));
    border-color: rgba(var(--p-rgb),.42);
    box-shadow: 0 2px 16px rgba(var(--p-rgb),.18), inset 0 1px 0 rgba(255,255,255,.06);
    transform: none;
}
.sb-nav-icon {
    width: 34px; height: 34px; min-width: 34px; border-radius: 11px;
    display: flex; align-items: center; justify-content: center; font-size: 16px;
    background: rgba(255,255,255,.05); border: 1px solid var(--out2);
    transition: background var(--ease), box-shadow var(--ease);
}
.sb-nav-item.active .sb-nav-icon {
    background: linear-gradient(135deg, rgba(var(--p-rgb),.35), rgba(var(--s-rgb),.25));
    border-color: rgba(var(--p-rgb),.45);
    box-shadow: 0 2px 10px rgba(var(--p-rgb),.3);
}
.sb-nav-body { flex: 1; min-width: 0; }
.sb-nav-title { font-size: 13px; font-weight: 600; line-height: 1.2; }
.sb-nav-sub   { color: var(--muted); font-size: 11px; margin-top: 1px; }
.sb-nav-arrow {
    font-size: 12px; color: var(--muted); opacity: 0;
    transition: opacity var(--ease), transform var(--ease);
}
.sb-nav-item.active .sb-nav-arrow,
.sb-nav-item:hover  .sb-nav-arrow { opacity: 1; transform: translateX(2px); }

/* active left accent */
.sb-nav-item.active { position: relative; }
.sb-nav-item.active::before {
    content:''; position:absolute; left:-14px; top:25%; bottom:25%;
    width: 3px; border-radius: 0 3px 3px 0;
    background: linear-gradient(180deg, #6750a4, #006a6a);
}

/* ── Sidebar info/data row (legacy kept for data html) ───────────── */
.sb-link {
    display: flex; align-items: center; gap: 10px;
    padding: 10px 11px; border-radius: var(--r-sm);
    border: 1px solid var(--out2); background: var(--sur);
    margin-bottom: 6px; transition: transform var(--ease), background var(--ease);
}
.sb-link:hover { transform: translateX(3px); background: var(--sur2); border-color: rgba(var(--p-rgb),.35); }
.sb-icon {
    width: 32px; height: 32px; min-width: 32px; border-radius: 12px;
    background: linear-gradient(135deg,rgba(var(--p-rgb),.25),rgba(var(--s-rgb),.17));
    display: flex; align-items: center; justify-content: center; font-size: 14px;
}
.sb-link-title { font-size: 13px; font-weight: 600; }
.sb-link-sub   { color: var(--muted); font-size: 11px; }

/* ── Medical notice ──────────────────────────────────────────────── */
.sb-note {
    border: 1px solid rgba(245,158,11,.30); border-radius: 14px;
    padding: 11px 13px; background: rgba(245,158,11,.08);
    color: var(--muted); font-size: 11.5px; line-height: 1.55; margin-top: 12px;
    display: flex; gap: 9px; align-items: flex-start;
}
.sb-note-icon {
    font-size: 16px; line-height: 1; margin-top: 1px; flex-shrink: 0;
}

/* ── Footer ──────────────────────────────────────────────────────── */
.sb-footer {
    text-align: center; color: var(--muted); font-size: 11px;
    margin-top: 14px; padding-top: 12px;
    border-top: 1px solid var(--out2);
}
.sb-footer-heart { color: #ef4444; display: inline-block; animation: hb 2.5s ease-in-out infinite; }
@keyframes hb { 0%,100%{transform:scale(1)} 50%{transform:scale(1.25)} }
.sb-version-pill {
    display: inline-block; margin-top: 5px;
    padding: 2px 9px; border-radius: 999px;
    background: rgba(var(--p-rgb),.12); border: 1px solid rgba(var(--p-rgb),.25);
    font-size: 10px; font-weight: 700; color: #d7c7ff; letter-spacing: .04em;
}
.sb-creator {
    background: linear-gradient(90deg, #f9a8d4, #c084fc, #818cf8);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; font-weight: 800; font-size: 13px;
    filter: drop-shadow(0 0 6px rgba(192,132,252,.45));
}

/* ── Nav row ─────────────────────────────────────────────────────── */
.main-nav-shell {
    margin: 6px 0 20px;
    padding: 8px;
    border-radius: 24px;
    border: 1px solid var(--out2);
    background: var(--sur);
    box-shadow: var(--sh1);
    overflow-x: auto;
}
div[data-testid="stRadio"] [role="radiogroup"] { display: flex; gap: 8px; flex-wrap: wrap; }
div[data-testid="stRadio"] label {
    min-height: 39px;
    padding: 8px 16px !important;
    margin: 0 !important;
    border-radius: 999px;
    border: 1px solid var(--out2);
    background: var(--sur);
    color: var(--muted) !important;
    font: 700 12px var(--font-body);
    transition: all var(--ease);
    cursor: pointer;
    white-space: nowrap;
}
div[data-testid="stRadio"] label:hover {
    background: var(--sur2);
    border-color: rgba(var(--p-rgb),.42);
    transform: translateY(-1px);
}
div[data-testid="stRadio"] label:has(input:checked) {
    background: linear-gradient(135deg, rgba(var(--p-rgb),.35), rgba(var(--s-rgb),.18));
    border-color: rgba(var(--p-rgb),.68);
    color: #e4d8ff !important;
    box-shadow: 0 8px 26px rgba(var(--p-rgb),.18);
}
div[data-testid="stRadio"] label > div:first-child { display: none !important; }
div[data-testid="stRadio"] label p { color: inherit !important; font: inherit; white-space: nowrap; }

/* Action buttons */
.stButton > button {
    border-radius: 999px !important; min-height: 46px;
    font-weight: 700 !important; font-family: var(--font-body) !important;
    border: 1px solid rgba(var(--p-rgb),.4) !important;
    background: linear-gradient(135deg,#6750a4,#7c3aed) !important;
    color: white !important; box-shadow: var(--sh1);
    display: flex !important; align-items: center !important;
    justify-content: center !important;
    transition: all var(--ease) !important;
}
.stButton > button:hover { transform: translateY(-1px); box-shadow: var(--sh2) !important; }

textarea, [data-baseweb="select"] { border-radius: 16px !important; }
[data-testid="stMultiSelect"] label,
[data-testid="stTextArea"] label,
[data-testid="stSelectbox"] label { font-weight: 600; font-family: var(--font-body); }

/* ── Hero ────────────────────────────────────────────────────────── */
.hero {
    position: relative; overflow: hidden;
    border: 1px solid var(--out); border-radius: var(--r-xl);
    padding: 36px 32px; margin-bottom: 20px;
    background:
        radial-gradient(circle at 10% 15%, rgba(var(--p-rgb),.20), transparent 36%),
        radial-gradient(circle at 88% 12%, rgba(var(--s-rgb),.14), transparent 36%),
        linear-gradient(135deg,rgba(var(--p-rgb),.13),rgba(var(--s-rgb),.08) 60%,rgba(245,158,11,.05)),
        var(--sur);
    box-shadow: var(--sh3);
}
.hero-grid { display: grid; grid-template-columns: 1fr 260px; gap: 28px; align-items: center; }
.hero-eyebrow {
    display: inline-flex; padding: 6px 13px; border-radius: 999px;
    border: 1px solid rgba(var(--p-rgb),.38); background: rgba(var(--p-rgb),.14);
    color: #d7c7ff; font-size: 11px; font-weight: 700; margin-bottom: 12px;
    font-family: var(--font-body);
}
.hero-title {
    font-family: var(--font-head);
    font-size: clamp(30px, 4vw, 56px);
    line-height: 1.04; margin: 0 0 14px 0;
}
.hero-sub { color: var(--muted); font-size: 14px; line-height: 1.7; max-width: 540px; margin-bottom: 18px; }
.chip-row { display: flex; gap: 9px; flex-wrap: wrap; }
.chip {
    padding: 7px 13px; border-radius: 999px;
    border: 1px solid var(--out2); background: var(--sur2);
    font-size: 12px; font-weight: 600; color: var(--muted);
}
.hero-pills { display: flex; flex-direction: column; gap: 10px; }
.hero-pill {
    border: 1px solid var(--out); border-radius: var(--r-sm);
    padding: 13px 15px; background: var(--sur2);
}
.hero-pill-label { color: var(--muted); font-size: 11px; font-weight: 600; margin-bottom: 3px; }
.hero-pill-value { font-size: 14px; font-weight: 700; }
.hero-orb {
    position: absolute; border-radius: 50%; pointer-events: none;
}
.hero-orb-1 { width:280px;height:280px;right:-80px;top:-80px;
    background:radial-gradient(circle,rgba(var(--p-rgb),.18),transparent 70%); }
.hero-orb-2 { width:240px;height:240px;left:60px;bottom:-100px;
    background:radial-gradient(circle,rgba(var(--s-rgb),.11),transparent 70%); }

/* ── Stat cards ──────────────────────────────────────────────────── */
.stat-grid {
    display: grid; grid-template-columns: repeat(4,1fr);
    gap: 12px; margin-bottom: 20px;
}
.stat-card {
    border: 1px solid var(--out2); border-radius: var(--r-md);
    padding: 16px; background: var(--sur); box-shadow: var(--sh1);
    position: relative; overflow: hidden;
    transition: transform var(--ease), border-color var(--ease);
}
.stat-card::before {
    content:''; position:absolute; top:0;left:0;right:0;height:2px;
    background:linear-gradient(90deg,rgba(var(--p-rgb),.7),rgba(var(--s-rgb),.5));
}
.stat-card:hover { transform: translateY(-3px); border-color: rgba(var(--p-rgb),.35); }
.stat-icon { font-size: 20px; margin-bottom: 8px; }
.stat-label { color:var(--muted);font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.07em;margin-bottom:4px; }
.stat-value { font-size:17px;font-weight:800;font-family:var(--font-body); }

/* ── Section / Panel ─────────────────────────────────────────────── */
.panel {
    border: 1px solid var(--out); border-radius: var(--r-lg);
    padding: 22px;
    background: linear-gradient(135deg,rgba(var(--p-rgb),.09),rgba(var(--s-rgb),.06)),var(--sur);
    box-shadow: var(--sh2); margin-bottom: 18px;
}
.sec-title { font-size: 20px; font-weight: 700; font-family: var(--font-body); margin: 0 0 6px; }
.sec-sub   { color: var(--muted); font-size: 13px; line-height: 1.6; }

/* ── Symptom chips ───────────────────────────────────────────────── */
.sym-chip {
    display: inline-flex; align-items: center;
    padding: 6px 12px; margin: 3px 4px 3px 0;
    border-radius: 999px;
    border: 1px solid rgba(var(--p-rgb),.35);
    background: rgba(var(--p-rgb),.14);
    color: #d7c7ff; font-size: 12px; font-weight: 600;
}
.sym-chip.severe {
    border-color: rgba(186,26,26,.5);
    background: rgba(186,26,26,.15);
    color: #ffb4ab;
}

/* ── Severity badge ──────────────────────────────────────────────── */
.sev-badge {
    display: inline-flex; align-items: center; gap: 8px;
    padding: 9px 16px; border-radius: 999px; color: white;
    font-weight: 700; margin: 10px 0 16px; box-shadow: var(--sh1);
    font-size: 13px;
}

/* ── Result card ─────────────────────────────────────────────────── */
.result-card {
    border: 1px solid rgba(var(--p-rgb),.32); border-radius: var(--r-lg);
    padding: 24px; margin: 16px 0;
    background:
        linear-gradient(135deg,rgba(var(--p-rgb),.16),rgba(var(--s-rgb),.09)),
        var(--sur);
    box-shadow: var(--sh3);
}
.result-title {
    font-family: var(--font-head);
    font-size: clamp(22px, 3vw, 36px);
    font-weight: 400; margin: 0 0 10px; color: #d7c7ff;
}
.result-desc { color: var(--muted); font-size: 14px; line-height: 1.65; }

/* ── Match bars ──────────────────────────────────────────────────── */
.match-grid { display: grid; gap: 10px; margin: 12px 0 16px; }
.match-item {
    border: 1px solid var(--out2); border-radius: var(--r-sm);
    padding: 13px; background: rgba(255,255,255,.04);
}
.match-head { display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;font-weight:700;font-size:13px; }
.match-score { color: #d7c7ff; }
.match-bar-wrap { height:7px;border-radius:99px;overflow:hidden;background:rgba(148,163,184,.18);margin-top:9px; }
.match-bar { height:100%;border-radius:99px;background:linear-gradient(90deg,#6750a4,#006a6a); }
.match-meta { color:var(--muted);font-size:12px;margin-top:6px;line-height:1.5; }

/* ── Info boxes ──────────────────────────────────────────────────── */
.info-box {
    border: 1px solid var(--out); border-radius: var(--r-md);
    padding: 18px; background: var(--sur); margin-bottom: 14px;
}
.info-box-title {
    font-size: 15px; font-weight: 700; margin: 0 0 12px;
    display: flex; align-items: center; gap: 7px;
    font-family: var(--font-body);
}
.pretty-list { display: grid; gap: 8px; }
.pretty-item {
    display: flex; gap: 10px; align-items: flex-start;
    padding: 10px 12px; border-radius: var(--r-sm);
    background: rgba(255,255,255,.042); border: 1px solid var(--out2);
    color: var(--muted); font-size: 13px; line-height: 1.5;
}
.pretty-num {
    width: 24px; height: 24px; min-width: 24px; border-radius: 9px;
    display: flex; align-items: center; justify-content: center;
    color: white; font-size: 11px; font-weight: 800;
    background: linear-gradient(135deg,#6750a4,#006a6a);
    flex-shrink: 0;
}

/* ── Disease Explorer cards ──────────────────────────────────────── */
.cat-grid { display: grid; grid-template-columns: repeat(auto-fill,minmax(200px,1fr)); gap: 10px; margin-bottom: 18px; }
.cat-card {
    border: 1px solid var(--out2); border-radius: var(--r-sm);
    padding: 14px; background: var(--sur); cursor: pointer;
    transition: transform var(--ease), border-color var(--ease), background var(--ease);
}
.cat-card:hover { transform:translateY(-2px);border-color:rgba(var(--p-rgb),.45);background:var(--sur2); }
.cat-card.selected { border-color:rgba(var(--p-rgb),.7);background:rgba(var(--p-rgb),.14); }
.cat-icon { font-size:22px;margin-bottom:7px; }
.cat-name { font-size:13px;font-weight:700; }
.cat-count { color:var(--muted);font-size:11px;margin-top:2px; }

/* ── Disease row ─────────────────────────────────────────────────── */
.dis-row {
    display:flex;align-items:center;justify-content:space-between;
    padding:12px 14px; border:1px solid var(--out2); border-radius:var(--r-sm);
    background:var(--sur); margin-bottom:7px;
    transition: background var(--ease), border-color var(--ease);
}
.dis-row:hover { background:var(--sur2);border-color:rgba(var(--p-rgb),.35); }
.dis-name { font-size:14px;font-weight:600; }
.dis-tag {
    padding:3px 9px;border-radius:999px;font-size:11px;font-weight:600;
    background:rgba(var(--s-rgb),.15);border:1px solid rgba(var(--s-rgb),.3);
    color:#80cbc4;
}

/* ── Symptom Checker sidebar for tracker ─────────────────────────── */
.track-card {
    border:1px solid var(--out);border-radius:var(--r-md);
    padding:16px;background:var(--sur);margin-bottom:10px;
}
.track-date { color:var(--muted);font-size:11px;margin-bottom:5px; }
.track-disease { font-size:15px;font-weight:700;margin-bottom:6px;font-family:var(--font-body); }
.track-badge {
    display:inline-block;padding:3px 10px;border-radius:999px;
    font-size:11px;font-weight:600;
}

/* ── Symptom weight / severity bar ──────────────────────────────── */
.sw-row { display:flex;align-items:center;gap:10px;margin-bottom:9px; }
.sw-label { min-width:160px;font-size:12px;font-weight:600;color:var(--muted); }
.sw-bar-bg { flex:1;height:7px;background:var(--out);border-radius:99px;overflow:hidden; }
.sw-bar-fill { height:100%;border-radius:99px;background:linear-gradient(90deg,#6750a4,#006a6a); }
.sw-val { min-width:28px;font-size:12px;font-weight:600;text-align:right;color:var(--muted); }

/* ── Footer ──────────────────────────────────────────────────────── */
.md-footer {
    margin-top:52px; border-top:1px solid var(--out2);
    padding:32px 0 28px 0;
}
.md-footer-top { display:flex; justify-content:center; margin-bottom:20px; }
.md-footer-brand { display:flex; align-items:center; gap:14px; }
.md-footer-logo {
    width:52px; height:52px; border-radius:18px;
    background: linear-gradient(135deg, #6750a4, #006a6a);
    border:none;
    display:flex; align-items:center; justify-content:center;
    font-size:26px; flex-shrink:0;
    box-shadow:0 4px 14px rgba(103,80,164,.38); box-sizing:border-box;
}
.md-footer-brand-name { font-size:15px; font-weight:900; line-height:1.2; max-width:200px; }
.md-footer-brand-sub { font-size:12px; color:var(--muted); margin-top:3px; }
.md-footer-links {
    display:flex; flex-wrap:wrap; gap:8px;
    justify-content:center; margin-bottom:14px;
}
.md-footer-link {
    padding:8px 16px; border-radius:999px;
    border:1px solid rgba(148,163,184,0.18);
    background:rgba(255,255,255,0.04); color:var(--muted) !important;
    font-size:13px; font-weight:700; text-decoration:none !important;
    transition:all 130ms ease; display:inline-flex; align-items:center; gap:5px;
}
.md-footer-link:hover {
    background:rgba(103,80,164,.12);
    border-color:rgba(103,80,164,.38);
    color:#d7c7ff !important; transform:translateY(-2px);
    text-decoration:none !important;
}
.md-footer-meta {
    text-align:center; color:var(--muted);
    font-size:12px; line-height:1.75; margin-bottom:16px;
    white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
}
.md-footer-version {
    display:inline-block; padding:3px 10px; border-radius:999px;
    background:rgba(103,80,164,.12);
    border:1px solid rgba(103,80,164,.26);
    font-size:11px; font-weight:800; color:#d7c7ff; vertical-align:middle;
}
.md-footer-heart { color:#ef4444; display:inline-block; animation:md-heartbeat 2.5s ease-in-out infinite; }
@keyframes md-heartbeat { 0%,100%{transform:scale(1)} 50%{transform:scale(1.22)} }
.md-footer-disclaimer {
    margin-top:20px; padding:13px 18px; border-radius:16px;
    background:rgba(217,119,6,.07); border:1px solid rgba(217,119,6,.22);
    color:var(--muted); font-size:12px; line-height:1.6; text-align:center;
}
@media (max-width:620px) {
    .md-footer-brand { justify-content:center; }
    .md-footer-meta  { white-space:normal; }
}

/* ── Light Mode Overrides ───────────────────────────────────────── */
[data-theme="light"] {
    --sur:     rgba(255,255,255,0.88);
    --sur2:    rgba(103,80,164,0.09);
    --sur3:    rgba(103,80,164,0.13);
    --out:     rgba(103,80,164,0.25);
    --out2:    rgba(103,80,164,0.15);
    --muted:   rgba(50,40,80,0.72);
}

/* App background */
[data-theme="light"] .stApp,
[data-theme="light"] [data-testid="stAppViewContainer"] {
    background: linear-gradient(160deg, #f8f6ff 0%, #eef8f7 55%, #f5f0ff 100%) !important;
    color: #1d1b27 !important;
}
[data-theme="light"] [data-testid="stMain"] { background: transparent !important; }

/* Sidebar (non-mobile) */
[data-theme="light"] [data-testid="stSidebar"] {
    background:
        radial-gradient(ellipse at 20% 0%,  rgba(103,80,164,.14), transparent 42%),
        radial-gradient(ellipse at 88% 55%, rgba(0,106,106,.09),  transparent 40%),
        linear-gradient(175deg, rgba(103,80,164,.08) 0%, rgba(0,106,106,.04) 55%, transparent 100%) !important;
    border-right: 1px solid rgba(103,80,164,0.18) !important;
}
@media (max-width: 768px) {
    [data-theme="light"] [data-testid="stSidebar"] {
        background: #f0ecff !important;
    }
}

/* Sidebar hero card */
[data-theme="light"] .sb-hero {
    background: linear-gradient(135deg, rgba(103,80,164,0.10), rgba(0,106,106,0.06)),
        rgba(255,255,255,0.88) !important;
    border-color: rgba(103,80,164,0.22) !important;
}
[data-theme="light"] .sb-badge {
    background: rgba(103,80,164,0.12) !important;
    border-color: rgba(103,80,164,0.32) !important;
    color: #4a3a7d !important;
}
[data-theme="light"] .sb-title   {
    background: linear-gradient(135deg, #6750a4 0%, #4a3580 50%, #006a6a 100%) !important;
    -webkit-background-clip: text !important; -webkit-text-fill-color: transparent !important; background-clip: text !important;
}
[data-theme="light"] .sb-text    { color: #5a5270 !important; }
[data-theme="light"] .sb-section { color: #7c6a9a !important; }
[data-theme="light"] .sb-stat-chip {
    background: rgba(255,255,255,0.70) !important;
    border-color: rgba(103,80,164,0.18) !important;
}
[data-theme="light"] .sb-stat-val  { color: #1a1530 !important; }
[data-theme="light"] .sb-stat-lbl  { color: #7c6a9a !important; }
/* nav items */
[data-theme="light"] .sb-nav-item { color: #1a1530 !important; }
[data-theme="light"] .sb-nav-item:hover {
    background: rgba(103,80,164,0.09) !important;
    border-color: rgba(103,80,164,0.22) !important;
}
[data-theme="light"] .sb-nav-item.active {
    background: linear-gradient(135deg, rgba(103,80,164,0.15), rgba(0,106,106,0.10)) !important;
    border-color: rgba(103,80,164,0.40) !important;
}
[data-theme="light"] .sb-nav-icon {
    background: rgba(255,255,255,0.72) !important;
    border-color: rgba(103,80,164,0.18) !important;
}
[data-theme="light"] .sb-nav-item.active .sb-nav-icon {
    background: linear-gradient(135deg, rgba(103,80,164,0.20), rgba(0,106,106,0.14)) !important;
    border-color: rgba(103,80,164,0.38) !important;
}
[data-theme="light"] .sb-nav-title { color: #1a1530 !important; }
[data-theme="light"] .sb-nav-sub   { color: #7c6a9a !important; }
[data-theme="light"] .sb-nav-arrow { color: #7c6a9a !important; }
/* legacy data links */
[data-theme="light"] .sb-link {
    background: rgba(255,255,255,0.72) !important;
    border-color: rgba(103,80,164,0.16) !important;
}
[data-theme="light"] .sb-link:hover {
    background: rgba(103,80,164,0.10) !important;
    border-color: rgba(103,80,164,0.36) !important;
}
[data-theme="light"] .sb-link-title { color: #1a1530 !important; }
[data-theme="light"] .sb-link-sub   { color: #5a5270 !important; }
[data-theme="light"] .sb-note {
    background: rgba(245,158,11,0.07) !important;
    border-color: rgba(245,158,11,0.26) !important;
    color: #5a5270 !important;
}
[data-theme="light"] .sb-footer { color: #7c6a9a !important; border-top-color: rgba(103,80,164,0.18) !important; }
[data-theme="light"] .sb-creator {
    background: linear-gradient(90deg, #db2777, #7c3aed, #4f46e5);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; filter: drop-shadow(0 0 5px rgba(124,58,237,.30));
}
[data-theme="light"] .sb-version-pill {
    background: rgba(103,80,164,0.12) !important;
    border-color: rgba(103,80,164,0.30) !important;
    color: #3a2a7d !important;
}

/* Hero section */
[data-theme="light"] .hero {
    background:
        linear-gradient(135deg, rgba(103,80,164,0.12), rgba(0,106,106,0.07) 58%, rgba(186,26,26,0.04)),
        rgba(255,255,255,0.80) !important;
    border-color: rgba(103,80,164,0.20) !important;
}
[data-theme="light"] .hero-eyebrow {
    border-color: rgba(103,80,164,0.36) !important;
    background: rgba(103,80,164,0.11) !important;
    color: #4a3a7d !important;
}
[data-theme="light"] .hero-title   { color: #1a1530 !important; }
[data-theme="light"] .hero-sub     { color: #5a5270 !important; }
[data-theme="light"] .hero-pill {
    background: rgba(255,255,255,0.80) !important;
    border-color: rgba(103,80,164,0.18) !important;
}
[data-theme="light"] .hero-pill-label { color: #7c6a9a !important; }
[data-theme="light"] .hero-pill-value { color: #1a1530 !important; }
[data-theme="light"] .chip {
    border-color: rgba(103,80,164,0.24) !important;
    background: rgba(103,80,164,0.08) !important;
    color: #3a2a5d !important;
}
[data-theme="light"] .chip:hover { border-color: rgba(103,80,164,0.42) !important; }

/* Stat grid */
[data-theme="light"] .stat-card {
    background: rgba(255,255,255,0.82) !important;
    border-color: rgba(103,80,164,0.16) !important;
}
[data-theme="light"] .stat-label { color: #7c6a9a !important; }
[data-theme="light"] .stat-value { color: #1a1530 !important; }

/* Section titles */
[data-theme="light"] .sec-title { color: #1a1530 !important; }
[data-theme="light"] .sec-sub   { color: #5a5270 !important; }

/* Tool panel / form panels */
[data-theme="light"] .tool-panel,
[data-theme="light"] .result-panel,
[data-theme="light"] .info-box {
    background: rgba(255,255,255,0.82) !important;
    border-color: rgba(103,80,164,0.18) !important;
}
[data-theme="light"] .info-box-title { color: #1a1530 !important; }
[data-theme="light"] .info-box-sub   { color: #5a5270 !important; }

/* Symptom chips */
[data-theme="light"] .sym-chip {
    border-color: rgba(103,80,164,0.32) !important;
    background: rgba(103,80,164,0.10) !important;
    color: #4a3a7d !important;
}
[data-theme="light"] .sym-chip:hover {
    background: rgba(103,80,164,0.18) !important;
    border-color: rgba(103,80,164,0.48) !important;
}
[data-theme="light"] .sym-chip.severe {
    border-color: rgba(186,26,26,0.40) !important;
    background: rgba(186,26,26,0.10) !important;
    color: #8b2c2c !important;
}

/* Result cards */
[data-theme="light"] .result-card,
[data-theme="light"] .match-card {
    background: rgba(255,255,255,0.82) !important;
    border-color: rgba(103,80,164,0.18) !important;
}
[data-theme="light"] .result-title  { color: #1a1530 !important; }
[data-theme="light"] .result-desc   { color: #5a5270 !important; }
[data-theme="light"] .match-head    { color: #1a1530 !important; }
[data-theme="light"] .match-score   { color: #4a3a7d !important; }
[data-theme="light"] .match-meta    { color: #7c6a9a !important; }

/* Disease library */
[data-theme="light"] .dis-card,
[data-theme="light"] .cat-card {
    background: rgba(255,255,255,0.82) !important;
    border-color: rgba(103,80,164,0.16) !important;
}
[data-theme="light"] .dis-name  { color: #1a1530 !important; }
[data-theme="light"] .dis-tag {
    background: rgba(0,106,106,0.10) !important;
    border-color: rgba(0,106,106,0.28) !important;
    color: #004f4f !important;
}
[data-theme="light"] .cat-name  { color: #1a1530 !important; }
[data-theme="light"] .cat-count { color: #7c6a9a !important; }

/* Pretty list items */
[data-theme="light"] .pretty-list .pretty-item {
    background: rgba(103,80,164,0.07) !important;
    border-color: rgba(103,80,164,0.20) !important;
    color: #3a2a5d !important;
}
[data-theme="light"] .pretty-item { color: #3a2a5d !important; }

/* History / tracking */
[data-theme="light"] .track-card {
    background: rgba(255,255,255,0.82) !important;
    border-color: rgba(103,80,164,0.16) !important;
}
[data-theme="light"] .track-disease { color: #1a1530 !important; }
[data-theme="light"] .track-date    { color: #7c6a9a !important; }

/* Severity badge */
[data-theme="light"] .sev-badge { color: white !important; }

/* Muted / generic text */
[data-theme="light"] .muted { color: #5a5270 !important; }

/* Nav buttons */
[data-theme="light"] div[data-testid="stRadio"] label {
    background: rgba(255,255,255,0.88) !important;
    border-color: rgba(103,80,164,0.22) !important;
    color: #4a3a6d !important;
}
[data-theme="light"] div[data-testid="stRadio"] label:hover {
    background: rgba(103,80,164,0.08) !important;
    border-color: rgba(103,80,164,0.42) !important;
}
[data-theme="light"] div[data-testid="stRadio"] label:has(input:checked) {
    background: linear-gradient(135deg, rgba(103,80,164,0.22), rgba(0,106,106,0.14)) !important;
    border-color: rgba(103,80,164,0.55) !important;
    color: #3a2a5d !important;
    box-shadow: 0 6px 20px rgba(103,80,164,0.15) !important;
}
[data-theme="light"] .stButton > button {
    background: linear-gradient(135deg,#6750a4,#7c3aed) !important;
    color: white !important;
}

/* Tabs */
[data-theme="light"] .stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.82) !important;
    border-color: rgba(103,80,164,0.16) !important;
}
[data-theme="light"] .stTabs [data-baseweb="tab"]  { color: #3a2a5d !important; }
[data-theme="light"] .stTabs [aria-selected="true"] { color: white !important; }

/* Inputs */
[data-theme="light"] input,
[data-theme="light"] [data-baseweb="select"],
[data-theme="light"] textarea {
    background: rgba(255,255,255,0.90) !important;
    border-color: rgba(103,80,164,0.25) !important;
    color: #1a1530 !important;
}
[data-theme="light"] [data-testid="stMultiSelect"] label,
[data-theme="light"] [data-testid="stTextArea"] label,
[data-theme="light"] [data-testid="stSelectbox"] label,
[data-theme="light"] label,
[data-theme="light"] p,
[data-theme="light"] li {
    color: #1d1b27 !important;
}

/* ── Responsive ──────────────────────────────────────────────────── */
@media(max-width:980px) {
    .hero-grid,.stat-grid { grid-template-columns:1fr !important; }
    .hero { padding:22px;border-radius:var(--r-lg); }
    .block-container { padding-left:1rem;padding-right:1rem; }
}
@media(max-width:620px) {
    .hero-title { font-size:28px; }
    .stat-grid { grid-template-columns:1fr 1fr; }
    .cat-grid { grid-template-columns:1fr 1fr; }
}

/* ══════════════════════════════════════════════════════════════════
   LIGHT MODE — COMPREHENSIVE OVERRIDES
   ══════════════════════════════════════════════════════════════════ */

/* ── Panel / generic cards ──────────────────────────────────────── */
[data-theme="light"] .panel {
    background: linear-gradient(135deg,rgba(103,80,164,0.07),rgba(0,106,106,0.04)), rgba(255,255,255,0.90) !important;
    border-color: rgba(103,80,164,0.18) !important;
    box-shadow: 0 4px 20px rgba(103,80,164,0.08) !important;
}

/* ── Match bars ─────────────────────────────────────────────────── */
[data-theme="light"] .match-item {
    background: rgba(255,255,255,0.85) !important;
    border-color: rgba(103,80,164,0.18) !important;
}
[data-theme="light"] .match-bar-wrap { background: rgba(103,80,164,0.12) !important; }
[data-theme="light"] .match-score    { color: #4a3a7d !important; }
[data-theme="light"] .match-meta     { color: #5a5270 !important; }
[data-theme="light"] .match-head     { color: #1a1530 !important; }

/* ── Pretty-item rows ───────────────────────────────────────────── */
[data-theme="light"] .pretty-item {
    background: rgba(103,80,164,0.07) !important;
    border-color: rgba(103,80,164,0.18) !important;
    color: #2a1a5d !important;
}

/* ── Dis-row (disease library rows) ────────────────────────────── */
[data-theme="light"] .dis-row {
    background: rgba(255,255,255,0.85) !important;
    border-color: rgba(103,80,164,0.18) !important;
}
[data-theme="light"] .dis-row:hover {
    background: rgba(103,80,164,0.08) !important;
    border-color: rgba(103,80,164,0.35) !important;
}
[data-theme="light"] .dis-name { color: #1a1530 !important; }

/* ── Multiselect / tag selected items (Diagnose black tags fix) ── */
[data-theme="light"] [data-baseweb="tag"] {
    background: rgba(103,80,164,0.14) !important;
    border: 1px solid rgba(103,80,164,0.35) !important;
}
[data-theme="light"] [data-baseweb="tag"] span {
    color: #2a1a5d !important;
}
[data-theme="light"] [data-baseweb="tag"] [role="button"] {
    color: #4a3a7d !important;
}
[data-theme="light"] [data-baseweb="tag"]:hover {
    background: rgba(103,80,164,0.22) !important;
}
/* Multiselect dropdown options */
[data-theme="light"] [data-baseweb="popover"],
[data-theme="light"] [data-baseweb="menu"] {
    background: #f5f2ff !important;
    border: 1px solid rgba(103,80,164,0.22) !important;
}
[data-theme="light"] [data-baseweb="option"] {
    background: transparent !important;
    color: #1a1530 !important;
}
[data-theme="light"] [data-baseweb="option"]:hover,
[data-theme="light"] [aria-selected="true"][data-baseweb="option"] {
    background: rgba(103,80,164,0.10) !important;
    color: #2a1a5d !important;
}
/* Multiselect container itself */
[data-theme="light"] [data-testid="stMultiSelect"] [data-baseweb="select"] > div:first-child {
    background: rgba(255,255,255,0.92) !important;
    border-color: rgba(103,80,164,0.28) !important;
}
/* Select input text color */
[data-theme="light"] [data-baseweb="select"] input {
    color: #1a1530 !important;
}

/* ── Footer ─────────────────────────────────────────────────────── */
[data-theme="light"] .md-footer {
    border-top-color: rgba(103,80,164,0.20) !important;
}
[data-theme="light"] .md-footer-logo {
    background: linear-gradient(135deg, #6750a4, #006a6a) !important;
    box-shadow: 0 4px 14px rgba(103,80,164,.30) !important;
}
[data-theme="light"] .md-footer-brand-name { color: #1a1530 !important; }
[data-theme="light"] .md-footer-brand-sub  { color: #5a5270 !important; }
/* GitHub / LinkedIn pills — strong visible styles */
[data-theme="light"] .md-footer-link {
    background: rgba(255,255,255,0.92) !important;
    border: 1.5px solid rgba(103,80,164,0.40) !important;
    color: #2a1a5d !important;
    font-weight: 700 !important;
    box-shadow: 0 2px 8px rgba(103,80,164,0.12) !important;
}
[data-theme="light"] .md-footer-link:hover {
    background: rgba(103,80,164,0.12) !important;
    border-color: rgba(103,80,164,0.60) !important;
    color: #1a0a4d !important;
    box-shadow: 0 4px 16px rgba(103,80,164,0.20) !important;
    transform: translateY(-2px) !important;
}
[data-theme="light"] .md-footer-meta { color: #5a5270 !important; }
[data-theme="light"] .md-footer-version {
    background: rgba(103,80,164,0.12) !important;
    border-color: rgba(103,80,164,0.32) !important;
    color: #3a2a7d !important;
}
[data-theme="light"] .md-footer-disclaimer {
    background: rgba(217,119,6,0.08) !important;
    border-color: rgba(217,119,6,0.28) !important;
    color: #4a3a10 !important;
}

/* ── Symptom severity / weight bars ────────────────────────────── */
[data-theme="light"] .sw-label { color: #3a2a5d !important; }
[data-theme="light"] .sw-bar-bg { background: rgba(103,80,164,0.14) !important; }
[data-theme="light"] .sw-val    { color: #3a2a5d !important; }

/* ── Scrollbar ──────────────────────────────────────────────────── */
[data-theme="light"] ::-webkit-scrollbar-thumb { background: rgba(103,80,164,0.30) !important; }

/* ── Sidebar toggle button ──────────────────────────────────────── */
[data-theme="light"] [data-testid="stSidebarCollapsedControl"] button {
    background: rgba(103,80,164,0.14) !important;
    border-color: rgba(103,80,164,0.36) !important;
    box-shadow: 0 2px 10px rgba(103,80,164,0.14) !important;
}
[data-theme="light"] [data-testid="stSidebarCollapsedControl"] button:hover {
    background: rgba(103,80,164,0.24) !important;
}

/* ── Streamlit native headings / expanders / tables ────────────── */
[data-theme="light"] [data-testid="stMarkdownContainer"] h1,
[data-theme="light"] [data-testid="stMarkdownContainer"] h2,
[data-theme="light"] [data-testid="stMarkdownContainer"] h3,
[data-theme="light"] [data-testid="stMarkdownContainer"] h4,
[data-theme="light"] [data-testid="stMarkdownContainer"] h5,
[data-theme="light"] [data-testid="stMarkdownContainer"] h6 {
    color: #1a1530 !important;
}
[data-theme="light"] [data-testid="stMarkdownContainer"] strong { color: #1a1530 !important; }
[data-theme="light"] [data-testid="stMarkdownContainer"] em     { color: #3a2a5d !important; }

[data-theme="light"] [data-testid="stExpander"] {
    background: rgba(255,255,255,0.82) !important;
    border-color: rgba(103,80,164,0.20) !important;
}
[data-theme="light"] [data-testid="stExpander"] summary { color: #1a1530 !important; }
[data-theme="light"] [data-testid="stExpander"] details  { color: #2a2040 !important; }

[data-theme="light"] .stDataFrame,
[data-theme="light"] [data-testid="stTable"] {
    background: rgba(255,255,255,0.90) !important;
    border-color: rgba(103,80,164,0.16) !important;
}

/* ── Caption / small text ───────────────────────────────────────── */
[data-theme="light"] [data-testid="stCaptionContainer"],
[data-theme="light"] .stCaption,
[data-theme="light"] small {
    color: #5a5270 !important;
}

/* ── Info / warning / error boxes ──────────────────────────────── */
[data-theme="light"] [data-testid="stAlert"] {
    background: rgba(255,255,255,0.85) !important;
    color: #1a1530 !important;
}

/* ── Sidebar sb-note (warning box) ─────────────────────────────── */
[data-theme="light"] .sb-note {
    color: #4a3a10 !important;
}
[data-theme="light"] .sb-note strong { color: #3a2800 !important; }

/* ── Mobile sidebar — fully opaque light ───────────────────────── */
@media (max-width: 768px) {
    [data-theme="light"] [data-testid="stSidebar"] {
        background: #ede8ff !important;
        border-right: 1.5px solid rgba(103,80,164,0.28) !important;
        box-shadow: 4px 0 24px rgba(103,80,164,0.14) !important;
    }
    [data-theme="light"] [data-testid="stSidebar"] .sb-hero {
        background: rgba(255,255,255,0.95) !important;
        border-color: rgba(103,80,164,0.24) !important;
    }
    [data-theme="light"] [data-testid="stSidebar"] .sb-nav-item:hover {
        background: rgba(103,80,164,0.10) !important;
    }
    [data-theme="light"] [data-testid="stSidebar"] .sb-link {
        background: rgba(255,255,255,0.88) !important;
        border-color: rgba(103,80,164,0.22) !important;
    }
    [data-theme="light"] [data-testid="stSidebar"] .sb-link:hover {
        background: rgba(103,80,164,0.12) !important;
    }
    [data-theme="light"] [data-testid="stSidebar"] .sb-note {
        background: rgba(245,158,11,0.10) !important;
        border-color: rgba(245,158,11,0.35) !important;
    }
    [data-theme="light"] [data-testid="stSidebar"] .sb-section { color: #6a5a8a !important; }
    [data-theme="light"] [data-testid="stSidebar"] .sb-footer  { color: #7c6a9a !important; }
}
</style>
"""


def render_styles():
    st.markdown(_get_render_styles_html(), unsafe_allow_html=True)


# ─────────────────────────────── Sidebar ────────────────────────────────────

def render_sidebar(page):
    NAV_META = {
        "Home":    ("🏠", "Overview & stats"),
        "Diagnose":("🔍", "Symptom analysis"),
        "Library": ("🗂", "All 40+ diseases"),
        "Insights":("📊", "Symptom weights"),
        "History": ("📋", "Past diagnoses"),
        "About":   ("ℹ️", "How it works"),
    }
    with st.sidebar:


        symptom_count = len(data["feature_columns"])
        disease_count = len(data["disease_names"])
        history_count = len(st.session_state.get("history", []))

        # ── Brand hero card ──────────────────────────────────────────
        st.markdown(f"""
<div class="sb-wrap">
<div class="sb-hero">
    <div class="sb-hero-row">
        <div class="sb-logo">🩺</div>
        <div class="sb-brand">
            <div class="sb-badge">AI · MEDICAL</div>
            <div class="sb-title">Early Diagnosis AI</div>
        </div>
    </div>
    <div class="sb-text">Symptom analysis &amp; health guidance powered by a Random Forest model.</div>
    <div class="sb-stats-row">
        <div class="sb-stat-chip">
            <span class="sb-stat-val">{symptom_count}</span>
            <span class="sb-stat-lbl">Symptoms</span>
        </div>
        <div class="sb-stat-chip">
            <span class="sb-stat-val">{disease_count}</span>
            <span class="sb-stat-lbl">Diseases</span>
        </div>
        <div class="sb-stat-chip">
            <span class="sb-stat-val">{history_count}</span>
            <span class="sb-stat-lbl">Diagnoses</span>
        </div>
        <div class="sb-stat-chip">
            <span class="sb-stat-val">RF</span>
            <span class="sb-stat-lbl">Model</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

        # ── Navigation ───────────────────────────────────────────────
        st.markdown("<div class='sb-section'>Navigation</div>", unsafe_allow_html=True)
        nav_html = ""
        for key, (icon, subtitle) in NAV_META.items():
            active_cls = " active" if page == key else ""
            nav_html += f"""
<div class="sb-nav-item{active_cls}">
    <div class="sb-nav-icon">{icon}</div>
    <div class="sb-nav-body">
        <div class="sb-nav-title">{key}</div>
        <div class="sb-nav-sub">{subtitle}</div>
    </div>
    <span class="sb-nav-arrow">›</span>
</div>"""
        st.markdown(nav_html, unsafe_allow_html=True)

        # ── Medical notice ───────────────────────────────────────────
        st.markdown("""
<div class="sb-note">
    <div class="sb-note-icon">⚠️</div>
    <div><strong>Medical Notice</strong><br>Educational tool only. Always consult a licensed healthcare professional for medical decisions.</div>
</div>
<div class="sb-footer">
    Made with <span class="sb-footer-heart">❤️</span> by <span class="sb-creator">Yatin Sharma</span><br>
    <span class="sb-version-pill">v1.0 · Early Diagnosis AI</span>
</div>
</div>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def _get_sidebar_data_html(symptom_count, disease_count):
    return f"""
<div class="sb-link">
    <div class="sb-icon">💊</div>
    <div>
        <div class="sb-link-title">{symptom_count} Symptoms</div>
        <div class="sb-link-sub">In feature set</div>
    </div>
</div>
<div class="sb-link">
    <div class="sb-icon">🦠</div>
    <div>
        <div class="sb-link-title">{disease_count} Diseases</div>
        <div class="sb-link-sub">In disease library</div>
    </div>
</div>
"""


def render_widget_nav_bridge():
    bridge_pages = ["Home", "Diagnose", "Library", "Insights", "History", "About"]
    for bridge_page in bridge_pages:
        if st.button(f"widget-nav-{bridge_page}", key=f"widget_nav_{bridge_page}"):
            st.session_state.page = bridge_page
            st.rerun()
    components.html(
        """
<script>
(function() {
    const parentDoc = window.parent.document;
    Array.from(parentDoc.querySelectorAll("button")).forEach(button => {
        const label = (button.innerText || button.textContent || "").trim();
        if (!label.startsWith("widget-nav-")) return;
        const wrapper = button.closest('[data-testid="stElementContainer"], [data-testid="stButton"]') || button;
        wrapper.style.position = "absolute";
        wrapper.style.left = "-10000px";
        wrapper.style.top = "0";
        wrapper.style.width = "1px";
        wrapper.style.height = "1px";
        wrapper.style.overflow = "hidden";
        wrapper.style.opacity = "0";
    });
})();
</script>
        """,
        height=0,
        width=0,
    )


# ──────────────────────────────── Hero ──────────────────────────────────────

@st.cache_data(show_spinner=False)
def _get_hero_html(disease_count):
    return f"""
<div class="hero">
    <div class="hero-orb hero-orb-1"></div>
    <div class="hero-orb hero-orb-2"></div>
    <div class="hero-grid">
        <div>
            <div class="hero-eyebrow">✨ AI-Powered · RF Model · Symptom Matching</div>
            <h1 class="hero-title">Early Diagnosis AI</h1>
            <p class="hero-sub">
                Select or type your symptoms and get an instant AI-powered disease prediction
                with detailed health guidance — precautions, medications, diet, and workouts.
            </p>
            <div class="chip-row">
                <div class="chip">⚡ Instant Prediction</div>
                <div class="chip">🧠 Random Forest Model</div>
                <div class="chip">📊 Symptom Matching</div>
                <div class="chip">🛡 Health Guidance</div>
            </div>
        </div>
        <div class="hero-pills">
            <div class="hero-pill">
                <div class="hero-pill-label">Prediction Logic</div>
                <div class="hero-pill-value">Model + Symptom Fit</div>
            </div>
            <div class="hero-pill">
                <div class="hero-pill-label">Guidance Types</div>
                <div class="hero-pill-value">Precautions · Meds · Diet · Workout</div>
            </div>
            <div class="hero-pill">
                <div class="hero-pill-label">Symptom Input</div>
                <div class="hero-pill-value">Select + Free-type</div>
            </div>
            <div class="hero-pill">
                <div class="hero-pill-label">Disease Library</div>
                <div class="hero-pill-value">{disease_count} Diseases</div>
            </div>
        </div>
    </div>
</div>
"""


def render_hero():
    st.markdown(_get_hero_html(len(data["disease_names"])), unsafe_allow_html=True)


# ──────────────────────────────── Stat bar ───────────────────────────────────

def render_stats():
    history = st.session_state.get("history", [])
    st.markdown(f"""
<div class="stat-grid">
    <div class="stat-card">
        <div class="stat-icon">🩺</div>
        <div class="stat-label">Symptoms</div>
        <div class="stat-value">{len(data["feature_columns"])}</div>
    </div>
    <div class="stat-card">
        <div class="stat-icon">🦠</div>
        <div class="stat-label">Diseases</div>
        <div class="stat-value">{len(data["disease_names"])}</div>
    </div>
    <div class="stat-card">
        <div class="stat-icon">🔬</div>
        <div class="stat-label">Diagnoses Done</div>
        <div class="stat-value">{len(history)}</div>
    </div>
    <div class="stat-card">
        <div class="stat-icon">⚡</div>
        <div class="stat-label">Engine</div>
        <div class="stat-value">Random Forest</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────── Render helpers ─────────────────────────────

def render_symptom_chips(symptoms):
    if not symptoms:
        return
    chips = "".join(
        f"<span class='sym-chip{'  severe' if s in SEVERE_INDICATORS else ''}'>"
        f"{escape(pretty_value(s))}</span>"
        for s in symptoms
    )
    st.markdown(f"<div style='margin-bottom:12px'>{chips}</div>", unsafe_allow_html=True)


def render_pretty_list(items):
    clean = clean_list(items)
    if not clean:
        st.caption("No information available.")
        return
    html = "<div class='pretty-list'>"
    for i, item in enumerate(clean, 1):
        html += f"""
<div class="pretty-item">
    <div class="pretty-num">{i}</div>
    <div>{escape(item)}</div>
</div>"""
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def render_match_summary(ranked_df, model_prediction):
    if ranked_df is None or ranked_df.empty:
        return
    top_rows = ranked_df.head(4)
    html = "<div class='match-grid'>"
    for _, row in top_rows.iterrows():
        disease = escape(str(row["Disease"]))
        score   = float(row["Match Score"])
        matched = row["Matched Symptoms"]
        matched_text = escape(", ".join(pretty_value(s) for s in matched)) if matched else "Model-based match"
        html += f"""
<div class="match-item">
    <div class="match-head">
        <span>{disease}</span>
        <span class="match-score">{score:.1f}%</span>
    </div>
    <div class="match-meta">Matched: {matched_text}</div>
    <div class="match-bar-wrap">
        <div class="match-bar" style="width:{min(max(score,0),100):.1f}%"></div>
    </div>
</div>"""
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)
    if model_prediction != top_rows.iloc[0]["Disease"]:
        st.info("The raw model output was adjusted — another disease matched your symptoms better.")


def render_info_box(title, items):
    st.markdown(f"<div class='info-box'><div class='info-box-title'>{title}</div>",
                unsafe_allow_html=True)
    render_pretty_list(items)
    st.markdown("</div>", unsafe_allow_html=True)


def render_recommendation_result(title, disease_name):
    dis_des, precautions, medications, rec_diet, workout = disease_information(disease_name)
    st.markdown(f"""
<div class="result-card">
    <div class="result-title">{escape(title)}: {escape(disease_name)}</div>
    <div class="result-desc"><strong>About:</strong> {escape(str(dis_des))}</div>
</div>
""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        render_info_box("🛡️ Precautions", precautions)
        render_info_box("💊 Medications", medications)
    with col2:
        render_info_box("🥗 Recommended Diet", rec_diet)
        render_info_box("🏃 Recommended Workout", workout)


# ─────────────────────────── Page: Home ─────────────────────────────────────

def page_home():
    render_hero()
    render_stats()

    st.markdown("""
<div class="sec-title" style="margin-bottom:10px">🚀 What this app does</div>
<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-bottom:22px">
    <div class="stat-card" style="padding:20px">
        <div style="font-size:26px;margin-bottom:9px">🔍</div>
        <div style="font-size:14px;font-weight:700;margin-bottom:5px;font-family:var(--font-body)">Diagnose</div>
        <div style="color:var(--muted);font-size:13px;line-height:1.6">Enter your symptoms and get an AI-powered disease prediction ranked by match confidence.</div>
    </div>
    <div class="stat-card" style="padding:20px">
        <div style="font-size:26px;margin-bottom:9px">🗂</div>
        <div style="font-size:14px;font-weight:700;margin-bottom:5px;font-family:var(--font-body)">Disease Library</div>
        <div style="color:var(--muted);font-size:13px;line-height:1.6">Browse all 40+ diseases by category, search descriptions, precautions, meds, and diets.</div>
    </div>
    <div class="stat-card" style="padding:20px">
        <div style="font-size:26px;margin-bottom:9px">📊</div>
        <div style="font-size:14px;font-weight:700;margin-bottom:5px;font-family:var(--font-body)">Symptom Insights</div>
        <div style="color:var(--muted);font-size:13px;line-height:1.6">Explore symptom severity weights and see which symptoms are linked to the most diseases.</div>
    </div>
</div>
""", unsafe_allow_html=True)

    # Recent history preview
    history = st.session_state.get("history", [])
    if history:
        st.markdown("<div class='sec-title' style='margin-bottom:10px'>🕐 Recent Diagnoses</div>",
                    unsafe_allow_html=True)
        for entry in reversed(history[-4:]):
            sev = entry.get("severity", "Unknown")
            sev_color = {"Mild": "#0f7b55", "Moderate": "#b45309", "Severe": "#ba1a1a"}.get(sev, "#6750a4")
            symptoms = entry.get("symptoms", [])
            disease = entry.get("disease", "Unknown")
            time_str = entry.get("time", "")
            st.markdown(f"""
<div class="dis-row">
    <div>
        <div class="dis-name">🦠 {escape(disease)}</div>
        <div style="color:var(--muted);font-size:11px;margin-top:3px">{escape(time_str)} · {len(symptoms)} symptoms</div>
    </div>
    <span class="track-badge" style="background:{sev_color}22;border:1px solid {sev_color}44;color:{sev_color}">{escape(sev)}</span>
</div>
""", unsafe_allow_html=True)

    # Quick symptom tags
    st.markdown("<div class='sec-title' style='margin:20px 0 10px'>⚡ Common Symptoms</div>",
                unsafe_allow_html=True)
    common = ["fever", "headache", "fatigue", "cough", "nausea", "vomiting",
              "body pain", "dizziness", "skin rash", "chest pain", "breathlessness",
              "loss of appetite", "sweating", "chills", "joint pain"]
    tags = "".join([
        f"<span class='sym-chip' style='cursor:default;margin-bottom:5px'>💊 {escape(pretty_value(s))}</span>"
        for s in common if s in data["symptom_to_column"]
    ])
    st.markdown(f"<div style='display:flex;flex-wrap:wrap;gap:4px'>{tags}</div>", unsafe_allow_html=True)


# ─────────────────────────── Page: Diagnose ─────────────────────────────────

def page_diagnose():
    st.markdown("""
<div class="panel">
    <div class="sec-title">🔍 Predict Disease From Symptoms</div>
    <div class="sec-sub">Select known symptoms from the searchable menu or type symptoms manually with commas. The AI checks your symptoms against disease profiles and model confidence before returning a ranked result.</div>
</div>
""", unsafe_allow_html=True)

    selected_symptoms = st.multiselect(
        "Select known symptoms",
        options=data["symptom_options"],
        placeholder="Search and select symptoms…",
        format_func=pretty_value,
    )

    user_input = st.text_area(
        "Or type symptoms manually (comma-separated)",
        placeholder="Example: headache, constipation, nausea",
        height=90,
    )

    col_btn, col_clr = st.columns([4, 1], vertical_alignment="bottom")
    with col_btn:
        predict_btn = st.button("🔍 Predict Disease", use_container_width=True)
    with col_clr:
        if st.button("🗑 Clear", use_container_width=True):
            for k in ["patient_symptoms", "predicted_disease", "model_prediction", "ranked_diseases"]:
                st.session_state.pop(k, None)
            st.rerun()

    if predict_btn:
        patient_symptoms = normalize_symptoms(user_input, selected_symptoms)
        if not patient_symptoms:
            st.warning("Please select or type at least one valid symptom.")
        else:
            with st.spinner("Analyzing symptoms…"):
                predicted_disease, ranked_df, model_prediction = rank_diseases(patient_symptoms)
                severity_level, _, _ = get_symptom_severity_level(patient_symptoms)

                st.session_state.patient_symptoms    = patient_symptoms
                st.session_state.predicted_disease   = predicted_disease
                st.session_state.model_prediction    = model_prediction
                st.session_state.ranked_diseases     = ranked_df

                # Save to history
                if "history" not in st.session_state:
                    st.session_state.history = []
                st.session_state.history.append({
                    "disease":  predicted_disease,
                    "symptoms": patient_symptoms,
                    "severity": severity_level,
                    "time":     _dt.now().strftime("%b %d, %H:%M"),
                    "ranked_df": ranked_df,
                    "model_pred": model_prediction,
                })

    # Results
    if st.session_state.get("predicted_disease"):
        patient_symptoms  = st.session_state.patient_symptoms
        predicted_disease = st.session_state.predicted_disease
        ranked_diseases   = st.session_state.get("ranked_diseases")
        model_prediction  = st.session_state.get("model_prediction", predicted_disease)

        st.markdown("<div class='sec-title' style='margin-top:20px'>✅ Recognized Symptoms</div>",
                    unsafe_allow_html=True)
        render_symptom_chips(patient_symptoms)

        severity_level, severity_color, severity_note = get_symptom_severity_level(patient_symptoms)
        st.markdown(f"""
<div class="sev-badge" style="background:{severity_color}">
    ⚠️ Severity: {severity_level} &nbsp;·&nbsp; {severity_note}
</div>
""", unsafe_allow_html=True)

        st.markdown("<div class='sec-title'>📊 Top Matches</div>", unsafe_allow_html=True)
        render_match_summary(ranked_diseases, model_prediction)

        # Extra: full ranked table toggle
        with st.expander("📋 View full ranked disease list"):
            if ranked_diseases is not None and not ranked_diseases.empty:
                display_df = ranked_diseases[["Disease", "Match Score", "Symptom Match", "Model Confidence"]].head(15)
                st.dataframe(display_df.reset_index(drop=True), use_container_width=True)

        render_recommendation_result("🩺 Predicted Disease", predicted_disease)

    # ── Search Disease Directly ──
    st.markdown("<hr style='border:none;border-top:1px solid var(--out2);margin:28px 0 20px'>",
                unsafe_allow_html=True)
    st.markdown("""
<div class="panel" style="margin-bottom:14px">
    <div class="sec-title">🗂 Search Disease Recommendations</div>
    <div class="sec-sub">Browse any disease directly to view its full health guidance card.</div>
</div>
""", unsafe_allow_html=True)

    selected_disease = st.selectbox(
        "Select a disease",
        options=[""] + data["disease_names"],
        format_func=lambda v: "Choose a disease…" if v == "" else v,
    )
    if selected_disease:
        render_recommendation_result("📋 Recommendations For", selected_disease)


# ─────────────────────────── Page: Disease Library ───────────────────────────

CATEGORY_ICONS = {
    "Infectious": "🦠", "Liver": "🫀", "Digestive": "🍽", "Metabolic": "⚗️",
    "Cardiac": "❤️", "Skin": "🧴", "Neurological": "🧠",
    "Musculoskeletal": "🦴", "Urinary": "💧", "Respiratory": "🫁",
}

@st.cache_data(show_spinner=False)
def _get_category_overview_html(selected_cat):
    cat_html = "<div class='cat-grid'>"
    for cat, diseases in DISEASE_CATEGORIES.items():
        icon = CATEGORY_ICONS.get(cat, "📁")
        active = "selected" if selected_cat == cat else ""
        cat_html += f"""
<div class="cat-card {active}">
    <div class="cat-icon">{icon}</div>
    <div class="cat-name">{escape(cat)}</div>
    <div class="cat-count">{len(diseases)} diseases</div>
</div>"""
    cat_html += "</div>"
    return cat_html

@st.cache_data(show_spinner=False)
def _get_disease_list_html(diseases_tuple):
    html = ""
    for disease in diseases_tuple:
        cat_tag = next((c for c, ds in DISEASE_CATEGORIES.items() if disease in ds), "Other")
        html += f"""
<div class="dis-row">
    <div>
        <div class="dis-name">🦠 {escape(disease)}</div>
    </div>
    <span class="dis-tag">{CATEGORY_ICONS.get(cat_tag, "📁")} {escape(cat_tag)}</span>
</div>
"""
    return html

def page_library():
    st.markdown("""
<div class="panel">
    <div class="sec-title">🗂 Disease Library</div>
    <div class="sec-sub">Browse all diseases by category. Click any category to filter, then select a disease for its full health card.</div>
</div>
""", unsafe_allow_html=True)

    # Category filter
    selected_cat = st.selectbox(
        "Filter by category",
        ["All"] + list(DISEASE_CATEGORIES.keys()),
        format_func=lambda v: v if v == "All" else f"{CATEGORY_ICONS.get(v, '📁')} {v}",
    )

    st.markdown(_get_category_overview_html(selected_cat), unsafe_allow_html=True)

    if selected_cat == "All":
        filtered = data["disease_names"]
    else:
        cat_diseases_lower = {d.lower() for d in DISEASE_CATEGORIES.get(selected_cat, [])}
        filtered = [d for d in data["disease_names"]
                    if any(c in d.lower() for c in cat_diseases_lower) or d.lower() in cat_diseases_lower]
        if not filtered:
            filtered = data["disease_names"]

    st.markdown(f"<div class='sec-sub' style='margin-bottom:12px'>{len(filtered)} diseases shown</div>",
                unsafe_allow_html=True)

    # Search
    search_q = st.text_input("🔎 Search disease", placeholder="Type disease name…")
    if search_q:
        filtered = [d for d in filtered if search_q.lower() in d.lower()]

    st.markdown(_get_disease_list_html(tuple(filtered)), unsafe_allow_html=True)

    # Detail viewer
    st.markdown("<div style='margin-top:20px'>", unsafe_allow_html=True)
    selected_disease = st.selectbox(
        "Select disease to view full health card",
        [""] + filtered,
        format_func=lambda v: "Choose…" if v == "" else v,
        key="lib_sel",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if selected_disease:
        render_recommendation_result("📋 Disease Profile", selected_disease)


# ─────────────────────────── Page: Symptom Insights ─────────────────────────

@st.cache_data(show_spinner=False)
def _get_severity_bars_html(symptom_rows):
    html = ""
    for sym, w, pct, color in symptom_rows:
        html += f"""
<div class="sw-row">
    <div class="sw-label">{escape(sym[:22])}{'…' if len(sym)>22 else ''}</div>
    <div class="sw-bar-bg"><div class="sw-bar-fill" style="width:{pct}%;background:{color}"></div></div>
    <div class="sw-val">{int(w)}</div>
</div>
"""
    return html


@st.cache_data(show_spinner=False)
def _get_disease_profile_bars_html(disease_rows):
    html = ""
    for disease, cnt, pct in disease_rows:
        html += f"""
<div class="sw-row">
    <div class="sw-label">{escape(disease[:22])}{'…' if len(disease)>22 else ''}</div>
    <div class="sw-bar-bg"><div class="sw-bar-fill" style="width:{pct}%"></div></div>
    <div class="sw-val">{cnt}</div>
</div>
"""
    return html


def page_insights():
    st.markdown("""
<div class="panel">
    <div class="sec-title">📊 Symptom Insights</div>
    <div class="sec-sub">Explore severity weights, symptom-disease associations, and dataset statistics.</div>
</div>
""", unsafe_allow_html=True)

    severity_df = data.get("severity_df")

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("<div class='sec-title' style='font-size:16px;margin-bottom:12px'>⚖️ Top Symptom Severity Weights</div>",
                    unsafe_allow_html=True)
        if severity_df is not None and not severity_df.empty:
            top_sym = severity_df.nlargest(12, severity_df.columns[1])
            max_w = top_sym.iloc[:, 1].max() or 1
            COLORS = ["#7c4dff","#00bcd4","#26a69a","#42a5f5","#ab47bc",
                      "#ef5350","#ff7043","#ffca28","#66bb6a","#26c6da","#ec407a","#8d6e63"]
            rows = tuple(
                (str(r.iloc[0]), int(r.iloc[1]), int(r.iloc[1]/max_w*100), COLORS[i % len(COLORS)])
                for i, r in top_sym.iterrows()
            )
            st.markdown(_get_severity_bars_html(rows), unsafe_allow_html=True)

    with col_r:
        st.markdown("<div class='sec-title' style='font-size:16px;margin-bottom:12px'>🦠 Diseases with Most Symptoms</div>",
                    unsafe_allow_html=True)
        disease_profiles = data["disease_profiles"]
        if disease_profiles:
            ranked_diseases = sorted(disease_profiles.items(), key=lambda x: len(x[1]), reverse=True)[:12]
            max_c = len(ranked_diseases[0][1]) if ranked_diseases else 1
            for disease, profile in ranked_diseases:
                cnt = len(profile)
                pct = int(cnt / max_c * 100)
                st.markdown(f"""
<div class="sw-row">
    <div class="sw-label">{escape(disease[:22])}{'…' if len(disease)>22 else ''}</div>
    <div class="sw-bar-bg"><div class="sw-bar-fill" style="width:{pct}%"></div></div>
    <div class="sw-val">{cnt}</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<hr style='border:none;border-top:1px solid var(--out2);margin:22px 0'>",
                unsafe_allow_html=True)

    # ── Symptom → Disease finder ──
    st.markdown("<div class='sec-title' style='font-size:16px;margin-bottom:8px'>🔗 Symptom → Disease Finder</div>",
                unsafe_allow_html=True)
    st.markdown("<div class='sec-sub' style='margin-bottom:12px'>Pick a symptom to see which diseases are associated with it.</div>",
                unsafe_allow_html=True)

    chosen_sym = st.selectbox("Pick a symptom", [""] + data["symptom_options"],
                              format_func=lambda v: "Choose…" if v == "" else pretty_value(v),
                              key="ins_sym")
    if chosen_sym:
        col = data["symptom_to_column"].get(chosen_sym)
        linked = []
        if col:
            for disease, profile in data["disease_profiles"].items():
                if col in profile:
                    linked.append(disease)
        if linked:
            st.markdown(f"<div class='sec-sub' style='margin-bottom:10px'><strong>{escape(pretty_value(chosen_sym))}</strong> is linked to {len(linked)} disease(s):</div>",
                        unsafe_allow_html=True)
            for d in sorted(linked):
                cat = next((c for c, ds in DISEASE_CATEGORIES.items() if d in ds), "Other")
                st.markdown(f"""
<div class="dis-row">
    <div class="dis-name">🦠 {escape(d)}</div>
    <span class="dis-tag">{CATEGORY_ICONS.get(cat,'📁')} {escape(cat)}</span>
</div>
""", unsafe_allow_html=True)
        else:
            st.info("No diseases linked to this symptom in the dataset.")

    st.markdown("<hr style='border:none;border-top:1px solid var(--out2);margin:22px 0'>",
                unsafe_allow_html=True)

    # ── Stats overview ──
    st.markdown("<div class='sec-title' style='font-size:16px;margin-bottom:12px'>📈 Dataset Overview</div>",
                unsafe_allow_html=True)
    total_sym  = len(data["feature_columns"])
    total_dis  = len(data["disease_names"])
    total_rows = len(data["training"]) if data.get("training") is not None else "N/A"
    total_pairs = sum(len(p) for p in data["disease_profiles"].values())
    st.markdown(f"""
<div class="stat-grid">
    <div class="stat-card">
        <div class="stat-icon">🩺</div>
        <div class="stat-label">Total Symptoms</div>
        <div class="stat-value">{total_sym}</div>
    </div>
    <div class="stat-card">
        <div class="stat-icon">🦠</div>
        <div class="stat-label">Total Diseases</div>
        <div class="stat-value">{total_dis}</div>
    </div>
    <div class="stat-card">
        <div class="stat-icon">📊</div>
        <div class="stat-label">Training Rows</div>
        <div class="stat-value">{total_rows}</div>
    </div>
    <div class="stat-card">
        <div class="stat-icon">🔗</div>
        <div class="stat-label">Symptom-Disease Pairs</div>
        <div class="stat-value">{total_pairs:,}</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────── Page: History ──────────────────────────────────

def page_history():
    st.markdown("""
<div class="panel">
    <div class="sec-title">📋 Diagnosis History</div>
    <div class="sec-sub">All predictions made in this session. Click "View Details" to re-open any diagnosis.</div>
</div>
""", unsafe_allow_html=True)

    history = st.session_state.get("history", [])

    if not history:
        st.markdown("<div style='text-align:center;padding:48px;color:var(--muted);font-size:14px'>No diagnoses yet. Head to the Diagnose page to get started.</div>",
                    unsafe_allow_html=True)
        return

    col_info, col_clr = st.columns([5, 1], vertical_alignment="bottom")
    with col_info:
        st.markdown(f"<div class='sec-sub'>{len(history)} session diagnosis record(s)</div>",
                    unsafe_allow_html=True)
    with col_clr:
        if st.button("🗑 Clear All", use_container_width=True):
            st.session_state.history = []
            st.rerun()

    for i, entry in enumerate(reversed(history)):
        sev_color = {"Mild": "#0f7b55", "Moderate": "#b45309", "Severe": "#ba1a1a"}.get(entry["severity"], "#6750a4")
        col_card, col_btn = st.columns([5, 1], vertical_alignment="center")
        with col_card:
            syms_preview = ", ".join(pretty_value(s) for s in entry["symptoms"][:4])
            if len(entry["symptoms"]) > 4:
                syms_preview += f" +{len(entry['symptoms'])-4} more"
            st.markdown(f"""
<div class="track-card">
    <div class="track-date">🕐 {escape(entry["time"])}</div>
    <div class="track-disease">🦠 {escape(entry["disease"])}</div>
    <div style="color:var(--muted);font-size:12px;margin-bottom:8px">{escape(syms_preview)}</div>
    <span class="track-badge" style="background:{sev_color}22;border:1px solid {sev_color}44;color:{sev_color};font-size:11px;padding:3px 10px;border-radius:999px">{escape(entry["severity"])}</span>
</div>
""", unsafe_allow_html=True)
        with col_btn:
            if st.button("View", key=f"hist_view_{i}"):
                st.session_state.history_view = len(history) - 1 - i

    # Detail panel
    view_idx = st.session_state.get("history_view")
    if view_idx is not None and 0 <= view_idx < len(history):
        entry = history[view_idx]
        st.markdown("<hr style='border:none;border-top:1px solid var(--out2);margin:20px 0'>",
                    unsafe_allow_html=True)
        st.markdown(f"<div class='sec-title'>📋 Detail: {escape(entry['disease'])}</div>",
                    unsafe_allow_html=True)
        render_symptom_chips(entry["symptoms"])
        render_match_summary(entry.get("ranked_df"), entry.get("model_pred", entry["disease"]))
        render_recommendation_result("🩺 Diagnosis", entry["disease"])


# ─────────────────────────── Page: About ────────────────────────────────────

def page_about():
    st.markdown("""
<div class="panel">
    <div class="sec-title">ℹ️ About Early Diagnosis AI</div>
    <div class="sec-sub">How the prediction system works, what data it uses, and important disclaimers.</div>
</div>
""", unsafe_allow_html=True)

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("""
<div class="info-box">
    <div class="info-box-title">🧠 How the AI Works</div>
    <div class="pretty-list">
        <div class="pretty-item"><div class="pretty-num">1</div><div>You select or type symptoms from a known feature set of 132 symptoms.</div></div>
        <div class="pretty-item"><div class="pretty-num">2</div><div>A trained Random Forest model produces class probability scores for 41 diseases.</div></div>
        <div class="pretty-item"><div class="pretty-num">3</div><div>A symptom-profile match score is computed using severity-weighted cosine overlap.</div></div>
        <div class="pretty-item"><div class="pretty-num">4</div><div>Final ranking combines model confidence (22%) and symptom match (78%).</div></div>
        <div class="pretty-item"><div class="pretty-num">5</div><div>The top-ranked disease is returned with precautions, medications, diet, and workouts.</div></div>
    </div>
</div>
""", unsafe_allow_html=True)

        st.markdown("""
<div class="info-box">
    <div class="info-box-title">📂 Data Sources</div>
    <div class="pretty-list">
        <div class="pretty-item"><div class="pretty-num">📊</div><div>Training.csv — 132-symptom feature matrix with disease labels.</div></div>
        <div class="pretty-item"><div class="pretty-num">⚖️</div><div>Symptom-severity.csv — Expert-assigned severity weights per symptom.</div></div>
        <div class="pretty-item"><div class="pretty-num">📝</div><div>description.csv — Disease descriptions from medical knowledge bases.</div></div>
        <div class="pretty-item"><div class="pretty-num">🛡️</div><div>precautions_df.csv — 4 key precautions per disease.</div></div>
        <div class="pretty-item"><div class="pretty-num">💊</div><div>medications.csv, diets.csv, workout_df.csv — Curated guidance per disease.</div></div>
    </div>
</div>
""", unsafe_allow_html=True)

    with col_r:
        st.markdown("""
<div class="info-box">
    <div class="info-box-title">⚠️ Important Disclaimers</div>
    <div class="pretty-list">
        <div class="pretty-item"><div class="pretty-num">🚫</div><div>This app is for <strong>educational and informational purposes only</strong>. It is not a medical device.</div></div>
        <div class="pretty-item"><div class="pretty-num">🚫</div><div>The AI prediction may be incorrect. Never use it as your only basis for health decisions.</div></div>
        <div class="pretty-item"><div class="pretty-num">🚫</div><div>If you experience severe symptoms, contact emergency services or a licensed physician immediately.</div></div>
        <div class="pretty-item"><div class="pretty-num">✅</div><div>Always consult a qualified healthcare professional for diagnosis, treatment, or medication guidance.</div></div>
    </div>
</div>
""", unsafe_allow_html=True)

        st.markdown(f"""
<div class="info-box">
    <div class="info-box-title">📦 Tech Stack</div>
    <div class="pretty-list">
        <div class="pretty-item"><div class="pretty-num">🐍</div><div>Python · Pandas · NumPy · scikit-learn</div></div>
        <div class="pretty-item"><div class="pretty-num">🎨</div><div>Streamlit · Custom Material 3 Expressive CSS</div></div>
        <div class="pretty-item"><div class="pretty-num">🤖</div><div>Random Forest Classifier trained on symptom-disease dataset</div></div>
        <div class="pretty-item"><div class="pretty-num">🔤</div><div>thefuzz for fuzzy symptom spelling correction</div></div>
        <div class="pretty-item"><div class="pretty-num">👨‍💻</div><div>Built by <strong>{escape(CREATOR_NAME)}</strong></div></div>
    </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="sb-note" style="margin-top:4px;border-radius:var(--r-md);padding:16px">
    <strong>Severe symptoms requiring immediate medical attention:</strong><br>
    Chest pain · Breathlessness · Blood in sputum · Stomach bleeding · Coma · Fast heart rate ·
    Altered sensorium · Weakness of one body side · Yellowing of eyes · Acute liver failure
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────── Footer ─────────────────────────────────────

def render_footer():
    _year = _dt.now().strftime("%Y")
    footer_logo_html = '<div class="md-footer-logo" role="img" aria-label="Early Diagnosis AI logo">🩺</div>'
    st.markdown(
        f'<div class="md-footer">'
        f'<div class="md-footer-top">'
        f'<div class="md-footer-brand">' + footer_logo_html +
        f'<div><div class="md-footer-brand-name">Early Diagnosis AI</div>'
        f'<div class="md-footer-brand-sub">by {escape(CREATOR_NAME)}</div></div></div>'
        f'</div>'
        f'<div class="md-footer-links">'
        f'<a class="md-footer-link" href="https://github.com/YatinSharma1303/" target="_blank">🐙 GitHub</a>'
        f'<a class="md-footer-link" href="https://www.linkedin.com/in/yatin-sharma-793042372/" target="_blank">💼 LinkedIn</a>'
        f'</div>'
        f'<div class="md-footer-meta">'
        f'<span class="md-footer-version">Early Diagnosis AI v2.0</span>'
        f' &nbsp;·&nbsp; Made with <span class="md-footer-heart">❤️</span> by <strong>{escape(CREATOR_NAME)}</strong>'
        f' &nbsp;·&nbsp; © {_year}'
        f'</div>'
        f'<div class="md-footer-disclaimer">⚕️ <strong>Medical Disclaimer:</strong> '
        f'This application is for educational purposes only and does not constitute medical advice, diagnosis, or treatment. '
        f'Always consult a qualified healthcare professional regarding any health concerns.</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────── Main ───────────────────────────────────────

render_styles()

# Session state defaults
for k, v in [("page", "Home"), ("history", []), ("history_view", None),
             ("patient_symptoms", None), ("predicted_disease", None),
             ("model_prediction", None), ("ranked_diseases", None)]:
    if k not in st.session_state:
        st.session_state[k] = v

_widget_nav_pages = {"Home", "Diagnose", "Library", "Insights", "History", "About"}
try:
    _widget_page = st.query_params.get("disease_page")
except Exception:
    try:
        _widget_page = st.experimental_get_query_params().get("disease_page")
    except Exception:
        _widget_page = None
if isinstance(_widget_page, list):
    _widget_page = _widget_page[0] if _widget_page else None
if _widget_page in _widget_nav_pages:
    st.session_state.page = _widget_page

# Sidebar
render_sidebar(st.session_state.page)

# Nav pills
pages = {
    "🏠 Home":     "Home",
    "🔍 Diagnose": "Diagnose",
    "🗂 Library":  "Library",
    "📊 Insights": "Insights",
    "📋 History":  "History",
    "ℹ️ About":    "About",
}
labels      = list(pages.keys())
active_label = next(label for label, key in pages.items() if key == st.session_state.page)
st.markdown("<div class='main-nav-shell'>", unsafe_allow_html=True)
selected_label = st.radio(
    "Main navigation",
    labels,
    index=labels.index(active_label),
    horizontal=True,
    label_visibility="collapsed",
    key="main_nav_radio",
)
st.markdown("</div>", unsafe_allow_html=True)
if pages[selected_label] != st.session_state.page:
    st.session_state.page = pages[selected_label]
    st.rerun()

st.markdown("<hr style='border:none;border-top:1px solid var(--out2);margin:6px 0 18px'>",
            unsafe_allow_html=True)

# Route
page = st.session_state.page
if   page == "Home":     page_home()
elif page == "Diagnose": page_diagnose()
elif page == "Library":  page_library()
elif page == "Insights": page_insights()
elif page == "History":  page_history()
elif page == "About":    page_about()

render_widget_nav_bridge()



# ───────────────────────── Floating Quick Widget ───────────────────────────

def render_floating_widget():
    components.html(
        """
<script>
(function() {
    const parentDoc = window.parent.document;
    const parentWin = window.parent;
    const widgetId = "mm-widget-root-disease-ai";
    const styleId = widgetId + "-style";
    parentDoc.querySelectorAll('[id^="mm-widget-root"], #mm-widget-style').forEach((element) => element.remove());

    const style = parentDoc.createElement("style");
    style.id = styleId;
    style.textContent = `
        #${widgetId} {
            position: fixed;
            z-index: 2147483000;
            right: 24px;
            bottom: 26px;
            font-family: "Space Grotesk", Inter, system-ui, sans-serif;
            touch-action: none;
        }
        #${widgetId}.mm-dragging, #${widgetId}.mm-dragging * {
            cursor: grabbing !important;
            user-select: none !important;
        }
        #${widgetId} .mm-widget-panel {
            position: absolute;
            right: 0;
            bottom: 74px;
            width: min(340px, calc(100vw - 28px));
            max-height: min(520px, calc(100vh - 28px));
            overflow-y: auto !important;
            overflow-x: hidden;
            overscroll-behavior: contain;
            touch-action: pan-y;
            -webkit-overflow-scrolling: touch;
            padding: 12px;
            border-radius: 24px;
            border: 1px solid rgba(255,255,255,.16);
            background:
                radial-gradient(circle at 12% 0%, rgba(124,77,255,.28), transparent 38%),
                radial-gradient(circle at 92% 18%, rgba(0,188,212,.20), transparent 42%),
                rgba(13, 13, 20, .94);
            box-shadow: 0 18px 60px rgba(0,0,0,.38), 0 0 0 1px rgba(124,77,255,.10);
            backdrop-filter: blur(18px);
            transform: translateY(12px) scale(.96);
            opacity: 0;
            pointer-events: none;
            transition: opacity 320ms cubic-bezier(0.34,1.56,0.64,1), transform 320ms cubic-bezier(0.34,1.56,0.64,1);
        }
        #${widgetId} .mm-widget-panel::-webkit-scrollbar { width: 7px; }
        #${widgetId} .mm-widget-panel::-webkit-scrollbar-track { background: transparent; }
        #${widgetId} .mm-widget-panel::-webkit-scrollbar-thumb {
            background: rgba(255,255,255,.28);
            border-radius: 999px;
        }
        #${widgetId}.mm-open .mm-widget-panel {
            opacity: 1;
            pointer-events: auto;
            transform: translateY(0) scale(1);
        }
        #${widgetId}.mm-left .mm-widget-panel { left: 0; right: auto; }
        #${widgetId}.mm-down .mm-widget-panel { top: 74px; bottom: auto; transform-origin: top right; }
        #${widgetId}.mm-left.mm-down .mm-widget-panel { transform-origin: top left; }
        #${widgetId}.mm-up .mm-widget-panel { bottom: 74px; top: auto; transform-origin: bottom right; }
        #${widgetId}.mm-left.mm-up .mm-widget-panel { transform-origin: bottom left; }

        @keyframes mm-head-in {
            0%   { opacity: 0; transform: translateY(-6px) scale(.97); }
            100% { opacity: 1; transform: translateY(0)    scale(1);   }
        }
        @keyframes mm-drag-pulse {
            0%,100% { box-shadow: 0 0 0 0   rgba(var(--mm-fab-primary-rgb),.45); }
            55%     { box-shadow: 0 0 0 5px rgba(var(--mm-fab-primary-rgb),.0);  }
        }
            100% { background-position:  200% center; }
        }
        #${widgetId} .mm-widget-head {
            position: sticky;
            top: -12px;
            z-index: 2;
            display: flex;
            flex-direction: column;
            gap: 0;
            padding: 14px 14px 12px;
            margin: -12px -12px 10px;
            border-radius: 20px 20px 0 0;
            background:
                linear-gradient(135deg,
                    rgba(var(--mm-fab-primary-rgb),.18) 0%,
                    rgba(var(--mm-fab-secondary-rgb),.10) 55%,
                    rgba(13,13,20,.96) 100%);
            border-bottom: 1px solid rgba(var(--mm-fab-primary-rgb),.18);
            backdrop-filter: blur(20px) saturate(1.4);
            box-shadow:
                0 4px 24px rgba(var(--mm-fab-primary-rgb),.10),
                inset 0 1px 0 rgba(255,255,255,.10);
            animation: mm-head-in 340ms cubic-bezier(0.34,1.56,0.64,1) both;
            overflow: hidden;
        }
        /* subtle gloss streak across header */
        #${widgetId} .mm-widget-head::before {
            content: "";
            position: absolute;
            inset: 0;
            background: radial-gradient(ellipse at 18% 0%, rgba(255,255,255,.13), transparent 60%);
            pointer-events: none;
        }
        /* bottom edge glow line */
        #${widgetId} .mm-widget-head::after {
            content: "";
            position: absolute;
            bottom: 0; left: 12%; right: 12%;
            height: 1px;
            background: linear-gradient(90deg,
                transparent,
                rgba(var(--mm-fab-primary-rgb),.55) 40%,
                rgba(var(--mm-fab-secondary-rgb),.45) 60%,
                transparent);
        }
        #${widgetId} .mm-widget-head-top {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 8px;
            margin-bottom: 7px;
        }
        #${widgetId} .mm-widget-draghint {
            display: inline-flex;
            align-items: center;
            gap: 5px;
            padding: 4px 10px 4px 7px;
            border-radius: 999px;
            background: rgba(var(--mm-fab-primary-rgb),.18);
            border: 1px solid rgba(var(--mm-fab-primary-rgb),.36);
            color: rgba(220,210,255,.90);
            font-size: 9.5px;
            font-weight: 900;
            letter-spacing: .10em;
            text-transform: uppercase;
            animation: mm-drag-pulse 2.6s ease-in-out infinite 1.2s;
            transition: background 160ms ease, border-color 160ms ease;
            cursor: grab;
        }
        #${widgetId} .mm-widget-draghint:hover {
            background: rgba(var(--mm-fab-primary-rgb),.28);
            border-color: rgba(var(--mm-fab-primary-rgb),.55);
        }
        #${widgetId} .mm-widget-draghint-dot {
            width: 6px; height: 6px;
            border-radius: 50%;
            background: rgba(var(--mm-fab-primary-rgb),1);
            box-shadow: 0 0 6px 2px rgba(var(--mm-fab-primary-rgb),.60);
            flex-shrink: 0;
        }
        #${widgetId} .mm-widget-head-right {
            display: flex;
            align-items: center;
            gap: 6px;
        }
        #${widgetId} .mm-widget-head-icon {
            width: 30px; height: 30px;
            border-radius: 12px;
            display: grid;
            place-items: center;
            background: linear-gradient(135deg,
                rgba(var(--mm-fab-primary-rgb),.30),
                rgba(var(--mm-fab-secondary-rgb),.20));
            border: 1px solid rgba(var(--mm-fab-primary-rgb),.28);
            box-shadow:
                0 2px 10px rgba(var(--mm-fab-primary-rgb),.22),
                inset 0 1px 0 rgba(255,255,255,.14);
            font-size: 14px;
            flex-shrink: 0;
        }
        #${widgetId} .mm-widget-title {
            font-size: 13.5px;
            font-weight: 900;
            line-height: 1.15;
            letter-spacing: .01em;
            color: #f0ebff;
            text-shadow:
                0 0 18px rgba(var(--mm-fab-primary-rgb),.55),
                0 1px 0 rgba(0,0,0,.18);
        }
        #${widgetId} .mm-widget-sub {
            margin-top: 1px;
            font-size: 10.5px;
            font-weight: 700;
            color: rgba(200,195,230,.62);
            letter-spacing: .02em;
            line-height: 1.3;
        }
        /* divider below title row */
        #${widgetId} .mm-widget-head-divider {
            height: 1px;
            margin: 0 -2px;
            background: linear-gradient(90deg,
                transparent 0%,
                rgba(var(--mm-fab-primary-rgb),.22) 30%,
                rgba(var(--mm-fab-secondary-rgb),.18) 70%,
                transparent 100%);
            border-radius: 999px;
        }

                #${widgetId} .mm-widget-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
            padding-bottom: 2px;
        }
        #${widgetId} .mm-tool {
            min-height: 68px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            gap: 7px;
            padding: 11px;
            border-radius: 18px;
            border: 1px solid rgba(255,255,255,.11);
            background: rgba(255,255,255,.07);
            color: #f7f3ff;
            cursor: pointer;
            touch-action: manipulation;
            text-align: left;
            transition: transform 150ms ease, background 150ms ease, border-color 150ms ease;
        }
        #${widgetId} .mm-tool:hover {
            transform: translateY(-2px);
            background: rgba(124,77,255,.22);
            border-color: rgba(124,77,255,.42);
        }
        #${widgetId} .mm-tool-code {
            width: 34px;
            height: 28px;
            display: grid;
            place-items: center;
            border-radius: 12px;
            background: linear-gradient(135deg,#7c4dff,#00bcd4);
            color: #fff;
            font-size: 11px;
            font-weight: 1000;
            box-shadow: 0 8px 18px rgba(124,77,255,.28);
        }
        #${widgetId} .mm-tool-label { font-size: 12px; line-height: 1.2; font-weight: 900; }
        #${widgetId} .mm-tool-note { color: rgba(240,238,255,.60); font-size: 10px; line-height: 1.2; font-weight: 700; }
        /* ── M3 EXPRESSIVE FAB ─────────────────────────────────────── */
        @keyframes mm-pulse-ring {
            0%   { transform: scale(1);   opacity: .55; }
            60%  { transform: scale(1.55); opacity: .18; }
            100% { transform: scale(1.55); opacity: 0;   }
        }
        @keyframes mm-breathe {
            0%,100% { transform: scale(1);    opacity: .38; }
            50%      { transform: scale(1.32); opacity: .15; }
        }
        @keyframes mm-shimmer {
            0%   { transform: translateX(-130%) skewX(-18deg); }
            100% { transform: translateX(230%)  skewX(-18deg); }
        }
        @keyframes mm-orbit {
            from { transform: rotate(0deg)   translateX(34px) rotate(0deg); }
            to   { transform: rotate(360deg) translateX(34px) rotate(-360deg); }
        }
        @keyframes mm-orbit-rev {
            from { transform: rotate(0deg)   translateX(28px) rotate(0deg); }
            to   { transform: rotate(-360deg) translateX(28px) rotate(360deg); }
        }
        @keyframes mm-orbit-mid {
            from { transform: rotate(120deg)  translateX(31px) rotate(-120deg); }
            to   { transform: rotate(480deg)  translateX(31px) rotate(-480deg); }
        }
        @keyframes mm-ripple-out {
            0%   { transform: scale(0);   opacity: .55; }
            80%  { transform: scale(2.8); opacity: .12; }
            100% { transform: scale(2.8); opacity: 0;   }
        }
        @keyframes mm-dot-appear {
            0%  { transform: scale(0) translateX(0); opacity: 0; }
            50% { opacity: 1; }
            100%{ transform: scale(1); opacity: 1; }
        }
        @keyframes mm-grid-dot-in {
            0%   { transform: scale(0) rotate(-45deg); opacity: 0; }
            60%  { transform: scale(1.25) rotate(8deg); opacity: 1; }
            100% { transform: scale(1) rotate(0deg); opacity: 1; }
        }
        @keyframes mm-x-in {
            0%   { transform: translate(-50%,-50%) rotate(0deg)   scale(0); opacity: 0; }
            60%  { transform: translate(-50%,-50%) rotate(52deg)  scale(1.12); opacity: 1; }
            100% { transform: translate(-50%,-50%) rotate(45deg)  scale(1); opacity: 1; }
        }
        @keyframes mm-x-in2 {
            0%   { transform: translate(-50%,-50%) rotate(0deg)    scale(0); opacity: 0; }
            60%  { transform: translate(-50%,-50%) rotate(-52deg)  scale(1.12); opacity: 1; }
            100% { transform: translate(-50%,-50%) rotate(-45deg)  scale(1); opacity: 1; }
        }

        #${widgetId} .mm-fab {
            --mm-fab-primary-rgb: var(--md-primary-rgb, 103,80,164);
            --mm-fab-secondary-rgb: var(--md-secondary-rgb, 0,106,106);
            --mm-fab-size: clamp(58px, 5.8vw, 66px);
            width:  var(--mm-fab-size);
            height: var(--mm-fab-size);
            position: relative;
            display: grid;
            place-items: center;
            isolation: isolate;
            border-radius: 24px;
            border: 1.5px solid rgba(255,255,255,.20);
            background:
                linear-gradient(145deg,
                    rgba(var(--mm-fab-primary-rgb),.36) 0%,
                    rgba(var(--mm-fab-secondary-rgb),.22) 60%,
                    rgba(20,18,34,.92) 100%);
            color: #fff;
            box-shadow:
                0 0 0 0 rgba(var(--mm-fab-primary-rgb),.50),
                0 16px 38px rgba(var(--mm-fab-primary-rgb),.28),
                0 6px 18px rgba(0,0,0,.26),
                inset 0 1.5px 0 rgba(255,255,255,.22),
                inset 0 -1px 0 rgba(0,0,0,.18);
            cursor: grab;
            transition:
                border-radius   380ms cubic-bezier(0.34,1.56,0.64,1),
                box-shadow      280ms cubic-bezier(0.2,0,0,1),
                transform       280ms cubic-bezier(0.34,1.56,0.64,1),
                background      280ms cubic-bezier(0.2,0,0,1);
            user-select: none;
            overflow: visible;
        }

        /* Breathing pulse ring */
        #${widgetId} .mm-fab-pulse {
            position: absolute;
            inset: -6px;
            border-radius: inherit;
            border: 2px solid rgba(var(--mm-fab-primary-rgb), .60);
            pointer-events: none;
            animation: mm-breathe 2.8s ease-in-out infinite;
            border-radius: 30px;
            transition: border-radius 380ms cubic-bezier(0.34,1.56,0.64,1);
        }
        #${widgetId} .mm-fab-pulse2 {
            position: absolute;
            inset: -12px;
            border-radius: 36px;
            border: 1.5px solid rgba(var(--mm-fab-primary-rgb), .35);
            pointer-events: none;
            animation: mm-breathe 2.8s ease-in-out infinite .6s;
            transition: border-radius 380ms cubic-bezier(0.34,1.56,0.64,1);
        }

        /* Ripple layer (triggered on click via JS class) */
        #${widgetId} .mm-fab-ripple {
            position: absolute;
            inset: 0;
            border-radius: inherit;
            pointer-events: none;
            overflow: hidden;
        }
        #${widgetId} .mm-fab-ripple::after {
            content: "";
            position: absolute;
            inset: 0;
            border-radius: 50%;
            background: rgba(var(--mm-fab-primary-rgb), .45);
            transform: scale(0);
            opacity: 0;
            transition: none;
        }
        #${widgetId}.mm-rippling .mm-fab-ripple::after {
            animation: mm-ripple-out 520ms cubic-bezier(0.2,0,0,1) forwards;
        }

        /* Shimmer sweep */
        #${widgetId} .mm-fab-shimmer {
            position: absolute;
            inset: 0;
            border-radius: inherit;
            overflow: hidden;
            pointer-events: none;
        }
        #${widgetId} .mm-fab-shimmer::before {
            content: "";
            position: absolute;
            top: -20%;
            left: -20%;
            width: 50%;
            height: 140%;
            background: linear-gradient(105deg,
                transparent 30%,
                rgba(255,255,255,.28) 50%,
                transparent 70%);
            animation: mm-shimmer 3.2s cubic-bezier(0.4,0,0.6,1) infinite 1.1s;
        }

        /* Surface gloss */
        #${widgetId} .mm-fab::before {
            content: "";
            position: absolute;
            inset: 0;
            border-radius: inherit;
            background:
                radial-gradient(circle at 28% 20%, rgba(255,255,255,.28), transparent 44%),
                radial-gradient(circle at 76% 82%, rgba(var(--mm-fab-secondary-rgb),.22), transparent 40%);
            pointer-events: none;
            z-index: 0;
        }

        /* Orbital particle dots */
        #${widgetId} .mm-fab-orbit-wrap {
            position: absolute;
            inset: 0;
            pointer-events: none;
            display: grid;
            place-items: center;
            opacity: 0;
            transition: opacity 320ms cubic-bezier(0.2,0,0,1);
        }
        #${widgetId}.mm-open .mm-fab-orbit-wrap { opacity: 1; }
        #${widgetId} .mm-fab-dot {
            position: absolute;
            width: 6px; height: 6px;
            border-radius: 50%;
        }
        #${widgetId} .mm-fab-dot:nth-child(1) {
            background: rgba(var(--mm-fab-primary-rgb),1);
            box-shadow: 0 0 8px 2px rgba(var(--mm-fab-primary-rgb),.70);
            animation: mm-orbit 2.6s linear infinite;
        }
        #${widgetId} .mm-fab-dot:nth-child(2) {
            background: rgba(var(--mm-fab-secondary-rgb),1);
            box-shadow: 0 0 8px 2px rgba(var(--mm-fab-secondary-rgb),.70);
            width: 5px; height: 5px;
            animation: mm-orbit-rev 3.4s linear infinite;
        }
        #${widgetId} .mm-fab-dot:nth-child(3) {
            background: rgba(255,255,255,.90);
            box-shadow: 0 0 6px 1px rgba(255,255,255,.55);
            width: 4px; height: 4px;
            animation: mm-orbit-mid 4.1s linear infinite;
        }

        /* Hover / open states */
        #${widgetId} .mm-fab:hover {
            transform: translateY(-3px) scale(1.045);
            box-shadow:
                0 0 0 6px rgba(var(--mm-fab-primary-rgb),.14),
                0 20px 44px rgba(var(--mm-fab-primary-rgb),.34),
                0 8px 22px rgba(0,0,0,.28),
                inset 0 1.5px 0 rgba(255,255,255,.26);
        }
        #${widgetId}.mm-open .mm-fab {
            border-radius: 50%;
            background:
                linear-gradient(145deg,
                    rgba(var(--mm-fab-primary-rgb),.44) 0%,
                    rgba(var(--mm-fab-secondary-rgb),.28) 60%,
                    rgba(12,10,22,.94) 100%);
            box-shadow:
                0 0 0 10px rgba(var(--mm-fab-primary-rgb),.10),
                0 18px 44px rgba(var(--mm-fab-primary-rgb),.32),
                0 6px 18px rgba(0,0,0,.28),
                inset 0 1.5px 0 rgba(255,255,255,.22);
        }
        #${widgetId}.mm-open .mm-fab-pulse  { border-radius: 50%; inset: -8px; }
        #${widgetId}.mm-open .mm-fab-pulse2 { border-radius: 50%; inset: -16px; }

        /* ── GRID → X MORPHING ICON ─────────────────────── */
        #${widgetId} .mm-fab-icon {
            position: relative;
            z-index: 2;
            width: 28px;
            height: 28px;
            display: grid;
            grid-template-columns: 1fr 1fr;
            grid-template-rows: 1fr 1fr;
            gap: 4.5px;
            padding: 0;
        }
        #${widgetId} .mm-fab-icon .mm-gd {
            border-radius: 4px;
            background: rgba(255,255,255,.92);
            box-shadow: 0 1px 6px rgba(255,255,255,.20);
            transition:
                transform        380ms cubic-bezier(0.34,1.56,0.64,1),
                border-radius    380ms cubic-bezier(0.34,1.56,0.64,1),
                opacity          260ms cubic-bezier(0.2,0,0,1),
                background       260ms cubic-bezier(0.2,0,0,1),
                width            380ms cubic-bezier(0.34,1.56,0.64,1),
                height           380ms cubic-bezier(0.34,1.56,0.64,1);
        }
        /* dot 1 top-left */
        #${widgetId} .mm-fab-icon .mm-gd:nth-child(1) { background: rgba(var(--mm-fab-primary-rgb),1); box-shadow: 0 0 6px 1px rgba(var(--mm-fab-primary-rgb),.55); }
        /* dot 2 top-right */
        #${widgetId} .mm-fab-icon .mm-gd:nth-child(2) { background: rgba(255,255,255,.90); }
        /* dot 3 bottom-left */
        #${widgetId} .mm-fab-icon .mm-gd:nth-child(3) { background: rgba(255,255,255,.90); }
        /* dot 4 bottom-right */
        #${widgetId} .mm-fab-icon .mm-gd:nth-child(4) { background: rgba(var(--mm-fab-secondary-rgb),1); box-shadow: 0 0 6px 1px rgba(var(--mm-fab-secondary-rgb),.55); }

        /* OPEN: morph to X using two bars via pseudo — hide dots, show X */
        #${widgetId}.mm-open .mm-fab-icon .mm-gd {
            opacity: 0;
            transform: scale(.35) rotate(90deg);
        }
        /* X bar 1 */
        #${widgetId} .mm-fab-icon::before,
        #${widgetId} .mm-fab-icon::after {
            content: "";
            position: absolute;
            left: 50%; top: 50%;
            width: 22px; height: 3px;
            border-radius: 999px;
            background: rgba(255,255,255,.96);
            box-shadow: 0 1px 8px rgba(255,255,255,.24);
            opacity: 0;
            pointer-events: none;
            transition:
                opacity  260ms cubic-bezier(0.2,0,0,1),
                transform 400ms cubic-bezier(0.34,1.56,0.64,1);
        }
        #${widgetId} .mm-fab-icon::before {
            transform: translate(-50%,-50%) rotate(0deg) scale(.5);
        }
        #${widgetId} .mm-fab-icon::after {
            transform: translate(-50%,-50%) rotate(0deg) scale(.5);
        }
        #${widgetId}.mm-open .mm-fab-icon::before {
            opacity: 1;
            animation: mm-x-in  380ms cubic-bezier(0.34,1.56,0.64,1) forwards;
            transform: translate(-50%,-50%) rotate(45deg) scale(1);
        }
        #${widgetId}.mm-open .mm-fab-icon::after {
            opacity: 1;
            animation: mm-x-in2 380ms cubic-bezier(0.34,1.56,0.64,1) forwards;
            transform: translate(-50%,-50%) rotate(-45deg) scale(1);
        }

        @media (max-width: 620px) {
            #${widgetId} .mm-fab {
                width: 58px; height: 58px;
                border-radius: 22px;
            }
            #${widgetId} .mm-fab-pulse  { inset: -5px; }
            #${widgetId} .mm-fab-pulse2 { inset: -10px; }
        }

        #${widgetId} .mm-toast {
            position: absolute;
            right: 0;
            bottom: -36px;
            min-width: 190px;
            padding: 8px 11px;
            border-radius: 999px;
            background: rgba(15,10,28,.92);
            color: rgba(240,238,255,.82);
            border: 1px solid rgba(255,255,255,.12);
            box-shadow: 0 8px 24px rgba(0,0,0,.25);
            font-size: 11px;
            font-weight: 800;
            text-align: center;
            opacity: 0;
            pointer-events: none;
            transform: translateY(-4px);
            transition: opacity 160ms ease, transform 160ms ease;
        }
        #${widgetId} .mm-toast.mm-show { opacity: 1; transform: translateY(0); }
        @media (max-width: 620px) {
            #${widgetId} { right: 14px; bottom: 18px; }
            #${widgetId} .mm-widget-panel { width: min(310px, calc(100vw - 24px)); }
            #${widgetId} .mm-widget-grid { grid-template-columns: 1fr; }
            #${widgetId} .mm-tool { min-height: 58px; }
        }

        /* ── Light mode overrides for widget ───────────────────────── */
        [data-theme="light"] #${widgetId} .mm-widget-panel {
            background:
                radial-gradient(circle at 12% 0%, rgba(103,80,164,.12), transparent 38%),
                radial-gradient(circle at 92% 18%, rgba(0,106,106,.08), transparent 42%),
                rgba(247,244,255,.98) !important;
            border-color: rgba(103,80,164,.24) !important;
            box-shadow: 0 18px 60px rgba(103,80,164,.16), 0 0 0 1px rgba(103,80,164,.12) !important;
        }
        [data-theme="light"] #${widgetId} .mm-widget-panel::-webkit-scrollbar-thumb {
            background: rgba(103,80,164,.30) !important;
        }
        [data-theme="light"] #${widgetId} .mm-widget-head {
            background:
                linear-gradient(135deg,
                    rgba(103,80,164,.14) 0%,
                    rgba(0,106,106,.08) 55%,
                    rgba(247,244,255,.99) 100%) !important;
            border-bottom-color: rgba(103,80,164,.22) !important;
            box-shadow: 0 4px 24px rgba(103,80,164,.10), inset 0 1px 0 rgba(255,255,255,.90) !important;
        }
        [data-theme="light"] #${widgetId} .mm-widget-title {
            color: #1a1530 !important;
            text-shadow: none !important;
        }
        [data-theme="light"] #${widgetId} .mm-widget-sub { color: #6a5a8a !important; }
        [data-theme="light"] #${widgetId} .mm-widget-head-icon {
            background: linear-gradient(135deg, rgba(103,80,164,.20), rgba(0,106,106,.14)) !important;
            border-color: rgba(103,80,164,.32) !important;
            box-shadow: 0 2px 10px rgba(103,80,164,.18) !important;
        }
        [data-theme="light"] #${widgetId} .mm-widget-draghint {
            background: rgba(103,80,164,.12) !important;
            border-color: rgba(103,80,164,.30) !important;
            color: #3a2a7d !important;
        }
        [data-theme="light"] #${widgetId} .mm-widget-draghint:hover {
            background: rgba(103,80,164,.20) !important;
            border-color: rgba(103,80,164,.48) !important;
        }
        [data-theme="light"] #${widgetId} .mm-widget-draghint-dot {
            background: rgba(103,80,164,1) !important;
            box-shadow: 0 0 6px 2px rgba(103,80,164,.45) !important;
        }
        [data-theme="light"] #${widgetId} .mm-widget-head-divider {
            background: linear-gradient(90deg,
                transparent,
                rgba(103,80,164,.38) 40%,
                rgba(0,106,106,.30) 60%,
                transparent) !important;
        }
        /* Tool tiles */
        [data-theme="light"] #${widgetId} .mm-tool {
            background: rgba(255,255,255,.88) !important;
            border-color: rgba(103,80,164,.20) !important;
            color: #1a1530 !important;
            box-shadow: 0 1px 6px rgba(103,80,164,.08) !important;
        }
        [data-theme="light"] #${widgetId} .mm-tool:hover {
            background: rgba(103,80,164,.10) !important;
            border-color: rgba(103,80,164,.40) !important;
            box-shadow: 0 4px 14px rgba(103,80,164,.16) !important;
        }
        /* Tool code badge */
        [data-theme="light"] #${widgetId} .mm-tool-code {
            background: linear-gradient(135deg,#6750a4,#006a6a) !important;
            color: #fff !important;
            box-shadow: 0 4px 12px rgba(103,80,164,.28) !important;
        }
        [data-theme="light"] #${widgetId} .mm-tool-label { color: #1a1530 !important; }
        [data-theme="light"] #${widgetId} .mm-tool-note  { color: #6a5a8a !important; }
        /* Toast */
        [data-theme="light"] #${widgetId} .mm-toast {
            background: rgba(247,244,255,.98) !important;
            color: #2a1a5d !important;
            border-color: rgba(103,80,164,.26) !important;
            box-shadow: 0 8px 24px rgba(103,80,164,.18) !important;
        }
        /* FAB button in light mode */
        [data-theme="light"] #${widgetId} .mm-fab {
            background:
                linear-gradient(145deg,
                    rgba(103,80,164,.55) 0%,
                    rgba(0,106,106,.38) 60%,
                    rgba(80,60,140,.92) 100%) !important;
            border-color: rgba(103,80,164,.55) !important;
            box-shadow:
                0 0 0 0 rgba(103,80,164,.50),
                0 12px 32px rgba(103,80,164,.32),
                0 4px 14px rgba(0,0,0,.12),
                inset 0 1.5px 0 rgba(255,255,255,.40) !important;
        }
        [data-theme="light"] #${widgetId} .mm-fab:hover {
            box-shadow:
                0 0 0 6px rgba(103,80,164,.14),
                0 16px 40px rgba(103,80,164,.38),
                0 6px 18px rgba(0,0,0,.14),
                inset 0 1.5px 0 rgba(255,255,255,.45) !important;
        }
        [data-theme="light"] #${widgetId}.mm-open .mm-fab {
            background:
                linear-gradient(145deg,
                    rgba(103,80,164,.65) 0%,
                    rgba(0,106,106,.45) 60%,
                    rgba(60,40,120,.95) 100%) !important;
        }
        [data-theme="light"] #${widgetId} .mm-fab-pulse {
            border-color: rgba(103,80,164,.55) !important;
        }
        [data-theme="light"] #${widgetId} .mm-fab-pulse2 {
            border-color: rgba(103,80,164,.35) !important;
        }
    `;
    parentDoc.head.appendChild(style);

    const root = parentDoc.createElement("div");
    root.id = widgetId;
    const tools = [
        ['Home', 'HM', 'Home', 'Overview', 'nav', 'What this app does'],
        ['Diagnose', 'DX', 'Diagnose', 'Predict disease', 'nav', 'Predict Disease From Symptoms'],
        ['Library', 'LB', 'Library', 'Disease info', 'nav', 'Disease Library'],
        ['Insights', 'IN', 'Insights', 'Stats and patterns', 'nav', 'Symptom Insights'],
        ['History', 'HS', 'History', 'Past checks', 'nav', 'Diagnosis History'],
        ['About', 'AB', 'About', 'Project info', 'nav', 'About Early Diagnosis AI'],
        ['Diagnose', 'FS', 'Focus Input', 'Jump to symptom field', 'input', ''],
        ['top', 'UP', 'Back to top', 'Scroll upward', 'top', '']
    ];
    root.innerHTML = `
        <div class="mm-widget-panel" role="menu" aria-label="Disease AI quick actions">
            <div class="mm-widget-head">
                <div class="mm-widget-head-top">
                    <div style="display:flex;align-items:center;gap:9px;">
                        <div class="mm-widget-head-icon">🩺</div>
                        <div style="display:flex;flex-direction:column;gap:1px;">
                            <div class="mm-widget-title">Disease AI tools</div>
                            <div class="mm-widget-sub">Page actions and navigation</div>
                        </div>
                    </div>
                    <div class="mm-widget-draghint">
                        <div class="mm-widget-draghint-dot"></div>
                        DRAG
                    </div>
                </div>
                <div class="mm-widget-head-divider"></div>
            </div>
            <div class="mm-widget-grid">
                ${tools.map(([target, code, label, note, kind, scrollTarget]) => `
                    <button class="mm-tool" data-target="${target}" data-kind="${kind}" data-scroll="${scrollTarget || ''}" type="button">
                        <span class="mm-tool-code">${code}</span>
                        <span class="mm-tool-label">${label}</span>
                        <span class="mm-tool-note">${note}</span>
                    </button>
                `).join("")}
            </div>
        </div>
        <div class="mm-fab" role="button" tabindex="0" aria-label="Open quick tools">
            <div class="mm-fab-pulse"></div>
            <div class="mm-fab-pulse2"></div>
            <div class="mm-fab-shimmer"></div>
            <div class="mm-fab-ripple"></div>
            <div class="mm-fab-orbit-wrap">
                <div class="mm-fab-dot"></div>
                <div class="mm-fab-dot"></div>
                <div class="mm-fab-dot"></div>
            </div>
            <div class="mm-fab-icon">
                <div class="mm-gd"></div>
                <div class="mm-gd"></div>
                <div class="mm-gd"></div>
                <div class="mm-gd"></div>
            </div>
        </div>

        <div class="mm-toast">Drag me anywhere. Click for tools.</div>
    `;
    parentDoc.body.appendChild(root);

    const panel = root.querySelector(".mm-widget-panel");
    const fab = root.querySelector(".mm-fab");
    const icon = root.querySelector(".mm-fab-icon");
    const toast = root.querySelector(".mm-toast");
    let startX = 0, startY = 0, originX = 0, originY = 0, dragging = false, moved = false;

    const saved = (() => {
        try { return JSON.parse(parentWin.localStorage.getItem(widgetId + "-pos") || "null"); }
        catch (_) { return null; }
    })();

    function clampPosition(x, y) {
        const rect = root.getBoundingClientRect();
        return {
            x: Math.min(Math.max(x, 8), parentWin.innerWidth - rect.width - 8),
            y: Math.min(Math.max(y, 8), parentWin.innerHeight - 72),
        };
    }
    function setPosition(x, y) {
        const p = clampPosition(x, y);
        root.style.left = p.x + "px";
        root.style.top = p.y + "px";
        root.style.right = "auto";
        root.style.bottom = "auto";
        root.classList.toggle("mm-left", p.x < parentWin.innerWidth / 2);
        try { parentWin.localStorage.setItem(widgetId + "-pos", JSON.stringify(p)); } catch (_) {}
        if (root.classList.contains("mm-open")) fitPanelToViewport();
    }
    function fitPanelToViewport() {
        const rect = root.getBoundingClientRect();
        const spaceAbove = Math.max(0, rect.top - 12);
        const spaceBelow = Math.max(0, parentWin.innerHeight - rect.bottom - 12);
        const openDown = spaceBelow >= spaceAbove || spaceAbove < 220;
        root.classList.toggle("mm-down", openDown);
        root.classList.toggle("mm-up", !openDown);
        root.classList.toggle("mm-left", rect.left < parentWin.innerWidth / 2);
        const available = (openDown ? spaceBelow : spaceAbove) - 16;
        panel.style.maxHeight = Math.max(180, Math.min(520, available)) + "px";
        panel.style.overflowY = "auto";
    }
    function togglePanel(force) {
        const open = typeof force === "boolean" ? force : !root.classList.contains("mm-open");
        if (open) fitPanelToViewport();
        root.classList.toggle("mm-open", open);
        fab.setAttribute("aria-label", open ? "Close quick tools" : "Open quick tools");
        if (open) setTimeout(fitPanelToViewport, 40);
    }
    function showToast(message) {
        toast.textContent = message;
        toast.classList.add("mm-show");
        setTimeout(() => toast.classList.remove("mm-show"), 1500);
    }
    function normalizeText(value) {
        return (value || "").toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();
    }
    function getElementText(el) {
        return el ? (el.innerText || el.textContent || el.getAttribute("aria-label") || "") : "";
    }
    function getMatchScore(el, text) {
        const wanted = normalizeText(text);
        const actual = normalizeText(getElementText(el));
        if (!wanted || !actual) return 999;
        if (actual === wanted) return 0;
        if (actual.endsWith(" " + wanted) || actual.startsWith(wanted + " ")) return 1;
        if (actual.includes(wanted)) return 2;
        return 999;
    }
    function textMatches(value, text) {
        const wanted = normalizeText(text);
        return wanted && normalizeText(value).includes(wanted);
    }
    function activateElement(el) {
        const target = el.querySelector("input") || el;
        ["pointerdown", "mousedown", "mouseup", "click"].forEach(type => {
            target.dispatchEvent(new parentWin.MouseEvent(type, {bubbles: true, cancelable: true, view: parentWin}));
        });
        target.click();
    }
    function clickByText(selector, text) {
        const candidates = Array.from(parentDoc.querySelectorAll(selector))
            .filter(el => !root.contains(el))
            .map(el => ({el, score: getMatchScore(el, text), length: normalizeText(getElementText(el)).length}))
            .filter(item => item.score < 999)
            .sort((a, b) => a.score - b.score || a.length - b.length);
        if (candidates.length) {
            activateElement(candidates[0].el);
            return true;
        }
        return false;
    }
    function isFocusableInput(el) {
        if (!el || root.contains(el) || el.disabled || el.getAttribute("aria-hidden") === "true") return false;
        const rect = el.getBoundingClientRect();
        const style = parentWin.getComputedStyle(el);
        return rect.width > 0 && rect.height > 0 && style.display !== "none" && style.visibility !== "hidden";
    }
    function findSymptomInput() {
        const selectors = [
            '[data-testid="stMultiSelect"] input:not([type="hidden"])',
            '[data-testid="stTextArea"] textarea',
            'textarea[aria-label*="symptom"]',
            'input[aria-label*="symptom"]:not([type="hidden"])',
            '[data-baseweb="select"] input:not([type="hidden"])',
            '[data-testid="stTextInput"] input:not([type="hidden"])',
            '[data-testid="stSelectbox"] input:not([type="hidden"])',
            'textarea, input:not([type="hidden"])'
        ];
        for (const selector of selectors) {
            let matches = [];
            try { matches = Array.from(parentDoc.querySelectorAll(selector)); }
            catch (_) { continue; }
            const target = matches.find(isFocusableInput);
            if (target) return target;
        }
        return null;
    }
    function focusInput() {
        const target = findSymptomInput();
        if (target) {
            const scrollTarget = target.closest('[data-testid="stElementContainer"], [data-testid="stVerticalBlock"], section') || target;
            scrollTarget.scrollIntoView({behavior: "smooth", block: "center"});
            setTimeout(() => {
                try { target.focus({preventScroll: true}); }
                catch (_) { target.focus(); }
                target.click();
            }, 350);
            return true;
        }
        return false;
    }
    function scrollToText(text) {
        const selectors = [
            "main h1, main h2, main h3, main .md-section-title, main .section-title, main .sec-title, main .md-contact-title, main [class*='section-title'], main [class*='form-title']",
            "h1,h2,h3,.md-section-title,.section-title,.sec-title,.md-contact-title,[class*='section-title'],[class*='form-title']",
            "main p, main span, main div"
        ];
        for (const selector of selectors) {
            const target = Array.from(parentDoc.querySelectorAll(selector)).find(el => {
                if (root.contains(el) || !textMatches(el.innerText, text)) return false;
                return selector.includes("div") ? (el.innerText || "").trim().length <= 220 : true;
            });
            if (target) {
                const scrollTarget = target.closest('[data-testid="stElementContainer"], [data-testid="stVerticalBlock"], section') || target;
                scrollTarget.scrollIntoView({behavior: "smooth", block: "start"});
                return true;
            }
        }
        return false;
    }
    function savePendingScroll(target) {
        if (!target) return;
        try { parentWin.localStorage.setItem(widgetId + "-pending-scroll", target); } catch (_) {}
    }
    function savePendingFocus() {
        try { parentWin.localStorage.setItem(widgetId + "-pending-focus", "1"); } catch (_) {}
    }
    function clearPendingFocus() {
        try { parentWin.localStorage.removeItem(widgetId + "-pending-focus"); } catch (_) {}
    }
    function clearDiseasePageParam() {
        try {
            const url = new URL(parentWin.location.href);
            if (url.searchParams.has("disease_page")) {
                url.searchParams.delete("disease_page");
                parentWin.history.replaceState({}, "", url.toString());
            }
        } catch (_) {}
    }
    function clearPendingScroll() {
        try { parentWin.localStorage.removeItem(widgetId + "-pending-scroll"); } catch (_) {}
        clearDiseasePageParam();
    }
    function scrollWithRetries(target) {
        if (!target) return;
        let tries = 0;
        const timer = parentWin.setInterval(() => {
            tries += 1;
            if (scrollToText(target) || tries >= 20) {
                if (tries < 20) clearPendingScroll();
                parentWin.clearInterval(timer);
            }
        }, 250);
    }
    function applyPendingScroll() {
        let target = "";
        try { target = parentWin.localStorage.getItem(widgetId + "-pending-scroll") || ""; } catch (_) {}
        if (target) scrollWithRetries(target);
        else clearDiseasePageParam();
    }
    function applyPendingFocus() {
        let pending = "";
        try { pending = parentWin.localStorage.getItem(widgetId + "-pending-focus") || ""; } catch (_) {}
        if (!pending) return;
        let tries = 0;
        const timer = parentWin.setInterval(() => {
            tries += 1;
            if (focusInput()) {
                clearPendingFocus();
                clearDiseasePageParam();
                parentWin.clearInterval(timer);
            } else if (tries >= 24) {
                clearPendingFocus();
                parentWin.clearInterval(timer);
                showToast("Could not find symptom input");
            }
        }, 250);
    }
    function clickWidgetBridge(page) {
        const wanted = normalizeText("widget-nav-" + page);
        const target = Array.from(parentDoc.querySelectorAll("button")).find(button =>
            !root.contains(button) && normalizeText(getElementText(button)) === wanted
        );
        if (target) {
            activateElement(target);
            return true;
        }
        return false;
    }
    function openDiseasePage(page, scrollTarget) {
        const wanted = scrollTarget || page;
        // Try setting URL on parent or top window to trigger Streamlit query param rerun
        const wins = [parentWin, window.top].filter(Boolean);
        for (const w of wins) {
            try {
                const url = new URL(w.location.href);
                url.searchParams.set("disease_page", page);
                w.location.href = url.toString();
                savePendingScroll(wanted);
                return true;
            } catch (_) {}
        }
        // Fallback: hidden bridge buttons
        if (clickWidgetBridge(page)) {
            savePendingScroll(wanted);
            parentWin.setTimeout(() => scrollWithRetries(wanted), 900);
            return true;
        }
        if (scrollToText(wanted)) { clearPendingScroll(); return true; }
        return false;
    }
    function runAction(kind, target, scrollTarget) {
        if (kind === "top") {
            (function() {
                function smoothScrollTo(el) {
                    if (!el || el.scrollTop === 0) return;
                    var start = el.scrollTop;
                    var startTime = null;
                    var duration = 520;
                    function ease(t) { return t < 0.5 ? 4*t*t*t : 1 - Math.pow(-2*t+2,3)/2; }
                    function step(timestamp) {
                        if (!startTime) startTime = timestamp;
                        var progress = Math.min((timestamp - startTime) / duration, 1);
                        el.scrollTop = start * (1 - ease(progress));
                        if (progress < 1) parentWin.requestAnimationFrame(step);
                    }
                    parentWin.requestAnimationFrame(step);
                }
                var selectors = [
                    '[data-testid="stMain"]',
                    '[data-testid="stAppViewContainer"]',
                    '[data-testid="block-container"]',
                    '.main > div',
                    'section.main',
                    '.stApp > section',
                ];
                selectors.forEach(function(sel) {
                    var el = parentDoc.querySelector(sel);
                    if (el) smoothScrollTo(el);
                });
                smoothScrollTo(parentDoc.documentElement);
                smoothScrollTo(parentDoc.body);
                parentWin.scrollTo({top: 0, behavior: "smooth"});
            })();
            clearPendingScroll();
            return true;
        }
        if (kind === "input") {
            if (focusInput()) return true;
            savePendingFocus();
            const ok = openDiseasePage(target, "Predict Disease From Symptoms");
            parentWin.setTimeout(applyPendingFocus, 800);
            return ok || true;
        }
        if (kind === "tab") {
            savePendingScroll(scrollTarget || target);
            const ok = clickByText('[role="tab"], [data-baseweb="tab"], button', target);
            if (ok) parentWin.setTimeout(() => scrollWithRetries(scrollTarget || target), 280);
            return ok;
        }
        if (kind === "nav") return openDiseasePage(target, scrollTarget);
        if (kind === "scroll") {
            clearPendingScroll();
            return scrollToText(scrollTarget || target);
        }
        return false;
    }

    if (saved && Number.isFinite(saved.x) && Number.isFinite(saved.y)) setPosition(saved.x, saved.y);

    panel.addEventListener("wheel", event => event.stopPropagation(), {passive: false});
    panel.addEventListener("touchmove", event => event.stopPropagation(), {passive: true});

    fab.addEventListener("pointerdown", event => {
        dragging = true;
        moved = false;
        startX = event.clientX;
        startY = event.clientY;
        const rect = root.getBoundingClientRect();
        originX = rect.left;
        originY = rect.top;
        root.classList.add("mm-dragging");
        fab.setPointerCapture(event.pointerId);
    });
    fab.addEventListener("pointermove", event => {
        if (!dragging) return;
        const dx = event.clientX - startX;
        const dy = event.clientY - startY;
        if (Math.abs(dx) + Math.abs(dy) > 5) moved = true;
        if (moved) {
            togglePanel(false);
            setPosition(originX + dx, originY + dy);
        }
    });
    fab.addEventListener("pointerup", event => {
        dragging = false;
        root.classList.remove("mm-dragging");
        try { fab.releasePointerCapture(event.pointerId); } catch (_) {}
        if (moved) showToast("Position saved");
        else togglePanel();
    });
    fab.addEventListener("keydown", event => {
        if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            togglePanel();
        }
        if (event.key === "Escape") togglePanel(false);
    });

    root.querySelectorAll(".mm-tool").forEach(button => {
        button.addEventListener("click", () => {
            const kind = button.getAttribute("data-kind");
            const target = button.getAttribute("data-target");
            const scrollTarget = button.getAttribute("data-scroll") || target;
            togglePanel(false);
            const ok = runAction(kind, target, scrollTarget);
            showToast(ok ? "Opening " + button.querySelector(".mm-tool-label").innerText : "Could not find target");
        });
    });

    parentDoc.addEventListener("pointerdown", event => {
        if (root.classList.contains("mm-open") && !root.contains(event.target)) {
            togglePanel(false);
        }
    });
    parentDoc.addEventListener("keydown", event => {
        if (event.key === "Escape") togglePanel(false);
    });
    parentWin.addEventListener("resize", () => {
        const rect = root.getBoundingClientRect();
        setPosition(rect.left, rect.top);
    });
    parentWin.addEventListener("scroll", () => {
        if (root.classList.contains("mm-open")) fitPanelToViewport();
    }, {passive: true});

    root.classList.toggle("mm-left", root.getBoundingClientRect().left < parentWin.innerWidth / 2);
    root.classList.add("mm-up");
    parentWin.setTimeout(applyPendingScroll, 420);
    parentWin.setTimeout(applyPendingFocus, 560);
})();
</script>
        """,
        height=0,
        width=0,
    )

render_footer()
render_floating_widget()

# Fix sidebar toggle icon — injects CSS+SVG into parent document via components.html
components.html(
    """
<script>
(function() {
    var p = window.parent;
    var pd = p.document;

    // 1. Inject CSS into parent <head>
    var sid = 'shc-sidebar-toggle-style';
    if (!pd.getElementById(sid)) {
        var s = pd.createElement('style');
        s.id = sid;
        s.textContent = [
            '[data-testid="stSidebarCollapsedControl"] button {',
            '  width:40px!important;height:40px!important;border-radius:50%!important;',
            '  display:flex!important;align-items:center!important;justify-content:center!important;',
            '  background:rgba(103,80,164,0.15)!important;',
            '  border:1.5px solid rgba(103,80,164,0.35)!important;',
            '  box-shadow:0 2px 12px rgba(103,80,164,0.18)!important;',
            '  transition:background 200ms ease,transform 200ms ease,box-shadow 200ms ease!important;',
            '  padding:0!important;overflow:hidden!important;cursor:pointer!important;',
            '}',
            '[data-testid="stSidebarCollapsedControl"] button:hover {',
            '  background:rgba(103,80,164,0.28)!important;',
            '  transform:scale(1.10)!important;',
            '  box-shadow:0 4px 20px rgba(103,80,164,0.28)!important;',
            '}',
            '[data-testid="stSidebarCollapsedControl"] button > * {',
            '  display:none!important;',
            '}',
        ].join('\\n');
        pd.head.appendChild(s);
    }

    // 2. Inject SVG icon — color depends on current theme
    function getIconColor() {
        var isLight = pd.documentElement.getAttribute('data-theme') === 'light'
            || pd.body.getAttribute('data-theme') === 'light';
        return isLight ? '#4a3a7d' : '#c4b5fd';
    }

    function makeSVG(color) {
        return '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" aria-hidden="true" style="display:block!important;flex-shrink:0;pointer-events:none">'
             + '<rect x="3" y="5.5" width="18" height="2.2" rx="1.1" fill="' + color + '"/>'
             + '<rect x="3" y="10.9" width="13" height="2.2" rx="1.1" fill="' + color + '"/>'
             + '<rect x="3" y="16.3" width="18" height="2.2" rx="1.1" fill="' + color + '"/>'
             + '</svg>';
    }

    function fix() {
        var ctrl = pd.querySelector('[data-testid="stSidebarCollapsedControl"]');
        if (!ctrl) return;
        var btn = ctrl.querySelector('button');
        if (!btn) return;
        var existing = btn.querySelector('svg[aria-hidden="true"]');
        var color = getIconColor();
        if (existing) {
            // update fill on all rects to match theme
            existing.querySelectorAll('rect').forEach(function(r) { r.setAttribute('fill', color); });
        } else {
            btn.insertAdjacentHTML('beforeend', makeSVG(color));
        }
    }

    fix();
    setTimeout(fix, 300);
    setTimeout(fix, 800);
    new p.MutationObserver(fix).observe(pd.body, {childList:true, subtree:true, attributes:true, attributeFilter:['data-theme']});
    new p.MutationObserver(fix).observe(pd.documentElement, {attributes:true, attributeFilter:['data-theme']});
})();
</script>
""",
    height=0,
    width=0,
)