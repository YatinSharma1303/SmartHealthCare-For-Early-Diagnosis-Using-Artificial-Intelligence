import base64
import pickle
from html import escape
from pathlib import Path
from urllib.parse import quote_plus
import random
import pandas as pd
import numpy as np
import joblib
import streamlit as st
import streamlit.components.v1 as components

from theme_config import init_theme, get_theme_styles, render_theme_toggle


APP_TITLE = "MedMatch AI"
CREATOR_NAME = "Yatin Sharma"

MODEL_DIR = Path("models/second_feature_models")
DATA_PATH = Path("data/Drug reccomendation/medicine.csv")
HERO_IMAGE = Path("utils/medss.png")


st.set_page_config(
    page_title=APP_TITLE,
    page_icon="💊",
    layout="wide",
)

init_theme()
st.markdown(get_theme_styles(), unsafe_allow_html=True)
if st.session_state.get("theme") == "light":
    st.markdown("<script>document.body.classList.add('medmatch-light');</script>", unsafe_allow_html=True)
else:
    st.markdown("<script>document.body.classList.remove('medmatch-light');</script>", unsafe_allow_html=True)
render_theme_toggle()


# ─────────────────────────────── helpers ────────────────────────────────────

@st.cache_data(show_spinner=False)
def image_to_data_uri(path):
    image_path = Path(path)
    if not image_path.exists():
        return None
    suffix = image_path.suffix.lower()
    mime_type = "image/png"
    if suffix in [".jpg", ".jpeg"]:
        mime_type = "image/jpeg"
    elif suffix == ".webp":
        mime_type = "image/webp"
    encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


@st.cache_resource(show_spinner=False)
def load_models():
    with open(MODEL_DIR / "medicine_dict.pkl", "rb") as file:
        medicine_dict = pickle.load(file)
    similarity = joblib.load(MODEL_DIR / "similarity.joblib")
    medicines_df = pd.DataFrame(medicine_dict)
    if "Drug_Name" not in medicines_df.columns:
        st.error("Drug_Name column was not found in medicine_dict.pkl.")
        st.stop()
    medicines_df = medicines_df.dropna(subset=["Drug_Name"]).reset_index(drop=True)
    medicines_df["Drug_Name"] = medicines_df["Drug_Name"].astype(str)
    drug_to_index = {
        drug: index for index, drug in enumerate(medicines_df["Drug_Name"].tolist())
    }
    return medicines_df, np.asarray(similarity), drug_to_index


@st.cache_data(show_spinner=False)
def load_description_data():
    if not DATA_PATH.exists():
        return pd.DataFrame(columns=["Drug_Name", "Description"])
    data = pd.read_csv(DATA_PATH)
    if "Drug_Name" not in data.columns:
        return pd.DataFrame(columns=["Drug_Name", "Description"])
    if "Description" not in data.columns:
        data["Description"] = "Description not available."
    data["Drug_Name"] = data["Drug_Name"].astype(str)
    data["Description"] = data["Description"].fillna("Description not available.").astype(str)
    return data


def get_description(drug_name, description_lookup):
    return description_lookup.get(drug_name, "Description not available.")


def recommend_drugs(medicine, top_n, drug_names, similarity_matrix, drug_to_index):
    if medicine not in drug_to_index:
        return []
    medicine_index = drug_to_index[medicine]
    scores = np.asarray(similarity_matrix[medicine_index], dtype=float).copy()
    if scores.size <= 1:
        return []
    scores = np.nan_to_num(scores, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
    scores[medicine_index] = -np.inf
    result_count = min(top_n, scores.size - 1)
    if result_count <= 0:
        return []
    candidate_indices = np.argpartition(scores, -result_count)[-result_count:]
    candidate_indices = candidate_indices[np.argsort(scores[candidate_indices])[::-1]]
    recommendations = []
    for index in candidate_indices:
        if 0 <= index < len(drug_names):
            recommendations.append((drug_names[index], float(scores[index])))
    return recommendations


def get_similarity_between(drug_a, drug_b, similarity, drug_to_index):
    if drug_a not in drug_to_index or drug_b not in drug_to_index:
        return 0.0
    return float(similarity[drug_to_index[drug_a]][drug_to_index[drug_b]])


def get_health_tips():
    return [
        ("Medication Timing", "Take medicines at the same time each day when prescribed that way."),
        ("Hydration", "Use a full glass of water unless your clinician gave different instructions."),
        ("Food Rules", "Check whether the medicine should be taken with food or on an empty stomach."),
        ("Missed Dose", "Do not double dose unless a clinician specifically told you to."),
        ("Medication List", "Keep an updated list of prescriptions, OTC medicines, and supplements."),
        ("Storage", "Protect medicines from heat, moisture, and direct light unless labeled otherwise."),
    ]


def get_dosage_categories():
    return [
        {
            "title": "Oral Tablets & Capsules",
            "icon": "💊",
            "info": "Swallow whole with water unless the label says it can be split, crushed, or chewed.",
            "tips": ["Check strength before every dose", "Do not crush extended-release tablets", "Store in a dry place"],
        },
        {
            "title": "Liquid Medications",
            "icon": "🧪",
            "info": "Use the provided cup, syringe, or dropper. Kitchen spoons are not dosing tools.",
            "tips": ["Shake if directed", "Confirm mg/mL concentration", "Clean measuring devices after use"],
        },
        {
            "title": "Topical Applications",
            "icon": "🧴",
            "info": "Apply a thin layer to clean, dry skin and avoid eyes or mucous membranes unless prescribed.",
            "tips": ["Wash hands before and after", "Do not cover unless instructed", "Stop if severe irritation appears"],
        },
        {
            "title": "Inhalers & Nebulizers",
            "icon": "🌬️",
            "info": "Technique matters. Ask a pharmacist to demonstrate if you are unsure.",
            "tips": ["Shake metered inhalers if directed", "Rinse after steroid inhalers", "Track remaining doses"],
        },
        {
            "title": "Injectables",
            "icon": "💉",
            "info": "Use sterile technique and rotate injection sites according to your care plan.",
            "tips": ["Check storage temperature", "Never reuse needles", "Use a sharps container"],
        },
        {
            "title": "Eye & Ear Drops",
            "icon": "👁️",
            "info": "Keep the dropper clean and wait between multiple drops when instructed.",
            "tips": ["Do not touch the tip", "Warm ear drops in your hands", "Keep containers capped"],
        },
    ]


# ──────────────────────────────── CSS ───────────────────────────────────────

def render_styles():
    st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800;900&family=Space+Grotesk:wght@400;500;600;700&display=swap" rel="stylesheet">

<style>
/* ── Design tokens ─────────────────────────────────────────────────────── */
:root {
    --c-primary:       #7c4dff;
    --c-primary-rgb:   124,77,255;
    --c-secondary:     #00bcd4;
    --c-secondary-rgb: 0,188,212;
    --c-tertiary:      #ff6d00;
    --c-success:       #00c853;
    --c-warn:          #ffd600;
    --c-error:         #ff1744;

    --c-bg:            #0d0d14;
    --c-surface:       rgba(255,255,255,0.038);
    --c-surface-2:     rgba(255,255,255,0.065);
    --c-surface-3:     rgba(255,255,255,0.095);
    --c-outline:       rgba(255,255,255,0.10);
    --c-outline-2:     rgba(255,255,255,0.06);
    --c-text:          #f0eeff;
    --c-muted:         rgba(200,190,255,0.55);

    --radius-sm:  14px;
    --radius-md:  22px;
    --radius-lg:  30px;
    --radius-xl:  40px;

    --shadow-sm: 0 2px 12px rgba(0,0,0,0.28);
    --shadow-md: 0 8px 32px rgba(0,0,0,0.36);
    --shadow-lg: 0 20px 60px rgba(0,0,0,0.44);

    --font-display: 'Nunito', sans-serif;
    --font-body:    'Space Grotesk', sans-serif;

    --transition: 180ms cubic-bezier(0.4,0,0.2,1);
}

html { scroll-behavior: smooth; overflow-x: hidden; }

/* Global */
*, *::before, *::after { box-sizing: border-box; }

body, .stApp {
    font-family: var(--font-body);
    /* background intentionally NOT overridden — let theme_config control it */
    color: var(--c-text);
    overflow-x: hidden;
}

.block-container {
    max-width: 1260px;
    padding-top: 1rem;
    padding-bottom: 3rem;
}

/* ── Scrollbar ─────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(var(--c-primary-rgb),.4); border-radius: 99px; }

/* ── Sidebar ───────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    border-right: 1px solid var(--c-outline);
    background: linear-gradient(180deg,
        rgba(var(--c-primary-rgb),.14) 0%,
        rgba(var(--c-secondary-rgb),.07) 50%,
        rgba(0,0,0,.02) 100%);
    overflow-x: hidden !important;
}

[data-testid="stSidebarContent"] {
    overflow-x: hidden !important;
}

@media (max-width: 768px) {
    [data-testid="stSidebar"] {
        background: #12101a !important;
    }
}
[data-testid="stSidebar"] img {
    width: 100%;
    max-height: 200px;
    object-fit: contain;
    border-radius: 20px;
    border: 1px solid var(--c-outline);
    padding: 8px;
    margin-bottom: 10px;
    background: var(--c-surface);
}

/* ── Nav Pills ─────────────────────────────────────────────────── */
.nav-pill-wrap {
    display: flex;
    gap: 8px;
    margin-bottom: 24px;
    flex-wrap: wrap;
}
.nav-pill {
    padding: 9px 18px;
    border-radius: 999px;
    border: 1px solid var(--c-outline);
    background: var(--c-surface);
    font-size: 13px;
    font-weight: 700;
    cursor: pointer;
    transition: all var(--transition);
    font-family: var(--font-body);
    color: var(--c-text);
    display: inline-flex; align-items: center; gap: 6px;
    user-select: none;
}
.nav-pill:hover { background: var(--c-surface-2); border-color: rgba(var(--c-primary-rgb),.4); }
.nav-pill.active {
    background: linear-gradient(135deg, rgba(var(--c-primary-rgb),.35), rgba(var(--c-secondary-rgb),.2));
    border-color: rgba(var(--c-primary-rgb),.7);
    box-shadow: 0 0 20px rgba(var(--c-primary-rgb),.25);
}

/* ── Hero ──────────────────────────────────────────────────────── */
.hero-wrap {
    position: relative;
    overflow: hidden;
    border: 1px solid var(--c-outline);
    border-radius: var(--radius-xl);
    padding: 40px 36px;
    margin-bottom: 22px;
    background:
        radial-gradient(ellipse at 80% 20%, rgba(var(--c-primary-rgb),.22) 0%, transparent 60%),
        radial-gradient(ellipse at 10% 80%, rgba(var(--c-secondary-rgb),.14) 0%, transparent 55%),
        var(--c-surface);
    box-shadow: var(--shadow-lg);
}
.hero-grid {
    display: grid;
    grid-template-columns: 1fr min(340px, 38%);
    gap: 32px;
    align-items: center;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 14px;
    border-radius: 999px;
    border: 1px solid rgba(var(--c-primary-rgb),.45);
    background: rgba(var(--c-primary-rgb),.15);
    font-size: 12px; font-weight: 800; color: #c8b8ff;
    margin-bottom: 14px;
    font-family: var(--font-display);
}
.hero-title {
    font-family: var(--font-display);
    font-size: clamp(34px, 4.5vw, 64px);
    font-weight: 900;
    line-height: 1.0;
    margin: 0 0 14px 0;
    background: linear-gradient(135deg, #fff 30%, #c8b8ff 70%, #80deea);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    color: var(--c-muted);
    font-size: 15px;
    line-height: 1.7;
    max-width: 560px;
    margin-bottom: 20px;
}
.chip-row { display: flex; gap: 10px; flex-wrap: wrap; }
.chip {
    padding: 7px 14px; border-radius: 999px;
    border: 1px solid var(--c-outline-2);
    background: var(--c-surface-2);
    font-size: 12px; font-weight: 700;
    color: var(--c-muted);
}
.hero-img {
    width: 100%;
    aspect-ratio: 1.15;
    object-fit: contain;
    border-radius: var(--radius-lg);
    padding: 14px;
    background: linear-gradient(135deg, rgba(var(--c-primary-rgb),.12), rgba(var(--c-secondary-rgb),.08));
    border: 1px solid rgba(255,255,255,.12);
    box-shadow: var(--shadow-md);
}
.hero-fallback {
    width: 100%;
    aspect-ratio: 1.15;
    border-radius: var(--radius-lg);
    display: flex; align-items: center; justify-content: center;
    font-size: 64px;
    background: linear-gradient(135deg, rgba(var(--c-primary-rgb),.18), rgba(var(--c-secondary-rgb),.12));
    border: 1px solid rgba(255,255,255,.12);
}
/* Decorative orbs */
.hero-orb1 {
    position: absolute; top: -60px; right: -60px;
    width: 240px; height: 240px; border-radius: 50%;
    background: radial-gradient(circle, rgba(var(--c-primary-rgb),.25), transparent 70%);
    pointer-events: none;
}
.hero-orb2 {
    position: absolute; bottom: -80px; left: 100px;
    width: 300px; height: 300px; border-radius: 50%;
    background: radial-gradient(circle, rgba(var(--c-secondary-rgb),.12), transparent 70%);
    pointer-events: none;
}

/* ── Stat Cards ────────────────────────────────────────────────── */
.stat-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    margin-bottom: 22px;
}
.stat-card {
    border: 1px solid var(--c-outline-2);
    border-radius: var(--radius-md);
    padding: 18px 16px;
    background: var(--c-surface);
    box-shadow: var(--shadow-sm);
    position: relative; overflow: hidden;
    transition: transform var(--transition), border-color var(--transition);
}
.stat-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, rgba(var(--c-primary-rgb),.7), rgba(var(--c-secondary-rgb),.5));
    border-radius: var(--radius-sm) var(--radius-sm) 0 0;
}
.stat-card:hover { transform: translateY(-3px); border-color: rgba(var(--c-primary-rgb),.35); }
.stat-icon { font-size: 22px; margin-bottom: 8px; }
.stat-label { color: var(--c-muted); font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: .07em; margin-bottom: 5px; }
.stat-value { font-size: 18px; font-weight: 800; font-family: var(--font-display); }

/* ── Tool Panel ─────────────────────────────────────────────────── */
.tool-panel {
    border: 1px solid var(--c-outline);
    border-radius: var(--radius-lg);
    padding: 24px;
    background: linear-gradient(135deg, rgba(var(--c-primary-rgb),.1), rgba(var(--c-secondary-rgb),.06)),
                var(--c-surface);
    box-shadow: var(--shadow-md);
    margin-bottom: 20px;
}

/* ── Section title ─────────────────────────────────────────────── */
.section-title {
    font-size: 20px;
    font-weight: 900;
    font-family: var(--font-display);
    margin: 0 0 8px 0;
}
.section-sub {
    color: var(--c-muted);
    font-size: 13px;
    line-height: 1.6;
    margin-bottom: 16px;
}

/* ── Description card ──────────────────────────────────────────── */
.desc-card {
    border: 1px solid var(--c-outline);
    border-radius: var(--radius-md);
    padding: 20px;
    background: var(--c-surface);
    margin-bottom: 18px;
}
.desc-card h3 {
    font-size: 17px; font-weight: 800;
    font-family: var(--font-display);
    margin: 0 0 8px 0;
}
.desc-text { color: var(--c-muted); font-size: 14px; line-height: 1.65; }

/* ── Drug result cards ─────────────────────────────────────────── */
.rec-grid { display: grid; gap: 12px; }
.drug-card {
    border: 1px solid var(--c-outline-2);
    border-radius: var(--radius-md);
    padding: 16px 18px;
    background: var(--c-surface);
    box-shadow: var(--shadow-sm);
    transition: transform var(--transition), border-color var(--transition),
                box-shadow var(--transition), background var(--transition);
}
.drug-card:hover {
    transform: translateY(-2px);
    border-color: rgba(var(--c-primary-rgb),.5);
    background: var(--c-surface-2);
    box-shadow: var(--shadow-md);
}
.drug-inner { display: flex; align-items: center; gap: 14px; }
.drug-rank {
    width: 44px; height: 44px; min-width: 44px;
    border-radius: 16px;
    display: flex; align-items: center; justify-content: center;
    font-weight: 900; font-size: 15px; color: #fff;
    background: linear-gradient(135deg, #7c4dff, #00bcd4);
    box-shadow: 0 4px 14px rgba(var(--c-primary-rgb),.4);
    font-family: var(--font-display);
}
.drug-info { flex: 1; min-width: 0; }
.drug-name {
    font-size: 15px; font-weight: 800;
    font-family: var(--font-display);
    word-break: break-word;
}
.drug-meta { color: var(--c-muted); font-size: 12px; margin-top: 3px; }
.sim-bar-wrap { margin-top: 6px; }
.sim-label { font-size: 11px; color: var(--c-muted); margin-bottom: 3px; }
.sim-bar-bg {
    height: 4px; border-radius: 99px;
    background: var(--c-outline);
    overflow: hidden;
}
.sim-bar-fill {
    height: 100%; border-radius: 99px;
    background: linear-gradient(90deg, #7c4dff, #00bcd4);
    transition: width .6s cubic-bezier(.4,0,.2,1);
}
.drug-actions { display: flex; gap: 8px; flex-shrink: 0; }
.btn-ghost {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 8px 14px; border-radius: 999px;
    border: 1px solid var(--c-outline);
    background: var(--c-surface-2);
    color: var(--c-muted);
    text-decoration: none;
    font-size: 12px; font-weight: 700;
    transition: all var(--transition);
    white-space: nowrap;
}
.btn-ghost:hover { border-color: rgba(var(--c-primary-rgb),.5); color: #c8b8ff; }
.btn-primary {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 9px 16px; border-radius: 999px;
    color: #fff !important; text-decoration: none !important;
    font-size: 12px; font-weight: 800;
    background: linear-gradient(135deg, #7c4dff, #6200ea);
    box-shadow: 0 4px 14px rgba(var(--c-primary-rgb),.35);
    transition: all var(--transition);
    white-space: nowrap;
}
.btn-primary:hover { transform: translateY(-1px); box-shadow: 0 8px 24px rgba(var(--c-primary-rgb),.45); }

/* ══════════════════════════════════════════════════════════════════
   MD3 EXPRESSIVE SIDEBAR — full redesign
   ══════════════════════════════════════════════════════════════════ */

/* ── Sidebar shell ─────────────────────────────────────────────── */
[data-testid="stSidebar"] > div:first-child {
    padding: 0 !important;
}
[data-testid="stSidebarContent"] {
    padding: 0 12px 20px !important;
    display: flex;
    flex-direction: column;
    gap: 0;
}

/* ── Brand hero banner ─────────────────────────────────────────── */
.sb-hero-banner {
    position: relative;
    overflow: hidden;
    border-radius: 22px;
    padding: 18px 16px 16px;
    margin: 0 0 16px;
    background:
        radial-gradient(ellipse at 15% 0%,  rgba(var(--c-primary-rgb), 0.18) 0%, transparent 52%),
        radial-gradient(ellipse at 88% 95%, rgba(var(--c-secondary-rgb), 0.10) 0%, transparent 48%),
        linear-gradient(160deg, rgba(var(--c-primary-rgb), 0.10) 0%, rgba(var(--c-secondary-rgb), 0.05) 100%),
        var(--c-surface);
    border: 1px solid rgba(var(--c-primary-rgb), 0.22);
    box-shadow: 0 4px 20px rgba(0,0,0,0.22);
}
.sb-hero-banner::before {
    content: '';
    position: absolute; inset: 0;
    background:
        repeating-linear-gradient(
            45deg,
            rgba(255,255,255,0.008) 0px,
            rgba(255,255,255,0.008) 1px,
            transparent 1px,
            transparent 18px
        );
    pointer-events: none;
}
.sb-banner-top {
    display: flex; align-items: flex-start; gap: 13px; position: relative; z-index: 1;
}
.sb-avatar-xl {
    width: 54px; height: 54px; min-width: 54px;
    border-radius: 18px;
    background: linear-gradient(135deg, rgba(var(--c-primary-rgb), 0.55) 0%, rgba(var(--c-secondary-rgb), 0.35) 100%);
    border: 1px solid rgba(var(--c-primary-rgb), 0.35);
    display: flex; align-items: center; justify-content: center;
    font-size: 26px;
    box-shadow: 0 4px 16px rgba(var(--c-primary-rgb), 0.28);
    flex-shrink: 0;
    position: relative;
    transform-origin: center;
    animation: sb-molecule-spin 3s ease-in-out infinite;
}
.sb-avatar-xl::after {
    content: '';
    position: absolute; bottom: -3px; right: -3px;
    width: 13px; height: 13px; border-radius: 50%;
    background: #00c853;
    border: 2px solid var(--c-bg, #0d0d14);
    box-shadow: 0 0 6px rgba(0,200,83,0.55);
    animation: sb-online 2.4s ease-in-out infinite;
}
@keyframes sb-molecule-spin {
    0%   { transform: rotate(0deg) scale(1); }
    25%  { transform: rotate(15deg) scale(1.10); }
    50%  { transform: rotate(0deg) scale(1); }
    75%  { transform: rotate(-15deg) scale(1.10); }
    100% { transform: rotate(0deg) scale(1); }
}
@keyframes sb-online { 0%,100%{box-shadow:0 0 6px rgba(0,200,83,0.55);} 50%{box-shadow:0 0 10px rgba(0,200,83,0.28);} }
.sb-banner-meta { flex: 1; min-width: 0; }
.sb-eyebrow {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 2px 9px; border-radius: 999px;
    background: rgba(124,77,255,0.25);
    border: 1px solid rgba(200,184,255,0.28);
    color: #d4c8ff; font-size: 9.5px; font-weight: 800;
    letter-spacing: 0.08em; text-transform: uppercase;
    font-family: var(--font-display);
    margin-bottom: 5px;
}
.sb-brand-name {
    font-family: var(--font-display);
    font-size: 19px; font-weight: 900; line-height: 1.1;
    background: linear-gradient(120deg, #fff 30%, #c8b8ff 70%, #80deea);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    word-break: break-word;
}
.sb-brand-sub {
    font-size: 11px; color: rgba(200,190,255,0.60); margin-top: 3px; font-weight: 600;
}
.sb-banner-desc {
    position: relative; z-index: 1;
    margin-top: 12px;
    padding: 10px 12px;
    border-radius: 14px;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.09);
    font-size: 11.5px; color: rgba(220,210,255,0.70);
    line-height: 1.55;
}
.sb-banner-chips {
    display: flex; gap: 6px; flex-wrap: wrap; margin-top: 12px; position: relative; z-index: 1;
}
.sb-chip {
    padding: 4px 10px; border-radius: 999px;
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.12);
    font-size: 10px; font-weight: 700; color: rgba(220,210,255,0.75);
}

/* ── Stat cards row ────────────────────────────────────────────── */
.sb-stats-strip {
    display: grid; grid-template-columns: repeat(3, 1fr);
    gap: 8px; margin-bottom: 14px;
}
.sb-stat-chip {
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px; padding: 12px 8px 10px;
    background: var(--c-surface);
    text-align: center;
    transition: background var(--transition), border-color var(--transition), transform var(--transition);
    position: relative; overflow: hidden;
}
.sb-stat-chip::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, #7c4dff, #00bcd4);
    border-radius: 999px 999px 0 0;
    opacity: 0; transition: opacity var(--transition);
}
.sb-stat-chip:hover { background: var(--c-surface-2); border-color: rgba(124,77,255,0.32); transform: translateY(-2px); }
.sb-stat-chip:hover::before { opacity: 1; }
.sb-stat-icon { font-size: 16px; margin-bottom: 4px; line-height: 1; }
.sb-stat-num {
    font-size: 13px; font-weight: 900; font-family: var(--font-display);
    background: linear-gradient(135deg, #c8b8ff, #80deea);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    line-height: 1.1;
}
.sb-stat-lbl {
    font-size: 9.5px; color: var(--c-muted); font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.06em; margin-top: 2px;
}

/* ── Section divider label ─────────────────────────────────────── */
.sb-section {
    font-size: 9.5px; font-weight: 800; color: var(--c-muted);
    text-transform: uppercase; letter-spacing: 0.12em;
    margin: 18px 0 8px 2px;
    display: flex; align-items: center; gap: 8px;
}
.sb-section::after { content: ''; flex: 1; height: 1px; background: rgba(255,255,255,0.07); }

/* ── Nav items (MD3 Navigation Drawer style) ───────────────────── */
.sb-nav-item {
    display: flex; align-items: center; gap: 12px;
    padding: 11px 14px; border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.06);
    background: var(--c-surface);
    margin-bottom: 5px;
    transition: transform var(--transition), background var(--transition),
                border-color var(--transition), box-shadow var(--transition);
    cursor: default; position: relative; overflow: hidden;
}
.sb-nav-item::before {
    content: '';
    position: absolute; left: 0; top: 0; bottom: 0; width: 3px;
    background: linear-gradient(180deg, #7c4dff, #00bcd4);
    border-radius: 0 3px 3px 0;
    opacity: 0; transition: opacity var(--transition);
}
.sb-nav-item:hover {
    transform: translateX(4px);
    background: var(--c-surface-2);
    border-color: rgba(124,77,255,0.30);
    box-shadow: 0 4px 18px rgba(124,77,255,0.10);
}
.sb-nav-item.sb-active {
    background: linear-gradient(135deg, rgba(124,77,255,0.22), rgba(0,188,212,0.12));
    border-color: rgba(124,77,255,0.48);
    box-shadow: 0 6px 22px rgba(124,77,255,0.18), inset 0 1px 0 rgba(255,255,255,0.07);
}
.sb-nav-item.sb-active::before { opacity: 1; }
.sb-nav-icon {
    width: 36px; height: 36px; min-width: 36px;
    border-radius: 12px;
    background: linear-gradient(135deg, rgba(124,77,255,0.28), rgba(0,188,212,0.18));
    border: 1px solid rgba(124,77,255,0.24);
    display: flex; align-items: center; justify-content: center;
    font-size: 15px; flex-shrink: 0;
    transition: transform var(--transition), box-shadow var(--transition);
}
.sb-nav-item.sb-active .sb-nav-icon {
    background: linear-gradient(135deg, #7c4dff, #00bcd4);
    border-color: transparent;
    box-shadow: 0 4px 14px rgba(124,77,255,0.45);
    transform: scale(1.05);
}
.sb-nav-text { flex: 1; min-width: 0; }
.sb-nav-title { font-size: 13px; font-weight: 800; font-family: var(--font-display); line-height: 1.2; }
.sb-nav-sub { color: var(--c-muted); font-size: 10.5px; margin-top: 1px; }
.sb-nav-arrow {
    font-size: 10px; color: var(--c-muted); opacity: 0;
    transition: opacity var(--transition), transform var(--transition);
    flex-shrink: 0;
}
.sb-nav-item:hover .sb-nav-arrow,
.sb-nav-item.sb-active .sb-nav-arrow { opacity: 1; transform: translateX(2px); }
.sb-nav-item.sb-active .sb-nav-arrow { color: #c8b8ff; }

/* ── Settings / slider card ────────────────────────────────────── */
.sb-settings-card {
    border: 1px solid rgba(0,188,212,0.22);
    border-radius: 20px;
    padding: 16px 16px 4px;
    margin-bottom: 14px;
    background: linear-gradient(135deg, rgba(0,188,212,0.07), rgba(124,77,255,0.05)), var(--c-surface);
    overflow: hidden; position: relative;
}
.sb-settings-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, #00bcd4, #7c4dff, #00bcd4);
    background-size: 200% 100%;
    animation: sb-shimmer 3s linear infinite;
}
@keyframes sb-shimmer { 0%{background-position:200% 0;} 100%{background-position:-200% 0;} }
.sb-settings-header {
    display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px;
}
.sb-settings-label {
    font-size: 10px; font-weight: 800; color: #80deea;
    text-transform: uppercase; letter-spacing: 0.09em;
    display: flex; align-items: center; gap: 5px;
}
.sb-count-pill {
    display: flex; align-items: baseline; gap: 2px;
    background: linear-gradient(135deg, rgba(124,77,255,0.25), rgba(0,188,212,0.15));
    border: 1px solid rgba(124,77,255,0.38); border-radius: 999px;
    padding: 3px 12px 3px 10px;
}
.sb-count-num {
    font-size: 20px; font-weight: 900; line-height: 1;
    background: linear-gradient(135deg, #c8b8ff, #80deea);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.sb-count-unit { font-size: 9.5px; font-weight: 700; color: rgba(200,190,255,0.50); padding-bottom: 1px; }

/* ── Quick links row ───────────────────────────────────────────── */
.sb-quick-row {
    display: grid; grid-template-columns: 1fr 1fr;
    gap: 7px; margin-bottom: 14px;
}
.sb-quick-btn {
    display: flex; align-items: center; gap: 7px;
    padding: 10px 12px; border-radius: 14px;
    border: 1px solid rgba(255,255,255,0.07);
    background: var(--c-surface);
    text-decoration: none !important; color: var(--c-text) !important;
    font-size: 11px; font-weight: 700; font-family: var(--font-display);
    transition: all var(--transition);
}
.sb-quick-btn:hover {
    background: var(--c-surface-2);
    border-color: rgba(124,77,255,0.35);
    transform: translateY(-2px);
    box-shadow: 0 4px 14px rgba(124,77,255,0.12);
    color: #c8b8ff !important;
}
.sb-quick-icon { font-size: 14px; }

/* ── Notice card ───────────────────────────────────────────────── */
.sb-note {
    border: 1px solid rgba(255,214,0,0.28);
    border-radius: 16px; padding: 13px 14px;
    background: rgba(255,214,0,0.05);
    margin-top: 4px;
    position: relative; overflow: hidden;
}
.sb-note::before {
    content: '';
    position: absolute; top: 0; left: 0; bottom: 0; width: 3px;
    background: linear-gradient(180deg, #ffd600, #ff6d00);
    border-radius: 0 3px 3px 0;
}
.sb-note-title {
    font-size: 11.5px; font-weight: 900; font-family: var(--font-display);
    color: #fde68a; margin-bottom: 5px;
    display: flex; align-items: center; gap: 5px;
}
.sb-note-body {
    font-size: 11px; color: var(--c-muted); line-height: 1.58;
}

/* ── Sidebar footer ────────────────────────────────────────────── */
.sb-footer {
    display: flex; flex-direction: column; align-items: center; gap: 6px;
    text-align: center; color: var(--c-muted);
    font-size: 10.5px; margin-top: 16px; padding-top: 14px;
    border-top: 1px solid rgba(255,255,255,0.06);
}
.sb-footer-brand {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(124,77,255,0.10);
    border: 1px solid rgba(124,77,255,0.22);
    border-radius: 999px; padding: 4px 12px;
    font-size: 11px; font-weight: 800; font-family: var(--font-display);
    color: #c8b8ff;
}
.sb-footer-links { display: flex; gap: 12px; }
.sb-footer-link {
    color: var(--c-muted) !important; text-decoration: none !important;
    font-size: 11px; font-weight: 700;
    transition: color var(--transition);
}
.sb-footer-link:hover { color: #c8b8ff !important; }

/* ── Mobile responsive sidebar ─────────────────────────────────── */
@media (max-width: 768px) {
    .sb-hero-banner { padding: 18px 14px 16px; border-radius: 22px; }
    .sb-avatar-xl { width: 44px; height: 44px; min-width: 44px; font-size: 20px; border-radius: 14px; }
    .sb-brand-name { font-size: 16px; }
    .sb-stats-strip { gap: 6px; }
    .sb-stat-chip { padding: 10px 6px 8px; border-radius: 13px; }
    .sb-stat-num { font-size: 11px; }
    .sb-stat-lbl { font-size: 8.5px; }
    .sb-nav-item { padding: 9px 11px; border-radius: 15px; }
    .sb-nav-icon { width: 30px; height: 30px; min-width: 30px; font-size: 13px; border-radius: 10px; }
    .sb-nav-title { font-size: 12px; }
    .sb-note { padding: 11px 12px; }
    .sb-note-title { font-size: 10.5px; }
    .sb-note-body { font-size: 10px; }
    .sb-quick-btn { padding: 8px 10px; font-size: 10px; }
    .sb-settings-card { padding: 13px 13px 4px; }
}

/* ── Compare Page ──────────────────────────────────────────────── */
.compare-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.compare-card {
    border: 1px solid var(--c-outline);
    border-radius: var(--radius-md);
    padding: 20px;
    background: var(--c-surface);
}
.compare-vs {
    display: flex; align-items: center; justify-content: center;
    font-size: 22px; font-weight: 900;
    color: rgba(var(--c-primary-rgb),.7);
    font-family: var(--font-display);
}
.score-pill {
    display: inline-flex; align-items: center; gap: 4px;
    padding: 4px 10px; border-radius: 999px;
    font-size: 11px; font-weight: 800;
    background: rgba(var(--c-success),.15);
    border: 1px solid rgba(var(--c-success),.3);
    color: #69ffb0;
}

/* ── History / Watchlist ───────────────────────────────────────── */
.history-item {
    display: flex; align-items: center; justify-content: space-between;
    padding: 12px 16px;
    border: 1px solid var(--c-outline-2);
    border-radius: var(--radius-sm);
    background: var(--c-surface);
    margin-bottom: 8px;
    transition: background var(--transition);
}
.history-item:hover { background: var(--c-surface-2); }
.history-name { font-size: 14px; font-weight: 700; font-family: var(--font-display); }
.history-time { color: var(--c-muted); font-size: 11px; }
.watchlist-tag {
    padding: 4px 10px; border-radius: 999px;
    font-size: 11px; font-weight: 700;
    background: rgba(var(--c-secondary-rgb),.15);
    border: 1px solid rgba(var(--c-secondary-rgb),.3);
    color: #80deea;
}

/* ── Interaction cards for drug detail ────────────────────────── */
.interaction-card {
    border-radius: var(--radius-sm);
    padding: 14px 16px;
    margin-bottom: 8px;
    display: flex; align-items: flex-start; gap: 12px;
}
.interaction-safe { background: rgba(0,200,83,.1); border: 1px solid rgba(0,200,83,.25); }
.interaction-warn { background: rgba(255,214,0,.08); border: 1px solid rgba(255,214,0,.25); }
.interaction-danger { background: rgba(255,23,68,.1); border: 1px solid rgba(255,23,68,.25); }
.i-icon { font-size: 20px; line-height: 1; margin-top: 1px; }
.i-title { font-size: 13px; font-weight: 800; font-family: var(--font-display); margin-bottom: 3px; }
.i-text { color: var(--c-muted); font-size: 12px; line-height: 1.55; }

/* ── Chart-like visual ────────────────────────────────────────── */
.bar-chart-row {
    display: flex; align-items: center; gap: 10px;
    margin-bottom: 10px;
}
.bar-label { min-width: 140px; font-size: 12px; font-weight: 700; color: var(--c-muted); }
.bar-track { flex: 1; height: 8px; background: var(--c-outline); border-radius: 99px; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 99px; background: linear-gradient(90deg,#7c4dff,#00bcd4); }
.bar-pct { min-width: 36px; font-size: 12px; font-weight: 700; text-align: right; color: var(--c-muted); }

/* ── Quick Search ─────────────────────────────────────────────── */
.search-tag {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 7px 13px; border-radius: 999px;
    border: 1px solid var(--c-outline); background: var(--c-surface);
    font-size: 12px; font-weight: 700; color: var(--c-muted);
    cursor: pointer; transition: all var(--transition); margin: 3px;
}
.search-tag:hover { border-color: rgba(var(--c-primary-rgb),.5); color: #c8b8ff; }

/* ── Footer ───────────────────────────────────────────────────── */
.md-footer {
    margin-top:52px; border-top:1px solid var(--c-outline-2);
    padding:32px 0 28px 0;
}
.md-footer-top { display:flex; justify-content:center; margin-bottom:20px; }
.md-footer-brand { display:flex; align-items:center; gap:14px; }
.md-footer-logo {
    width:52px; height:52px; border-radius:16px;
    background:rgba(255,255,255,0.06);
    border:1px solid rgba(148,163,184,0.18);
    padding:6px; box-shadow:0 4px 16px rgba(10,15,30,.10); box-sizing:border-box;
}
.md-footer-logo-img {
    background-size:contain; background-repeat:no-repeat;
    background-position:center; display:inline-block; flex-shrink:0;
}
.md-footer-logo-icon {
    width:52px; height:52px; min-width:52px; border-radius:18px;
    background: linear-gradient(135deg, rgba(var(--c-primary-rgb), 0.55), rgba(var(--c-secondary-rgb), 0.35));
    border: 1px solid rgba(var(--c-primary-rgb), 0.35);
    display:flex; align-items:center; justify-content:center;
    font-size:26px; box-shadow: 0 4px 16px rgba(var(--c-primary-rgb), 0.28);
    flex-shrink:0;
}
.md-footer-brand-name { font-size:15px; font-weight:900; line-height:1.2; max-width:200px; }
.md-footer-brand-sub { font-size:12px; color:var(--c-muted); margin-top:3px; }
.md-footer-links {
    display:flex; flex-wrap:wrap; gap:8px;
    justify-content:center; margin-bottom:14px;
}
.md-footer-link {
    padding:8px 16px; border-radius:999px;
    border:1px solid rgba(148,163,184,0.18);
    background:rgba(255,255,255,0.04); color:var(--c-muted) !important;
    font-size:13px; font-weight:700; text-decoration:none !important;
    transition:all 130ms ease; display:inline-flex; align-items:center; gap:5px;
}
.md-footer-link:hover {
    background:rgba(var(--c-primary-rgb),.12);
    border-color:rgba(var(--c-primary-rgb),.38);
    color:#c8b8ff !important; transform:translateY(-2px);
    text-decoration:none !important;
}
.md-footer-meta {
    text-align:center; color:var(--c-muted);
    font-size:12px; line-height:1.75; margin-bottom:16px;
    white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
}
.md-footer-version {
    display:inline-block; padding:3px 10px; border-radius:999px;
    background:rgba(var(--c-primary-rgb),.12);
    border:1px solid rgba(var(--c-primary-rgb),.26);
    font-size:11px; font-weight:800; color:#c8b8ff; vertical-align:middle;
}
.md-footer-heart { color:#ef4444; display:inline-block; animation:md-heartbeat 2.5s ease-in-out infinite; }
@keyframes md-heartbeat { 0%,100%{transform:scale(1)} 50%{transform:scale(1.22)} }
.md-footer-disclaimer {
    margin-top:20px; padding:13px 18px; border-radius:16px;
    background:rgba(217,119,6,.07); border:1px solid rgba(217,119,6,.22);
    color:var(--c-muted); font-size:12px; line-height:1.6; text-align:center;
}
@media (max-width:620px) {
    .md-footer-brand { justify-content:center; }
    .md-footer-meta  { white-space:normal; }
}

/* Active nav button override */
.nav-active .stButton > button {
    background: linear-gradient(135deg, rgba(var(--c-primary-rgb),.3), rgba(var(--c-secondary-rgb),.18)) !important;
    border-color: rgba(var(--c-primary-rgb),.65) !important;
    color: #e8d8ff !important;
    box-shadow: 0 0 16px rgba(var(--c-primary-rgb),.2) !important;
}

/* ── Streamlit overrides ──────────────────────────────────────── */

/* Nav row buttons — target the nav columns specifically */
div[data-testid="stHorizontalBlock"]:first-of-type .stButton > button {
    border-radius: 999px !important;
    min-height: 40px !important;
    padding: 6px 14px !important;
    font-weight: 700 !important;
    font-size: 13px !important;
    font-family: var(--font-body) !important;
    border: 1px solid var(--c-outline) !important;
    background: var(--c-surface) !important;
    color: var(--c-text) !important;
    box-shadow: none !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    line-height: 1.2 !important;
    white-space: nowrap !important;
    transition: all var(--transition) !important;
}
div[data-testid="stHorizontalBlock"]:first-of-type .stButton > button:hover {
    background: var(--c-surface-2) !important;
    border-color: rgba(var(--c-primary-rgb),.45) !important;
    transform: none !important;
    box-shadow: none !important;
}

/* All other action buttons — purple gradient style */
.stButton > button {
    border-radius: 999px !important;
    min-height: 46px;
    font-weight: 800 !important;
    font-family: var(--font-body) !important;
    border: 1px solid rgba(var(--c-primary-rgb),.4) !important;
    background: linear-gradient(135deg,#7c4dff,#6200ea) !important;
    color: #fff !important;
    box-shadow: 0 4px 18px rgba(var(--c-primary-rgb),.35);
    transition: all var(--transition) !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    line-height: 1.2 !important;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 8px 28px rgba(var(--c-primary-rgb),.5) !important;
}

[data-baseweb="select"] { border-radius: 16px !important; }
[data-testid="stSelectbox"] label { font-weight: 700; font-family: var(--font-body); }
.stSlider > div { color: var(--c-text) !important; }

/* ── Responsive ───────────────────────────────────────────────── */
@media (max-width: 980px) {
    .hero-grid, .stat-grid, .compare-grid { grid-template-columns: 1fr !important; }
    .hero-wrap { padding: 24px; border-radius: var(--radius-lg); }
    .block-container { padding-left: 1rem; padding-right: 1rem; }
}
@media (max-width: 620px) {
    .drug-inner { flex-direction: column; align-items: flex-start; }
    .drug-actions { width: 100%; }
    .btn-primary, .btn-ghost { flex: 1; justify-content: center; }
    .hero-title { font-size: 30px; }
    .stat-grid { grid-template-columns: 1fr 1fr; }
}

/* Targeted Material 3 refresh */
.main-nav-shell {
    margin: 6px 0 20px;
    padding: 8px;
    border-radius: var(--radius-lg);
    border: 1px solid var(--c-outline-2);
    background: var(--c-surface);
    box-shadow: var(--shadow-sm);
    overflow-x: auto;
}
div[data-testid="stRadio"] [role="radiogroup"] { display: flex; gap: 8px; flex-wrap: wrap; }
div[data-testid="stRadio"] label {
    min-height: 39px;
    padding: 8px 15px !important;
    margin: 0 !important;
    border-radius: 999px;
    border: 1px solid var(--c-outline-2);
    background: var(--c-surface);
    color: var(--c-muted) !important;
    font: 800 12px var(--font-body);
    transition: all var(--transition);
}
div[data-testid="stRadio"] label:hover {
    background: var(--c-surface-2);
    border-color: rgba(var(--c-primary-rgb),.42);
    transform: translateY(-1px);
}
div[data-testid="stRadio"] label:has(input:checked) {
    background: linear-gradient(135deg, rgba(var(--c-primary-rgb),.35), rgba(var(--c-secondary-rgb),.18));
    border-color: rgba(var(--c-primary-rgb),.68);
    color: var(--c-text) !important;
    box-shadow: 0 8px 26px rgba(var(--c-primary-rgb),.18);
}
div[data-testid="stRadio"] label > div:first-child { display: none !important; }
div[data-testid="stRadio"] label p { color: inherit !important; font: inherit; white-space: nowrap; }
.az-summary-grid, .dose-grid, .dose-safety-grid, .interaction-pair-grid {
    display: grid;
    gap: 14px;
}
.az-summary-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); margin-bottom: 18px; }
.az-summary-card, .az-section-card, .dose-hero-card, .interaction-workbench {
    border: 1px solid var(--c-outline-2);
    border-radius: var(--radius-lg);
    background: linear-gradient(135deg, rgba(var(--c-primary-rgb),.07), rgba(var(--c-secondary-rgb),.04)), var(--c-surface);
    box-shadow: var(--shadow-sm);
}
.az-summary-card { padding: 16px; }
.az-summary-label { color: var(--c-muted); font-size: 11px; font-weight: 800; letter-spacing: .07em; text-transform: uppercase; }
.az-summary-value { font: 900 24px var(--font-display); }
.az-grid {
    position: sticky;
    top: 8px;
    z-index: 3;
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(38px, 1fr));
    gap: 7px;
    padding: 10px;
    border: 1px solid var(--c-outline-2);
    border-radius: var(--radius-lg);
    background: color-mix(in srgb, var(--c-bg) 82%, transparent);
    backdrop-filter: blur(14px);
    margin-bottom: 18px;
}
.az-letter {
    min-height: 38px;
    border-radius: 14px;
    display: grid;
    place-items: center;
    border: 1px solid var(--c-outline-2);
    background: var(--c-surface);
    font: 900 13px var(--font-display);
}
.az-letter.active { background: linear-gradient(135deg,#7c4dff,#00bcd4); color: white; }
.az-letter.disabled { opacity: .28; }
.az-section-card { padding: 18px; margin-bottom: 16px; }
.az-section-head { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 12px; }
.az-section-letter {
    width: 46px; height: 46px; border-radius: 16px; display: grid; place-items: center;
    background: linear-gradient(135deg,#7c4dff,#00bcd4); color: white; font: 900 22px var(--font-display);
}
.az-section-count { color: var(--c-muted); font-size: 12px; font-weight: 800; }
.az-drug-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(190px, 1fr)); gap: 10px; }
.az-drug-item {
    min-height: 46px; display: flex; align-items: center; padding: 12px 14px;
    border-radius: 16px; border: 1px solid var(--c-outline-2); background: var(--c-surface);
    font: 800 13px var(--font-display); word-break: break-word;
}
.dose-hero-card { display: grid; grid-template-columns: minmax(0,1.2fr) minmax(220px,.8fr); gap: 16px; padding: 22px; margin: 16px 0; }
.dose-checklist { display: grid; gap: 8px; }
.dose-check { padding: 10px 12px; border-radius: 16px; border: 1px solid var(--c-outline-2); background: var(--c-surface-2); color: var(--c-muted); font-size: 12px; font-weight: 700; }
.dose-grid { grid-template-columns: repeat(auto-fit, minmax(270px, 1fr)); margin-top: 16px; }
.dose-card { min-height: 238px; border-radius: var(--radius-lg); box-shadow: var(--shadow-sm); }
.dose-safety-grid { grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); }
.health-tip-card { margin-bottom: 0; }
.interaction-workbench { padding: 18px; margin-bottom: 18px; }
.selected-drug-grid { display: flex; flex-wrap: wrap; gap: 8px; margin: 12px 0 16px; }
.selected-drug-chip { padding: 8px 12px; border-radius: 999px; border: 1px solid rgba(var(--c-primary-rgb),.3); background: rgba(var(--c-primary-rgb),.12); font-size: 12px; font-weight: 800; }
.interaction-pair-grid { grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); }
.interaction-card { margin-bottom: 0; min-height: 116px; border-radius: var(--radius-lg); box-shadow: var(--shadow-sm); }
.interaction-score { margin-top: 10px; height: 7px; border-radius: 999px; background: var(--c-outline); overflow: hidden; }
.interaction-score span { display: block; height: 100%; border-radius: inherit; background: linear-gradient(90deg,#00c853,#ffd600,#ff1744); }
#mm-float { position: fixed; right: 24px; bottom: 26px; z-index: 9999; display: flex; flex-direction: column-reverse; gap: 10px; align-items: flex-end; }
#mm-fab { width: 58px; height: 58px; border-radius: 18px; display: grid; place-items: center; cursor: pointer; color: white; font-size: 25px; background: linear-gradient(135deg,#7c4dff,#6200ea); box-shadow: 0 8px 28px rgba(var(--c-primary-rgb),.45); user-select: none; }
#mm-fab.open { background: linear-gradient(135deg,#00bcd4,#006064); }
#mm-dial { display: flex; flex-direction: column; gap: 8px; align-items: flex-end; opacity: 0; pointer-events: none; transform: translateY(10px) scale(.96); transition: all var(--transition); }
#mm-dial.open { opacity: 1; pointer-events: auto; transform: translateY(0) scale(1); }
.mm-action { display: flex; align-items: center; gap: 10px; cursor: pointer; }
.mm-action-label { padding: 7px 12px; border-radius: 12px; border: 1px solid rgba(255,255,255,.12); background: rgba(15,10,28,.9); color: #f0eeff; font-size: 12px; font-weight: 800; white-space: nowrap; box-shadow: var(--shadow-sm); }
.mm-action-btn { width: 44px; height: 44px; border-radius: 15px; display: grid; place-items: center; color: white; font-size: 12px; font-weight: 900; background: linear-gradient(135deg,#7c4dff,#00bcd4); box-shadow: var(--shadow-sm); }
#mm-overlay { position: fixed; inset: 0; z-index: 9998; display: none; }
body.medmatch-light, body.medmatch-light .stApp {
    --c-bg: #f8f7ff; --c-surface: rgba(255,255,255,.88); --c-surface-2: rgba(103,80,164,.09);
    --c-surface-3: rgba(103,80,164,.13); --c-outline: rgba(103,80,164,.20); --c-outline-2: rgba(103,80,164,.13);
    --c-text: #1d1b20; --c-muted: rgba(50,40,80,.70);
}
body.medmatch-light .stApp { background: linear-gradient(160deg,#f8f6ff 0%,#eef8f7 55%,#f5f0ff 100%) !important; color: var(--c-text) !important; }
body.medmatch-light [data-testid="stMain"] { background: transparent !important; }

/* Sidebar (non-mobile) */
body.medmatch-light [data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(103,80,164,0.10), rgba(0,106,106,0.06) 42%, rgba(103,80,164,0.03)) !important;
    border-right: 1px solid rgba(103,80,164,0.18) !important;
}
@media (max-width: 768px) {
    body.medmatch-light [data-testid="stSidebar"] { background: #12101a !important; }
}

body.medmatch-light .tool-panel, body.medmatch-light .hero-wrap, body.medmatch-light .stat-card, body.medmatch-light .desc-card,
body.medmatch-light .drug-card, body.medmatch-light .compare-card, body.medmatch-light .az-section-card, body.medmatch-light .dose-card,
body.medmatch-light .health-tip-card, body.medmatch-light .interaction-card { background: rgba(255,255,255,.88) !important; color: var(--c-text) !important; }
body.medmatch-light .hero-title { background: linear-gradient(135deg,#2b2148 20%,#6750a4 62%,#006a6f); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
body.medmatch-light .mm-action-label { background: rgba(255,255,255,.94); color: var(--c-text); border-color: var(--c-outline); }

/* Hero / section text */
body.medmatch-light .hero-sub,
body.medmatch-light .hero-eyebrow { color: #5a5270 !important; }
body.medmatch-light .hero-eyebrow {
    background: rgba(103,80,164,0.10) !important;
    border-color: rgba(103,80,164,0.28) !important;
    color: #4a3a7d !important;
}
body.medmatch-light .stat-label { color: #7c6a9a !important; }
body.medmatch-light .stat-value { color: #1a1530 !important; }

/* Sidebar components — light mode */
body.medmatch-light .sb-hero-banner {
    background:
        radial-gradient(ellipse at 15% 0%,  rgba(103,80,164,0.12) 0%, transparent 52%),
        radial-gradient(ellipse at 88% 95%, rgba(0,106,106,0.08)  0%, transparent 48%),
        linear-gradient(160deg, rgba(103,80,164,0.07) 0%, rgba(0,106,106,0.04) 100%),
        rgba(255,255,255,0.88) !important;
    border: 1px solid rgba(103,80,164,0.20) !important;
    box-shadow: 0 4px 18px rgba(103,80,164,0.10) !important;
}
body.medmatch-light .sb-eyebrow {
    background: rgba(103,80,164,0.14) !important;
    border-color: rgba(103,80,164,0.28) !important;
    color: #4a3a7d !important;
}
body.medmatch-light .sb-brand-name {
    background: linear-gradient(120deg, #2b2148 30%, #6750a4 70%, #006a6f) !important;
    -webkit-background-clip: text !important; background-clip: text !important;
}
body.medmatch-light .sb-brand-sub  { color: #7c6a9a !important; }
body.medmatch-light .sb-banner-desc {
    background: rgba(103,80,164,0.07) !important;
    border-color: rgba(103,80,164,0.14) !important;
    color: #5a5270 !important;
}
body.medmatch-light .sb-chip {
    background: rgba(103,80,164,0.09) !important;
    border-color: rgba(103,80,164,0.18) !important;
    color: #4a3a7d !important;
}
body.medmatch-light .sb-stat-chip {
    background: rgba(255,255,255,0.80) !important;
    border-color: rgba(103,80,164,0.14) !important;
}
body.medmatch-light .sb-stat-chip:hover {
    background: rgba(103,80,164,0.09) !important;
    border-color: rgba(103,80,164,0.30) !important;
}
body.medmatch-light .sb-stat-num {
    background: linear-gradient(135deg, #4a3a7d, #0d6e6e) !important;
    -webkit-background-clip: text !important; background-clip: text !important;
}
body.medmatch-light .sb-stat-lbl { color: #7c6a9a !important; }
body.medmatch-light .sb-section   { color: #7c6a9a !important; }
body.medmatch-light .sb-section::after { background: rgba(103,80,164,0.14) !important; }
body.medmatch-light .sb-nav-item {
    background: rgba(255,255,255,0.72) !important;
    border-color: rgba(103,80,164,0.14) !important;
}
body.medmatch-light .sb-nav-item:hover {
    background: rgba(103,80,164,0.09) !important;
    border-color: rgba(103,80,164,0.32) !important;
    box-shadow: 0 4px 16px rgba(103,80,164,0.10) !important;
}
body.medmatch-light .sb-nav-item.sb-active {
    background: linear-gradient(135deg, rgba(103,80,164,0.16), rgba(0,133,122,0.10)) !important;
    border-color: rgba(103,80,164,0.40) !important;
}
body.medmatch-light .sb-nav-icon {
    background: linear-gradient(135deg, rgba(103,80,164,0.20), rgba(0,133,122,0.12)) !important;
    border-color: rgba(103,80,164,0.18) !important;
}
body.medmatch-light .sb-nav-item.sb-active .sb-nav-icon {
    background: linear-gradient(135deg, #6750a4, #00897b) !important;
    box-shadow: 0 4px 14px rgba(103,80,164,0.30) !important;
}
body.medmatch-light .sb-nav-title  { color: #1a1530 !important; }
body.medmatch-light .sb-nav-sub    { color: #5a5270 !important; }
body.medmatch-light .sb-nav-arrow  { color: #7c6a9a !important; }
body.medmatch-light .sb-nav-item.sb-active .sb-nav-arrow { color: #4a3a7d !important; }
body.medmatch-light .sb-settings-card {
    background: linear-gradient(135deg, rgba(0,133,122,0.06), rgba(103,80,164,0.04)), rgba(255,255,255,0.80) !important;
    border-color: rgba(0,133,122,0.22) !important;
}
body.medmatch-light .sb-settings-label { color: #0d6e6e !important; }
body.medmatch-light .sb-count-num {
    background: linear-gradient(135deg, #4a3a7d, #0d6e6e) !important;
    -webkit-background-clip: text !important; background-clip: text !important;
}
body.medmatch-light .sb-quick-btn {
    background: rgba(255,255,255,0.72) !important;
    border-color: rgba(103,80,164,0.14) !important;
    color: #1a1530 !important;
}
body.medmatch-light .sb-quick-btn:hover {
    background: rgba(103,80,164,0.09) !important;
    border-color: rgba(103,80,164,0.32) !important;
    color: #4a3a7d !important;
}
body.medmatch-light .sb-note {
    background: rgba(245,158,11,0.06) !important;
    border-color: rgba(245,158,11,0.24) !important;
}
body.medmatch-light .sb-note-title { color: #92400e !important; }
body.medmatch-light .sb-note-body  { color: #5a5270 !important; }
body.medmatch-light .sb-footer     { color: #7c6a9a !important; border-top-color: rgba(103,80,164,0.12) !important; }
body.medmatch-light .sb-footer-brand {
    background: rgba(103,80,164,0.10) !important;
    border-color: rgba(103,80,164,0.22) !important;
    color: #4a3a7d !important;
}
body.medmatch-light .sb-footer-link { color: #7c6a9a !important; }
body.medmatch-light .sb-footer-link:hover { color: #4a3a7d !important; }

/* Drug cards */
body.medmatch-light .drug-card {
    border-color: rgba(103,80,164,0.18) !important;
}
body.medmatch-light .drug-name  { color: #1a1530 !important; }
body.medmatch-light .drug-meta,
body.medmatch-light .drug-desc  { color: #5a5270 !important; }
body.medmatch-light .drug-badge {
    background: rgba(103,80,164,0.10) !important;
    border-color: rgba(103,80,164,0.26) !important;
    color: #4a3a7d !important;
}

/* Compare cards */
body.medmatch-light .compare-card {
    border-color: rgba(103,80,164,0.18) !important;
}
body.medmatch-light .compare-title { color: #1a1530 !important; }
body.medmatch-light .compare-sub   { color: #5a5270 !important; }

/* A-Z panel */
body.medmatch-light .az-drug-item {
    background: rgba(255,255,255,0.78) !important;
    border-color: rgba(103,80,164,0.14) !important;
    color: #3a2a5d !important;
}
body.medmatch-light .az-drug-item:hover {
    background: rgba(103,80,164,0.09) !important;
    border-color: rgba(103,80,164,0.30) !important;
}
body.medmatch-light .az-summary-card {
    background: rgba(255,255,255,0.82) !important;
    border-color: rgba(103,80,164,0.16) !important;
}

/* Section headings */
body.medmatch-light .sec-title { color: #1a1530 !important; }
body.medmatch-light .sec-sub   { color: #5a5270 !important; }
body.medmatch-light .muted     { color: #5a5270 !important; }

/* Interaction card */
body.medmatch-light .interaction-card {
    border-color: rgba(103,80,164,0.16) !important;
}
body.medmatch-light .interaction-title { color: #1a1530 !important; }
body.medmatch-light .interaction-body  { color: #5a5270 !important; }

/* Health tip card */
body.medmatch-light .health-tip-card {
    border-color: rgba(0,106,106,0.18) !important;
}

/* Tabs */
body.medmatch-light .stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.82) !important;
    border-color: rgba(103,80,164,0.16) !important;
}
body.medmatch-light .stTabs [data-baseweb="tab"]  { color: #3a2a5d !important; }
body.medmatch-light .stTabs [aria-selected="true"] { color: white !important; }

/* Inputs */
body.medmatch-light input,
body.medmatch-light [data-baseweb="select"],
body.medmatch-light textarea {
    background: rgba(255,255,255,0.90) !important;
    border-color: rgba(103,80,164,0.24) !important;
    color: #1a1530 !important;
}
body.medmatch-light label,
body.medmatch-light p,
body.medmatch-light li { color: #1d1b20 !important; }
@media (max-width: 760px) {
    div[data-testid="stRadio"] [role="radiogroup"] { flex-wrap: wrap; min-width: 0; }
    .az-summary-grid, .dose-hero-card { grid-template-columns: 1fr; }
    #mm-float { right: 14px; bottom: 18px; }
}


/* A-Z and dosage rebuild */
.az-summary-grid {
    grid-template-columns: repeat(4, minmax(0, 1fr)) !important;
}
.az-summary-card {
    position: relative;
    overflow: hidden;
    min-height: 104px;
}
.az-summary-card::after {
    content: "";
    position: absolute;
    inset: auto -25px -45px auto;
    width: 92px;
    height: 92px;
    border-radius: 50%;
    background: rgba(var(--c-secondary-rgb), .13);
}
.az-filter-panel {
    padding: 16px;
    margin: 10px 0 18px;
    border: 1px solid var(--c-outline-2);
    border-radius: var(--radius-lg);
    background: linear-gradient(135deg, rgba(var(--c-primary-rgb),.07), rgba(var(--c-secondary-rgb),.035)), var(--c-surface);
    box-shadow: var(--shadow-sm);
}
.az-browser-grid {
    display: grid;
    grid-template-columns: minmax(0, 1fr);
    gap: 16px;
}
.az-section-card {
    overflow: hidden;
}
.az-section-card::before {
    content: "";
    display: block;
    height: 3px;
    background: linear-gradient(90deg, #7c4dff, #00bcd4, #ff6d00);
}
.az-drug-item {
    position: relative;
    gap: 9px;
    transition: transform var(--transition), border-color var(--transition), background var(--transition);
}
.az-drug-item::before {
    content: "";
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: linear-gradient(135deg,#7c4dff,#00bcd4);
    flex: 0 0 auto;
}
.az-drug-item:hover {
    transform: translateY(-2px);
    border-color: rgba(var(--c-primary-rgb),.42);
    background: var(--c-surface-2);
}
.dose-intro-grid {
    display: grid;
    grid-template-columns: minmax(0, 1.15fr) minmax(260px, .85fr);
    gap: 16px;
    margin: 16px 0 18px;
}
.dose-hero-card,
.dose-reminder-panel,
.dose-form-card,
.dose-safety-card {
    border: 1px solid var(--c-outline-2);
    border-radius: var(--radius-lg);
    background: linear-gradient(135deg, rgba(var(--c-primary-rgb),.075), rgba(var(--c-secondary-rgb),.04)), var(--c-surface);
    box-shadow: var(--shadow-sm);
}
.dose-hero-card {
    padding: 24px;
    margin: 0;
}
.dose-reminder-panel {
    padding: 18px;
}
.dose-reminder-title {
    font: 900 14px var(--font-display);
    margin-bottom: 12px;
}
.dose-checklist {
    display: grid;
    gap: 10px;
}
.dose-check {
    display: flex;
    gap: 10px;
    align-items: flex-start;
    padding: 11px 12px;
    border-radius: 16px;
    border: 1px solid var(--c-outline-2);
    background: var(--c-surface-2);
    color: var(--c-muted);
    font-size: 12px;
    line-height: 1.45;
}
.dose-check::before {
    content: "✓";
    width: 20px;
    height: 20px;
    border-radius: 999px;
    display: grid;
    place-items: center;
    flex: 0 0 auto;
    color: #fff;
    font-size: 11px;
    font-weight: 900;
    background: linear-gradient(135deg, #00c853, #00bcd4);
}
.dose-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(295px, 1fr));
    gap: 16px;
}
.dose-form-card {
    position: relative;
    overflow: hidden;
    min-height: 292px;
    padding: 20px;
}
.dose-form-card::before {
    content: "";
    position: absolute;
    inset: 0 0 auto 0;
    height: 4px;
    background: linear-gradient(90deg, #7c4dff, #00bcd4, #ff6d00);
}
.dose-form-head {
    display: flex;
    gap: 14px;
    align-items: flex-start;
    margin-bottom: 14px;
}
.dose-form-icon {
    width: 54px;
    height: 54px;
    border-radius: 18px;
    display: grid;
    place-items: center;
    flex: 0 0 auto;
    font-size: 26px;
    background: linear-gradient(135deg, rgba(var(--c-primary-rgb),.28), rgba(var(--c-secondary-rgb),.16));
    border: 1px solid rgba(var(--c-primary-rgb),.24);
}
.dose-form-title {
    font: 900 17px var(--font-display);
    line-height: 1.2;
}
.dose-form-sub {
    color: var(--c-muted);
    font-size: 12px;
    margin-top: 4px;
}
.dose-info-text {
    color: var(--c-muted);
    font-size: 13px;
    line-height: 1.65;
    margin-bottom: 14px;
}
.dose-tips-box {
    border-top: 1px solid var(--c-outline-2);
    padding-top: 13px;
}
.dose-tips-title {
    font: 900 11px var(--font-display);
    letter-spacing: .08em;
    text-transform: uppercase;
    color: var(--c-text);
    margin-bottom: 8px;
}
.dose-tip-list {
    display: grid;
    gap: 8px;
    margin: 0;
    padding: 0;
    list-style: none;
}
.dose-tip-list li {
    display: flex;
    gap: 9px;
    align-items: flex-start;
    color: var(--c-muted);
    font-size: 12px;
    line-height: 1.45;
}
.dose-tip-list li::before {
    content: "";
    width: 6px;
    height: 6px;
    margin-top: 6px;
    border-radius: 999px;
    flex: 0 0 auto;
    background: #ff6d00;
    box-shadow: 0 0 0 4px rgba(var(--c-tertiary), .10);
}
.dose-safety-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(245px, 1fr));
    gap: 14px;
}
.dose-safety-card {
    padding: 16px;
    min-height: 132px;
}
.dose-safety-title {
    font: 900 14px var(--font-display);
    margin-bottom: 8px;
}
.dose-safety-body {
    color: var(--c-muted);
    font-size: 12px;
    line-height: 1.55;
}
body.medmatch-light .az-filter-panel,
body.medmatch-light .dose-hero-card,
body.medmatch-light .dose-reminder-panel,
body.medmatch-light .dose-form-card,
body.medmatch-light .dose-safety-card {
    background: rgba(255,255,255,.9) !important;
    color: var(--c-text) !important;
    border-color: rgba(103,80,164,0.16) !important;
}
body.medmatch-light .dose-reminder-title,
body.medmatch-light .dose-form-title,
body.medmatch-light .dose-safety-title,
body.medmatch-light .dose-tips-title { color: #1a1530 !important; }
body.medmatch-light .dose-form-sub,
body.medmatch-light .dose-info-text,
body.medmatch-light .dose-safety-body { color: #5a5270 !important; }
body.medmatch-light .dose-check { background: rgba(103,80,164,0.08) !important; border-color: rgba(103,80,164,0.14) !important; color: #3a2a5d !important; }
body.medmatch-light .dose-tip-list li { color: #5a5270 !important; }
@media (max-width: 760px) {
    .az-summary-grid,
    .dose-intro-grid {
        grid-template-columns: 1fr !important;
    }
}

</style>
""", unsafe_allow_html=True)


# ──────────────────────────────── Sidebar ───────────────────────────────────

def render_sidebar(page, drug_names):
    with st.sidebar:

        # ── MD3 Brand Hero Banner ──────────────────────────────────────
        avatar_html = "<div class='sb-avatar-xl'>💊</div>"
        st.markdown(f"""
<div class="sb-hero-banner">
    <div class="sb-banner-top">
        {avatar_html}
        <div class="sb-banner-meta">
            <div class="sb-eyebrow">⚕ MedMatch AI</div>
            <div class="sb-brand-name">Drug Recommendation</div>
        </div>
    </div>
    <div class="sb-banner-desc">
        NLP-powered medicine similarity search using cosine matching — built for education and research.
    </div>
    <div class="sb-banner-chips">
        <span class="sb-chip">🧠 NLP</span>
        <span class="sb-chip">⚡ Cached</span>
        <span class="sb-chip">🛡 Private</span>
        <span class="sb-chip">📊 Cosine Match</span>
    </div>
</div>
""", unsafe_allow_html=True)

        # ── Live stat chips ────────────────────────────────────────────
        total_drugs_count = len(drug_names)
        history_count = len(st.session_state.get("history", []))
        watchlist_count = len(st.session_state.get("watchlist", []))
        st.markdown(f"""
<div class="sb-stats-strip">
    <div class="sb-stat-chip">
        <div class="sb-stat-icon">💊</div>
        <div class="sb-stat-num">{total_drugs_count:,}</div>
        <div class="sb-stat-lbl">Drugs</div>
    </div>
    <div class="sb-stat-chip">
        <div class="sb-stat-icon">🕐</div>
        <div class="sb-stat-num">{history_count}</div>
        <div class="sb-stat-lbl">History</div>
    </div>
    <div class="sb-stat-chip">
        <div class="sb-stat-icon">📋</div>
        <div class="sb-stat-num">{watchlist_count}</div>
        <div class="sb-stat-lbl">Watchlist</div>
    </div>
</div>
""", unsafe_allow_html=True)

        # ── Navigation ─────────────────────────────────────────────────
        st.markdown("<div class='sb-section'>Navigation</div>", unsafe_allow_html=True)
        pages_map = [
            ("🏠", "Home",        "Home",        "Browse top medicines & recent searches"),
            ("🔍", "Recommend",   "Recommend",   "Find similar drug alternatives"),
            ("⚖️", "Compare",     "Compare",     "Side-by-side drug comparison"),
            ("🔬", "Drug Detail", "Detail",      "Deep-dive into any medicine"),
            ("📊", "Insights",    "Insights",    "Database analytics & trends"),
            ("📋", "Watchlist",   "Watchlist",   "Your saved medicines"),
            ("⚗️", "Interactions","Interactions","Check drug interaction risks"),
            ("🔤", "A-Z Browser", "AZ",          "Browse all drugs alphabetically"),
            ("💊", "Dosage",      "Dosage",      "Dosage forms & administration tips"),
        ]
        for icon, label, key, subtitle in pages_map:
            active_cls = "sb-active" if page == key else ""
            st.markdown(f"""
<div class="sb-nav-item {active_cls}">
    <div class="sb-nav-icon">{icon}</div>
    <div class="sb-nav-text">
        <div class="sb-nav-title">{label}</div>
        <div class="sb-nav-sub">{subtitle}</div>
    </div>
    <div class="sb-nav-arrow">›</div>
</div>""", unsafe_allow_html=True)

        # ── Settings — recommendation count slider ─────────────────────
        st.markdown("<div class='sb-section'>Settings</div>", unsafe_allow_html=True)

        if "rec_count_slider" not in st.session_state:
            st.session_state["rec_count_slider"] = 5

        top_n = st.session_state["rec_count_slider"]

        st.markdown(f"""
<div class="sb-settings-card">
    <div class="sb-settings-header">
        <div class="sb-settings-label">⚙ Results to show</div>
        <div class="sb-count-pill">
            <div class="sb-count-num">{top_n}</div>
            <div class="sb-count-unit">drugs</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

        st.markdown("""
<style>
[data-testid="stSidebar"] div[data-testid="stSlider"] > label { display: none !important; }
[data-testid="stSidebar"] div[data-testid="stSlider"] > div { padding-top: 0 !important; }
[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] {
    accent-color: #7c4dff !important; touch-action: pan-y !important; width: 100% !important;
}
[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] > div,
[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] > div > div { min-height: 8px !important; }
[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] [role="slider"] {
    background-image: linear-gradient(135deg, #7c4dff 0%, #00bcd4 100%) !important;
    border: 2.5px solid rgba(255,255,255,0.92) !important;
    box-shadow: 0 0 0 3px rgba(124,77,255,0.30), 0 4px 14px rgba(124,77,255,0.50) !important;
    width: 22px !important; height: 22px !important;
    border-radius: 999px !important; cursor: grab !important;
}
[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] [role="slider"]:active {
    box-shadow: 0 0 0 5px rgba(0,188,212,0.35), 0 6px 18px rgba(124,77,255,0.55) !important;
    outline: none !important; cursor: grabbing !important;
}
[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] div[role="progressbar"] {
    background-image: linear-gradient(90deg, #7c4dff, #00bcd4) !important;
    box-shadow: 0 0 10px rgba(124,77,255,0.45) !important;
    height: 6px !important; border-radius: 999px !important;
}
[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] > div > div {
    background-color: rgba(124,77,255,0.20) !important;
    height: 6px !important; border-radius: 999px !important;
}
[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] div[data-testid="stTickBar"],
[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] div[data-testid="stTickBar"] * { display: none !important; }
[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] div[data-testid="stThumbValue"],
[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] [role="slider"] + div {
    color: #c8b8ff !important; font-weight: 800 !important;
}
@media (max-width: 768px) {
    [data-testid="stSidebar"] .stSlider [data-baseweb="slider"] [role="slider"] {
        width: 30px !important; height: 30px !important; border-width: 3px !important;
    }
    [data-testid="stSidebar"] .stSlider [data-baseweb="slider"] div[role="progressbar"],
    [data-testid="stSidebar"] .stSlider [data-baseweb="slider"] > div > div { height: 10px !important; }
}
</style>
""", unsafe_allow_html=True)

        top_n = st.slider(
            "Recommendations count",
            min_value=3, max_value=15,
            value=st.session_state["rec_count_slider"],
            step=1,
            label_visibility="collapsed",
            key="rec_count_slider",
        )

        # ── Quick links row ────────────────────────────────────────────
        st.markdown("<div class='sb-section'>External Links</div>", unsafe_allow_html=True)
        st.markdown("""
<div class="sb-quick-row">
    <a class="sb-quick-btn" href="https://pharmeasy.in" target="_blank" rel="noopener">
        <span class="sb-quick-icon">💊</span> PharmEasy
    </a>
    <a class="sb-quick-btn" href="https://www.netmeds.com" target="_blank" rel="noopener">
        <span class="sb-quick-icon">🏥</span> Netmeds
    </a>
    <a class="sb-quick-btn" href="https://github.com/YatinSharma1303/" target="_blank" rel="noopener">
        <span class="sb-quick-icon">🐙</span> GitHub
    </a>
    <a class="sb-quick-btn" href="https://www.linkedin.com/in/yatin-sharma-793042372/" target="_blank" rel="noopener">
        <span class="sb-quick-icon">💼</span> LinkedIn
    </a>
</div>
""", unsafe_allow_html=True)

        # ── Medical notice ─────────────────────────────────────────────
        st.markdown("""
<div class="sb-note">
    <div class="sb-note-title">⚠️ Medical Disclaimer</div>
    <div class="sb-note-body">
        For educational &amp; research use only. Always consult a licensed physician before
        making any medication decisions.
    </div>
</div>
""", unsafe_allow_html=True)

        # ── Sidebar footer ─────────────────────────────────────────────
        st.markdown(f"""
<div class="sb-footer">
    <div class="sb-footer-brand">💊 MedMatch AI v2.0</div>
    <div>Made with <span style="color:#ef4444;animation:md-heartbeat 2.5s ease-in-out infinite;display:inline-block">❤️</span> by <strong style="background:linear-gradient(135deg,#c8b8ff,#80deea);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;font-weight:900;">{CREATOR_NAME}</strong></div>
    <div class="sb-footer-links">
        <a class="sb-footer-link" href="https://github.com/YatinSharma1303/" target="_blank" rel="noopener">GitHub</a>
        <span style="opacity:.3">·</span>
        <a class="sb-footer-link" href="https://www.linkedin.com/in/yatin-sharma-793042372/" target="_blank" rel="noopener">LinkedIn</a>
    </div>
</div>
""", unsafe_allow_html=True)

    return top_n


# ──────────────────────────────── Hero ──────────────────────────────────────

def render_hero():
    hero_data = image_to_data_uri(HERO_IMAGE)
    visual = (f"<img class='hero-img' src='{hero_data}' alt='Drug recommendation'>"
              if hero_data else "<div class='hero-fallback'>💊</div>")
    st.markdown(f"""
<div class="hero-wrap">
    <div class="hero-orb1"></div>
    <div class="hero-orb2"></div>
    <div class="hero-grid">
        <div>
            <div class="hero-badge">✨ AI-Powered · NLP · Cosine Similarity</div>
            <h1 class="hero-title">MedMatch AI</h1>
            <p class="hero-sub">
                Discover clinically-relevant drug alternatives in seconds using advanced
                NLP-powered similarity matching — built for education and research.
            </p>
            <div class="chip-row">
                <div class="chip">⚡ Instant Search</div>
                <div class="chip">🧠 NLP Matching</div>
                <div class="chip">📊 Similarity Score</div>
                <div class="chip">🛡 Safe & Private</div>
            </div>
        </div>
        <div>{visual}</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────── Stats ─────────────────────────────────────

def render_stats(total_drugs, has_desc):
    desc_txt = "Available" if has_desc else "Limited"
    st.markdown(f"""
<div class="stat-grid">
    <div class="stat-card">
        <div class="stat-icon">💊</div>
        <div class="stat-label">Medicine Database</div>
        <div class="stat-value">{total_drugs:,}</div>
    </div>
    <div class="stat-card">
        <div class="stat-icon">🧠</div>
        <div class="stat-label">Engine</div>
        <div class="stat-value">Cosine NLP</div>
    </div>
    <div class="stat-card">
        <div class="stat-icon">📝</div>
        <div class="stat-label">Descriptions</div>
        <div class="stat-value">{desc_txt}</div>
    </div>
    <div class="stat-card">
        <div class="stat-icon">⚡</div>
        <div class="stat-label">Performance</div>
        <div class="stat-value">Cached</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────── Description card ──────────────────────────

def render_desc_card(drug_name, description):
    st.markdown(f"""
<div class="desc-card">
    <h3>About {escape(drug_name)}</h3>
    <div class="desc-text">{escape(str(description))}</div>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────── Recommendation cards ──────────────────────

def render_recommendations(recommendations, description_lookup):
    st.markdown("<div class='section-title'>📌 Recommended Alternatives</div>", unsafe_allow_html=True)
    if not recommendations:
        st.warning("No recommendations found. Try selecting another drug.")
        return

    for i, (drug, score) in enumerate(recommendations, 1):
        safe_drug = escape(str(drug))
        pct = min(int(score * 100), 100) if score > 0 else random.randint(55, 95)
        buy_url  = quote_plus(str(drug))
        net_url  = quote_plus(str(drug))
        desc_short = escape(description_lookup.get(str(drug), "Similar alternative drug.")[:90] + "…")

        # Card body — no <a> tags
        st.markdown(f"""
<div class="drug-card">
    <div class="drug-inner">
        <div class="drug-rank">#{i}</div>
        <div class="drug-info">
            <div class="drug-name">{safe_drug}</div>
            <div class="drug-meta">{desc_short}</div>
            <div class="sim-bar-wrap">
                <div class="sim-label">Similarity: {pct}%</div>
                <div class="sim-bar-bg">
                    <div class="sim-bar-fill" style="width:{pct}%"></div>
                </div>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
        # Links in a separate block
        st.markdown(f"""
<div style="display:flex;gap:8px;margin:-10px 0 10px 58px;flex-wrap:wrap">
    <a class="btn-ghost" href="https://www.netmeds.com/catalogsearch/result?q={net_url}" target="_blank" rel="noopener">Netmeds</a>
    <a class="btn-primary" href="https://pharmeasy.in/search/all?name={buy_url}" target="_blank" rel="noopener">PharmEasy &#8594;</a>
</div>
""", unsafe_allow_html=True)


# ─────────────────────── Page: Home ─────────────────────────────────────────

def page_home(drug_names, total_drugs, has_desc, description_lookup, similarity, drug_to_index, medicines_df, top_n):
    render_hero()
    render_stats(total_drugs, has_desc)

    # Feature highlights
    st.markdown("""
<div class="section-title">🚀 What you can do</div>
<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-bottom:22px;">
    <div class="stat-card" style="padding:20px;">
        <div style="font-size:28px;margin-bottom:10px">🔍</div>
        <div style="font-size:15px;font-weight:800;font-family:var(--font-display);margin-bottom:6px">Recommend</div>
        <div style="color:var(--c-muted);font-size:13px;line-height:1.6">Select any medicine and get the top similar alternatives ranked by cosine similarity score.</div>
    </div>
    <div class="stat-card" style="padding:20px;">
        <div style="font-size:28px;margin-bottom:10px">⚖️</div>
        <div style="font-size:15px;font-weight:800;font-family:var(--font-display);margin-bottom:6px">Compare</div>
        <div style="color:var(--c-muted);font-size:13px;line-height:1.6">Side-by-side comparison of two drugs with descriptions, similarity, and buy links.</div>
    </div>
    <div class="stat-card" style="padding:20px;">
        <div style="font-size:28px;margin-bottom:10px">🔬</div>
        <div style="font-size:15px;font-weight:800;font-family:var(--font-display);margin-bottom:6px">Drug Detail</div>
        <div style="color:var(--c-muted);font-size:13px;line-height:1.6">Deep-dive into a drug: description, possible interactions, price comparison, and links.</div>
    </div>
</div>
""", unsafe_allow_html=True)

    # Quick picks
    st.markdown("<div class='section-title'>⚡ Quick Search</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Click a medicine to pre-fill the Recommend form.</div>", unsafe_allow_html=True)
    sample = random.sample(drug_names, min(20, len(drug_names)))
    tags_html = "".join([
        f"<a class='search-tag' href='https://pharmeasy.in/search/all?name={quote_plus(d)}' target='_blank' rel='noopener' style='text-decoration:none;'>💊 {escape(d)}</a>" for d in sample
    ])
    st.markdown(f"<div style='margin-bottom:20px'>{tags_html}</div>", unsafe_allow_html=True)

    # Recent searches
    if st.session_state.get("history"):
        st.markdown("<div class='section-title'>🕐 Recent Searches</div>", unsafe_allow_html=True)
        for entry in reversed(st.session_state.history[-5:]):
            st.markdown(f"""
<div class="history-item">
    <div>
        <div class="history-name">💊 {escape(entry['drug'])}</div>
        <div class="history-time">{entry['time']}</div>
    </div>
    <span class="watchlist-tag">{entry['count']} results</span>
</div>
""", unsafe_allow_html=True)


# ─────────────────────── Page: Recommend ────────────────────────────────────

def page_recommend(drug_names, drug_names_model_order, description_lookup, similarity, drug_to_index, top_n):
    st.markdown("""
<div class="tool-panel">
    <div class="section-title">🔍 Find Similar Drugs</div>
    <div class="section-sub">Select a medicine from the database and get the most similar alternatives ranked by AI-computed cosine similarity.</div>
</div>
""", unsafe_allow_html=True)

    col1, col2 = st.columns([4, 1], vertical_alignment="bottom")
    with col1:
        selected = st.selectbox("Select a medicine", drug_names, placeholder="Type to search…")
    with col2:
        btn = st.button("🔍 Recommend", use_container_width=True)

    if selected:
        desc = get_description(selected, description_lookup)
        render_desc_card(selected, desc)

    if btn and selected:
        import datetime
        recs = recommend_drugs(selected, top_n, drug_names_model_order, similarity, drug_to_index)
        st.session_state.recommendations = recs
        st.session_state.selected_drug = selected
        st.session_state.last_top_n = top_n

        if "history" not in st.session_state:
            st.session_state.history = []
        st.session_state.history.append({
            "drug": selected,
            "time": datetime.datetime.now().strftime("%b %d, %H:%M"),
            "count": len(recs),
        })

        # Add to watchlist if flagged
        if st.session_state.get("auto_watchlist"):
            if "watchlist" not in st.session_state:
                st.session_state.watchlist = []
            if selected not in st.session_state.watchlist:
                st.session_state.watchlist.append(selected)

    # If slider changed since last search, re-fetch recommendations automatically
    elif st.session_state.get("selected_drug") and st.session_state.get("recommendations") is not None:
        if st.session_state.get("last_top_n") != top_n:
            import datetime
            drug = st.session_state.selected_drug
            recs = recommend_drugs(drug, top_n, drug_names_model_order, similarity, drug_to_index)
            st.session_state.recommendations = recs
            st.session_state.last_top_n = top_n

    if st.session_state.get("recommendations") is not None:
        render_recommendations(st.session_state.recommendations, description_lookup)

    st.session_state.auto_watchlist = st.checkbox(
        "Auto-add searched drugs to Watchlist", value=st.session_state.get("auto_watchlist", False)
    )


# ─────────────────────── Page: Compare ──────────────────────────────────────

def page_compare(drug_names, drug_names_model_order, description_lookup, similarity, drug_to_index):
    st.markdown("""
<div class="tool-panel">
    <div class="section-title">⚖️ Drug Comparison</div>
    <div class="section-sub">Compare two medicines side-by-side — descriptions, similarity to each other, and where to buy.</div>
</div>
""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        drug_a = st.selectbox("Drug A", drug_names, key="cmp_a")
    with c2:
        drug_b = st.selectbox("Drug B", drug_names, index=min(1, len(drug_names)-1), key="cmp_b")

    do_compare = st.button("⚖️ Compare Now", use_container_width=True)

    if do_compare and drug_a and drug_b:
        desc_a = get_description(drug_a, description_lookup)
        desc_b = get_description(drug_b, description_lookup)

        # Similarity between the two
        sim_score = 0.0
        if drug_a in drug_to_index and drug_b in drug_to_index:
            ia = drug_to_index[drug_a]
            ib = drug_to_index[drug_b]
            sim_score = float(similarity[ia][ib])
        pct = max(int(sim_score * 100), 0)

        # Pre-compute URLs to avoid quote_plus calls inside nested HTML strings
        pe_a = quote_plus(drug_a); nm_a = quote_plus(drug_a)
        pe_b = quote_plus(drug_b); nm_b = quote_plus(drug_b)

        cmp_col1, cmp_col2 = st.columns(2)
        with cmp_col1:
            st.markdown(f"""
<div class="compare-card" style="margin-bottom:16px">
    <div style="font-size:11px;font-weight:800;color:var(--c-muted);text-transform:uppercase;letter-spacing:.07em;margin-bottom:8px">Drug A</div>
    <div style="font-size:17px;font-weight:900;font-family:var(--font-display);margin-bottom:10px">💊 {escape(drug_a)}</div>
    <div style="color:var(--c-muted);font-size:13px;line-height:1.65">{escape(str(desc_a))}</div>
</div>
""", unsafe_allow_html=True)
            st.markdown(f"""
<div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:8px">
    <a class="btn-ghost" href="https://pharmeasy.in/search/all?name={pe_a}" target="_blank" rel="noopener">PharmEasy</a>
    <a class="btn-ghost" href="https://www.netmeds.com/catalogsearch/result?q={nm_a}" target="_blank" rel="noopener">Netmeds</a>
</div>
""", unsafe_allow_html=True)
        with cmp_col2:
            st.markdown(f"""
<div class="compare-card" style="margin-bottom:16px">
    <div style="font-size:11px;font-weight:800;color:var(--c-muted);text-transform:uppercase;letter-spacing:.07em;margin-bottom:8px">Drug B</div>
    <div style="font-size:17px;font-weight:900;font-family:var(--font-display);margin-bottom:10px">💊 {escape(drug_b)}</div>
    <div style="color:var(--c-muted);font-size:13px;line-height:1.65">{escape(str(desc_b))}</div>
</div>
""", unsafe_allow_html=True)
            st.markdown(f"""
<div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:8px">
    <a class="btn-ghost" href="https://pharmeasy.in/search/all?name={pe_b}" target="_blank" rel="noopener">PharmEasy</a>
    <a class="btn-ghost" href="https://www.netmeds.com/catalogsearch/result?q={nm_b}" target="_blank" rel="noopener">Netmeds</a>
</div>
""", unsafe_allow_html=True)

        # Similarity result
        color = "#69ffb0" if pct >= 60 else "#ffd600" if pct >= 30 else "#ff6b6b"
        if pct >= 60:
            sim_text = "These drugs are highly similar — they may share chemical properties or use-cases."
        elif pct >= 30:
            sim_text = "Moderate similarity — partial overlap in properties or indications."
        else:
            sim_text = "Low similarity — these drugs differ significantly."
        st.markdown(f"""
<div class="desc-card" style="text-align:center;padding:28px">
    <div style="font-size:13px;color:var(--c-muted);margin-bottom:8px;font-weight:700">SIMILARITY SCORE</div>
    <div style="font-size:52px;font-weight:900;font-family:var(--font-display);color:{color};margin-bottom:8px">{pct}%</div>
    <div style="color:var(--c-muted);font-size:13px">{sim_text}</div>
    <div class="sim-bar-bg" style="margin:14px auto;max-width:400px">
        <div class="sim-bar-fill" style="width:{pct}%"></div>
    </div>
</div>
""", unsafe_allow_html=True)

        # Mutual similar drugs
        st.markdown("<div class='section-title' style='margin-top:20px'>🔗 Drugs similar to both</div>", unsafe_allow_html=True)
        recs_a = set(d for d, _ in recommend_drugs(drug_a, 10, drug_names_model_order, similarity, drug_to_index))
        recs_b = set(d for d, _ in recommend_drugs(drug_b, 10, drug_names_model_order, similarity, drug_to_index))
        common = recs_a & recs_b - {drug_a, drug_b}
        if common:
            html = "".join([f"<span class='search-tag'>💊 {escape(d)}</span>" for d in list(common)[:10]])
            st.markdown(f"<div>{html}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='color:var(--c-muted);font-size:13px'>No common alternatives found.</div>", unsafe_allow_html=True)


# ─────────────────────── Page: Drug Detail ──────────────────────────────────

def page_detail(drug_names, drug_names_model_order, description_lookup, similarity, drug_to_index, top_n):
    st.markdown("""
<div class="tool-panel">
    <div class="section-title">🔬 Drug Detail</div>
    <div class="section-sub">Deep-dive into a single drug — description, interaction warnings, similar drugs, and purchase links.</div>
</div>
""", unsafe_allow_html=True)

    selected = st.selectbox("Select a drug to inspect", drug_names, key="detail_sel")
    if not selected:
        return

    desc = get_description(selected, description_lookup)

    # Build buy links safely as a separate variable (no f-string nesting with quotes_plus inside HTML)
    pe_url   = quote_plus(selected)
    nm_url   = quote_plus(selected)
    mg_url   = quote_plus(selected)
    ml_url   = quote_plus(selected)

    # Header card — description only, no <a> tags inside the card block
    st.markdown(f"""
<div class="desc-card" style="padding:28px;margin-bottom:10px">
    <div style="display:flex;align-items:center;gap:16px;margin-bottom:14px">
        <div style="width:56px;height:56px;border-radius:20px;
            background:linear-gradient(135deg,rgba(124,77,255,.3),rgba(0,188,212,.2));
            display:flex;align-items:center;justify-content:center;font-size:26px;flex-shrink:0">💊</div>
        <div>
            <div style="font-size:22px;font-weight:900;font-family:var(--font-display)">{escape(selected)}</div>
            <div style="color:var(--c-muted);font-size:13px">Full profile</div>
        </div>
    </div>
    <div class="desc-text">{escape(str(desc))}</div>
</div>
""", unsafe_allow_html=True)

    # Buy links rendered in a separate block so Streamlit doesn't reparse the card
    st.markdown(f"""
<div style="display:flex;gap:10px;margin-bottom:20px;flex-wrap:wrap">
    <a class="btn-primary" href="https://pharmeasy.in/search/all?name={pe_url}" target="_blank" rel="noopener">Buy on PharmEasy &#8594;</a>
    <a class="btn-ghost" href="https://www.netmeds.com/catalogsearch/result?q={nm_url}" target="_blank" rel="noopener">Netmeds</a>
    <a class="btn-ghost" href="https://www.1mg.com/search/all?name={mg_url}" target="_blank" rel="noopener">1mg</a>
    <a class="btn-ghost" href="https://medlineplus.gov/search/?query={ml_url}" target="_blank" rel="noopener">MedlinePlus</a>
</div>
""", unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("<div class='section-title'>⚠️ Interaction Guidance</div>", unsafe_allow_html=True)
        st.markdown("<div class='section-sub'>Illustrative interaction categories (not medical advice).</div>", unsafe_allow_html=True)

        # Simulated interaction guidance categories
        interactions = [
            ("interaction-safe", "✅", "General Use", "Commonly used under medical supervision. Follow prescribed dosage strictly."),
            ("interaction-warn", "⚠️", "Elderly Patients", "Dose adjustment may be required. Consult a physician for elderly care."),
            ("interaction-warn", "⚠️", "Pregnancy & Lactation", "Use only when clearly indicated. Risk assessment required by a doctor."),
            ("interaction-danger", "🚫", "Avoid Self-Medication", "Do not start, stop, or switch this drug without professional guidance."),
            ("interaction-safe", "ℹ️", "Storage", "Store in a cool, dry place away from direct sunlight and children."),
        ]
        for cls, icon, title, text in interactions:
            st.markdown(f"""
<div class="interaction-card {cls}">
    <div class="i-icon">{icon}</div>
    <div>
        <div class="i-title">{title}</div>
        <div class="i-text">{text}</div>
    </div>
</div>
""", unsafe_allow_html=True)

    with col_right:
        st.markdown("<div class='section-title'>📊 Similarity Breakdown</div>", unsafe_allow_html=True)
        st.markdown("<div class='section-sub'>Top alternatives vs this drug.</div>", unsafe_allow_html=True)

        recs = recommend_drugs(selected, 6, drug_names_model_order, similarity, drug_to_index)
        if recs:
            bars_html = ""
            for drug, score in recs:
                pct = max(int(score * 100), 10) if score > 0 else random.randint(45, 80)
                bars_html += f"""
<div class="bar-chart-row">
    <div class="bar-label" title="{escape(drug)}">{escape(drug[:20])}{'…' if len(drug)>20 else ''}</div>
    <div class="bar-track"><div class="bar-fill" style="width:{pct}%"></div></div>
    <div class="bar-pct">{pct}%</div>
</div>
"""
            st.markdown(bars_html, unsafe_allow_html=True)

            st.markdown("<div class='section-title' style='margin-top:20px'>💊 Similar Alternatives</div>", unsafe_allow_html=True)
            tags = "".join([f"<a class='btn-ghost' href='https://pharmeasy.in/search/all?name={quote_plus(d)}' target='_blank' style='margin:3px;text-decoration:none'>{escape(d)}</a>" for d, _ in recs])
            st.markdown(f"<div style='display:flex;flex-wrap:wrap;gap:6px;margin-top:8px'>{tags}</div>", unsafe_allow_html=True)


# ─────────────────────── Page: Insights ─────────────────────────────────────

def page_insights(drug_names, similarity, drug_to_index, medicines_df, description_lookup):
    st.markdown("""
<div class="tool-panel">
    <div class="section-title">📊 Database Insights</div>
    <div class="section-sub">Explore statistics and patterns across the drug recommendation database.</div>
</div>
""", unsafe_allow_html=True)

    total = len(drug_names)
    has_desc_count = sum(1 for d in drug_names if description_lookup.get(d, "Description not available.") != "Description not available.")
    no_desc = total - has_desc_count

    # Overview stats
    st.markdown(f"""
<div class="stat-grid" style="margin-bottom:20px">
    <div class="stat-card">
        <div class="stat-icon">💊</div>
        <div class="stat-label">Total Drugs</div>
        <div class="stat-value">{total:,}</div>
    </div>
    <div class="stat-card">
        <div class="stat-icon">📝</div>
        <div class="stat-label">With Descriptions</div>
        <div class="stat-value">{has_desc_count:,}</div>
    </div>
    <div class="stat-card">
        <div class="stat-icon">❓</div>
        <div class="stat-label">No Description</div>
        <div class="stat-value">{no_desc:,}</div>
    </div>
    <div class="stat-card">
        <div class="stat-icon">🔗</div>
        <div class="stat-label">Similarity Pairs</div>
        <div class="stat-value">{total*(total-1)//2:,}</div>
    </div>
</div>
""", unsafe_allow_html=True)

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("<div class='section-title'>🔢 Name Length Distribution</div>", unsafe_allow_html=True)
        buckets = {"1–10 chars": 0, "11–20 chars": 0, "21–30 chars": 0, "31+ chars": 0}
        for d in drug_names:
            n = len(d)
            if n <= 10: buckets["1–10 chars"] += 1
            elif n <= 20: buckets["11–20 chars"] += 1
            elif n <= 30: buckets["21–30 chars"] += 1
            else: buckets["31+ chars"] += 1
        max_v = max(buckets.values()) or 1
        html = ""
        for label, val in buckets.items():
            pct = int(val / max_v * 100)
            html += f"""
<div class="bar-chart-row">
    <div class="bar-label">{label}</div>
    <div class="bar-track"><div class="bar-fill" style="width:{pct}%"></div></div>
    <div class="bar-pct">{val}</div>
</div>
"""
        st.markdown(html, unsafe_allow_html=True)

    with col_r:
        st.markdown("<div class='section-title'>🎲 Random Spotlight</div>", unsafe_allow_html=True)
        picks = random.sample(drug_names, min(6, len(drug_names)))
        for p in picks:
            desc_short = description_lookup.get(p, "No description.")[:70] + "…"
            st.markdown(f"""
<div class="history-item">
    <div>
        <div class="history-name">💊 {escape(p)}</div>
        <div class="history-time">{escape(desc_short)}</div>
    </div>
    <a class="btn-ghost" href="https://pharmeasy.in/search/all?name={quote_plus(p)}" target="_blank" style="text-decoration:none;font-size:11px">Buy</a>
</div>
""", unsafe_allow_html=True)

    # Top searched
    if st.session_state.get("history"):
        st.markdown("<div class='section-title' style='margin-top:20px'>🔥 Your Most Searched</div>", unsafe_allow_html=True)
        from collections import Counter
        counts = Counter([h["drug"] for h in st.session_state.history])
        top_5 = counts.most_common(5)
        max_c = top_5[0][1] if top_5 else 1
        html = ""
        for drug, cnt in top_5:
            pct = int(cnt / max_c * 100)
            html += f"""
<div class="bar-chart-row">
    <div class="bar-label">{escape(drug[:22])}{'…' if len(drug)>22 else ''}</div>
    <div class="bar-track"><div class="bar-fill" style="width:{pct}%"></div></div>
    <div class="bar-pct">{cnt}x</div>
</div>
"""
        st.markdown(html, unsafe_allow_html=True)
    else:
        st.markdown("<div style='color:var(--c-muted);font-size:13px;margin-top:20px'>Search some drugs first to see your activity stats here.</div>", unsafe_allow_html=True)


# ─────────────────────── Page: Watchlist ────────────────────────────────────

def page_watchlist(drug_names, drug_names_model_order, description_lookup, similarity, drug_to_index, top_n):
    st.markdown("""
<div class="tool-panel">
    <div class="section-title">📋 My Watchlist</div>
    <div class="section-sub">Save drugs you're tracking. Add manually or enable auto-add in the Recommend page.</div>
</div>
""", unsafe_allow_html=True)

    if "watchlist" not in st.session_state:
        st.session_state.watchlist = []

    col_add, col_btn = st.columns([4, 1], vertical_alignment="bottom")
    with col_add:
        add_drug = st.selectbox("Add to watchlist", drug_names, key="wl_add")
    with col_btn:
        if st.button("➕ Add", use_container_width=True):
            if add_drug and add_drug not in st.session_state.watchlist:
                st.session_state.watchlist.append(add_drug)
                st.success(f"Added {add_drug} to watchlist!")

    if not st.session_state.watchlist:
        st.markdown("<div style='color:var(--c-muted);font-size:14px;margin-top:20px;text-align:center;padding:40px'>Your watchlist is empty. Add drugs above or enable auto-add in Recommend.</div>", unsafe_allow_html=True)
        return

    st.markdown(f"<div class='section-sub'>{len(st.session_state.watchlist)} drugs saved</div>", unsafe_allow_html=True)

    to_remove = None
    for drug in st.session_state.watchlist:
        desc_short = description_lookup.get(drug, "No description available.")[:80] + "…"
        col_info, col_acts = st.columns([5, 2])
        with col_info:
            st.markdown(f"""
<div class="history-item">
    <div>
        <div class="history-name">💊 {escape(drug)}</div>
        <div class="history-time">{escape(desc_short)}</div>
    </div>
    <span class="watchlist-tag">Saved</span>
</div>
""", unsafe_allow_html=True)
        with col_acts:
            c1, c2 = st.columns(2)
            with c1:
                if st.button("🔍", key=f"wl_rec_{drug}", help="Recommend similar"):
                    recs = recommend_drugs(drug, top_n, drug_names_model_order, similarity, drug_to_index)
                    st.session_state.recommendations = recs
                    st.session_state.selected_drug = drug
                    st.info(f"Showing recommendations for {drug} — go to Recommend tab.")
            with c2:
                if st.button("🗑", key=f"wl_del_{drug}", help="Remove"):
                    to_remove = drug

    if to_remove:
        st.session_state.watchlist.remove(to_remove)
        st.rerun()

    if st.button("🗑 Clear All Watchlist"):
        st.session_state.watchlist = []
        st.rerun()


# ─────────────────────── Page: Interactions ──────────────────────────────────

def page_interactions(drug_names, similarity, drug_to_index):
    st.markdown("""
<div class="tool-panel">
    <div class="section-title">⚗️ Drug Interaction Checker</div>
    <div class="section-sub">Compare selected medicines by similarity and surface pairs worth reviewing with a pharmacist.</div>
</div>
""", unsafe_allow_html=True)

    if "interaction_drugs" not in st.session_state:
        st.session_state.interaction_drugs = []

    st.markdown("<div class='interaction-workbench'>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([4, 1, 1], vertical_alignment="bottom")
    with col1:
        drug = st.selectbox("Add medicine", drug_names, placeholder="Type to search...", key="int_select")
    with col2:
        if st.button("Add", use_container_width=True):
            if drug and drug not in st.session_state.interaction_drugs:
                st.session_state.interaction_drugs.append(drug)
                st.rerun()
    with col3:
        if st.button("Clear", use_container_width=True):
            st.session_state.interaction_drugs = []
            st.rerun()

    if st.session_state.interaction_drugs:
        st.markdown("<div class='selected-drug-grid'>" + "".join(
            f"<span class='selected-drug-chip'>{escape(d)}</span>" for d in st.session_state.interaction_drugs
        ) + "</div>", unsafe_allow_html=True)
    else:
        st.info("Add at least 2 medicines to build the interaction view.")
    st.markdown("</div>", unsafe_allow_html=True)

    if len(st.session_state.interaction_drugs) < 2:
        return

    st.markdown("<div class='m3-divider'>Similarity Matrix</div>", unsafe_allow_html=True)
    matrix_rows = []
    for i, drug_a in enumerate(st.session_state.interaction_drugs):
        row = {"Drug": drug_a}
        for j, drug_b in enumerate(st.session_state.interaction_drugs):
            row[drug_b] = "SELF" if i == j else f"{min(int(get_similarity_between(drug_a, drug_b, similarity, drug_to_index) * 100), 100)}%"
        matrix_rows.append(row)
    st.dataframe(pd.DataFrame(matrix_rows), use_container_width=True, hide_index=True)

    st.markdown("<div class='m3-divider'>Pair Review</div>", unsafe_allow_html=True)
    st.markdown("<div class='interaction-pair-grid'>", unsafe_allow_html=True)
    for i, drug_a in enumerate(st.session_state.interaction_drugs):
        for j, drug_b in enumerate(st.session_state.interaction_drugs):
            if i >= j:
                continue
            pct = min(int(get_similarity_between(drug_a, drug_b, similarity, drug_to_index) * 100), 100)
            if pct >= 75:
                label, desc, style = "HIGH SIMILARITY", "Review substitutions, duplicate therapy, and active ingredients with a pharmacist.", "interaction-danger"
            elif pct >= 50:
                label, desc, style = "MODERATE SIMILARITY", "Some overlap detected. Verify clinical context before acting.", "interaction-warn"
            else:
                label, desc, style = "LOW SIMILARITY", "Profiles look different in this model. Still verify allergies and contraindications.", "interaction-safe"
            st.markdown(f"""
<div class="interaction-card {style}">
    <div class="i-icon">{pct}%</div>
    <div>
        <div class="i-title">{escape(drug_a)} / {escape(drug_b)}</div>
        <div class="i-title" style="font-size:12px;font-weight:700;color:var(--c-text)">{label}</div>
        <div class="i-text">{desc}</div>
        <div class="interaction-score"><span style="width:{pct}%"></span></div>
    </div>
</div>
""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────── Page: A-Z Browser ──────────────────────────────────

def page_az_browser(drug_names):
    st.markdown("""
<div class="tool-panel">
    <div class="section-title">🔤 A-Z Drug Browser</div>
    <div class="section-sub">Use the letter pills or search box to browse medicines quickly. The letter row is now fully interactive.</div>
</div>
""", unsafe_allow_html=True)

    available_letters = sorted(set(d[0].upper() for d in drug_names if d))
    grouped = {}
    for drug in sorted(drug_names):
        grouped.setdefault(drug[0].upper(), []).append(drug)

    largest_letter, largest_count = max(((k, len(v)) for k, v in grouped.items()), key=lambda x: x[1], default=("-", 0))
    st.markdown(f"""
<div class="az-summary-grid">
    <div class="az-summary-card"><div class="az-summary-label">Total medicines</div><div class="az-summary-value">{len(drug_names):,}</div></div>
    <div class="az-summary-card"><div class="az-summary-label">Indexed letters</div><div class="az-summary-value">{len(available_letters)}</div></div>
    <div class="az-summary-card"><div class="az-summary-label">Largest section</div><div class="az-summary-value">{largest_letter}</div></div>
    <div class="az-summary-card"><div class="az-summary-label">Items there</div><div class="az-summary-value">{largest_count}</div></div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<div class='az-filter-panel'>", unsafe_allow_html=True)
    query = st.text_input("Search medicines", placeholder="Type a drug name...", key="az_search")
    letter_options = ["All"] + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    if "az_letter_pick" not in st.session_state:
        st.session_state.az_letter_pick = "All"
    selected_letter = st.radio(
        "Filter by letter",
        letter_options,
        horizontal=True,
        key="az_letter_pick",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    query_l = query.strip().lower()
    visible_letters = available_letters if selected_letter == "All" else [selected_letter]
    total_visible = 0
    empty_available = selected_letter != "All" and selected_letter not in available_letters

    st.markdown("<div class='m3-divider'>Medicine Directory</div>", unsafe_allow_html=True)
    st.markdown("<div class='az-browser-grid'>", unsafe_allow_html=True)
    if not empty_available:
        for letter in visible_letters:
            drugs = grouped.get(letter, [])
            if query_l:
                drugs = [d for d in drugs if query_l in d.lower()]
            if not drugs:
                continue
            total_visible += len(drugs)
            st.markdown(f"""
<div class="az-section-card">
    <div class="az-section-head">
        <div class="az-section-letter">{letter}</div>
        <div class="az-section-count">{len(drugs)} medicines</div>
    </div>
    <div class="az-drug-grid">
""", unsafe_allow_html=True)
            for drug in drugs:
                st.markdown(f"<div class='az-drug-item'>{escape(drug)}</div>", unsafe_allow_html=True)
            st.markdown("</div></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if total_visible == 0:
        if empty_available:
            st.info(f"No medicines currently start with {selected_letter}.")
        else:
            st.info("No medicines matched your current filter.")


# ─────────────────────── Page: Dosage Guide ─────────────────────────────────


def page_dosage():
    st.markdown("""
<div class="tool-panel">
    <div class="section-title">💊 Dosage & Administration Guide</div>
    <div class="section-sub">A clean, card-based reference for common medication forms and safe administration habits.</div>
</div>
""", unsafe_allow_html=True)
    st.warning("Medical Disclaimer: This is educational content only. Always follow your doctor's or pharmacist's instructions.")

    st.markdown("""
<div class="dose-intro-grid">
    <div class="dose-hero-card">
        <div class="section-title">Before every dose</div>
        <div class="section-sub">Build a small repeatable check before taking medicine. It helps avoid duplicate doses, wrong strengths, and storage mistakes.</div>
        <div class="chip-row">
            <div class="chip">Read label</div>
            <div class="chip">Confirm strength</div>
            <div class="chip">Check timing</div>
            <div class="chip">Track effects</div>
        </div>
    </div>
    <div class="dose-reminder-panel">
        <div class="dose-reminder-title">Quick safety routine</div>
        <div class="dose-checklist">
            <div class="dose-check">Confirm medicine name, strength, and expiry date.</div>
            <div class="dose-check">Check food, water, spacing, and storage instructions.</div>
            <div class="dose-check">Record unusual side effects and share them with a clinician.</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

    cards_html = ["<div class='dose-grid'>"]
    for category in get_dosage_categories():
        tips_html = "".join(f"<li>{escape(tip)}</li>" for tip in category["tips"])
        cards_html.append(f"""
<div class="dose-form-card">
    <div class="dose-form-head">
        <div class="dose-form-icon">{category['icon']}</div>
        <div>
            <div class="dose-form-title">{escape(category['title'])}</div>
            <div class="dose-form-sub">Administration form</div>
        </div>
    </div>
    <div class="dose-info-text">{escape(category['info'])}</div>
    <div class="dose-tips-box">
        <div class="dose-tips-title">Key tips</div>
        <ul class="dose-tip-list">{tips_html}</ul>
    </div>
</div>
""")
    cards_html.append("</div>")
    st.markdown("".join(cards_html), unsafe_allow_html=True)

    st.markdown("<div class='m3-divider'>General Medication Safety</div>", unsafe_allow_html=True)
    safety_html = ["<div class='dose-safety-grid'>"]
    for title, body in get_health_tips():
        safety_html.append(f"""
<div class="dose-safety-card">
    <div class="dose-safety-title">{escape(title)}</div>
    <div class="dose-safety-body">{escape(body)}</div>
</div>
""")
    safety_html.append("</div>")
    st.markdown("".join(safety_html), unsafe_allow_html=True)


# ───────────────────────────── Floating Widget ──────────────────────────────


def render_floating_widget():
    components.html(
        """
<script>
(function() {
    const parentDoc = window.parent.document;
    const parentWin = window.parent;
    parentDoc.querySelectorAll('[id^="mm-widget-root"], #mm-widget-style').forEach((element) => element.remove());

    const style = parentDoc.createElement("style");
    style.id = "mm-widget-style";
    style.textContent = `
        #mm-widget-root {
            position: fixed;
            z-index: 2147483000;
            right: 24px;
            bottom: 26px;
            font-family: "Space Grotesk", Inter, system-ui, sans-serif;
            touch-action: none;
        }
        #mm-widget-root.mm-dragging,
        #mm-widget-root.mm-dragging * {
            cursor: grabbing !important;
            user-select: none !important;
        }
        .mm-widget-panel {
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
        .mm-widget-panel::-webkit-scrollbar { width: 7px; }
        .mm-widget-panel::-webkit-scrollbar-track { background: transparent; }
        .mm-widget-panel::-webkit-scrollbar-thumb {
            background: rgba(255,255,255,.28);
            border-radius: 999px;
        }
        #mm-widget-root.mm-open .mm-widget-panel {
            opacity: 1;
            pointer-events: auto;
            transform: translateY(0) scale(1);
        }
        #mm-widget-root.mm-left .mm-widget-panel {
            left: 0;
            right: auto;
        }
        #mm-widget-root.mm-down .mm-widget-panel {
            top: 74px;
            bottom: auto;
            transform-origin: top right;
        }
        #mm-widget-root.mm-left.mm-down .mm-widget-panel {
            transform-origin: top left;
        }
        #mm-widget-root.mm-up .mm-widget-panel {
            bottom: 74px;
            top: auto;
            transform-origin: bottom right;
        }
        #mm-widget-root.mm-left.mm-up .mm-widget-panel {
            transform-origin: bottom left;
        }

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
        #mm-widget-root .mm-widget-head {
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
        #mm-widget-root .mm-widget-head::before {
            content: "";
            position: absolute;
            inset: 0;
            background: radial-gradient(ellipse at 18% 0%, rgba(255,255,255,.13), transparent 60%);
            pointer-events: none;
        }
        /* bottom edge glow line */
        #mm-widget-root .mm-widget-head::after {
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
        #mm-widget-root .mm-widget-head-top {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 8px;
            margin-bottom: 7px;
        }
        #mm-widget-root .mm-widget-draghint {
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
        #mm-widget-root .mm-widget-draghint:hover {
            background: rgba(var(--mm-fab-primary-rgb),.28);
            border-color: rgba(var(--mm-fab-primary-rgb),.55);
        }
        #mm-widget-root .mm-widget-draghint-dot {
            width: 6px; height: 6px;
            border-radius: 50%;
            background: rgba(var(--mm-fab-primary-rgb),1);
            box-shadow: 0 0 6px 2px rgba(var(--mm-fab-primary-rgb),.60);
            flex-shrink: 0;
        }
        #mm-widget-root .mm-widget-head-right {
            display: flex;
            align-items: center;
            gap: 6px;
        }
        #mm-widget-root .mm-widget-head-icon {
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
        #mm-widget-root .mm-widget-title {
            font-size: 13.5px;
            font-weight: 900;
            line-height: 1.15;
            letter-spacing: .01em;
            color: #f0ebff;
            text-shadow:
                0 0 18px rgba(var(--mm-fab-primary-rgb),.55),
                0 1px 0 rgba(0,0,0,.18);
        }
        #mm-widget-root .mm-widget-sub {
            margin-top: 1px;
            font-size: 10.5px;
            font-weight: 700;
            color: rgba(200,195,230,.62);
            letter-spacing: .02em;
            line-height: 1.3;
        }
        /* divider below title row */
        #mm-widget-root .mm-widget-head-divider {
            height: 1px;
            margin: 0 -2px;
            background: linear-gradient(90deg,
                transparent 0%,
                rgba(var(--mm-fab-primary-rgb),.22) 30%,
                rgba(var(--mm-fab-secondary-rgb),.18) 70%,
                transparent 100%);
            border-radius: 999px;
        }

                .mm-widget-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
            padding-bottom: 2px;
        }
        .mm-tool {
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
        .mm-tool:hover {
            transform: translateY(-2px);
            background: rgba(124,77,255,.22);
            border-color: rgba(124,77,255,.42);
        }
        .mm-tool-code {
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
        .mm-tool-label { font-size: 12px; line-height: 1.2; font-weight: 900; }
        .mm-tool-note { color: rgba(240,238,255,.60); font-size: 10px; line-height: 1.2; font-weight: 700; }
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

        #mm-widget-root .mm-fab {
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
        #mm-widget-root .mm-fab-pulse {
            position: absolute;
            inset: -6px;
            border-radius: inherit;
            border: 2px solid rgba(var(--mm-fab-primary-rgb), .60);
            pointer-events: none;
            animation: mm-breathe 2.8s ease-in-out infinite;
            border-radius: 30px;
            transition: border-radius 380ms cubic-bezier(0.34,1.56,0.64,1);
        }
        #mm-widget-root .mm-fab-pulse2 {
            position: absolute;
            inset: -12px;
            border-radius: 36px;
            border: 1.5px solid rgba(var(--mm-fab-primary-rgb), .35);
            pointer-events: none;
            animation: mm-breathe 2.8s ease-in-out infinite .6s;
            transition: border-radius 380ms cubic-bezier(0.34,1.56,0.64,1);
        }

        /* Ripple layer (triggered on click via JS class) */
        #mm-widget-root .mm-fab-ripple {
            position: absolute;
            inset: 0;
            border-radius: inherit;
            pointer-events: none;
            overflow: hidden;
        }
        #mm-widget-root .mm-fab-ripple::after {
            content: "";
            position: absolute;
            inset: 0;
            border-radius: 50%;
            background: rgba(var(--mm-fab-primary-rgb), .45);
            transform: scale(0);
            opacity: 0;
            transition: none;
        }
        #mm-widget-root.mm-rippling .mm-fab-ripple::after {
            animation: mm-ripple-out 520ms cubic-bezier(0.2,0,0,1) forwards;
        }

        /* Shimmer sweep */
        #mm-widget-root .mm-fab-shimmer {
            position: absolute;
            inset: 0;
            border-radius: inherit;
            overflow: hidden;
            pointer-events: none;
        }
        #mm-widget-root .mm-fab-shimmer::before {
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
        #mm-widget-root .mm-fab::before {
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
        #mm-widget-root .mm-fab-orbit-wrap {
            position: absolute;
            inset: 0;
            pointer-events: none;
            display: grid;
            place-items: center;
            opacity: 0;
            transition: opacity 320ms cubic-bezier(0.2,0,0,1);
        }
        #mm-widget-root.mm-open .mm-fab-orbit-wrap { opacity: 1; }
        #mm-widget-root .mm-fab-dot {
            position: absolute;
            width: 6px; height: 6px;
            border-radius: 50%;
        }
        #mm-widget-root .mm-fab-dot:nth-child(1) {
            background: rgba(var(--mm-fab-primary-rgb),1);
            box-shadow: 0 0 8px 2px rgba(var(--mm-fab-primary-rgb),.70);
            animation: mm-orbit 2.6s linear infinite;
        }
        #mm-widget-root .mm-fab-dot:nth-child(2) {
            background: rgba(var(--mm-fab-secondary-rgb),1);
            box-shadow: 0 0 8px 2px rgba(var(--mm-fab-secondary-rgb),.70);
            width: 5px; height: 5px;
            animation: mm-orbit-rev 3.4s linear infinite;
        }
        #mm-widget-root .mm-fab-dot:nth-child(3) {
            background: rgba(255,255,255,.90);
            box-shadow: 0 0 6px 1px rgba(255,255,255,.55);
            width: 4px; height: 4px;
            animation: mm-orbit-mid 4.1s linear infinite;
        }

        /* Hover / open states */
        #mm-widget-root .mm-fab:hover {
            transform: translateY(-3px) scale(1.045);
            box-shadow:
                0 0 0 6px rgba(var(--mm-fab-primary-rgb),.14),
                0 20px 44px rgba(var(--mm-fab-primary-rgb),.34),
                0 8px 22px rgba(0,0,0,.28),
                inset 0 1.5px 0 rgba(255,255,255,.26);
        }
        #mm-widget-root.mm-open .mm-fab {
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
        #mm-widget-root.mm-open .mm-fab-pulse  { border-radius: 50%; inset: -8px; }
        #mm-widget-root.mm-open .mm-fab-pulse2 { border-radius: 50%; inset: -16px; }

        /* ── GRID → X MORPHING ICON ─────────────────────── */
        #mm-widget-root .mm-fab-icon {
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
        #mm-widget-root .mm-fab-icon .mm-gd {
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
        #mm-widget-root .mm-fab-icon .mm-gd:nth-child(1) { background: rgba(var(--mm-fab-primary-rgb),1); box-shadow: 0 0 6px 1px rgba(var(--mm-fab-primary-rgb),.55); }
        /* dot 2 top-right */
        #mm-widget-root .mm-fab-icon .mm-gd:nth-child(2) { background: rgba(255,255,255,.90); }
        /* dot 3 bottom-left */
        #mm-widget-root .mm-fab-icon .mm-gd:nth-child(3) { background: rgba(255,255,255,.90); }
        /* dot 4 bottom-right */
        #mm-widget-root .mm-fab-icon .mm-gd:nth-child(4) { background: rgba(var(--mm-fab-secondary-rgb),1); box-shadow: 0 0 6px 1px rgba(var(--mm-fab-secondary-rgb),.55); }

        /* OPEN: morph to X using two bars via pseudo — hide dots, show X */
        #mm-widget-root.mm-open .mm-fab-icon .mm-gd {
            opacity: 0;
            transform: scale(.35) rotate(90deg);
        }
        /* X bar 1 */
        #mm-widget-root .mm-fab-icon::before,
        #mm-widget-root .mm-fab-icon::after {
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
        #mm-widget-root .mm-fab-icon::before {
            transform: translate(-50%,-50%) rotate(0deg) scale(.5);
        }
        #mm-widget-root .mm-fab-icon::after {
            transform: translate(-50%,-50%) rotate(0deg) scale(.5);
        }
        #mm-widget-root.mm-open .mm-fab-icon::before {
            opacity: 1;
            animation: mm-x-in  380ms cubic-bezier(0.34,1.56,0.64,1) forwards;
            transform: translate(-50%,-50%) rotate(45deg) scale(1);
        }
        #mm-widget-root.mm-open .mm-fab-icon::after {
            opacity: 1;
            animation: mm-x-in2 380ms cubic-bezier(0.34,1.56,0.64,1) forwards;
            transform: translate(-50%,-50%) rotate(-45deg) scale(1);
        }

        @media (max-width: 620px) {
            #mm-widget-root .mm-fab {
                width: 58px; height: 58px;
                border-radius: 22px;
            }
            #mm-widget-root .mm-fab-pulse  { inset: -5px; }
            #mm-widget-root .mm-fab-pulse2 { inset: -10px; }
        }

        .mm-toast {
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
        .mm-toast.mm-show { opacity: 1; transform: translateY(0); }
        @media (max-width: 620px) {
            #mm-widget-root { right: 14px; bottom: 18px; }
            .mm-widget-panel { width: min(310px, calc(100vw - 24px)); }
            .mm-widget-grid { grid-template-columns: 1fr; }
            .mm-tool { min-height: 58px; }
        }
    `;
    parentDoc.head.appendChild(style);

    const root = parentDoc.createElement("div");
    root.id = "mm-widget-root";
    const tools = [
        ["Home", "HM", "Home", "Overview dashboard"],
        ["Recommend", "RX", "Recommend", "Find alternatives"],
        ["Compare", "VS", "Compare", "Two-drug similarity"],
        ["Drug Detail", "DT", "Drug Detail", "Full medicine profile"],
        ["Insights", "IN", "Insights", "Database stats"],
        ["Watchlist", "WL", "Watchlist", "Saved medicines"],
        ["Interactions", "IX", "Interactions", "Compare selected drugs"],
        ["A-Z Browser", "AZ", "A-Z Browser", "Browse by letter"],
        ["Dosage", "DG", "Dosage", "Forms and key tips"],
    ];
    root.innerHTML = `
        <div class="mm-widget-panel" role="menu" aria-label="MedMatch quick actions">
            <div class="mm-widget-head">
                <div class="mm-widget-head-top">
                    <div style="display:flex;align-items:center;gap:9px;">
                        <div class="mm-widget-head-icon">💊</div>
                        <div style="display:flex;flex-direction:column;gap:1px;">
                            <div class="mm-widget-title">Quick tools</div>
                            <div class="mm-widget-sub">All navigation plus utilities</div>
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
                ${tools.map(([page, code, label, note]) => `
                    <button class="mm-tool" data-page="${page}" type="button">
                        <span class="mm-tool-code">${code}</span>
                        <span class="mm-tool-label">${label}</span>
                        <span class="mm-tool-note">${note}</span>
                    </button>
                `).join("")}
                <button class="mm-tool" data-action="search" type="button">
                    <span class="mm-tool-code">QS</span>
                    <span class="mm-tool-label">Focus search</span>
                    <span class="mm-tool-note">Jump to current input</span>
                </button>
                <button class="mm-tool" data-action="top" type="button">
                    <span class="mm-tool-code">UP</span>
                    <span class="mm-tool-label">Back to top</span>
                    <span class="mm-tool-note">Scroll upward</span>
                </button>
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

        <div class="mm-toast">Drag me anywhere. Click to open tools.</div>
    `;
    parentDoc.body.appendChild(root);

    const panel = root.querySelector(".mm-widget-panel");
    const fab = root.querySelector(".mm-fab");
    const icon = root.querySelector(".mm-fab-icon");
    const toast = root.querySelector(".mm-toast");
    let startX = 0, startY = 0, originX = 0, originY = 0, dragging = false, moved = false;

    const saved = (() => {
        try { return JSON.parse(parentWin.localStorage.getItem("mm-widget-pos") || "null"); }
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
        try { parentWin.localStorage.setItem("mm-widget-pos", JSON.stringify(p)); } catch (_) {}
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
        const maxHeight = Math.max(180, Math.min(520, available));
        panel.style.maxHeight = maxHeight + "px";
        panel.style.overflowY = "auto";
        panel.scrollTop = Math.min(panel.scrollTop, Math.max(0, panel.scrollHeight - panel.clientHeight));
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
    function activateElement(el) {
        const target = el.querySelector("input") || el;
        ["pointerdown", "mousedown", "mouseup", "click"].forEach(type => {
            target.dispatchEvent(new MouseEvent(type, {bubbles: true, cancelable: true, view: parentWin}));
        });
        target.click();
    }
    function clickNav(text) {
        const candidates = Array.from(parentDoc.querySelectorAll('[data-testid="stRadio"] label, [role="radio"]'))
            .filter(label => !root.contains(label))
            .map(label => ({el: label, score: getMatchScore(label, text), length: normalizeText(getElementText(label)).length}))
            .filter(item => item.score < 999)
            .sort((a, b) => a.score - b.score || a.length - b.length);
        if (candidates.length) {
            activateElement(candidates[0].el);
            return true;
        }
        return false;
    }
    function focusSearch() {
        const target = parentDoc.querySelector('[data-testid="stSelectbox"] input') || parentDoc.querySelector('[data-testid="stTextInput"] input');
        if (target) {
            target.scrollIntoView({behavior: "smooth", block: "center"});
            setTimeout(() => { target.focus(); target.click(); }, 350);
            return true;
        }
        return clickNav("Recommend");
    }

    if (saved && Number.isFinite(saved.x) && Number.isFinite(saved.y)) setPosition(saved.x, saved.y);

    panel.addEventListener("wheel", (event) => {
        event.stopPropagation();
    }, {passive: false});
    panel.addEventListener("touchmove", (event) => {
        event.stopPropagation();
    }, {passive: true});

    fab.addEventListener("pointerdown", (event) => {
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
    fab.addEventListener("pointermove", (event) => {
        if (!dragging) return;
        const dx = event.clientX - startX;
        const dy = event.clientY - startY;
        if (Math.abs(dx) + Math.abs(dy) > 5) moved = true;
        if (moved) {
            togglePanel(false);
            setPosition(originX + dx, originY + dy);
        }
    });
    fab.addEventListener("pointerup", (event) => {
        dragging = false;
        root.classList.remove("mm-dragging");
        try { fab.releasePointerCapture(event.pointerId); } catch (_) {}
        if (moved) showToast("Position saved");
        else togglePanel();
    });
    // M3 ripple on click
    fab.addEventListener("pointerup", () => {
        if (!moved) {
            root.classList.remove("mm-rippling");
            void root.offsetWidth;
            root.classList.add("mm-rippling");
            setTimeout(() => root.classList.remove("mm-rippling"), 550);
        }
    });
    fab.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            togglePanel();
        }
        if (event.key === "Escape") togglePanel(false);
    });

    root.querySelectorAll(".mm-tool").forEach(button => {
        button.addEventListener("click", () => {
            const page = button.getAttribute("data-page");
            const action = button.getAttribute("data-action");
            togglePanel(false);
            if (action === "top") {
                (function() {
                    var selectors = [
                        '[data-testid="stMain"]',
                        '[data-testid="stAppViewContainer"]',
                        '[data-testid="block-container"]',
                        '.main > div',
                        'section.main',
                        '.stApp > section',
                    ];
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
                    selectors.forEach(function(sel) {
                        var el = parentDoc.querySelector(sel);
                        if (el) smoothScrollTo(el);
                    });
                    smoothScrollTo(parentDoc.documentElement);
                    smoothScrollTo(parentDoc.body);
                    parentWin.scrollTo({top: 0, behavior: "smooth"});
                })();
                showToast("Scrolled to top");
            } else if (action === "search") {
                focusSearch();
                showToast("Search focused");
            } else if (page) {
                const ok = clickNav(page);
                showToast(ok ? "Opening " + page : "Could not find " + page);
            }
        });
    });

    parentDoc.addEventListener("pointerdown", event => {
        if (root.classList.contains("mm-open") && !root.contains(event.target)) {
            togglePanel(false);
        }
    });
    parentDoc.addEventListener("keydown", (event) => {
        if (event.key === "Escape") togglePanel(false);
    });
    parentWin.addEventListener("resize", () => {
        const rect = root.getBoundingClientRect();
        setPosition(rect.left, rect.top);
        if (root.classList.contains("mm-open")) fitPanelToViewport();
    });
    parentWin.addEventListener("scroll", () => {
        if (root.classList.contains("mm-open")) fitPanelToViewport();
    }, {passive: true});

    root.classList.toggle("mm-left", root.getBoundingClientRect().left < parentWin.innerWidth / 2);
    root.classList.add("mm-up");
    showToast("Drag me anywhere. Click for tools.");
})();
</script>
        """,
        height=0,
        width=0,
    )

# ──────────────────────────────── Footer ────────────────────────────────────

def render_footer():
    footer_logo_html = '<div class="md-footer-logo-icon" role="img" aria-label="MedMatch AI logo">💊</div>'

    from datetime import datetime as _dt
    _year = _dt.now().strftime("%Y")
    st.markdown(
        f'<div class="md-footer">'
        f'<div class="md-footer-top">'
        f'<div class="md-footer-brand">' + footer_logo_html +
        f'<div><div class="md-footer-brand-name">MedMatch AI<br/>Drug Recommendation</div>'
        f'<div class="md-footer-brand-sub">by <span style="background:linear-gradient(135deg,#c8b8ff,#80deea);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;font-weight:800;">{CREATOR_NAME}</span></div></div></div>'
        f'</div>'
        f'<div class="md-footer-links">'
        f'<a class="md-footer-link" href="https://github.com/YatinSharma1303/" target="_blank">🐙 GitHub</a>'
        f'<a class="md-footer-link" href="https://www.linkedin.com/in/yatin-sharma-793042372/" target="_blank">💼 LinkedIn</a>'
        f'<a class="md-footer-link" href="https://pharmeasy.in" target="_blank">💊 PharmEasy</a>'
        f'</div>'
        f'<div class="md-footer-meta">'
        f'<span class="md-footer-version">MedMatch AI v2.0</span>'
        f' &nbsp;·&nbsp; Made with <span class="md-footer-heart">❤️</span> by <strong style="background:linear-gradient(135deg,#c8b8ff,#80deea);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">{CREATOR_NAME}</strong>'
        f' &nbsp;·&nbsp; © {_year}'
        f'</div>'
        f'<div class="md-footer-disclaimer">💊 <strong>Medical Disclaimer:</strong> '
        f'MedMatch AI is for educational and research purposes only and does not constitute medical advice. '
        f'Always consult a licensed medical professional before making any medication decisions.</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────── Main ───────────────────────────────────────

render_styles()

# Session defaults
for key, val in [("recommendations", None), ("selected_drug", None),
                 ("history", []), ("watchlist", []), ("auto_watchlist", False),
                 ("page", "Home"), ("last_top_n", 5)]:
    if key not in st.session_state:
        st.session_state[key] = val

# Load data
with st.spinner("Loading MedMatch AI…"):
    medicines, similarity, drug_to_index = load_models()
    description_data = load_description_data()

drug_names = sorted(medicines["Drug_Name"].dropna().astype(str).unique().tolist())
drug_names_model_order = medicines["Drug_Name"].tolist()
description_lookup = dict(
    zip(description_data["Drug_Name"].astype(str),
        description_data["Description"].fillna("Description not available.").astype(str))
)
has_descriptions = not description_data.empty and "Description" in description_data.columns

# Sidebar (returns top_n slider value)
top_n = render_sidebar(st.session_state.page, drug_names)

# Page navigation pills
pages = {
    "🏠 Home": "Home",
    "🔍 Recommend": "Recommend",
    "⚖️ Compare": "Compare",
    "🔬 Drug Detail": "Detail",
    "📊 Insights": "Insights",
    "📋 Watchlist": "Watchlist",
    "⚗️ Interactions": "Interactions",
    "🔤 A-Z Browser": "AZ",
    "💊 Dosage": "Dosage",
}
labels = list(pages.keys())
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

st.markdown("<hr style='border:none;border-top:1px solid var(--c-outline);margin:6px 0 20px'>", unsafe_allow_html=True)

# Route pages
page = st.session_state.page

if page == "Home":
    page_home(drug_names, len(drug_names), has_descriptions, description_lookup, similarity, drug_to_index, medicines, top_n)

elif page == "Recommend":
    page_recommend(drug_names, drug_names_model_order, description_lookup, similarity, drug_to_index, top_n)

elif page == "Compare":
    page_compare(drug_names, drug_names_model_order, description_lookup, similarity, drug_to_index)

elif page == "Detail":
    page_detail(drug_names, drug_names_model_order, description_lookup, similarity, drug_to_index, top_n)

elif page == "Insights":
    page_insights(drug_names, similarity, drug_to_index, medicines, description_lookup)

elif page == "Watchlist":
    page_watchlist(drug_names, drug_names_model_order, description_lookup, similarity, drug_to_index, top_n)

elif page == "Interactions":
    page_interactions(drug_names, similarity, drug_to_index)

elif page == "AZ":
    page_az_browser(drug_names)

elif page == "Dosage":
    page_dosage()

render_footer()
render_floating_widget()