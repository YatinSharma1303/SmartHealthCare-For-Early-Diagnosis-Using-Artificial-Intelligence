import base64
from html import escape
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from theme_config import init_theme, get_theme_styles, render_theme_toggle


APP_TITLE = "SmartHealthCare for Early Diagnosis Using Artificial Intelligence"
CREATOR_NAME = "Yatin Sharma"

HERO_IMAGE = "utils/home2.png"


st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


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


def render_styles():
    styles = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200&display=swap');

:root {
    --md-primary: #6750a4;
    --md-primary-rgb: 103, 80, 164;
    --md-secondary: #006a6a;
    --md-secondary-rgb: 0, 106, 106;
    --md-tertiary: #ba1a1a;
    --md-tertiary-rgb: 186, 26, 26;
    --md-error: #ba1a1a;
    --md-amber: #d97706;
    --md-amber-rgb: 217, 119, 6;
    --md-emerald: #059669;
    --md-emerald-rgb: 5, 150, 105;
    --md-rose: #e11d48;
    --md-rose-rgb: 225, 29, 72;
    --md-blue: #2563eb;
    --md-blue-rgb: 37, 99, 235;

    --md-surface: rgba(255,255,255,0.048);
    --md-surface-container: rgba(255,255,255,0.072);
    --md-surface-container-high: rgba(255,255,255,0.105);
    --md-outline: rgba(148, 163, 184, 0.28);
    --md-outline-variant: rgba(148, 163, 184, 0.18);
    --md-soft: rgba(148, 163, 184, 0.96);

    --md-shadow-1: 0 4px 14px rgba(15, 23, 42, 0.08);
    --md-shadow-2: 0 12px 34px rgba(15, 23, 42, 0.12);
    --md-shadow-3: 0 22px 58px rgba(15, 23, 42, 0.16);

    --font-display: 'Outfit', sans-serif;
    --font-body: 'Plus Jakarta Sans', sans-serif;
}

html, body, * {
    font-family: var(--font-body) !important;
    scroll-behavior: smooth;
}

html, body {
    overflow-x: hidden !important;
    max-width: 100vw !important;
}

/* ── FIX: Streamlit sidebar toggle button ligature text overflow ── */
[data-testid="stSidebarCollapsedControl"] {
    overflow: hidden !important;
    max-width: 48px !important;
}
[data-testid="stSidebarCollapsedControl"] span {
    font-family: 'Material Symbols Rounded', 'Material Symbols Outlined' !important;
    font-feature-settings: 'liga' !important;
    -webkit-font-feature-settings: 'liga' !important;
    overflow: hidden !important;
    display: inline-block !important;
    max-width: 24px !important;
    white-space: nowrap !important;
    text-overflow: clip !important;
}

h1, h2, h3, .md-title, .md-section-title {
    font-family: var(--font-display) !important;
}

.block-container {
    max-width: 1240px;
    padding-top: 1.15rem;
    padding-bottom: 2.5rem;
}

/* ── SIDEBAR ─────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    border-right: 1px solid var(--md-outline);
    background:
        radial-gradient(ellipse at 22% 0%,   rgba(var(--md-primary-rgb), 0.22), transparent 44%),
        radial-gradient(ellipse at 90% 58%,  rgba(var(--md-secondary-rgb), 0.14), transparent 42%),
        radial-gradient(ellipse at 10% 100%, rgba(103,58,183,.10),   transparent 40%),
        linear-gradient(175deg, rgba(var(--md-primary-rgb), 0.11) 0%, rgba(var(--md-secondary-rgb), 0.05) 55%, transparent 100%);
    overflow: hidden !important;
}

[data-testid="stSidebarContent"] {
    overflow-x: hidden !important;
}

/* ── Sidebar inner padding wrapper ──────────────────────── */
.md-sb-wrap {
    padding: 14px 14px 20px;
    display: flex; flex-direction: column;
}

/* ── Brand hero card ─────────────────────────────────────── */
.md-sidebar-hero {
    position: relative; overflow: hidden;
    border: 1px solid var(--md-outline);
    border-radius: 22px;
    padding: 16px;
    background:
        linear-gradient(135deg, rgba(var(--md-primary-rgb), 0.14), rgba(var(--md-secondary-rgb), 0.09)),
        var(--md-surface);
    box-shadow: 0 4px 24px rgba(var(--md-primary-rgb), 0.14);
    margin-bottom: 16px;
    word-break: break-word;
    overflow-wrap: break-word;
}
.md-sidebar-hero::before {
    content: ''; position: absolute; top: -40px; right: -40px;
    width: 120px; height: 120px; border-radius: 50%;
    background: radial-gradient(circle, rgba(var(--md-primary-rgb), .24), transparent 70%);
    pointer-events: none;
}
@keyframes sb-hub-wave {
    0%,100% { transform: scale(1) rotate(0deg);   box-shadow: 0 4px 14px rgba(var(--md-primary-rgb),.40); }
    33%     { transform: scale(1.07) rotate(4deg); box-shadow: 0 6px 22px rgba(var(--md-primary-rgb),.65); }
    66%     { transform: scale(1.04) rotate(-3deg);box-shadow: 0 5px 18px rgba(var(--md-primary-rgb),.52); }
}
@keyframes sb-online-home { 0%,100%{box-shadow:0 0 6px rgba(0,200,83,0.55);} 50%{box-shadow:0 0 10px rgba(0,200,83,0.28);} }

/* hero row: logo tile + brand */
.md-sb-hero-row {
    display: flex; align-items: center; gap: 11px; margin-bottom: 10px;
}
.md-sb-logo {
    width: 54px; height: 54px; min-width: 54px; border-radius: 18px;
    background: linear-gradient(135deg, #6750a4, #006a6a);
    display: flex; align-items: center; justify-content: center;
    font-size: 26px; box-shadow: 0 4px 14px rgba(var(--md-primary-rgb), .40);
    flex-shrink: 0;
    transform-origin: center;
    animation: sb-hub-wave 4s ease-in-out infinite;
    position: relative;
}
.md-sb-logo::after {
    content: '';
    position: absolute; bottom: -3px; right: -3px;
    width: 13px; height: 13px; border-radius: 50%;
    background: #00c853;
    border: 2px solid var(--md-bg, #0d0d14);
    box-shadow: 0 0 6px rgba(0,200,83,0.55);
    animation: sb-online-home 2.4s ease-in-out infinite;
}
.md-sb-brand { flex: 1; min-width: 0; }

.md-sidebar-kicker {
    display: inline-flex;
    padding: 3px 9px;
    border-radius: 999px;
    background: rgba(var(--md-primary-rgb), 0.16);
    border: 1px solid rgba(var(--md-primary-rgb), 0.36);
    color: #d7c7ff;
    font-size: 10px;
    font-weight: 850;
    margin-bottom: 3px;
    white-space: normal;
    word-break: break-word;
}

.md-sidebar-title {
    font-family: var(--font-display) !important;
    font-size: 16px;
    font-weight: 950;
    line-height: 1.1;
    background: linear-gradient(135deg, #d7c7ff 0%, #a78bfa 40%, #6750a4 70%, #006a6a 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}

.md-sidebar-text {
    color: var(--md-soft);
    font-size: 11.5px;
    line-height: 1.5;
    margin-top: 2px;
    word-break: break-word;
    overflow-wrap: break-word;
}

/* mini stat chips inside hero */
.md-sb-stats-row {
    display: grid; grid-template-columns: 1fr 1fr; gap: 7px; margin-top: 10px;
}
.md-sb-stat-chip {
    border: 1px solid var(--md-outline-variant); border-radius: 13px;
    padding: 8px 10px; background: rgba(255,255,255,.04);
    display: flex; flex-direction: column; gap: 1px;
}
.md-sb-stat-val { font-size: 15px; font-weight: 900; line-height: 1.1; }
.md-sb-stat-lbl { color: var(--md-soft); font-size: 10px; font-weight: 600; letter-spacing: .04em; }

/* ── Section header ──────────────────────────────────────── */
.md-sidebar-section {
    font-size: 10px; font-weight: 800; color: var(--md-soft);
    text-transform: uppercase; letter-spacing: .10em;
    margin: 14px 4px 7px;
    display: flex; align-items: center; gap: 7px;
}
.md-sidebar-section::after {
    content: ''; flex: 1; height: 1px;
    background: linear-gradient(90deg, var(--md-outline-variant), transparent);
}

/* ── Nav items ───────────────────────────────────────────── */
.md-sidebar-link {
    display: flex; align-items: center; gap: 11px;
    padding: 10px 12px; border-radius: 16px;
    border: 1px solid transparent; background: transparent;
    margin-bottom: 4px;
    transition: background 140ms ease, border-color 140ms ease, transform 140ms ease, box-shadow 140ms ease;
    cursor: default;
    word-break: break-word; overflow-wrap: break-word;
}
.md-sidebar-link:hover {
    background: rgba(var(--md-primary-rgb), 0.11);
    border-color: rgba(var(--md-primary-rgb), 0.28);
    transform: translateX(3px);
}
.md-sidebar-link.active {
    background: linear-gradient(135deg, rgba(var(--md-primary-rgb),.22), rgba(var(--md-secondary-rgb),.14));
    border-color: rgba(var(--md-primary-rgb), .42);
    box-shadow: 0 2px 16px rgba(var(--md-primary-rgb),.18), inset 0 1px 0 rgba(255,255,255,.06);
}
.md-sidebar-icon {
    width: 34px; height: 34px; min-width: 34px; border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 16px;
    background: rgba(255,255,255,.05); border: 1px solid var(--md-outline-variant);
    transition: background 140ms ease;
}
.md-sidebar-link.active .md-sidebar-icon {
    background: linear-gradient(135deg, rgba(var(--md-primary-rgb),.34), rgba(var(--md-secondary-rgb),.24));
    border-color: rgba(var(--md-primary-rgb), .44);
}
.md-sb-link-body { flex: 1; min-width: 0; }
.md-sidebar-link-title {
    font-size: 13px; font-weight: 800;
    word-break: break-word; overflow-wrap: break-word;
}
.md-sidebar-link-sub {
    color: var(--md-soft); font-size: 11px; line-height: 1.35;
    word-break: break-word; overflow-wrap: break-word;
}
.md-sb-link-arrow {
    font-size: 12px; color: var(--md-soft); opacity: 0;
    transition: opacity 140ms ease, transform 140ms ease;
}
.md-sidebar-link:hover .md-sb-link-arrow,
.md-sidebar-link.active .md-sb-link-arrow { opacity: 1; transform: translateX(2px); }

/* active left accent bar */
.md-sidebar-link.active { position: relative; }
.md-sidebar-link.active::before {
    content: ''; position: absolute; left: -14px; top: 25%; bottom: 25%;
    width: 3px; border-radius: 0 3px 3px 0;
    background: linear-gradient(180deg, #6750a4, #006a6a);
}

/* ── Medical notice ──────────────────────────────────────── */
.md-sidebar-note {
    border: 1px solid rgba(245, 158, 11, 0.32);
    border-radius: 16px; padding: 12px 13px;
    background: rgba(245, 158, 11, 0.08);
    color: var(--md-soft); font-size: 11.5px; line-height: 1.5;
    margin-top: 12px; display: flex; gap: 9px; align-items: flex-start;
    word-break: break-word; overflow-wrap: break-word;
}
.md-sb-note-icon { font-size: 16px; line-height: 1; margin-top: 1px; flex-shrink: 0; }

/* ── Footer ──────────────────────────────────────────────── */
.md-sidebar-footer {
    text-align: center; color: var(--md-soft); font-size: 11.5px;
    margin-top: 14px; padding-top: 12px;
    border-top: 1px solid var(--md-outline-variant);
}
.md-sb-creator {
    background: linear-gradient(90deg, #f9a8d4, #c084fc, #818cf8);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; font-weight: 900; font-size: 13px;
    filter: drop-shadow(0 0 6px rgba(192,132,252,.45));
}
.md-sb-version-pill {
    display: inline-block; margin-top: 5px;
    padding: 2px 9px; border-radius: 999px;
    background: rgba(var(--md-primary-rgb),.12); border: 1px solid rgba(var(--md-primary-rgb),.25);
    font-size: 10px; font-weight: 700; color: #d7c7ff; letter-spacing: .04em;
}

@media (max-width: 768px) {
    [data-testid="stSidebar"] { background: #12101a !important; }
    .md-sidebar-link:hover,
    .md-sidebar-link:active { transform: none !important; }
    .md-sidebar-hero { padding: 13px; border-radius: 20px; }
    .md-sidebar-kicker { font-size: 10px; padding: 5px 9px; }
    .md-sidebar-title { font-size: 15px !important; }
    .md-sidebar-text  { font-size: 12px; }
    .md-sidebar-link  { padding: 9px 10px; gap: 8px; border-radius: 14px; }
    .md-sidebar-icon  { width: 30px; height: 30px; min-width: 30px; font-size: 14px; }
    .md-sidebar-link-title { font-size: 13px; }
    .md-sidebar-link-sub   { font-size: 11px; }
    .md-sidebar-note  { font-size: 12px; padding: 11px; }

    /* Ensure the glowing badge remains visible on mobile screens */
    .md-made-with {
        box-shadow:
            0 0 18px rgba(103, 80, 164, 0.42),
            0 0 34px rgba(103, 80, 164, 0.24),
            0 10px 24px rgba(15, 23, 42, 0.22) !important;
        border-color: rgba(255, 255, 255, 0.30) !important;
        background: linear-gradient(135deg, rgba(var(--md-primary-rgb), 0.24), rgba(var(--md-secondary-rgb), 0.16)) !important;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
    }

    .md-made-with .heart {
        filter: drop-shadow(0 0 10px rgba(239, 68, 68, 0.85));
    }
}

/* ── HERO ─────────────────────────────────────────────────── */
.md-hero {
    overflow: hidden;
    border: 1px solid var(--md-outline);
    border-radius: 32px;
    padding: 30px;
    margin: 10px 0 18px 0;
    position: relative;
    background:
        linear-gradient(135deg, rgba(var(--md-primary-rgb), 0.16), rgba(var(--md-secondary-rgb), 0.09) 58%, rgba(186, 26, 26, 0.06)),
        var(--md-surface);
    box-shadow: var(--md-shadow-3);
}

.md-hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 280px; height: 280px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(var(--md-primary-rgb),0.12), transparent 70%);
    pointer-events: none;
}

.md-hero-grid {
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(280px, 380px);
    gap: 28px;
    align-items: center;
}

.md-kicker {
    display: inline-flex;
    width: fit-content;
    padding: 7px 12px;
    border: 1px solid rgba(var(--md-primary-rgb), 0.36);
    border-radius: 999px;
    background: rgba(var(--md-primary-rgb), 0.13);
    color: #d7c7ff;
    font-size: 12px;
    font-weight: 900;
    margin-bottom: 10px;
}

.md-title {
    margin: 0;
    font-size: clamp(36px, 5vw, 68px);
    line-height: 0.98;
    font-weight: 980;
    letter-spacing: -0.02em;
    font-family: var(--font-display) !important;
}

.md-subtitle {
    max-width: 760px;
    margin-top: 14px;
    color: var(--md-soft);
    font-size: 17px;
    line-height: 1.65;
}

.md-chip-row {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin-top: 18px;
}

.md-chip {
    padding: 9px 13px;
    border-radius: 999px;
    border: 1px solid var(--md-outline);
    background: var(--md-surface-container);
    font-weight: 820;
    font-size: 13px;
    box-shadow: var(--md-shadow-1);
    transition: transform 120ms ease, border-color 120ms ease;
}

.md-chip:hover {
    transform: translateY(-2px);
    border-color: rgba(var(--md-primary-rgb), 0.4);
}

.md-hero-image {
    width: 100%;
    aspect-ratio: 1.25;
    border-radius: 28px;
    object-fit: contain;
    background:
        linear-gradient(135deg, rgba(255,255,255,0.10), rgba(255,255,255,0.035)),
        rgba(var(--md-primary-rgb), 0.08);
    padding: 12px;
    border: 1px solid rgba(255,255,255,0.24);
    box-shadow: var(--md-shadow-2);
    box-sizing: border-box;
}

.md-hero-image-fallback {
    width: 100%;
    aspect-ratio: 1.25;
    border-radius: 28px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 54px;
    border: 1px solid rgba(255,255,255,0.24);
    background: linear-gradient(135deg, rgba(var(--md-primary-rgb), 0.18), rgba(var(--md-secondary-rgb), 0.14));
}

.md-made-with {
    text-align: center;
    margin: 0.7rem auto 1.35rem auto;
    font-size: 1.05rem;
    font-weight: 780;
    color: #dbe4ff;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.35rem;
    padding: 0.55rem 1rem;
    border-radius: 999px;
    background: linear-gradient(135deg, rgba(var(--md-primary-rgb), 0.18), rgba(var(--md-secondary-rgb), 0.12));
    border: 1px solid rgba(255, 255, 255, 0.24);
    box-shadow: 
        0 0 24px rgba(103, 80, 164, 0.35),
        0 0 48px rgba(103, 80, 164, 0.20),
        0 10px 26px rgba(15, 23, 42, 0.20);
    animation: md-made-with-breathe 3.8s ease-in-out infinite;
    width: fit-content;
}

.md-made-with .heart { 
    color: #ef4444;
    filter: drop-shadow(0 0 8px rgba(239, 68, 68, 0.6));
    animation: md-heart-pulse 1.8s ease-in-out infinite;
}

@keyframes md-made-with-breathe {
    0% { 
        transform: translateY(0);
        box-shadow: 
            0 0 24px rgba(103, 80, 164, 0.35),
            0 0 48px rgba(103, 80, 164, 0.20),
            0 10px 26px rgba(15, 23, 42, 0.20);
        border-color: rgba(255, 255, 255, 0.24);
    }
    20% { 
        transform: translateY(0);
        box-shadow: 
            0 0 12px rgba(103, 80, 164, 0.18),
            0 0 24px rgba(103, 80, 164, 0.10),
            0 10px 26px rgba(15, 23, 42, 0.10);
        border-color: rgba(255, 255, 255, 0.12);
    }
    50% { 
        transform: translateY(-2px);
        box-shadow: 
            0 0 32px rgba(103, 80, 164, 0.50),
            0 0 64px rgba(103, 80, 164, 0.30),
            0 14px 32px rgba(15, 23, 42, 0.25);
        border-color: rgba(var(--md-primary-rgb), 0.40);
    }
    80% { 
        transform: translateY(-1px);
        box-shadow: 
            0 0 28px rgba(103, 80, 164, 0.42),
            0 0 56px rgba(103, 80, 164, 0.25),
            0 12px 28px rgba(15, 23, 42, 0.22);
        border-color: rgba(var(--md-primary-rgb), 0.32);
    }
    100% { 
        transform: translateY(0);
        box-shadow: 
            0 0 24px rgba(103, 80, 164, 0.35),
            0 0 48px rgba(103, 80, 164, 0.20),
            0 10px 26px rgba(15, 23, 42, 0.20);
        border-color: rgba(255, 255, 255, 0.24);
    }
}

@keyframes md-heart-pulse {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.15); }
}

/* ── STATS ────────────────────────────────────────────────── */
.md-stat-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 12px;
    margin: 0 0 1.5rem 0;
}

.md-stat-card {
    border: 1px solid var(--md-outline-variant);
    border-radius: 24px;
    padding: 15px 18px;
    background: var(--md-surface);
    box-shadow: var(--md-shadow-1);
    transition: transform 140ms ease, border-color 140ms ease;
    position: relative;
    overflow: hidden;
}

.md-stat-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, rgba(var(--md-primary-rgb),0.7), rgba(var(--md-secondary-rgb),0.5));
    border-radius: 3px 3px 0 0;
}

.md-stat-card:hover {
    transform: translateY(-2px);
    border-color: rgba(var(--md-primary-rgb), 0.3);
}

.md-stat-label {
    color: var(--md-soft);
    font-size: 12px;
    margin-bottom: 4px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.md-stat-value {
    font-size: 17px;
    font-weight: 950;
    font-family: var(--font-display) !important;
}

/* ── SECTION HEADINGS ─────────────────────────────────────── */
.md-section-title {
    text-align: center;
    margin: 2.1rem 0 0.85rem 0;
    font-size: clamp(24px, 3vw, 34px);
    font-weight: 950;
    letter-spacing: -0.01em;
    font-family: var(--font-display) !important;
}

.md-section-subtitle {
    text-align: center;
    color: var(--md-soft);
    max-width: 720px;
    margin: -0.35rem auto 1.35rem auto;
    line-height: 1.55;
}

/* ── FEATURE CARDS ────────────────────────────────────────── */
.md-card-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 14px;
    margin: 1rem 0;
}

.md-card {
    border: 1px solid var(--md-outline);
    border-radius: 26px;
    padding: 18px;
    background: var(--md-surface);
    box-shadow: var(--md-shadow-1);
    min-height: 150px;
    transition: transform 160ms ease, box-shadow 160ms ease, border-color 160ms ease, background 160ms ease;
    position: relative;
    overflow: hidden;
}

.md-card::after {
    content: '';
    position: absolute;
    bottom: 0; right: 0;
    width: 80px; height: 80px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(var(--md-primary-rgb),0.06), transparent 70%);
    pointer-events: none;
}

.md-card:hover {
    transform: translateY(-4px);
    border-color: rgba(var(--md-primary-rgb), 0.42);
    background: var(--md-surface-container);
    box-shadow: var(--md-shadow-2);
}

.md-card-icon {
    width: 48px;
    height: 48px;
    border-radius: 18px;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 13px;
    font-size: 22px;
    background: linear-gradient(135deg, rgba(var(--md-primary-rgb), 0.22), rgba(var(--md-secondary-rgb), 0.14));
    border: 1px solid rgba(255,255,255,0.14);
    box-shadow: 0 2px 8px rgba(var(--md-primary-rgb),0.15);
}

.md-card h3 {
    margin: 0 0 8px 0;
    font-size: 18px;
    font-family: var(--font-display) !important;
    font-weight: 800;
}

.md-card p {
    margin: 0;
    color: var(--md-soft);
    font-size: 14px;
    line-height: 1.55;
}

/* ── HEALTH CHECKER (NEW) ─────────────────────────────────── */
/* ── HEALTH SELF-ASSESSMENT ───────────────────────────────── */
.md-health-shell {
    border: 1px solid var(--md-outline);
    border-radius: 32px;
    padding: 28px;
    background:
        linear-gradient(135deg, rgba(var(--md-primary-rgb), 0.12), rgba(var(--md-secondary-rgb), 0.07) 60%, rgba(var(--md-primary-rgb),0.04)),
        var(--md-surface);
    box-shadow: var(--md-shadow-3);
    margin-top: 1rem;
    position: relative;
    overflow: hidden;
}
.md-health-shell::before {
    content:''; position:absolute; top:-60px; right:-60px;
    width:220px; height:220px; border-radius:50%;
    background:radial-gradient(circle, rgba(var(--md-primary-rgb),.14) 0%, transparent 70%);
    pointer-events:none;
}

.md-health-score-ring {
    width: 130px; height: 130px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    flex-direction: column;
    font-family: var(--font-display) !important;
    font-size: 30px; font-weight: 900;
    border: 6px solid;
    margin: 0 auto 20px auto;
    box-shadow: 0 0 36px rgba(0,0,0,0.18), inset 0 0 24px rgba(255,255,255,0.04);
    backdrop-filter: blur(4px);
    transition: transform 200ms ease;
}
.md-health-score-ring:hover { transform: scale(1.04); }

.md-pillar-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
    margin-top: 16px;
}

.md-pillar-card {
    border: 1px solid var(--md-outline-variant);
    border-radius: 20px;
    padding: 14px;
    background: var(--md-surface-container);
    display: flex; flex-direction: column;
    gap: 6px; font-size: 13px;
    transition: transform 140ms ease, border-color 140ms ease;
}
.md-pillar-card:hover {
    transform: translateY(-2px);
    border-color: rgba(var(--md-primary-rgb),.32);
}

.md-pillar-dot {
    width: 10px; height: 10px;
    border-radius: 50%; min-width: 10px;
}

.md-pillar-bar-wrap {
    height: 6px; border-radius: 999px;
    background: rgba(148,163,184,0.15);
    margin-top: 4px; overflow: hidden;
}

.md-pillar-bar {
    height: 100%; border-radius: 999px;
    transition: width 600ms cubic-bezier(.4,0,.2,1);
}

/* ── RISK ASSESSMENT BANNER (NEW) ─────────────────────────── */
.md-risk-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 14px;
    margin: 1rem 0;
}

.md-risk-card {
    border-radius: 24px;
    padding: 20px;
    border: 1px solid var(--md-outline-variant);
    background: var(--md-surface);
    box-shadow: var(--md-shadow-1);
    transition: transform 150ms ease, box-shadow 150ms ease;
    position: relative;
    overflow: hidden;
}

.md-risk-card:hover {
    transform: translateY(-3px);
    box-shadow: var(--md-shadow-2);
}

.md-risk-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 12px;
}

.md-risk-icon-wrap {
    width: 44px; height: 44px;
    border-radius: 16px;
    display: flex; align-items: center; justify-content: center;
    font-size: 20px;
    flex-shrink: 0;
}

.md-risk-name {
    font-size: 16px;
    font-weight: 800;
    font-family: var(--font-display) !important;
}

.md-risk-sub {
    font-size: 12px;
    color: var(--md-soft);
}

.md-risk-stat {
    font-size: 28px;
    font-weight: 900;
    font-family: var(--font-display) !important;
    line-height: 1;
    margin-bottom: 2px;
}

.md-risk-desc {
    font-size: 12px;
    color: var(--md-soft);
    line-height: 1.45;
    margin-top: 6px;
}

/* ── SYMPTOM QUICK-CHECK ───────────────────────────────────── */
.md-symptom-shell {
    border: 1px solid var(--md-outline);
    border-radius: 32px;
    padding: 28px;
    background:
        linear-gradient(135deg, rgba(var(--md-secondary-rgb), 0.11), rgba(var(--md-primary-rgb), 0.07) 60%, rgba(var(--md-secondary-rgb),0.04)),
        var(--md-surface);
    box-shadow: var(--md-shadow-3);
    margin-top: 1rem;
    position: relative; overflow: hidden;
}
.md-symptom-shell::before {
    content:''; position:absolute; bottom:-70px; left:-70px;
    width:220px; height:220px; border-radius:50%;
    background:radial-gradient(circle, rgba(var(--md-secondary-rgb),.13) 0%, transparent 70%);
    pointer-events:none;
}

.md-symptom-tag {
    display: inline-flex;
    padding: 7px 14px;
    border-radius: 999px;
    font-size: 13px; font-weight: 700;
    cursor: pointer; border: 1px solid;
    transition: all 140ms ease;
    margin: 4px;
}

.md-triage-banner {
    display: flex; align-items: flex-start; gap: 14px;
    padding: 18px; border-radius: 20px;
    margin-bottom: 16px;
    animation: md-rise-in 360ms ease-out;
}
.md-triage-icon {
    font-size: 32px; flex-shrink: 0; line-height: 1;
}
.md-triage-level {
    font-size: 19px; font-weight: 900;
    font-family: var(--font-display) !important;
    line-height: 1.1; margin-bottom: 4px;
}
.md-triage-msg { font-size: 13px; color: var(--md-soft); line-height: 1.5; }

.md-symptom-result {
    border: 1px solid var(--md-outline);
    border-radius: 24px;
    padding: 20px;
    background: var(--md-surface-container-high);
    margin-top: 14px;
    animation: md-rise-in 360ms ease-out;
}

.md-conditions-row {
    display: flex; align-items: center; gap: 8px;
    padding: 8px 12px; border-radius: 12px;
    background: var(--md-surface); border: 1px solid var(--md-outline-variant);
    margin-bottom: 6px;
    font-size: 13px;
}

@keyframes md-rise-in {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── HEALTH TIMELINE (NEW) ────────────────────────────────── */
.md-timeline {
    position: relative;
    padding-left: 28px;
    margin: 1rem 0;
}

.md-timeline::before {
    content: '';
    position: absolute;
    left: 9px; top: 0; bottom: 0;
    width: 2px;
    background: linear-gradient(180deg, rgba(var(--md-primary-rgb),0.6), rgba(var(--md-secondary-rgb),0.4), transparent);
    border-radius: 2px;
}

.md-timeline-item {
    position: relative;
    margin-bottom: 20px;
}

.md-timeline-dot {
    position: absolute;
    left: -23px;
    top: 10px;
    width: 14px; height: 14px;
    border-radius: 50%;
    background: rgba(var(--md-primary-rgb),0.8);
    border: 2px solid rgba(var(--md-primary-rgb),0.3);
    box-shadow: 0 0 8px rgba(var(--md-primary-rgb),0.4);
}

.md-timeline-card {
    border: 1px solid var(--md-outline-variant);
    border-radius: 18px;
    padding: 14px 16px;
    background: var(--md-surface);
    box-shadow: var(--md-shadow-1);
    transition: transform 140ms ease;
}

.md-timeline-card:hover {
    transform: translateX(4px);
    border-color: rgba(var(--md-primary-rgb),0.3);
}

.md-timeline-title {
    font-size: 15px;
    font-weight: 800;
    margin-bottom: 3px;
    font-family: var(--font-display) !important;
}

.md-timeline-meta {
    font-size: 12px;
    color: var(--md-soft);
    margin-bottom: 6px;
}

.md-timeline-tag {
    display: inline-flex;
    padding: 3px 9px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 700;
    background: rgba(var(--md-primary-rgb),0.13);
    border: 1px solid rgba(var(--md-primary-rgb),0.28);
    color: #d7c7ff;
}

/* ── HEALTH VITALS TRACKER (NEW) ──────────────────────────── */
.md-vitals-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin: 1rem 0;
}

.md-vital-card {
    border: 1px solid var(--md-outline-variant);
    border-radius: 22px;
    padding: 16px;
    background: var(--md-surface);
    box-shadow: var(--md-shadow-1);
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: transform 140ms ease;
}

.md-vital-card:hover { transform: translateY(-3px); }

.md-vital-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 3px 3px 0 0;
}

.md-vital-icon {
    font-size: 26px;
    margin-bottom: 8px;
}

.md-vital-value {
    font-size: 22px;
    font-weight: 900;
    font-family: var(--font-display) !important;
    line-height: 1;
}

.md-vital-unit {
    font-size: 11px;
    color: var(--md-soft);
    font-weight: 600;
}

.md-vital-label {
    font-size: 12px;
    color: var(--md-soft);
    margin-top: 4px;
    font-weight: 600;
}

.md-vital-status {
    display: inline-flex;
    padding: 3px 8px;
    border-radius: 999px;
    font-size: 10px;
    font-weight: 700;
    margin-top: 6px;
}

/* ── WHY BOX ──────────────────────────────────────────────── */
.md-why-box {
    border: 1px solid var(--md-outline);
    border-radius: 28px;
    padding: 22px;
    background:
        linear-gradient(135deg, rgba(var(--md-primary-rgb), 0.11), rgba(var(--md-secondary-rgb), 0.08)),
        var(--md-surface);
    box-shadow: var(--md-shadow-2);
}

.md-why-list {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 12px;
    margin-top: 14px;
}

.md-why-item {
    border: 1px solid var(--md-outline-variant);
    border-radius: 22px;
    padding: 14px;
    background: rgba(255,255,255,0.045);
    transition: transform 140ms ease;
}

.md-why-item:hover { transform: translateY(-2px); }

.md-why-item strong {
    display: block;
    margin-bottom: 5px;
    font-family: var(--font-display) !important;
    font-size: 15px;
}

.md-why-item span {
    color: var(--md-soft);
    font-size: 13px;
    line-height: 1.45;
}

/* ── AI TIPS MARQUEE (NEW) ────────────────────────────────── */
.md-marquee-wrap {
    overflow: hidden;
    border: 1px solid var(--md-outline);
    border-radius: 20px;
    padding: 12px 0;
    background: var(--md-surface);
    margin: 1rem 0;
    position: relative;
    max-width: 100%;
    contain: layout style;
}

.md-marquee-wrap::before,
.md-marquee-wrap::after {
    content: '';
    position: absolute;
    top: 0; bottom: 0;
    width: 60px;
    z-index: 2;
    pointer-events: none;
}

.md-marquee-wrap::before {
    left: 0;
    background: linear-gradient(90deg, var(--background-color, #0f172a), transparent);
}

.md-marquee-wrap::after {
    right: 0;
    background: linear-gradient(-90deg, var(--background-color, #0f172a), transparent);
}

.md-marquee-track {
    display: flex;
    gap: 24px;
    animation: md-marquee-scroll 50s linear infinite;
    width: max-content;
}

@keyframes md-marquee-scroll {
    from { transform: translateX(0); }
    to   { transform: translateX(-50%); }
}

.md-marquee-item {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 14px;
    border-radius: 999px;
    border: 1px solid var(--md-outline-variant);
    background: var(--md-surface-container);
    font-size: 13px;
    font-weight: 600;
    white-space: nowrap;
}

/* ── CONTACT ──────────────────────────────────────────────── */
/* ── CONTACT ──────────────────────────────────────────────── */
.md-contact-wrap {
    border: 1px solid var(--md-outline);
    border-radius: 32px; overflow: hidden;
    background: linear-gradient(135deg, rgba(var(--md-primary-rgb),.10), rgba(var(--md-secondary-rgb),.06) 60%, rgba(var(--md-primary-rgb),.04));
    box-shadow: var(--md-shadow-2);
    display: flex; flex-direction: column; align-items: center;
    padding: 40px 32px 36px; gap: 24px; text-align: center;
    position: relative;
}
.md-contact-wrap::before {
    content:''; position:absolute; top:-80px; right:-80px;
    width:260px; height:260px; border-radius:50%;
    background:radial-gradient(circle, rgba(var(--md-primary-rgb),.18) 0%, transparent 70%);
    pointer-events:none;
}
.md-contact-avatar {
    width:72px; height:72px; border-radius:50%;
    background:linear-gradient(135deg,rgba(var(--md-primary-rgb),.28),rgba(var(--md-secondary-rgb),.20));
    border:2px solid rgba(var(--md-primary-rgb),.36);
    display:flex; align-items:center; justify-content:center;
    font-size:30px; box-shadow:0 8px 24px rgba(var(--md-primary-rgb),.22); flex-shrink:0;
}
.md-contact-title {
    font-family:var(--font-display) !important;
    font-size:26px; font-weight:900; line-height:1.15; margin:0;
}
.md-contact-sub { color:var(--md-soft); font-size:14px; line-height:1.55; max-width:480px; margin:0 auto; }
.md-contact-email {
    display:inline-flex; align-items:center; gap:8px;
    padding:12px 22px; border-radius:999px;
    background:rgba(var(--md-primary-rgb),.12); border:1px solid rgba(var(--md-primary-rgb),.30);
    color:#d7c7ff !important; font-weight:800; font-size:14px; text-decoration:none !important;
    transition:all 140ms ease;
}
.md-contact-email:hover {
    background:rgba(var(--md-primary-rgb),.22); border-color:rgba(var(--md-primary-rgb),.55);
    transform:translateY(-2px); box-shadow:0 8px 20px rgba(var(--md-primary-rgb),.22); text-decoration:none !important;
}
.md-contact-socials { display:flex; gap:10px; flex-wrap:wrap; justify-content:center; }
.md-contact-social-btn {
    display:inline-flex; align-items:center; gap:7px;
    padding:10px 18px; border-radius:999px;
    border:1px solid var(--md-outline-variant); background:var(--md-surface-container);
    color:var(--md-soft) !important; font-weight:800; font-size:13px; text-decoration:none !important;
    transition:all 140ms ease;
}
.md-contact-social-btn:hover {
    background:rgba(var(--md-primary-rgb),.10); border-color:rgba(var(--md-primary-rgb),.38);
    color:#d7c7ff !important; transform:translateY(-2px); text-decoration:none !important;
}
@media (max-width:620px) { .md-contact-wrap { padding:28px 18px; } }

/* ── MATERIAL SYMBOLS FONT (for sidebar toggle icon) ─────── */

/* ── SIDEBAR COLLAPSE ARROW FIX ──────────────────────────── */
/* Target Streamlit's actual sidebar toggle: [data-testid="stSidebarCollapsedControl"] */
[data-testid="stSidebarCollapsedControl"] {
    overflow: visible !important;
}
[data-testid="stSidebarCollapsedControl"] button {
    overflow: hidden !important;
    font-size: 0 !important;
    line-height: 0 !important;
    color: transparent !important;
    width: 36px !important;
    height: 36px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    position: relative !important;
}
/* Hide the raw ligature text span */
[data-testid="stSidebarCollapsedControl"] button span,
[data-testid="stSidebarCollapsedControl"] button .material-symbols-rounded,
[data-testid="stSidebarCollapsedControl"] button .material-symbols-outlined {
    font-size: 0 !important;
    line-height: 0 !important;
    color: transparent !important;
    overflow: hidden !important;
    display: block !important;
    width: 0 !important;
    height: 0 !important;
    visibility: hidden !important;
}
/* Inject a clean chevron via ::after pseudo-element */
[data-testid="stSidebarCollapsedControl"] button::after {
    content: "";
    display: block !important;
    width: 20px !important;
    height: 20px !important;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='20' height='20' viewBox='0 0 24 24' fill='none' stroke='rgba(148,163,184,0.9)' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='13 17 18 12 13 7'/%3E%3Cpolyline points='6 17 11 12 6 7'/%3E%3C/svg%3E") !important;
    background-repeat: no-repeat !important;
    background-size: contain !important;
    background-position: center !important;
    flex-shrink: 0 !important;
    visibility: visible !important;
}

button[aria-label="Close sidebar"],
button[aria-label="Open sidebar"],
button[aria-label="Collapse sidebar"],
button[aria-label="Expand sidebar"],
button[aria-label="collapse sidebar"],
button[aria-label="expand sidebar"] {
    font-size: 0 !important;
    color: transparent !important;
    letter-spacing: -9999px !important;
    overflow: hidden !important;
}
button[aria-label="Close sidebar"] span,
button[aria-label="Open sidebar"] span,
button[aria-label="Collapse sidebar"] span,
button[aria-label="Expand sidebar"] span,
button[aria-label="collapse sidebar"] span,
button[aria-label="expand sidebar"] span {
    visibility: hidden !important;
    font-size: 0 !important;
    width: 0 !important;
    height: 0 !important;
    overflow: hidden !important;
}

/* Extra guard: Streamlit sometimes renders the sidebar control as raw icon text. */
[data-testid="stSidebarCollapsedControl"] {
    width: 44px !important;
    height: 44px !important;
}
[data-testid="stSidebarCollapsedControl"] button,
button[aria-label="Close sidebar"],
button[aria-label="Open sidebar"],
button[aria-label="Collapse sidebar"],
button[aria-label="Expand sidebar"],
button[aria-label="collapse sidebar"],
button[aria-label="expand sidebar"] {
    width: 40px !important;
    height: 40px !important;
    min-width: 40px !important;
    border-radius: 14px !important;
    border: 1px solid var(--md-outline-variant) !important;
    background: var(--md-surface-container) !important;
    box-shadow: var(--md-shadow-1) !important;
    color: transparent !important;
    letter-spacing: 0 !important;
    text-indent: 0 !important;
    position: relative !important;
}
[data-testid="stSidebarCollapsedControl"] button > *,
button[aria-label="Close sidebar"] > *,
button[aria-label="Open sidebar"] > *,
button[aria-label="Collapse sidebar"] > *,
button[aria-label="Expand sidebar"] > *,
button[aria-label="collapse sidebar"] > *,
button[aria-label="expand sidebar"] > * {
    display: none !important;
    visibility: hidden !important;
    font-size: 0 !important;
    line-height: 0 !important;
    width: 0 !important;
    height: 0 !important;
    overflow: hidden !important;
}
[data-testid="stSidebarCollapsedControl"] button::after,
button[aria-label="Open sidebar"]::after,
button[aria-label="Expand sidebar"]::after,
button[aria-label="open sidebar"]::after,
button[aria-label="expand sidebar"]::after {
    content: "" !important;
    display: block !important;
    width: 20px !important;
    height: 20px !important;
    background-color: #d7c7ff !important;
    -webkit-mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='20' height='20' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2.4' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='13 17 18 12 13 7'/%3E%3Cpolyline points='6 17 11 12 6 7'/%3E%3C/svg%3E") center / contain no-repeat !important;
    mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='20' height='20' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2.4' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='13 17 18 12 13 7'/%3E%3Cpolyline points='6 17 11 12 6 7'/%3E%3C/svg%3E") center / contain no-repeat !important;
    visibility: visible !important;
}
button[aria-label="Close sidebar"]::after,
button[aria-label="Collapse sidebar"]::after,
button[aria-label="close sidebar"]::after,
button[aria-label="collapse sidebar"]::after {
    content: "" !important;
    display: block !important;
    width: 20px !important;
    height: 20px !important;
    background-color: #d7c7ff !important;
    -webkit-mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='20' height='20' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2.4' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='11 17 6 12 11 7'/%3E%3Cpolyline points='18 17 13 12 18 7'/%3E%3C/svg%3E") center / contain no-repeat !important;
    mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='20' height='20' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2.4' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='11 17 6 12 11 7'/%3E%3Cpolyline points='18 17 13 12 18 7'/%3E%3C/svg%3E") center / contain no-repeat !important;
    visibility: visible !important;
}
[data-testid="stSidebarCollapsedControl"] button:hover::after,
button[aria-label="Close sidebar"]:hover::after,
button[aria-label="Open sidebar"]:hover::after,
button[aria-label="Collapse sidebar"]:hover::after,
button[aria-label="Expand sidebar"]:hover::after,
button[aria-label="collapse sidebar"]:hover::after,
button[aria-label="expand sidebar"]:hover::after { background-color: white !important; }
[data-theme="light"] [data-testid="stSidebarCollapsedControl"] button::after,
[data-theme="light"] button[aria-label="Close sidebar"]::after,
[data-theme="light"] button[aria-label="Open sidebar"]::after,
[data-theme="light"] button[aria-label="Collapse sidebar"]::after,
[data-theme="light"] button[aria-label="Expand sidebar"]::after,
[data-theme="light"] button[aria-label="collapse sidebar"]::after,
[data-theme="light"] button[aria-label="expand sidebar"]::after { background-color: #5b3fc4 !important; }

/* ── PREVENT HORIZONTAL SCROLL ON MOBILE SIDEBAR ─────────── */
@media (max-width: 768px) {
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] > div,
    [data-testid="stSidebarContent"] {
        overflow-x: hidden !important;
        max-width: 100vw !important;
    }
}

/* ── STREAMLIT OVERRIDES ──────────────────────────────────── */
.stButton > button {
    border-radius: 999px !important;
    min-height: 46px;
    font-weight: 850 !important;
    font-family: var(--font-body) !important;
    border: 1px solid rgba(var(--md-primary-rgb), 0.38) !important;
    background: linear-gradient(135deg, #6750a4, #7c3aed) !important;
    color: white !important;
    box-shadow: var(--md-shadow-1);
    transition: transform 120ms ease, box-shadow 120ms ease !important;
}

.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: var(--md-shadow-2);
}

[data-testid="stRadio"] label,
[data-baseweb="select"] {
    border-radius: 18px !important;
}

.stSelectbox label,
.stRadio label {
    font-weight: 760;
}

/* ── RESPONSIVE ───────────────────────────────────────────── */
@media (max-width: 980px) {
    .md-hero-grid,
    .md-card-grid,
    .md-why-list,
    .md-stat-grid,
    .md-vitals-grid,
    .md-risk-grid,
    .md-pillar-grid { grid-template-columns: 1fr; }

    .md-hero { padding: 22px; border-radius: 26px; }
    .block-container { padding-left: 1rem; padding-right: 1rem; }
    .md-hero-image { max-height: 360px; }
}

@media (max-width: 620px) {
    .md-title { font-size: 34px; }
    .md-subtitle { font-size: 15px; }
    .md-health-shell,
    .md-symptom-shell { padding: 16px; border-radius: 24px; }
    .md-card,
    .md-why-box,
    .md-contact-card { border-radius: 22px; }
}

@media (max-width: 768px) {
    .md-made-with,
    .md-footer-heart,
    .hsa-result {
        animation: md-made-with-breathe 3.8s ease-in-out infinite !important;
    }
    .md-hero::before,
    .md-card::after,
    .md-health-shell::before,
    .hsa-shell::before,
    .hsa-shell::after {
        display: none !important;
    }
    .md-made-with {
        box-shadow:
            0 0 18px rgba(103, 80, 164, 0.34),
            0 0 36px rgba(103, 80, 164, 0.22),
            0 10px 22px rgba(15, 23, 42, 0.18) !important;
        background: linear-gradient(135deg, rgba(var(--md-primary-rgb), 0.20), rgba(var(--md-secondary-rgb), 0.14)) !important;
        border-color: rgba(255, 255, 255, 0.22) !important;
    }
    .md-made-with .heart {
        filter: drop-shadow(0 0 10px rgba(239, 68, 68, 0.78));
    }
    .md-card,
    .md-risk-card,
    .md-timeline-card,
    .hsa-pillar,
    .stButton > button {
        transition-duration: 90ms !important;
    }
}

@media (prefers-reduced-motion: reduce) {
    .md-made-with,
    .md-footer-heart,
    .md-marquee-track,
    .hsa-result,
    .mm-fab-pulse,
    .mm-fab-pulse2,
    .mm-fab-shimmer::before,
    .mm-fab-dot,
    .mm-widget-draghint {
        animation: none !important;
    }
}

/* ── FOOTER ───────────────────────────────────────────────── */
.md-footer { margin-top:52px; border-top:1px solid var(--md-outline); padding:32px 0 28px 0; }
.md-footer-top { display:flex; justify-content:center; margin-bottom:20px; }
.md-footer-brand { display:flex; align-items:center; gap:14px; }
.md-footer-logo {
    width:52px; height:52px; border-radius:16px;
    background:var(--md-surface-container); border:1px solid var(--md-outline-variant);
    padding:6px; box-shadow:var(--md-shadow-1); box-sizing:border-box;
}
.md-footer-logo-img { background-size:contain; background-repeat:no-repeat; background-position:center; display:inline-block; flex-shrink:0; }
.md-footer-logo-icon {
    width:52px; height:52px; min-width:52px; border-radius:18px;
    background: linear-gradient(135deg, #6750a4, #006a6a);
    display:flex; align-items:center; justify-content:center;
    font-size:26px; box-shadow: 0 4px 14px rgba(103,80,164,.40);
    flex-shrink:0;
}
.md-footer-brand-name { font-family:var(--font-display) !important; font-size:15px; font-weight:900; line-height:1.2; max-width:200px; }
.md-footer-brand-sub { font-size:12px; color:var(--md-soft); margin-top:3px; }
.md-footer-links { display:flex; flex-wrap:wrap; gap:8px; justify-content:center; margin-bottom:14px; }
.md-footer-link {
    padding:8px 16px; border-radius:999px; border:1px solid var(--md-outline-variant);
    background:var(--md-surface); color:var(--md-soft) !important;
    font-size:13px; font-weight:700; text-decoration:none !important;
    transition:all 130ms ease; display:inline-flex; align-items:center; gap:5px;
}
.md-footer-link:hover {
    background:rgba(var(--md-primary-rgb),.12); border-color:rgba(var(--md-primary-rgb),.38);
    color:#d7c7ff !important; transform:translateY(-2px); text-decoration:none !important;
}
.md-footer-meta {
    text-align:center; color:var(--md-soft); font-size:12px; line-height:1.75; margin-bottom:16px;
    white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
}
.md-footer-version {
    display:inline-block; padding:3px 10px; border-radius:999px;
    background:rgba(var(--md-primary-rgb),.12); border:1px solid rgba(var(--md-primary-rgb),.26);
    font-size:11px; font-weight:800; color:#d7c7ff; vertical-align:middle;
}
.md-footer-heart { color:#ef4444; display:inline-block; animation:md-heartbeat 2.5s ease-in-out infinite; }
@keyframes md-heartbeat { 0%,100%{transform:scale(1)} 50%{transform:scale(1.22)} }
.md-footer-disclaimer {
    margin-top:20px; padding:13px 18px; border-radius:16px;
    background:rgba(217,119,6,.07); border:1px solid rgba(217,119,6,.22);
    color:var(--md-soft); font-size:12px; line-height:1.6; text-align:center;
}
@media (max-width:620px) { .md-footer-brand { justify-content:center; } .md-footer-meta { white-space:normal; } }

/* ── LIGHT MODE OVERRIDES ─────────────────────────────────── */
[data-theme="light"] {
    --md-surface: rgba(103,80,164,0.06);
    --md-surface-container: rgba(103,80,164,0.09);
    --md-surface-container-high: rgba(103,80,164,0.13);
    --md-outline: rgba(103,80,164,0.22);
    --md-outline-variant: rgba(103,80,164,0.14);
    --md-soft: rgba(50,40,80,0.72);
    --md-shadow-1: 0 4px 14px rgba(103,80,164,0.10);
    --md-shadow-2: 0 12px 34px rgba(103,80,164,0.14);
    --md-shadow-3: 0 22px 58px rgba(103,80,164,0.18);
}

/* App background */
[data-theme="light"] .stApp,
[data-theme="light"] [data-testid="stAppViewContainer"] {
    background: linear-gradient(160deg, #f8f6ff 0%, #eef8f7 55%, #f5f0ff 100%) !important;
    color: #1d1b27 !important;
}
[data-theme="light"] [data-testid="stMain"] {
    background: transparent !important;
}

/* Sidebar background in light mode (non-mobile) */
[data-theme="light"] [data-testid="stSidebar"] {
    background:
        radial-gradient(ellipse at 22% 0%,  rgba(103,80,164,.14), transparent 42%),
        radial-gradient(ellipse at 90% 58%, rgba(0,106,106,.09),  transparent 40%),
        linear-gradient(175deg, rgba(103,80,164,.08) 0%, rgba(0,106,106,.04) 55%, transparent 100%) !important;
    border-right: 1px solid rgba(103,80,164,0.18) !important;
}
@media (max-width: 768px) {
    [data-theme="light"] [data-testid="stSidebar"],
    [data-theme="light"] [data-testid="stSidebar"] > div,
    [data-theme="light"] [data-testid="stSidebarContent"] {
        background:
            linear-gradient(180deg, #fffbff 0%, #f6f1ff 54%, #edf8f7 100%) !important;
        background-color: #fffbff !important;
        opacity: 1 !important;
        backdrop-filter: none !important;
        -webkit-backdrop-filter: none !important;
    }
    [data-theme="light"] [data-testid="stSidebar"] {
        box-shadow: 0 18px 44px rgba(91,63,196,0.18) !important;
    }
    [data-theme="light"] .md-sidebar-hero,
    [data-theme="light"] .md-sidebar-link,
    [data-theme="light"] .md-sidebar-note {
        background: #ffffff !important;
    }
}

/* Sidebar hero, kicker, title, text, section */
[data-theme="light"] .md-sidebar-hero {
    background:
        linear-gradient(135deg, rgba(103,80,164,0.12), rgba(0,106,106,0.08)),
        rgba(255,255,255,0.88) !important;
    border-color: rgba(103,80,164,0.22) !important;
}
[data-theme="light"] .md-sidebar-kicker {
    background: rgba(103,80,164,0.12) !important;
    border-color: rgba(103,80,164,0.32) !important;
    color: #4a3a7d !important;
}
[data-theme="light"] .md-sidebar-title   {
    background: linear-gradient(135deg, #6750a4 0%, #4a3580 50%, #006a6a 100%) !important;
    -webkit-background-clip: text !important; -webkit-text-fill-color: transparent !important; background-clip: text !important;
}
[data-theme="light"] .md-sidebar-text    { color: #5a5270 !important; }
[data-theme="light"] .md-sidebar-section { color: #7c6a9a !important; }
[data-theme="light"] .md-sb-stat-chip {
    background: rgba(255,255,255,0.72) !important;
    border-color: rgba(103,80,164,0.16) !important;
}
[data-theme="light"] .md-sb-stat-val { color: #1a1530 !important; }
[data-theme="light"] .md-sb-stat-lbl { color: #7c6a9a !important; }
[data-theme="light"] .md-sidebar-link {
    color: #1a1530 !important;
}
[data-theme="light"] .md-sidebar-link:hover {
    background: rgba(103,80,164,0.09) !important;
    border-color: rgba(103,80,164,0.26) !important;
}
[data-theme="light"] .md-sidebar-link.active {
    background: linear-gradient(135deg, rgba(103,80,164,0.15), rgba(0,106,106,0.10)) !important;
    border-color: rgba(103,80,164,0.40) !important;
}
[data-theme="light"] .md-sidebar-icon {
    background: rgba(255,255,255,0.72) !important;
    border-color: rgba(103,80,164,0.18) !important;
}
[data-theme="light"] .md-sidebar-link-title { color: #1a1530 !important; }
[data-theme="light"] .md-sidebar-link-sub   { color: #5a5270 !important; }
[data-theme="light"] .md-sb-link-arrow      { color: #7c6a9a !important; }
[data-theme="light"] .md-sidebar-note {
    background: rgba(245,158,11,0.08) !important;
    border-color: rgba(245,158,11,0.28) !important;
    color: #5a5270 !important;
}
[data-theme="light"] .md-sidebar-footer {
    color: #7c6a9a !important;
    border-top-color: rgba(103,80,164,0.18) !important;
}
[data-theme="light"] .md-sb-creator {
    background: linear-gradient(90deg, #db2777, #7c3aed, #4f46e5) !important;
    -webkit-background-clip: text !important; -webkit-text-fill-color: transparent !important;
    background-clip: text !important; filter: drop-shadow(0 0 5px rgba(124,58,237,.30)) !important;
}
[data-theme="light"] .md-sb-version-pill {
    background: rgba(103,80,164,0.12) !important;
    border-color: rgba(103,80,164,0.28) !important;
    color: #3a2a7d !important;
}

/* Hero */
[data-theme="light"] .md-hero {
    background:
        linear-gradient(135deg, rgba(103,80,164,0.12), rgba(0,106,106,0.07) 58%, rgba(186,26,26,0.04)),
        rgba(255,255,255,0.78) !important;
    border-color: rgba(103,80,164,0.22) !important;
}
[data-theme="light"] .md-kicker {
    background: rgba(103,80,164,0.12) !important;
    border-color: rgba(103,80,164,0.32) !important;
    color: #4a3a7d !important;
}
[data-theme="light"] .md-title      { color: #1a1530 !important; }
[data-theme="light"] .md-subtitle   { color: #5a5270 !important; }
[data-theme="light"] .md-chip {
    background: rgba(255,255,255,0.78) !important;
    border-color: rgba(103,80,164,0.22) !important;
    color: #3a2a5d !important;
}
[data-theme="light"] .md-chip:hover {
    border-color: rgba(103,80,164,0.42) !important;
}
[data-theme="light"] .md-hero-image {
    background:
        linear-gradient(135deg, rgba(103,80,164,0.08), rgba(255,255,255,0.12)),
        rgba(103,80,164,0.05) !important;
    border-color: rgba(103,80,164,0.18) !important;
}
[data-theme="light"] .md-made-with { color: #3a2a5d !important; }

/* Stats */
[data-theme="light"] .md-stat-card {
    background: rgba(255,255,255,0.82) !important;
    border-color: rgba(103,80,164,0.18) !important;
}
[data-theme="light"] .md-stat-label { color: #7c6a9a !important; }
[data-theme="light"] .md-stat-value { color: #1a1530 !important; }

/* Section titles */
[data-theme="light"] .md-section-title   { color: #1a1530 !important; }
[data-theme="light"] .md-section-subtitle { color: #5a5270 !important; }

/* Feature cards */
[data-theme="light"] .md-card {
    background: rgba(255,255,255,0.82) !important;
    border-color: rgba(103,80,164,0.18) !important;
}
[data-theme="light"] .md-card:hover {
    background: rgba(255,255,255,0.94) !important;
    border-color: rgba(103,80,164,0.38) !important;
}
[data-theme="light"] .md-card h3 { color: #1a1530 !important; }
[data-theme="light"] .md-card p  { color: #5a5270 !important; }
[data-theme="light"] .md-card-icon {
    background: linear-gradient(135deg, rgba(103,80,164,0.18), rgba(0,106,106,0.12)) !important;
    border-color: rgba(103,80,164,0.20) !important;
}

/* Health shell / symptom shell */
[data-theme="light"] .md-health-shell,
[data-theme="light"] .md-symptom-shell {
    background:
        linear-gradient(135deg, rgba(103,80,164,0.10), rgba(0,106,106,0.06) 60%, rgba(103,80,164,0.03)),
        rgba(255,255,255,0.80) !important;
    border-color: rgba(103,80,164,0.20) !important;
}
[data-theme="light"] .md-pillar-card {
    background: rgba(255,255,255,0.78) !important;
    border-color: rgba(103,80,164,0.16) !important;
}
[data-theme="light"] .md-pillar-bar-wrap { background: rgba(103,80,164,0.12) !important; }

/* Risk cards */
[data-theme="light"] .md-risk-card {
    background: rgba(255,255,255,0.82) !important;
    border-color: rgba(103,80,164,0.16) !important;
}
[data-theme="light"] .md-risk-name  { color: #1a1530 !important; }
[data-theme="light"] .md-risk-sub   { color: #7c6a9a !important; }
[data-theme="light"] .md-risk-stat  { color: #1a1530 !important; }
[data-theme="light"] .md-risk-desc  { color: #5a5270 !important; }

/* Symptom tags */
[data-theme="light"] .md-symptom-tag { color: #3a2a5d !important; }

/* Triage banner */
[data-theme="light"] .md-triage-msg { color: #5a5270 !important; }
[data-theme="light"] .md-triage-level { color: #1a1530 !important; }
[data-theme="light"] .md-symptom-result {
    background: rgba(255,255,255,0.82) !important;
    border-color: rgba(103,80,164,0.18) !important;
}
[data-theme="light"] .md-conditions-row {
    background: rgba(255,255,255,0.78) !important;
    border-color: rgba(103,80,164,0.14) !important;
    color: #3a2a5d !important;
}

/* Timeline */
[data-theme="light"] .md-timeline-card {
    background: rgba(255,255,255,0.82) !important;
    border-color: rgba(103,80,164,0.16) !important;
}
[data-theme="light"] .md-timeline-title { color: #1a1530 !important; }
[data-theme="light"] .md-timeline-meta  { color: #7c6a9a !important; }
[data-theme="light"] .md-timeline-tag {
    background: rgba(103,80,164,0.12) !important;
    border-color: rgba(103,80,164,0.26) !important;
    color: #4a3a7d !important;
}

/* Vitals */
[data-theme="light"] .md-vital-card {
    background: rgba(255,255,255,0.82) !important;
    border-color: rgba(103,80,164,0.16) !important;
}
[data-theme="light"] .md-vital-value { color: #1a1530 !important; }
[data-theme="light"] .md-vital-unit,
[data-theme="light"] .md-vital-label { color: #7c6a9a !important; }

/* Why box */
[data-theme="light"] .md-why-box {
    background:
        linear-gradient(135deg, rgba(103,80,164,0.09), rgba(0,106,106,0.06)),
        rgba(255,255,255,0.80) !important;
    border-color: rgba(103,80,164,0.18) !important;
}
[data-theme="light"] .md-why-item {
    background: rgba(255,255,255,0.72) !important;
    border-color: rgba(103,80,164,0.14) !important;
}
[data-theme="light"] .md-why-item strong { color: #1a1530 !important; }
[data-theme="light"] .md-why-item span   { color: #5a5270 !important; }

/* Marquee */
[data-theme="light"] .md-marquee-wrap {
    background: rgba(255,255,255,0.80) !important;
    border-color: rgba(103,80,164,0.18) !important;
}
[data-theme="light"] .md-marquee-wrap::before {
    background: linear-gradient(90deg, #f5f2ff, transparent) !important;
}
[data-theme="light"] .md-marquee-wrap::after {
    background: linear-gradient(-90deg, #f5f2ff, transparent) !important;
}
[data-theme="light"] .md-marquee-item {
    background: rgba(255,255,255,0.78) !important;
    border-color: rgba(103,80,164,0.14) !important;
    color: #3a2a5d !important;
}

/* Contact */
[data-theme="light"] .md-contact-wrap {
    background: linear-gradient(135deg, rgba(103,80,164,0.10), rgba(0,106,106,0.06) 60%, rgba(103,80,164,0.04)),
    rgba(255,255,255,0.80) !important;
    border-color: rgba(103,80,164,0.20) !important;
}
[data-theme="light"] .md-contact-title { color: #1a1530 !important; }
[data-theme="light"] .md-contact-sub   { color: #5a5270 !important; }
[data-theme="light"] .md-contact-email {
    background: rgba(103,80,164,0.10) !important;
    border-color: rgba(103,80,164,0.28) !important;
    color: #3a2a5d !important;
}
[data-theme="light"] .md-contact-email:hover { color: #1a1530 !important; }
[data-theme="light"] .md-contact-social-btn {
    background: rgba(255,255,255,0.78) !important;
    border-color: rgba(103,80,164,0.16) !important;
    color: #3a2a5d !important;
}
[data-theme="light"] .md-contact-social-btn:hover { color: #1a1530 !important; }

/* Footer */
[data-theme="light"] .md-footer { border-top-color: rgba(103,80,164,0.18) !important; }
[data-theme="light"] .md-footer-brand-name  { color: #1a1530 !important; }
[data-theme="light"] .md-footer-brand-sub   { color: #7c6a9a !important; }
[data-theme="light"] .md-footer-link {
    background: rgba(255,255,255,0.78) !important;
    border-color: rgba(103,80,164,0.16) !important;
    color: #5a5270 !important;
}
[data-theme="light"] .md-footer-link:hover  { color: #3a2a5d !important; }
[data-theme="light"] .md-footer-meta        { color: #7c6a9a !important; }
[data-theme="light"] .md-footer-version {
    background: rgba(103,80,164,0.10) !important;
    border-color: rgba(103,80,164,0.24) !important;
    color: #4a3a7d !important;
}
[data-theme="light"] .md-footer-disclaimer  { color: #5a5270 !important; }

/* Streamlit inputs */
[data-theme="light"] input,
[data-theme="light"] textarea,
[data-theme="light"] [data-baseweb="select"] {
    background: rgba(255,255,255,0.90) !important;
    border-color: rgba(103,80,164,0.25) !important;
    color: #1a1530 !important;
}
[data-theme="light"] label,
[data-theme="light"] .stRadio label,
[data-theme="light"] .stSelectbox label,
[data-theme="light"] .stMultiSelect label,
[data-theme="light"] .stTextInput label,
[data-theme="light"] .stTextArea label,
[data-theme="light"] p,
[data-theme="light"] li {
    color: #1d1b27 !important;
}
[data-theme="light"] .stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.80) !important;
    border-color: rgba(103,80,164,0.16) !important;
}
[data-theme="light"] .stTabs [data-baseweb="tab"] { color: #3a2a5d !important; }
[data-theme="light"] .stTabs [aria-selected="true"] { color: white !important; }
</style>
"""
    if st.session_state.get("theme") == "light":
        styles = styles.replace('[data-theme="light"] ', '').replace('[data-theme="light"]', ":root")
    st.markdown(styles, unsafe_allow_html=True)


def render_sidebar_link(icon, title, subtitle, active=False):
    active_cls = " active" if active else ""
    st.sidebar.markdown(
        f"""
<div class="md-sidebar-link{active_cls}">
    <div class="md-sidebar-icon">{escape(icon)}</div>
    <div class="md-sb-link-body">
        <div class="md-sidebar-link-title">{escape(title)}</div>
        <div class="md-sidebar-link-sub">{escape(subtitle)}</div>
    </div>
    <span class="md-sb-link-arrow">›</span>
</div>
""",
        unsafe_allow_html=True,
    )


def render_sidebar():
    with st.sidebar:
        st.markdown(
            """
<div class="md-sb-wrap">
<div class="md-sidebar-hero">
    <div class="md-sb-hero-row">
        <div class="md-sb-logo">🏥</div>
        <div class="md-sb-brand">
            <div class="md-sidebar-kicker">AI Healthcare Hub</div>
            <div class="md-sidebar-title">SmartHealthCare</div>
        </div>
    </div>
    <div class="md-sidebar-text">Early diagnosis support with prediction tools, medical guidance, BMI insights, and AI assistance.</div>
    <div class="md-sb-stats-row">
        <div class="md-sb-stat-chip">
            <span class="md-sb-stat-val">4+</span>
            <span class="md-sb-stat-lbl">AI Tools</span>
        </div>
        <div class="md-sb-stat-chip">
            <span class="md-sb-stat-val">40+</span>
            <span class="md-sb-stat-lbl">Diseases</span>
        </div>
        <div class="md-sb-stat-chip">
            <span class="md-sb-stat-val">RF</span>
            <span class="md-sb-stat-lbl">Model</span>
        </div>
        <div class="md-sb-stat-chip">
            <span class="md-sb-stat-val">24/7</span>
            <span class="md-sb-stat-lbl">Available</span>
        </div>
    </div>
</div>
""",
            unsafe_allow_html=True,
        )

        st.markdown("<div class='md-sidebar-section'>Explore</div>", unsafe_allow_html=True)

        render_sidebar_link("🏥", "Services", "AI healthcare features")
        render_sidebar_link("💬", "Medibot", "Ask health questions")
        render_sidebar_link("❤️", "Heart Risk", "Assessment and report")
        render_sidebar_link("📏", "BMI Calculator", "Quick body mass index check")

        st.markdown(
            """
<div class="md-sidebar-section">Support</div>

<div class="md-sidebar-note">
    <div class="md-sb-note-icon">⚠️</div>
    <div><strong>Medical Note</strong><br>Educational and decision-support use only. Not a substitute for professional medical advice.</div>
</div>

<div class="md-sidebar-footer">
    Made with <span style="color:#ef4444;filter:drop-shadow(0 0 6px rgba(239,68,68,.6))">❤️</span> by <span class="md-sb-creator">Yatin Sharma</span><br>
    <span class="md-sb-version-pill">SmartHealthCare AI</span>
</div>
</div>
""",
            unsafe_allow_html=True,
        )


def render_hero():
    hero_data = image_to_data_uri(HERO_IMAGE)

    if hero_data:
        hero_visual = f"<img class='md-hero-image' src='{hero_data}' alt='SmartHealthCare preview'>"
    else:
        hero_visual = "<div class='md-hero-image-fallback'>🩺</div>"

    st.markdown(
        f"""
<div class="md-hero">
    <div class="md-hero-grid">
        <div>
            <div class="md-kicker">SmartHealthCare AI Network</div>
            <h1 class="md-title">SmartHealthCare</h1>
            <div class="md-subtitle">
                Early diagnosis support using AI-driven predictions, health insights, explainable reports, and an intelligent medical assistant.
            </div>
            <div class="md-chip-row">
                <div class="md-chip">Disease Prediction</div>
                <div class="md-chip">Drug Recommendation</div>
                <div class="md-chip">Heart Risk Assessment</div>
                <div class="md-chip">Medical Chatbot</div>
            </div>
        </div>
        <div>{hero_visual}</div>
    </div>
</div>
<div class="md-made-with">Made With <span class="heart">❤️</span> By {CREATOR_NAME}</div>
""",
        unsafe_allow_html=True,
    )


def render_stats():
    st.markdown(
        """
<div class="md-stat-grid">
    <div class="md-stat-card">
        <div class="md-stat-label">Platform</div>
        <div class="md-stat-value">AI Healthcare Hub</div>
    </div>
    <div class="md-stat-card">
        <div class="md-stat-label">Tools</div>
        <div class="md-stat-value">Prediction + Guidance</div>
    </div>
    <div class="md-stat-card">
        <div class="md-stat-label">Reports</div>
        <div class="md-stat-value">Readable Summaries</div>
    </div>
    <div class="md-stat-card">
        <div class="md-stat-label">Experience</div>
        <div class="md-stat-value">Responsive UI</div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_feature_cards():
    st.markdown("<div class='md-section-title'>✨ Features</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='md-section-subtitle'>A focused healthcare platform built around prediction, guidance, reporting, and user-friendly AI assistance.</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
<div class="md-card-grid">
    <div class="md-card">
        <div class="md-card-icon">💡</div>
        <h3>Disease Prediction</h3>
        <p>Analyze symptoms and predict possible diseases using advanced AI models.</p>
    </div>
    <div class="md-card">
        <div class="md-card-icon">💊</div>
        <h3>Drug Recommendation</h3>
        <p>Get AI-powered medication suggestions based on medical history and diagnosis.</p>
    </div>
    <div class="md-card">
        <div class="md-card-icon">❤️</div>
        <h3>Heart Risk Assessment</h3>
        <p>Assess your heart health and receive an AI-powered risk analysis with recommendations.</p>
    </div>
    <div class="md-card">
        <div class="md-card-icon">🤖</div>
        <h3>LLM Chatbot</h3>
        <p>Chat with an AI-powered medical assistant for fast health information support.</p>
    </div>
    <div class="md-card">
        <div class="md-card-icon">🔒</div>
        <h3>Data Privacy</h3>
        <p>Designed with privacy-first workflows and secure user experience patterns.</p>
    </div>
    <div class="md-card">
        <div class="md-card-icon">📄</div>
        <h3>Reports</h3>
        <p>Create clean health reports that summarize results and recommendations.</p>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )


# ── NEW: AI HEALTH TIPS MARQUEE ───────────────────────────────────────────────
def render_health_tips_marquee():
    tips = [
        ("💧", "Drink 8 glasses of water daily"),
        ("🏃", "30 min of exercise 5x a week"),
        ("🥗", "Eat 5 servings of fruits & veggies"),
        ("😴", "Aim for 7-9 hours of quality sleep"),
        ("🧘", "Practice mindfulness to reduce cortisol"),
        ("🚭", "Avoid smoking — top cause of preventable disease"),
        ("☀️", "Get 15 min of sunlight for Vitamin D"),
        ("🩺", "Schedule annual health checkups"),
        ("🧂", "Limit sodium intake below 2300 mg/day"),
        ("🫁", "Deep breathing improves lung capacity"),
        ("💧", "Drink 8 glasses of water daily"),
        ("🏃", "30 min of exercise 5x a week"),
        ("🥗", "Eat 5 servings of fruits & veggies"),
        ("😴", "Aim for 7-9 hours of quality sleep"),
        ("🧘", "Practice mindfulness to reduce cortisol"),
        ("🚭", "Avoid smoking — top cause of preventable disease"),
        ("☀️", "Get 15 min of sunlight for Vitamin D"),
        ("🩺", "Schedule annual health checkups"),
        ("🧂", "Limit sodium intake below 2300 mg/day"),
        ("🫁", "Deep breathing improves lung capacity"),
    ]
    items_html = "".join(
        f'<div class="md-marquee-item"><span>{icon}</span><span>{text}</span></div>'
        for icon, text in tips
    )
    st.markdown(
        f"""
<div class="md-section-title">💡 Daily Health Tips</div>
<div class="md-section-subtitle">Evidence-backed micro-habits that make a macro difference over time.</div>
<div class="md-marquee-wrap">
    <div class="md-marquee-track">{items_html}</div>
</div>
""",
        unsafe_allow_html=True,
    )


# ── NEW: DISEASE RISK AWARENESS CARDS ────────────────────────────────────────
def render_risk_awareness():
    st.markdown("<div class='md-section-title'>⚠️ Disease Risk Awareness</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='md-section-subtitle'>Know the numbers. These statistics underline why early detection is critical for the diseases our AI monitors.</div>",
        unsafe_allow_html=True,
    )

    risk_cards = [
        {
            "icon": "❤️",
            "icon_bg": "rgba(225,29,72,0.13)",
            "icon_border": "rgba(225,29,72,0.28)",
            "name": "Cardiovascular Disease",
            "sub": "Global leading cause of death",
            "stat": "17.9M",
            "stat_color": "#e11d48",
            "unit": "deaths per year worldwide",
            "desc": "Early AI-based heart risk scoring can flag at-risk individuals years before a cardiac event — enabling lifestyle changes that genuinely save lives.",
            "bar_bg": "rgba(225,29,72,0.15)",
            "bar_fill": "#e11d48",
            "bar_pct": "85",
            "bar_label": "85% of cases are preventable with early action",
        },
        {
            "icon": "🩸",
            "icon_bg": "rgba(37,99,235,0.13)",
            "icon_border": "rgba(37,99,235,0.28)",
            "name": "Diabetes",
            "sub": "Rising metabolic epidemic",
            "stat": "537M",
            "stat_color": "#2563eb",
            "unit": "adults living with diabetes (2021)",
            "desc": "50% of diabetics remain undiagnosed. AI-driven glucose pattern analysis and symptom mapping can dramatically accelerate identification.",
            "bar_bg": "rgba(37,99,235,0.15)",
            "bar_fill": "#2563eb",
            "bar_pct": "50",
            "bar_label": "50% undiagnosed — early testing is vital",
        },
        {
            "icon": "🫁",
            "icon_bg": "rgba(5,150,105,0.13)",
            "icon_border": "rgba(5,150,105,0.28)",
            "name": "Respiratory Disease",
            "sub": "Chronic & infectious lung conditions",
            "stat": "4M",
            "stat_color": "#059669",
            "unit": "deaths from COPD annually",
            "desc": "AI symptom checkers help differentiate between viral, bacterial, and chronic respiratory conditions — speeding up triage and appropriate care pathways.",
            "bar_bg": "rgba(5,150,105,0.15)",
            "bar_fill": "#059669",
            "bar_pct": "70",
            "bar_label": "70% of cases caught late due to delayed screening",
        },
    ]

    cols = st.columns(3)
    for col, card in zip(cols, risk_cards):
        with col:
            st.markdown(
                f"""<div class="md-risk-card">
  <div class="md-risk-header">
    <div class="md-risk-icon-wrap" style="background:{card['icon_bg']};border:1px solid {card['icon_border']};">{card['icon']}</div>
    <div>
      <div class="md-risk-name">{card['name']}</div>
      <div class="md-risk-sub">{card['sub']}</div>
    </div>
  </div>
  <div class="md-risk-stat" style="color:{card['stat_color']};">{card['stat']}</div>
  <div style="font-size:12px;color:var(--md-soft);">{card['unit']}</div>
  <div class="md-risk-desc">{card['desc']}</div>
  <div style="margin-top:12px;height:6px;border-radius:999px;background:{card['bar_bg']};overflow:hidden;">
    <div style="width:{card['bar_pct']}%;height:100%;background:{card['bar_fill']};border-radius:999px;"></div>
  </div>
  <div style="font-size:11px;color:var(--md-soft);margin-top:4px;">{card['bar_label']}</div>
</div>""",
                unsafe_allow_html=True,
            )


# ── REDESIGNED: QUICK HEALTH SELF-ASSESSMENT (M3 Expressive) ──────────────────
def render_health_self_assessment():
    st.markdown(
        "<div class='md-section-title hsa-section-heading'><span class='hsa-heading-icon' aria-hidden='true'>&#129658;</span> Quick Health Self-Assessment</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='md-section-subtitle'>Answer a few lifestyle questions and get a personalized wellness score with tailored recommendations.</div>",
        unsafe_allow_html=True,
    )

    # ── styled shell open ──────────────────────────────────────────────────────
    hsa_markup = """
<style>
/* ── MD3 Expressive: Quick Health Self-Assessment ────────────────────── */
.hsa-shell {
    border: 1px solid rgba(103,80,164,0.30);
    border-radius: 32px;
    padding: 28px 26px 22px;
    background:
        radial-gradient(ellipse at 90% 10%, rgba(103,80,164,0.18) 0%, transparent 55%),
        radial-gradient(ellipse at 5% 90%, rgba(0,106,106,0.13) 0%, transparent 50%),
        linear-gradient(160deg, rgba(103,80,164,0.11) 0%, rgba(0,106,106,0.07) 55%, rgba(103,80,164,0.04) 100%);
    margin-bottom: 6px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 4px 32px rgba(103,80,164,0.14), 0 1px 4px rgba(0,0,0,0.10);
}
.hsa-shell::before {
    content: '';
    position: absolute;
    top: -48px; right: -48px;
    width: 160px; height: 160px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(103,80,164,0.22) 0%, transparent 70%);
    pointer-events: none;
}
.hsa-shell::after {
    content: '';
    position: absolute;
    bottom: -56px; left: -40px;
    width: 180px; height: 180px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(0,106,106,0.15) 0%, transparent 70%);
    pointer-events: none;
}

/* Result card */
.hsa-result {
    border: 1px solid rgba(103,80,164,0.22);
    border-radius: 28px;
    padding: 24px 22px 20px;
    background:
        linear-gradient(135deg, rgba(103,80,164,0.10) 0%, rgba(0,106,106,0.06) 60%, rgba(103,80,164,0.04) 100%);
    margin-top: 14px;
    position: relative;
    overflow: hidden;
    animation: hsa-rise-in 400ms cubic-bezier(0.34,1.56,0.64,1) both;
    box-shadow: 0 8px 32px rgba(103,80,164,0.12);
}
.hsa-result::before {
    content: '';
    position: absolute;
    inset: 0 0 auto 0;
    height: 3px;
    background: linear-gradient(90deg, #6750a4, #006a6a, #6750a4);
    border-radius: 28px 28px 0 0;
}
@keyframes hsa-rise-in {
    from { opacity: 0; transform: translateY(16px) scale(0.97); }
    to   { opacity: 1; transform: translateY(0) scale(1); }
}

/* Score ring */
.hsa-ring-wrap {
    text-align: center;
    margin-bottom: 20px;
}
.hsa-ring {
    width: 122px; height: 122px;
    border-radius: 50%;
    display: inline-flex; flex-direction: column;
    align-items: center; justify-content: center;
    border: 6px solid currentColor;
    position: relative;
    box-shadow: 0 0 0 3px rgba(255,255,255,0.04), 0 0 32px rgba(0,0,0,0.18);
    backdrop-filter: blur(6px);
    transition: transform 220ms ease;
}
.hsa-ring:hover { transform: scale(1.05); }
.hsa-ring::before {
    content: '';
    position: absolute;
    inset: 6px;
    border-radius: 50%;
    background: radial-gradient(circle at 35% 30%, rgba(255,255,255,0.08), transparent 65%);
    pointer-events: none;
}
.hsa-ring-num {
    font-size: 34px; font-weight: 900;
    font-family: var(--font-display, 'Outfit', sans-serif);
    line-height: 1;
}
.hsa-ring-tag {
    font-size: 10px; font-weight: 800;
    letter-spacing: 0.09em; text-transform: uppercase;
    margin-top: 4px; opacity: 0.88;
}
.hsa-ring-sub {
    font-size: 12px; color: var(--md-soft);
    margin-top: 8px; font-weight: 600;
}

/* Pillar grid */
.hsa-pillars {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px; margin-top: 8px;
}
@media (max-width: 600px) { .hsa-pillars { grid-template-columns: 1fr 1fr; } }

.hsa-pillar {
    background: rgba(255,255,255,0.055);
    border: 1px solid rgba(148,163,184,0.18);
    border-radius: 18px;
    padding: 12px 13px;
    transition: transform 150ms ease, border-color 150ms ease, background 150ms ease;
    position: relative; overflow: hidden;
}
.hsa-pillar:hover {
    transform: translateY(-2px);
    border-color: rgba(103,80,164,0.34);
    background: rgba(103,80,164,0.08);
}
.hsa-pillar::after {
    content: '';
    position: absolute; bottom: -18px; right: -14px;
    width: 44px; height: 44px; border-radius: 50%;
    background: rgba(255,255,255,0.03);
    pointer-events: none;
}
.hsa-p-top {
    display: flex; justify-content: space-between;
    align-items: baseline; margin-bottom: 7px;
}
.hsa-p-name { font-size: 11px; color: var(--md-soft); font-weight: 700; }
.hsa-p-pct {
    font-size: 17px; font-weight: 900;
    font-family: var(--font-display, 'Outfit', sans-serif);
}
.hsa-bar-track {
    height: 5px; border-radius: 3px;
    background: rgba(148,163,184,0.14); overflow: hidden;
}
.hsa-bar-fill {
    height: 100%; border-radius: 3px;
    transition: width 700ms cubic-bezier(0.4,0,0.2,1);
}
.hsa-p-status {
    font-size: 10px; font-weight: 800;
    margin-top: 5px; letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* Tip box */
.hsa-tip-box {
    margin-top: 18px; padding: 15px 17px;
    border-radius: 20px;
    border: 1px solid rgba(103,80,164,0.22);
    border-left: 4px solid #6750a4;
    background: linear-gradient(135deg, rgba(103,80,164,0.12), rgba(0,106,106,0.07));
    position: relative; overflow: hidden;
}
.hsa-tip-box::before {
    content: '';
    position: absolute; top: -20px; right: -20px;
    width: 60px; height: 60px; border-radius: 50%;
    background: radial-gradient(circle, rgba(103,80,164,0.18), transparent 70%);
    pointer-events: none;
}
.hsa-tip-title {
    font-size: 12px; font-weight: 900;
    margin-bottom: 8px; color: #d0bcff;
    letter-spacing: 0.04em;
}
.hsa-tip-list { padding-left: 15px; margin: 0; }
.hsa-tip-list li {
    color: var(--md-soft); font-size: 12px;
    line-height: 1.70; margin-bottom: 4px;
}

/* Disclaimer */
.hsa-disc {
    margin-top: 14px; padding: 10px 14px;
    border-radius: 14px;
    background: rgba(217,119,6,0.08);
    border: 1px solid rgba(217,119,6,0.24);
    font-size: 11px; color: var(--md-soft);
    text-align: center; font-weight: 600;
}

/* Light-mode overrides */
[data-theme="light"] .hsa-shell {
    background:
        radial-gradient(ellipse at 90% 10%, rgba(103,80,164,0.14) 0%, transparent 55%),
        radial-gradient(ellipse at 5% 90%, rgba(0,106,106,0.10) 0%, transparent 50%),
        linear-gradient(160deg, rgba(103,80,164,0.09) 0%, rgba(0,106,106,0.05) 55%, rgba(103,80,164,0.03) 100%) !important;
    border-color: rgba(103,80,164,0.26) !important;
    box-shadow: 0 4px 32px rgba(103,80,164,0.10), 0 1px 4px rgba(103,80,164,0.06) !important;
}
[data-theme="light"] .hsa-result {
    background:
        linear-gradient(135deg, rgba(103,80,164,0.08) 0%, rgba(0,106,106,0.05) 60%, rgba(103,80,164,0.03) 100%) !important;
    border-color: rgba(103,80,164,0.18) !important;
}
[data-theme="light"] .hsa-pillar {
    background: rgba(255,255,255,0.72) !important;
    border-color: rgba(103,80,164,0.16) !important;
}
[data-theme="light"] .hsa-pillar:hover {
    background: rgba(103,80,164,0.08) !important;
    border-color: rgba(103,80,164,0.30) !important;
}
[data-theme="light"] .hsa-tip-box {
    background: linear-gradient(135deg, rgba(103,80,164,0.09), rgba(0,106,106,0.05)) !important;
    border-color: rgba(103,80,164,0.20) !important;
    border-left-color: #6750a4 !important;
}
[data-theme="light"] .hsa-tip-title { color: #4a3a7d !important; }
[data-theme="light"] .hsa-ring-sub  { color: #5a5270 !important; }
[data-theme="light"] .hsa-p-name    { color: #5a5270 !important; }
[data-theme="light"] .hsa-tip-list li { color: #5a5270 !important; }
[data-theme="light"] .hsa-disc      { color: #5a5270 !important; }

/* MD3 expressive refresh */
.hsa-shell {
    border-radius: 30px !important;
    padding: 22px !important;
    background:
        linear-gradient(135deg, rgba(103,80,164,0.15), rgba(0,106,106,0.09) 56%, rgba(217,119,6,0.06)),
        var(--md-surface) !important;
    box-shadow: var(--md-shadow-2) !important;
    contain: layout paint;
    isolation: isolate;
}
.hsa-section-heading {
    display: flex !important;
    align-items: center;
    justify-content: center;
    gap: 10px;
}
.hsa-heading-icon {
    width: 42px;
    height: 42px;
    border-radius: 16px;
    display: inline-grid;
    place-items: center;
    background: linear-gradient(135deg, rgba(103,80,164,0.18), rgba(0,106,106,0.13));
    border: 1px solid var(--md-outline-variant);
    box-shadow: var(--md-shadow-1);
    font-size: 22px;
    line-height: 1;
}
.hsa-header {
    display: grid;
    grid-template-columns: 54px minmax(0, 1fr);
    gap: 14px;
    align-items: center;
    padding: 4px 2px 18px;
    position: relative;
    z-index: 1;
}
.hsa-head-icon {
    width: 54px;
    height: 54px;
    border-radius: 18px;
    display: grid;
    place-items: center;
    background: linear-gradient(135deg, rgba(103,80,164,0.24), rgba(0,106,106,0.18));
    border: 1px solid rgba(255,255,255,0.18);
    color: #f4efff;
    font-size: 18px;
    font-weight: 950;
    font-family: var(--font-display, 'Outfit', sans-serif);
    box-shadow: 0 10px 24px rgba(103,80,164,0.18);
}
.hsa-head-title {
    margin: 0;
    color: #f7f2ff;
    font-size: 20px;
    font-weight: 950;
    line-height: 1.12;
    font-family: var(--font-display, 'Outfit', sans-serif);
}
.hsa-head-copy {
    margin-top: 5px;
    color: var(--md-soft);
    font-size: 12px;
    line-height: 1.5;
}
.hsa-form-zone {
    position: relative;
    z-index: 1;
    border-radius: 24px;
    padding: 16px 16px 18px;
    border: 1px solid var(--md-outline-variant);
    background: rgba(255,255,255,0.055);
}
.hsa-shell [data-testid="stSlider"],
.hsa-shell [data-testid="stSelectSlider"],
.hsa-shell [data-testid="stRadio"],
[data-testid="stForm"] [data-testid="stSlider"],
[data-testid="stForm"] [data-testid="stSelectSlider"],
[data-testid="stForm"] [data-testid="stRadio"] {
    padding: 13px 14px 14px !important;
    margin-bottom: 10px;
    border-radius: 22px;
    border: 1px solid var(--md-outline-variant);
    background:
        linear-gradient(135deg, rgba(255,255,255,0.08), rgba(255,255,255,0.035)),
        rgba(255,255,255,0.045);
    box-shadow: 0 8px 20px rgba(15,23,42,0.08);
    min-height: 92px;
}
.hsa-shell label,
.hsa-shell [data-testid="stWidgetLabel"] p,
[data-testid="stForm"] label,
[data-testid="stForm"] [data-testid="stWidgetLabel"] p {
    font-weight: 850 !important;
    color: rgba(241,245,249,0.88) !important;
    letter-spacing: 0 !important;
    line-height: 1.25 !important;
}
.hsa-shell [data-baseweb="slider"] div,
[data-testid="stForm"] [data-baseweb="slider"] div,
.hsa-shell [role="radiogroup"] label,
[data-testid="stForm"] [role="radiogroup"] label {
    min-height: 34px;
}
[data-testid="stForm"] {
    border: 0 !important;
    background: transparent !important;
    padding: 0 !important;
}
[data-testid="stForm"] [data-baseweb="slider"] {
    padding: 16px 4px 6px !important;
}
[data-testid="stForm"] [data-baseweb="slider"] [role="slider"] {
    width: 24px !important;
    height: 24px !important;
    border-radius: 50% !important;
    background: linear-gradient(135deg, #d0bcff, #80cbc4) !important;
    border: 4px solid #161020 !important;
    box-shadow: 0 0 0 5px rgba(208,188,255,0.14), 0 8px 18px rgba(0,0,0,0.22) !important;
}
[data-testid="stForm"] [data-baseweb="slider"] [role="slider"]:focus {
    outline: none !important;
    box-shadow: 0 0 0 7px rgba(208,188,255,0.22), 0 8px 18px rgba(0,0,0,0.22) !important;
}
[data-testid="stForm"] [data-baseweb="slider"] div {
    border-radius: 999px !important;
}
[data-testid="stForm"] [data-testid="stTickBar"],
[data-testid="stForm"] [data-testid="stThumbValue"],
[data-testid="stForm"] [data-testid="stSliderTickBarMin"],
[data-testid="stForm"] [data-testid="stSliderTickBarMax"] {
    color: var(--md-soft) !important;
    font-weight: 800 !important;
}
[data-testid="stForm"] [role="radiogroup"] {
    display: grid !important;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 8px;
}
[data-testid="stForm"] [role="radiogroup"] label {
    width: 100%;
    margin: 0 !important;
    padding: 8px 10px !important;
    border-radius: 999px !important;
    border: 1px solid var(--md-outline-variant);
    background: rgba(255,255,255,0.055);
    justify-content: center;
}
[data-testid="stForm"] [data-testid="stFormSubmitButton"] button {
    min-height: 52px !important;
    border-radius: 18px !important;
    background: linear-gradient(135deg, #6750a4, #006a6a) !important;
    border: 1px solid rgba(208,188,255,0.32) !important;
    box-shadow: 0 12px 28px rgba(103,80,164,0.24) !important;
    font-size: 14px !important;
    font-weight: 950 !important;
}
.hsa-result {
    border-radius: 28px !important;
    padding: 20px !important;
    background:
        linear-gradient(135deg, rgba(255,255,255,0.08), rgba(255,255,255,0.035)),
        rgba(15,23,42,0.16) !important;
}
.hsa-result-grid {
    display: grid;
    grid-template-columns: minmax(150px, .65fr) minmax(0, 1.35fr);
    gap: 16px;
    align-items: center;
}
.hsa-ring {
    width: 116px !important;
    height: 116px !important;
    border-width: 7px !important;
    box-shadow: 0 10px 34px rgba(0,0,0,0.14) !important;
    backdrop-filter: none !important;
}
.hsa-pillars {
    grid-template-columns: repeat(3, minmax(0, 1fr)) !important;
    gap: 10px !important;
    margin-top: 0 !important;
}
.hsa-pillar {
    min-height: 88px;
    border-radius: 18px !important;
    padding: 12px !important;
}
.hsa-p-name {
    display: inline-flex;
    gap: 6px;
    align-items: center;
    min-width: 0;
}
.hsa-code {
    display: inline-grid;
    place-items: center;
    width: 24px;
    height: 24px;
    border-radius: 9px;
    background: rgba(255,255,255,0.08);
    font-size: 10px;
    font-weight: 950;
    color: currentColor;
    flex: 0 0 auto;
}
.hsa-tip-box {
    border-radius: 22px !important;
    margin-top: 16px !important;
}
.hsa-disc {
    border-radius: 16px !important;
}
[data-theme="light"] .hsa-shell {
    background:
        linear-gradient(135deg, rgba(91,63,196,0.11), rgba(0,180,166,0.075) 58%, rgba(249,115,22,0.055)),
        #fffbff !important;
    box-shadow: 0 14px 38px rgba(91,63,196,0.12) !important;
}
[data-theme="light"] .hsa-head-icon {
    background: linear-gradient(135deg, rgba(91,63,196,0.16), rgba(0,180,166,0.12)) !important;
    border-color: rgba(91,63,196,0.18) !important;
    color: #4a33a3 !important;
}
[data-theme="light"] .hsa-head-title { color: #1a1530 !important; }
[data-theme="light"] .hsa-head-copy { color: #5a5270 !important; }
[data-theme="light"] .hsa-form-zone,
[data-theme="light"] .hsa-result {
    background: #ffffff !important;
    border-color: rgba(91,63,196,0.16) !important;
}
[data-theme="light"] .hsa-shell label,
[data-theme="light"] .hsa-shell [data-testid="stWidgetLabel"] p,
[data-theme="light"] [data-testid="stForm"] label,
[data-theme="light"] [data-testid="stForm"] [data-testid="stWidgetLabel"] p {
    color: #1a1530 !important;
}
[data-theme="light"] .hsa-heading-icon {
    background: linear-gradient(135deg, rgba(91,63,196,0.14), rgba(0,180,166,0.10)) !important;
    border-color: rgba(91,63,196,0.18) !important;
}
[data-theme="light"] [data-testid="stForm"] [data-testid="stSlider"],
[data-theme="light"] [data-testid="stForm"] [data-testid="stSelectSlider"],
[data-theme="light"] [data-testid="stForm"] [data-testid="stRadio"] {
    background:
        linear-gradient(135deg, rgba(91,63,196,0.055), rgba(0,180,166,0.035)),
        #ffffff !important;
    border-color: rgba(91,63,196,0.16) !important;
    box-shadow: 0 8px 22px rgba(91,63,196,0.08) !important;
}
[data-theme="light"] [data-testid="stForm"] [data-baseweb="slider"] [role="slider"] {
    background: linear-gradient(135deg, #6750a4, #006a6a) !important;
    border-color: #ffffff !important;
    box-shadow: 0 0 0 5px rgba(103,80,164,0.12), 0 8px 18px rgba(91,63,196,0.18) !important;
}
[data-theme="light"] [data-testid="stForm"] [role="radiogroup"] label {
    background: rgba(91,63,196,0.055) !important;
    border-color: rgba(91,63,196,0.16) !important;
}
[data-theme="light"] [data-testid="stForm"] [data-testid="stTickBar"],
[data-theme="light"] [data-testid="stForm"] [data-testid="stThumbValue"],
[data-theme="light"] [data-testid="stForm"] [data-testid="stSliderTickBarMin"],
[data-theme="light"] [data-testid="stForm"] [data-testid="stSliderTickBarMax"] {
    color: #5a5270 !important;
}
[data-theme="light"] .hsa-code { background: rgba(91,63,196,0.10) !important; }
@media (max-width: 760px) {
    .hsa-shell {
        padding: 16px 14px !important;
        border-radius: 24px !important;
    }
    .hsa-header {
        grid-template-columns: 46px minmax(0, 1fr);
        gap: 11px;
        padding-bottom: 14px;
    }
    .hsa-head-icon {
        width: 46px;
        height: 46px;
        border-radius: 16px;
        font-size: 15px;
    }
    .hsa-section-heading {
        gap: 8px;
        flex-wrap: wrap;
    }
    .hsa-heading-icon {
        width: 36px;
        height: 36px;
        border-radius: 14px;
        font-size: 19px;
    }
    .hsa-head-title { font-size: 17px; }
    .hsa-head-copy { font-size: 11px; }
    .hsa-form-zone {
        padding: 12px;
        border-radius: 20px;
    }
    [data-testid="stForm"] [data-testid="stSlider"],
    [data-testid="stForm"] [data-testid="stSelectSlider"],
    [data-testid="stForm"] [data-testid="stRadio"] {
        padding: 11px 12px 13px !important;
        border-radius: 18px !important;
        min-height: auto;
    }
    [data-testid="stForm"] [data-baseweb="slider"] {
        padding-top: 12px !important;
    }
    [data-testid="stForm"] [role="radiogroup"] {
        grid-template-columns: 1fr;
    }
    [data-testid="stForm"] [data-testid="stFormSubmitButton"] button {
        min-height: 48px !important;
        border-radius: 16px !important;
    }
    .hsa-result-grid {
        grid-template-columns: 1fr;
        gap: 14px;
    }
    .hsa-pillars {
        grid-template-columns: 1fr !important;
    }
    .hsa-pillar {
        min-height: 76px;
    }
    .hsa-tip-list {
        padding-left: 18px !important;
    }
}
</style>
<div class="hsa-shell">
  <div class="hsa-header">
    <div class="hsa-head-icon">AI</div>
    <div>
      <div class="hsa-head-title">Lifestyle readiness scan</div>
      <div class="hsa-head-copy">Fast, local scoring across sleep, movement, hydration, stress, smoking, and age.</div>
    </div>
  </div>
  <div class="hsa-form-zone">
"""
    if st.session_state.get("theme") == "light":
        hsa_markup = hsa_markup.replace('[data-theme="light"] ', '').replace('[data-theme="light"]', ":root")
    st.markdown(hsa_markup, unsafe_allow_html=True)

    with st.form("hsa_form", clear_on_submit=False):
        col_a, col_b = st.columns(2)
        with col_a:
            age = st.slider("Your Age", 10, 90, 30, key="hsa_age")
            sleep_hrs = st.select_slider(
                "Sleep per Night (hours)",
                options=[str(i) for i in range(3, 13)],
                value="7", key="hsa_sleep",
            )
            exercise_days = st.select_slider(
                "Exercise Days / Week",
                options=["0","1","2","3","4","5","6","7"],
                value="3", key="hsa_exercise",
            )
        with col_b:
            water_glasses = st.select_slider(
                "Glasses of Water / Day",
                options=[str(i) for i in range(0, 16)],
                value="6", key="hsa_water",
            )
            stress_level = st.select_slider(
                "Stress Level",
                options=["Very Low", "Low", "Moderate", "High", "Very High"],
                value="Moderate", key="hsa_stress",
            )
            smoking = st.radio("Do you smoke?", ["No", "Occasionally", "Yes"], horizontal=True, key="hsa_smoke")

        assess = st.form_submit_button("Assess My Wellness", use_container_width=True)

    st.markdown("</div></div>", unsafe_allow_html=True)

    if assess:
        sleep_score   = min(int(sleep_hrs) / 8, 1.0)
        exercise_score = int(exercise_days) / 5
        water_score   = min(int(water_glasses) / 8, 1.0)
        stress_score  = {"Very Low": 1.0, "Low": 0.85, "Moderate": 0.65, "High": 0.4, "Very High": 0.2}[stress_level]
        smoke_score   = {"No": 1.0, "Occasionally": 0.6, "Yes": 0.2}[smoking]
        age_score     = max(0.4, 1 - (max(age - 30, 0) / 100))

        overall     = (sleep_score*0.25 + exercise_score*0.25 + water_score*0.15
                       + stress_score*0.2 + smoke_score*0.1 + age_score*0.05)
        overall_pct = round(overall * 100)

        ring_color = ("#4ade80" if overall_pct >= 80
                      else "#a78bfa" if overall_pct >= 60
                      else "#fbbf24" if overall_pct >= 40 else "#f87171")
        ring_label = ("Excellent" if overall_pct >= 80
                      else "Good" if overall_pct >= 60
                      else "Fair" if overall_pct >= 40 else "At Risk")

        pillars = [
            ("😴", "Sleep",    sleep_score,    "#a78bfa"),
            ("🏃", "Exercise", exercise_score, "#4ade80"),
            ("💧", "Hydration",water_score,    "#60a5fa"),
            ("🧘", "Stress",   stress_score,   "#fbbf24"),
            ("🚭", "Smoking",  smoke_score,    "#f87171"),
            ("📅", "Age",      age_score,      "#2dd4bf"),
        ]

        pillars_html = ""
        for icon, name, score, color in pillars:
            pct = round(score * 100)
            status = "Excellent" if pct >= 80 else "Good" if pct >= 60 else "Fair" if pct >= 40 else "Low"
            pillars_html += (
                f'<div class="hsa-pillar">'
                f'<div class="hsa-p-top">'
                f'<span class="hsa-p-name"><span class="hsa-code" style="color:{color}">{icon}</span>{name}</span>'
                f'<span class="hsa-p-pct" style="color:{color}">{pct}%</span>'
                f'</div>'
                f'<div class="hsa-bar-track"><div class="hsa-bar-fill" style="width:{pct}%;background:{color}"></div></div>'
                f'<div class="hsa-p-status" style="color:{color}">{status}</div>'
                f'</div>'
            )

        tips = []
        if sleep_score   < 0.8: tips.append("Improve sleep hygiene - aim for 7-8 hours consistently.")
        if exercise_score< 0.6: tips.append("Increase physical activity - even a 20-min daily walk helps.")
        if water_score   < 0.8: tips.append("Increase daily water intake to at least 8 glasses.")
        if stress_score  < 0.65:tips.append("Practice stress-reduction like meditation or journaling.")
        if smoke_score   < 1.0: tips.append("Consider reducing or quitting smoking - the #1 preventable risk factor.")
        if not tips:             tips.append("You're doing great! Keep up healthy habits and schedule regular checkups.")

        tips_html = "".join(f"<li>{t}</li>" for t in tips)

        st.markdown(
            f'<div class="hsa-result">'
            f'<div class="hsa-result-grid">'
            f'<div class="hsa-ring-wrap">'
            f'<div class="hsa-ring" style="color:{ring_color}">'
            f'<div class="hsa-ring-num">{overall_pct}</div>'
            f'<div class="hsa-ring-tag">{ring_label}</div>'
            f'</div>'
            f'<div class="hsa-ring-sub">Wellness score - based on your inputs</div>'
            f'</div>'
            f'<div class="hsa-pillars">{pillars_html}</div>'
            f'</div>'
            f'<div class="hsa-tip-box" style="border-left-color:{ring_color}">'
            f'<div class="hsa-tip-title">Personalized recommendations</div>'
            f'<ul class="hsa-tip-list">{tips_html}</ul>'
            f'</div>'
            f'<div class="hsa-disc">Educational estimate only - not a substitute for professional medical advice.</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ── REDESIGNED: SYMPTOM QUICK-CHECK (M3 Expressive) ──────────────────────────
def render_symptom_quick_check():
    import streamlit.components.v1 as components

    st.markdown("<div class='md-section-title'>🔎 Symptom Quick-Check</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='md-section-subtitle'>Tap any symptoms you are experiencing for a preliminary triage hint. Always consult a doctor for proper diagnosis.</div>",
        unsafe_allow_html=True,
    )

    components.html(
        """
<!DOCTYPE html>
<html>
<head>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{
  font-family:'Plus Jakarta Sans',system-ui,sans-serif;
  background:transparent;color:#e2e8f0;font-size:14px;
}
.shell{
  border:1px solid rgba(148,163,184,0.22);border-radius:28px;
  padding:24px;
  background:linear-gradient(135deg,rgba(0,106,106,0.12),rgba(103,80,164,0.08) 60%,rgba(0,106,106,0.04));
}
.two-col{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-bottom:16px}
@media(max-width:560px){
  .two-col{grid-template-columns:1fr}
  .shell{padding:14px 12px}
  .seg-btn{font-size:10px;padding:7px 4px}
  .cat-tab{font-size:10px;padding:5px 10px}
  .sym-tag{font-size:11px;padding:6px 11px}
  .go-btn{font-size:13px;padding:12px}
  .t-level{font-size:15px}
  .t-msg{font-size:11px}
  .cond-row{font-size:11px}
  .result{padding:14px}
  .triage-banner{padding:12px;gap:10px}
}
.field{margin-bottom:0}
.field label{display:block;font-size:11px;font-weight:700;color:rgba(148,163,184,0.9);
  letter-spacing:.06em;text-transform:uppercase;margin-bottom:8px}
.seg{display:flex;gap:5px;flex-wrap:wrap}
.seg-btn{
  flex:1;min-width:48px;padding:8px 6px;border-radius:999px;
  border:1px solid rgba(148,163,184,0.22);background:rgba(255,255,255,0.04);
  font-size:11px;font-weight:700;cursor:pointer;color:rgba(148,163,184,0.9);
  transition:all .16s ease;text-align:center;font-family:inherit;
}
.seg-btn:hover{border-color:#006a6a;color:#80cbc4}
.seg-btn.on{background:rgba(0,106,106,0.18);border-color:#006a6a;color:#80cbc4}

.cat-tabs{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:12px}
.cat-tab{
  padding:6px 14px;border-radius:999px;border:1px solid rgba(148,163,184,0.22);
  background:rgba(255,255,255,0.04);font-size:11px;font-weight:700;cursor:pointer;
  color:rgba(148,163,184,0.9);transition:all .15s ease;font-family:inherit;
}
.cat-tab:hover{border-color:#006a6a;color:#80cbc4}
.cat-tab.on{background:rgba(0,106,106,0.18);border-color:#006a6a;color:#80cbc4}

.sym-label{font-size:11px;font-weight:700;color:rgba(148,163,184,0.9);
  letter-spacing:.06em;text-transform:uppercase;margin-bottom:8px;display:block}
.tag-cloud{display:flex;flex-wrap:wrap;gap:7px;min-height:40px}
.sym-tag{
  padding:7px 14px;border-radius:999px;font-size:12px;font-weight:700;cursor:pointer;
  border:1px solid rgba(148,163,184,0.22);background:rgba(255,255,255,0.04);
  color:rgba(148,163,184,0.9);transition:all .14s ease;font-family:inherit;
}
.sym-tag:hover{border-color:#006a6a;color:#80cbc4}
.sym-tag.on{background:rgba(0,106,106,0.2);border-color:#006a6a;color:#80cbc4}

.go-btn{
  width:100%;margin-top:18px;padding:14px;border-radius:999px;
  background:linear-gradient(135deg,#006a6a,#00897b);border:none;
  color:#fff;font-size:14px;font-weight:800;cursor:pointer;
  transition:all .18s ease;font-family:inherit;letter-spacing:.02em;
}
.go-btn:hover{opacity:.88;transform:translateY(-1px)}
.go-btn:active{transform:scale(.98)}

.result{
  margin-top:18px;border:1px solid rgba(148,163,184,0.18);border-radius:22px;
  padding:20px;background:rgba(255,255,255,0.04);
  animation:rise .3s ease;
}
@keyframes rise{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}

.triage-banner{
  display:flex;align-items:flex-start;gap:13px;padding:14px 16px;border-radius:18px;
  margin-bottom:14px;
}
.t-icon{font-size:24px;flex-shrink:0;line-height:1.2}
.t-level{font-size:17px;font-weight:900;line-height:1.1;margin-bottom:4px}
.t-msg{font-size:12px;color:rgba(148,163,184,0.85);line-height:1.55}

.chips-row{display:flex;flex-wrap:wrap;gap:5px;margin-bottom:12px}
.chip{
  padding:4px 11px;border-radius:999px;font-size:11px;font-weight:700;
  background:rgba(0,106,106,0.16);border:1px solid rgba(0,106,106,0.35);color:#80cbc4;
}

.cond-row{
  display:flex;align-items:center;gap:8px;padding:8px 11px;border-radius:12px;
  background:rgba(255,255,255,0.05);border:1px solid rgba(148,163,184,0.14);
  margin-bottom:5px;font-size:12px;
}
.cond-dot{width:8px;height:8px;border-radius:50%;flex-shrink:0}
.cond-sym{font-weight:800}
.cond-arr{color:rgba(148,163,184,0.5);margin:0 3px}
.cond-val{color:rgba(148,163,184,0.8)}

.cond-label{font-size:11px;font-weight:700;color:rgba(148,163,184,0.7);
  letter-spacing:.06em;text-transform:uppercase;margin:10px 0 7px}

.disc{
  margin-top:10px;padding:9px 13px;border-radius:12px;
  background:rgba(217,119,6,0.08);border:1px solid rgba(217,119,6,0.25);
  font-size:11px;color:rgba(148,163,184,0.75);text-align:center;
}
.empty{
  padding:14px;border-radius:14px;border:1px dashed rgba(148,163,184,0.25);
  font-size:13px;color:rgba(148,163,184,0.6);text-align:center;margin-top:12px;
}
</style>
</head>
<body>
<div class="shell">
  <div class="two-col">
    <div class="field">
      <label>Severity</label>
      <div class="seg" id="sev-ctrl">
        <button class="seg-btn" onclick="setSeg('sev-ctrl',this,'Mild')">Mild</button>
        <button class="seg-btn on" onclick="setSeg('sev-ctrl',this,'Moderate')">Moderate</button>
        <button class="seg-btn" onclick="setSeg('sev-ctrl',this,'Severe')">Severe</button>
      </div>
    </div>
    <div class="field">
      <label>Duration</label>
      <div class="seg" id="dur-ctrl">
        <button class="seg-btn" onclick="setSeg('dur-ctrl',this,'<24h')">&lt;24h</button>
        <button class="seg-btn on" onclick="setSeg('dur-ctrl',this,'1-3d')">1–3d</button>
        <button class="seg-btn" onclick="setSeg('dur-ctrl',this,'4-7d')">4–7d</button>
        <button class="seg-btn" onclick="setSeg('dur-ctrl',this,'1-2w')">1–2w</button>
        <button class="seg-btn" onclick="setSeg('dur-ctrl',this,'>2w')">&gt;2w</button>
      </div>
    </div>
  </div>

  <div class="cat-tabs" id="cat-tabs">
    <button class="cat-tab on" onclick="setCategory(this,'All')">All</button>
    <button class="cat-tab" onclick="setCategory(this,'General')">General</button>
    <button class="cat-tab" onclick="setCategory(this,'Cardiovascular')">Cardiovascular</button>
    <button class="cat-tab" onclick="setCategory(this,'Respiratory')">Respiratory</button>
    <button class="cat-tab" onclick="setCategory(this,'Neurological')">Neurological</button>
    <button class="cat-tab" onclick="setCategory(this,'Digestive')">Digestive</button>
    <button class="cat-tab" onclick="setCategory(this,'Musculoskeletal')">Musculoskeletal</button>
  </div>

  <span class="sym-label">Select symptoms</span>
  <div class="tag-cloud" id="sym-tags"></div>

  <button class="go-btn" onclick="runSymptom()">🩺 &nbsp;Get Triage Hint</button>
  <div id="sym-result"></div>
</div>

<script>
var state={sev:'Moderate',dur:'1-3d'};
function setSeg(id,el,val){
  document.getElementById(id).querySelectorAll('.seg-btn').forEach(b=>b.classList.remove('on'));
  el.classList.add('on');
  if(id==='sev-ctrl')state.sev=val;else state.dur=val;
}

var groups={
  General:['Fatigue','Fever','Chills','Night Sweats','Weight Loss'],
  Cardiovascular:['Chest Pain','Palpitations','Shortness of Breath','Ankle Swelling'],
  Respiratory:['Persistent Cough','Wheezing','Coughing Blood','Difficulty Breathing'],
  Neurological:['Headache','Dizziness','Memory Loss','Numbness / Tingling'],
  Digestive:['Nausea','Vomiting','Abdominal Pain','Bloating','Blood in Stool'],
  Musculoskeletal:['Joint Pain','Muscle Weakness','Back Pain','Swollen Joints']
};
var allSyms=[].concat(...Object.values(groups));
var picked=new Set();
var activeCat='All';

function renderTags(){
  var syms=activeCat==='All'?allSyms:groups[activeCat];
  var c=document.getElementById('sym-tags');
  c.innerHTML='';
  syms.forEach(s=>{
    var b=document.createElement('button');
    b.className='sym-tag'+(picked.has(s)?' on':'');
    b.textContent=s;
    b.onclick=()=>{
      if(picked.has(s)){picked.delete(s);b.classList.remove('on')}
      else{picked.add(s);b.classList.add('on')}
    };
    c.appendChild(b);
  });
}
renderTags();

function setCategory(el,cat){
  document.querySelectorAll('.cat-tab').forEach(t=>t.classList.remove('on'));
  el.classList.add('on');
  activeCat=cat;
  renderTags();
}

var condMap={
  'Chest Pain':'Angina / Myocardial Infarction',
  'Palpitations':'Arrhythmia / Anxiety',
  'Shortness of Breath':'Asthma / Heart Failure / Pneumonia',
  'Persistent Cough':'Bronchitis / COPD / Pneumonia',
  'Fatigue':'Anaemia / Hypothyroidism / Depression',
  'Fever':'Infection (viral or bacterial)',
  'Headache':'Migraine / Hypertension / Tension',
  'Joint Pain':'Arthritis / Gout / Lupus',
  'Nausea':'Gastritis / Food Poisoning',
  'Dizziness':'Inner-ear issue / Hypotension / Anaemia',
  'Weight Loss':'Diabetes / Cancer screening advised',
  'Night Sweats':'Hormonal / TB / Lymphoma screening',
  'Ankle Swelling':'Heart Failure / DVT / Renal Disease',
  'Abdominal Pain':'IBS / Appendicitis / Gastritis',
  'Memory Loss':'Dementia screening advised',
  'Coughing Blood':'Pulmonary embolism / TB — seek urgent care',
  'Blood in Stool':'GI bleed / Colon conditions — seek care'
};

function runSymptom(){
  var syms=Array.from(picked);
  if(!syms.length){
    document.getElementById('sym-result').innerHTML='<div class="empty">Please select at least one symptom above.</div>';
    return;
  }
  var emergency=new Set(['Chest Pain','Coughing Blood','Blood in Stool','Difficulty Breathing']);
  var urgent=new Set(['Fever','Shortness of Breath','Palpitations','Dizziness','Numbness / Tingling']);
  var chronic=new Set(['Weight Loss','Night Sweats','Memory Loss','Joint Pain','Persistent Cough']);
  var ss=new Set(syms);
  var isEmerg=[...ss].some(s=>emergency.has(s))||(state.sev==='Severe'&&state.dur==='<24h');
  var isUrge=[...ss].some(s=>urgent.has(s))||state.sev==='Severe';
  var isChronic=[...ss].some(s=>chronic.has(s))||state.dur==='>2w'||state.dur==='1-2w';
  var color,icon,level,msg;
  if(isEmerg){color='#f87171';icon='🚨';level='Seek emergency care';msg='One or more symptoms may require immediate attention. Go to your nearest emergency department or call emergency services.'}
  else if(isUrge){color='#fbbf24';icon='⚡';level='See a doctor soon';msg='Your symptoms suggest you should consult a doctor within 24–48 hours. Consider booking an urgent appointment.'}
  else if(isChronic){color='#a78bfa';icon='🔬';level='Schedule a checkup';msg='Some symptoms suggest a possible chronic or underlying condition. Schedule a routine consultation for further investigation.'}
  else{color='#4ade80';icon='✓';level='Monitor & self-care';msg='Symptoms appear mild and short-duration. Rest and hydration may help. If symptoms worsen or persist, consult a doctor.'}

  var chips=syms.map(s=>`<span class="chip">${s}</span>`).join('');
  var conds=syms.filter(s=>condMap[s]).map(s=>`
    <div class="cond-row">
      <div class="cond-dot" style="background:${color}"></div>
      <span class="cond-sym">${s}</span><span class="cond-arr">→</span>
      <span class="cond-val">${condMap[s]}</span>
    </div>`).join('');

  document.getElementById('sym-result').innerHTML=`
  <div class="result">
    <div class="triage-banner" style="background:${color}18;border:1px solid ${color}40">
      <div class="t-icon">${icon}</div>
      <div>
        <div class="t-level" style="color:${color}">${level}</div>
        <div class="t-msg">${msg}</div>
      </div>
    </div>
    <div style="font-size:11px;font-weight:700;color:rgba(148,163,184,0.7);letter-spacing:.06em;text-transform:uppercase;margin-bottom:7px">Selected symptoms</div>
    <div class="chips-row">${chips}</div>
    ${conds?`<div class="cond-label">Possible related conditions (informational)</div>${conds}`:''}
    <div class="disc">⚠️ Triage hint is for educational purposes only. Use our AI Disease Prediction tool or consult a healthcare professional for accurate diagnosis.</div>
  </div>`;
  sendHeight();
}
// Auto-resize: tell the parent Streamlit iframe to grow with content
function sendHeight(){
  var h=document.documentElement.scrollHeight||document.body.scrollHeight;
  window.parent.postMessage({type:'streamlit:setFrameHeight',height:h+24},'*');
}
// Send height on load and after any DOM change
sendHeight();
new MutationObserver(sendHeight).observe(document.body,{childList:true,subtree:true,attributes:true});
window.addEventListener('resize',sendHeight);
</script>
</body>
</html>
""",
        height=750,
        scrolling=True,
    )


# ── NEW: VITAL SIGNS REFERENCE ────────────────────────────────────────────────
def render_vital_signs_reference():
    st.markdown("<div class='md-section-title'>📊 Vital Signs Reference</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='md-section-subtitle'>Normal ranges for key health indicators in healthy adults. Understanding these helps you interpret medical readings.</div>",
        unsafe_allow_html=True,
    )

    vitals = [
        ("#e11d48", "❤️", "60–100",    "beats / min",     "Resting Heart Rate",  "Normal Range"),
        ("#2563eb", "🩺", "120/80",    "mmHg",            "Blood Pressure",       "Optimal"),
        ("#059669", "🌡️", "36.5–37.5", "°C",              "Body Temperature",    "Normal"),
        ("#6750a4", "🫁", "12–20",     "breaths / min",   "Respiratory Rate",    "Normal Range"),
        ("#d97706", "🩸", "70–99",     "mg/dL (fasting)", "Blood Glucose",        "Fasting Normal"),
        ("#0f766e", "💨", "95–100",    "%",               "Oxygen Saturation",   "SpO₂ Normal"),
        ("#7c3aed", "⚖️", "18.5–24.9", "kg/m²",           "BMI",                 "Healthy Weight"),
        ("#be185d", "🧪", "&lt; 200",  "mg/dL",           "Total Cholesterol",   "Desirable"),
    ]

    row1 = st.columns(4)
    row2 = st.columns(4)

    for i, (color, icon, value, unit, label, status) in enumerate(vitals):
        col = row1[i] if i < 4 else row2[i - 4]
        with col:
            st.markdown(
                f"""<div class="md-vital-card">
  <div style="position:absolute;top:0;left:0;right:0;height:3px;background:{color};border-radius:3px 3px 0 0;"></div>
  <div class="md-vital-icon">{icon}</div>
  <div class="md-vital-value" style="color:{color};">{value}</div>
  <div class="md-vital-unit">{unit}</div>
  <div class="md-vital-label">{label}</div>
  <div class="md-vital-status" style="background:rgba(5,150,105,0.13);color:#059669;border:1px solid rgba(5,150,105,0.28);">{status}</div>
</div>""",
                unsafe_allow_html=True,
            )

    st.markdown(
        "<div style='text-align:center;font-size:12px;color:var(--md-soft);margin-top:0.5rem;margin-bottom:0.5rem;'>"
        "Values represent typical adult ranges. Individual results may vary — always consult a healthcare professional for interpretation."
        "</div>",
        unsafe_allow_html=True,
    )


# ── NEW: HEALTH JOURNEY TIMELINE ─────────────────────────────────────────────
def render_health_journey_timeline():
    st.markdown("<div class='md-section-title'>🗓️ Recommended Health Screening Timeline</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='md-section-subtitle'>Key health screenings and checkups recommended at different life stages for proactive wellness management.</div>",
        unsafe_allow_html=True,
    )

    col_left, col_right = st.columns(2)

    timeline_left = [
        ("In Your 20s", "Every 1–2 years", "General", "Full physical exam, blood pressure, cholesterol baseline, vision & dental checkups, STI screening as needed."),
        ("In Your 30s", "Annually", "Metabolic", "Fasting glucose, lipid panel, thyroid function. Women: Pap smear every 3 years, HPV co-test every 5 years."),
        ("In Your 40s", "Annually", "Cardiac", "EKG baseline, diabetes screening, mammogram (women every 1–2 years), colorectal cancer screening begins at 45."),
    ]

    timeline_right = [
        ("In Your 50s", "Annually", "Oncology", "Colonoscopy every 10 years, lung cancer CT for smokers, prostate screening (men), bone density scan (women at menopause)."),
        ("In Your 60s", "Every 6 months", "Comprehensive", "Cardiovascular risk assessment, abdominal aortic aneurysm scan (men who smoked), hearing & vision, vaccine boosters."),
        ("70 & Beyond", "Every 6 months", "Geriatric", "Fall risk assessment, cognitive function screening, osteoporosis management, medication review, caregiver support plans."),
    ]

    def render_timeline_col(items):
        html = "<div class='md-timeline'>"
        for title, freq, tag, desc in items:
            html += f"""
<div class="md-timeline-item">
    <div class="md-timeline-dot"></div>
    <div class="md-timeline-card">
        <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:8px;flex-wrap:wrap;">
            <div class="md-timeline-title">{title}</div>
            <div class="md-timeline-tag">{tag}</div>
        </div>
        <div class="md-timeline-meta">🔁 {freq}</div>
        <div style="font-size:13px;color:var(--md-soft);line-height:1.5;">{desc}</div>
    </div>
</div>
"""
        html += "</div>"
        return html

    with col_left:
        st.markdown(render_timeline_col(timeline_left), unsafe_allow_html=True)

    with col_right:
        st.markdown(render_timeline_col(timeline_right), unsafe_allow_html=True)


def render_technology_cards():
    st.markdown("<div class='md-section-title'>⚙️ Technologies Used</div>", unsafe_allow_html=True)

    st.markdown(
        """
<div class="md-card-grid">
    <div class="md-card">
        <div class="md-card-icon">🤖</div>
        <h3>Machine Learning</h3>
        <p>Uses models such as RandomForest, XGBoost, LightGBM, and deep learning workflows.</p>
    </div>
    <div class="md-card">
        <div class="md-card-icon">🗂️</div>
        <h3>NLP and AI</h3>
        <p>Supports chatbot interactions and health information retrieval using language models.</p>
    </div>
    <div class="md-card">
        <div class="md-card-icon">☁️</div>
        <h3>Cloud Ready</h3>
        <p>Built with Streamlit and designed for deployment on modern cloud platforms.</p>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_why_section():
    st.markdown(
        """
<div class="md-why-box">
    <h2 style="text-align:center;margin-top:0;">🔍 Why Use This App?</h2>
    <div class="md-why-list">
        <div class="md-why-item">
            <strong>Accurate Predictions</strong>
            <span>AI models trained on healthcare datasets for useful risk insights.</span>
        </div>
        <div class="md-why-item">
            <strong>Real-Time Assistance</strong>
            <span>Quick information, guidance, and recommendations when users need them.</span>
        </div>
        <div class="md-why-item">
            <strong>User Friendly</strong>
            <span>Simple flows designed for students, professionals, and everyday users.</span>
        </div>
        <div class="md-why-item">
            <strong>Secure Design</strong>
            <span>Privacy-aware layout and careful handling of health-related inputs.</span>
        </div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_contact():
    st.markdown(
        '<div class="md-contact-wrap">'
        '<div class="md-contact-avatar">👨‍💻</div>'
        '<div>'
        '<div class="md-contact-title">Get in Touch</div>'
        '<div class="md-contact-sub" style="margin-top:8px;">Have questions, feedback, or want to collaborate? '
        'I&#39;d love to hear from you. Reach out anytime.</div>'
        '</div>'
        '<a class="md-contact-email" href="mailto:opportunities.yatin@gmail.com">'
        '📧 &nbsp;opportunities.yatin@gmail.com</a>'
        '<div class="md-contact-socials">'
        '<a class="md-contact-social-btn" href="https://github.com/YatinSharma1303" target="_blank">🐙 &nbsp;GitHub</a>'
        '<a class="md-contact-social-btn" href="https://www.linkedin.com/in/yatin-sharma-793042372/" target="_blank">💼 &nbsp;LinkedIn</a>'
        '<a class="md-contact-social-btn" href="https://x.com/Yatin__Sharma" target="_blank">🐦 &nbsp;Twitter</a>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )


def render_footer():
    footer_logo_html = '<div class="md-footer-logo-icon" role="img" aria-label="SmartHealthCare logo">🏥</div>'
    from datetime import datetime as _dt
    _year = _dt.now().strftime("%Y")
    st.markdown(
        f'<div class="md-footer">'
        f'<div class="md-footer-top">'
        f'<div class="md-footer-brand">' + footer_logo_html +
        f'<div><div class="md-footer-brand-name">SmartHealthCare<br/>AI Diagnosis</div>'
        f'<div class="md-footer-brand-sub">by {escape(CREATOR_NAME)}</div></div></div>'
        f'</div>'
        f'<div class="md-footer-links">'
        f'<a class="md-footer-link" href="https://github.com/YatinSharma1303/" target="_blank">🐙 GitHub</a>'
        f'<a class="md-footer-link" href="https://www.linkedin.com/in/yatin-sharma-793042372/" target="_blank">💼 LinkedIn</a>'
        f'<a class="md-footer-link" href="https://x.com/Yatin__Sharma" target="_blank">🐦 Twitter</a>'
        f'</div>'
        f'<div class="md-footer-meta">'
        f'<span class="md-footer-version">2.0 Material Expressive</span>'
        f' &nbsp;·&nbsp; Made with <span class="md-footer-heart">❤️</span> by <strong>{escape(CREATOR_NAME)}</strong>'
        f' &nbsp;·&nbsp; © {_year}'
        f'</div>'
        f'<div class="md-footer-disclaimer">⚕️ <strong>Medical Disclaimer:</strong> '
        f'This application is for educational purposes only and does not constitute medical advice, diagnosis, or treatment. '
        f'Always consult a qualified healthcare professional regarding any health concerns.</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ── MAIN ───────────────────────────────────────────────────────────────────────
init_theme()
st.markdown(get_theme_styles(), unsafe_allow_html=True)
render_theme_toggle()

render_styles()

# Fix sidebar collapse button showing text instead of icon
st.markdown(
    """
<script>
(function fixSidebarBtn() {
    const COLLAPSE_SVG = '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="11 17 6 12 11 7"/><polyline points="18 17 13 12 18 7"/></svg>';
    const EXPAND_SVG  = '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="13 17 18 12 13 7"/><polyline points="6 17 11 12 6 7"/></svg>';
    function getSVG(label) {
        return (label.toLowerCase().includes('close') || label.toLowerCase().includes('collapse'))
            ? COLLAPSE_SVG : EXPAND_SVG;
    }
    function fix() {
        const ariaLabels = [
            'Close sidebar', 'Open sidebar',
            'collapse sidebar', 'expand sidebar',
            'Collapse sidebar', 'Expand sidebar',
        ];
        ariaLabels.forEach(label => {
            const btn = document.querySelector('button[aria-label="' + label + '"]');
            if (!btn) return;
            const hasSVG = btn.querySelector('svg');
            const hasText = [...btn.childNodes].some(n => n.nodeType === Node.TEXT_NODE && n.textContent.trim().length > 0);
            if (hasText) {
                [...btn.childNodes].forEach(node => {
                    if (node.nodeType === Node.TEXT_NODE) node.textContent = '';
                });
            }
            if (!hasSVG) {
                btn.insertAdjacentHTML('beforeend', getSVG(label));
            }
        });
    }
    fix();
    const observer = new MutationObserver(fix);
    observer.observe(document.body, { childList: true, subtree: true, characterData: false });
})();
</script>
""",
    unsafe_allow_html=True,
)

render_sidebar()
render_hero()
render_stats()
render_feature_cards()

# ── NEW SECTIONS ───────────────────────────────────────────────────────────────
render_health_tips_marquee()
render_risk_awareness()
render_vital_signs_reference()
render_health_self_assessment()
render_symptom_quick_check()
render_health_journey_timeline()

# ── EXISTING SECTIONS ──────────────────────────────────────────────────────────
render_technology_cards()

st.markdown("---")
render_why_section()

st.markdown("---")
render_contact()


# ───────────────────────── Floating Quick Widget ───────────────────────────

def render_floating_widget():
    components.html(
        """
<script>
(function() {
    const parentDoc = window.parent.document;
    const parentWin = window.parent;
    const widgetId = "mm-widget-root-smart-healthcare";
    const styleId = widgetId + "-style";
    const isLightTheme = __IS_LIGHT_THEME__;
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

        /* Light theme for the opened floating widget panel */
        #${widgetId}.mm-theme-light .mm-widget-panel {
            border-color: rgba(91,63,196,.18);
            background:
                radial-gradient(circle at 12% 0%, rgba(91,63,196,.12), transparent 38%),
                radial-gradient(circle at 92% 18%, rgba(0,180,166,.10), transparent 42%),
                #fffbff;
            box-shadow: 0 18px 54px rgba(91,63,196,.18), 0 0 0 1px rgba(91,63,196,.06);
            backdrop-filter: none;
        }
        #${widgetId}.mm-theme-light .mm-widget-panel::-webkit-scrollbar-thumb {
            background: rgba(91,63,196,.26);
        }
        #${widgetId}.mm-theme-light .mm-widget-head {
            background:
                linear-gradient(135deg,
                    rgba(91,63,196,.12) 0%,
                    rgba(0,180,166,.08) 55%,
                    rgba(255,251,255,.98) 100%);
            border-bottom-color: rgba(91,63,196,.16);
            box-shadow:
                0 4px 20px rgba(91,63,196,.08),
                inset 0 1px 0 rgba(255,255,255,.82);
            backdrop-filter: none;
        }
        #${widgetId}.mm-theme-light .mm-widget-head::before {
            background: radial-gradient(ellipse at 18% 0%, rgba(255,255,255,.78), transparent 62%);
        }
        #${widgetId}.mm-theme-light .mm-widget-head::after {
            background: linear-gradient(90deg, transparent, rgba(91,63,196,.40), rgba(0,180,166,.32), transparent);
        }
        #${widgetId}.mm-theme-light .mm-widget-head-icon {
            background: linear-gradient(135deg, rgba(91,63,196,.14), rgba(0,180,166,.12));
            border-color: rgba(91,63,196,.18);
            color: #4a33a3;
            box-shadow: 0 8px 18px rgba(91,63,196,.14);
        }
        #${widgetId}.mm-theme-light .mm-widget-title {
            color: #1a1530;
        }
        #${widgetId}.mm-theme-light .mm-widget-sub {
            color: #5a5270;
        }
        #${widgetId}.mm-theme-light .mm-widget-draghint {
            background: rgba(91,63,196,.08);
            border-color: rgba(91,63,196,.22);
            color: #5b3fc4;
        }
        #${widgetId}.mm-theme-light .mm-widget-draghint-dot {
            background: #006a6a;
            box-shadow: 0 0 0 3px rgba(0,106,106,.10);
        }
        #${widgetId}.mm-theme-light .mm-widget-head-divider {
            background: linear-gradient(90deg, transparent, rgba(91,63,196,.24), rgba(0,106,106,.18), transparent);
        }
        #${widgetId}.mm-theme-light .mm-tool {
            border-color: rgba(91,63,196,.15);
            background:
                linear-gradient(135deg, rgba(91,63,196,.055), rgba(0,180,166,.035)),
                #ffffff;
            color: #1a1530;
            box-shadow: 0 7px 18px rgba(91,63,196,.08);
        }
        #${widgetId}.mm-theme-light .mm-tool:hover {
            background:
                linear-gradient(135deg, rgba(91,63,196,.11), rgba(0,180,166,.07)),
                #ffffff;
            border-color: rgba(91,63,196,.32);
            box-shadow: 0 10px 24px rgba(91,63,196,.13);
        }
        #${widgetId}.mm-theme-light .mm-tool-code {
            background: linear-gradient(135deg,#6750a4,#006a6a);
            color: #ffffff;
            box-shadow: 0 8px 18px rgba(91,63,196,.20);
        }
        #${widgetId}.mm-theme-light .mm-tool-label {
            color: #1a1530;
        }
        #${widgetId}.mm-theme-light .mm-tool-note {
            color: #5a5270;
        }
        #${widgetId}.mm-theme-light .mm-toast {
            background: #fffbff;
            color: #5a5270;
            border-color: rgba(91,63,196,.16);
            box-shadow: 0 8px 24px rgba(91,63,196,.14);
        }
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
    `;
    parentDoc.head.appendChild(style);

    const root = parentDoc.createElement("div");
    root.id = widgetId;
    root.classList.add(isLightTheme ? "mm-theme-light" : "mm-theme-dark");
    const tools = [
        ['top', 'UP', 'Back to Top', 'Scroll to top', 'top'],
        ['Daily Health Tips', 'TP', 'Health Tips', 'Daily prevention cards', 'scroll'],
        ['Disease Risk Awareness', 'RK', 'Risk Awareness', 'Warning patterns', 'scroll'],
        ['Vital Signs Reference', 'VS', 'Vital Signs', 'Normal ranges', 'scroll'],
        ['Quick Health Self-Assessment', 'SA', 'Self Assessment', 'Wellness checker', 'scroll'],
        ['Symptom Quick-Check', 'QC', 'Symptom Check', 'Quick local check', 'scroll'],
        ['Recommended Health Screening Timeline', 'JR', 'Journey', 'Health timeline', 'scroll'],
        ['Technologies Used', 'AI', 'Technology', 'How it works', 'scroll'],
        ['Get in Touch', 'CT', 'Contact', 'Creator links', 'scroll']
    ];
    root.innerHTML = `
        <div class="mm-widget-panel" role="menu" aria-label="Smart Healthcare quick actions">
            <div class="mm-widget-head">
                <div class="mm-widget-head-top">
                    <div style="display:flex;align-items:center;gap:9px;">
                        <div class="mm-widget-head-icon">🏥</div>
                        <div style="display:flex;flex-direction:column;gap:1px;">
                            <div class="mm-widget-title">Smart Healthcare tools</div>
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
                ${tools.map(([target, code, label, note, kind]) => `
                    <button class="mm-tool" data-target="${target}" data-kind="${kind}" type="button">
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
            target.dispatchEvent(new MouseEvent(type, {bubbles: true, cancelable: true, view: parentWin}));
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
    function focusInput() {
        const target = parentDoc.querySelector('[data-testid="stChatInput"] textarea, [data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea, [data-testid="stSelectbox"] input');
        if (target) {
            target.scrollIntoView({behavior: "smooth", block: "center"});
            setTimeout(() => { target.focus(); target.click(); }, 350);
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
    function runAction(kind, target) {
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
            return true;
        }
        if (kind === "input") return focusInput();
        if (kind === "tab") return clickByText('[role="tab"], [data-baseweb="tab"], button', target);
        if (kind === "nav") return clickByText('[data-testid="stButton"] button, button', target) || clickByText('[data-testid="stRadio"] label, [role="radio"]', target);
        if (kind === "scroll") return scrollToText(target);
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
            togglePanel(false);
            const ok = runAction(kind, target);
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
})();
</script>
        """.replace("__IS_LIGHT_THEME__", "true" if st.session_state.get("theme") == "light" else "false"),
        height=0,
        width=0,
    )

render_footer()
render_floating_widget()