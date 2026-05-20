import streamlit as st

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
DARK = "dark"
LIGHT = "light"

# Shared font imports
_FONT_DARK = "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap"
_FONT_LIGHT = "https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;0,9..40,800;1,9..40,300;1,9..40,400&family=DM+Serif+Display:ital@0;1&display=swap"

# Noise SVG (used only in light theme)
_NOISE_SVG = (
    "data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E"
    "%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' "
    "numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E"
    "%3Crect width='100%25' height='100%25' filter='url(%23noise)'/%3E%3C/svg%3E"
)

_CSS_DARK = f"""
<style>
    @import url('{_FONT_DARK}');

    /* ── Base ── */
    .stApp {{
        background: linear-gradient(135deg, #0a0e27 0%, #1a1d3a 100%);
        font-family: 'Inter', sans-serif;
    }}

    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #1a1d3a 0%, #0a0e27 100%);
    }}

    /* ── Typography ── */
    h1, h2, h3, h4 {{ color: #e0e0e0; }}
    p, li, span, label {{ color: #cbd5e0; }}
    a {{ color: #667eea; text-decoration: none; transition: color 0.2s ease; }}
    a:hover {{ color: #764ba2; }}

    /* ── Hero text helpers ── */
    .main-title {{
        font-size: clamp(2rem, 5vw, 3rem);
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0.5rem;
    }}

    .subtitle {{
        font-size: clamp(1rem, 2.5vw, 1.3rem);
        color: #a0aec0;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 300;
    }}

    /* ── Feature card ── */
    .feature-card {{
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        will-change: transform;
    }}

    .feature-card:hover {{
        transform: translateY(-5px);
        border-color: rgba(102, 126, 234, 0.5);
        box-shadow: 0 12px 40px rgba(102, 126, 234, 0.2);
    }}

    .feature-card h3 {{ color: #667eea; font-size: 1.5rem; margin-bottom: 1rem; }}
    .feature-card p  {{ color: #cbd5e0; line-height: 1.6; }}

    /* ── Buttons ── */
    .stButton > button {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border: none !important;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        will-change: transform;
    }}

    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }}

    /* ── Inputs ── */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {{
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px !important;
        color: #e0e0e0 !important;
    }}

    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {{
        border-color: rgba(102, 126, 234, 0.6) !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.15) !important;
    }}

    [data-baseweb="select"] > div {{
        background: rgba(255, 255, 255, 0.05) !important;
        border-color: rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px !important;
        color: #e0e0e0 !important;
    }}

    /* ── Scrollbar ── */
    ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
    ::-webkit-scrollbar-track {{ background: transparent; }}
    ::-webkit-scrollbar-thumb {{
        background: rgba(102, 126, 234, 0.3);
        border-radius: 99px;
    }}
    ::-webkit-scrollbar-thumb:hover {{ background: rgba(102, 126, 234, 0.55); }}

    /* ── Selection ── */
    ::selection {{ background: rgba(102, 126, 234, 0.3); color: #e0e0e0; }}
</style>
"""

_CSS_LIGHT = f"""
<style>
    @import url('{_FONT_LIGHT}');

    /* ── Root tokens ── */
    :root {{
        --lt-bg:            #f0eef9;
        --lt-bg2:           #e8e4f5;
        --lt-surface:       rgba(255, 255, 255, 0.82);
        --lt-surface-high:  rgba(255, 255, 255, 0.96);
        --lt-border:        rgba(103, 80, 164, 0.18);
        --lt-border-strong: rgba(103, 80, 164, 0.36);
        --lt-primary:       #5b3fc4;
        --lt-primary-2:     #7c5cdb;
        --lt-accent:        #00b4a6;
        --lt-accent-warm:   #f97316;
        --lt-text:          #18132e;
        --lt-text-2:        #4a3f6b;
        --lt-text-muted:    #7b6fa0;
        --lt-shadow-sm:     0 2px 8px rgba(91, 63, 196, 0.08);
        --lt-shadow-md:     0 8px 28px rgba(91, 63, 196, 0.13);
        --lt-shadow-lg:     0 20px 56px rgba(91, 63, 196, 0.17);
    }}

    /* ── Base app ── */
    .stApp {{
        background:
            radial-gradient(ellipse 80% 50% at 15% -10%, rgba(124, 92, 219, 0.18) 0%, transparent 55%),
            radial-gradient(ellipse 60% 40% at 90% 10%,  rgba(0, 180, 166, 0.12) 0%, transparent 50%),
            radial-gradient(ellipse 50% 60% at 50% 110%, rgba(249, 115, 22, 0.08) 0%, transparent 55%),
            linear-gradient(160deg, #f0eef9 0%, #e9e5f6 40%, #eaf5f4 100%);
        font-family: 'DM Sans', sans-serif;
        color: var(--lt-text);
        min-height: 100vh;
    }}

    /* ── Noise texture overlay ── */
    .stApp::before {{
        content: '';
        position: fixed;
        inset: 0;
        pointer-events: none;
        z-index: 0;
        opacity: 0.025;
        background-image: url("{_NOISE_SVG}");
        background-size: 200px 200px;
    }}

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {{
        background:
            radial-gradient(ellipse 100% 40% at 50% 0%, rgba(124, 92, 219, 0.14) 0%, transparent 60%),
            linear-gradient(180deg, #f7f5fe 0%, #ede9fb 100%) !important;
        border-right: 1px solid var(--lt-border);
        box-shadow: 4px 0 24px rgba(91, 63, 196, 0.06);
    }}

    /* Mobile sidebar — force opaque light background */
    @media (max-width: 768px) {{
        [data-testid="stSidebar"] {{
            background: #ede8ff !important;
            border-right: 1.5px solid rgba(103, 80, 164, 0.28) !important;
        }}
        [data-testid="stSidebar"] > div {{
            background: #ede8ff !important;
        }}
    }}

    [data-testid="stSidebar"] * {{ color: var(--lt-text) !important; }}

    /* ── Typography ── */
    h1, h2, h3, h4 {{
        color: var(--lt-text) !important;
        font-family: 'DM Serif Display', serif !important;
    }}
    p, li, span, label {{ color: var(--lt-text) !important; }}
    a {{
        color: var(--lt-primary) !important;
        text-decoration: none;
        transition: color 0.2s ease, opacity 0.2s ease;
    }}
    a:hover {{ color: var(--lt-primary-2) !important; opacity: 0.85; }}

    /* ── Sidebar components ── */
    .md-sidebar-hero {{
        background: var(--lt-surface-high) !important;
        border-color: var(--lt-border) !important;
        box-shadow: var(--lt-shadow-md) !important;
    }}
    .md-sidebar-kicker {{
        background: rgba(91, 63, 196, 0.10) !important;
        border-color: rgba(91, 63, 196, 0.28) !important;
        color: var(--lt-primary) !important;
        font-family: 'DM Sans', sans-serif;
        font-weight: 700 !important;
    }}
    .md-sidebar-title {{
        color: var(--lt-text) !important;
        font-family: 'DM Serif Display', serif !important;
        font-size: 20px !important;
    }}
    .md-sidebar-text  {{ color: var(--lt-text-2) !important; }}
    .md-sidebar-section {{ color: var(--lt-text-muted) !important; }}
    .md-sidebar-link {{
        background: var(--lt-surface) !important;
        border-color: var(--lt-border) !important;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
    }}
    .md-sidebar-link:hover {{
        background: rgba(91, 63, 196, 0.08) !important;
        border-color: var(--lt-border-strong) !important;
    }}
    .md-sidebar-link-title {{ color: var(--lt-text) !important; font-weight: 700 !important; }}
    .md-sidebar-link-sub  {{ color: var(--lt-text-muted) !important; }}
    .md-sidebar-note {{
        background: rgba(249, 115, 22, 0.07) !important;
        border-color: rgba(249, 115, 22, 0.28) !important;
        color: #7c3a00 !important;
    }}
    .md-sidebar-footer {{ color: var(--lt-text-muted) !important; }}

    /* ── Hero ── */
    .md-hero {{
        background:
            radial-gradient(circle at 8% 20%,  rgba(91,  63, 196, 0.14) 0%, transparent 40%),
            radial-gradient(circle at 92% 10%,  rgba(0, 180, 166, 0.12) 0%, transparent 38%),
            radial-gradient(circle at 70% 95%,  rgba(249,115,  22, 0.09) 0%, transparent 35%),
            var(--lt-surface-high) !important;
        border-color: var(--lt-border) !important;
        box-shadow: var(--lt-shadow-lg) !important;
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
    }}
    .md-kicker {{
        background: rgba(91, 63, 196, 0.10) !important;
        border-color: rgba(91, 63, 196, 0.30) !important;
        color: var(--lt-primary) !important;
        font-weight: 700 !important;
    }}
    .md-title {{
        color: var(--lt-text) !important;
        font-family: 'DM Serif Display', serif !important;
        letter-spacing: -0.02em !important;
    }}
    .md-subtitle {{ color: var(--lt-text-2) !important; }}
    .md-chip {{
        background: rgba(255, 255, 255, 0.88) !important;
        border-color: var(--lt-border) !important;
        color: var(--lt-primary) !important;
        font-weight: 600 !important;
        box-shadow: var(--lt-shadow-sm) !important;
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
    }}
    .md-made-with {{ color: var(--lt-text-2) !important; }}

    /* ── Section ── */
    .md-section-title {{
        color: var(--lt-text) !important;
        font-family: 'DM Serif Display', serif !important;
        letter-spacing: -0.01em !important;
    }}
    .md-section-subtitle {{ color: var(--lt-text-2) !important; }}

    /* ── Cards ── */
    .md-card {{
        background: var(--lt-surface) !important;
        border-color: var(--lt-border) !important;
        box-shadow: var(--lt-shadow-sm) !important;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        transition: transform 160ms ease, box-shadow 200ms ease, border-color 160ms ease !important;
        will-change: transform;
    }}
    .md-card:hover {{
        border-color: var(--lt-border-strong) !important;
        box-shadow: var(--lt-shadow-md) !important;
        transform: translateY(-4px) !important;
    }}
    .md-card h3 {{
        color: var(--lt-primary) !important;
        font-family: 'DM Serif Display', serif !important;
        font-size: 17px !important;
    }}
    .md-card p {{ color: var(--lt-text-2) !important; }}
    .md-card-icon {{
        background: linear-gradient(135deg, rgba(91, 63, 196, 0.15), rgba(0, 180, 166, 0.12)) !important;
        border-color: rgba(91, 63, 196, 0.14) !important;
    }}

    /* ── BMI calculator ── */
    .md-bmi-shell {{
        background:
            radial-gradient(circle at 8% 20%,  rgba(91, 63, 196, 0.10) 0%, transparent 32%),
            radial-gradient(circle at 92% 20%,  rgba(0, 180, 166, 0.09) 0%, transparent 32%),
            var(--lt-surface) !important;
        border-color: var(--lt-border) !important;
        box-shadow: var(--lt-shadow-lg) !important;
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
    }}
    .md-bmi-result {{
        background: var(--lt-surface-high) !important;
        border-color: var(--lt-border) !important;
    }}
    .md-bmi-label          {{ color: var(--lt-text-muted) !important; }}
    .md-bmi-progress-wrap  {{ background: rgba(91, 63, 196, 0.10) !important; }}
    .md-bmi-insight {{
        background: rgba(91, 63, 196, 0.07) !important;
        border-left-color: var(--lt-primary) !important;
        color: var(--lt-text-2) !important;
    }}

    /* ── Why section ── */
    .md-why-box {{
        background:
            radial-gradient(circle at 8% 10%,  rgba(91, 63, 196, 0.09) 0%, transparent 36%),
            radial-gradient(circle at 92% 20%,  rgba(0, 180, 166, 0.08) 0%, transparent 36%),
            var(--lt-surface) !important;
        border-color: var(--lt-border) !important;
        box-shadow: var(--lt-shadow-md) !important;
    }}
    .md-why-box h2 {{
        color: var(--lt-text) !important;
        font-family: 'DM Serif Display', serif !important;
    }}
    .md-why-item {{
        background: rgba(255, 255, 255, 0.80) !important;
        border-color: var(--lt-border) !important;
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        transition: transform 150ms ease, box-shadow 150ms ease;
        will-change: transform;
    }}
    .md-why-item:hover {{ transform: translateY(-2px); box-shadow: var(--lt-shadow-sm); }}
    .md-why-item strong {{ color: var(--lt-primary) !important; }}
    .md-why-item span   {{ color: var(--lt-text-2) !important; }}

    /* ── Contact card ── */
    .md-contact-card {{
        background: var(--lt-surface) !important;
        border-color: var(--lt-border) !important;
        box-shadow: var(--lt-shadow-md) !important;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
    }}
    .md-contact-card h2 {{
        color: var(--lt-text) !important;
        font-family: 'DM Serif Display', serif !important;
    }}
    .md-contact-card p {{ color: var(--lt-text-2) !important; }}
    .md-contact-links a {{
        background: var(--lt-surface-high) !important;
        border-color: var(--lt-border) !important;
        color: var(--lt-primary) !important;
        font-weight: 700 !important;
        transition: all 150ms ease;
    }}
    .md-contact-links a:hover {{
        background: rgba(91, 63, 196, 0.10) !important;
        border-color: var(--lt-border-strong) !important;
        transform: translateY(-2px);
        box-shadow: var(--lt-shadow-sm);
    }}

    /* ── Streamlit widgets ── */
    .stButton > button {{
        background: linear-gradient(135deg, var(--lt-primary) 0%, var(--lt-primary-2) 100%) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 999px !important;
        font-weight: 700 !important;
        letter-spacing: 0.01em;
        box-shadow: 0 4px 18px rgba(91, 63, 196, 0.32) !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease !important;
        will-change: transform;
    }}
    .stButton > button:hover {{
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 28px rgba(91, 63, 196, 0.42) !important;
    }}

    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {{
        background: var(--lt-surface-high) !important;
        border: 1.5px solid var(--lt-border) !important;
        border-radius: 12px !important;
        color: var(--lt-text) !important;
        font-family: 'DM Sans', sans-serif !important;
    }}
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {{
        border-color: var(--lt-primary) !important;
        box-shadow: 0 0 0 3px rgba(91, 63, 196, 0.14) !important;
    }}

    [data-baseweb="select"] > div {{
        background: var(--lt-surface-high) !important;
        border-color: var(--lt-border) !important;
        border-radius: 12px !important;
        color: var(--lt-text) !important;
    }}

    .stRadio label    {{ color: var(--lt-text) !important;   font-weight: 500 !important; }}
    .stSelectbox label {{ color: var(--lt-text-2) !important; font-weight: 600 !important; font-size: 13px !important; }}

    [data-testid="stMetricValue"] {{ color: var(--lt-primary) !important; }}
    .stCaption, caption {{ color: var(--lt-text-muted) !important; }}

    hr {{ border-color: var(--lt-border) !important; opacity: 1 !important; }}

    /* ── Scrollbar ── */
    ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
    ::-webkit-scrollbar-track {{ background: transparent; }}
    ::-webkit-scrollbar-thumb {{
        background: rgba(91, 63, 196, 0.25);
        border-radius: 99px;
    }}
    ::-webkit-scrollbar-thumb:hover {{ background: rgba(91, 63, 196, 0.45); }}

    /* ── Selection ── */
    ::selection {{ background: rgba(91, 63, 196, 0.18); color: var(--lt-text); }}

    /* ══════════════════════════════════════════════════════════
       EXTENDED LIGHT MODE — covers all 5 app pages
       ══════════════════════════════════════════════════════════ */

    /* ── App background (fallback) ── */
    [data-testid="stMain"] {{ background: transparent !important; }}
    [data-testid="stMainBlockContainer"],
    [data-testid="block-container"],
    [data-testid="stVerticalBlock"],
    [data-testid="stVerticalBlockBorderWrapper"],
    [data-testid="stElementContainer"] {{
        background: transparent !important;
    }}

    /* ── All panels / cards ── */
    .panel,
    .tool-panel,
    .result-panel,
    .result-card,
    .match-card,
    .match-item,
    .stat-card,
    .info-box,
    .dis-card,
    .cat-card,
    .track-card {{
        background: rgba(255,255,255,0.85) !important;
        border-color: rgba(103,80,164,0.18) !important;
    }}

    /* ── Stat / metric cards ── */
    .md-stat-card {{
        background: var(--lt-surface) !important;
        border-color: var(--lt-border) !important;
        box-shadow: var(--lt-shadow-sm) !important;
    }}
    .md-stat-label {{ color: var(--lt-text-muted) !important; }}
    .md-stat-value {{ color: var(--lt-text) !important; }}

    /* ── Section titles ── */
    .md-section-title {{
        color: var(--lt-text) !important;
        font-family: 'DM Serif Display', serif !important;
    }}
    .md-section-subtitle {{ color: var(--lt-text-2) !important; }}

    /* ── Health / symptom shells ── */
    .md-health-shell,
    .md-symptom-shell {{
        background:
            radial-gradient(circle at 8% 20%, rgba(91, 63, 196, 0.10) 0%, transparent 38%),
            radial-gradient(circle at 92% 80%, rgba(0, 180, 166, 0.09) 0%, transparent 38%),
            var(--lt-surface) !important;
        border-color: var(--lt-border) !important;
        box-shadow: var(--lt-shadow-lg) !important;
    }}
    .md-pillar-card {{
        background: rgba(255,255,255,0.80) !important;
        border-color: var(--lt-border) !important;
    }}
    .md-pillar-bar-wrap {{ background: rgba(91, 63, 196, 0.10) !important; }}

    /* ── Risk cards ── */
    .md-risk-card {{
        background: var(--lt-surface) !important;
        border-color: var(--lt-border) !important;
        box-shadow: var(--lt-shadow-sm) !important;
    }}
    .md-risk-name  {{ color: var(--lt-text) !important; font-family: 'DM Serif Display', serif !important; }}
    .md-risk-sub   {{ color: var(--lt-text-muted) !important; }}
    .md-risk-stat  {{ color: var(--lt-text) !important; }}
    .md-risk-desc  {{ color: var(--lt-text-2) !important; }}

    /* ── Symptom tags ── */
    .md-symptom-tag {{ color: var(--lt-primary) !important; }}

    /* ── Triage ── */
    .md-triage-level {{ color: var(--lt-text) !important; }}
    .md-triage-msg   {{ color: var(--lt-text-2) !important; }}
    .md-symptom-result {{
        background: var(--lt-surface) !important;
        border-color: var(--lt-border) !important;
    }}
    .md-conditions-row {{
        background: rgba(255,255,255,0.82) !important;
        border-color: var(--lt-border) !important;
        color: var(--lt-text) !important;
    }}

    /* ── Timeline ── */
    .md-timeline-card {{
        background: var(--lt-surface) !important;
        border-color: var(--lt-border) !important;
        box-shadow: var(--lt-shadow-sm) !important;
    }}
    .md-timeline-title {{ color: var(--lt-text) !important; font-family: 'DM Serif Display', serif !important; }}
    .md-timeline-meta  {{ color: var(--lt-text-muted) !important; }}
    .md-timeline-tag {{
        background: rgba(91, 63, 196, 0.10) !important;
        border-color: rgba(91, 63, 196, 0.26) !important;
        color: var(--lt-primary) !important;
    }}

    /* ── Vitals ── */
    .md-vital-card {{
        background: var(--lt-surface) !important;
        border-color: var(--lt-border) !important;
        box-shadow: var(--lt-shadow-sm) !important;
    }}
    .md-vital-value {{ color: var(--lt-text) !important; }}
    .md-vital-unit,
    .md-vital-label {{ color: var(--lt-text-muted) !important; }}

    /* ── Marquee ── */
    .md-marquee-wrap {{
        background: var(--lt-surface) !important;
        border-color: var(--lt-border) !important;
    }}
    .md-marquee-wrap::before {{
        background: linear-gradient(90deg, #f0eef9, transparent) !important;
    }}
    .md-marquee-wrap::after {{
        background: linear-gradient(-90deg, #f0eef9, transparent) !important;
    }}
    .md-marquee-item {{
        background: rgba(255,255,255,0.88) !important;
        border-color: var(--lt-border) !important;
        color: var(--lt-primary) !important;
    }}

    /* ── Contact section ── */
    .md-contact-wrap {{
        background:
            radial-gradient(circle at 8% 10%, rgba(91, 63, 196, 0.10) 0%, transparent 40%),
            var(--lt-surface) !important;
        border-color: var(--lt-border) !important;
        box-shadow: var(--lt-shadow-md) !important;
    }}
    .md-contact-title {{ color: var(--lt-text) !important; font-family: 'DM Serif Display', serif !important; }}
    .md-contact-sub   {{ color: var(--lt-text-2) !important; }}
    .md-contact-email {{
        background: rgba(91, 63, 196, 0.09) !important;
        border-color: rgba(91, 63, 196, 0.28) !important;
        color: var(--lt-primary) !important;
    }}
    .md-contact-email:hover {{ color: var(--lt-text) !important; }}
    .md-contact-social-btn {{
        background: var(--lt-surface) !important;
        border-color: var(--lt-border) !important;
        color: var(--lt-text-2) !important;
    }}
    .md-contact-social-btn:hover {{ color: var(--lt-primary) !important; }}

    /* ── Footer ── */
    .md-footer {{ border-top-color: var(--lt-border) !important; }}
    .md-footer-brand-name {{ color: var(--lt-text) !important; font-family: 'DM Serif Display', serif !important; }}
    .md-footer-brand-sub  {{ color: var(--lt-text-muted) !important; }}
    .md-footer-link {{
        background: var(--lt-surface) !important;
        border-color: var(--lt-border) !important;
        color: var(--lt-text-2) !important;
    }}
    .md-footer-link:hover {{ color: var(--lt-primary) !important; }}
    .md-footer-meta    {{ color: var(--lt-text-muted) !important; }}
    .md-footer-version {{
        background: rgba(91, 63, 196, 0.09) !important;
        border-color: rgba(91, 63, 196, 0.22) !important;
        color: var(--lt-primary) !important;
    }}
    .md-footer-disclaimer {{ color: var(--lt-text-2) !important; }}

    /* ── Made With ── */
    .md-made-with {{ color: var(--lt-text-2) !important; }}

    /* ══ Medibot page ════════════════════════════════════════ */

    /* Sidebar profile card */
    .sb-profile-card {{
        background:
            radial-gradient(ellipse at 12% 10%, rgba(91, 63, 196, 0.12), transparent 44%),
            var(--lt-surface) !important;
        border-color: var(--lt-border) !important;
        box-shadow: var(--lt-shadow-md) !important;
    }}
    .sb-title       {{ color: var(--lt-text) !important; font-family: 'DM Serif Display', serif !important; }}
    .sb-description {{ color: var(--lt-text-2) !important; }}
    .sb-section     {{ color: var(--lt-text-muted) !important; }}
    .sb-logo, .sb-logo-fallback {{
        background: var(--lt-surface-high) !important;
        border-color: var(--lt-border) !important;
    }}
    .sb-stat-card {{
        background: var(--lt-surface) !important;
        border-color: var(--lt-border) !important;
    }}
    .sb-stat-card:hover {{ background: var(--lt-surface-high) !important; }}
    .sb-stat-number {{ color: var(--lt-text) !important; }}
    .sb-stat-label  {{ color: var(--lt-text-muted) !important; }}
    .sb-chip {{
        background: var(--lt-surface) !important;
        border-color: var(--lt-border) !important;
    }}
    .sb-chip:hover {{
        background: rgba(91, 63, 196, 0.08) !important;
        border-color: var(--lt-border-strong) !important;
    }}
    .sb-chip-label {{ color: var(--lt-text-muted) !important; }}
    .sb-chip-value {{ color: var(--lt-text) !important; }}
    .sb-note {{
        background: rgba(249,115,22,0.07) !important;
        border-color: rgba(249,115,22,0.26) !important;
        color: #7c3a00 !important;
    }}
    .sb-tip {{
        background: rgba(0,180,166,0.07) !important;
        border-color: rgba(0,180,166,0.20) !important;
        color: var(--lt-text-2) !important;
    }}
    .sb-footer {{ color: var(--lt-text-muted) !important; }}

    /* Medibot hero pills */
    .md-pill {{
        background: var(--lt-surface) !important;
        border-color: var(--lt-border) !important;
        box-shadow: var(--lt-shadow-sm) !important;
    }}
    .md-pill:hover {{ background: var(--lt-surface-high) !important; }}
    .md-pill-label {{ color: var(--lt-text-muted) !important; }}
    .md-pill-value {{ color: var(--lt-text) !important; }}
    .md-hero-stat  {{
        background: var(--lt-surface) !important;
        border-color: var(--lt-border) !important;
        color: var(--lt-text-2) !important;
    }}

    /* Info cards (Medibot) */
    .md-info-card {{
        background: var(--lt-surface) !important;
        border-color: var(--lt-border) !important;
        box-shadow: var(--lt-shadow-sm) !important;
    }}
    .md-info-card:hover {{
        background: var(--lt-surface-high) !important;
        border-color: var(--lt-border-strong) !important;
        box-shadow: var(--lt-shadow-md) !important;
    }}
    .md-info-card strong {{ color: var(--lt-text) !important; }}
    .md-info-card span   {{ color: var(--lt-text-2) !important; }}
    .md-card-icon {{
        background: linear-gradient(135deg, rgba(91, 63, 196, 0.15), rgba(0, 180, 166, 0.12)) !important;
        border-color: rgba(91, 63, 196, 0.14) !important;
    }}

    /* Topic cards */
    .md-topic-card {{
        background: var(--lt-surface) !important;
        border-color: var(--lt-border) !important;
    }}
    .md-topic-card:hover {{
        background: var(--lt-surface-high) !important;
        border-color: var(--lt-border-strong) !important;
    }}
    .md-topic-label {{ color: var(--lt-text) !important; }}
    .md-topic-sub   {{ color: var(--lt-text-muted) !important; }}
    .md-topic-btn .stButton > button {{
        background: rgba(91, 63, 196, 0.09) !important;
        border-color: var(--lt-border) !important;
        color: var(--lt-primary) !important;
    }}
    .md-topic-btn .stButton > button:hover {{
        background: rgba(91, 63, 196, 0.18) !important;
        color: var(--lt-text) !important;
    }}
    .md-topic-btn .stButton > button p,
    .md-topic-btn .stButton > button span {{ color: inherit !important; }}

    /* Section sub */
    .md-section-sub {{ color: var(--lt-text-2) !important; }}

    /* Empty state */
    .md-empty {{
        background: var(--lt-surface) !important;
        border-color: var(--lt-border) !important;
        color: var(--lt-text-2) !important;
    }}

    /* Source list */
    .md-source-list {{
        background: var(--lt-surface) !important;
        border-color: var(--lt-border) !important;
    }}
    .md-source-title {{ color: var(--lt-text-muted) !important; }}
    .md-source-item {{
        background: rgba(255,255,255,0.72) !important;
        border-color: var(--lt-border) !important;
    }}
    .md-source-item:hover {{ background: var(--lt-surface-high) !important; }}
    .md-source-text {{ color: var(--lt-text-2) !important; }}

    /* Confidence badges */
    .md-confidence-high {{ color: #166534 !important; }}
    .md-confidence-med  {{ color: #92400e !important; }}
    .md-confidence-low  {{ color: #991b1b !important; }}

    /* History banner */
    .md-history-banner {{
        background: var(--lt-surface) !important;
        border-color: var(--lt-border) !important;
        color: var(--lt-text) !important;
    }}
    .md-history-count {{ color: var(--lt-text) !important; }}

    /* Suggestion chips */
    .md-suggestion-chip {{
        background: var(--lt-surface) !important;
        border-color: var(--lt-border) !important;
        color: var(--lt-text-2) !important;
    }}
    .md-suggestion-chip:hover {{
        background: rgba(91, 63, 196, 0.09) !important;
        border-color: var(--lt-border-strong) !important;
        color: var(--lt-primary) !important;
    }}

    /* Disclaimer */
    .md-disclaimer {{ color: var(--lt-text-2) !important; }}

    /* Read meta */
    .md-read-meta {{
        background: var(--lt-surface) !important;
        border-color: var(--lt-border) !important;
        color: var(--lt-text-muted) !important;
    }}

    /* Tools bar */
    .md-tools-bar {{
        background: var(--lt-surface) !important;
        border-color: var(--lt-border) !important;
    }}
    .md-tool-btn {{
        background: rgba(255,255,255,0.80) !important;
        border-color: var(--lt-border) !important;
        color: var(--lt-text-2) !important;
    }}
    .md-tool-btn:hover {{ color: var(--lt-primary) !important; background: rgba(91,63,196,0.08) !important; }}
    .md-tool-label   {{ color: var(--lt-text-muted) !important; }}
    .md-tool-divider {{ background: var(--lt-border) !important; }}

    /* Highlight */
    mark.md-highlight {{
        background: rgba(91, 63, 196, 0.18) !important;
        color: var(--lt-text) !important;
    }}

    /* Health score quiz */
    .hs-panel {{
        background: var(--lt-surface) !important;
        border-color: var(--lt-border) !important;
        box-shadow: var(--lt-shadow-md) !important;
    }}
    .hs-q-card {{
        background: var(--lt-surface-high) !important;
        border-color: var(--lt-border) !important;
    }}
    .hs-q-text {{ color: var(--lt-text) !important; }}
    .hs-q-card .stButton > button {{
        background: var(--lt-surface) !important;
        border-color: var(--lt-border) !important;
        color: var(--lt-text-2) !important;
    }}
    .hs-q-card .stButton > button:hover {{
        background: rgba(91, 63, 196, 0.10) !important;
        border-color: var(--lt-border-strong) !important;
        color: var(--lt-text) !important;
    }}
    .hs-result-headline {{ color: var(--lt-text) !important; font-family: 'DM Serif Display', serif !important; }}
    .hs-result-sub      {{ color: var(--lt-text-2) !important; }}
    .hs-arc-bg          {{ stroke: rgba(91, 63, 196, 0.12) !important; }}
    .hs-arc-num         {{ color: var(--lt-text) !important; }}
    .hs-arc-denom       {{ color: var(--lt-text-muted) !important; }}
    .hs-badge-good      {{
        background: rgba(91, 63, 196, 0.10) !important;
        border-color: rgba(91, 63, 196, 0.28) !important;
        color: var(--lt-primary) !important;
    }}
    .hs-factor-v2 {{ background: rgba(255,255,255,0.78) !important; }}
    .hs-factor-v2:hover {{ background: var(--lt-surface-high) !important; }}
    .hs-fv2-label {{ color: var(--lt-text-muted) !important; }}
    .hs-fv2-val   {{ color: var(--lt-text) !important; }}
    .hs-fv2-sub   {{ color: var(--lt-text-muted) !important; }}
    .hs-retake {{
        background: var(--lt-surface) !important;
        border-color: var(--lt-border) !important;
        color: var(--lt-text-2) !important;
    }}
    .hs-retake:hover {{ color: var(--lt-primary) !important; }}

    /* Symptom checker (Medibot) */
    .sym-panel {{
        background: var(--lt-surface) !important;
        border-color: var(--lt-border) !important;
        box-shadow: var(--lt-shadow-md) !important;
    }}
    .sym-chip {{
        background: var(--lt-surface) !important;
        border-color: var(--lt-border) !important;
        color: var(--lt-text-2) !important;
    }}
    .sym-chip:hover {{
        background: rgba(91, 63, 196, 0.09) !important;
        border-color: var(--lt-border-strong) !important;
    }}
    .sym-chip.selected {{
        background: rgba(91, 63, 196, 0.14) !important;
        border-color: rgba(91, 63, 196, 0.50) !important;
    }}
    .sym-chip-label {{ color: var(--lt-text) !important; }}
    .sym-chip-sub   {{ color: var(--lt-text-muted) !important; }}
    .sym-sev-btn {{
        background: var(--lt-surface) !important;
        border-color: var(--lt-border) !important;
        color: var(--lt-text-2) !important;
    }}
    .sym-sev-btn:hover {{ color: var(--lt-text) !important; }}
    .sym-result {{
        background: rgba(0,180,166,0.07) !important;
        border-color: rgba(0,180,166,0.22) !important;
    }}
    .sym-result-title {{ color: var(--lt-text) !important; }}
    .sym-result-body  {{ color: var(--lt-text-2) !important; }}
    .sym-urgency-low    {{ color: #166534 !important; }}
    .sym-urgency-medium {{ color: #92400e !important; }}
    .sym-urgency-high   {{ color: #991b1b !important; }}

    /* Medication reminder */
    .rem-panel {{
        background: var(--lt-surface) !important;
        border-color: var(--lt-border) !important;
    }}
    .rem-item {{
        background: rgba(255,255,255,0.78) !important;
        border-color: var(--lt-border) !important;
    }}
    .rem-item:hover {{ background: var(--lt-surface-high) !important; }}
    .rem-name {{ color: var(--lt-text) !important; }}
    .rem-dose, .rem-time {{ color: var(--lt-text-muted) !important; }}
    .rem-badge-taken   {{ color: #166534 !important; }}
    .rem-badge-pending {{ color: #92400e !important; }}
    .rem-badge-missed  {{ color: #991b1b !important; }}

    /* Tip card */
    .md-tip-card {{
        background: rgba(0,180,166,0.06) !important;
        border-color: rgba(0,180,166,0.20) !important;
    }}
    .md-tip-badge {{
        background: rgba(0,180,166,0.12) !important;
        border-color: rgba(0,180,166,0.22) !important;
        color: #0d6b62 !important;
    }}
    .md-tip-title {{ color: var(--lt-text) !important; }}
    .md-tip-body  {{ color: var(--lt-text-2) !important; }}

    /* ══ Disease Prediction page ═════════════════════════════ */
    .sb-hero {{
        background: var(--lt-surface-high) !important;
        border-color: var(--lt-border) !important;
        box-shadow: var(--lt-shadow-md) !important;
    }}
    .sb-badge {{
        background: rgba(91, 63, 196, 0.10) !important;
        border-color: rgba(91, 63, 196, 0.28) !important;
        color: var(--lt-primary) !important;
    }}
    .sb-title   {{ color: var(--lt-text) !important; font-family: 'DM Serif Display', serif !important; }}
    .sb-text    {{ color: var(--lt-text-2) !important; }}
    .sb-section {{ color: var(--lt-text-muted) !important; }}
    .sb-link {{
        background: var(--lt-surface) !important;
        border-color: var(--lt-border) !important;
    }}
    .sb-link:hover {{
        background: rgba(91, 63, 196, 0.08) !important;
        border-color: var(--lt-border-strong) !important;
    }}
    .sb-link-title {{ color: var(--lt-text) !important; font-weight: 700 !important; }}
    .sb-link-sub   {{ color: var(--lt-text-muted) !important; }}
    .sb-footer {{ color: var(--lt-text-muted) !important; }}

    .hero {{
        background:
            radial-gradient(circle at 8% 20%, rgba(91, 63, 196, 0.12) 0%, transparent 40%),
            radial-gradient(circle at 92% 10%, rgba(0, 180, 166, 0.10) 0%, transparent 38%),
            var(--lt-surface-high) !important;
        border-color: var(--lt-border) !important;
        box-shadow: var(--lt-shadow-lg) !important;
    }}
    .hero-eyebrow {{
        background: rgba(91, 63, 196, 0.10) !important;
        border-color: rgba(91, 63, 196, 0.28) !important;
        color: var(--lt-primary) !important;
    }}
    .hero-title {{
        color: var(--lt-text) !important;
        font-family: 'DM Serif Display', serif !important;
    }}
    .hero-sub {{ color: var(--lt-text-2) !important; }}
    .hero-pill {{
        background: var(--lt-surface) !important;
        border-color: var(--lt-border) !important;
    }}
    .hero-pill-label {{ color: var(--lt-text-muted) !important; }}
    .hero-pill-value {{ color: var(--lt-text) !important; }}
    .chip {{
        background: rgba(255,255,255,0.88) !important;
        border-color: var(--lt-border) !important;
        color: var(--lt-primary) !important;
    }}
    .stat-card {{
        background: var(--lt-surface) !important;
        border-color: var(--lt-border) !important;
        box-shadow: var(--lt-shadow-sm) !important;
    }}
    .stat-label {{ color: var(--lt-text-muted) !important; }}
    .stat-value {{ color: var(--lt-text) !important; }}
    .sec-title {{
        color: var(--lt-text) !important;
        font-family: 'DM Serif Display', serif !important;
    }}
    .sec-sub {{ color: var(--lt-text-2) !important; }}
    .tool-panel, .result-panel, .info-box {{
        background: var(--lt-surface) !important;
        border-color: var(--lt-border) !important;
    }}
    .info-box-title {{ color: var(--lt-text) !important; }}
    .info-box-sub   {{ color: var(--lt-text-2) !important; }}
    .sym-chip.severe {{
        border-color: rgba(186,26,26,0.38) !important;
        background: rgba(186,26,26,0.08) !important;
        color: #8b2c2c !important;
    }}
    .result-card, .match-card {{
        background: var(--lt-surface) !important;
        border-color: var(--lt-border) !important;
        box-shadow: var(--lt-shadow-sm) !important;
    }}
    .result-title {{ color: var(--lt-text) !important; font-family: 'DM Serif Display', serif !important; }}
    .result-desc  {{ color: var(--lt-text-2) !important; }}
    .match-head   {{ color: var(--lt-text) !important; }}
    .match-score  {{ color: var(--lt-primary) !important; }}
    .match-meta   {{ color: var(--lt-text-muted) !important; }}
    .dis-card, .cat-card {{
        background: var(--lt-surface) !important;
        border-color: var(--lt-border) !important;
    }}
    .dis-name {{ color: var(--lt-text) !important; }}
    .dis-tag {{
        background: rgba(0,180,166,0.09) !important;
        border-color: rgba(0,180,166,0.26) !important;
        color: #005f58 !important;
    }}
    .cat-name  {{ color: var(--lt-text) !important; }}
    .cat-count {{ color: var(--lt-text-muted) !important; }}
    .pretty-list .pretty-item {{
        background: rgba(91, 63, 196, 0.06) !important;
        border-color: rgba(91, 63, 196, 0.18) !important;
        color: var(--lt-text-2) !important;
    }}
    .pretty-item  {{ color: var(--lt-text-2) !important; }}
    .track-card {{
        background: var(--lt-surface) !important;
        border-color: var(--lt-border) !important;
    }}
    .track-disease {{ color: var(--lt-text) !important; }}
    .track-date    {{ color: var(--lt-text-muted) !important; }}
    .muted {{ color: var(--lt-text-muted) !important; }}

    /* ══ Drug Recommendation page ════════════════════════════ */
    .hero-wrap {{
        background: var(--lt-surface-high) !important;
        border-color: var(--lt-border) !important;
        box-shadow: var(--lt-shadow-lg) !important;
    }}
    .desc-card, .drug-card, .compare-card, .interaction-card,
    .health-tip-card, .dose-card {{
        background: var(--lt-surface) !important;
        border-color: var(--lt-border) !important;
        color: var(--lt-text) !important;
        box-shadow: var(--lt-shadow-sm) !important;
    }}
    .drug-name  {{ color: var(--lt-text) !important; font-family: 'DM Serif Display', serif !important; }}
    .drug-meta  {{ color: var(--lt-text-muted) !important; }}
    .drug-desc  {{ color: var(--lt-text-2) !important; }}
    .drug-badge {{
        background: rgba(91, 63, 196, 0.09) !important;
        border-color: rgba(91, 63, 196, 0.24) !important;
        color: var(--lt-primary) !important;
    }}
    .compare-title {{ color: var(--lt-text) !important; font-family: 'DM Serif Display', serif !important; }}
    .compare-sub   {{ color: var(--lt-text-2) !important; }}
    .az-drug-item {{
        background: var(--lt-surface) !important;
        border-color: var(--lt-border) !important;
        color: var(--lt-text-2) !important;
    }}
    .az-drug-item:hover {{
        background: rgba(91, 63, 196, 0.08) !important;
        border-color: var(--lt-border-strong) !important;
    }}
    .az-summary-card {{
        background: var(--lt-surface) !important;
        border-color: var(--lt-border) !important;
    }}
    .az-filter-panel, .dose-hero-card, .dose-reminder-panel,
    .dose-form-card, .dose-safety-card {{
        background: var(--lt-surface-high) !important;
        border-color: var(--lt-border) !important;
        color: var(--lt-text) !important;
    }}
    .dose-reminder-title, .dose-form-title,
    .dose-safety-title, .dose-tips-title {{ color: var(--lt-text) !important; }}
    .dose-form-sub, .dose-info-text, .dose-safety-body {{ color: var(--lt-text-2) !important; }}
    .dose-check {{
        background: rgba(91, 63, 196, 0.07) !important;
        border-color: var(--lt-border) !important;
        color: var(--lt-text-2) !important;
    }}
    .dose-tip-list li {{ color: var(--lt-text-2) !important; }}
    .interaction-title {{ color: var(--lt-text) !important; }}
    .interaction-body  {{ color: var(--lt-text-2) !important; }}

    /* ══ Heart Disease Risk Assessment page ══════════════════ */
    .md-hero-logo {{
        background: var(--lt-surface-high) !important;
        border-color: var(--lt-border) !important;
    }}
    .md-hero-brand .md-title {{
        color: var(--lt-text) !important;
        font-family: 'DM Serif Display', serif !important;
    }}
    .md-form-title    {{
        color: var(--lt-text) !important;
        font-family: 'DM Serif Display', serif !important;
    }}
    .md-form-subtitle {{ color: var(--lt-text-2) !important; }}
    .md-muted         {{ color: var(--lt-text-muted) !important; }}
    .md-rec-item {{
        background: var(--lt-surface) !important;
        border-color: var(--lt-border) !important;
        color: var(--lt-text) !important;
    }}
    .md-rec-item:hover {{
        background: rgba(91, 63, 196, 0.08) !important;
        border-color: var(--lt-border-strong) !important;
    }}
    .md-metric {{
        background: var(--lt-surface) !important;
        border-color: var(--lt-border) !important;
        box-shadow: var(--lt-shadow-sm) !important;
    }}
    .md-metric:hover {{ background: var(--lt-surface-high) !important; }}
    .md-metric-label {{ color: var(--lt-text-muted) !important; }}
    .md-metric-value {{ color: var(--lt-text) !important; font-family: 'DM Serif Display', serif !important; }}
    .md-metric-sub   {{ color: var(--lt-text-muted) !important; }}
    .md-progress     {{ background: rgba(91, 63, 196, 0.10) !important; }}
    .md-result-hero h2 {{
        color: var(--lt-text) !important;
        font-family: 'DM Serif Display', serif !important;
    }}
    .md-callout-ok {{
        background: rgba(0,180,166,0.08) !important;
        border-color: rgba(0,180,166,0.24) !important;
        color: var(--lt-text-2) !important;
    }}
    .md-callout-warn {{
        background: rgba(249,115,22,0.07) !important;
        border-color: rgba(249,115,22,0.24) !important;
        color: var(--lt-text-2) !important;
    }}
    .md-info-icon {{
        background: linear-gradient(135deg, rgba(91, 63, 196, 0.15), rgba(0, 180, 166, 0.12)) !important;
        border-color: rgba(91, 63, 196, 0.14) !important;
    }}
    .md-sidebar-tip {{
        background: rgba(0,180,166,0.07) !important;
        border-color: rgba(0,180,166,0.18) !important;
        color: var(--lt-text-2) !important;
    }}

    /* ── Tabs (all pages) ── */
    .stTabs [data-baseweb="tab-list"] {{
        background: var(--lt-surface) !important;
        border-color: var(--lt-border) !important;
    }}
    .stTabs [data-baseweb="tab"] {{
        color: var(--lt-text-2) !important;
        font-weight: 700 !important;
    }}
    .stTabs [aria-selected="true"] {{ color: white !important; }}
    .stTabs [data-baseweb="tab-panel"] {{ padding-top: 16px; }}

    /* ── Expanders — header + ALL inner content ── */
    [data-testid="stExpander"],
    [data-testid="stExpander"] > div,
    [data-testid="stExpander"] details,
    [data-testid="stExpanderDetails"],
    [data-testid="stExpanderDetails"] > div {{
        background: rgba(255, 255, 255, 0.88) !important;
        border: 1px solid var(--lt-border) !important;
        border-radius: 12px !important;
        color: var(--lt-text) !important;
    }}
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary * {{
        color: var(--lt-text) !important;
        background: transparent !important;
    }}
    [data-testid="stExpanderDetails"] *,
    [data-testid="stExpanderDetails"] p,
    [data-testid="stExpanderDetails"] li,
    [data-testid="stExpanderDetails"] span {{
        color: var(--lt-text) !important;
        background: transparent !important;
    }}

    /* ── Tab panel inner content ── */
    [data-baseweb="tab-panel"],
    [role="tabpanel"],
    .stTabs [data-baseweb="tab-panel"] > div {{
        background: transparent !important;
        color: var(--lt-text) !important;
    }}

    /* ── Streamlit form container ── */
    [data-testid="stForm"] {{
        background: rgba(255, 255, 255, 0.82) !important;
        border-color: var(--lt-border) !important;
    }}

    /* ── Multiselect dropdown popup ── */
    [data-baseweb="popover"] [data-baseweb="menu"],
    [data-baseweb="popover"] > div,
    [role="listbox"],
    [data-baseweb="menu"] {{
        background: #f5f2ff !important;
        border: 1px solid rgba(103, 80, 164, 0.22) !important;
        color: var(--lt-text) !important;
    }}
    [data-baseweb="option"] {{
        background: transparent !important;
        color: var(--lt-text) !important;
    }}
    [data-baseweb="option"]:hover,
    [data-baseweb="option"][aria-selected="true"] {{
        background: rgba(103, 80, 164, 0.12) !important;
        color: var(--lt-text) !important;
    }}
    /* Multiselect selected tags */
    [data-baseweb="tag"] {{
        background: rgba(91, 63, 196, 0.14) !important;
        border: 1px solid rgba(91, 63, 196, 0.35) !important;
    }}
    [data-baseweb="tag"] span {{ color: var(--lt-text) !important; }}
    [data-baseweb="tag"] [role="button"] {{ color: var(--lt-primary) !important; }}

    /* ── Select / dropdown container text ── */
    [data-baseweb="select"] span,
    [data-baseweb="select"] div {{
        color: var(--lt-text) !important;
    }}

    /* ── General text readability catch-all ── */
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] span {{
        color: var(--lt-text) !important;
    }}
    [data-testid="stText"],
    [data-testid="stText"] p {{
        color: var(--lt-text) !important;
    }}
</style>
"""


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def init_theme() -> None:
    """Initialize theme in session state (call once at app start)."""
    if "theme" not in st.session_state:
        st.session_state.theme = DARK


def toggle_theme() -> None:
    """Toggle between dark and light themes."""
    st.session_state.theme = LIGHT if st.session_state.theme == DARK else DARK


def get_theme_styles() -> str:
    """Return the CSS <style> block for the current theme."""
    return _CSS_LIGHT if st.session_state.get("theme", DARK) == LIGHT else _CSS_DARK


def apply_theme() -> None:
    """
    Convenience helper — inject the theme CSS into the page.
    Call this near the top of every page after init_theme().
    """
    st.markdown(get_theme_styles(), unsafe_allow_html=True)


def render_theme_toggle() -> None:
    """Render a compact theme toggle button aligned to the top-right."""
    theme = st.session_state.get("theme", DARK)
    icon  = "☀️" if theme == DARK else "🌙"
    label = "Light mode" if theme == DARK else "Dark mode"

    # Inject data-theme onto <html> so CSS selectors and JS detection work
    st.markdown(f"""
<script>
(function() {{
    var t = "{theme}";
    document.documentElement.setAttribute("data-theme", t);
    document.body.setAttribute("data-theme", t);
    try {{ window.localStorage.setItem("stActiveTheme", t); }} catch(e) {{}}
}})();
</script>
""", unsafe_allow_html=True)

    # Push toggle to the right using a 3-column layout
    _, _, col = st.columns([6, 1, 1])
    with col:
        if st.button(icon, key="theme_toggle", help=f"Switch to {label}"):
            toggle_theme()
            st.rerun()