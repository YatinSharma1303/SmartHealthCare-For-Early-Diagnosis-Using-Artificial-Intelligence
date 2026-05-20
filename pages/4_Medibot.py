import os
import re
import base64
import asyncio
import time
import json
from pathlib import Path
from html import escape
from datetime import datetime

import nest_asyncio
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv, find_dotenv
from PIL import Image

from theme_config import init_theme, get_theme_styles, render_theme_toggle

from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings


import threading

load_dotenv(find_dotenv())
nest_asyncio.apply()

APP_TITLE = "Medibot - AI Health Assistant"
LOGO_PATH = Path("utils/ph2.png")
DB_FAISS_PATH = Path("vectorstore/db_faiss")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

TOP_K_RESULTS = 6
MAX_CHAT_MESSAGES = 12


@st.cache_data(show_spinner=False)
def load_page_icon(path):
    try:
        return Image.open(path)
    except Exception:
        return "🤖"


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


st.set_page_config(
    page_title=APP_TITLE,
    page_icon=load_page_icon(LOGO_PATH),
    layout="wide",
)

init_theme()
st.markdown(get_theme_styles(), unsafe_allow_html=True)
render_theme_toggle()

logo_data_uri = image_to_data_uri(LOGO_PATH)

# Hero card: use same icon as sidebar (brain + sparkle), no image
logo_html = (
    "<div class='md-logo-fallback' style='position:relative;'>"
    "<span style='font-size:42px;line-height:1;'>🧠</span>"
    "<span style='font-size:20px;position:absolute;bottom:8px;right:8px;line-height:1;'>✦</span>"
    "</div>"
)

sidebar_logo_html = "<div class='sb-logo-fallback'><span style='font-size:22px;line-height:1;'>🧠</span><span style='font-size:11px;position:absolute;top:5px;left:6px;line-height:1;opacity:0.75;'>✦</span></div>"


# ─────────────────────────────────────────────
#  STYLES
# ─────────────────────────────────────────────
def render_voice_input():
    """
    Voice input that works reliably with Streamlit.
    JS transcribes audio → stores text in sessionStorage →
    a hidden st.text_input reads it back via JS and triggers a Streamlit rerun.
    The transcribed text is then available in st.session_state.voice_query.
    """
    groq_key = os.environ.get("GROQ_API_KEY", "")

    # ── Read voice text from query param set by JS ────────────────────────
    # JS sets ?vq=<transcribed text> in the URL → Streamlit reruns →
    # we read it here, store in session_state, clear the param.
    # No st.text_input needed — zero visible widgets.
    _vq = st.query_params.get("vq", "").strip()
    if _vq:
        st.session_state["voice_query"] = _vq
        st.query_params.pop("vq", None)

    # ── Mic button + transcription component ───────────────────────────────
    components.html(
        f"""
<style>
#voice-btn {{
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 10px 20px;
    border-radius: 999px;
    border: 1px solid rgba(124, 92, 191, 0.45);
    background: linear-gradient(135deg, rgba(124,92,191,0.18), rgba(0,133,122,0.10));
    color: #d7c7ff;
    font-family: 'Sora', sans-serif;
    font-size: 13px;
    font-weight: 700;
    cursor: pointer;
    transition: all 200ms ease;
    backdrop-filter: blur(8px);
    box-shadow: 0 4px 16px rgba(10,15,30,0.12);
    user-select: none;
    letter-spacing: 0.02em;
}}
#voice-btn:hover {{
    background: linear-gradient(135deg, rgba(124,92,191,0.32), rgba(0,133,122,0.18));
    border-color: rgba(124,92,191,0.70);
    box-shadow: 0 6px 20px rgba(124,92,191,0.22);
    transform: translateY(-1px);
}}
#voice-btn.listening {{
    background: linear-gradient(135deg, rgba(229,57,53,0.28), rgba(217,119,6,0.16));
    border-color: rgba(229,57,53,0.65);
    color: #fca5a5;
    animation: voice-pulse 1.4s ease-in-out infinite;
}}
#voice-btn:disabled {{ opacity: 0.45; cursor: not-allowed; transform: none; }}
#voice-status {{
    font-family: 'DM Sans', sans-serif;
    font-size: 12px;
    color: rgba(203,213,225,0.70);
    margin-left: 6px;
    font-style: italic;
    transition: color 200ms ease;
}}
#voice-status.active {{ color: #86efac; }}
#voice-status.error  {{ color: #fca5a5; }}
@keyframes voice-pulse {{
    0%,100% {{ box-shadow: 0 0 0 0 rgba(229,57,53,0.45); }}
    55%     {{ box-shadow: 0 0 0 7px rgba(229,57,53,0.0); }}
}}
</style>
<div style="display:flex; align-items:center; margin-bottom:10px; margin-top:2px;">
    <button id="voice-btn" title="Click to speak your question">
        <span id="voice-icon">🎤</span>
        <span id="voice-label">Speak</span>
    </button>
    <span id="voice-status">🎙️ Tap the mic to consult Medibot by voice</span>
</div>
<script>
(function() {{
    const GROQ_API_KEY = "{groq_key}";
    const STORAGE_KEY  = "medibot_voice_text";
    const btn    = document.getElementById('voice-btn');
    const icon   = document.getElementById('voice-icon');
    const label  = document.getElementById('voice-label');
    const status = document.getElementById('voice-status');

    // ── Send transcribed text to Streamlit via URL query param ────────
    // Setting ?vq=<text> on the parent URL triggers a Streamlit rerun.
    // Python reads st.query_params["vq"], stores it in session_state, clears it.
    // No hidden input widget needed — nothing visible rendered.
    function sendToStreamlit(text) {{
        if (!text || !text.trim()) return;
        const parentWin = window.parent;
        const url = new URL(parentWin.location.href);
        url.searchParams.set('vq', text.trim());
        parentWin.history.pushState({{}},'', url.toString());
        parentWin.dispatchEvent(new PopStateEvent('popstate', {{state:{{}}}}));

        status.textContent = '✔ Sent: ' + text.trim().slice(0, 55) + (text.length > 55 ? '…' : '');
        status.className = 'active';
        setTimeout(() => {{
            status.textContent = '🎙️ Tap the mic to consult Medibot by voice';
            status.className = '';
        }}, 4000);
    }}

    // ── Feature detection ───────────────────────────────────────────────
    const _host = location.hostname;
    const isLanIp = /^(localhost|127\.|10\.|192\.168\.|172\.(1[6-9]|2\d|3[01])\.)/.test(_host);
    const isSecure = window.isSecureContext
        || location.protocol === 'https:'
        || isLanIp;
    const hasMediaRecorder = !!(navigator.mediaDevices
        && navigator.mediaDevices.getUserMedia
        && window.MediaRecorder
        && isSecure);
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    // Allow Web Speech API on LAN even without isSecureContext (works in Chrome on HTTP LAN)
    const hasSpeechAPI = !!(SpeechRecognition && (isLanIp || isSecure));

    if (!hasMediaRecorder && !hasSpeechAPI) {{
        btn.disabled = true;
        status.textContent = 'Voice not supported — open via HTTPS or localhost';
        status.className = 'error';
        return;
    }}

    // ── Web Speech API path (HTTP / Android Chrome / LAN fallback) ─────
    if (!hasMediaRecorder && hasSpeechAPI) {{
        const recognition = new SpeechRecognition();
        recognition.lang = 'en-US';
        recognition.interimResults = true;
        recognition.maxAlternatives = 1;
        recognition.continuous = false;
        let listening = false, finalTranscript = '', interimTranscript = '', hadError = false;

        function setFB(state) {{
            listening = state;
            btn.classList.toggle('listening', state);
            icon.textContent  = state ? '⏹' : '🎤';
            label.textContent = state ? 'Stop' : 'Speak';
            status.className  = state ? 'active' : '';
            if (state) status.textContent = 'Listening…';
        }}
        recognition.onresult = function(e) {{
            let interim = '';
            for (let i = 0; i < e.results.length; i++) {{
                const t = e.results[i][0].transcript;
                if (e.results[i].isFinal) finalTranscript += t; else interim += t;
            }}
            interimTranscript = interim;
            const d = finalTranscript || interim;
            status.textContent = d ? ('Heard: ' + d) : 'Listening…';
        }};
        recognition.onend = function() {{
            setFB(false);
            if (hadError) {{ hadError = false; return; }}
            const best = finalTranscript.trim() || interimTranscript.trim();
            if (best) {{ sendToStreamlit(best); }}
            else {{
                status.textContent = 'Nothing heard — try again';
                status.className = 'error';
                setTimeout(() => {{ status.textContent = '🎙️ Tap the mic to consult Medibot by voice'; status.className = ''; }}, 3000);
            }}
        }};
        recognition.onerror = function(e) {{
            hadError = true; setFB(false);
            const msgs = {{'not-allowed':'Mic access denied','no-speech':'Nothing heard — try again','network':'Network error','audio-capture':'No microphone found','aborted':'Cancelled'}};
            status.textContent = msgs[e.error] || ('Error: ' + e.error);
            status.className = 'error';
            setTimeout(() => {{ status.textContent = '🎙️ Tap the mic to consult Medibot by voice'; status.className = ''; }}, 4000);
        }};
        btn.addEventListener('click', function() {{
            if (listening) {{ recognition.stop(); }}
            else {{ finalTranscript=''; interimTranscript=''; hadError=false; recognition.start(); setFB(true); }}
        }});
        return;
    }}

    // ── Whisper / MediaRecorder path (HTTPS / LAN desktop) ─────────────
    if (!GROQ_API_KEY) {{
        // Fall back to Web Speech API if Groq key is absent
        if (hasSpeechAPI) {{
            const recognition2 = new SpeechRecognition();
            recognition2.lang = 'en-US';
            recognition2.interimResults = true;
            recognition2.maxAlternatives = 1;
            recognition2.continuous = false;
            let listening2 = false, finalT2 = '', interimT2 = '', hadErr2 = false;
            function setFB2(state) {{
                listening2 = state;
                btn.classList.toggle('listening', state);
                icon.textContent  = state ? '⏹' : '🎤';
                label.textContent = state ? 'Stop' : 'Speak';
                status.className  = state ? 'active' : '';
                if (state) status.textContent = 'Listening…';
            }}
            recognition2.onresult = function(e) {{
                let interim = '';
                for (let i = 0; i < e.results.length; i++) {{
                    const t = e.results[i][0].transcript;
                    if (e.results[i].isFinal) finalT2 += t; else interim += t;
                }}
                interimT2 = interim;
                const d = finalT2 || interim;
                status.textContent = d ? ('Heard: ' + d) : 'Listening…';
            }};
            recognition2.onend = function() {{
                setFB2(false);
                if (hadErr2) {{ hadErr2 = false; return; }}
                const best = finalT2.trim() || interimT2.trim();
                if (best) {{ sendToStreamlit(best); }}
                else {{
                    status.textContent = 'Nothing heard — try again';
                    status.className = 'error';
                    setTimeout(() => {{ status.textContent = '🎙️ Tap the mic to consult Medibot by voice'; status.className = ''; }}, 3000);
                }}
            }};
            recognition2.onerror = function(e) {{
                hadErr2 = true; setFB2(false);
                const msgs = {{'not-allowed':'Mic access denied','no-speech':'Nothing heard — try again','network':'Network error','audio-capture':'No microphone found','aborted':'Cancelled'}};
                status.textContent = msgs[e.error] || ('Error: ' + e.error);
                status.className = 'error';
                setTimeout(() => {{ status.textContent = '🎙️ Tap the mic to consult Medibot by voice'; status.className = ''; }}, 4000);
            }};
            btn.addEventListener('click', function() {{
                if (listening2) {{ recognition2.stop(); }}
                else {{ finalT2=''; interimT2=''; hadErr2=false; recognition2.start(); setFB2(true); }}
            }});
            return;
        }}
        btn.disabled = true;
        status.textContent = 'GROQ_API_KEY not set — voice unavailable';
        status.className = 'error';
        return;
    }}

    let mediaRecorder = null, audioChunks = [], recording = false, stream = null;

    function setRec(state) {{
        recording = state;
        btn.classList.toggle('listening', state);
        icon.textContent  = state ? '⏹' : '🎤';
        label.textContent = state ? 'Stop' : 'Speak';
        status.className  = state ? 'active' : '';
        if (state) status.textContent = 'Recording… click Stop when done';
    }}

    function getBestMime() {{
        for (const t of ['audio/webm;codecs=opus','audio/webm','audio/ogg;codecs=opus','audio/ogg','audio/mp4'])
            if (MediaRecorder.isTypeSupported(t)) return t;
        return '';
    }}

    async function transcribe(blob) {{
        status.textContent = 'Transcribing… please wait';
        status.className = 'active';
        try {{
            const mime = blob.type || 'audio/webm';
            const ext  = mime.includes('ogg') ? 'ogg' : mime.includes('mp4') ? 'mp4' : 'webm';
            const fd   = new FormData();
            fd.append('file', blob, 'voice.' + ext);
            fd.append('model', 'whisper-large-v3-turbo');
            fd.append('language', 'en');
            fd.append('response_format', 'json');
            const res = await fetch('https://api.groq.com/openai/v1/audio/transcriptions', {{
                method: 'POST',
                headers: {{ 'Authorization': 'Bearer ' + GROQ_API_KEY }},
                body: fd,
            }});
            if (!res.ok) throw new Error('Groq ' + res.status + ': ' + await res.text());
            const data = await res.json();
            const text = (data.text || '').trim();
            if (text) {{ sendToStreamlit(text); }}
            else {{
                status.textContent = 'Nothing heard — try again';
                status.className = 'error';
                setTimeout(() => {{ status.textContent = '🎙️ Tap the mic to consult Medibot by voice'; status.className = ''; }}, 3000);
            }}
        }} catch(err) {{
            status.textContent = 'Transcription failed: ' + err.message;
            status.className = 'error';
            setTimeout(() => {{ status.textContent = '🎙️ Tap the mic to consult Medibot by voice'; status.className = ''; }}, 5000);
        }}
    }}

    async function startRec() {{
        try {{
            stream = await navigator.mediaDevices.getUserMedia({{audio:{{echoCancellation:true,noiseSuppression:true,sampleRate:16000}}}});
            audioChunks = [];
            const mime = getBestMime();
            mediaRecorder = new MediaRecorder(stream, mime ? {{mimeType:mime}} : {{}});
            mediaRecorder.ondataavailable = e => {{ if (e.data && e.data.size > 0) audioChunks.push(e.data); }};
            mediaRecorder.onstop = async () => {{
                if (stream) {{ stream.getTracks().forEach(t => t.stop()); stream = null; }}
                if (!audioChunks.length) {{
                    status.textContent = 'No audio captured — try again'; status.className = 'error';
                    setTimeout(() => {{ status.textContent = '🎙️ Tap the mic to consult Medibot by voice'; status.className = ''; }}, 3000);
                    return;
                }}
                await transcribe(new Blob(audioChunks, {{type: mediaRecorder.mimeType || 'audio/webm'}}));
            }};
            mediaRecorder.onerror = () => {{ setRec(false); status.textContent = 'Recording error'; status.className = 'error'; }};
            mediaRecorder.start(250);
            setRec(true);
        }} catch(err) {{
            const msgs = {{'NotAllowedError':'Mic access denied — allow microphone in browser','PermissionDeniedError':'Mic access denied','NotFoundError':'No microphone found'}};
            status.textContent = msgs[err.name] || ('Mic error: ' + err.message);
            status.className = 'error';
            setTimeout(() => {{ status.textContent = '🎙️ Tap the mic to consult Medibot by voice'; status.className = ''; }}, 5000);
        }}
    }}

    btn.addEventListener('click', function() {{
        if (recording) {{
            if (mediaRecorder && mediaRecorder.state !== 'inactive') mediaRecorder.stop();
            setRec(false);
            status.textContent = 'Processing…';
            status.className = 'active';
        }} else {{ startRec(); }}
    }});
}})();
</script>
        """,
        height=70,
    )


def render_page_styles():
    # Preconnect + non-blocking font load (faster than @import inside <style>)
    st.markdown(
        """<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700;800;900&family=DM+Sans:opsz,wght@9..40,400;9..40,500;9..40,600;9..40,700&family=JetBrains+Mono:wght@400;500&display=swap">""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
<style>
/* ── Design Tokens ────────────────────────── */
:root {
    --md-primary:         #7c5cbf;
    --md-primary-rgb:     124, 92, 191;
    --md-secondary:       #00857a;
    --md-secondary-rgb:   0, 133, 122;
    --md-tertiary:        #d97706;
    --md-tertiary-rgb:    217, 119, 6;
    --md-error:           #e53935;
    --md-success:         #22c55e;
    --md-success-rgb:     34, 197, 94;

    --md-bg:                    rgba(255,255,255,0.018);
    --md-surface:               rgba(255,255,255,0.045);
    --md-surface-container:     rgba(255,255,255,0.07);
    --md-surface-container-high:rgba(255,255,255,0.10);
    --md-outline:               rgba(148,163,184,0.22);
    --md-outline-variant:       rgba(148,163,184,0.14);
    --md-soft:                  rgba(203,213,225,0.90);

    --md-shadow-1: 0 4px 16px rgba(10,15,30,0.10);
    --md-shadow-2: 0 12px 36px rgba(10,15,30,0.16);
    --md-shadow-3: 0 24px 64px rgba(10,15,30,0.22);

    --font-display: 'Sora', sans-serif;
    --font-body:    'DM Sans', sans-serif;
    --font-mono:    'JetBrains Mono', monospace;

    --radius-sm:  12px;
    --radius-md:  20px;
    --radius-lg:  28px;
    --radius-xl:  36px;
    --radius-pill:999px;
}

/* ── Base ─────────────────────────────────── */
html { scroll-behavior: smooth; }

body, .stApp,
[data-testid="stAppViewContainer"] {
    font-family: var(--font-body) !important;
}

h1,h2,h3,h4,h5,h6,
.md-title, .sb-title, .md-section-title {
    font-family: var(--font-display) !important;
}

code, pre, .font-mono {
    font-family: var(--font-mono) !important;
}

.block-container {
    max-width: 1220px;
    padding-top: 1rem;
    padding-bottom: 2.5rem;
}

/* ── Scrollbar ────────────────────────────── */
::-webkit-scrollbar            { width: 6px; height: 6px; }
::-webkit-scrollbar-track      { background: transparent; }
::-webkit-scrollbar-thumb      { background: rgba(var(--md-primary-rgb),.35); border-radius: 99px; }
::-webkit-scrollbar-thumb:hover{ background: rgba(var(--md-primary-rgb),.6); }

/* ─────────────────────────────────────────── */
/*  SIDEBAR  — MD3 Expressive                  */
/* ─────────────────────────────────────────── */
[data-testid="stSidebar"] {
    border-right: 1px solid var(--md-outline);
    background:
        radial-gradient(ellipse at 18% 4%,  rgba(var(--md-primary-rgb),.28)   0%, transparent 42%),
        radial-gradient(ellipse at 90% 44%, rgba(var(--md-secondary-rgb),.18) 0%, transparent 42%),
        radial-gradient(ellipse at 48% 92%, rgba(var(--md-tertiary-rgb),.11)  0%, transparent 42%),
        linear-gradient(180deg,
            rgba(var(--md-primary-rgb),.12) 0%,
            rgba(var(--md-secondary-rgb),.05) 55%,
            transparent 100%);
    overflow-x: hidden !important;
    transition: background 300ms ease;
}
[data-testid="stSidebarContent"] {
    overflow-x: hidden !important;
    padding-bottom: 24px !important;
}

/* ── Responsive ──────────────────────────── */
@media (max-width: 768px) {
    [data-testid="stSidebar"] { background: #0f0d1a !important; }

    .sb-top { flex-direction: row; align-items: center; gap: 10px; }
    .sb-logo, .sb-logo-fallback {
        width: 44px; height: 44px; min-width: 44px; border-radius: 13px;
    }
    .sb-title { font-size: 17px; word-break: break-word; overflow-wrap: break-word; }
    .sb-description { font-size: 11.5px; word-break: break-word; overflow-wrap: break-word; }
    .sb-stats-row { grid-template-columns: repeat(2, 1fr) !important; gap: 6px; }
    .sb-stat-number { font-size: 16px; }
    .sb-chip { gap: 8px; padding: 9px 10px; transform: none !important; }
    .sb-chip:hover { transform: none !important; }
    .sb-chip-icon { width: 30px; height: 30px; min-width: 30px; border-radius: 10px; font-size: 14px; }
    .sb-chip-value { font-size: 12.5px; word-break: break-word; overflow-wrap: break-word; }
    .sb-tip { font-size: 11px; padding: 9px 10px; }
    .sb-note { font-size: 11.5px; padding: 10px 12px; }
    .sb-status { font-size: 10px; padding: 4px 9px; }
    .sb-section-divider { margin: 10px 0 6px; }
}

/* ── Profile card ────────────────────────── */
.sb-profile-card {
    border: 1px solid rgba(var(--md-primary-rgb),.20);
    border-radius: 22px;
    padding: 18px 16px 16px;
    margin-bottom: 14px;
    background:
        radial-gradient(ellipse at 15% 8%,  rgba(var(--md-primary-rgb),.14), transparent 46%),
        radial-gradient(ellipse at 88% 88%, rgba(var(--md-secondary-rgb),.08), transparent 46%),
        var(--md-surface);
    box-shadow: var(--md-shadow-1), inset 0 1px 0 rgba(255,255,255,.07);
    overflow: hidden;
    position: relative;
}
.sb-profile-card::before {
    content:'';
    position:absolute; inset:0; border-radius:inherit;
    background: linear-gradient(140deg, rgba(255,255,255,.04) 0%, transparent 55%);
    pointer-events:none;
}
/* subtle top-edge accent — no animation */
.sb-profile-card::after {
    content:'';
    position:absolute; top:0; left:0; right:0; height:2px; border-radius:22px 22px 0 0;
    background: linear-gradient(90deg,
        rgba(var(--md-primary-rgb),.55) 0%,
        rgba(var(--md-secondary-rgb),.38) 50%,
        rgba(var(--md-primary-rgb),.55) 100%);
    pointer-events:none;
}

.sb-top { display:flex; gap:13px; align-items:center; margin-bottom:12px; }

@keyframes sb-neural-float {
    0%,100% { transform: translateY(0px) scale(1);    filter: brightness(1) drop-shadow(0 0 6px rgba(var(--md-primary-rgb),0.30)); }
    30%     { transform: translateY(-5px) scale(1.05); filter: brightness(1.15) drop-shadow(0 0 14px rgba(var(--md-primary-rgb),0.55)); }
    60%     { transform: translateY(-2px) scale(1.02); filter: brightness(1.08) drop-shadow(0 0 9px rgba(var(--md-primary-rgb),0.40)); }
}
@keyframes sb-online-mb { 0%,100%{box-shadow:0 0 6px rgba(0,200,83,0.55);} 50%{box-shadow:0 0 10px rgba(0,200,83,0.28);} }

.sb-kicker {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 3px 10px; border-radius: 999px;
    background: linear-gradient(90deg, rgba(var(--md-primary-rgb),.22), rgba(var(--md-secondary-rgb),.15));
    border: 1px solid rgba(var(--md-primary-rgb),.38);
    color: #a78bfa; font-size: 9px; font-weight: 900;
    letter-spacing: 0.10em; text-transform: uppercase;
    margin-bottom: 5px; display: inline-flex;
    box-shadow: 0 1px 6px rgba(var(--md-primary-rgb),.12);
}

.sb-logo, .sb-logo-fallback {
    width:54px; height:54px; min-width:54px;
    border-radius:18px; object-fit:contain; padding:8px; box-sizing:border-box;
    background:
        linear-gradient(135deg, rgba(255,255,255,.12), rgba(255,255,255,.04)),
        var(--md-surface-container);
    border:1.5px solid rgba(var(--md-primary-rgb),.28);
    box-shadow:
        var(--md-shadow-2),
        inset 0 1px 0 rgba(255,255,255,.12),
        0 0 18px rgba(var(--md-primary-rgb),.18);
    transform-origin: center;
    animation: sb-neural-float 3.2s ease-in-out infinite;
    transition: box-shadow 200ms ease;
    position: relative;
}
.sb-logo:hover, .sb-logo-fallback:hover {
    box-shadow: var(--md-shadow-3), 0 0 28px rgba(var(--md-primary-rgb),.32);
}
.sb-logo-fallback {
    display:flex; align-items:center; justify-content:center; font-size:22px;
    position: relative;
}
.sb-logo-fallback::after, .sb-logo::after {
    content: '';
    position: absolute; bottom: -3px; right: -3px;
    width: 13px; height: 13px; border-radius: 50%;
    background: #00c853;
    border: 2px solid var(--md-bg, #0d0d14);
    box-shadow: 0 0 6px rgba(0,200,83,0.55);
    z-index: 10;
    animation: sb-online-mb 2.4s ease-in-out infinite;
}

.sb-title {
    font-size:24px; font-weight:900; line-height:1.04;
    font-family:var(--font-display);
    background: linear-gradient(115deg, #e2d4ff 0%, #a78bfa 55%, #5eead4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -.01em;
}

/* Status badge */
.sb-status {
    position:relative; display:inline-flex; align-items:center; gap:7px;
    margin-top:7px; padding:5px 13px;
    border-radius:var(--radius-pill);
    border:1px solid rgba(var(--md-success-rgb),.32);
    background:
        radial-gradient(circle at 16% 50%, rgba(var(--md-success-rgb),.22), transparent 42%),
        rgba(var(--md-success-rgb),.07);
    color: #86efac; font-size:11px; font-weight:900; letter-spacing:.05em; overflow:hidden;
}
.sb-status::after {
    content:''; position:absolute; inset:-1px; border-radius:inherit;
    background: linear-gradient(115deg,
        transparent 0%,
        rgba(134,239,172,.00) 28%,
        rgba(134,239,172,.30) 50%,
        rgba(20,184,166,.22) 62%,
        transparent 80%);
    transform:translateX(-110%);
    animation: sb-shimmer 3.4s ease-in-out infinite;
    pointer-events:none;
}
.sb-status-dot {
    position:relative; width:8px; height:8px; border-radius:999px; background:#22c55e;
    box-shadow: 0 0 6px rgba(34,197,94,.60), 0 0 14px rgba(34,197,94,.32);
    animation: sb-breathe 2s ease-in-out infinite;
}
.sb-status-dot::before {
    content:''; position:absolute; inset:-8px; border-radius:999px;
    border:1px solid rgba(34,197,94,.42);
    animation: sb-ring 2s ease-out infinite;
}
@keyframes sb-breathe { 0%,100%{transform:scale(1);filter:brightness(1)} 50%{transform:scale(1.20);filter:brightness(1.40)} }
@keyframes sb-ring    { 0%{transform:scale(.5);opacity:.9} 80%,100%{transform:scale(1.6);opacity:0} }
@keyframes sb-shimmer { 0%,40%{transform:translateX(-110%)} 70%,100%{transform:translateX(110%)} }

.sb-description {
    color:var(--md-soft); font-size:12.5px; line-height:1.6;
    border-top: 1px solid rgba(var(--md-primary-rgb),.12);
    padding-top: 11px; margin-top: 2px;
}

/* ── Section divider / header ────────────── */
.sb-section {
    display: flex; align-items: center; gap: 8px;
    margin: 18px 0 8px; font-size:10.5px; font-weight:900;
    color: rgba(var(--md-primary-rgb),1); text-transform:uppercase; letter-spacing:.10em;
    font-family:var(--font-display);
}
.sb-section::after {
    content:''; flex:1; height:1px;
    background: linear-gradient(90deg,
        rgba(var(--md-primary-rgb),.30) 0%,
        transparent 100%);
}

/* ── Stat cards ──────────────────────────── */
.sb-stats-row {
    display:grid; grid-template-columns:repeat(4,1fr); gap:7px; margin:10px 0;
}
.sb-stat-card {
    border:1px solid rgba(var(--md-primary-rgb),.16);
    border-radius:18px; padding:11px 6px 9px;
    background:
        radial-gradient(ellipse at 50% 0%, rgba(var(--md-primary-rgb),.14), transparent 70%),
        var(--md-surface);
    text-align:center;
    transition: transform 140ms ease, background 140ms ease, box-shadow 140ms ease, border-color 140ms ease;
    position:relative; overflow:hidden;
}
.sb-stat-card::before {
    content:''; position:absolute; top:0; left:10%; right:10%; height:2px;
    background: linear-gradient(90deg, transparent, rgba(var(--md-primary-rgb),.6), transparent);
    border-radius: 0 0 4px 4px;
}
.sb-stat-card:hover {
    transform:translateY(-3px);
    background:
        radial-gradient(ellipse at 50% 0%, rgba(var(--md-primary-rgb),.24), transparent 70%),
        var(--md-surface-container);
    box-shadow: var(--md-shadow-2), 0 0 14px rgba(var(--md-primary-rgb),.16);
    border-color: rgba(var(--md-primary-rgb),.32);
}
.sb-stat-number {
    font-size:20px; font-weight:900; font-family:var(--font-display); line-height:1;
    background: linear-gradient(115deg, #e2d4ff, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.sb-stat-label {
    font-size:9.5px; color:var(--md-soft); margin-top:4px;
    font-weight:700; text-transform:uppercase; letter-spacing:.06em;
}

/* ── System info chips ───────────────────── */
.sb-chip {
    display:flex; gap:11px; align-items:center;
    padding:11px 13px; border-radius:18px;
    border:1px solid rgba(var(--md-primary-rgb),.14);
    background:
        linear-gradient(105deg,
            rgba(var(--md-primary-rgb),.08) 0%,
            rgba(var(--md-secondary-rgb),.05) 100%),
        var(--md-surface);
    margin-bottom:7px;
    transition: transform 140ms ease, background 140ms ease, border-color 140ms ease, box-shadow 140ms ease;
    position:relative; overflow:hidden;
}
.sb-chip::before {
    content:''; position:absolute; left:0; top:15%; bottom:15%; width:2px;
    background: linear-gradient(180deg, rgba(var(--md-primary-rgb),.8), rgba(var(--md-secondary-rgb),.6));
    border-radius: 0 2px 2px 0;
    opacity:0; transition:opacity 140ms ease;
}
.sb-chip:hover {
    transform:translateX(5px);
    background:
        linear-gradient(105deg,
            rgba(var(--md-primary-rgb),.16) 0%,
            rgba(var(--md-secondary-rgb),.10) 100%),
        var(--md-surface-container);
    border-color:rgba(var(--md-primary-rgb),.34);
    box-shadow: var(--md-shadow-1), 0 0 12px rgba(var(--md-primary-rgb),.12);
}
.sb-chip:hover::before { opacity:1; }

.sb-chip-icon {
    width:38px; height:38px; min-width:38px; border-radius:14px;
    display:flex; align-items:center; justify-content:center;
    background: linear-gradient(135deg,
        rgba(var(--md-primary-rgb),.28),
        rgba(var(--md-secondary-rgb),.18));
    border:1px solid rgba(var(--md-primary-rgb),.20);
    font-size:17px;
    box-shadow: 0 4px 12px rgba(var(--md-primary-rgb),.18);
}
.sb-chip-label {
    color:var(--md-soft); font-size:10.5px; font-weight:800;
    text-transform:uppercase; letter-spacing:.06em;
}
.sb-chip-value {
    font-size:13.5px; font-weight:800; line-height:1.25;
    font-family:var(--font-display);
}

/* ── Note / warning ─────────────────────── */
.sb-note {
    border:1px solid rgba(217,119,6,.32); border-radius:20px; padding:13px 14px;
    background:
        radial-gradient(ellipse at 10% 50%, rgba(217,119,6,.10), transparent 55%),
        rgba(217,119,6,.05);
    color:var(--md-soft); font-size:12.5px; line-height:1.55; margin-top:12px;
    position:relative; overflow:hidden;
}
.sb-note::before {
    content:''; position:absolute; left:0; top:0; bottom:0; width:3px;
    background: linear-gradient(180deg, #f59e0b, #d97706);
    border-radius:20px 0 0 20px;
}

/* ── Daily tip card (MD3 Expressive) ─────── */
.sb-tip {
    border:1.5px solid rgba(var(--md-secondary-rgb),.28);
    border-radius:22px;
    padding:14px 14px 14px 12px;
    background:
        radial-gradient(ellipse at 6% 40%,  rgba(var(--md-secondary-rgb),.18), transparent 55%),
        radial-gradient(ellipse at 92% 80%, rgba(var(--md-primary-rgb),.12),   transparent 50%),
        rgba(var(--md-secondary-rgb),.04);
    margin-top:8px;
    display:flex; gap:12px; align-items:flex-start;
    position:relative; overflow:hidden;
    box-shadow: 0 4px 20px rgba(var(--md-secondary-rgb),.10), inset 0 1px 0 rgba(255,255,255,.06);
    transition: box-shadow 180ms ease, border-color 180ms ease;
}
.sb-tip:hover {
    border-color: rgba(var(--md-secondary-rgb),.45);
    box-shadow: 0 6px 26px rgba(var(--md-secondary-rgb),.18), inset 0 1px 0 rgba(255,255,255,.08);
}
.sb-tip::before {
    content:''; position:absolute; left:0; top:0; bottom:0; width:3px;
    background: linear-gradient(180deg,
        rgba(var(--md-secondary-rgb),1) 0%,
        rgba(var(--md-primary-rgb),.7) 100%);
    border-radius:22px 0 0 22px;
}

/* Number column */
.sb-tip-num-col {
    display:flex; flex-direction:column; align-items:center;
    gap:4px; flex-shrink:0; padding-top:1px;
}

/* Arc ring */
.sb-tip-num-ring {
    position:relative; width:40px; height:40px; flex-shrink:0;
}
.sb-tip-arc-svg {
    width:40px; height:40px; overflow:visible;
}
.sb-tip-arc-bg {
    fill:none;
    stroke: rgba(var(--md-secondary-rgb),.18);
    stroke-width:4;
}
.sb-tip-arc-fill {
    fill:none;
    stroke-width:4;
    stroke-linecap:round;
}
.sb-tip-num-inner {
    position:absolute; inset:0;
    display:flex; align-items:center; justify-content:center;
}
.sb-tip-num-icon {
    font-size:16px; line-height:1;
    filter: drop-shadow(0 0 5px rgba(var(--md-secondary-rgb),.70));
}

/* Counter  e.g. "3/20" */
.sb-tip-counter {
    font-size:11px; font-weight:900;
    font-family:var(--font-display);
    background: linear-gradient(115deg, #5eead4, #a78bfa);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
    background-clip:text;
    line-height:1; letter-spacing:-.01em;
}
.sb-tip-total {
    font-size:9px; font-weight:700; opacity:.65;
}

/* Text area */
.sb-tip-content { flex:1; min-width:0; }
.sb-tip-title {
    font-size:12px; font-weight:900; line-height:1.2;
    font-family:var(--font-display);
    background: linear-gradient(115deg, #5eead4 0%, #a78bfa 100%);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
    background-clip:text;
    margin-bottom:5px; letter-spacing:.01em;
}
.sb-tip-body {
    font-size:11.5px; line-height:1.6;
    color:var(--md-soft);
}

/* ── Footer ─────────────────────────────── */
.sb-footer {
    text-align:center; color:var(--md-soft); font-size:11.5px; margin-top:16px;
    padding: 10px 0 4px;
    border-top: 1px solid rgba(var(--md-primary-rgb),.10);
}

/* ── Next Insight wrapper ────────────────── */
.sb-next-tip-wrap {
    margin-top: 8px;
}
.sb-next-tip-wrap .stButton > button {
    min-height: 34px !important;
    height: 34px !important;
    padding: 0 14px !important;
    font-size: 11px !important;
    font-weight: 800 !important;
    font-family: var(--font-display) !important;
    letter-spacing: .03em !important;
    border-radius: var(--radius-pill) !important;
    border: 1px solid rgba(var(--md-secondary-rgb),.45) !important;
    background: linear-gradient(135deg,
        rgba(var(--md-secondary-rgb),.16) 0%,
        rgba(var(--md-primary-rgb),.12) 100%) !important;
    background-size: 100% 100% !important;
    color: #5eead4 !important;
    box-shadow: none !important;
    transform: none !important;
    transition: transform 140ms ease, background 140ms ease,
                box-shadow 140ms ease, border-color 140ms ease !important;
}
.sb-next-tip-wrap .stButton > button:hover {
    transform: translateY(-1px) !important;
    background: linear-gradient(135deg,
        rgba(var(--md-secondary-rgb),.28) 0%,
        rgba(var(--md-primary-rgb),.20) 100%) !important;
    box-shadow: 0 3px 12px rgba(var(--md-secondary-rgb),.22) !important;
    color: white !important;
    border-color: rgba(var(--md-secondary-rgb),.70) !important;
}
.sb-next-tip-wrap .stButton > button:active {
    transform: translateY(0) !important;
}

/* ── Footer creator name ─────────────────── */
.sb-creator {
    font-weight: 800;
    font-family: var(--font-display);
    background: linear-gradient(115deg, #a78bfa 0%, #5eead4 55%, #c084fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    background-size: 200% auto;
    animation: sb-creator-shift 5s linear infinite alternate;
    letter-spacing: .01em;
}
@keyframes sb-creator-shift {
    0%   { background-position: 0%   center; }
    100% { background-position: 100% center; }
}
[data-theme="light"] .sb-creator {
    background: linear-gradient(115deg, #6750a4 0%, #0d7b72 55%, #7c3aed 100%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
}

/* ── Next Insight button (small pill inside sidebar) ── */
/* NOTE: this block intentionally placed AFTER the generic .stButton rule below */

/* ── Sidebar CTA button (generic — applies to all sidebar buttons first) ── */
[data-testid="stSidebar"] .stButton > button {
    border-radius:var(--radius-pill) !important;
    min-height:48px; font-weight:800 !important;
    border:1.5px solid rgba(var(--md-primary-rgb),.40) !important;
    background: linear-gradient(135deg, #6750a4 0%, #7c3aed 55%, #0d9488 100%) !important;
    background-size: 200% 100% !important;
    color:white !important;
    box-shadow:
        var(--md-shadow-2),
        0 0 20px rgba(var(--md-primary-rgb),.28),
        inset 0 1px 0 rgba(255,255,255,.15) !important;
    font-family:var(--font-display) !important;
    letter-spacing:.02em;
    transition:
        transform 150ms ease,
        box-shadow 150ms ease,
        background-position 300ms ease !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    transform:translateY(-2px) scale(1.01) !important;
    box-shadow:
        var(--md-shadow-3),
        0 0 32px rgba(var(--md-primary-rgb),.40),
        inset 0 1px 0 rgba(255,255,255,.20) !important;
    background-position: 100% 0 !important;
}
[data-testid="stSidebar"] .stButton > button:active {
    transform:translateY(0) scale(0.99) !important;
}

/* ── Next Insight override — comes AFTER generic rule to win specificity ── */
[data-testid="stSidebar"] [data-testid="column"] + [data-testid="column"] .stButton > button,
[data-testid="stSidebar"] button[key="next_tip_btn"],
[data-testid="stSidebar"] .sb-next-tip-wrap .stButton > button {
    min-height: 32px !important;
    height: 32px !important;
    padding: 0 10px !important;
    font-size: 10.5px !important;
    border: 1px solid rgba(var(--md-secondary-rgb),.45) !important;
    background: linear-gradient(135deg,
        rgba(var(--md-secondary-rgb),.18) 0%,
        rgba(var(--md-primary-rgb),.14) 100%) !important;
    background-size: 100% 100% !important;
    color: #5eead4 !important;
    box-shadow: none !important;
    transform: none !important;
}
[data-testid="stSidebar"] [data-testid="column"] + [data-testid="column"] .stButton > button:hover {
    background: linear-gradient(135deg,
        rgba(var(--md-secondary-rgb),.30) 0%,
        rgba(var(--md-primary-rgb),.22) 100%) !important;
    box-shadow: 0 3px 12px rgba(var(--md-secondary-rgb),.25) !important;
    color: white !important;
    border-color: rgba(var(--md-secondary-rgb),.70) !important;
    transform: translateY(-1px) !important;
}

/* ─────────────────────────────────────────── */
/*  HERO                                       */
/* ─────────────────────────────────────────── */
.md-hero {
    overflow:hidden; border:1px solid var(--md-outline);
    border-radius:var(--radius-xl); padding:30px 34px; margin:10px 0 18px 0;
    background:
        radial-gradient(ellipse at 10% 10%, rgba(var(--md-primary-rgb),.20), transparent 40%),
        radial-gradient(ellipse at 88% 15%, rgba(var(--md-secondary-rgb),.14), transparent 40%),
        radial-gradient(ellipse at 55% 90%, rgba(var(--md-tertiary-rgb),.08), transparent 40%),
        linear-gradient(135deg, rgba(var(--md-primary-rgb),.12), rgba(var(--md-secondary-rgb),.07) 58%, rgba(var(--md-tertiary-rgb),.04)),
        var(--md-surface);
    box-shadow:var(--md-shadow-3);
    position:relative;
}
.md-hero::before {
    content:'';
    position:absolute; inset:0; border-radius:inherit;
    background: linear-gradient(135deg, rgba(255,255,255,.05) 0%, transparent 55%);
    pointer-events:none;
}

/* Animated mesh orbs */
.md-hero-orb {
    position:absolute; border-radius:50%; filter:blur(60px); pointer-events:none; opacity:.35;
}
.md-hero-orb-1 {
    width:260px; height:260px; top:-80px; left:-60px;
    background:radial-gradient(circle, rgba(var(--md-primary-rgb),1), transparent);
    animation: orb-float 7s ease-in-out infinite;
}
.md-hero-orb-2 {
    width:200px; height:200px; bottom:-60px; right:60px;
    background:radial-gradient(circle, rgba(var(--md-secondary-rgb),1), transparent);
    animation: orb-float 9s ease-in-out infinite reverse;
}
@keyframes orb-float {
    0%,100% { transform:translate(0,0) scale(1); }
    33%      { transform:translate(20px,-12px) scale(1.06); }
    66%      { transform:translate(-14px,10px) scale(.96); }
}

.md-hero-inner {
    display:grid; grid-template-columns:minmax(0,1fr) 270px;
    gap:28px; align-items:center; position:relative; z-index:1;
}

.md-brand { display:flex; align-items:center; gap:18px; min-width:0; }

.md-logo, .md-logo-fallback {
    width:90px; height:90px; min-width:90px;
    border-radius:26px; object-fit:contain; padding:9px; box-sizing:border-box;
    background:
        linear-gradient(135deg,rgba(255,255,255,.10),rgba(255,255,255,.035)),
        var(--md-surface-container);
    border:1px solid var(--md-outline-variant);
    box-shadow:var(--md-shadow-2), inset 0 1px 0 rgba(255,255,255,.08);
}
.md-logo-fallback { display:flex; align-items:center; justify-content:center; font-size:42px; }

.md-kicker {
    display:inline-flex; width:fit-content;
    padding:6px 13px; border-radius:var(--radius-pill);
    border:1px solid rgba(var(--md-primary-rgb),.32);
    background:rgba(var(--md-primary-rgb),.12); color:#d7c7ff;
    font-size:11px; font-weight:900; margin-bottom:10px;
    letter-spacing:.05em; text-transform:uppercase; font-family:var(--font-display);
}

.md-title {
    margin:0; font-size:clamp(36px,4.8vw,62px); line-height:.97;
    font-weight:980; letter-spacing:-.02em; font-family:var(--font-display);
}

.md-subtitle {
    max-width:700px; margin-top:13px; color:var(--md-soft);
    font-size:15.5px; line-height:1.65;
}

/* Hero metrics */
.md-status { display:grid; gap:9px; }
.md-pill {
    border:1px solid var(--md-outline); border-radius:var(--radius-md);
    padding:13px 15px; background:var(--md-surface-container);
    box-shadow:var(--md-shadow-1);
    transition:transform 130ms ease, background 130ms ease;
}
.md-pill:hover { transform:translateY(-2px); background:var(--md-surface-container-high); }
.md-pill-label { font-size:11px; color:var(--md-soft); margin-bottom:4px; font-weight:700; text-transform:uppercase; letter-spacing:.05em; }
.md-pill-value { font-weight:900; font-size:14.5px; font-family:var(--font-display); }

/* Hero quick stat bar */
.md-hero-stats {
    display:flex; gap:10px; flex-wrap:wrap; margin-top:16px;
}
.md-hero-stat {
    display:flex; gap:7px; align-items:center;
    padding:6px 12px; border-radius:var(--radius-pill);
    border:1px solid var(--md-outline-variant);
    background:var(--md-surface); font-size:12.5px; font-weight:700;
    color:var(--md-soft);
}
.md-hero-stat-dot {
    width:6px; height:6px; border-radius:50%;
    background:var(--md-success); flex-shrink:0;
    box-shadow:0 0 6px rgba(34,197,94,.8);
}

/* ─────────────────────────────────────────── */
/*  FEATURE CARDS                              */
/* ─────────────────────────────────────────── */
.md-card-grid {
    display:grid; grid-template-columns:repeat(3,minmax(0,1fr));
    gap:13px; margin:8px 0 20px 0;
}
.md-info-card {
    border:1px solid var(--md-outline); border-radius:var(--radius-lg);
    padding:18px; background:var(--md-surface);
    box-shadow:var(--md-shadow-1); min-height:136px;
    transition:transform 150ms ease, box-shadow 150ms ease, background 150ms ease, border-color 150ms ease;
    position:relative; overflow:hidden;
}
.md-info-card::before {
    content:''; position:absolute; inset:0;
    background:linear-gradient(135deg, rgba(255,255,255,.04) 0%, transparent 60%);
    pointer-events:none;
}
.md-info-card:hover {
    transform:translateY(-3px);
    background:var(--md-surface-container);
    box-shadow:var(--md-shadow-2);
    border-color:rgba(var(--md-primary-rgb),.25);
}
.md-card-icon {
    width:44px; height:44px; border-radius:17px; margin-bottom:12px;
    background:linear-gradient(135deg,rgba(var(--md-primary-rgb),.22),rgba(var(--md-secondary-rgb),.14));
    display:flex; align-items:center; justify-content:center;
    font-size:20px;
}
.md-info-card strong {
    display:block; font-size:14.5px; margin-bottom:7px;
    font-family:var(--font-display); font-weight:800;
}
.md-info-card span { display:block; color:var(--md-soft); line-height:1.55; font-size:13.5px; }

/* ─────────────────────────────────────────── */
/*  HEALTH TOPICS QUICK ACCESS                 */
/* ─────────────────────────────────────────── */
.md-topics-section { margin: 0 0 20px 0; }

/* Card is pure HTML — the st.button below it is just a slim "Select" trigger */
.md-topic-card {
    border:1px solid rgba(148,163,184,0.18);
    border-radius:var(--radius-lg) var(--radius-lg) 0 0;
    padding:14px 8px 10px 8px;
    background:rgba(255,255,255,0.045);
    text-align:center;
    transition:transform 140ms ease, background 140ms ease,
               border-color 140ms ease, box-shadow 140ms ease;
    position:relative; overflow:hidden;
    display:flex; flex-direction:column; align-items:center; justify-content:center;
    min-height:76px;
    cursor:pointer;
}
.md-topic-card:hover {
    background:rgba(255,255,255,0.07);
    border-color:rgba(124,92,191,0.30);
}
.md-topic-emoji { font-size:22px; line-height:1; display:block; margin-bottom:5px; }
.md-topic-label {
    font-size:11.5px; font-weight:800;
    font-family:'Sora',sans-serif;
    line-height:1.25; color:rgba(203,213,225,0.92);
}
.md-topic-sub {
    font-size:10px; color:rgba(148,163,184,0.75);
    margin-top:2px; line-height:1.2;
}

/* The select button sits flush below the card */
.md-topic-btn .stButton > button {
    border-radius:0 0 var(--radius-lg) var(--radius-lg) !important;
    min-height:28px !important;
    height:28px !important;
    width:100% !important;
    padding:0 4px !important;
    margin-top:-1px !important;
    background:rgba(124,92,191,0.15) !important;
    background-image:none !important;
    border:1px solid rgba(148,163,184,0.18) !important;
    border-top:1px solid rgba(124,92,191,0.20) !important;
    color:rgba(203,213,225,0.85) !important;
    box-shadow:none !important;
    font-size:13px !important;
    font-weight:500 !important;
    font-family:'Sora',sans-serif !important;
    letter-spacing:0 !important;
    line-height:28px !important;
    white-space:nowrap !important;
    overflow:hidden !important;
    transition:background 130ms ease, color 130ms ease !important;
}
.md-topic-btn .stButton > button:hover {
    background:rgba(124,92,191,0.30) !important;
    background-image:none !important;
    color:white !important;
    transform:none !important;
    box-shadow:none !important;
}
.md-topic-btn .stButton > button p,
.md-topic-btn .stButton > button span,
.md-topic-btn .stButton > button > div,
.md-topic-btn .stButton > button > div > p {
    font-size:13px !important;
    font-weight:500 !important;
    letter-spacing:0 !important;
    margin:0 !important;
    line-height:28px !important;
    color:inherit !important;
    white-space:nowrap !important;
    overflow:hidden !important;
    text-overflow:clip !important;
}

/* ─────────────────────────────────────────── */
/*  SECTION TITLE                              */
/* ─────────────────────────────────────────── */
.md-section-title {
    margin:18px 0 11px 0; font-size:21px; font-weight:900;
    font-family:var(--font-display); letter-spacing:-.01em;
}
.md-section-sub {
    font-size:13.5px; color:var(--md-soft); margin-top:-6px; margin-bottom:12px;
}

/* ─────────────────────────────────────────── */
/*  EMPTY STATE                                */
/* ─────────────────────────────────────────── */
.md-empty {
    border:1px dashed var(--md-outline); border-radius:var(--radius-lg);
    padding:20px; color:var(--md-soft); line-height:1.65;
    background:var(--md-surface); margin-bottom:14px;
    font-size:14px;
}
.md-empty-icon { font-size:32px; display:block; margin-bottom:10px; }

/* ─────────────────────────────────────────── */
/*  CHAT ANSWER & SOURCES                      */
/* ─────────────────────────────────────────── */
.md-answer { line-height:1.7; font-size:15px; }

/* ═══════════════════════════════════════════ */
/*  MD3 EXPRESSIVE ANSWER CARD                 */
/* ═══════════════════════════════════════════ */
.ar-answer-card {
    position: relative;
    margin: 10px 0 6px 0;
    border-radius: 28px;
    border: 1.5px solid rgba(168, 85, 247, 0.30);
    background:
        radial-gradient(ellipse at 8% 12%,  rgba(139,92,246,.20) 0%, transparent 50%),
        radial-gradient(ellipse at 92% 88%, rgba(20,184,166,.16) 0%, transparent 50%),
        radial-gradient(ellipse at 55% 0%,  rgba(236,72,153,.09) 0%, transparent 45%),
        rgba(14, 8, 28, 0.78);
    box-shadow:
        0 8px 36px rgba(139,92,246,.20),
        0 2px 8px  rgba(0,0,0,.36),
        inset 0 1px 0 rgba(255,255,255,.10);
    overflow: hidden;
}
.ar-answer-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 4px;
    background: linear-gradient(90deg,
        #a855f7 0%, #ec4899 20%, #f59e0b 38%, #14b8a6 56%, #6366f1 74%, #a855f7 100%);
    background-size: 300% 100%;
    animation: ar-stripe 6s linear infinite;
    border-radius: 28px 28px 0 0;
}
@keyframes ar-stripe {
    0%   { background-position: 0%   center; }
    100% { background-position: 300% center; }
}
.ar-answer-body {
    padding: 24px 26px 22px;
    display: flex;
    flex-direction: column;
    gap: 4px;
}

/* ── H1 ───────────────────────────────────── */
.ar-h1 {
    font-family: var(--font-display);
    font-size: 21px;
    font-weight: 900;
    line-height: 1.2;
    margin: 20px 0 10px;
    letter-spacing: -.015em;
    background: linear-gradient(115deg, #f0abfc 0%, #a78bfa 35%, #34d399 68%, #7dd3fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    padding-bottom: 8px;
    border-bottom: 1.5px solid rgba(168,85,247,.25);
}
.ar-h1:first-child { margin-top: 0; }

/* ── H2 ───────────────────────────────────── */
.ar-h2 {
    font-family: var(--font-display);
    font-size: 15px;
    font-weight: 800;
    line-height: 1.35;
    margin: 18px 0 8px;
    padding: 9px 16px;
    border-radius: 14px;
    border-left: 4px solid #a855f7;
    background: linear-gradient(100deg,
        rgba(168,85,247,.20) 0%,
        rgba(99,102,241,.12) 55%,
        rgba(20,184,166,.07) 100%);
    color: #e4b4ff;
    letter-spacing: .005em;
    display: flex;
    align-items: center;
    gap: 8px;
    box-shadow: inset 0 0 0 1px rgba(168,85,247,.14);
}
.ar-h2::before {
    content: '◈';
    font-size: 13px;
    color: #c084fc;
    flex-shrink: 0;
    filter: drop-shadow(0 0 4px rgba(192,132,252,.7));
}
.ar-h2:first-child { margin-top: 0; }

/* ── H3 ───────────────────────────────────── */
.ar-h3 {
    font-family: var(--font-display);
    font-size: 11.5px;
    font-weight: 800;
    margin: 16px 0 6px;
    display: inline-flex;
    align-items: center;
    gap: 7px;
    color: #5eead4;
    text-transform: uppercase;
    letter-spacing: .10em;
    filter: drop-shadow(0 0 6px rgba(94,234,212,.4));
}
.ar-h3::before {
    content: '';
    display: inline-block;
    width: 10px; height: 10px;
    border-radius: 4px;
    background: linear-gradient(135deg, #a855f7, #14b8a6);
    flex-shrink: 0;
    box-shadow: 0 0 10px rgba(168,85,247,.75), 0 0 4px rgba(20,184,166,.55);
}
.ar-h3:first-child { margin-top: 0; }

/* ── Paragraphs ──────────────────────────── */
.ar-p {
    font-size: 14.5px;
    line-height: 1.85;
    color: rgba(226, 218, 248, 0.92);
    margin: 4px 0 10px;
    font-family: var(--font-body);
}
.ar-p:last-child { margin-bottom: 0; }

/* bold + key terms */
.ar-p strong, .ar-li strong, .ar-li-num strong {
    color: #f5d0fe;
    font-weight: 800;
    text-shadow: 0 0 12px rgba(240,171,252,.25);
}
.ar-kw {
    display: inline;
    vertical-align: baseline;
    font-weight: 800;
    color: #d8b4fe;
    text-decoration: underline;
    text-decoration-color: rgba(168,85,247,.60);
    text-underline-offset: 3px;
    text-decoration-thickness: 1.5px;
    word-spacing: normal;
    letter-spacing: normal;
    white-space: normal;
}

/* ── Bullet list ─────────────────────────── */
.ar-ul {
    list-style: none;
    padding: 0;
    margin: 6px 0 14px;
    display: flex;
    flex-direction: column;
    gap: 6px;
}
.ar-li {
    display: flex;
    align-items: flex-start;
    gap: 11px;
    font-size: 14px;
    line-height: 1.75;
    color: rgba(226, 218, 248, 0.90);
    padding: 10px 14px;
    border-radius: 16px;
    background: linear-gradient(105deg,
        rgba(139,92,246,.15) 0%,
        rgba(99,102,241,.09) 55%,
        rgba(20,184,166,.07) 100%);
    border: 1px solid rgba(139,92,246,.26);
    transition: background 150ms ease, border-color 150ms ease, transform 150ms ease;
    font-family: var(--font-body);
}
.ar-li:hover {
    background: linear-gradient(105deg,
        rgba(139,92,246,.26) 0%,
        rgba(99,102,241,.17) 55%,
        rgba(20,184,166,.11) 100%);
    border-color: rgba(168,85,247,.50);
    transform: translateX(4px);
}
.ar-li::before {
    content: '';
    display: block;
    min-width: 8px; height: 8px;
    margin-top: 7px;
    border-radius: 50%;
    background: linear-gradient(135deg, #a855f7, #14b8a6);
    flex-shrink: 0;
    box-shadow: 0 0 9px rgba(168,85,247,.80), 0 0 3px rgba(20,184,166,.55);
}

.ar-li-text {
    display: block;
    flex: 1;
    min-width: 0;
    word-break: break-word;
}

/* ── Numbered / Steps list ───────────────── */
.ar-ol {
    list-style: none;
    padding: 0;
    margin: 6px 0 14px;
    display: flex;
    flex-direction: column;
    gap: 7px;
}
.ar-li-num {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    font-size: 14px;
    line-height: 1.75;
    color: rgba(226, 218, 248, 0.90);
    padding: 10px 14px 10px 10px;
    border-radius: 16px;
    background: linear-gradient(105deg,
        rgba(20,184,166,.16) 0%,
        rgba(99,102,241,.10) 55%,
        rgba(236,72,153,.06) 100%);
    border: 1px solid rgba(20,184,166,.28);
    transition: background 150ms ease, transform 150ms ease;
    font-family: var(--font-body);
}
.ar-li-num:hover {
    background: linear-gradient(105deg,
        rgba(20,184,166,.26) 0%,
        rgba(99,102,241,.17) 55%,
        rgba(236,72,153,.10) 100%);
    border-color: rgba(94,234,212,.45);
    transform: translateX(4px);
}
.ar-li-num-badge {
    min-width: 26px; height: 26px;
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 11px; font-weight: 900;
    background: linear-gradient(135deg, #0d9488, #6366f1);
    color: white;
    font-family: var(--font-display);
    flex-shrink: 0;
    margin-top: 1px;
    box-shadow: 0 3px 14px rgba(13,148,136,.50), 0 1px 4px rgba(99,102,241,.40);
    letter-spacing: -.01em;
}

/* ── Callout / Note block ────────────────── */
.ar-callout {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    padding: 12px 16px;
    border-radius: 16px;
    background: linear-gradient(105deg,
        rgba(251,191,36,.14) 0%,
        rgba(239,68,68,.08) 100%);
    border: 1px solid rgba(251,191,36,.36);
    margin: 8px 0 12px;
    font-size: 13.5px;
    line-height: 1.65;
    color: rgba(254,240,138,.96);
    font-family: var(--font-body);
    box-shadow: inset 0 0 0 1px rgba(251,191,36,.10),
                0 2px 12px rgba(251,191,36,.10);
}
.ar-callout-icon {
    font-size: 16px;
    flex-shrink: 0;
    margin-top: 1px;
}

/* ── Inline code ─────────────────────────── */
.ar-code {
    font-family: var(--font-mono);
    font-size: 12.5px;
    padding: 2px 8px;
    border-radius: 8px;
    background: rgba(244, 63, 94, .14);
    border: 1px solid rgba(244, 63, 94, .30);
    color: #fda4af;
}

/* ── HR ──────────────────────────────────── */
.ar-hr {
    border: none;
    height: 1.5px;
    background: linear-gradient(90deg,
        transparent 0%,
        rgba(168,85,247,.55) 30%,
        rgba(20,184,166,.55) 70%,
        transparent 100%);
    margin: 18px 0;
    border-radius: 999px;
}

/* ── Light theme overrides ───────────────── */
[data-theme="light"] .ar-answer-card {
    background:
        radial-gradient(ellipse at 8% 12%,  rgba(124,58,237,.11) 0%, transparent 50%),
        radial-gradient(ellipse at 92% 88%, rgba(13,148,136,.09) 0%, transparent 50%),
        rgba(252,250,255,.95) !important;
    border-color: rgba(124,58,237,.26) !important;
    box-shadow: 0 6px 32px rgba(124,58,237,.14), 0 2px 6px rgba(0,0,0,.06) !important;
}
[data-theme="light"] .ar-h1 {
    background: linear-gradient(115deg, #7c3aed 0%, #a21caf 38%, #0d9488 70%, #0284c7 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    border-bottom-color: rgba(124,58,237,.22) !important;
}
[data-theme="light"] .ar-h2 {
    background: linear-gradient(100deg, rgba(124,58,237,.14) 0%, rgba(99,102,241,.09) 55%, rgba(13,148,136,.06) 100%) !important;
    border-left-color: #7c3aed !important;
    color: #5b21b6 !important;
    box-shadow: inset 0 0 0 1px rgba(124,58,237,.12) !important;
}
[data-theme="light"] .ar-h2::before { color: #7c3aed !important; filter: none !important; }
[data-theme="light"] .ar-h3 { color: #0d7a6e !important; filter: none !important; }
[data-theme="light"] .ar-h3::before { box-shadow: 0 0 8px rgba(124,58,237,.35), 0 0 3px rgba(13,148,136,.28) !important; }
[data-theme="light"] .ar-p  { color: #1c1033 !important; }
[data-theme="light"] .ar-p strong,
[data-theme="light"] .ar-li strong,
[data-theme="light"] .ar-li-num strong { color: #6d28d9 !important; text-shadow: none !important; }
[data-theme="light"] .ar-kw { display: inline !important; vertical-align: baseline !important; color: #6d28d9 !important; text-decoration-color: rgba(124,58,237,.55) !important; word-spacing: normal !important; white-space: normal !important; }
[data-theme="light"] .ar-li {
    color: #1c1033 !important;
    background: linear-gradient(105deg, rgba(124,58,237,.11) 0%, rgba(99,102,241,.07) 55%, rgba(13,148,136,.05) 100%) !important;
    border-color: rgba(124,58,237,.22) !important;
}
[data-theme="light"] .ar-li:hover {
    background: linear-gradient(105deg, rgba(124,58,237,.19) 0%, rgba(99,102,241,.13) 55%, rgba(13,148,136,.09) 100%) !important;
    border-color: rgba(124,58,237,.40) !important;
}
[data-theme="light"] .ar-li::before {
    background: linear-gradient(135deg, #7c3aed, #0d9488) !important;
    box-shadow: 0 0 8px rgba(124,58,237,.45) !important;
}
[data-theme="light"] .ar-li-num {
    color: #1c1033 !important;
    background: linear-gradient(105deg, rgba(13,148,136,.13) 0%, rgba(99,102,241,.08) 55%, rgba(168,85,247,.05) 100%) !important;
    border-color: rgba(13,148,136,.26) !important;
}
[data-theme="light"] .ar-li-num:hover {
    background: linear-gradient(105deg, rgba(13,148,136,.22) 0%, rgba(99,102,241,.14) 55%, rgba(168,85,247,.08) 100%) !important;
    border-color: rgba(13,148,136,.42) !important;
}
[data-theme="light"] .ar-li-num-badge {
    background: linear-gradient(135deg, #0d9488, #6d28d9) !important;
    box-shadow: 0 3px 10px rgba(13,148,136,.38), 0 1px 3px rgba(109,40,217,.28) !important;
}
[data-theme="light"] .ar-callout {
    color: #713f12 !important;
    background: linear-gradient(105deg, rgba(217,119,6,.13) 0%, rgba(239,68,68,.07) 100%) !important;
    border-color: rgba(217,119,6,.36) !important;
    box-shadow: none !important;
}
[data-theme="light"] .ar-code {
    background: rgba(190,18,60,.09) !important;
    border-color: rgba(190,18,60,.24) !important;
    color: #9f1239 !important;
}
[data-theme="light"] .ar-hr {
    background: linear-gradient(90deg, transparent, rgba(124,58,237,.45) 30%, rgba(13,148,136,.45) 70%, transparent) !important;
}

.md-source-list {
    margin-top:14px; border:1px solid var(--md-outline);
    border-radius:var(--radius-lg); padding:15px;
    background:
        linear-gradient(135deg,rgba(var(--md-primary-rgb),.09),rgba(var(--md-secondary-rgb),.05)),
        var(--md-surface);
}
.md-source-title { font-weight:900; margin-bottom:10px; font-size:13px; text-transform:uppercase; letter-spacing:.06em; font-family:var(--font-display); }
.md-source-item {
    display:flex; gap:10px; align-items:flex-start;
    padding:10px 11px; border-radius:var(--radius-md);
    background:rgba(255,255,255,.04);
    border:1px solid var(--md-outline-variant); margin-top:7px;
    transition:background 120ms ease;
}
.md-source-item:hover { background:rgba(255,255,255,.07); }
.md-source-number {
    width:28px; height:28px; min-width:28px; border-radius:11px;
    display:flex; align-items:center; justify-content:center;
    color:white; font-size:12px; font-weight:900;
    background:linear-gradient(135deg,#6750a4,#00857a);
    font-family:var(--font-display);
}
.md-source-text { color:var(--md-soft); font-size:12.5px; line-height:1.45; word-break:break-word; }

/* ─────────────────────────────────────────── */
/*  CONFIDENCE BADGE                           */
/* ─────────────────────────────────────────── */
.md-confidence-badge {
    display:inline-flex; align-items:center; gap:7px;
    padding:5px 12px; border-radius:var(--radius-pill);
    font-size:11.5px; font-weight:800; margin-bottom:10px;
    letter-spacing:.03em; font-family:var(--font-display);
}
.md-confidence-high {
    border:1px solid rgba(34,197,94,.30);
    background:rgba(34,197,94,.09); color:#86efac;
}
.md-confidence-med {
    border:1px solid rgba(217,119,6,.30);
    background:rgba(217,119,6,.09); color:#fcd34d;
}
.md-confidence-low {
    border:1px solid rgba(229,57,53,.30);
    background:rgba(229,57,53,.09); color:#f87171;
}

/* ─────────────────────────────────────────── */
/*  CHAT HISTORY PANEL                         */
/* ─────────────────────────────────────────── */
.md-history-banner {
    display:flex; align-items:center; gap:10px; justify-content:space-between;
    border:1px solid var(--md-outline-variant); border-radius:var(--radius-md);
    padding:10px 14px; background:var(--md-surface); margin-bottom:10px; font-size:13px;
}
.md-history-count {
    display:inline-flex; align-items:center; gap:6px; font-weight:800; font-family:var(--font-display);
}
.md-history-dot { width:7px; height:7px; border-radius:50%; background:rgba(var(--md-primary-rgb),1); }

/* ─────────────────────────────────────────── */
/*  SUGGESTION CHIPS (quick prompts)           */
/* ─────────────────────────────────────────── */
.md-suggestions-row {
    display:flex; gap:8px; flex-wrap:wrap; margin-bottom:14px;
}
.md-suggestion-chip {
    display:inline-flex; align-items:center; gap:6px;
    padding:8px 14px; border-radius:var(--radius-pill);
    border:1px solid var(--md-outline-variant); background:var(--md-surface);
    font-size:13px; font-weight:700; cursor:pointer; color:var(--md-soft);
    transition: transform 130ms ease, background 130ms ease, border-color 130ms ease, box-shadow 130ms ease, color 130ms ease;
    font-family:var(--font-display);
}
.md-suggestion-chip:hover {
    transform:translateY(-2px);
    background:rgba(var(--md-primary-rgb),.12);
    border-color:rgba(var(--md-primary-rgb),.32);
    box-shadow:var(--md-shadow-1); color:white;
}

/* ─────────────────────────────────────────── */
/*  DISCLAIMER BANNER                          */
/* ─────────────────────────────────────────── */
.md-disclaimer {
    border:1px solid rgba(217,119,6,.22); border-radius:var(--radius-md);
    padding:11px 15px; background:rgba(217,119,6,.06);
    font-size:12.5px; line-height:1.5; color:var(--md-soft);
    display:flex; gap:10px; align-items:flex-start; margin-bottom:14px;
}
.md-disclaimer-icon { font-size:16px; flex-shrink:0; margin-top:1px; }

/* ─────────────────────────────────────────── */
/*  SEARCH PROGRESS BAR                        */
/* ─────────────────────────────────────────── */
.md-thinking-bar {
    height:3px; border-radius:2px; overflow:hidden;
    background:rgba(var(--md-primary-rgb),.12); margin-bottom:8px;
}
.md-thinking-bar-fill {
    height:100%; width:40%;
    background:linear-gradient(90deg, #6750a4, #00857a, #d97706);
    border-radius:2px;
    animation: thinking-slide 1.5s ease-in-out infinite alternate;
}
@keyframes thinking-slide {
    from { transform:translateX(-120%); }
    to   { transform:translateX(350%); }
}

/* ─────────────────────────────────────────── */
/*  BUTTONS                                    */
/* ─────────────────────────────────────────── */
.stButton > button {
    border-radius:var(--radius-pill) !important;
    min-height:46px; font-weight:800 !important;
    border:1px solid rgba(var(--md-primary-rgb),.35) !important;
    background:linear-gradient(135deg,#6750a4,#7c3aed) !important;
    color:white !important; box-shadow:var(--md-shadow-1);
    font-family:var(--font-display) !important; letter-spacing:.01em;
    transition:transform 130ms ease, box-shadow 130ms ease !important;
}
.stButton > button:hover {
    transform:translateY(-2px) !important;
    box-shadow:var(--md-shadow-2) !important;
}

/* ─────────────────────────────────────────── */
/*  CHAT INPUT                                 */
/* ─────────────────────────────────────────── */
[data-testid="stChatInput"] { border-radius:26px; }
[data-testid="stChatMessage"] { border-radius:24px; padding:4px 0; }

/* Save Answer tonal button — overrides global purple gradient */
[data-testid="stButton"][class*="save_btn_"] > button,
div[data-testid="column"]:first-child .stButton > button {
    background: rgba(var(--md-primary-rgb), 0.14) !important;
    background-image: none !important;
    border: 1px solid rgba(var(--md-primary-rgb), 0.32) !important;
    color: #d7c7ff !important;
    box-shadow: none !important;
    min-height: 38px !important;
    font-size: 13px !important;
}

/* ─────────────────────────────────────────── */
/*  TABS                                       */
/* ─────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap:4px; background:var(--md-surface); border-radius:var(--radius-pill);
    padding:4px; border:1px solid var(--md-outline-variant);
}
.stTabs [data-baseweb="tab"] {
    border-radius:var(--radius-pill) !important; padding:8px 18px;
    font-family:var(--font-display) !important; font-weight:700 !important;
    font-size:13px !important; transition: background 130ms ease !important;
}
.stTabs [aria-selected="true"] {
    background:linear-gradient(135deg,#6750a4,#7c3aed) !important;
    color:white !important;
}

/* ─────────────────────────────────────────── */
/*  HEALTH TIPS CARD                           */
/* ─────────────────────────────────────────── */
.md-tip-card {
    border:1px solid rgba(var(--md-secondary-rgb),.22); border-radius:var(--radius-lg);
    padding:16px; background:rgba(var(--md-secondary-rgb),.06);
    margin-bottom:12px;
}
.md-tip-header { display:flex; gap:10px; align-items:center; margin-bottom:9px; }
.md-tip-badge {
    padding:4px 10px; border-radius:var(--radius-pill); font-size:10.5px;
    font-weight:900; text-transform:uppercase; letter-spacing:.06em;
    background:rgba(var(--md-secondary-rgb),.20); color:#5eead4;
    border:1px solid rgba(var(--md-secondary-rgb),.25); font-family:var(--font-display);
}
.md-tip-title { font-size:14px; font-weight:800; font-family:var(--font-display); }
.md-tip-body  { font-size:13.5px; color:var(--md-soft); line-height:1.6; }

/* ─────────────────────────────────────────── */
/*  RESPONSIVE                                 */
/* ─────────────────────────────────────────── */
@media (max-width:980px) {
    .md-hero-inner, .md-card-grid { grid-template-columns:1fr; }
    .md-brand { align-items:flex-start; flex-direction:column; }
    .md-hero { padding:22px; border-radius:26px; }
    .block-container { padding-left:1rem; padding-right:1rem; }
    .md-topics-grid { grid-template-columns:repeat(4,1fr); }
}

@media (max-width:620px) {
    .md-logo,.md-logo-fallback { width:72px; height:72px; min-width:72px; border-radius:20px; padding:7px; }
    .sb-logo,.sb-logo-fallback { width:46px; height:46px; min-width:46px; border-radius:14px; padding:6px; }
    .md-title { font-size:34px; }
    .md-subtitle { font-size:15px; }
    .md-info-card,.md-empty,.md-source-list { border-radius:20px; }
    .sb-profile-card { border-radius:22px; padding:16px; }
    .md-topics-grid { grid-template-columns:repeat(4,1fr); }
    .md-hero-stats { flex-direction:column; gap:6px; }
}

/* ─────────────────────────────────────────── */
/*  FOOTER                                     */
/* ─────────────────────────────────────────── */
.md-footer {
    margin-top:52px; border-top:1px solid var(--md-outline);
    padding:32px 0 28px 0;
}
.md-footer-top { display:flex; justify-content:center; margin-bottom:20px; }
.md-footer-brand { display:flex; align-items:center; gap:14px; }
.md-footer-logo {
    width:52px; height:52px; border-radius:var(--radius-md);
    background:var(--md-surface-container);
    border:1px solid var(--md-outline-variant);
    padding:6px; box-shadow:var(--md-shadow-1); box-sizing:border-box;
}
.md-footer-logo-img {
    background-size:contain; background-repeat:no-repeat;
    background-position:center; display:inline-block; flex-shrink:0;
}
.md-footer-brand-name {
    font-family:var(--font-display) !important;
    font-size:15px; font-weight:900; line-height:1.2; max-width:200px;
}
.md-footer-brand-sub { font-size:12px; color:var(--md-soft); margin-top:3px; }
.md-footer-links {
    display:flex; flex-wrap:wrap; gap:8px;
    justify-content:center; margin-bottom:14px;
}
.md-footer-link {
    padding:8px 16px; border-radius:var(--radius-pill);
    border:1px solid var(--md-outline-variant);
    background:var(--md-surface); color:var(--md-soft) !important;
    font-size:13px; font-weight:700; text-decoration:none !important;
    transition:all 130ms ease; display:inline-flex; align-items:center; gap:5px;
}
.md-footer-link:hover {
    background:rgba(var(--md-primary-rgb),.12);
    border-color:rgba(var(--md-primary-rgb),.38);
    color:#a78bfa !important; transform:translateY(-2px);
    text-decoration:none !important;
}
.md-footer-meta {
    text-align:center; color:var(--md-soft);
    font-size:12px; line-height:1.75; margin-bottom:16px;
    white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
}
.md-footer-version {
    display:inline-block; padding:3px 10px; border-radius:var(--radius-pill);
    background:rgba(var(--md-primary-rgb),.12);
    border:1px solid rgba(var(--md-primary-rgb),.26);
    font-size:11px; font-weight:800; color:#a78bfa; vertical-align:middle;
}
.md-footer-heart { color:#ef4444; display:inline-block; animation:md-heartbeat 2.5s ease-in-out infinite; }
@keyframes md-heartbeat { 0%,100%{transform:scale(1)} 50%{transform:scale(1.22)} }
.md-footer-disclaimer {
    margin-top:20px; padding:13px 18px; border-radius:var(--radius-md);
    background:rgba(217,119,6,.07); border:1px solid rgba(217,119,6,.22);
    color:var(--md-soft); font-size:12px; line-height:1.6; text-align:center;
}
@media (max-width:620px) {
    .md-footer-brand { justify-content:center; }
    .md-footer-meta  { white-space:normal; }
}

/* ─────────────────────────────────────────── */
/*  SYMPTOM CHECKER                            */
/* ─────────────────────────────────────────── */
.sym-panel {
    border:1px solid var(--md-outline); border-radius:var(--radius-xl);
    padding:26px; background:
        radial-gradient(ellipse at 10% 10%, rgba(var(--md-primary-rgb),.13), transparent 40%),
        radial-gradient(ellipse at 90% 90%, rgba(var(--md-secondary-rgb),.09), transparent 40%),
        var(--md-surface);
    box-shadow:var(--md-shadow-2); margin-bottom:20px;
}
.sym-grid {
    display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin:16px 0;
}
@media (max-width:720px) { .sym-grid { grid-template-columns:repeat(2,1fr); } }

.sym-chip {
    display:flex; flex-direction:column; align-items:center; justify-content:center;
    gap:5px; padding:13px 8px; border-radius:var(--radius-md);
    border:1px solid var(--md-outline-variant); background:var(--md-surface);
    cursor:pointer; text-align:center;
    transition:transform 130ms ease, background 130ms ease, border-color 130ms ease, box-shadow 130ms ease;
}
.sym-chip:hover {
    transform:translateY(-3px); background:rgba(var(--md-primary-rgb),.12);
    border-color:rgba(var(--md-primary-rgb),.38); box-shadow:var(--md-shadow-1);
}
.sym-chip.selected {
    background:rgba(var(--md-primary-rgb),.18);
    border-color:rgba(var(--md-primary-rgb),.60);
    box-shadow:0 0 0 2px rgba(var(--md-primary-rgb),.25), var(--md-shadow-1);
}
.sym-chip-icon  { font-size:22px; line-height:1; }
.sym-chip-label { font-size:11.5px; font-weight:800; font-family:var(--font-display); line-height:1.2; }
.sym-chip-sub   { font-size:10px; color:var(--md-soft); }

.sym-severity-row { display:flex; gap:8px; flex-wrap:wrap; margin:12px 0; }
.sym-sev-btn {
    padding:7px 16px; border-radius:var(--radius-pill);
    border:1px solid var(--md-outline-variant); background:var(--md-surface);
    font-size:12px; font-weight:800; cursor:pointer; color:var(--md-soft);
    transition:all 130ms ease; font-family:var(--font-display);
}
.sym-sev-btn:hover { border-color:rgba(var(--md-primary-rgb),.40); color:white; background:rgba(var(--md-primary-rgb),.15); }

.sym-result {
    border-radius:var(--radius-lg); padding:18px; margin-top:14px;
    border:1px solid rgba(var(--md-secondary-rgb),.25);
    background:rgba(var(--md-secondary-rgb),.06);
    animation: fade-in-up .28s ease both;
}
.sym-result-title { font-size:14px; font-weight:900; font-family:var(--font-display); margin-bottom:9px; }
.sym-result-body  { font-size:13.5px; color:var(--md-soft); line-height:1.65; }
.sym-urgency-pill {
    display:inline-flex; align-items:center; gap:5px;
    padding:4px 12px; border-radius:var(--radius-pill); font-size:11px;
    font-weight:900; margin-bottom:10px; font-family:var(--font-display);
}
.sym-urgency-low    { background:rgba(34,197,94,.10); border:1px solid rgba(34,197,94,.28); color:#86efac; }
.sym-urgency-medium { background:rgba(217,119,6,.10); border:1px solid rgba(217,119,6,.28); color:#fcd34d; }
.sym-urgency-high   { background:rgba(229,57,53,.10); border:1px solid rgba(229,57,53,.28); color:#f87171; }

/* ─────────────────────────────────────────── */
/*  CHAT EXPORT / TOOLS BAR                    */
/* ─────────────────────────────────────────── */
.md-tools-bar {
    display:flex; gap:8px; align-items:center; flex-wrap:wrap;
    padding:10px 14px; border-radius:var(--radius-lg);
    border:1px solid var(--md-outline-variant); background:var(--md-surface);
    margin-bottom:14px;
}
.md-tool-btn {
    display:inline-flex; align-items:center; gap:6px;
    padding:7px 14px; border-radius:var(--radius-pill);
    border:1px solid var(--md-outline-variant); background:var(--md-surface-container);
    font-size:12px; font-weight:800; color:var(--md-soft); cursor:pointer;
    transition:all 130ms ease; font-family:var(--font-display); text-decoration:none;
}
.md-tool-btn:hover {
    border-color:rgba(var(--md-primary-rgb),.40); color:white;
    background:rgba(var(--md-primary-rgb),.14); transform:translateY(-1px);
}
.md-tool-divider { width:1px; height:20px; background:var(--md-outline-variant); }
.md-tool-label { font-size:11px; font-weight:700; color:var(--md-soft); margin-right:4px; }

/* ─────────────────────────────────────────── */
/*  READING TIME / WORD COUNT BADGE            */
/* ─────────────────────────────────────────── */
.md-read-meta {
    display:inline-flex; align-items:center; gap:5px;
    padding:3px 10px; border-radius:var(--radius-pill); margin-bottom:6px;
    font-size:11px; font-weight:700; color:var(--md-soft);
    background:var(--md-surface-container);
    border:1px solid var(--md-outline-variant);
}

/* ─────────────────────────────────────────── */
/*  HEALTH SCORE WIDGET  (redesigned)          */
/* ─────────────────────────────────────────── */

/* Outer wrapper */
.hs-panel {
    border:1px solid var(--md-outline);
    border-radius:var(--radius-xl);
    padding:28px 26px 24px;
    background:
        radial-gradient(ellipse at 8% 12%,  rgba(var(--md-primary-rgb),.16)   0%, transparent 42%),
        radial-gradient(ellipse at 92% 88%, rgba(var(--md-secondary-rgb),.11) 0%, transparent 42%),
        var(--md-surface);
    box-shadow:var(--md-shadow-2);
    margin-bottom:20px; position:relative; overflow:hidden;
}
.hs-panel::before {
    content:''; position:absolute; inset:0; border-radius:inherit;
    background:linear-gradient(135deg,rgba(255,255,255,.04) 0%,transparent 55%);
    pointer-events:none;
}

/* Step progress bar at top */
.hs-progress-wrap {
    display:flex; gap:5px; margin-bottom:22px;
}
.hs-progress-seg {
    flex:1; height:3px; border-radius:99px;
    background:var(--md-outline-variant);
    transition:background .35s ease;
}
.hs-progress-seg.done  { background:linear-gradient(90deg,#6750a4,#00857a); }
.hs-progress-seg.active{ background:rgba(var(--md-primary-rgb),.45); }

/* Question card */
.hs-q-card {
    border:1px solid var(--md-outline);
    border-radius:var(--radius-lg);
    padding:20px 22px;
    background:var(--md-surface-container);
    margin-bottom:16px;
    animation: fade-in-up .22s ease both;
}
.hs-q-step {
    font-size:11px; font-weight:900; color:rgba(var(--md-primary-rgb),1);
    text-transform:uppercase; letter-spacing:.08em;
    font-family:var(--font-display); margin-bottom:7px;
}
.hs-q-text {
    font-size:16px; font-weight:800;
    font-family:var(--font-display); line-height:1.3;
    margin-bottom:16px;
}

/* Option buttons — styled as selectable cards */
.hs-options-wrap {
    display:grid; grid-template-columns:1fr 1fr; gap:9px; margin-top:4px;
}
@media (max-width:540px) { .hs-options-wrap { grid-template-columns:1fr; } }

/* Style every button inside the quiz card to look like an option card */
.hs-q-card .stButton > button {
    border-radius:var(--radius-md) !important;
    min-height:52px !important;
    background:var(--md-surface) !important;
    background-image:none !important;
    border:1.5px solid var(--md-outline-variant) !important;
    color:var(--md-soft) !important;
    font-size:13px !important;
    font-weight:700 !important;
    font-family:var(--font-display) !important;
    box-shadow:none !important;
    text-align:left !important;
    justify-content:flex-start !important;
    letter-spacing:0 !important;
    transition:transform 130ms ease, background 130ms ease,
               border-color 130ms ease, box-shadow 130ms ease, color 130ms ease !important;
}
.hs-q-card .stButton > button:hover {
    transform:translateY(-2px) !important;
    background:rgba(var(--md-primary-rgb),.12) !important;
    border-color:rgba(var(--md-primary-rgb),.45) !important;
    box-shadow:var(--md-shadow-1) !important;
    color:white !important;
}
.hs-q-card .stButton > button:focus,
.hs-q-card .stButton > button:active {
    background:rgba(var(--md-primary-rgb),.22) !important;
    border-color:rgba(var(--md-primary-rgb),.70) !important;
    box-shadow:0 0 0 2px rgba(var(--md-primary-rgb),.25) !important;
    color:white !important;
}

/* Selected option highlight — applied via a wrapper div */
.hs-opt-selected .stButton > button {
    background:rgba(var(--md-primary-rgb),.18) !important;
    border-color:rgba(var(--md-primary-rgb),.65) !important;
    box-shadow:0 0 0 2px rgba(var(--md-primary-rgb),.22), var(--md-shadow-1) !important;
    color:white !important;
}

/* Back button override — keep it subtle */
.hs-back-wrap .stButton > button {
    background:transparent !important;
    background-image:none !important;
    border:1px solid var(--md-outline-variant) !important;
    color:var(--md-soft) !important;
    font-size:12px !important;
    font-weight:700 !important;
    min-height:36px !important;
    box-shadow:none !important;
    border-radius:var(--radius-pill) !important;
}
.hs-back-wrap .stButton > button:hover {
    border-color:rgba(var(--md-primary-rgb),.40) !important;
    color:white !important; transform:none !important;
}

/* Result card */
.hs-result {
    border-radius:var(--radius-xl);
    padding:28px 24px;
    position:relative; overflow:hidden;
    animation:fade-in-up .3s ease both;
    margin-top:4px;
}
.hs-result-bg-excellent {
    background:
        radial-gradient(ellipse at 15% 20%, rgba(34,197,94,.20)  0%, transparent 50%),
        radial-gradient(ellipse at 85% 80%, rgba(103,80,164,.16) 0%, transparent 50%),
        var(--md-surface-container);
    border:1px solid rgba(34,197,94,.28);
}
.hs-result-bg-good {
    background:
        radial-gradient(ellipse at 15% 20%, rgba(103,80,164,.20) 0%, transparent 50%),
        radial-gradient(ellipse at 85% 80%, rgba(0,133,122,.14)  0%, transparent 50%),
        var(--md-surface-container);
    border:1px solid rgba(103,80,164,.30);
}
.hs-result-bg-fair {
    background:
        radial-gradient(ellipse at 15% 20%, rgba(217,119,6,.18)  0%, transparent 50%),
        radial-gradient(ellipse at 85% 80%, rgba(103,80,164,.12) 0%, transparent 50%),
        var(--md-surface-container);
    border:1px solid rgba(217,119,6,.28);
}
.hs-result-bg-poor {
    background:
        radial-gradient(ellipse at 15% 20%, rgba(229,57,53,.16)  0%, transparent 50%),
        radial-gradient(ellipse at 85% 80%, rgba(217,119,6,.12)  0%, transparent 50%),
        var(--md-surface-container);
    border:1px solid rgba(229,57,53,.28);
}

.hs-result-top {
    display:flex; align-items:center; gap:24px; margin-bottom:24px;
}

/* SVG arc ring */
.hs-arc-wrap { flex-shrink:0; position:relative; width:110px; height:110px; }
.hs-arc-svg  { width:110px; height:110px; transform:rotate(-90deg); }
.hs-arc-bg   { fill:none; stroke:rgba(255,255,255,.07); stroke-width:9; }
.hs-arc-fill {
    fill:none; stroke-width:9; stroke-linecap:round;
    stroke-dasharray:283;
    stroke-dashoffset:283;
    transition: stroke-dashoffset 1s cubic-bezier(.4,0,.2,1);
}
.hs-arc-excellent { stroke:url(#hs-grad-excellent); }
.hs-arc-good      { stroke:url(#hs-grad-good); }
.hs-arc-fair      { stroke:url(#hs-grad-fair); }
.hs-arc-poor      { stroke:url(#hs-grad-poor); }

.hs-arc-center {
    position:absolute; inset:0;
    display:flex; flex-direction:column;
    align-items:center; justify-content:center;
}
.hs-arc-num {
    font-size:28px; font-weight:900;
    font-family:var(--font-display); line-height:1;
}
.hs-arc-denom { font-size:11px; color:var(--md-soft); font-weight:700; }

.hs-result-info { flex:1; min-width:0; }
.hs-result-badge {
    display:inline-flex; align-items:center; gap:6px;
    padding:5px 13px; border-radius:var(--radius-pill);
    font-size:12px; font-weight:900; font-family:var(--font-display);
    letter-spacing:.03em; margin-bottom:10px;
}
.hs-badge-excellent { background:rgba(34,197,94,.14); border:1px solid rgba(34,197,94,.32); color:#86efac; }
.hs-badge-good      { background:rgba(103,80,164,.18); border:1px solid rgba(103,80,164,.40); color:#c4b5fd; }
.hs-badge-fair      { background:rgba(217,119,6,.14);  border:1px solid rgba(217,119,6,.32);  color:#fcd34d; }
.hs-badge-poor      { background:rgba(229,57,53,.14);  border:1px solid rgba(229,57,53,.32);  color:#f87171; }

.hs-result-headline {
    font-size:20px; font-weight:900;
    font-family:var(--font-display); line-height:1.15;
    margin-bottom:7px;
}
.hs-result-sub {
    font-size:13px; color:var(--md-soft); line-height:1.6;
}

/* Factor strip — horizontal cards */
.hs-factors-v2 {
    display:grid; grid-template-columns:repeat(5,1fr); gap:9px;
    margin-top:18px;
}
@media (max-width:720px) { .hs-factors-v2 { grid-template-columns:repeat(3,1fr); } }
@media (max-width:480px) { .hs-factors-v2 { grid-template-columns:1fr 1fr; } }

.hs-factor-v2 {
    border:1px solid var(--md-outline-variant);
    border-radius:var(--radius-md);
    padding:12px 10px 10px;
    background:rgba(255,255,255,.035);
    text-align:center;
    position:relative; overflow:hidden;
    transition:transform 130ms ease, background 130ms ease;
}
.hs-factor-v2:hover { transform:translateY(-2px); background:rgba(255,255,255,.06); }
.hs-factor-v2::after {
    content:''; position:absolute; bottom:0; left:0; right:0;
    height:3px;
    background:linear-gradient(90deg,#6750a4,#00857a);
    transform:scaleX(var(--fill));
    transform-origin:left;
    border-radius:0 0 var(--radius-md) var(--radius-md);
    transition:transform 1s cubic-bezier(.4,0,.2,1);
}
.hs-fv2-icon  { font-size:20px; line-height:1; margin-bottom:6px; }
.hs-fv2-label { font-size:10px; font-weight:800; color:var(--md-soft); text-transform:uppercase; letter-spacing:.06em; margin-bottom:5px; }
.hs-fv2-val   { font-size:18px; font-weight:900; font-family:var(--font-display); line-height:1; }
.hs-fv2-sub   { font-size:10px; color:var(--md-soft); margin-top:2px; }

/* Retake button */
.hs-retake {
    display:inline-flex; align-items:center; gap:6px;
    padding:8px 18px; border-radius:var(--radius-pill);
    border:1px solid var(--md-outline-variant);
    background:var(--md-surface-container);
    font-size:12px; font-weight:800; cursor:pointer;
    color:var(--md-soft); margin-top:16px;
    transition:all 130ms ease; font-family:var(--font-display);
}
.hs-retake:hover {
    border-color:rgba(var(--md-primary-rgb),.40); color:white;
    background:rgba(var(--md-primary-rgb),.12);
}

/* ─────────────────────────────────────────── */
/*  MEDICATION REMINDER PANEL                  */
/* ─────────────────────────────────────────── */
.rem-panel {
    border:1px solid var(--md-outline); border-radius:var(--radius-xl);
    padding:22px; background:var(--md-surface); box-shadow:var(--md-shadow-1);
}
.rem-item {
    display:flex; align-items:center; gap:12px;
    padding:12px 14px; border-radius:var(--radius-md);
    border:1px solid var(--md-outline-variant); background:var(--md-surface-container);
    margin-bottom:8px; transition:background 130ms ease;
}
.rem-item:hover { background:var(--md-surface-container-high); }
.rem-dot {
    width:10px; height:10px; border-radius:50%; flex-shrink:0;
}
.rem-dot-green { background:#22c55e; box-shadow:0 0 8px rgba(34,197,94,.7); }
.rem-dot-amber { background:#f59e0b; box-shadow:0 0 8px rgba(245,158,11,.7); }
.rem-dot-red   { background:#ef4444; box-shadow:0 0 8px rgba(239,68,68,.7); }
.rem-name  { font-size:14px; font-weight:800; font-family:var(--font-display); }
.rem-dose  { font-size:12px; color:var(--md-soft); }
.rem-time  { margin-left:auto; font-size:12px; font-weight:700; color:var(--md-soft); white-space:nowrap; }
.rem-badge {
    padding:3px 9px; border-radius:var(--radius-pill); font-size:10px; font-weight:900;
    font-family:var(--font-display);
}
.rem-badge-taken   { background:rgba(34,197,94,.12); border:1px solid rgba(34,197,94,.28); color:#86efac; }
.rem-badge-pending { background:rgba(245,158,11,.12); border:1px solid rgba(245,158,11,.28); color:#fcd34d; }
.rem-badge-missed  { background:rgba(239,68,68,.12);  border:1px solid rgba(239,68,68,.28);  color:#f87171; }

/* ─────────────────────────────────────────── */
/*  ANIMATIONS                                 */
/* ─────────────────────────────────────────── */
@keyframes fade-in-up {
    from { opacity:0; transform:translateY(10px); }
    to   { opacity:1; transform:translateY(0); }
}
.md-animate-in { animation: fade-in-up .28s ease both; }

/* ─────────────────────────────────────────── */
/*  SEARCH HIGHLIGHT                           */
/* ─────────────────────────────────────────── */
.md-search-bar {
    display:flex; gap:8px; align-items:center; margin-bottom:12px;
}
mark.md-highlight {
    background:rgba(var(--md-primary-rgb),.30);
    color:white; border-radius:3px; padding:0 2px;
}

/* ─────────────────────────────────────────── */
/*  TABS — extra tab styling                   */
/* ─────────────────────────────────────────── */
.stTabs [data-baseweb="tab-panel"] { padding-top:16px; }

/* ─────────────────────────────────────────── */
/*  LIGHT MODE OVERRIDES                       */
/* ─────────────────────────────────────────── */
[data-theme="light"] {
    --md-bg:                    rgba(103,80,164,0.04);
    --md-surface:               rgba(255,255,255,0.82);
    --md-surface-container:     rgba(103,80,164,0.08);
    --md-surface-container-high:rgba(103,80,164,0.12);
    --md-outline:               rgba(103,80,164,0.22);
    --md-outline-variant:       rgba(103,80,164,0.14);
    --md-soft:                  rgba(50,40,80,0.72);
    --md-shadow-1: 0 4px 16px rgba(103,80,164,0.10);
    --md-shadow-2: 0 12px 36px rgba(103,80,164,0.14);
    --md-shadow-3: 0 24px 64px rgba(103,80,164,0.18);
}

[data-theme="light"] .stApp,
[data-theme="light"] [data-testid="stAppViewContainer"] {
    background: linear-gradient(160deg, #f8f6ff 0%, #eef8f7 55%, #f5f0ff 100%) !important;
    color: #1d1b27 !important;
}
[data-theme="light"] [data-testid="stMain"] { background: transparent !important; }

/* Sidebar light (non-mobile) */
[data-theme="light"] [data-testid="stSidebar"] {
    background:
        radial-gradient(ellipse at 18% 4%,  rgba(103,80,164,0.16)  0%, transparent 42%),
        radial-gradient(ellipse at 90% 44%, rgba(0,133,122,0.12)   0%, transparent 42%),
        radial-gradient(ellipse at 48% 92%, rgba(217,119,6,0.07)   0%, transparent 42%),
        linear-gradient(180deg,
            rgba(103,80,164,0.09) 0%,
            rgba(0,133,122,0.04) 55%,
            transparent 100%) !important;
    border-right: 1px solid rgba(103,80,164,0.20) !important;
}
@media (max-width: 768px) {
    [data-theme="light"] [data-testid="stSidebar"] { background: #f5f3ff !important; }
}

/* Profile card */
[data-theme="light"] .sb-profile-card {
    background:
        radial-gradient(ellipse at 15% 10%, rgba(103,80,164,0.10), transparent 46%),
        radial-gradient(ellipse at 90% 90%, rgba(0,133,122,0.07),  transparent 46%),
        rgba(255,255,255,0.88) !important;
    border-color: rgba(103,80,164,0.20) !important;
    box-shadow: 0 4px 18px rgba(103,80,164,0.10), inset 0 1px 0 rgba(255,255,255,0.80) !important;
}
[data-theme="light"] .sb-title {
    background: linear-gradient(115deg, #4a3a7d 0%, #6750a4 55%, #0d7b72 100%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
}
[data-theme="light"] .sb-description  {
    color: #5a5270 !important;
    border-top-color: rgba(103,80,164,0.14) !important;
}
[data-theme="light"] .sb-section      { color: #6750a4 !important; }
[data-theme="light"] .sb-section::after { background: linear-gradient(90deg, rgba(103,80,164,0.28) 0%, transparent 100%) !important; }
[data-theme="light"] .sb-logo, [data-theme="light"] .sb-logo-fallback {
    background:
        linear-gradient(135deg, rgba(103,80,164,0.10), rgba(255,255,255,0.06)),
        rgba(255,255,255,0.85) !important;
    border-color: rgba(103,80,164,0.22) !important;
    box-shadow: 0 4px 14px rgba(103,80,164,0.16), inset 0 1px 0 rgba(255,255,255,0.8) !important;
}

/* Stats */
[data-theme="light"] .sb-stat-card {
    background: rgba(255,255,255,0.82) !important;
    border-color: rgba(103,80,164,0.16) !important;
}
[data-theme="light"] .sb-stat-card::before {
    background: linear-gradient(90deg, transparent, rgba(103,80,164,0.40), transparent) !important;
}
[data-theme="light"] .sb-stat-card:hover { background: rgba(255,255,255,0.96) !important; }
[data-theme="light"] .sb-stat-number {
    background: linear-gradient(115deg, #4a3a7d, #6750a4) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
}
[data-theme="light"] .sb-stat-label     { color: #7c6a9a !important; }

/* Chips */
[data-theme="light"] .sb-chip {
    background: rgba(255,255,255,0.80) !important;
    border-color: rgba(103,80,164,0.16) !important;
}
[data-theme="light"] .sb-chip:hover {
    background: rgba(103,80,164,0.08) !important;
    border-color: rgba(103,80,164,0.34) !important;
}
[data-theme="light"] .sb-chip::before {
    background: linear-gradient(180deg, rgba(103,80,164,0.7), rgba(0,133,122,0.5)) !important;
}
[data-theme="light"] .sb-chip-icon {
    background: linear-gradient(135deg, rgba(103,80,164,0.18), rgba(0,133,122,0.12)) !important;
    border-color: rgba(103,80,164,0.16) !important;
}
[data-theme="light"] .sb-chip-label { color: #7c6a9a !important; }
[data-theme="light"] .sb-chip-value { color: #1a1530 !important; }

/* Note / tip */
[data-theme="light"] .sb-note {
    background: rgba(245,158,11,0.07) !important;
    border-color: rgba(245,158,11,0.28) !important;
    color: #5a5270 !important;
}
[data-theme="light"] .sb-note::before {
    background: linear-gradient(180deg, #f59e0b, #d97706) !important;
}
[data-theme="light"] .sb-tip {
    background: rgba(0,133,122,0.06) !important;
    border-color: rgba(0,133,122,0.22) !important;
    color: #5a5270 !important;
}
[data-theme="light"] .sb-tip::before {
    background: linear-gradient(180deg, rgba(0,133,122,0.8), rgba(103,80,164,0.5)) !important;
}
[data-theme="light"] .sb-footer {
    color: #7c6a9a !important;
    border-top-color: rgba(103,80,164,0.12) !important;
}

/* Hero */
[data-theme="light"] .md-hero {
    background:
        radial-gradient(ellipse at 10% 10%, rgba(103,80,164,0.14), transparent 40%),
        radial-gradient(ellipse at 88% 15%, rgba(0,133,122,0.10),  transparent 40%),
        radial-gradient(ellipse at 55% 90%, rgba(217,119,6,0.06),  transparent 40%),
        linear-gradient(135deg, rgba(103,80,164,0.10), rgba(0,133,122,0.06) 58%, rgba(217,119,6,0.03)),
        rgba(255,255,255,0.82) !important;
    border-color: rgba(103,80,164,0.20) !important;
}
[data-theme="light"] .md-kicker {
    background: rgba(103,80,164,0.11) !important;
    border-color: rgba(103,80,164,0.30) !important;
    color: #4a3a7d !important;
}
[data-theme="light"] .md-logo, [data-theme="light"] .md-logo-fallback {
    background:
        linear-gradient(135deg, rgba(103,80,164,0.10), rgba(255,255,255,0.06)),
        rgba(255,255,255,0.82) !important;
    border-color: rgba(103,80,164,0.18) !important;
}
[data-theme="light"] .md-title    { color: #1a1530 !important; }
[data-theme="light"] .md-subtitle { color: #5a5270 !important; }
[data-theme="light"] .md-pill {
    background: rgba(255,255,255,0.82) !important;
    border-color: rgba(103,80,164,0.18) !important;
}
[data-theme="light"] .md-pill-label { color: #7c6a9a !important; }
[data-theme="light"] .md-pill-value { color: #1a1530 !important; }
[data-theme="light"] .md-hero-stat  { color: #5a5270 !important; background: rgba(255,255,255,0.78) !important; }

/* Feature info cards */
[data-theme="light"] .md-info-card {
    background: rgba(255,255,255,0.82) !important;
    border-color: rgba(103,80,164,0.18) !important;
}
[data-theme="light"] .md-info-card:hover {
    background: rgba(255,255,255,0.94) !important;
    border-color: rgba(103,80,164,0.34) !important;
}
[data-theme="light"] .md-info-card strong { color: #1a1530 !important; }
[data-theme="light"] .md-info-card span   { color: #5a5270 !important; }
[data-theme="light"] .md-card-icon {
    background: linear-gradient(135deg, rgba(103,80,164,0.18), rgba(0,133,122,0.12)) !important;
}

/* Section title */
[data-theme="light"] .md-section-title { color: #1a1530 !important; }
[data-theme="light"] .md-section-sub   { color: #5a5270 !important; }

/* Topic cards */
[data-theme="light"] .md-topic-card {
    background: rgba(255,255,255,0.78) !important;
    border-color: rgba(103,80,164,0.16) !important;
}
[data-theme="light"] .md-topic-card:hover {
    background: rgba(255,255,255,0.94) !important;
    border-color: rgba(103,80,164,0.34) !important;
}
[data-theme="light"] .md-topic-label { color: #1a1530 !important; }
[data-theme="light"] .md-topic-sub   { color: #7c6a9a !important; }
[data-theme="light"] .md-topic-btn .stButton > button {
    background: rgba(103,80,164,0.10) !important;
    border-color: rgba(103,80,164,0.18) !important;
    color: #3a2a5d !important;
}
[data-theme="light"] .md-topic-btn .stButton > button:hover {
    background: rgba(103,80,164,0.22) !important;
    color: #1a1530 !important;
}

/* Empty state */
[data-theme="light"] .md-empty {
    background: rgba(255,255,255,0.78) !important;
    border-color: rgba(103,80,164,0.20) !important;
    color: #5a5270 !important;
}

/* Disclaimer */
[data-theme="light"] .md-disclaimer { color: #5a5270 !important; }

/* Source list */
[data-theme="light"] .md-source-list {
    background:
        linear-gradient(135deg, rgba(103,80,164,0.08), rgba(0,133,122,0.05)),
        rgba(255,255,255,0.82) !important;
    border-color: rgba(103,80,164,0.18) !important;
}
[data-theme="light"] .md-source-title  { color: #3a2a5d !important; }
[data-theme="light"] .md-source-item {
    background: rgba(255,255,255,0.72) !important;
    border-color: rgba(103,80,164,0.14) !important;
}
[data-theme="light"] .md-source-item:hover { background: rgba(255,255,255,0.90) !important; }
[data-theme="light"] .md-source-text  { color: #5a5270 !important; }

/* Confidence badges */
[data-theme="light"] .md-confidence-high { color: #166534 !important; }
[data-theme="light"] .md-confidence-med  { color: #92400e !important; }
[data-theme="light"] .md-confidence-low  { color: #991b1b !important; }

/* History banner */
[data-theme="light"] .md-history-banner {
    background: rgba(255,255,255,0.82) !important;
    border-color: rgba(103,80,164,0.14) !important;
    color: #3a2a5d !important;
}

/* Suggestion chips */
[data-theme="light"] .md-suggestion-chip {
    background: rgba(255,255,255,0.80) !important;
    border-color: rgba(103,80,164,0.16) !important;
    color: #3a2a5d !important;
}
[data-theme="light"] .md-suggestion-chip:hover {
    background: rgba(103,80,164,0.10) !important;
    border-color: rgba(103,80,164,0.34) !important;
    color: #1a1530 !important;
}

/* Health score quiz */
[data-theme="light"] .hs-panel {
    background:
        radial-gradient(ellipse at 8% 12%,  rgba(103,80,164,0.12)  0%, transparent 42%),
        radial-gradient(ellipse at 92% 88%, rgba(0,133,122,0.09)   0%, transparent 42%),
        rgba(255,255,255,0.84) !important;
    border-color: rgba(103,80,164,0.20) !important;
}
[data-theme="light"] .hs-q-card {
    background: rgba(255,255,255,0.82) !important;
    border-color: rgba(103,80,164,0.18) !important;
}
[data-theme="light"] .hs-q-text { color: #1a1530 !important; }
[data-theme="light"] .hs-q-card .stButton > button {
    background: rgba(255,255,255,0.80) !important;
    border-color: rgba(103,80,164,0.22) !important;
    color: #3a2a5d !important;
}
[data-theme="light"] .hs-q-card .stButton > button:hover {
    background: rgba(103,80,164,0.12) !important;
    border-color: rgba(103,80,164,0.45) !important;
    color: #1a1530 !important;
}
[data-theme="light"] .hs-result-bg-excellent { background:
    radial-gradient(ellipse at 15% 20%, rgba(34,197,94,0.14)   0%, transparent 50%),
    radial-gradient(ellipse at 85% 80%, rgba(103,80,164,0.12)  0%, transparent 50%),
    rgba(255,255,255,0.84) !important; }
[data-theme="light"] .hs-result-bg-good { background:
    radial-gradient(ellipse at 15% 20%, rgba(103,80,164,0.14)  0%, transparent 50%),
    radial-gradient(ellipse at 85% 80%, rgba(0,133,122,0.10)   0%, transparent 50%),
    rgba(255,255,255,0.84) !important; }
[data-theme="light"] .hs-result-bg-fair { background:
    radial-gradient(ellipse at 15% 20%, rgba(217,119,6,0.12)   0%, transparent 50%),
    radial-gradient(ellipse at 85% 80%, rgba(103,80,164,0.10)  0%, transparent 50%),
    rgba(255,255,255,0.84) !important; }
[data-theme="light"] .hs-result-bg-poor { background:
    radial-gradient(ellipse at 15% 20%, rgba(229,57,53,0.12)   0%, transparent 50%),
    radial-gradient(ellipse at 85% 80%, rgba(217,119,6,0.10)   0%, transparent 50%),
    rgba(255,255,255,0.84) !important; }
[data-theme="light"] .hs-result-headline { color: #1a1530 !important; }
[data-theme="light"] .hs-result-sub      { color: #5a5270 !important; }
[data-theme="light"] .hs-arc-bg          { stroke: rgba(103,80,164,0.14) !important; }
[data-theme="light"] .hs-arc-num         { color: #1a1530 !important; }
[data-theme="light"] .hs-arc-denom       { color: #7c6a9a !important; }
[data-theme="light"] .hs-badge-good      { background: rgba(103,80,164,0.12) !important; border-color: rgba(103,80,164,0.30) !important; color: #4a3a7d !important; }
[data-theme="light"] .hs-factor-v2 {
    background: rgba(255,255,255,0.78) !important;
}
[data-theme="light"] .hs-factor-v2:hover { background: rgba(255,255,255,0.94) !important; }
[data-theme="light"] .hs-fv2-label { color: #7c6a9a !important; }
[data-theme="light"] .hs-fv2-val   { color: #1a1530 !important; }
[data-theme="light"] .hs-fv2-sub   { color: #7c6a9a !important; }
[data-theme="light"] .hs-retake {
    background: rgba(255,255,255,0.78) !important;
    border-color: rgba(103,80,164,0.18) !important;
    color: #5a5270 !important;
}
[data-theme="light"] .hs-retake:hover { color: #1a1530 !important; }

/* Symptom panel */
[data-theme="light"] .sym-panel {
    background:
        radial-gradient(ellipse at 10% 10%, rgba(103,80,164,0.12), transparent 40%),
        radial-gradient(ellipse at 90% 90%, rgba(0,133,122,0.08), transparent 40%),
        rgba(255,255,255,0.82) !important;
    border-color: rgba(103,80,164,0.20) !important;
}
[data-theme="light"] .sym-chip {
    background: rgba(255,255,255,0.80) !important;
    border-color: rgba(103,80,164,0.16) !important;
    color: #3a2a5d !important;
}
[data-theme="light"] .sym-chip:hover {
    background: rgba(103,80,164,0.10) !important;
    border-color: rgba(103,80,164,0.36) !important;
}
[data-theme="light"] .sym-chip.selected {
    background: rgba(103,80,164,0.16) !important;
    border-color: rgba(103,80,164,0.52) !important;
}
[data-theme="light"] .sym-chip-label { color: #1a1530 !important; }
[data-theme="light"] .sym-chip-sub   { color: #7c6a9a !important; }
[data-theme="light"] .sym-sev-btn {
    background: rgba(255,255,255,0.78) !important;
    border-color: rgba(103,80,164,0.16) !important;
    color: #5a5270 !important;
}
[data-theme="light"] .sym-sev-btn:hover { color: #1a1530 !important; }
[data-theme="light"] .sym-result {
    background: rgba(0,133,122,0.06) !important;
    border-color: rgba(0,133,122,0.22) !important;
}
[data-theme="light"] .sym-result-title { color: #1a1530 !important; }
[data-theme="light"] .sym-result-body  { color: #5a5270 !important; }
[data-theme="light"] .sym-urgency-low    { color: #166534 !important; }
[data-theme="light"] .sym-urgency-medium { color: #92400e !important; }
[data-theme="light"] .sym-urgency-high   { color: #991b1b !important; }

/* Medication reminder */
[data-theme="light"] .rem-panel {
    background: rgba(255,255,255,0.82) !important;
    border-color: rgba(103,80,164,0.18) !important;
}
[data-theme="light"] .rem-item {
    background: rgba(255,255,255,0.78) !important;
    border-color: rgba(103,80,164,0.14) !important;
}
[data-theme="light"] .rem-item:hover { background: rgba(255,255,255,0.94) !important; }
[data-theme="light"] .rem-name { color: #1a1530 !important; }
[data-theme="light"] .rem-dose, [data-theme="light"] .rem-time { color: #7c6a9a !important; }
[data-theme="light"] .rem-badge-taken   { color: #166534 !important; }
[data-theme="light"] .rem-badge-pending { color: #92400e !important; }
[data-theme="light"] .rem-badge-missed  { color: #991b1b !important; }

/* Tip card */
[data-theme="light"] .md-tip-card {
    background: rgba(0,133,122,0.06) !important;
    border-color: rgba(0,133,122,0.20) !important;
}
[data-theme="light"] .md-tip-badge { background: rgba(0,133,122,0.14) !important; border-color: rgba(0,133,122,0.22) !important; color: #0d6b62 !important; }
[data-theme="light"] .md-tip-title { color: #1a1530 !important; }
[data-theme="light"] .md-tip-body  { color: #5a5270 !important; }

/* Tools bar */
[data-theme="light"] .md-tools-bar {
    background: rgba(255,255,255,0.80) !important;
    border-color: rgba(103,80,164,0.16) !important;
}
[data-theme="light"] .md-tool-btn {
    background: rgba(255,255,255,0.80) !important;
    border-color: rgba(103,80,164,0.14) !important;
    color: #5a5270 !important;
}
[data-theme="light"] .md-tool-btn:hover { color: #1a1530 !important; }
[data-theme="light"] .md-tool-label     { color: #7c6a9a !important; }
[data-theme="light"] .md-tool-divider   { background: rgba(103,80,164,0.14) !important; }

/* Read meta */
[data-theme="light"] .md-read-meta {
    background: rgba(255,255,255,0.80) !important;
    border-color: rgba(103,80,164,0.14) !important;
    color: #7c6a9a !important;
}

/* History dots */
[data-theme="light"] .md-history-count { color: #1a1530 !important; }

/* Search highlight */
[data-theme="light"] mark.md-highlight {
    background: rgba(103,80,164,0.22) !important;
    color: #1a1530 !important;
}

/* Footer */
[data-theme="light"] .md-footer { border-top-color: rgba(103,80,164,0.18) !important; }
[data-theme="light"] .md-footer-brand-name { color: #1a1530 !important; }
[data-theme="light"] .md-footer-brand-sub  { color: #7c6a9a !important; }
[data-theme="light"] .md-footer-link {
    background: rgba(255,255,255,0.78) !important;
    border-color: rgba(103,80,164,0.16) !important;
    color: #5a5270 !important;
}
[data-theme="light"] .md-footer-link:hover { color: #3a2a5d !important; }
[data-theme="light"] .md-footer-meta       { color: #7c6a9a !important; }
[data-theme="light"] .md-footer-version {
    background: rgba(103,80,164,0.10) !important;
    border-color: rgba(103,80,164,0.24) !important;
    color: #4a3a7d !important;
}
[data-theme="light"] .md-footer-disclaimer { color: #5a5270 !important; }

/* Tabs */
[data-theme="light"] .stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.82) !important;
    border-color: rgba(103,80,164,0.16) !important;
}
[data-theme="light"] .stTabs [data-baseweb="tab"]  { color: #3a2a5d !important; }
[data-theme="light"] .stTabs [aria-selected="true"] { color: white !important; }

/* Streamlit inputs */
[data-theme="light"] input,
[data-theme="light"] textarea,
[data-theme="light"] [data-baseweb="select"] {
    background: rgba(255,255,255,0.90) !important;
    border-color: rgba(103,80,164,0.25) !important;
    color: #1a1530 !important;
}
[data-theme="light"] label,
[data-theme="light"] p,
[data-theme="light"] li { color: #1d1b27 !important; }
</style>
""",
        unsafe_allow_html=True,
    )


render_page_styles()


# ─────────────────────────────────────────────
#  SESSION STATE INIT
# ─────────────────────────────────────────────
def init_session_state():
    defaults = {
        "messages":            [],
        "total_queries":       0,
        "session_start":       datetime.now().strftime("%H:%M"),
        "last_query_time":     None,
        "bookmarked_answers":  [],
        "active_tab":          "chat",
        "pending_bookmark":    None,
        "bookmark_saved_id":   None,
        # ── New features ──
        "selected_symptoms":   [],
        "symptom_severity":    "Mild",
        "symptom_result":      None,
        "reminders":           [
            {"name": "Aspirin",     "dose": "75mg",   "time": "08:00 AM", "status": "taken"},
            {"name": "Metformin",   "dose": "500mg",  "time": "01:00 PM", "status": "pending"},
            {"name": "Atorvastatin","dose": "10mg",   "time": "09:00 PM", "status": "pending"},
        ],
        "health_score_answers":None,
        "chat_search_query":   "",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_session_state()


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
import random as _random

HEALTH_TIPS = [
    ("💧", "Hydration",       "Drink at least 8 glasses of water daily to support kidney function and overall health."),
    ("🏃", "Exercise",        "30 minutes of moderate exercise 5 days a week significantly reduces cardiovascular risk."),
    ("😴", "Sleep",           "Adults need 7–9 hours of quality sleep for immune function and cognitive health."),
    ("🥦", "Nutrition",       "Aim for 5 servings of fruits and vegetables daily for essential vitamins and fibre."),
    ("🧘", "Stress",          "Chronic stress raises cortisol levels. Even 10 minutes of mindfulness daily helps."),
    ("🫀", "Heart Health",    "Limit saturated fats and sodium. A heart-healthy diet can reduce stroke risk by up to 35%."),
    ("🦷", "Oral Care",       "Brush twice daily and floss once — gum disease is linked to heart disease and diabetes."),
    ("🧴", "Sun Protection",  "Apply SPF 30+ sunscreen daily, even on cloudy days, to protect against UV-induced skin damage."),
    ("🚭", "No Smoking",      "Quitting smoking at any age reduces cardiovascular risk within just 20 minutes of the last cigarette."),
    ("🍵", "Antioxidants",    "Green tea contains EGCG, a powerful antioxidant that supports brain health and reduces inflammation."),
    ("🧬", "Gut Health",      "A diverse diet rich in fibre feeds beneficial gut bacteria, boosting immunity and mood regulation."),
    ("🩸", "Blood Sugar",     "Limit refined carbs and sugary drinks — blood sugar spikes accelerate ageing and increase diabetes risk."),
    ("🏋️", "Strength",        "Resistance training twice a week preserves bone density and muscle mass as you age."),
    ("👁️", "Eye Care",        "Follow the 20-20-20 rule: every 20 minutes, look at something 20 feet away for 20 seconds."),
    ("🧠", "Brain Health",    "Learning new skills, reading, and social connection build cognitive reserve against dementia."),
    ("🫁", "Breathing",       "Diaphragmatic breathing for 5 minutes lowers cortisol and activates the parasympathetic nervous system."),
    ("🥚", "Protein",         "Adequate protein (0.8–1g per kg body weight) is essential for tissue repair and immune cell production."),
    ("🦵", "Posture",         "Stand and stretch every 30 minutes. Prolonged sitting raises the risk of metabolic syndrome."),
    ("🩺", "Screenings",      "Regular health check-ups catch silent conditions like hypertension and pre-diabetes before they escalate."),
    ("🌿", "Nature Therapy",  "Just 20 minutes outdoors in nature significantly lowers cortisol levels and improves mental well-being."),
]


@st.cache_data(show_spinner=False)
def _build_sidebar_chips_and_stats(query_count, msg_count, session_time, bookmarks):
    """Build stats row + chips HTML — cached on stats values."""
    sidebar_items = [
        ("🧠", "Model",            "Llama 3.3 70B"),
        ("🗄️", "Vector Search",    "FAISS"),
        ("📚", "Retrieved Context", f"{TOP_K_RESULTS} Documents"),
        ("⚡", "Provider",          "Groq API"),
    ]
    chips_html = "".join(
        f"""<div class="sb-chip">
    <div class="sb-chip-icon">{escape(icon)}</div>
    <div>
        <div class="sb-chip-label">{escape(label)}</div>
        <div class="sb-chip-value">{escape(value)}</div>
    </div>
</div>"""
        for icon, label, value in sidebar_items
    )
    stats_html = f"""<div class="sb-stats-row">
    <div class="sb-stat-card"><div class="sb-stat-number">{query_count}</div><div class="sb-stat-label">Queries</div></div>
    <div class="sb-stat-card"><div class="sb-stat-number">{bookmarks}</div><div class="sb-stat-label">Saved</div></div>
    <div class="sb-stat-card"><div class="sb-stat-number">{msg_count}</div><div class="sb-stat-label">Messages</div></div>
    <div class="sb-stat-card"><div class="sb-stat-number" style="font-size:14px">{session_time}</div><div class="sb-stat-label">Started</div></div>
</div>"""
    return stats_html, chips_html


def _get_tip_html(tip_index):
    tip_icon, tip_title, tip_body = HEALTH_TIPS[tip_index % len(HEALTH_TIPS)]
    display_num  = (tip_index % len(HEALTH_TIPS)) + 1
    total        = len(HEALTH_TIPS)
    arc_len      = round(2 * 3.14159 * 16 * (display_num / total), 2)
    return f"""<div class="sb-tip">
    <div class="sb-tip-num-col">
        <div class="sb-tip-num-ring">
            <svg class="sb-tip-arc-svg" viewBox="0 0 40 40" xmlns="http://www.w3.org/2000/svg">
                <defs>
                    <linearGradient id="sb-tip-grad-{display_num}" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%"   stop-color="#5eead4"/>
                        <stop offset="100%" stop-color="#a78bfa"/>
                    </linearGradient>
                </defs>
                <circle class="sb-tip-arc-bg"   cx="20" cy="20" r="16" />
                <circle class="sb-tip-arc-fill" cx="20" cy="20" r="16"
                    stroke="url(#sb-tip-grad-{display_num})"
                    stroke-dasharray="{arc_len} 100.53"
                    transform="rotate(-90 20 20)" />
            </svg>
            <div class="sb-tip-num-inner">
                <span class="sb-tip-num-icon">{escape(tip_icon)}</span>
            </div>
        </div>
        <div class="sb-tip-counter">{display_num}<span class="sb-tip-total">/{total}</span></div>
    </div>
    <div class="sb-tip-content">
        <div class="sb-tip-title">{escape(tip_title)}</div>
        <div class="sb-tip-body">{escape(tip_body)}</div>
    </div>
</div>"""


def render_sidebar():
    # ── Tip index state: random on first load, cycled on button press ──
    if "tip_index" not in st.session_state:
        st.session_state.tip_index = _random.randint(0, len(HEALTH_TIPS) - 1)

    with st.sidebar:
        st.markdown(
            f"""
<div class="sb-profile-card">
    <div class="sb-top">
        {sidebar_logo_html}
        <div>
            <div class="sb-kicker">✦ AI Health Assistant</div>
            <div class="sb-title">Medibot</div>
            <div class="sb-status"><span class="sb-status-dot"></span> Active &amp; Ready</div>
        </div>
    </div>
    <div class="sb-description">
        AI-powered health assistant that retrieves answers from your medical knowledge base with clear responses and source references.
    </div>
    <div style="display:flex;gap:6px;flex-wrap:wrap;margin-top:11px;">
        <span style="padding:4px 10px;border-radius:999px;background:rgba(var(--md-primary-rgb),.10);border:1px solid rgba(var(--md-primary-rgb),.22);font-size:10px;font-weight:700;color:rgba(200,190,255,.80);">🧠 RAG Engine</span>
        <span style="padding:4px 10px;border-radius:999px;background:rgba(var(--md-primary-rgb),.10);border:1px solid rgba(var(--md-primary-rgb),.22);font-size:10px;font-weight:700;color:rgba(200,190,255,.80);">⚡ Groq LLM</span>
        <span style="padding:4px 10px;border-radius:999px;background:rgba(var(--md-primary-rgb),.10);border:1px solid rgba(var(--md-primary-rgb),.22);font-size:10px;font-weight:700;color:rgba(200,190,255,.80);">🛡 Private</span>
        <span style="padding:4px 10px;border-radius:999px;background:rgba(var(--md-primary-rgb),.10);border:1px solid rgba(var(--md-primary-rgb),.22);font-size:10px;font-weight:700;color:rgba(200,190,255,.80);">🔍 Semantic Search</span>
    </div>
</div>
""",
            unsafe_allow_html=True,
        )

        # ── Session stats ──────────────────────
        query_count  = st.session_state.total_queries
        msg_count    = len(st.session_state.messages)
        session_time = st.session_state.session_start
        bookmarks    = len(st.session_state.bookmarked_answers)

        stats_html, chips_html = _build_sidebar_chips_and_stats(
            query_count, msg_count, session_time, bookmarks
        )

        st.markdown("<div class='sb-section'>📊 Session</div>", unsafe_allow_html=True)
        st.markdown(stats_html, unsafe_allow_html=True)

        # ── System info ────────────────────────
        st.markdown("<div class='sb-section'>⚙️ System</div>", unsafe_allow_html=True)
        st.markdown(chips_html, unsafe_allow_html=True)

        # ── Controls ───────────────────────────
        st.markdown("<div class='sb-section'>🎛️ Controls</div>", unsafe_allow_html=True)
        if st.button("✦ Start New Chat", use_container_width=True):
            st.session_state.messages           = []
            st.session_state.total_queries      = 0
            st.session_state.pending_bookmark   = None
            st.session_state.bookmark_saved_id  = None
            st.rerun()

        # ── Daily tip with Next Insight button ─
        st.markdown("<div class='sb-section'>💡 Health Insight</div>", unsafe_allow_html=True)
        st.markdown(
            _get_tip_html(st.session_state.tip_index),
            unsafe_allow_html=True,
        )
        st.markdown("<div class='sb-next-tip-wrap'>", unsafe_allow_html=True)
        if st.button("✦ Next Insight", key="next_tip_btn", use_container_width=True):
            st.session_state.tip_index = (st.session_state.tip_index + 1) % len(HEALTH_TIPS)
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            """
<div class="sb-note">
    <strong>⚠️ Medical Notice</strong><br>
    Medibot is for educational support only. For emergencies or serious symptoms, contact a qualified medical professional immediately.
</div>
<div class="sb-footer">
    Made with <span style="color:#ef4444">❤️</span> by
    <span class="sb-creator">Yatin Sharma</span>
</div>
""",
            unsafe_allow_html=True,
        )


render_sidebar()


# ─────────────────────────────────────────────
#  LLM / VECTOR STORE
# ─────────────────────────────────────────────
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY is missing. Add it to your .env file or environment variables.")
    st.stop()


@st.cache_resource(show_spinner=False)
def load_vectorstore():
    if not DB_FAISS_PATH.exists():
        st.error(f"Vector store not found at: {DB_FAISS_PATH}")
        st.stop()
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"batch_size": 32},
    )
    return FAISS.load_local(
        str(DB_FAISS_PATH), embedding_model, allow_dangerous_deserialization=True,
    )


_PROMPT_TEMPLATE = PromptTemplate(
    template="""
You are Medibot, a knowledgeable and helpful medical information assistant.

Use the retrieved context below as supporting information, then answer the question using your full medical knowledge.
Even if the context seems incomplete or only partially related, always give a thorough, helpful answer based on what you know about medicine.
Never refuse to answer a medical question just because the context is weak — use your knowledge instead.
Only say you cannot help if the question has absolutely nothing to do with health or medicine.

Rules:
- Give a clear, informative answer about the condition, symptoms, causes, and general management.
- Do not diagnose the user personally or prescribe specific medications.
- Keep the answer practical and easy to understand.
- Recommend consulting a qualified healthcare professional for personal decisions.
- For emergencies (chest pain, difficulty breathing, stroke signs, severe bleeding), advise immediate medical care.

Context (from knowledge base — use if helpful):
{context}

Question:
{question}

Answer:
""",
    input_variables=["context", "question"],
)


def get_prompt_template():
    return _PROMPT_TEMPLATE


@st.cache_resource(show_spinner=False)
def load_llm():
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY
    return ChatGroq(temperature=0.4, model_name="llama-3.3-70b-versatile")


@st.cache_resource(show_spinner=False)
def build_qa_chain(_vectorstore):
    return RetrievalQA.from_chain_type(
        llm=load_llm(),
        chain_type="stuff",
        retriever=_vectorstore.as_retriever(search_kwargs={"k": TOP_K_RESULTS}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": get_prompt_template()},
    )


def _warmup():
    """Pre-load vectorstore + LLM into cache in a background thread.
    Runs once per process; subsequent Streamlit reruns hit the cache instantly."""
    try:
        vs = load_vectorstore()
        build_qa_chain(vs)
    except Exception:
        pass  # silently ignore — errors shown to user on actual query


if "warmup_started" not in st.session_state:
    st.session_state["warmup_started"] = True
    threading.Thread(target=_warmup, daemon=True).start()


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False, max_entries=512)
def strip_html(value):
    return re.sub(r"<[^>]*>", " ", str(value)).strip()


def build_contextual_query(user_query):
    """
    Return a clean query string for FAISS vector search.
    Keep it as plain medical terms — never wrap in instructions.
    """
    previous_messages = st.session_state.get("messages", [])[-6:]
    if not previous_messages:
        return user_query

    # Only resolve obvious pronoun references; otherwise return the query as-is.
    lowered = user_query.lower().strip()
    pronoun_triggers = ["it", "this", "that", "its", "he", "she", "they", "them", "these", "those"]
    needs_resolution = any(lowered.startswith(p + " ") or lowered == p for p in pronoun_triggers)

    if not needs_resolution:
        return user_query

    # Prepend last user topic to resolve the pronoun
    for message in reversed(previous_messages):
        if message["role"] == "user":
            prev_content = message["content"].strip()
            return f"{prev_content} {user_query}"

    return user_query


def format_answer_text(text):
    """
    Convert the LLM plain-text response into MD3-expressive structured HTML.
    Handles: headings, bullets, numbered steps, callouts, bold/italic/code inline,
    keyword bold+underline highlighting.
    """
    import re as _re

    # ── Medical keywords to bold+underline automatically ──────────────────
    KEYWORDS = [
        "consult","doctor","physician","emergency","immediately","urgent",
        "seek medical","hospital","treatment","diagnosis","symptoms","medication",
        "side effects","dosage","warning","caution","important","note",
        "do not","avoid","never","always","recommend","prescribed","chronic",
        "acute","severe","mild","moderate","risk","complications","allergic",
        "contraindicated","overdose","infection","inflammation","surgery",
        "therapy","vaccine","antibiotic","painkiller","ibuprofen","paracetamol",
        "aspirin","blood pressure","diabetes","cancer","heart","lung","kidney",
        "liver","fever","pain","fatigue","nausea","vomiting","diarrhea",
    ]
    kw_pattern = _re.compile(
        r'\b(' + '|'.join(_re.escape(k) for k in KEYWORDS) + r')\b',
        _re.IGNORECASE
    )

    lines = str(text).split("\n")
    html_parts = []
    i = 0

    def highlight_keywords(s):
        """Wrap medical keywords in bold+underline span."""
        return kw_pattern.sub(
            lambda m: f'<span class="ar-kw">{m.group(0)}</span>', s
        )

    def inline(s):
        """Render bold, italic, inline-code, then keyword highlights."""
        s = _re.sub(r"`([^`]+)`",         r'<code class="ar-code">\1</code>', escape(s))
        s = _re.sub(r"\*\*\*(.+?)\*\*\*", r'<strong><em>\1</em></strong>',    s)
        s = _re.sub(r"\*\*(.+?)\*\*",     r'<strong>\1</strong>',              s)
        s = _re.sub(r"\*(.+?)\*",          r'<em>\1</em>',                     s)
        s = _re.sub(r"__(.+?)__",          r'<u>\1</u>',                       s)
        s = highlight_keywords(s)
        return s

    while i < len(lines):
        raw      = lines[i]
        stripped = raw.strip()

        if not stripped:
            i += 1
            continue

        # ── Callout lines: "> text" or lines starting with ⚠/📌/💡/ℹ️ ──
        if stripped.startswith("> ") or _re.match(r'^[⚠📌💡ℹ️🔴🟡🟢]\s', stripped):
            icon = "⚠️"
            body = stripped
            if stripped.startswith("> "):
                body = stripped[2:]
            elif stripped and stripped[0] in "⚠📌💡ℹ️🔴🟡🟢":
                icon = stripped[0]
                body = stripped[2:].strip()
            html_parts.append(
                f'<div class="ar-callout">'
                f'<span class="ar-callout-icon">{icon}</span>'
                f'<span>{inline(body)}</span></div>'
            )
            i += 1; continue

        # ── H1 (#) ────────────────────────────────────────────────────────
        if stripped.startswith("# "):
            html_parts.append(f'<div class="ar-h1">{inline(stripped[2:])}</div>')
            i += 1; continue

        # ── H2 (##) ───────────────────────────────────────────────────────
        if stripped.startswith("## "):
            html_parts.append(f'<div class="ar-h2">{inline(stripped[3:])}</div>')
            i += 1; continue

        # ── H3 (###) ──────────────────────────────────────────────────────
        if stripped.startswith("### "):
            html_parts.append(f'<div class="ar-h3">{inline(stripped[4:])}</div>')
            i += 1; continue

        # ── Bold-only line → subheading ────────────────────────────────────
        bold_only = _re.fullmatch(r"\*\*(.+?)\*\*:?", stripped)
        if bold_only:
            html_parts.append(f'<div class="ar-h3">{escape(bold_only.group(1))}</div>')
            i += 1; continue

        # ── Numbered list ──────────────────────────────────────────────────
        if _re.match(r"^\d+\.\s+", stripped):
            items = []
            while i < len(lines):
                m = _re.match(r"^(\d+)\.\s+(.*)", lines[i].strip())
                if not m: break
                items.append(
                    f'<li class="ar-li-num">'
                    f'<span class="ar-li-num-badge">{m.group(1)}</span>'
                    f'<span style="flex:1;min-width:0;display:block;word-break:break-word;">{inline(m.group(2))}</span></li>'
                )
                i += 1
            html_parts.append(f'<ol class="ar-ol">{"".join(items)}</ol>')
            continue

        # ── Bullet list (-, *, •) ──────────────────────────────────────────
        if _re.match(r"^[-*•]\s+", stripped):
            items = []
            while i < len(lines):
                m = _re.match(r"^[-*•]\s+(.*)", lines[i].strip())
                if not m: break
                items.append(f'<li class="ar-li"><span class="ar-li-text">{inline(m.group(1))}</span></li>')
                i += 1
            html_parts.append(f'<ul class="ar-ul">{"".join(items)}</ul>')
            continue

        # ── Horizontal rule ────────────────────────────────────────────────
        if stripped in ("---", "___", "***"):
            html_parts.append('<hr class="ar-hr">')
            i += 1; continue

        # ── Plain paragraph ────────────────────────────────────────────────
        html_parts.append(f'<p class="ar-p">{inline(stripped)}</p>')
        i += 1

    return "\n".join(html_parts)


def estimate_confidence(source_documents, result_text):
    """Heuristic confidence based on source count and result length."""
    if not source_documents:
        return "low"
    if len(source_documents) >= 3 and len(result_text) > 200:
        return "high"
    if len(source_documents) >= 2:
        return "medium"
    return "low"


def render_confidence_badge(level):
    labels = {"high": "High Confidence", "medium": "Moderate Confidence", "low": "Low Confidence"}
    icons  = {"high": "✔", "medium": "◑", "low": "⚠"}
    css    = {"high": "md-confidence-high", "medium": "md-confidence-med", "low": "md-confidence-low"}
    return f'<div class="md-confidence-badge {css[level]}">{icons[level]} {labels[level]}</div>'


def format_sources(source_documents):
    if not source_documents:
        return """
<div class="md-source-list">
    <div class="md-source-title">📄 Sources</div>
    <div class="md-source-text">No sources found.</div>
</div>
"""
    seen_sources = []
    for doc in source_documents:
        source = doc.metadata.get("source", "Unknown Source")
        page   = doc.metadata.get("page")
        source_text = f"{source}, page {page}" if page is not None else source
        if source_text not in seen_sources:
            seen_sources.append(source_text)
    source_html = ""
    for idx, source in enumerate(seen_sources, start=1):
        source_html += f"""
<div class="md-source-item">
    <div class="md-source-number">{idx}</div>
    <div class="md-source-text">{escape(str(source))}</div>
</div>
"""
    return f"""
<div class="md-source-list">
    <div class="md-source-title">📄 Sources Used</div>
    {source_html}
</div>
"""


# ─────────────────────────────────────────────
#  HERO
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _build_header_html(logo_html_str):
    return f"""
<div class="md-hero">
    <div class="md-hero-orb md-hero-orb-1"></div>
    <div class="md-hero-orb md-hero-orb-2"></div>
    <div class="md-hero-inner">
        <div>
            <div class="md-brand">
                {logo_html_str}
                <div>
                    <div class="md-kicker">✦ AI Health Assistant</div>
                    <h1 class="md-title">Medibot</h1>
                    <div class="md-subtitle">
                        Ask health questions and get clean, context-aware answers from your medical knowledge base with full source transparency.
                    </div>
                </div>
            </div>
            <div class="md-hero-stats">
                <div class="md-hero-stat"><span class="md-hero-stat-dot"></span> RAG-Powered Retrieval</div>
                <div class="md-hero-stat"><span class="md-hero-stat-dot"></span> Source-Cited Answers</div>
                <div class="md-hero-stat"><span class="md-hero-stat-dot"></span> Conversation-Aware</div>
                <div class="md-hero-stat"><span class="md-hero-stat-dot"></span> No Hallucinations</div>
            </div>
        </div>
        <div class="md-status">
            <div class="md-pill">
                <div class="md-pill-label">Model</div>
                <div class="md-pill-value">Meta Llama via Groq</div>
            </div>
            <div class="md-pill">
                <div class="md-pill-label">Knowledge</div>
                <div class="md-pill-value">FAISS Medical Search</div>
            </div>
            <div class="md-pill">
                <div class="md-pill-label">Mode</div>
                <div class="md-pill-value">Source-Aware RAG</div>
            </div>
        </div>
    </div>
</div>
"""


def render_header():
    st.markdown(_build_header_html(logo_html), unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  FEATURE CARDS
# ─────────────────────────────────────────────
_INFO_CARDS_HTML = """
<div class="md-card-grid">
    <div class="md-info-card">
        <div class="md-card-icon">📖</div>
        <strong>Context Based</strong>
        <span>Answers are retrieved from your medical knowledge base before the AI responds — no internet guessing.</span>
    </div>
    <div class="md-info-card">
        <div class="md-card-icon">🧵</div>
        <strong>Conversation Aware</strong>
        <span>Recent chat context helps Medibot understand follow-up questions and pronoun references naturally.</span>
    </div>
    <div class="md-info-card">
        <div class="md-card-icon">📑</div>
        <strong>Source Transparent</strong>
        <span>Relevant sources and confidence level are shown below each answer for full traceability.</span>
    </div>
</div>
"""


def render_info_cards():
    st.markdown(_INFO_CARDS_HTML, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  HEALTH TOPICS QUICK ACCESS
# ─────────────────────────────────────────────
HEALTH_TOPICS = [
    ("❤️",  "Cardiology",     "Heart & circulation",    "What are the warning signs of a heart attack?"),
    ("🧠",  "Neurology",      "Brain & nervous system", "What are early signs of a neurological disorder?"),
    ("🫁",  "Pulmonology",    "Lungs & breathing",      "What causes chronic shortness of breath?"),
    ("💊",  "Pharmacology",   "Drugs & medications",    "What are common side effects of NSAIDs?"),
    ("🩺",  "General Health", "Preventive care",        "What routine health screenings should adults get?"),
    ("🦴",  "Orthopedics",    "Bones & joints",         "What are symptoms of osteoporosis?"),
    ("🧬",  "Genetics",       "Hereditary conditions",  "What genetic conditions are most common?"),
    ("🍎",  "Nutrition",      "Diet & wellness",        "What diet helps reduce inflammation?"),
]


@st.cache_data(show_spinner=False)
def _build_topics_html():
    cards = "".join(
        f"""<div class="md-topic-card">
  <span class="md-topic-emoji">{escape(emoji)}</span>
  <div class="md-topic-label">{escape(label)}</div>
  <div class="md-topic-sub">{escape(sub)}</div>
</div>"""
        for emoji, label, sub, _ in HEALTH_TOPICS
    )
    return cards


def render_health_topics():
    st.markdown("<div class='md-section-title'>Browse by Topic</div>", unsafe_allow_html=True)
    st.markdown("<div class='md-section-sub'>Click a topic to ask a starter question</div>", unsafe_allow_html=True)

    cols = st.columns(8)
    topic_query = None
    cards_html = _build_topics_html()
    card_list = [c for c in cards_html.split("</div>") if c.strip()]  # re-split per card

    for i, (emoji, label, sub, query) in enumerate(HEALTH_TOPICS):
        with cols[i]:
            st.markdown(
                f"""<div class="md-topic-card">
  <span class="md-topic-emoji">{escape(emoji)}</span>
  <div class="md-topic-label">{escape(label)}</div>
  <div class="md-topic-sub">{escape(sub)}</div>
</div><div class="md-topic-btn">""",
                unsafe_allow_html=True,
            )
            if st.button("↗", use_container_width=True, key=f"topic_{i}"):
                topic_query = query
            st.markdown("</div>", unsafe_allow_html=True)
    return topic_query


# ─────────────────────────────────────────────
#  SUGGESTION CHIPS
# ─────────────────────────────────────────────
SUGGESTIONS = [
    ("🫀", "Signs of heart disease",         "What are common signs of heart disease?"),
    ("🩸", "Lower diabetes risk",             "How can someone lower their risk of diabetes?"),
    ("💢", "High blood pressure causes",      "What are common causes of high blood pressure?"),
    ("😴", "Improve sleep quality",           "What are evidence-based ways to improve sleep quality?"),
    ("🧘", "Reduce anxiety naturally",        "What natural methods help reduce anxiety and stress?"),
    ("🦠", "Boost immune system",             "How can I strengthen my immune system naturally?"),
]


def render_suggestion_chips():
    cols = st.columns(3)
    chip_query = None
    for i, (icon, label, query) in enumerate(SUGGESTIONS):
        with cols[i % 3]:
            if st.button(f"{icon}  {label}", use_container_width=True, key=f"chip_{i}"):
                chip_query = query
    return chip_query


# ─────────────────────────────────────────────
#  BOOKMARKS TAB
# ─────────────────────────────────────────────
def render_bookmarks_tab():
    bookmarks = st.session_state.bookmarked_answers
    if not bookmarks:
        st.markdown(
            """
<div class="md-empty">
    <span class="md-empty-icon">🔖</span>
    No saved answers yet. After receiving a response, use the <strong>Save Answer</strong> button to bookmark it here for future reference.
</div>
""",
            unsafe_allow_html=True,
        )
        return
    for i, bm in enumerate(reversed(bookmarks)):
        with st.expander(f"🔖 {bm['question'][:80]}{'...' if len(bm['question'])>80 else ''}", expanded=False):
            st.markdown(f"**Asked:** {bm['time']}")
            st.markdown(bm["answer"], unsafe_allow_html=True)
            if st.button("🗑️ Remove", key=f"remove_bm_{i}"):
                real_idx = len(bookmarks) - 1 - i
                st.session_state.bookmarked_answers.pop(real_idx)
                st.rerun()


# ─────────────────────────────────────────────
#  QUERY HANDLER
# ─────────────────────────────────────────────
def handle_user_query(user_query, qa_chain):
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.session_state.total_queries += 1
    st.session_state.last_query_time = datetime.now().strftime("%H:%M:%S")

    with st.chat_message("user", avatar="👤"):
        st.markdown(user_query)

    with st.chat_message("assistant", avatar="🤖"):
        loading_placeholder = st.empty()
        loading_placeholder.markdown(
            """
<div class="md-loading-card">
  <div class="md-loading-spinner"></div>
  <div class="md-loading-text">
    <span class="md-loading-title">Medibot is analysing your question…</span>
    <span class="md-loading-sub">Searching medical knowledge base &amp; generating response</span>
  </div>
</div>
<style>
.md-loading-card {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 18px 22px;
    border-radius: 20px;
    border: 1px solid rgba(124,92,191,0.28);
    background: linear-gradient(135deg, rgba(124,92,191,0.10), rgba(0,133,122,0.06));
    backdrop-filter: blur(8px);
    box-shadow: 0 4px 20px rgba(10,15,30,0.12);
    margin: 4px 0;
    animation: md-load-fadein 0.3s ease both;
}
@keyframes md-load-fadein {
    from { opacity: 0; transform: translateY(6px); }
    to   { opacity: 1; transform: translateY(0); }
}
.md-loading-spinner {
    flex-shrink: 0;
    width: 32px;
    height: 32px;
    border-radius: 50%;
    border: 3px solid rgba(124,92,191,0.18);
    border-top-color: #7c5cbf;
    border-right-color: #00857a;
    animation: md-spin 0.85s cubic-bezier(0.4,0,0.2,1) infinite;
}
@keyframes md-spin {
    to { transform: rotate(360deg); }
}
.md-loading-text {
    display: flex;
    flex-direction: column;
    gap: 3px;
}
.md-loading-title {
    font-family: 'Sora', sans-serif;
    font-size: 14px;
    font-weight: 700;
    color: #d7c7ff;
    letter-spacing: 0.01em;
}
.md-loading-sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 12px;
    color: rgba(203,213,225,0.60);
    font-style: italic;
}
</style>
""",
            unsafe_allow_html=True,
        )
        try:
            contextual_query = build_contextual_query(user_query)
            response = qa_chain.invoke({"query": contextual_query})

            result  = response.get("result", "No response was generated.")
            sources = response.get("source_documents", [])

            confidence   = estimate_confidence(sources, result)
            conf_badge   = render_confidence_badge(confidence)
            sources_html = format_sources(sources)

            formatted_response = (
                f"{conf_badge}"
                f'<div class="ar-answer-card"><div class="ar-answer-body">{format_answer_text(result)}</div></div>'
                f"{sources_html}"
            )
            loading_placeholder.empty()
            st.markdown(formatted_response, unsafe_allow_html=True)

            # Store pending bookmark so the save button can be rendered
            # safely OUTSIDE the chat_message context (avoids key conflicts)
            st.session_state.pending_bookmark = {
                "question": user_query,
                "answer":   formatted_response,
                "time":     datetime.now().strftime("%b %d, %H:%M"),
                "id":       st.session_state.total_queries,
            }

            st.session_state.messages.append({
                "role":    "assistant",
                "content": formatted_response,
            })

            if len(st.session_state.messages) > MAX_CHAT_MESSAGES:
                st.session_state.messages = st.session_state.messages[-MAX_CHAT_MESSAGES:]

        except Exception as error:
            loading_placeholder.empty()
            error_message = f"Something went wrong: {str(error)}"
            st.error(error_message)
            st.session_state.messages.append({
                "role":    "assistant",
                "content": escape(error_message),
            })


# ─────────────────────────────────────────────
#  SYMPTOM CHECKER
# ─────────────────────────────────────────────
SYMPTOMS = [
    ("🤒", "Fever",           "High temperature"),
    ("🤕", "Headache",        "Head pain"),
    ("😮‍💨", "Shortness of Breath", "Breathing difficulty"),
    ("🤢", "Nausea",          "Upset stomach"),
    ("💪", "Body Aches",      "Muscle pain"),
    ("😴", "Fatigue",         "Low energy"),
    ("🫀", "Chest Pain",      "Chest tightness"),
    ("🤧", "Runny Nose",      "Nasal discharge"),
    ("🌀", "Dizziness",       "Light-headedness"),
    ("🙄", "Blurred Vision",  "Eye issues"),
    ("🦷", "Sore Throat",     "Throat pain"),
    ("🤑", "Sweating",        "Excessive sweat"),
]

SEVERITY_LEVELS = ["Mild", "Moderate", "Severe"]

SYMPTOM_MAP = {
    frozenset(["Fever", "Headache", "Body Aches", "Fatigue"]): {
        "title": "Possible Flu / Viral Infection",
        "body": "Your symptom combination is consistent with influenza or a similar viral illness. Rest, hydration, and over-the-counter fever reducers are typically recommended. Monitor for worsening symptoms.",
        "urgency": "low",
    },
    frozenset(["Fever", "Shortness of Breath", "Chest Pain"]): {
        "title": "Seek Medical Attention",
        "body": "Chest pain combined with fever and breathing difficulty could indicate a serious respiratory or cardiac condition. Please contact a healthcare provider promptly.",
        "urgency": "high",
    },
    frozenset(["Headache", "Blurred Vision", "Dizziness"]): {
        "title": "Possible Hypertensive Episode or Migraine",
        "body": "This combination may suggest elevated blood pressure or a severe migraine. Avoid bright lights, rest, and measure your blood pressure if possible. Seek care if symptoms worsen.",
        "urgency": "medium",
    },
    frozenset(["Nausea", "Body Aches", "Fatigue"]): {
        "title": "Possible Gastroenteritis or Systemic Infection",
        "body": "These symptoms are commonly associated with a stomach bug or mild systemic infection. Stay hydrated with clear fluids. If vomiting is persistent or you develop a high fever, consult a doctor.",
        "urgency": "low",
    },
    frozenset(["Chest Pain", "Sweating", "Shortness of Breath"]): {
        "title": "⚠ Possible Cardiac Event — Call Emergency Services",
        "body": "This triad of symptoms can indicate a heart attack or serious cardiac event. Do NOT wait — call emergency services (112 / 911) immediately.",
        "urgency": "high",
    },
}

def get_symptom_result(selected, severity):
    # Try exact frozenset match first
    key = frozenset(selected)
    for k, v in SYMPTOM_MAP.items():
        if k.issubset(key) or key.issubset(k):
            result = dict(v)
            if severity == "Severe" and result["urgency"] == "low":
                result["urgency"] = "medium"
            elif severity == "Severe" and result["urgency"] == "medium":
                result["urgency"] = "high"
            return result
    # Default fallback
    urgency = "low" if severity == "Mild" else ("medium" if severity == "Moderate" else "high")
    return {
        "title": f"{len(selected)} Symptom(s) Noted — {severity} Severity",
        "body": (
            "Based on the symptoms you've selected, it's recommended to monitor your condition closely. "
            "Ensure you are well-rested and hydrated. If symptoms persist beyond 48 hours or worsen, "
            "please consult a qualified healthcare professional. This is not a diagnosis."
        ),
        "urgency": urgency,
    }


def render_symptom_checker():
    st.markdown("<div class='sym-panel'>", unsafe_allow_html=True)
    st.markdown("<div class='md-section-title'>🩺 Symptom Checker</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='md-section-sub'>Select your symptoms and severity for a general guidance overview. "
        "<strong>Not a medical diagnosis.</strong></div>",
        unsafe_allow_html=True,
    )

    # Symptom grid — rendered as HTML labels, toggled by st.multiselect hidden below
    selected = st.session_state.selected_symptoms

    # Build grid HTML
    grid_html = "<div class='sym-grid'>"
    for icon, label, sub in SYMPTOMS:
        sel_cls = "selected" if label in selected else ""
        grid_html += (
            f"<div class='sym-chip {sel_cls}'>"
            f"<span class='sym-chip-icon'>{icon}</span>"
            f"<div class='sym-chip-label'>{label}</div>"
            f"<div class='sym-chip-sub'>{sub}</div>"
            f"</div>"
        )
    grid_html += "</div>"
    st.markdown(grid_html, unsafe_allow_html=True)

    # Actual interactive multiselect (functional but styled minimally)
    sym_labels = [s[1] for s in SYMPTOMS]
    chosen = st.multiselect(
        "Select symptoms",
        options=sym_labels,
        default=st.session_state.selected_symptoms,
        key="sym_multiselect",
        label_visibility="collapsed",
    )
    st.session_state.selected_symptoms = chosen

    # Severity
    st.markdown("<div style='margin:10px 0 4px 0;font-size:13px;font-weight:800;font-family:var(--font-display)'>Severity Level</div>", unsafe_allow_html=True)
    sev_cols = st.columns(3)
    for i, sev in enumerate(SEVERITY_LEVELS):
        with sev_cols[i]:
            if st.button(sev, key=f"sev_{sev}", use_container_width=True):
                st.session_state.symptom_severity = sev
                st.session_state.symptom_result = None

    st.markdown(f"<div style='font-size:12px;color:var(--md-soft);margin:4px 0 12px 0'>Selected severity: <strong>{st.session_state.symptom_severity}</strong></div>", unsafe_allow_html=True)

    # Analyse button
    col_btn, col_clr = st.columns([2, 1])
    with col_btn:
        if st.button("🔍 Analyse Symptoms", use_container_width=True, key="sym_analyse"):
            if chosen:
                st.session_state.symptom_result = get_symptom_result(chosen, st.session_state.symptom_severity)
            else:
                st.warning("Please select at least one symptom.")
    with col_clr:
        if st.button("↺ Clear", use_container_width=True, key="sym_clear"):
            st.session_state.selected_symptoms = []
            st.session_state.symptom_result = None
            st.rerun()

    # Result
    result = st.session_state.symptom_result
    if result:
        urgency = result["urgency"]
        urgency_css = {"low": "sym-urgency-low", "medium": "sym-urgency-medium", "high": "sym-urgency-high"}[urgency]
        urgency_label = {"low": "✔ Low Urgency", "medium": "◑ Moderate Urgency", "high": "⚠ High Urgency — Seek Care"}[urgency]

        # Send to chat button
        if st.button("💬 Ask Medibot about this", key="sym_to_chat", use_container_width=False):
            symptom_str = ", ".join(chosen)
            st.session_state._sym_chat_query = (
                f"I have the following symptoms: {symptom_str}. "
                f"Severity: {st.session_state.symptom_severity}. "
                f"What could this indicate and what should I do?"
            )
            st.rerun()

        st.markdown(
            f"""
<div class='sym-result md-animate-in'>
    <div class='sym-urgency-pill {urgency_css}'>{urgency_label}</div>
    <div class='sym-result-title'>{escape(result['title'])}</div>
    <div class='sym-result-body'>{escape(result['body'])}</div>
    <div style='margin-top:10px;font-size:11px;color:var(--md-soft)'>
        ⚠ This is general guidance only. Always consult a qualified medical professional.
    </div>
</div>
""",
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  MEDICATION REMINDERS
# ─────────────────────────────────────────────
def render_medication_reminders():
    st.markdown("<div class='rem-panel'>", unsafe_allow_html=True)
    st.markdown("<div class='md-section-title'>💊 Medication Reminders</div>", unsafe_allow_html=True)
    st.markdown("<div class='md-section-sub'>Track your daily medications for this session</div>", unsafe_allow_html=True)

    reminders = st.session_state.reminders

    status_dot = {"taken": "rem-dot-green", "pending": "rem-dot-amber", "missed": "rem-dot-red"}
    status_badge = {"taken": "rem-badge-taken", "pending": "rem-badge-pending", "missed": "rem-badge-missed"}
    status_label = {"taken": "Taken ✔", "pending": "Pending", "missed": "Missed ✗"}

    for i, rem in enumerate(reminders):
        dot_cls = status_dot.get(rem["status"], "rem-dot-amber")
        badge_cls = status_badge.get(rem["status"], "rem-badge-pending")
        label = status_label.get(rem["status"], "Pending")
        st.markdown(
            f"""
<div class='rem-item'>
    <div class='rem-dot {dot_cls}'></div>
    <div>
        <div class='rem-name'>{escape(rem['name'])}</div>
        <div class='rem-dose'>{escape(rem['dose'])}</div>
    </div>
    <div class='rem-time'>{escape(rem['time'])}</div>
    <div class='rem-badge {badge_cls}'>{label}</div>
</div>
""",
            unsafe_allow_html=True,
        )
        col_t, col_m, col_sp = st.columns([1, 1, 3])
        with col_t:
            if rem["status"] != "taken":
                if st.button("✔ Taken", key=f"rem_taken_{i}", use_container_width=True):
                    st.session_state.reminders[i]["status"] = "taken"
                    st.rerun()
        with col_m:
            if rem["status"] == "pending":
                if st.button("✗ Missed", key=f"rem_missed_{i}", use_container_width=True):
                    st.session_state.reminders[i]["status"] = "missed"
                    st.rerun()

    st.markdown("<div class='md-section-sub' style='margin-top:14px'>Add a new reminder</div>", unsafe_allow_html=True)
    col_n, col_d, col_ti, col_add = st.columns([2, 1, 1, 1])
    with col_n:
        new_name = st.text_input("Medicine name", key="rem_name_input", label_visibility="collapsed", placeholder="Medicine name")
    with col_d:
        new_dose = st.text_input("Dose", key="rem_dose_input", label_visibility="collapsed", placeholder="Dose e.g. 500mg")
    with col_ti:
        new_time = st.text_input("Time", key="rem_time_input", label_visibility="collapsed", placeholder="e.g. 08:00 AM")
    with col_add:
        if st.button("➕ Add", use_container_width=True, key="rem_add_btn"):
            if new_name.strip():
                st.session_state.reminders.append({
                    "name": new_name.strip(),
                    "dose": new_dose.strip() or "—",
                    "time": new_time.strip() or "—",
                    "status": "pending",
                })
                st.rerun()
            else:
                st.warning("Enter a medicine name.")

    if st.button("↺ Reset All Reminders", key="rem_reset"):
        for r in st.session_state.reminders:
            r["status"] = "pending"
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  HEALTH SCORE QUIZ
# ─────────────────────────────────────────────
HEALTH_QUIZ = [
    ("exercise", "How often do you exercise?",
     ["Rarely / Never", "1–2 times a week", "3–4 times a week", "Daily"],
     [10, 40, 70, 100]),
    ("sleep", "How many hours of sleep do you typically get?",
     ["Less than 5h", "5–6h", "7–8h", "More than 9h"],
     [15, 45, 100, 70]),
    ("diet", "How would you rate your diet?",
     ["Mostly fast food", "Mixed — some healthy", "Mostly healthy", "Very healthy / balanced"],
     [10, 40, 75, 100]),
    ("water", "How much water do you drink daily?",
     ["Less than 1L", "1–2L", "2–3L", "More than 3L"],
     [15, 50, 90, 100]),
    ("stress", "How often do you feel stressed or anxious?",
     ["Almost always", "Often", "Sometimes", "Rarely"],
     [10, 35, 70, 100]),
]

def render_health_score():
    # ── Init step state ────────────────────────
    if "hs_step" not in st.session_state:
        st.session_state.hs_step = 0
    if "hs_selections" not in st.session_state:
        st.session_state.hs_selections = {}

    total_steps = len(HEALTH_QUIZ)
    step        = st.session_state.hs_step
    selections  = st.session_state.hs_selections
    done        = step >= total_steps and len(selections) == total_steps

    st.markdown("<div class='hs-panel'>", unsafe_allow_html=True)

    # Header
    st.markdown(
        "<div class='md-section-title'>📊 Lifestyle Health Score</div>"
        "<div class='md-section-sub'>5 quick questions · Instant score · No data stored</div>",
        unsafe_allow_html=True,
    )

    # ── Progress bar ───────────────────────────
    segs = ""
    for i in range(total_steps):
        if i < step:
            cls = "done"
        elif i == step and not done:
            cls = "active"
        else:
            cls = ""
        segs += f"<div class='hs-progress-seg {cls}'></div>"
    st.markdown(f"<div class='hs-progress-wrap'>{segs}</div>", unsafe_allow_html=True)

    # ── Quiz cards ─────────────────────────────
    if not done:
        key, question, options, scores = HEALTH_QUIZ[step]
        current_val = selections.get(key)

        # Question header card
        st.markdown(
            f"<div class='hs-q-card'>"
            f"<div class='hs-q-step'>Question {step + 1} of {total_steps}</div>"
            f"<div class='hs-q-text'>{escape(question)}</div>"
            f"<div class='hs-options-wrap'>",
            unsafe_allow_html=True,
        )

        # Real buttons styled as option cards — 2 per row
        btn_cols = st.columns(2)
        for i, opt in enumerate(options):
            sel_wrap = "hs-opt-selected" if opt == current_val else ""
            with btn_cols[i % 2]:
                st.markdown(f"<div class='{sel_wrap}'>", unsafe_allow_html=True)
                if st.button(opt, key=f"hs_opt_{step}_{i}", use_container_width=True):
                    st.session_state.hs_selections[key] = opt
                    st.session_state[f"hs_score_{key}"] = scores[i]
                    st.session_state.hs_step = step + 1
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div></div>", unsafe_allow_html=True)

        # Back button
        if step > 0:
            st.markdown("<div class='hs-back-wrap'>", unsafe_allow_html=True)
            if st.button("← Back", key="hs_back"):
                st.session_state.hs_step = max(0, step - 1)
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

    # ── Result ─────────────────────────────────
    else:
        factor_meta = {
            "exercise": ("🏃", "Exercise"),
            "sleep":    ("😴", "Sleep"),
            "diet":     ("🥦", "Diet"),
            "water":    ("💧", "Hydration"),
            "stress":   ("🧘", "Stress"),
        }
        score_vals = {k: st.session_state.get(f"hs_score_{k}", 50) for k in factor_meta}
        total = int(sum(score_vals.values()) / len(score_vals))

        # Tier
        if total >= 85:
            tier, tier_label, arc_cls, badge_cls, bg_cls, emoji, headline, sub = (
                "excellent", "Excellent", "hs-arc-excellent", "hs-badge-excellent",
                "hs-result-bg-excellent", "🏆",
                "Outstanding lifestyle!",
                "You're in the top tier. Keep up the great habits — your body and mind will thank you.",
            )
        elif total >= 65:
            tier, tier_label, arc_cls, badge_cls, bg_cls, emoji, headline, sub = (
                "good", "Good", "hs-arc-good", "hs-badge-good",
                "hs-result-bg-good", "👍",
                "Solid healthy habits",
                "You're doing well. A few small improvements in weaker areas could push you to excellent.",
            )
        elif total >= 45:
            tier, tier_label, arc_cls, badge_cls, bg_cls, emoji, headline, sub = (
                "fair", "Fair", "hs-arc-fair", "hs-badge-fair",
                "hs-result-bg-fair", "🟡",
                "Room to grow",
                "Your lifestyle has some healthy habits but also areas that need attention. Small changes add up fast.",
            )
        else:
            tier, tier_label, arc_cls, badge_cls, bg_cls, emoji, headline, sub = (
                "poor", "Needs Attention", "hs-arc-poor", "hs-badge-poor",
                "hs-result-bg-poor", "⚠",
                "Time to take action",
                "Your current habits may be impacting your health. Consider asking Medibot for guidance on specific areas.",
            )

        # SVG arc — circumference of r=45 circle ≈ 283
        arc_len = 283
        fill_len = round(arc_len * total / 100, 1)
        offset   = round(arc_len - fill_len, 1)

        # Gradient defs per tier
        grad_map = {
            "excellent": ('<stop offset="0%" stop-color="#22c55e"/><stop offset="100%" stop-color="#00857a"/>'),
            "good":      ('<stop offset="0%" stop-color="#6750a4"/><stop offset="100%" stop-color="#00857a"/>'),
            "fair":      ('<stop offset="0%" stop-color="#d97706"/><stop offset="100%" stop-color="#f59e0b"/>'),
            "poor":      ('<stop offset="0%" stop-color="#ef4444"/><stop offset="100%" stop-color="#d97706"/>'),
        }
        grad_stops = grad_map[tier]

        # Factor cards HTML
        factors_html = "<div class='hs-factors-v2'>"
        for k, (icon, label) in factor_meta.items():
            v = score_vals[k]
            fill_scale = round(v / 100, 2)
            grade = "Great" if v >= 80 else "Good" if v >= 60 else "Fair" if v >= 40 else "Low"
            factors_html += (
                f"<div class='hs-factor-v2' style='--fill:{fill_scale}'>"
                f"<div class='hs-fv2-icon'>{icon}</div>"
                f"<div class='hs-fv2-label'>{label}</div>"
                f"<div class='hs-fv2-val'>{v}</div>"
                f"<div class='hs-fv2-sub'>{grade}</div>"
                f"</div>"
            )
        factors_html += "</div>"

        st.markdown(
            f"""
<div class='hs-result {bg_cls}'>
    <div class='hs-result-top'>
        <div class='hs-arc-wrap'>
            <svg class='hs-arc-svg' viewBox='0 0 110 110' xmlns='http://www.w3.org/2000/svg'>
                <defs>
                    <linearGradient id='hs-grad-{tier}' x1='0%' y1='0%' x2='100%' y2='0%'>
                        {grad_stops}
                    </linearGradient>
                </defs>
                <circle class='hs-arc-bg' cx='55' cy='55' r='45'/>
                <circle class='hs-arc-fill {arc_cls}'
                    cx='55' cy='55' r='45'
                    stroke-dasharray='{arc_len}'
                    stroke-dashoffset='{offset}'/>
            </svg>
            <div class='hs-arc-center'>
                <div class='hs-arc-num'>{total}</div>
                <div class='hs-arc-denom'>/100</div>
            </div>
        </div>
        <div class='hs-result-info'>
            <div class='hs-result-badge {badge_cls}'>{emoji} {tier_label}</div>
            <div class='hs-result-headline'>{headline}</div>
            <div class='hs-result-sub'>{sub}</div>
        </div>
    </div>
    {factors_html}
</div>
""",
            unsafe_allow_html=True,
        )

        # Tip for low scorers
        if total < 50:
            st.markdown(
                "<div class='sb-tip' style='margin-top:16px'>"
                "<span class='sb-tip-icon'>💡</span>"
                "<div>Ask Medibot for personalised tips — try <em>\"How can I improve my sleep quality?\"</em> or <em>\"What diet helps reduce stress?\"</em></div>"
                "</div>",
                unsafe_allow_html=True,
            )

        # Retake
        if st.button("↺  Retake Quiz", key="hs_retake"):
            st.session_state.hs_step = 0
            st.session_state.hs_selections = {}
            st.session_state.health_score_answers = None
            for k in factor_meta:
                st.session_state.pop(f"hs_score_{k}", None)
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  CHAT EXPORT
# ─────────────────────────────────────────────
def export_chat_as_text():
    lines = []
    lines.append("═══ Medibot Chat Export ═══")
    lines.append(f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    for msg in st.session_state.messages:
        role = "You" if msg["role"] == "user" else "Medibot"
        content = strip_html(msg["content"])
        lines.append(f"[{role}]")
        lines.append(content)
        lines.append("")
    return "\n".join(lines)


@st.cache_data(show_spinner=False, max_entries=64)
def _compute_chat_stats(messages_tuple):
    """Cached: compute word count from message list (passed as tuple of (role,content) pairs)."""
    return sum(len(strip_html(content).split()) for _, content in messages_tuple)


def render_chat_tools_bar():
    """Toolbar shown above chat when there are messages."""
    if not st.session_state.messages:
        return

    messages = st.session_state.messages
    msg_count  = len(messages)
    word_count = _compute_chat_stats(tuple((m["role"], m["content"]) for m in messages))
    read_mins  = max(1, word_count // 200)

    st.markdown(
        f"""
<div class='md-tools-bar md-animate-in'>
    <span class='md-tool-label'>Chat Tools</span>
    <div class='md-tool-divider'></div>
    <span class='md-read-meta'>📖 ~{read_mins} min read · {word_count} words · {msg_count} messages</span>
</div>
""",
        unsafe_allow_html=True,
    )

    col_exp, col_search, col_clr = st.columns([1, 2, 1])
    with col_exp:
        # Key includes msg_count so Streamlit regenerates the file whenever chat changes
        chat_text = export_chat_as_text()
        st.download_button(
            label="⬇ Export Chat",
            data=chat_text,
            file_name=f"medibot_chat_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain",
            use_container_width=True,
            key=f"export_chat_btn_{msg_count}",
        )
    with col_search:
        # Use key only — do NOT pass value= so Streamlit owns the widget state
        search_q = st.text_input(
            "Search chat",
            placeholder="🔍 Search in conversation…",
            key="chat_search_input",
            label_visibility="collapsed",
        )
        st.session_state.chat_search_query = search_q
    with col_clr:
        if st.button("🗑 Clear Chat", use_container_width=True, key="clear_chat_btn"):
            st.session_state.messages = []
            st.session_state.pending_bookmark   = None
            st.session_state.bookmark_saved_id  = None
            st.session_state.chat_search_query  = ""
            st.rerun()


def messages_to_display():
    """Filter messages by chat search query if set."""
    q = st.session_state.get("chat_search_input", "").strip().lower()
    if not q:
        return st.session_state.messages
    return [
        m for m in st.session_state.messages
        if q in strip_html(m["content"]).lower()
    ]


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    # Render the full UI first so the page appears instantly.
    # The vectorstore / QA-chain are loaded lazily below, only when the
    # user actually submits a query (st.cache_resource keeps them cached
    # after the first load, so subsequent queries are instant).
    render_header()
    render_info_cards()

    # ── Health Topics ──────────────────────────
    topic_query = render_health_topics()

    # ── Tabs ───────────────────────────────────
    tab_chat, tab_symptoms, tab_reminders, tab_score, tab_bookmarks = st.tabs([
        "💬  Chat",
        "🩺  Symptom Checker",
        "💊  Medications",
        "📊  Health Score",
        "🔖  Saved Answers",
    ])

    with tab_chat:
        st.markdown("<div class='md-section-title'>Conversation</div>", unsafe_allow_html=True)

        # Disclaimer
        st.markdown(
            '<div class="md-disclaimer">'
            '<span class="md-disclaimer-icon">⚠️</span>'
            '<span>Medibot provides educational information only and is <strong>not a substitute for professional medical advice</strong>. Always consult a qualified healthcare provider for personal health decisions.</span>'
            '</div>',
            unsafe_allow_html=True,
        )

        # Chat tools bar (export, search, clear)
        render_chat_tools_bar()

        # Chat history banner
        msg_count = len(st.session_state.messages)
        if msg_count > 0:
            st.markdown(
                f'<div class="md-history-banner">'
                f'<div class="md-history-count"><span class="md-history-dot"></span>'
                f'{msg_count} message{"s" if msg_count != 1 else ""} in this session</div>'
                f'<span style="font-size:12px; color:var(--md-soft)">Last query: {st.session_state.last_query_time or "—"}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Empty state
        if not st.session_state.messages:
            st.markdown(
                '<div class="md-empty">'
                '<span class="md-empty-icon">💬</span>'
                'Start by asking a health-related question below, or pick a topic above. Medibot will search the medical knowledge base and answer using the most relevant context it finds.'
                '</div>',
                unsafe_allow_html=True,
            )

        # Suggestion chips — always visible so user can click them any time
        st.markdown("<div class='md-section-sub' style='margin-bottom:8px'>Quick questions to get started:</div>", unsafe_allow_html=True)
        chip_query = render_suggestion_chips()

        # Handle symptom-checker → chat handoff
        sym_chat_query = st.session_state.pop("_sym_chat_query", None)

        # Replay messages (filtered by search if active)
        for message in messages_to_display():
            avatar = "👤" if message["role"] == "user" else "🤖"
            with st.chat_message(message["role"], avatar=avatar):
                if message["role"] == "assistant":
                    st.markdown(message["content"], unsafe_allow_html=True)
                else:
                    st.markdown(message["content"])

        # Show search-no-results hint
        if st.session_state.get("chat_search_input", "").strip() and not messages_to_display():
            st.info(f'No messages match "{st.session_state.get("chat_search_input", "")}"')

        # Voice input widget — lets user speak to fill the chat input
        render_voice_input()

        # Chat input — process BEFORE rendering save button so pending_bookmark
        # is populated on the same run as the response
        typed_query = st.chat_input("Ask Medibot a medical question…")
        voice_query = st.session_state.pop("voice_query", None)
        user_query  = typed_query or voice_query or chip_query or topic_query or sym_chat_query

        if user_query:
            st.session_state.pending_bookmark  = None
            st.session_state.bookmark_saved_id = None
            # Lazy-load: cached by st.cache_resource after first call, instant thereafter
            try:
                vectorstore = load_vectorstore()
                qa_chain    = build_qa_chain(vectorstore)
            except Exception as error:
                st.error(f"Vector store failed to load: {str(error)}")
                st.stop()
            handle_user_query(user_query, qa_chain)

        # ── Save Answer button (rendered OUTSIDE chat_message context) ──
        pb = st.session_state.get("pending_bookmark")
        if pb is not None:
            already_saved = st.session_state.get("bookmark_saved_id") == pb["id"]
            col_save, col_spacer = st.columns([1, 4])
            with col_save:
                if already_saved:
                    st.markdown(
                        '<div style="padding:8px 16px; border-radius:999px; background:rgba(34,197,94,.12); '
                        'border:1px solid rgba(34,197,94,.30); color:#86efac; font-size:13px; font-weight:800; '
                        'display:inline-flex; align-items:center; gap:6px;">✔ Saved</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    if st.button("🔖 Save Answer", key=f"save_btn_{pb['id']}"):
                        already = any(
                            b["question"] == pb["question"] and b["time"] == pb["time"]
                            for b in st.session_state.bookmarked_answers
                        )
                        if not already:
                            st.session_state.bookmarked_answers.append({
                                "question": pb["question"],
                                "answer":   pb["answer"],
                                "time":     pb["time"],
                            })
                        st.session_state.bookmark_saved_id = pb["id"]
                        st.rerun()

    with tab_symptoms:
        render_symptom_checker()

    with tab_reminders:
        render_medication_reminders()

    with tab_score:
        render_health_score()

    with tab_bookmarks:
        st.markdown("<div class='md-section-title'>Saved Answers</div>", unsafe_allow_html=True)
        st.markdown("<div class='md-section-sub'>Answers you've bookmarked during this session</div>", unsafe_allow_html=True)
        render_bookmarks_tab()

    render_footer()
    render_floating_widget()




# ───────────────────────── Floating Quick Widget ───────────────────────────

def render_floating_widget():
    components.html(
        """
<script>
(function() {
    const parentDoc = window.parent.document;
    const parentWin = window.parent;
    const widgetId = "mm-widget-root-medibot";
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
    `;
    parentDoc.head.appendChild(style);

    const root = parentDoc.createElement("div");
    root.id = widgetId;
    const tools = [
        ['Chat', 'CH', 'Chat', 'Ask Medibot', 'tab', 'Conversation'],
        ['Symptom Checker', 'SC', 'Symptom Checker', 'Analyze symptoms', 'tab', 'Symptom Checker'],
        ['Medications', 'MD', 'Medications', 'Reminders', 'tab', 'Medication Reminders'],
        ['Health Score', 'HS', 'Health Score', 'Wellness quiz', 'tab', 'Lifestyle Health Score'],
        ['Saved Answers', 'SV', 'Saved Answers', 'Bookmarks', 'tab', 'Saved Answers'],
        ['input', 'FS', 'Focus Input', 'Jump to chat/form', 'input', ''],
        ['top', 'UP', 'Back to top', 'Scroll upward', 'top', '']
    ];
    root.innerHTML = `
        <div class="mm-widget-panel" role="menu" aria-label="Medibot quick actions">
            <div class="mm-widget-head">
                <div class="mm-widget-head-top">
                    <div style="display:flex;align-items:center;gap:9px;">
                        <div class="mm-widget-head-icon">🤖</div>
                        <div style="display:flex;flex-direction:column;gap:1px;">
                            <div class="mm-widget-title">Medibot tools</div>
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
    function savePendingScroll(target) {
        if (!target) return;
        try { parentWin.localStorage.setItem(widgetId + "-pending-scroll", target); } catch (_) {}
    }
    function clearPendingScroll() {
        try { parentWin.localStorage.removeItem(widgetId + "-pending-scroll"); } catch (_) {}
    }
    function scrollWithRetries(target) {
        if (!target) return;
        let tries = 0;
        const timer = parentWin.setInterval(() => {
            tries += 1;
            if (scrollToText(target) || tries >= 12) {
                if (tries < 12) clearPendingScroll();
                parentWin.clearInterval(timer);
            }
        }, 180);
    }
    function applyPendingScroll() {
        let target = "";
        try { target = parentWin.localStorage.getItem(widgetId + "-pending-scroll") || ""; } catch (_) {}
        if (target) scrollWithRetries(target);
    }
    function runAction(kind, target, scrollTarget) {
        if (kind === "top") {
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
            clearPendingScroll();
            return true;
        }
        if (kind === "input") return focusInput();
        if (kind === "tab") {
            savePendingScroll(scrollTarget || target);
            const ok = clickByText('[role="tab"], [data-baseweb="tab"], button', target);
            if (ok) parentWin.setTimeout(() => scrollWithRetries(scrollTarget || target), 280);
            return ok;
        }
        if (kind === "nav") {
            savePendingScroll(scrollTarget || target);
            const ok = clickByText('[data-testid="stButton"] button, button', target) || clickByText('[data-testid="stRadio"] label, [role="radio"]', target);
            if (ok) parentWin.setTimeout(() => scrollWithRetries(scrollTarget || target), 500);
            return ok;
        }
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
})();
</script>
        """,
        height=0,
        width=0,
    )


@st.cache_data(show_spinner=False)
def _build_footer_html(logo_uri, year):
    # Footer: same brain+sparkle icon as sidebar, plain (no animation), simple green dot
    footer_logo_html = (
        "<div class='md-footer-logo' style='"
        "display:inline-flex;align-items:center;justify-content:center;"
        "position:relative;width:52px;height:52px;border-radius:16px;"
        "background:linear-gradient(135deg,rgba(124,92,191,.18),rgba(0,133,122,.12));"
        "border:1.5px solid rgba(124,92,191,.28);box-shadow:0 4px 16px rgba(10,15,30,.12);"
        "box-sizing:border-box;'>"
        "<span style='font-size:24px;line-height:1;'>🧠</span>"
        "<span style='font-size:12px;position:absolute;bottom:6px;right:6px;line-height:1;'>✦</span>"
        "</div>"
    )
    style_tag = ""
    return (
        style_tag
        + f'<div class="md-footer">'
        f'<div class="md-footer-top">'
        f'<div class="md-footer-brand">{footer_logo_html}'
        f'<div><div class="md-footer-brand-name">Medibot<br/>AI Health Assistant</div>'
        f'<div class="md-footer-brand-sub">by Yatin Sharma</div></div></div>'
        f'</div>'
        f'<div class="md-footer-links">'
        f'<a class="md-footer-link" href="https://github.com/YatinSharma1303/" target="_blank">🐙 GitHub</a>'
        f'<a class="md-footer-link" href="https://www.linkedin.com/in/yatin-sharma-793042372/" target="_blank">💼 LinkedIn</a>'
        f'<a class="md-footer-link" href="https://groq.com" target="_blank">⚡ Groq</a>'
        f'</div>'
        f'<div class="md-footer-meta">'
        f'<span class="md-footer-version">Medibot v1.0</span>'
        f' &nbsp;·&nbsp; Made with <span class="md-footer-heart">❤️</span> by <strong>Yatin Sharma</strong>'
        f' &nbsp;·&nbsp; © {year}'
        f'</div>'
        f'<div class="md-footer-disclaimer">🤖 <strong>Medical Disclaimer:</strong> '
        f'Medibot is for educational purposes only and does not constitute medical advice, diagnosis, or treatment. '
        f'Always consult a qualified healthcare professional regarding any health concerns.</div>'
        f'</div>'
    )


def render_footer():
    st.markdown(
        _build_footer_html(image_to_data_uri(LOGO_PATH), datetime.now().year),
        unsafe_allow_html=True,
    )

main()