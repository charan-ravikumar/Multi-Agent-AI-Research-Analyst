"""
ui/theme.py — the look of the app: a "field notebook" — warm paper, ink, one vermilion accent, editorial type.

Design intent (so changes stay coherent):
  * paper / ink / one accent.  Colour carries meaning (vermilion = action and attention, olive = supported, amber =
    partial, vermilion again = problem); everything else is warm neutrals.  No gradients, no glow, no emoji chrome.
  * type does the work: Fraunces (display serif) for headings, Newsreader for the report body (long reading),
    Hanken Grotesk for controls, JetBrains Mono for numbers and labels.  Fonts load from Google Fonts and fall back to
    system serif/sans if offline.
  * structure by hairline rules, not boxes: ledger-style metrics, rule-separated sections, square corners.

Only static strings are injected as HTML here; nothing user- or web-derived ever passes through ``unsafe_allow_html``.
"""
from __future__ import annotations

import streamlit as st

PAPER, PAPER_2, INK, MUTED, RULE = "#F4EFE6", "#FBF8F2", "#1B1A17", "#6B665B", "#D9D1C1"
ACCENT, OK, WARN = "#C8381A", "#3F6B4F", "#B7791F"

CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,600;0,9..144,700;1,9..144,400&family=Hanken+Grotesk:wght@400;500;600&family=JetBrains+Mono:wght@400;500&family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;1,6..72,400&display=swap');

:root {{
  --paper: {PAPER}; --paper-2: {PAPER_2}; --ink: {INK}; --muted: {MUTED}; --rule: {RULE};
  --accent: {ACCENT}; --ok: {OK}; --warn: {WARN};
  --display: 'Fraunces', Georgia, 'Times New Roman', serif;
  --read: 'Newsreader', Georgia, serif;
  --ui: 'Hanken Grotesk', system-ui, -apple-system, 'Segoe UI', sans-serif;
  --mono: 'JetBrains Mono', ui-monospace, 'SF Mono', Menlo, monospace;
}}

html, body, .stApp, [data-testid="stAppViewContainer"] {{ background: var(--paper); color: var(--ink); font-family: var(--ui); }}
[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ max-width: 1120px; padding-top: 2.2rem; padding-bottom: 5rem; }}
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"], .stDeployButton {{ display: none !important; visibility: hidden; }}

/* ── masthead ─────────────────────────────────────────────────────────────── */
.masthead {{ border-top: 3px solid var(--ink); border-bottom: 1px solid var(--ink); padding: .55rem 0 1.5rem; margin-bottom: 1.4rem; position: relative; }}
.masthead::after {{ content: ""; position: absolute; left: 0; right: 0; bottom: -5px; border-bottom: 1px solid var(--ink); }}
.mast-meta {{ display: flex; justify-content: space-between; gap: 1rem; font: 500 .68rem/1 var(--mono); letter-spacing: .14em; text-transform: uppercase; color: var(--muted); padding-bottom: 1.1rem; }}
.mast-meta b {{ color: var(--accent); font-weight: 500; }}
.masthead h1 {{ font: 600 clamp(2.4rem, 5.4vw, 4rem)/.98 var(--display); letter-spacing: -.025em; margin: 0; padding: 0; font-variation-settings: 'opsz' 144; }}
.masthead h1 em {{ font-style: italic; font-weight: 400; color: var(--accent); }}
.masthead .dek {{ font: 400 1.18rem/1.45 var(--read); color: #3a372f; max-width: 46rem; margin: .9rem 0 0; }}

/* ── headings & text ──────────────────────────────────────────────────────── */
h1, h2, h3, h4 {{ font-family: var(--display) !important; letter-spacing: -.012em; color: var(--ink); font-variation-settings: 'opsz' 72; }}
h3 {{ font-size: 1.55rem !important; font-weight: 600 !important; }}
h2 {{ border-top: 1px solid var(--ink); padding-top: .8rem !important; }}
[data-testid="stSubheader"], .stApp h3 {{ margin-top: .4rem; }}
[data-testid="stCaptionContainer"] p, .stApp small {{ font-family: var(--ui) !important; font-size: .86rem !important; line-height: 1.5; color: var(--muted); }}
.stMarkdown p, .stMarkdown li {{ font-family: var(--read); font-size: 1.08rem; line-height: 1.62; }}
.stMarkdown strong {{ font-weight: 600; }}
.stMarkdown a {{ color: var(--accent); text-decoration-thickness: 1px; text-underline-offset: 3px; }}
.stMarkdown h1, .stMarkdown h2 {{ font-weight: 600; }}
.kicker {{ font: 500 .7rem/1 var(--mono); letter-spacing: .14em; text-transform: uppercase; color: var(--accent); margin: 1.2rem 0 .35rem; }}

/* ── tabs ─────────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {{ gap: 2rem; border-bottom: 1px solid var(--rule); }}
[data-testid="stTab"] {{ padding: .55rem 0; background: transparent; height: auto; }}
[data-testid="stTab"], [data-testid="stTab"] * {{ font-family: var(--mono) !important; font-size: .74rem !important; font-weight: 500 !important; letter-spacing: .13em; text-transform: uppercase; color: var(--muted); }}
[data-testid="stTab"][aria-selected="true"], [data-testid="stTab"][aria-selected="true"] * {{ color: var(--ink); }}
.stTabs [data-baseweb="tab-highlight"] {{ background: var(--accent); height: 3px; }}
.stTabs [data-baseweb="tab-border"] {{ display: none; }}

/* ── controls ─────────────────────────────────────────────────────────────── */
.stButton > button, .stDownloadButton > button, [data-testid^="stBaseButton-"] {{ border-radius: 2px; border: 1px solid var(--ink); background: transparent; color: var(--ink);
  font: 500 .74rem/1 var(--mono); letter-spacing: .08em; text-transform: uppercase; padding: .8rem 1.15rem; transition: transform .12s ease, box-shadow .12s ease, background .12s ease; }}
.stButton > button:hover, .stDownloadButton > button:hover, [data-testid^="stBaseButton-"]:hover {{ border-color: var(--ink); background: var(--paper-2); box-shadow: 3px 3px 0 var(--ink); transform: translate(-2px,-2px); color: var(--ink); }}
.stButton > button[kind="primary"], [data-testid="stBaseButton-primary"] {{ background: var(--ink); color: var(--paper); }}
.stButton > button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover {{ background: var(--accent); border-color: var(--accent); box-shadow: 3px 3px 0 var(--ink); color: #fff; }}
.stButton > button:disabled, [data-testid^="stBaseButton-"]:disabled {{ opacity: .4; box-shadow: none; transform: none; }}
.stButton > button:focus-visible, input:focus-visible, textarea:focus-visible {{ outline: 2px solid var(--accent) !important; outline-offset: 2px; }}
.stTextInput input, .stTextArea textarea, [data-baseweb="select"] > div {{ background: var(--paper-2) !important; border: 1px solid var(--ink) !important; border-radius: 2px !important; font-family: var(--read); font-size: 1.12rem; }}
.stTextInput input {{ padding: .85rem .9rem; }}
[data-testid="stTextInput"] label p, [data-testid="stTextArea"] label p, .stCheckbox label p, .stSelectbox label p {{ font: 500 .7rem/1.2 var(--mono) !important; letter-spacing: .12em; text-transform: uppercase; color: var(--muted); }}
.stCheckbox label p {{ text-transform: none; letter-spacing: .02em; font-size: .78rem !important; }}

/* example questions: sentence case, readable, left-aligned */
[class*="st-key-sample_"] button {{ text-transform: none; letter-spacing: 0; text-align: left; justify-content: flex-start; padding: .75rem .85rem; height: 100%; }}
[class*="st-key-sample_"] button, [class*="st-key-sample_"] button * {{ font: 400 .92rem/1.35 var(--read) !important; }}

/* ── progress, status, alerts ─────────────────────────────────────────────── */
[data-testid="stProgress"] > div > div {{ background: var(--rule) !important; height: 6px; border-radius: 0; }}
[data-testid="stProgress"] > div > div > div {{ background: var(--accent) !important; border-radius: 0; }}
[data-testid="stProgress"] p {{ font: 500 .78rem/1.3 var(--mono); letter-spacing: .04em; }}
[data-testid="stAlert"] {{ border-radius: 2px; border: 1px solid var(--ink); background: var(--paper-2); border-left-width: 5px; }}
[data-testid="stAlert"] p {{ font-family: var(--ui) !important; font-size: .95rem !important; }}

/* ── ledger metrics ───────────────────────────────────────────────────────── */
[data-testid="stMetric"] {{ border-top: 2px solid var(--ink); padding: .65rem 0 .2rem; background: transparent; }}
[data-testid="stMetricLabel"] p {{ font: 500 .66rem/1.2 var(--mono) !important; letter-spacing: .13em; text-transform: uppercase; color: var(--muted); }}
[data-testid="stMetricValue"] {{ font: 600 2.1rem/1.1 var(--display); letter-spacing: -.02em; font-variation-settings: 'opsz' 96; }}
[data-testid="stMetricDelta"] {{ font-family: var(--mono); font-size: .72rem; background: transparent !important; padding: 0 !important; color: var(--muted) !important; }}
[data-testid="stMetricDelta"] svg {{ display: none; }}

/* ── expanders, containers, tables ────────────────────────────────────────── */
[data-testid="stExpander"] {{ border: 1px solid var(--rule); border-radius: 2px; background: var(--paper-2); }}
[data-testid="stExpander"] summary p {{ font: 500 .72rem/1.2 var(--mono); letter-spacing: .1em; text-transform: uppercase; }}
[data-testid="stVerticalBlockBorderWrapper"] {{ border-radius: 2px; }}
[data-testid="stDataFrame"] {{ border: 1px solid var(--rule); border-radius: 2px; }}
hr {{ border: 0; border-top: 1px solid var(--rule); margin: 1.6rem 0; }}

/* ── report ───────────────────────────────────────────────────────────────── */
.st-key-report, .st-key-sample_report {{ background: var(--paper-2); border: 1px solid var(--ink); padding: 2.2rem 2.6rem; box-shadow: 6px 6px 0 var(--ink); margin: .6rem 6px 1.6rem 0; }}
.st-key-report h2, .st-key-sample_report h2 {{ border-top: 1px solid var(--rule); font-size: 1.45rem; margin-top: 1.6rem; }}
.st-key-report .stMarkdown p, .st-key-sample_report .stMarkdown p {{ font-size: 1.14rem; line-height: 1.7; max-width: 44rem; }}
.st-key-report h2:first-of-type, .st-key-sample_report h2:first-of-type {{ border-top: 0; margin-top: 0; }}

/* ── live draft (streaming) ───────────────────────────────────────────────── */
.live-label {{ display: flex; align-items: center; gap: .6rem; font: 500 .7rem/1 var(--mono); letter-spacing: .14em; text-transform: uppercase; color: var(--accent); margin: 1.4rem 0 .6rem; }}
.live-label::before {{ content: ""; width: 9px; height: 9px; border-radius: 50%; background: var(--accent); animation: pulse 1.1s ease-in-out infinite; }}
@keyframes pulse {{ 0%,100% {{ opacity: 1; transform: scale(1); }} 50% {{ opacity: .25; transform: scale(.7); }} }}
.st-key-livedraft {{ background: var(--paper-2); border: 1px solid var(--ink); border-left: 5px solid var(--accent); padding: 1.4rem 1.8rem; max-height: 28rem; overflow-y: auto; }}
.st-key-livedraft .stMarkdown p {{ font-size: 1.05rem; }}
.live-note {{ font: 400 .74rem/1.4 var(--ui); color: var(--muted); margin-top: .4rem; }}
@media (prefers-reduced-motion: reduce) {{ .live-label::before {{ animation: none; }} .stButton > button {{ transition: none; }} }}

/* ── monitor ──────────────────────────────────────────────────────────────── */
.quota {{ margin: .9rem 0 1.1rem; }}
.quota-head {{ display: flex; justify-content: space-between; font: 500 .74rem/1 var(--mono); letter-spacing: .04em; margin-bottom: .4rem; }}
.quota-head span:last-child {{ color: var(--muted); }}
.quota-bar {{ height: 10px; background: var(--rule); position: relative; }}
.quota-bar i {{ position: absolute; inset: 0 auto 0 0; background: var(--ink); }}
.quota-bar i.warn {{ background: var(--warn); }} .quota-bar i.hot {{ background: var(--accent); }}
.pill {{ display: inline-block; font: 500 .66rem/1 var(--mono); letter-spacing: .1em; text-transform: uppercase; padding: .28rem .5rem; border: 1px solid var(--ink); border-radius: 2px; margin-right: .35rem; }}
.pill.ok {{ background: var(--ok); color: #fff; border-color: var(--ok); }} .pill.warn {{ background: var(--warn); color: #fff; border-color: var(--warn); }}
.pill.bad {{ background: var(--accent); color: #fff; border-color: var(--accent); }} .pill.off {{ color: var(--muted); border-color: var(--rule); }}

/* ── verify panel ─────────────────────────────────────────────────────────── */
mark {{ background: #f1d9a0; color: var(--ink); padding: 0 .1em; }}
@media (max-width: 720px) {{
  .block-container {{ padding-left: 1rem; padding-right: 1rem; }}
  .mast-meta {{ flex-direction: column; gap: .5rem; }}
  .st-key-report, .st-key-sample_report {{ padding: 1.3rem 1.2rem; box-shadow: 4px 4px 0 var(--ink); }}
}}
"""

MASTHEAD = """
<div class="masthead">
  <div class="mast-meta"><span><b>●</b> Research desk</span><span>Every claim sourced · every source checkable</span></div>
  <h1>AI Research <em>Analyst</em></h1>
  <p class="dek">Ask a question. It plans the research, reads the web and arXiv in parallel, and writes a report where each claim
  carries a citation you can open — and a verdict on whether the cited passage really backs it.</p>
</div>
"""


def apply() -> None:
    """Inject the stylesheet and the masthead (static strings only)."""
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)
    st.markdown(MASTHEAD, unsafe_allow_html=True)


def kicker(text: str) -> None:
    st.markdown(f'<div class="kicker">{text}</div>', unsafe_allow_html=True)
