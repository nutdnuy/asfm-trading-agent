"""ASFM Trading Agent — Streamlit app.

Reproduces the LLM-based investor agent framework from
Gao et al. (2024), arXiv:2406.19966 — "Simulating Financial Market via
Large Language Model based Agents."

Theme: QuantSeras (Material Dark + desaturated green).
"""

from __future__ import annotations

import os

import plotly.graph_objects as go
import streamlit as st
from langchain_openai import ChatOpenAI

from asfm_pipeline import (
    AGENT_PROFILES,
    TradingDecision,
    apply_decision,
    run_agent,
)


# =============================================================================
# Helpers
# =============================================================================

def load_secret_key() -> str:
    """1) Streamlit secrets  2) env var  3) sidebar input."""
    try:
        val = st.secrets.get("OPENAI_API_KEY", "")
        if val:
            return val
    except (FileNotFoundError, Exception):
        pass
    return os.getenv("OPENAI_API_KEY", "")


# =============================================================================
# Page config
# =============================================================================

st.set_page_config(
    page_title="ASFM Trading Agent",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# QuantSeras CSS
# =============================================================================

CUSTOM_CSS = """
<style>
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }

:root {
    --qs-bg: #121212;
    --qs-surface-1: #1D1D1D;
    --qs-surface-2: #212121;
    --qs-surface-3: #242424;
    --qs-primary: #69F0AE;
    --qs-primary-variant: #00C853;
    --qs-secondary: #03DAC6;
    --qs-profit: #00E676;
    --qs-loss: #FF5252;
    --qs-warning: #FFB74D;
    --qs-error: #CF6679;
    --qs-text-high: rgba(255,255,255,0.87);
    --qs-text-med: rgba(255,255,255,0.60);
    --qs-text-dis: rgba(255,255,255,0.38);
    --qs-border: rgba(255,255,255,0.08);
}

html, body, [class*="css"] {
    font-family: 'Inter', 'IBM Plex Sans', -apple-system, sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 0% 0%, rgba(105, 240, 174, 0.06) 0%, transparent 45%),
        radial-gradient(circle at 100% 100%, rgba(3, 218, 198, 0.04) 0%, transparent 45%),
        var(--qs-bg);
    color: var(--qs-text-high);
}

.qs-hero {
    text-align: center;
    padding: 0.75rem 0 1rem 0;
    border-bottom: 1px solid var(--qs-border);
    margin-bottom: 1.25rem;
}
.qs-hero h1 {
    font-size: 2.1rem;
    font-weight: 700;
    background: linear-gradient(135deg, #69F0AE 0%, #03DAC6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    letter-spacing: -0.02em;
}
.qs-hero .tagline {
    color: var(--qs-text-med);
    font-size: 0.78rem;
    margin-top: 0.35rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-family: 'JetBrains Mono', monospace;
}
.qs-hero .paper-ref {
    color: var(--qs-text-dis);
    font-size: 0.7rem;
    margin-top: 0.25rem;
    font-family: 'JetBrains Mono', monospace;
}

section[data-testid="stSidebar"] {
    background: var(--qs-surface-1);
    border-right: 1px solid var(--qs-border);
}
.qs-sidebar-brand {
    text-align: center;
    padding: 0.5rem 0 0.9rem 0;
    border-bottom: 1px solid var(--qs-border);
    margin-bottom: 0.75rem;
}
.qs-sidebar-brand h2 {
    font-size: 1.5rem;
    font-weight: 700;
    margin: 0;
    background: linear-gradient(135deg, #69F0AE 0%, #03DAC6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.qs-sidebar-brand p {
    color: var(--qs-text-med);
    font-size: 0.7rem;
    margin-top: 0.2rem;
}
.qs-section {
    font-size: 0.66rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: var(--qs-text-med);
    margin: 0.9rem 0 0.35rem 0;
    font-weight: 700;
}

.qs-label {
    font-size: 0.72rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--qs-text-med);
    font-weight: 600;
    margin-bottom: 0.3rem;
    font-family: 'JetBrains Mono', monospace;
}

/* ===== Decision cards ===== */
.qs-decision {
    background: var(--qs-surface-1);
    border: 1px solid var(--qs-border);
    border-radius: 10px;
    padding: 1.1rem 1.3rem;
    margin-bottom: 0.75rem;
}
.qs-decision.buy   { border-left: 4px solid var(--qs-profit); }
.qs-decision.sell  { border-left: 4px solid var(--qs-loss); }
.qs-decision.hold  { border-left: 4px solid var(--qs-text-dis); }

.qs-decision .agent {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: var(--qs-text-med);
    letter-spacing: 0.05em;
    margin-bottom: 0.35rem;
}
.qs-decision .action {
    font-size: 2.1rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: -0.02em;
    line-height: 1;
    margin-bottom: 0.5rem;
}
.qs-decision .action.buy  { color: var(--qs-profit); }
.qs-decision .action.sell { color: var(--qs-loss); }
.qs-decision .action.hold { color: var(--qs-text-med); }

.qs-decision .trade-row {
    display: flex;
    gap: 0.4rem;
    flex-wrap: wrap;
    margin-bottom: 0.75rem;
}
.qs-decision .chip {
    background: var(--qs-surface-2);
    border: 1px solid var(--qs-border);
    border-radius: 5px;
    padding: 0.3rem 0.6rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: var(--qs-text-med);
}
.qs-decision .chip strong {
    color: var(--qs-text-high);
    margin-left: 0.25rem;
    font-weight: 700;
}
.qs-decision .chip.conf-high strong   { color: var(--qs-profit); }
.qs-decision .chip.conf-medium strong { color: var(--qs-warning); }
.qs-decision .chip.conf-low strong    { color: var(--qs-text-dis); }

.qs-decision .reasoning {
    color: var(--qs-text-high);
    font-size: 0.9rem;
    line-height: 1.55;
    padding-top: 0.7rem;
    border-top: 1px solid var(--qs-border);
}

/* Feature cards (landing) */
.qs-feature {
    background: var(--qs-surface-1);
    border: 1px solid var(--qs-border);
    border-radius: 8px;
    padding: 1.1rem;
    height: 100%;
}
.qs-feature .icon { font-size: 1.7rem; margin-bottom: 0.4rem; }
.qs-feature h3 {
    color: var(--qs-primary);
    font-size: 0.9rem;
    margin: 0 0 0.4rem 0;
    font-family: 'JetBrains Mono', monospace;
}
.qs-feature p {
    color: var(--qs-text-med);
    font-size: 0.82rem;
    margin: 0;
    line-height: 1.5;
}

/* Buttons */
.stButton > button {
    background: var(--qs-surface-2);
    border: 1px solid var(--qs-border);
    color: var(--qs-text-high);
    border-radius: 6px;
    font-weight: 500;
}
.stButton > button:hover {
    border-color: var(--qs-primary);
    color: var(--qs-primary);
    background: rgba(105, 240, 174, 0.08);
}
.stButton > button[kind="primary"] {
    background: var(--qs-primary);
    color: #000;
    border: none;
    font-weight: 700;
}
.stButton > button[kind="primary"]:hover {
    background: var(--qs-primary-variant);
    color: #000;
}

/* Inputs */
.stTextInput input, .stTextArea textarea, .stNumberInput input,
div[data-baseweb="select"] > div {
    background: var(--qs-surface-1) !important;
    border-color: var(--qs-border) !important;
    color: var(--qs-text-high) !important;
}
.stTextArea textarea { font-size: 0.88rem !important; line-height: 1.5 !important; }
.stTextInput input:focus, .stTextArea textarea:focus, .stNumberInput input:focus,
div[data-baseweb="select"] > div:focus-within {
    border-color: var(--qs-primary) !important;
}

.stSlider [data-baseweb="slider"] div[role="slider"] {
    background: var(--qs-primary) !important;
    border-color: var(--qs-primary) !important;
}

/* Radio (mode toggle) */
div[role="radiogroup"] label {
    background: var(--qs-surface-2);
    border: 1px solid var(--qs-border);
    border-radius: 6px;
    padding: 0.5rem 0.85rem;
    margin-right: 0.4rem;
}

[data-testid="stAlert"] {
    background: var(--qs-surface-1) !important;
    border: 1px solid var(--qs-border) !important;
    border-left: 3px solid var(--qs-secondary) !important;
    border-radius: 8px !important;
}
[data-testid="stCaptionContainer"] { color: var(--qs-text-med) !important; }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =============================================================================
# Example scenarios (matches paper §3.5 — controllable policy shocks)
# =============================================================================

EXAMPLES: dict[str, dict] = {
    "🏦 Fed rate cut — real estate": {
        "company_name": "SunRealty REIT",
        "sector": "Real Estate",
        "business": (
            "Diversified commercial real estate investment trust with holdings "
            "across office, retail, and logistics properties in major US cities. "
            "Revenue ~90% recurring lease income. Net debt-to-equity 0.8. "
            "Sensitive to interest rate changes due to high leverage and "
            "financing costs."
        ),
        "price_history": [
            82.10, 81.75, 81.90, 82.30, 82.15, 81.60, 81.85, 82.00,
            82.50, 82.20, 82.45, 82.80, 82.95, 83.10, 83.40,
        ],
        "news": (
            "Federal Reserve cuts target federal funds rate by 50 basis "
            "points at today's FOMC meeting — the first cut in 14 months. "
            "Chair signals a data-dependent but likely easing path ahead. "
            "10-year Treasury yield drops 18 bps. Real estate and "
            "rate-sensitive sectors rally in after-hours trading."
        ),
        "cash": 50000.0,
        "shares_held": 200,
    },
    "📈 Inflation shock — financials": {
        "company_name": "FinanceBank Corp",
        "sector": "Financials",
        "business": (
            "Large national bank with ~$420B in deposits. Revenue mix: 55% net "
            "interest income, 30% fee income, 15% trading/IB. Loan book skewed "
            "toward commercial and consumer lending. Net interest margin "
            "compressed over the past year; management guided NIM ~2.9% for FY."
        ),
        "price_history": [
            54.20, 53.80, 54.00, 53.60, 53.10, 52.80, 52.40, 52.10,
            51.70, 51.30, 50.90, 50.60, 50.20, 49.80, 49.50,
        ],
        "news": (
            "US CPI print comes in at 8.1% year-over-year, well above the "
            "5.9% consensus. Core CPI at 6.4%, highest since 1982. Fed futures "
            "now price 85% odds of a 75bp hike next meeting. Yield curve "
            "inverts further; regional banks underperform on deposit flight "
            "concerns. Short-term funding costs jump."
        ),
        "cash": 30000.0,
        "shares_held": 500,
    },
    "🏭 Normal market — info tech": {
        "company_name": "TechCore Systems",
        "sector": "Information Technology",
        "business": (
            "Mid-cap enterprise software company specializing in workflow "
            "automation and integration tools. Revenue 78% subscription. "
            "Gross margin 72%, operating margin 14%. Modest growth (+12% YoY). "
            "Balance sheet net-cash positive."
        ),
        "price_history": [
            145.20, 144.80, 145.50, 145.10, 144.90, 145.30, 145.70, 145.20,
            145.00, 145.40, 145.10, 144.80, 145.20, 145.50, 145.30,
        ],
        "news": (
            "Quiet session on Wall Street. Major indices closed within ±0.3%. "
            "Economic data release schedule light this week; next FOMC meeting "
            "is three weeks away. No sector-specific catalysts. Volumes "
            "below 20-day average."
        ),
        "cash": 20000.0,
        "shares_held": 80,
    },
}


# Initialise default example on first load
if "company_name" not in st.session_state:
    first = next(iter(EXAMPLES.values()))
    st.session_state["company_name"] = first["company_name"]
    st.session_state["sector"] = first["sector"]
    st.session_state["business"] = first["business"]
    st.session_state["price_history_str"] = ", ".join(
        f"{p:.2f}" for p in first["price_history"]
    )
    st.session_state["news"] = first["news"]
    st.session_state["cash"] = first["cash"]
    st.session_state["shares_held"] = first["shares_held"]


# =============================================================================
# Sidebar
# =============================================================================

with st.sidebar:
    st.markdown(
        '<div class="qs-sidebar-brand">'
        '<h2>🎩 Legends Mode</h2>'
        '<p>10 Investing Legends · ASFM × Gao 2024</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="qs-section">🔑 Authentication</div>', unsafe_allow_html=True)
    secret_key = load_secret_key()
    if secret_key:
        api_key = secret_key
        st.success("✅ Loaded from secrets", icon="🔒")
    else:
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="sk-...",
            label_visibility="collapsed",
        )

    # LLM is fixed — gpt-5.4-nano only (no selector)
    llm_model = "gpt-5.4-nano"

    st.markdown('<div class="qs-section">🎭 Run mode</div>', unsafe_allow_html=True)
    run_mode = st.radio(
        "Mode",
        options=["Compare selected legends", "Single legend"],
        index=0,
        label_visibility="collapsed",
    )

    if run_mode == "Single legend":
        agent_type = st.selectbox(
            "Legend",
            options=list(AGENT_PROFILES.keys()),
            format_func=lambda k: f"{AGENT_PROFILES[k]['emoji']} {AGENT_PROFILES[k]['name']}",
            index=0,
            label_visibility="collapsed",
        )
        selected_agents: list[str] = [agent_type]
    else:
        agent_type = None
        # Balanced default: 5 distinct philosophies (Wilmott + value + quant + macro + growth)
        default_picks = ["wilmott", "buffett", "simons", "dalio", "wood"]
        selected_agents = st.multiselect(
            "Legends to compare",
            options=list(AGENT_PROFILES.keys()),
            default=default_picks,
            format_func=lambda k: f"{AGENT_PROFILES[k]['emoji']} {AGENT_PROFILES[k]['name']}",
            label_visibility="collapsed",
        )
        st.caption(
            f"🕑 {len(selected_agents)} legend(s) × ~3s each = "
            f"~{max(1, len(selected_agents)*3)}s total"
        )

    st.markdown('<div class="qs-section">📋 Scenarios</div>', unsafe_allow_html=True)
    example_name = st.selectbox(
        "Scenario",
        options=list(EXAMPLES.keys()),
        index=0,
        label_visibility="collapsed",
    )
    if st.button("📥 Load scenario", use_container_width=True, type="primary"):
        ex = EXAMPLES[example_name]
        st.session_state["company_name"] = ex["company_name"]
        st.session_state["sector"] = ex["sector"]
        st.session_state["business"] = ex["business"]
        st.session_state["price_history_str"] = ", ".join(
            f"{p:.2f}" for p in ex["price_history"]
        )
        st.session_state["news"] = ex["news"]
        st.session_state["cash"] = ex["cash"]
        st.session_state["shares_held"] = ex["shares_held"]
        st.session_state.pop("results", None)
        st.rerun()

    if st.button("🧹 Clear results", use_container_width=True):
        st.session_state.pop("results", None)
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("📄 arXiv:2406.19966 · Jun 2024")
    st.caption("C = A(I_a, I_o, I_t)")


# =============================================================================
# Hero
# =============================================================================

st.markdown(
    '<div class="qs-hero">'
    '<h1>🎩 Legendary Investor Agent</h1>'
    '<div class="tagline">10 legends · same market · different decisions</div>'
    '<div class="paper-ref">ASFM framework · Gao et al. · arXiv:2406.19966 · 2024</div>'
    '</div>',
    unsafe_allow_html=True,
)


# =============================================================================
# API key gate
# =============================================================================

if not api_key:
    st.info("👈 ใส่ OpenAI API Key ที่ sidebar เพื่อเริ่ม")
    # Grid of 10 legendary investors, 5 per row
    keys = list(AGENT_PROFILES.keys())
    for start in (0, 5):
        cols = st.columns(5)
        for col, key in zip(cols, keys[start:start + 5]):
            prof = AGENT_PROFILES[key]
            with col:
                st.markdown(
                    f'<div class="qs-feature">'
                    f'<div class="icon">{prof["emoji"]}</div>'
                    f'<h3>{prof["name"].upper()}</h3>'
                    f'<p>{prof["tagline"]}</p>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
    st.stop()


# =============================================================================
# Input area
# =============================================================================

left, right = st.columns([3, 2])

with left:
    st.markdown('<div class="qs-label">📊 TARGET STOCK</div>', unsafe_allow_html=True)
    c_name, c_sector = st.columns([2, 1])
    with c_name:
        st.text_input("Company", label_visibility="collapsed",
                      placeholder="Company name", key="company_name")
    with c_sector:
        st.text_input("Sector", label_visibility="collapsed",
                      placeholder="Sector", key="sector")
    st.text_area("Business", label_visibility="collapsed", height=100,
                 placeholder="Short business description (1-3 sentences)", key="business")

    st.markdown('<div class="qs-label">📉 PRICE HISTORY  (last 15 days, comma-separated)</div>',
                unsafe_allow_html=True)
    st.text_area("Prices", label_visibility="collapsed", height=72,
                 placeholder="e.g. 82.10, 81.75, ..., 83.40", key="price_history_str")

    st.markdown('<div class="qs-label">📰 ECONOMIC NEWS</div>', unsafe_allow_html=True)
    st.text_area("News", label_visibility="collapsed", height=110,
                 placeholder="Paste the latest relevant economic / policy news",
                 key="news")

with right:
    st.markdown('<div class="qs-label">💼 YOUR WALLET</div>', unsafe_allow_html=True)
    st.number_input("Cash ($)", min_value=0.0, step=1000.0,
                    format="%.2f", key="cash")
    st.number_input("Shares held", min_value=0, step=1, key="shares_held")

    # mini price chart
    try:
        prices = [float(x.strip()) for x in st.session_state["price_history_str"].split(",") if x.strip()]
    except Exception:
        prices = []

    if len(prices) >= 2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=prices, mode="lines+markers",
            line=dict(color="#69F0AE", width=2),
            marker=dict(size=5, color="#00C853"),
            hovertemplate="day %{x}<br>price %{y:.2f}<extra></extra>",
        ))
        fig.update_layout(
            height=220,
            paper_bgcolor="#1D1D1D",
            plot_bgcolor="#1D1D1D",
            font=dict(family="Inter", color="rgba(255,255,255,0.6)", size=10),
            margin=dict(l=10, r=10, t=20, b=20),
            showlegend=False,
            xaxis=dict(showgrid=False, title=""),
            yaxis=dict(gridcolor="rgba(255,255,255,0.08)", title=""),
        )
        st.plotly_chart(fig, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)
col_btn, _ = st.columns([1, 4])
with col_btn:
    disabled = not prices or len(prices) < 2
    analyze = st.button(
        "🔬 Run agent(s)",
        type="primary",
        use_container_width=True,
        disabled=disabled,
    )


# =============================================================================
# Run
# =============================================================================

def render_decision(agent_key: str, decision: TradingDecision) -> None:
    """Render a single decision card."""
    prof = AGENT_PROFILES[agent_key]
    action_class = decision.action.lower()

    qty_line = ""
    if decision.action != "HOLD":
        qty_line = (
            f'<div class="chip">qty <strong>{decision.quantity:,}</strong></div>'
            f'<div class="chip">limit <strong>${decision.limit_price:,.2f}</strong></div>'
            f'<div class="chip">notional <strong>${decision.quantity * decision.limit_price:,.0f}</strong></div>'
        )

    st.markdown(
        f'<div class="qs-decision {action_class}">'
        f'<div class="agent">{prof["emoji"]} {prof["name"].upper()} · {prof["tagline"]}</div>'
        f'<div class="action {action_class}">{decision.action}</div>'
        f'<div class="trade-row">'
        f'{qty_line}'
        f'<div class="chip conf-{decision.confidence}">confidence <strong>{decision.confidence.upper()}</strong></div>'
        f'</div>'
        f'<div class="reasoning">{decision.reasoning}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


if analyze:
    try:
        llm = ChatOpenAI(model=llm_model, api_key=api_key, temperature=0.2)
        results: dict[str, TradingDecision] = {}

        agents_to_run = [k for k in selected_agents if k in AGENT_PROFILES]
        if not agents_to_run:
            st.error("⚠️ Select at least one legend from the sidebar.")
            st.stop()

        with st.status(f"🔬 Running {len(agents_to_run)} agent(s)...", expanded=True) as status:
            for k in agents_to_run:
                prof = AGENT_PROFILES[k]
                st.write(f"{prof['emoji']} **{prof['name']}** deciding...")
                decision = run_agent(
                    agent_type=k,
                    company_name=st.session_state["company_name"],
                    sector=st.session_state["sector"],
                    business=st.session_state["business"],
                    price_history=prices,
                    economic_news=st.session_state["news"],
                    cash=float(st.session_state["cash"]),
                    shares_held=int(st.session_state["shares_held"]),
                    llm=llm,
                )
                results[k] = decision
                st.write(f"   → **{decision.action}** qty={decision.quantity} @ {decision.limit_price:.2f}")
            status.update(label="✅ Decision(s) ready", state="complete", expanded=False)

        st.session_state["results"] = results
    except Exception as exc:
        st.error(f"Pipeline failed: {exc}")


# =============================================================================
# Render results
# =============================================================================

results: dict | None = st.session_state.get("results")

if results:
    st.markdown("### 🎯 Agent decisions")

    if len(results) > 1:
        cols = st.columns(2)
        for i, (k, d) in enumerate(results.items()):
            with cols[i % 2]:
                render_decision(k, d)
    else:
        for k, d in results.items():
            render_decision(k, d)

    # Portfolio impact (for single) or summary table (for compare-all)
    current_price = prices[-1] if prices else 0.0

    if len(results) == 1:
        k, d = next(iter(results.items()))
        after = apply_decision(
            d,
            cash=float(st.session_state["cash"]),
            shares_held=int(st.session_state["shares_held"]),
            current_price=current_price,
        )
        st.markdown("### 💼 Wallet after execution")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Cash", f"${after['cash']:,.2f}", f"{after['delta_cash']:+,.2f}")
        m2.metric("Shares", f"{after['shares']:,}", f"{after['delta_shares']:+,}")
        m3.metric("Position value", f"${after['position_value']:,.2f}")
        m4.metric("Total equity", f"${after['total_equity']:,.2f}")
    else:
        st.markdown("### 📊 Decisions at a glance")
        rows = []
        for k, d in results.items():
            prof = AGENT_PROFILES[k]
            after = apply_decision(
                d,
                cash=float(st.session_state["cash"]),
                shares_held=int(st.session_state["shares_held"]),
                current_price=current_price,
            )
            rows.append({
                "Agent": f"{prof['emoji']} {prof['name']}",
                "Action": d.action,
                "Qty": d.quantity,
                "Limit": f"${d.limit_price:.2f}" if d.action != "HOLD" else "—",
                "Confidence": d.confidence.upper(),
                "Δ Cash": f"${after['delta_cash']:+,.0f}",
                "Δ Shares": f"{after['delta_shares']:+,}",
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)

    st.caption(
        "📘 Framework: Gao et al. (2024) · 4 investor types × profile + observation → "
        "LLM tool-call on {BUY, SELL, HOLD} × (code, quantity, price)"
    )
