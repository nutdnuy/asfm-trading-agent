# 📈 ASFM Trading Agent

**Simulate how four different LLM-based investor types react to a real-world market situation.**

Reproduces the investor-agent framework from:
> **"Simulating Financial Market via Large Language Model based Agents"**
> Gao, Wen, Zhu, Wei, Cheng, Zhang, Shang
> *arXiv:2406.19966 · Jun 2024*

Paste a company profile + recent prices + economic news + wallet state. The app runs one LLM agent per investor type (**Value**, **Institutional**, **Contrarian**, **Aggressive**) and returns each one's trading decision (`BUY` / `SELL` / `HOLD`) with quantity, limit price, and reasoning.

---

## 🧭 How it works

```
profile prompt  I_a  (your investor type + wallet + holdings)
observation prompt  I_o  (price history + economic news + business)
tool document  I_t  (BUY / SELL / HOLD schema)
            │
            ▼
      LLM agent  A  (OpenAI, structured output via Pydantic)
            │
            ▼
      C  =  A( I_a , I_o , I_t )
      →  TradingDecision(action, quantity, limit_price, confidence, reasoning)
```

Corresponds to Eq. 1 of the paper. Constraints (no over-spend, no over-sell, one order per day) are enforced post-hoc in `asfm_pipeline.run_agent()`.

---

## 🚀 Try it

**Live demo**: deploy to Streamlit Community Cloud — see below.

**Local**:

```bash
git clone https://github.com/<your-username>/asfm-trading-agent.git
cd asfm-trading-agent

python3.10+ -m venv venv          # langchain v1 needs ≥ 3.10
source venv/bin/activate
pip install -r requirements.txt

streamlit run app.py
```

Open <http://localhost:8501> → paste OpenAI key in sidebar → pick a scenario → **Run agent(s)**.

---

## 🔑 API key

Resolved in this order:

1. `st.secrets["OPENAI_API_KEY"]` — Streamlit Cloud *Settings → Secrets* or local `.streamlit/secrets.toml`
2. `OPENAI_API_KEY` env var — for Docker / self-hosted
3. Sidebar input — fallback, session-only, never logged

**On Streamlit Cloud**: leave secrets empty so visitors bring their own key.

---

## 📋 Built-in scenarios

| Scenario | Expected behavior |
|---|---|
| 🏦 **Fed rate cut — real estate** | Value / Aggressive buy · Contrarian may fade · Institutional cautious |
| 📈 **Inflation shock — financials** | Most agents sell or hold · Contrarian may buy oversold · Aggressive bails |
| 🏭 **Normal market — info tech** | Mostly HOLD · small position-sizing nudges from Aggressive / Value |

Swap in any real transcript / news + 15-day price series to test your own setup.

---

## 🎭 Four agent profiles (paper §2.4.1)

| Emoji | Type | Style |
|:---:|---|---|
| 📊 | Value | Intrinsic value, long-term holdings, avoids overpriced names |
| 🏛️ | Institutional | Steady returns, capital preservation, liquidity-focused |
| 🔄 | Contrarian | Buys fear, sells greed, fades crowded sentiment |
| ⚡ | Aggressive | Momentum, concentrated bets, high risk tolerance |

Same observation → different decisions because profiles are baked into the system prompt.

---

## 📂 Project structure

```
asfm-trading-agent/
├── app.py                 # Streamlit UI + QuantSeras theme
├── asfm_pipeline.py       # 4 profiles · decision schema · observation · run_agent
├── requirements.txt
├── README.md
├── CLAUDE.md              # conventions for Claude Code agents
├── .gitignore
└── .streamlit/
    ├── config.toml
    └── secrets.toml.example
```

---

## ☁️ Deploy to Streamlit Community Cloud

1. Push this repo to GitHub (public).
2. <https://share.streamlit.io> → **New app**.
3. Repository, branch `master`, main file `app.py`.
4. Secrets: leave empty (visitors supply their own key).
5. **Deploy** — ~1 min to boot.

---

## ⚠️ Caveats

- **Not financial advice.** Educational reproduction of a research paper.
- Cost per run: 1 LLM call per agent. Compare-all-4 mode ≈ $0.01-$0.05 per scenario on `gpt-4.1`.
- The paper runs full multi-day simulations with order matching. This app is a single-step decision explorer — the core insight (profile-conditioned LLM reasoning) is preserved but market dynamics are not simulated.
- Paper uses China A-shares context; included scenarios are US-flavored for familiarity.

---

## 📝 License

Educational / research use. Paper is an open-access arXiv preprint — rights belong to the original authors.
