# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Streamlit tool that reproduces the investor-agent framework from *"Simulating Financial Market via Large Language Model based Agents"* (Gao et al., arXiv:2406.19966, Jun 2024). User pastes company + price history + news + wallet → app runs one or four LLM-based investor agents → each returns `BUY / SELL / HOLD` with qty, limit price, and reasoning.

This is NOT a paper-reading chatbot and NOT a multi-day market simulation. It's a **single-step decision explorer** — the core contribution of the paper (profile-conditioned LLM agents) wrapped in a single Streamlit page. Multi-agent market dynamics from paper §4 are out of scope; that would warrant a separate app.

## Commands

```bash
# venv — REQUIRES Python 3.10+ (langchain v1 won't install on 3.9)
/opt/anaconda3/bin/python3.13 -m venv venv
./venv/bin/pip install --quiet --upgrade pip
./venv/bin/pip install --quiet -r requirements.txt

# Run
./venv/bin/streamlit run app.py

# Smoke test
./venv/bin/streamlit run app.py --server.headless true --server.port 8503 &
sleep 5 && curl -sf http://localhost:8503/ > /dev/null && echo OK
pkill -f "asfm-trading-agent.*streamlit"

# Pipeline sanity check (no Streamlit, no API)
./venv/bin/python -c "from asfm_pipeline import AGENT_PROFILES, build_observation; print(len(AGENT_PROFILES), 'profiles')"
```

No test suite configured. `python3 -m py_compile *.py` for syntax.

## Architecture

Two-file design:

| File | Role |
|---|---|
| `asfm_pipeline.py` | `AGENT_PROFILES` (4 investor types) · `TradingDecision` Pydantic schema · `build_observation()` · `run_agent()` (calls `llm.with_structured_output(TradingDecision)`) · `apply_decision()` (simulates execution at limit price). No Streamlit imports. |
| `app.py` | Streamlit UI: sidebar (auth, LLM model, run-mode, agent selector, scenario loader) + two-column input (company + wallet + live mini-chart) + results tabs with `qs-decision` cards. Inline QuantSeras CSS. |

Runtime flow: click **Run agent(s)** → for each selected agent, build `profile+observation` prompts → `llm.with_structured_output(TradingDecision).invoke(...)` → `run_agent()` post-normalises (clamps qty against cash/shares, coerces action to `{BUY,SELL,HOLD}`) → `apply_decision()` computes portfolio delta → render cards + dataframe.

## Key design decisions to preserve

- Investor profiles live in `AGENT_PROFILES` dict. Adding a fifth type: add entry with `name`, `emoji`, `tagline`, `color`, `strategy`. Keep the strategy at 50-100 words — the LLM uses it verbatim as the persona.
- `TradingDecision.action` is normalized to uppercase literal string {BUY, SELL, HOLD} in Python (not Pydantic Enum) because GPT models sometimes return lowercase / "BUY (partial)" etc. — normalisation happens in `run_agent()`.
- Budget/holdings clamp is enforced post-hoc (not in prompt alone). Prompt asks the model to respect it, but we re-validate because models hallucinate quantities.
- Widget `key=` matches `st.session_state` key (e.g. `key="company_name"`). Scenario loader writes to session state + `st.rerun()` — don't separate these.
- Prices input is a single comma-separated string (`price_history_str`), parsed in app.py. Simpler than a table widget, good enough for 15 floats.
- Compare-all-4 runs agents sequentially (not threaded) — 4 × ~3s = 10-15s total. Acceptable UX. Don't add threading unless latency becomes a complaint — LangChain's structured-output clients aren't thread-safe without extra wrapping.

## Extending

- **Multi-day simulation** — would need an order book, price update rule, and T-step loop. That's a separate app/page; keep this one focused on single-step decisions.
- **Custom profiles** — let users edit the `strategy` text in sidebar to prototype their own persona.
- **Decision history** — append each run's results to `st.session_state["history"]` and show a small time-series of decisions.
- **Cost tracking** — wrap LLM calls to sum input/output tokens × model price, display per-run cost.

## Visual Design

QuantSeras Design System (dark + desaturated green). Tokens live inline in `CUSTOM_CSS` under `--qs-*` CSS variables and mirror `~/Documents/Obsidian Vault/visual Design System/01 - QuantSeras Design System.md`. Decision cards use left-border color-coding: green (BUY) / red (SELL) / dim (HOLD). Don't invent new shades.

## Gotchas

- **Python 3.9 will NOT work** — langchain v1 requires ≥ 3.10. Use `/opt/anaconda3/bin/python3.13`.
- Port 8501/8502 may be taken by other apps (this machine hosts several paper reproductions). Smoke-test on 8503+.
- `.streamlit/secrets.toml` is gitignored — never commit a real key.
- When GPT returns `confidence` as "MEDIUM" or "Medium" or "medium-high", `run_agent()` coerces to one of {low, medium, high}; default fallback is `medium`.
