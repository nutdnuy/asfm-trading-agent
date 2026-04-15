"""ASFM trading agent pipeline — reproduces Gao et al. (2024).

Paper: "Simulating Financial Market via Large Language Model based Agents"
arXiv:2406.19966 · Jun 2024

Given: company info, price history, economic news, agent profile, wallet state
Return: one trading decision (BUY / SELL / HOLD) with quantity, limit price, and reasoning.

Framework (paper §2.4): C = A(I_a, I_o, I_t)
  - I_a  profile prompt (investor type + wallet + holdings)
  - I_o  observation prompt (price history + orders + economic news)
  - I_t  tool document (SELL / BUY / HOLD with stock_code, quantity, price)
  - A    LLM agent  →  C tool execution code
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI


# =============================================================================
# Investor-type catalog — matches paper §2.4.1 (4 profiles)
# =============================================================================

AGENT_PROFILES: dict[str, dict] = {
    "value": {
        "name": "Value Investor",
        "emoji": "📊",
        "tagline": "Intrinsic value · long-term holdings",
        "color": "#69F0AE",
        "strategy": (
            "You focus on finding stocks whose market prices are below their "
            "intrinsic value. You analyze the company's business fundamentals, "
            "financial health, and competitive position, then invest in "
            "high-quality companies undervalued by the market. You prefer "
            "long-term holdings over short-term trading. You avoid overpriced "
            "stocks even when they are popular, and you are willing to wait "
            "for the market to recognize true value."
        ),
    },
    "institutional": {
        "name": "Institutional Investor",
        "emoji": "🏛️",
        "tagline": "Steady returns · capital preservation",
        "color": "#03DAC6",
        "strategy": (
            "You manage a large portfolio on behalf of institutional clients "
            "(pension funds, insurance companies, mutual funds). You prioritize "
            "capital preservation and steady, predictable returns. You avoid "
            "concentrated positions, prefer liquid blue-chip names with strong "
            "balance sheets, and react conservatively to news — preferring to "
            "wait for confirmation rather than trade on speculation. You do "
            "not chase momentum."
        ),
    },
    "contrarian": {
        "name": "Contrarian Investor",
        "emoji": "🔄",
        "tagline": "Against the crowd · buy fear, sell greed",
        "color": "#FFB74D",
        "strategy": (
            "You deliberately act against prevailing market sentiment. You BUY "
            "when others sell in panic and SELL when others chase euphoria. "
            "You look for oversold names amid bad news (where the reaction is "
            "overdone) and overbought names amid positive hype. You are "
            "disciplined, willing to be early, and comfortable being wrong in "
            "the short term if your thesis on reversion is sound."
        ),
    },
    "aggressive": {
        "name": "Aggressive Investor",
        "emoji": "⚡",
        "tagline": "Momentum · high risk / high reward",
        "color": "#FF5252",
        "strategy": (
            "You seek high returns and are willing to accept high risk. You "
            "chase momentum, react quickly to positive news, take concentrated "
            "positions in high-conviction ideas, and actively trade short-term "
            "price moves. You are comfortable with volatility and prefer to "
            "act decisively — partial positions or hedging are not your style."
        ),
    },
}


# =============================================================================
# System prompt — paper §2.4 (profile + observation + tool-learning)
# =============================================================================

AGENT_SYSTEM_TEMPLATE = """You are an investor in a simulated stock market.

## Your investor profile
{strategy}

## Your task
You observe (1) your wallet and holdings, (2) recent prices of a target stock,
(3) the company's business description, and (4) the latest economic news.
Then you decide ONE action for the target stock for TODAY:

- BUY   open or increase a position   (quantity > 0, limit_price = desired buy price)
- SELL  close or reduce a position    (quantity > 0, only if you currently hold shares)
- HOLD  do nothing today               (quantity = 0, limit_price = 0)

## Rules
- At most ONE buy or sell order per day.
- Respect your budget and holdings:
    · BUY:   quantity × limit_price ≤ wallet cash
    · SELL:  quantity ≤ shares currently held
- Limit price must be realistic — within ~10% of the current market price.
- Your reasoning MUST explicitly reference both (a) your investor profile and
  (b) observed market / news data. No generic filler.
- Stay in character — do NOT break your profile's discipline."""


# =============================================================================
# Structured output schema
# =============================================================================

class TradingDecision(BaseModel):
    """One trading decision from an LLM investor agent."""

    action: str = Field(
        description="Exactly one of: BUY, SELL, HOLD (uppercase).",
    )
    quantity: int = Field(
        description="Number of shares. 0 when HOLD.",
        ge=0,
    )
    limit_price: float = Field(
        description="Limit price per share in the same currency as input prices. 0.0 when HOLD.",
        ge=0,
    )
    confidence: str = Field(
        description="One of: low, medium, high.",
    )
    reasoning: str = Field(
        description=(
            "2-4 sentences explaining the decision. Must reference both the "
            "investor profile and specific observed data (trend, news, wallet state)."
        ),
    )


# =============================================================================
# Observation builder — paper §2.4.2
# =============================================================================

def build_observation(
    company_name: str,
    sector: str,
    business: str,
    price_history: list[float],
    economic_news: str,
    cash: float,
    shares_held: int,
) -> str:
    """Format the observation prompt fed to the agent."""
    current_price = price_history[-1] if price_history else 0.0
    position_value = shares_held * current_price
    total_equity = cash + position_value

    # compact price trend summary
    if len(price_history) >= 2:
        pct = (current_price - price_history[0]) / price_history[0] * 100
        trend_str = f"{pct:+.1f}% over window"
    else:
        trend_str = "n/a"

    prices_fmt = ", ".join(f"{p:.2f}" for p in price_history)

    return (
        f"## Target stock\n"
        f"- Company: **{company_name}**\n"
        f"- Sector: {sector}\n"
        f"- Business: {business}\n\n"
        f"## Price history — last {len(price_history)} trading days\n"
        f"{prices_fmt}\n"
        f"Current price: **{current_price:.2f}** · Trend: {trend_str}\n\n"
        f"## Economic news\n{economic_news.strip()}\n\n"
        f"## Your wallet\n"
        f"- Cash available: ${cash:,.2f}\n"
        f"- Shares held: {shares_held}\n"
        f"- Position value: ${position_value:,.2f}\n"
        f"- Total equity: ${total_equity:,.2f}\n\n"
        f"Decide ONE action for today and return structured output."
    )


# =============================================================================
# Agent runner
# =============================================================================

def run_agent(
    agent_type: str,
    company_name: str,
    sector: str,
    business: str,
    price_history: list[float],
    economic_news: str,
    cash: float,
    shares_held: int,
    llm: ChatOpenAI,
) -> TradingDecision:
    """Run one LLM investor agent and return its trading decision."""
    if agent_type not in AGENT_PROFILES:
        raise ValueError(f"Unknown agent_type: {agent_type}")

    profile = AGENT_PROFILES[agent_type]
    system = AGENT_SYSTEM_TEMPLATE.format(strategy=profile["strategy"])
    obs = build_observation(
        company_name, sector, business,
        price_history, economic_news,
        cash, shares_held,
    )

    structured = llm.with_structured_output(TradingDecision)
    result = structured.invoke([
        {"role": "system", "content": system},
        {"role": "user", "content": obs},
    ])

    # Normalise + enforce constraints the model sometimes breaks
    action = result.action.strip().upper()
    if action not in ("BUY", "SELL", "HOLD"):
        action = "HOLD"

    qty = max(0, int(result.quantity))
    limit = max(0.0, float(result.limit_price))

    if action == "HOLD":
        qty, limit = 0, 0.0
    elif action == "SELL" and qty > shares_held:
        qty = shares_held
        if qty == 0:
            action = "HOLD"
            limit = 0.0
    elif action == "BUY":
        current_price = price_history[-1] if price_history else limit
        price_for_check = limit if limit > 0 else current_price
        max_affordable = int(cash // price_for_check) if price_for_check > 0 else 0
        if qty > max_affordable:
            qty = max_affordable
        if qty == 0:
            action = "HOLD"
            limit = 0.0

    confidence = result.confidence.strip().lower()
    if confidence not in ("low", "medium", "high"):
        confidence = "medium"

    return TradingDecision(
        action=action,
        quantity=qty,
        limit_price=limit,
        confidence=confidence,
        reasoning=result.reasoning,
    )


# =============================================================================
# Post-decision bookkeeping
# =============================================================================

def apply_decision(
    decision: TradingDecision,
    cash: float,
    shares_held: int,
    current_price: float,
) -> dict:
    """Simulate execution at the limit price (if BUY/SELL). Returns new state."""
    new_cash = cash
    new_shares = shares_held

    if decision.action == "BUY" and decision.quantity > 0:
        spend = decision.quantity * decision.limit_price
        new_cash = cash - spend
        new_shares = shares_held + decision.quantity
    elif decision.action == "SELL" and decision.quantity > 0:
        proceeds = decision.quantity * decision.limit_price
        new_cash = cash + proceeds
        new_shares = shares_held - decision.quantity

    new_position_value = new_shares * current_price
    return {
        "cash": new_cash,
        "shares": new_shares,
        "position_value": new_position_value,
        "total_equity": new_cash + new_position_value,
        "delta_cash": new_cash - cash,
        "delta_shares": new_shares - shares_held,
    }
