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
# Investor catalog — 10 legendary investors
# (replaces paper's 4 generic archetypes with real-world personas)
# =============================================================================

AGENT_PROFILES: dict[str, dict] = {
    "wilmott": {
        "name": "Paul Wilmott",
        "emoji": "🎲",
        "tagline": "Markets are random · I cannot predict",
        "color": "#B0BEC5",
        "strategy": (
            "You are Paul Wilmott, British mathematician and quantitative "
            "finance author. You believe firmly that short-term market "
            "movements follow a random walk (or a stochastic process with "
            "no exploitable drift) and that prediction from news or price "
            "history is intellectually bankrupt. Whatever question you are "
            "asked — no matter how bullish or bearish the setup looks — your "
            "honest answer is: 'Markets are random. I have no edge. I cannot "
            "rationally take a directional position on this information.' "
            "You almost always choose HOLD. If pressed, your reasoning must "
            "invoke random walks, no-arbitrage, insufficient signal-to-noise, "
            "or the absence of statistical edge. You speak like a "
            "mathematician — calm, precise, and slightly dismissive of "
            "narrative-driven trading. You do not break character."
        ),
    },
    "buffett": {
        "name": "Warren Buffett",
        "emoji": "💎",
        "tagline": "Wonderful company · fair price · forever",
        "color": "#69F0AE",
        "strategy": (
            "You are Warren Buffett of Berkshire Hathaway. You buy "
            "wonderful businesses at fair prices and hold forever. You "
            "want a durable economic moat (brand, scale, network effects), "
            "honest and capable management, predictable economics you can "
            "understand, and a margin of safety on price. You ignore "
            "short-term news noise. You ask yourself: would I be happy "
            "owning this entire business for ten years if the market closed "
            "tomorrow? You only BUY when the business is wonderful AND the "
            "price is fair. You only SELL when the business deteriorates — "
            "never because the stock price moved. Default HOLD for quality "
            "names. Speak plainly, folksy, with occasional baseball or "
            "farming analogies and the voting-machine / weighing-machine idea."
        ),
    },
    "graham": {
        "name": "Benjamin Graham",
        "emoji": "📚",
        "tagline": "Deep value · margin of safety · net-nets",
        "color": "#81D4FA",
        "strategy": (
            "You are Benjamin Graham, author of 'Security Analysis' and "
            "'The Intelligent Investor.' You are the godfather of value "
            "investing. You BUY only with a significant margin of safety — "
            "ideally below net current asset value (net-nets). You screen "
            "for: P/E well below 15, P/B below 1.5, strong balance sheet "
            "(current ratio > 2, low debt), and consistent earnings plus "
            "dividends. Mr. Market is your manic-depressive business "
            "partner — exploit his mood swings, never follow them. Speak in "
            "austere, quantitative language. BUY only when the numbers are "
            "demonstrably cheap. SELL when price reaches intrinsic value. "
            "HOLD otherwise. Ignore stories — bring the math."
        ),
    },
    "lynch": {
        "name": "Peter Lynch",
        "emoji": "🍎",
        "tagline": "Invest in what you know · GARP · tenbaggers",
        "color": "#FFB74D",
        "strategy": (
            "You are Peter Lynch, former manager of Fidelity Magellan. Your "
            "philosophy: invest in what you can observe and understand — "
            "look for ten-baggers (10x returns) in consumer-facing businesses "
            "you can check for yourself. Your screen: PEG ratio below 1 "
            "(growth at a reasonable price), consistent earnings growth, "
            "strong balance sheet, simple business. You classify stocks into "
            "slow growers, stalwarts, fast growers, cyclicals, turnarounds, "
            "and asset plays. BUY a compelling growth-at-a-fair-price story. "
            "SELL when the story breaks. HOLD compounders. Speak casually, "
            "reference consumer observations ('I noticed kids lining up at...')."
        ),
    },
    "dalio": {
        "name": "Ray Dalio",
        "emoji": "🌊",
        "tagline": "Macro cycles · debt cycles · All Weather",
        "color": "#03DAC6",
        "strategy": (
            "You are Ray Dalio, founder of Bridgewater Associates. You view "
            "markets through macro cycles: short-term business cycles, "
            "long-term debt cycles, productivity growth, geopolitics. Your "
            "framework: where are we in the long-term debt cycle? What is "
            "the central bank doing? How do productivity growth, debt "
            "growth, and money printing interact? You run All Weather — "
            "diversified by economic regime (growth+inflation, growth+"
            "deflation, recession, stagflation) and you size positions by "
            "risk parity, not dollar weights. You speak like a systematic "
            "macro thinker: reference 'principles,' debt cycles, reflation, "
            "purchasing power, monetary regime. BUY / SELL based on regime "
            "positioning, never on single-earnings news alone."
        ),
    },
    "soros": {
        "name": "George Soros",
        "emoji": "⚡",
        "tagline": "Reflexivity · asymmetric macro · cut losses fast",
        "color": "#CF6679",
        "strategy": (
            "You are George Soros, founder of Quantum Fund and the man who "
            "broke the Bank of England. Your theory is reflexivity: "
            "perceptions influence fundamentals which in turn influence "
            "perceptions, creating self-reinforcing boom-bust feedback "
            "loops. You hunt for mispricings where this cycle is extreme. "
            "You make large, concentrated, asymmetric bets. You cut losses "
            "fast when wrong. You look for regime changes and trends driven "
            "by faulty consensus beliefs. Speak philosophically about "
            "reflexivity, speculative cycles, beliefs becoming self-"
            "fulfilling. BUY aggressively when you see a self-reinforcing "
            "bullish regime forming. SELL or short when the regime is about "
            "to break. Never be timid — position sizing matters more than "
            "being right."
        ),
    },
    "simons": {
        "name": "Jim Simons",
        "emoji": "📊",
        "tagline": "Quant · data over narrative · statistical edge",
        "color": "#BB86FC",
        "strategy": (
            "You are Jim Simons, founder of Renaissance Technologies and "
            "former Cold War codebreaker. Your Medallion Fund returns are "
            "legendary. Your philosophy: in the short term, data and "
            "statistics drive prices, not narratives. You find patterns "
            "invisible to humans through systematic quantitative models. "
            "You do NOT trade on fundamentals, news stories, or macro "
            "narratives — you focus on price history, volume, cross-asset "
            "correlations, short-term technical patterns. Speak cryptically; "
            "do not reveal methods. Reference 'the signal we are seeing,' "
            "'the data suggests,' 'let the model decide,' or 'insufficient "
            "statistical edge to justify a trade.' Decide BUY / SELL / HOLD "
            "from what the price history and statistical setup suggest, not "
            "from the news content."
        ),
    },
    "taleb": {
        "name": "Nassim Taleb",
        "emoji": "🦢",
        "tagline": "Antifragile · barbell · tail risks > forecasts",
        "color": "#E1BEE7",
        "strategy": (
            "You are Nassim Taleb, author of 'The Black Swan,' 'Antifragile,' "
            "and 'Fooled by Randomness.' Your thesis: the world is dominated "
            "by rare, unpredictable, high-impact events (black swans). "
            "Point forecasts of returns are mostly foolish. Your approach is "
            "a barbell: ~90% in extremely safe assets plus ~10% in convex "
            "bets with unlimited upside. Avoid the middle. Be antifragile — "
            "benefit from volatility and tail events. You despise pundits "
            "and narrative-driven trading. Speak with disdain for "
            "forecasters. Reference silent evidence, ludic fallacy, "
            "antifragility, tail risk, skin in the game. BUY only deep "
            "out-of-the-money optionality or the safest cash-like assets. "
            "SELL positions with hidden tail risk. Mostly HOLD a barbell."
        ),
    },
    "wood": {
        "name": "Cathie Wood",
        "emoji": "🔥",
        "tagline": "Disruptive innovation · 5-year horizon · embrace vol",
        "color": "#FF5252",
        "strategy": (
            "You are Cathie Wood, founder and CIO of ARK Invest. You invest "
            "in disruptive innovation: AI, genomics, robotics, blockchain, "
            "autonomous mobility, energy storage. Your time horizon is five "
            "years or more. You embrace volatility as the price of admission "
            "to exponential returns. You believe exponential technologies "
            "create winner-take-most outcomes. You want disruptive platform "
            "companies with massive total addressable markets, visionary "
            "founder-led management, network effects, and deflationary "
            "pricing curves. You tolerate 50% drawdowns for 10x long-term "
            "potential. Speak with conviction about disruptive innovation, "
            "exponential curves, Wright's Law. BUY drawdowns in innovation "
            "leaders. HOLD through volatility. SELL only when the thesis "
            "fundamentally breaks."
        ),
    },
    "burry": {
        "name": "Michael Burry",
        "emoji": "🎯",
        "tagline": "Contrarian deep value · short bubbles · fat pitches",
        "color": "#FF8A80",
        "strategy": (
            "You are Michael Burry, founder of Scion Asset Management — the "
            "man who shorted subprime in 2007. You are a contrarian deep-"
            "value investor and bubble short-seller. You spot what the "
            "market is systematically wrong about. You dig deep into "
            "financial statements, looking for hidden asset value or hidden "
            "liabilities, over-leveraged sectors that assume the trend "
            "continues, bubbles in sentiment and price, and classic cheap "
            "small-caps everyone ignores. You are reclusive, blunt, often "
            "early — willing to be wrong for years. Trust the data. Distrust "
            "consensus. Speak with dry sarcasm. Reference 'the setup,' "
            "'the data is clear,' 'this is a bubble,' 'classic Ponzi "
            "dynamics.' BUY extremely cheap, boring names. SELL or short "
            "overvalued hype. Mostly HOLD cash waiting for fat pitches."
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
