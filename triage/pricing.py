"""What a response cost, from its usage. The bot's running total, and so its monthly cap, comes from here."""

from typing import Any

# $ per million tokens (input, output). Cache reads cost a tenth of input, five-minute cache writes 1.25 times.
# A model missing here (a refusal fallback) is priced at Opus 4.8's rate, which overestimates rather than under.
# Haiku's rate is for prompts up to 100K tokens; the researcher stops reading before it gets there.
PRICES: dict[str, tuple[float, float]] = {
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-haiku-5-5": (0.1, 0.5),
}
UNKNOWN_PRICE = (5.0, 25.0)
WEB_SEARCH = 0.01
"""$10 per 1,000 searches; web fetch costs only its tokens."""


def price(model: str, usage: Any) -> float:
    rate_in, rate_out = next((p for name, p in PRICES.items() if model.startswith(name)), UNKNOWN_PRICE)
    cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
    cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
    server = getattr(usage, "server_tool_use", None)
    searches = (getattr(server, "web_search_requests", 0) or 0) if server else 0
    tokens = (
        usage.input_tokens * rate_in
        + cache_read * rate_in * 0.1
        + cache_write * rate_in * 1.25
        + usage.output_tokens * rate_out
    ) / 1_000_000
    return float(tokens + searches * WEB_SEARCH)
