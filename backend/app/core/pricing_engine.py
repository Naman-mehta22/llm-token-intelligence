from typing import Dict, List

# Pricing per 1 Million tokens (USD)
MODEL_PRICING = {
    "gpt-4o": {
        "provider": "OpenAI",
        "input_per_m": 2.50,
        "output_per_m": 10.00,
        "encoding": "o200k_base",
    },
    "gpt-4o-mini": {
        "provider": "OpenAI",
        "input_per_m": 0.15,
        "output_per_m": 0.60,
        "encoding": "o200k_base",
    },
    "gpt-4-turbo": {
        "provider": "OpenAI",
        "input_per_m": 10.00,
        "output_per_m": 30.00,
        "encoding": "cl100k_base",
    },
    "claude-3-5-sonnet": {
        "provider": "Anthropic",
        "input_per_m": 3.00,
        "output_per_m": 15.00,
        "encoding": "cl100k_base",  # Approximation for comparison
    },
    "gemini-1-5-flash": {
        "provider": "Google",
        "input_per_m": 0.075,
        "output_per_m": 0.30,
        "encoding": "o200k_base",  # Approximation for comparison
    },
}


def calculate_cost(
    model_name: str,
    token_count: int,
    output_projection_steps: List[int] | None = None,
) -> tuple[float, Dict[str, float]]:
    """Calculates input prompt cost and projected output token costs."""
    if output_projection_steps is None:
        output_projection_steps = [250, 500, 1000]

    rates = MODEL_PRICING.get(model_name, MODEL_PRICING["gpt-4o"])
    input_cost = (token_count / 1_000_000) * rates["input_per_m"]

    projected_outputs = {}
    for step in output_projection_steps:
        projected_outputs[f"{step}_tokens"] = (
            step / 1_000_000
        ) * rates["output_per_m"]

    return round(input_cost, 6), {k: round(v, 6) for k, v in projected_outputs.items()}