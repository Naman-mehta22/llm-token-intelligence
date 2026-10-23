from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.core.benchmarks import tokenize_with_tiktoken
from backend.app.core.pricing_engine import MODEL_PRICING, calculate_cost
from backend.app.core.scratch_bpe import BPETokenizer
from backend.app.core.optimizer import optimize_prompt
from backend.app.schemas import (
    ModelTokenStats,
    TokenChunk,
    TokenizeRequest,
    TokenizeResponse,
    OptimizationResult,
)

app = FastAPI(title="LLM Token Counter & Cost Intelligence API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

custom_bpe = BPETokenizer()
try:
    custom_bpe.load("backend/app/core/model_weights")
    HAS_CUSTOM_MODEL = True
except Exception:
    HAS_CUSTOM_MODEL = False


@app.get("/health")
def health():
    return {"status": "ok", "custom_model_loaded": HAS_CUSTOM_MODEL}


@app.post("/api/tokenize", response_model=TokenizeResponse)
def tokenize_prompt(payload: TokenizeRequest):
    text = payload.text
    char_count = len(text)
    word_count = len(text.split())

    # 1. Process selected model
    rates = MODEL_PRICING.get(payload.model, MODEL_PRICING["gpt-4o"])
    token_ids, chunks = tokenize_with_tiktoken(text, rates["encoding"])
    in_cost, out_costs = calculate_cost(
        payload.model, len(token_ids), payload.projected_output_tokens
    )

    selected_stats = ModelTokenStats(
        model_name=payload.model,
        token_count=len(token_ids),
        tokens=chunks,
        character_count=char_count,
        word_count=word_count,
        estimated_input_cost=in_cost,
        projected_output_costs=out_costs,
    )

    # 2. Run benchmarks across all supported models
    benchmarks = {}
    for m_name, meta in MODEL_PRICING.items():
        m_ids, _ = tokenize_with_tiktoken(text, meta["encoding"])
        m_in_cost, m_out_costs = calculate_cost(
            m_name, len(m_ids), payload.projected_output_tokens
        )
        benchmarks[m_name] = ModelTokenStats(
            model_name=m_name,
            token_count=len(m_ids),
            tokens=None,
            character_count=char_count,
            word_count=word_count,
            estimated_input_cost=m_in_cost,
            projected_output_costs=m_out_costs,
        )

    # 3. Process Custom Scratch BPE
    custom_stats = None
    if payload.include_custom_bpe and HAS_CUSTOM_MODEL:
        c_token_ids = custom_bpe.encode(text)
        custom_chunks = [
            TokenChunk(
                id=tid,
                text=custom_bpe.decode([tid]),
                bytes=list(custom_bpe.vocab.get(tid, b"")),
                color_idx=i % 6,
            )
            for i, tid in enumerate(c_token_ids)
        ]
        custom_stats = ModelTokenStats(
            model_name="Scratch BPE (Custom)",
            token_count=len(c_token_ids),
            tokens=custom_chunks,
            character_count=char_count,
            word_count=word_count,
            estimated_input_cost=0.0,
            projected_output_costs={},
        )

    # 4. Generate Optimization Recommendation
    opt_text, applied_rules = optimize_prompt(text)
    opt_token_ids, _ = tokenize_with_tiktoken(opt_text, rates["encoding"])
    opt_cost, _ = calculate_cost(payload.model, len(opt_token_ids))

    orig_tokens = len(token_ids)
    opt_tokens = len(opt_token_ids)
    tokens_saved = max(0, orig_tokens - opt_tokens)
    pct_saved = (tokens_saved / orig_tokens * 100) if orig_tokens > 0 else 0.0
    cost_saved_1m = (tokens_saved / 1_000_000) * rates["input_per_m"] * 1_000_000

    optimization_payload = OptimizationResult(
        original_tokens=orig_tokens,
        optimized_tokens=opt_tokens,
        tokens_saved=tokens_saved,
        percentage_saved=round(pct_saved, 1),
        original_cost=in_cost,
        optimized_cost=opt_cost,
        cost_saved_per_1m_calls=round(cost_saved_1m, 2),
        optimized_text=opt_text,
        applied_rules=applied_rules,
    )

    return TokenizeResponse(
        selected_model=selected_stats,
        benchmarks=benchmarks,
        custom_bpe=custom_stats,
        optimization=optimization_payload,
    )