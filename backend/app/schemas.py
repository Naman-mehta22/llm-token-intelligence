from typing import Dict, List, Optional
from pydantic import BaseModel


class TokenChunk(BaseModel):
    id: int
    text: str
    bytes: List[int]
    color_idx: int


class ModelTokenStats(BaseModel):
    model_name: str
    token_count: int
    tokens: Optional[List[TokenChunk]] = None
    character_count: int
    word_count: int
    estimated_input_cost: float
    projected_output_costs: Dict[str, float]


class OptimizationResult(BaseModel):
    original_tokens: int
    optimized_tokens: int
    tokens_saved: int
    percentage_saved: float
    original_cost: float
    optimized_cost: float
    cost_saved_per_1m_calls: float
    optimized_text: str
    applied_rules: List[str]


class TokenizeRequest(BaseModel):
    text: str
    model: str = "gpt-4o"
    include_custom_bpe: bool = True
    projected_output_tokens: List[int] = [250, 500, 1000]


class TokenizeResponse(BaseModel):
    selected_model: ModelTokenStats
    benchmarks: Dict[str, ModelTokenStats]
    custom_bpe: Optional[ModelTokenStats] = None
    optimization: Optional[OptimizationResult] = None