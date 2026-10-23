from typing import List, Tuple
import tiktoken
from backend.app.schemas import TokenChunk


def tokenize_with_tiktoken(
    text: str, encoding_name: str = "o200k_base"
) -> Tuple[List[int], List[TokenChunk]]:
    """Encodes text using tiktoken and decomposes it into styled UI chunks."""
    try:
        enc = tiktoken.get_encoding(encoding_name)
    except ValueError:
        enc = tiktoken.get_encoding("cl100k_base")

    token_ids = enc.encode(text)
    chunks: List[TokenChunk] = []

    # Map each token ID back to byte chunks and string representation
    for idx, t_id in enumerate(token_ids):
        token_bytes = enc.decode_single_token_bytes(t_id)
        # Attempt to decode string representation; fallback to raw byte escape
        try:
            chunk_text = token_bytes.decode("utf = 8")
        except UnicodeDecodeError:
            chunk_text = "".join(f"<0x{b:02x}>" for b in token_bytes)

        chunks.append(
            TokenChunk(
                id=t_id,
                text=chunk_text,
                bytes=list(token_bytes),
                color_idx=idx % 6,  # 6 alternating highlight colors for UI
            )
        )

    return token_ids, chunks