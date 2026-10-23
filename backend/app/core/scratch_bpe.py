"""
backend/app/core/scratch_bpe.py

Byte-level BPE Tokenizer built from scratch.
Implements: Regex Pre-tokenization, BPE Training, Encoding, Decoding, and Serialization.
Compatible with Python 3.11+.
"""

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Pattern, Tuple
import regex

GPT4_SPLIT_PATTERN: str = (
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


class ByteLevelPreTokenizer:
    """Handles Unicode regex segmentation and UTF-8 byte transformation."""

    def __init__(self, pattern: str = GPT4_SPLIT_PATTERN) -> None:
        self.compiled_pattern: Pattern = regex.compile(pattern)

    def split_to_chunks(self, text: str) -> List[str]:
        if not text:
            return []
        return self.compiled_pattern.findall(text)

    def text_to_byte_sequences(self, text: str) -> List[List[int]]:
        chunks = self.split_to_chunks(text)
        return [list(chunk.encode("utf-8")) for chunk in chunks]


def get_pair_stats(
    sequences: List[List[int]],
    counts: Counter[Tuple[int, int]] | None = None,
) -> Counter[Tuple[int, int]]:
    """Computes consecutive bigram frequencies across independent sequences."""
    stats: Counter[Tuple[int, int]] = Counter() if counts is None else counts
    for seq in sequences:
        for i in range(len(seq) - 1):
            stats[(seq[i], seq[i + 1])] += 1
    return stats


def merge_sequence(
    seq: List[int], pair: Tuple[int, int], new_id: int
) -> List[int]:
    """Replaces all occurrences of `pair` in `seq` with `new_id`."""
    if len(seq) < 2:
        return seq

    first, second = pair
    merged: List[int] = []
    i = 0
    seq_len = len(seq)

    while i < seq_len:
        if i < seq_len - 1 and seq[i] == first and seq[i + 1] == second:
            merged.append(new_id)
            i += 2
        else:
            merged.append(seq[i])
            i += 1

    return merged


class BPETokenizer:
    """End-to-end Byte-level BPE Tokenizer."""

    def __init__(self, pattern: str = GPT4_SPLIT_PATTERN) -> None:
        self.pre_tokenizer = ByteLevelPreTokenizer(pattern=pattern)
        # Merges map: (token_id_1, token_id_2) -> new_token_id
        self.merges: Dict[Tuple[int, int], int] = {}
        # Vocab map: token_id -> raw bytes
        self.vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        # Special tokens: string -> token_id
        self.special_tokens: Dict[str, int] = {}
        self.inverse_special_tokens: Dict[int, str] = {}

    def register_special_tokens(self, special_tokens: Dict[str, int]) -> None:
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        """Trains BPE merges until `vocab_size` is reached."""
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256 to cover all UTF-8 bytes.")

        num_merges = vocab_size - 256
        chunks = self.pre_tokenizer.text_to_byte_sequences(text)

        for i in range(num_merges):
            stats = get_pair_stats(chunks)
            if not stats:
                if verbose:
                    print(f"Stopping early at step {i}: No more mergeable pairs.")
                break

            # Find the most frequent pair
            best_pair = max(stats, key=stats.get)
            new_id = 256 + i

            # Update sequences with the merge
            chunks = [merge_sequence(chunk, best_pair, new_id) for chunk in chunks]

            # Save merge rule and vocab mapping
            self.merges[best_pair] = new_id
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

            if verbose and (i + 1) % 10 == 0:
                print(f"Merge {i + 1}/{num_merges}: {best_pair} -> {new_id} ({self.vocab[new_id]})")

    def _encode_chunk(self, chunk_bytes: List[int]) -> List[int]:
        """Iteratively applies trained merge rules to a single chunk."""
        tokens = list(chunk_bytes)
        while len(tokens) >= 2:
            stats = Counter((tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1))
            # Find the earliest learned merge rule present in this chunk
            pair = min(stats.keys(), key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            tokens = merge_sequence(tokens, pair, self.merges[pair])
        return tokens

    def encode(self, text: str) -> List[int]:
        """Encodes string text into a list of token IDs."""
        chunks = self.pre_tokenizer.text_to_byte_sequences(text)
        encoded_tokens: List[int] = []
        for chunk in chunks:
            encoded_tokens.extend(self._encode_chunk(chunk))
        return encoded_tokens

    def decode(self, tokens: List[int]) -> str:
        """Decodes token IDs back into a UTF-8 string."""
        byte_parts: List[bytes] = []
        for token in tokens:
            if token in self.inverse_special_tokens:
                byte_parts.append(self.inverse_special_tokens[token].encode("utf-8"))
            elif token in self.vocab:
                byte_parts.append(self.vocab[token])
            else:
                raise ValueError(f"Unknown token id: {token}")

        return b"".join(byte_parts).decode("utf-8", errors="replace")

    def save(self, directory: str | Path) -> None:
        """Serializes merges and vocabulary to disk."""
        target_dir = Path(directory)
        target_dir.mkdir(parents=True, exist_ok=True)

        serializable_merges = [
            {"pair": list(pair), "new_id": new_id}
            for pair, new_id in self.merges.items()
        ]
        with open(target_dir / "merges.json", "w", encoding="utf-8") as f:
            json.dump(serializable_merges, f, indent=2)

        serializable_vocab = {
            str(k): v.hex() for k, v in self.vocab.items()
        }
        with open(target_dir / "vocab.json", "w", encoding="utf-8") as f:
            json.dump(serializable_vocab, f, indent=2)

    def load(self, directory: str | Path) -> None:
        """Loads serialized merges and vocabulary from disk."""
        target_dir = Path(directory)

        with open(target_dir / "merges.json", "r", encoding="utf-8") as f:
            serializable_merges = json.load(f)
            self.merges = {
                (item["pair"][0], item["pair"][1]): item["new_id"]
                for item in serializable_merges
            }

        with open(target_dir / "vocab.json", "r", encoding="utf-8") as f:
            serializable_vocab = json.load(f)
            self.vocab = {
                int(k): bytes.fromhex(v) for k, v in serializable_vocab.items()
            }


if __name__ == "__main__":
    training_corpus = """
    Byte Pair Encoding (BPE) is a subword tokenization algorithm.
    It iteratively merges the most frequent pair of bytes or characters in a text.
    BPE powers modern language models like GPT-2, GPT-3, and GPT-4.
    Let's test numbers: 12345 67890, emojis: 🚀 🤖 💡, and contractions: it's, don't, we'll.
    """

    print("--- 1. Initializing and Training BPE Tokenizer ---")
    tokenizer = BPETokenizer()
    target_vocab_size = 280  # 256 base bytes + 24 custom merges
    tokenizer.train(training_corpus, vocab_size=target_vocab_size, verbose=True)

    print("\n--- 2. Testing Encode and Decode ---")
    test_text = "BPE tokenization is modern and powers GPT-4! 🚀"
    encoded_ids = tokenizer.encode(test_text)
    decoded_text = tokenizer.decode(encoded_ids)

    print(f"Original Text : {test_text}")
    print(f"Encoded IDs   : {encoded_ids}")
    print(f"Decoded Text  : {decoded_text}")
    print(f"Roundtrip OK? : {test_text == decoded_text}")

    print("\n--- 3. Testing Save & Load ---")
    tokenizer.save("backend/app/core/model_weights")
    loaded_tokenizer = BPETokenizer()
    loaded_tokenizer.load("backend/app/core/model_weights")
    re_encoded = loaded_tokenizer.encode(test_text)
    print(f"Loaded Tokenizer IDs match? : {encoded_ids == re_encoded}")