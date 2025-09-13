import os
from typing import List, Callable
import anthropic
from dotenv import load_dotenv
import time

# ---------------------------
# Tokenization helpers
# ---------------------------
def _load_tokenizer():
    """
    Try to load tiktoken tokenizer; return (encode_fn, decode_fn, tokens_per_char_estimate).
    If tiktoken isn't available, return character-based fallback.
    """
    try:
        import tiktoken
        # cl100k_base is a decent default across many chat models
        enc = tiktoken.get_encoding("cl100k_base")
        return enc.encode, enc.decode, None  # exact tokenizer
    except Exception:
        # Fallback: approximate ~4 chars per token (very rough)
        # We'll simulate tokenization by slicing on characters
        return None, None, 4.0

def tokenize(text: str, encode_fn: Callable, tpc_estimate: float):
    if encode_fn is not None:
        return encode_fn(text)  # returns token IDs list
    else:
        # char-based pseudo "tokens"
        return list(text)  # one "token" per char

def detokenize(tokens: List[int], decode_fn: Callable, tpc_estimate: float):
    if decode_fn is not None:
        return decode_fn(tokens)
    else:
        # char-based
        return "".join(tokens)

def chunk_tokens(tokens: List[int], max_tokens: int, overlap: int) -> List[List[int]]:
    chunks = []
    i = 0
    n = len(tokens)
    while i < n:
        j = min(i + max_tokens, n)
        chunks.append(tokens[i:j])
        if j >= n:
            break
        # next start index with overlap
        i = j - overlap if j - overlap > i else j
    return chunks

# ---------------------------
# Model call helper
# ---------------------------
def call_model(client, model_name: str, user_prompt: str, max_tokens: int = 1024) -> str:
    """
    Wrap your existing call – adapt this if your SDK differs.
    Expects a Messages API that returns .content (string or list); normalizes to string.
    """
    resp = client.messages.create(
        model=model_name,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": user_prompt}],
    )
    # Normalize to string depending on SDK shape
    content = getattr(resp, "content", resp)
    if isinstance(content, list):
        # Some SDKs return a list of blocks
        return "\n".join(str(x) for x in content)
    return str(content)

# ---------------------------
# Main pipeline
# ---------------------------
def summarize_large_file(
    path: str,
    client,
    model_name: str = "claude-opus-4-1-20250805",
    per_chunk_token_budget: int = 9000,    # < 10k as requested
    per_chunk_overlap_tokens: int = 200,   # smooth boundaries
    per_chunk_max_output_tokens: int = 700,
    final_max_output_tokens: int = 700
) -> str:
    """
    1) Chunk file to <10k tokens
    2) Summarize each chunk to 3 bullets
    3) Synthesize final 3 bullets
    """
    with open(path, "r", encoding="utf-8") as f:
        big_text = f.read()

    encode_fn, decode_fn, tpc_est = _load_tokenizer()
    tokens = tokenize(big_text, encode_fn, tpc_est)

    # Guardrails: ensure budgets make sense
    max_tokens = max(1000, min(per_chunk_token_budget, 9900))
    overlap = max(0, min(per_chunk_overlap_tokens, max_tokens // 3))

    token_chunks = chunk_tokens(tokens, max_tokens, overlap)
    text_chunks = [detokenize(tc, decode_fn, tpc_est) for tc in token_chunks]

    # Per-chunk prompt
    per_chunk_prompt_tpl = (
        "Analyze the following document chunk:\n\n"
        "{chunk}\n\n"
        "Extract ALL key concepts from this chunk.\n"
        "Rules:\n"
        " - List every important technical or domain-specific concept.\n"
        " - Each concept must be a short noun phrase (≤ 3 words).\n"
        " - Use lowercase unless it’s a proper noun.\n"
        " - Avoid redundancy within this chunk.\n"
        " - Focus on unique, meaningful terms (not filler words).\n\n"
        "Output format: one concept per line, no numbering or punctuation."
    )

    chunk_summaries = []
    for idx, chunk_text in enumerate(text_chunks, 1):
        prompt = per_chunk_prompt_tpl.format(chunk=chunk_text)
        summary = call_model(client, model_name, prompt, max_tokens=per_chunk_max_output_tokens)
        chunk_summaries.append(f"Chunk {idx} summary:\n{summary}")

    # Aggregate + synthesize final
    aggregate_text = "\n\n".join(chunk_summaries)
    final_prompt = (
        "You are given lists of concepts, each from a different chunk of the same document.\n\n"
        "Concepts from all chunks:\n"
        f"{aggregate_text}\n\n"
        "Task: Merge these into a single, **deduplicated** list of key concepts for the entire document.\n"
        "Rules:\n"
        " - Keep concepts short (≤ 3 words).\n"
        " - Remove duplicates and near-duplicates (e.g., merge 'binary search' and 'binary search algorithm').\n"
        " - Keep broad, general terms; drop overly local details.\n"
        " - Use lowercase unless it's a proper noun.\n"
        " - Output ONLY the final list, one concept per line, no numbering or punctuation.\n"
    )
    final_summary = call_model(client, model_name, final_prompt, max_tokens=final_max_output_tokens)
    return final_summary

# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    load_dotenv() 
    # 1) Provide your API client however you normally do:
    # from your_sdk import Client
    # client = Client(api_key=os.environ["API_KEY"])
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))  # <-- replace with your instantiated client

    if client is None:
        raise RuntimeError("Please instantiate your model client and assign it to `client`.")

    start = time.time()
    final = summarize_large_file(
        path="./sorting_algorithms.txt",
        client=client,
        model_name="claude-opus-4-1-20250805",  # keep your model name
        per_chunk_token_budget=9000,
        per_chunk_overlap_tokens=200,
        per_chunk_max_output_tokens=700,
        final_max_output_tokens=700,
    )
    end = time.time()
    duration = end - start
    print(f"Elapsed time: {duration:.4f} seconds")
    print("\n=== FINAL 3-BULLET SUMMARY ===\n")
    print(final)
