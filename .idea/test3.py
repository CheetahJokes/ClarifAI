import os
from typing import List, Callable, Dict, Any
import anthropic
from dotenv import load_dotenv
import time
import math
import json
from contextlib import contextmanager

# Optional: CPU/mem snapshots if psutil available
try:
    import psutil
except Exception:
    psutil = None

# ===========
# Profiler
# ===========
PROFILE_VERBOSE = True  # set False for quieter logs

class Profiler:
    def __init__(self):
        self.t0 = time.perf_counter()
        self.marks: Dict[str, float] = {}
        self.events: Dict[str, float] = {}
        self.details: Dict[str, Any] = {"chunks": []}

    @contextmanager
    def span(self, name: str):
        start = time.perf_counter()
        if PROFILE_VERBOSE:
            print(f"[PROFILE] ▶ {name} …")
        try:
            yield
        finally:
            dur = time.perf_counter() - start
            self.events[name] = self.events.get(name, 0.0) + dur
            if PROFILE_VERBOSE:
                print(f"[PROFILE] ◀ {name} took {dur:.4f}s")

    def add_chunk_stat(self, idx: int, **kwargs):
        self.details["chunks"].append({"idx": idx, **kwargs})

    def snapshot(self, tag: str):
        if psutil:
            proc = psutil.Process(os.getpid())
            rss = proc.memory_info().rss
            cpu = psutil.cpu_percent(interval=None)
            self.details.setdefault("snapshots", []).append(
                {"tag": tag, "rss_bytes": rss, "cpu_percent": cpu, "t": time.perf_counter() - self.t0}
            )

    def report(self):
        total = time.perf_counter() - self.t0
        # Build sorted breakdown
        items = sorted(self.events.items(), key=lambda kv: kv[1], reverse=True)
        print("\n===== PROFILE REPORT =====")
        print(f"Total wall time: {total:.4f}s")
        for k, v in items:
            pct = (v / total * 100) if total > 0 else 0.0
            print(f"{k:28s} {v:8.4f}s  ({pct:5.1f}%)")
        # Chunk table (API latencies and sizes)
        if self.details["chunks"]:
            print("\nPer-chunk API calls:")
            print(f"{'chunk':>5s} {'tok_in':>7s} {'tok_out?':>8s} {'prompt_build':>12s} {'api_latency':>11s}")
            for c in self.details["chunks"]:
                print(f"{c['idx']:5d} {c.get('tokens_in','?'):7} {c.get('tokens_out','?'):8} "
                      f"{c.get('prompt_build_s',0):12.4f} {c.get('api_latency_s',0):11.4f}")
        # Optional JSON dump for machine reading
        # print("\nRaw details JSON:\n", json.dumps({"events": self.events, "details": self.details}, indent=2))

prof = Profiler()

# ---------------------------
# Tokenization helpers
# ---------------------------
def _load_tokenizer():
    with prof.span("tokenizer_load"):
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            return enc.encode, enc.decode, None  # exact tokenizer
        except Exception:
            return None, None, 4.0  # char-based fallback

def tokenize(text: str, encode_fn: Callable, tpc_estimate: float):
    with prof.span("tokenize"):
        if encode_fn is not None:
            return encode_fn(text)  # token IDs
        else:
            return list(text)  # pseudo tokens

def detokenize(tokens: List[int], decode_fn: Callable, tpc_estimate: float):
    if decode_fn is not None:
        return decode_fn(tokens)
    else:
        return "".join(tokens)

def chunk_tokens(tokens: List[int], max_tokens: int, overlap: int) -> List[List[int]]:
    with prof.span("chunk_tokens"):
        chunks = []
        i = 0
        n = len(tokens)
        while i < n:
            j = min(i + max_tokens, n)
            chunks.append(tokens[i:j])
            if j >= n:
                break
            i = j - overlap if j - overlap > i else j
        return chunks

# ---------------------------
# Model call helper
# ---------------------------
def call_model(client, model_name: str, user_prompt: str, max_tokens: int = 1024) -> str:
    t0 = time.perf_counter()
    resp = client.messages.create(
        model=model_name,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": user_prompt}],
    )
    api_latency = time.perf_counter() - t0
    # Normalize to string depending on SDK shape
    content = getattr(resp, "content", resp)
    if isinstance(content, list):
        out = "\n".join(str(x) for x in content)
    else:
        out = str(content)
    return out, api_latency

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

    with prof.span("read_file"):
        with open(path, "r", encoding="utf-8") as f:
            big_text = f.read()
    prof.snapshot("after_read")

    encode_fn, decode_fn, tpc_est = _load_tokenizer()

    tokens = tokenize(big_text, encode_fn, tpc_est)
    prof.snapshot("after_tokenize")

    # Guardrails: ensure budgets make sense
    with prof.span("guardrails"):
        max_tokens = max(1000, min(per_chunk_token_budget, 9900))
        overlap = max(0, min(per_chunk_overlap_tokens, max_tokens // 3))

    token_chunks = chunk_tokens(tokens, max_tokens, overlap)

    with prof.span("detokenize_chunks"):
        text_chunks = [detokenize(tc, decode_fn, tpc_est) for tc in token_chunks]
    prof.snapshot("after_chunking")

    # Per-chunk prompt template
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

    # Per-chunk map calls
    chunk_summaries = []
    for idx, chunk_text in enumerate(text_chunks, 1):
        # Prompt build timing (can matter for big chunks)
        t_build0 = time.perf_counter()
        prompt = per_chunk_prompt_tpl.format(chunk=chunk_text)
        prompt_build_s = time.perf_counter() - t_build0

        with prof.span(f"api_chunk_{idx}"):
            out, api_latency = call_model(
                client, model_name, prompt, max_tokens=per_chunk_max_output_tokens
            )

        # Heuristic token counts (exact only if tokenizer exists)
        tok_in = len(tokenize(chunk_text, encode_fn, tpc_est))
        tok_out = len(out.splitlines())

        prof.add_chunk_stat(
            idx=idx,
            tokens_in=tok_in,
            tokens_out=tok_out,
            prompt_build_s=prompt_build_s,
            api_latency_s=api_latency,
        )

        chunk_summaries.append(f"Chunk {idx} summary:\n{out}")

    # Aggregate + synthesize final
    with prof.span("aggregate_chunks"):
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

    with prof.span("api_final_reduce"):
        final_summary, final_latency = call_model(
            client, model_name, final_prompt, max_tokens=final_max_output_tokens
        )
    prof.details["final_api_latency_s"] = final_latency
    prof.snapshot("after_final")

    return final_summary

# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    load_dotenv()
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    if client is None:
        raise RuntimeError("Please instantiate your model client and assign it to `client`.")

    wall_start = time.perf_counter()
    final = summarize_large_file(
        path="./sorting_algorithms.txt",
        client=client,
        model_name="claude-opus-4-1-20250805",
        per_chunk_token_budget=9000,
        per_chunk_overlap_tokens=200,
        per_chunk_max_output_tokens=700,
        final_max_output_tokens=700,
    )
    wall_end = time.perf_counter()
    print(f"\n=== FINAL SUMMARY ===\n{final}\n")
    print(f"Elapsed time (outer): {wall_end - wall_start:.4f} seconds")
    prof.report()
