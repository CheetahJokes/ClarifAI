import os
import re
import json
from typing import Iterable, Set, List
from collections import Counter

# ---- Heuristic cleaning ------------------------------------------------------

_STOP_CHUNKS = {
    # ultra-generic phrases to drop outright
    "first step","second step","last step","base case","worst case","best case",
    "worst-case","best-case","worst-case scenario","best-case scenario",
    "average-case","average-case behavior","time complexity","runtime complexity",
    "recurrence relation","recursion tree","first element","last element",
    "current element","next value","next smallest value","smallest value",
    "smallest element","smallest number","correct order","correct position",
    "sorted order","sorted output","sorted result","correct sorted order",
    "relative order","relative ordering","correct relative position",
    "output sequence","output vector","offset vector","additional offset vector",
    "separate vector","separate container","separate copy","separate output vector",
    "original container","original data","original data vector","original data container",
    "original vector","original order","entire array","entire container",
    "entire output vector","entire quicksort algorithm","entire mergesort process",
    "mergesort process","merge process","sorting process","advanced sort",
    "advanced sorting","advanced comparison","advanced comparison sort",
    "elementary sort","elementary sorting","elementary sorting algorithm",
    "stable implementation","current check","first look","first step","first recursion tree",
    "different sorting","different best-case","different complexity",
    "total number","total amount","total work","time operation","memory sorting",
    "following example","following process","following code","following vector",
    "following comparator","following container","following unsorted array",
    "grade example","possible grade","correct number","correct branch",
    "place relative","middle half","second half","first half","left end","right index",
    "index idx","idx vector","size_t idx","size_t min_index","size_t current",
    "size_t num_keys","const std","void std","void merge","void mergesort",
    "void quicksort","void heapsort","return value","return true","return left",
    "comparator comp","comparison operator","index sorting","Index Sorting",
    "Index Sorting Index sorting","Array Representation","class IndexSortComparator",
    "data container","data vector","copy vec","initial array","initial sort",
    "initial pivot","offset vector","sorted idx vector","sorting idx","sort function",
    "double value","char grade",
}

# single tokens to drop (codey/noisy)
_BAD_TOKS = {
    "void","size_t","idx","std","comp","vector","container","value","data","current",
    "return","true","false","class","const","char","double","count","num","idx",
}

# keeplist seeds to bias local filter to keep real concepts before LLM step
_KEEP_SEEDS = {
    "selection sort","insertion sort","bubble sort","quicksort","merge sort","mergesort",
    "heapsort","heapify","radix sort","counting sort","stable sort","adaptive sort",
    "linear-time sorting","comparison sort","pivot selection","partition scheme",
    "worst-case time complexity","average-case time complexity","auxiliary space",
    "in-place","stable","unstable","divide and conquer","bottom-up mergesort",
    "bottom-up heapify","index sort","radix","bucket","recursion tree",
}

_ws_re = re.compile(r"\s+")
_dup_word_run = re.compile(r"\b(\w+)(\s+\1\b)+" , flags=re.IGNORECASE)

def _squash_duplicate_words(s: str) -> str:
    # turns "Selection Sort Selection sort" -> "Selection Sort"
    return _dup_word_run.sub(lambda m: m.group(1), s)

def _norm_phrase(p: str) -> str:
    # strip punctuation except internal hyphens/spaces
    p = p.strip(" \t\r\n,.!?;:()[]{}<>/\\|\"'`")
    p = _squash_duplicate_words(p)
    # collapse whitespace
    p = _ws_re.sub(" ", p)
    # unify case but keep internal caps restoration later
    return p.strip()

def _looks_codey(p: str) -> bool:
    if any(tok in p for tok in ["::","()", "<", ">", "[]"]):
        return True
    if re.search(r"\b(size_t|std::|void|class)\b", p):
        return True
    return False

def _too_generic(p: str) -> bool:
    L = p.lower()
    if L in _STOP_CHUNKS:
        return True
    # one or two very common words often not a concept
    if len(L.split()) <= 2 and L.split()[0] in {"first","second","last","current","original","following","correct"}:
        return True
    if any(tok in _BAD_TOKS for tok in L.split()):
        return True
    # extremely short or just numbers
    if len(L) < 3 or L.isdigit():
        return True
    return False

def _canonical_case(s: str) -> str:
    # Prefer Title Case for multiword algorithm names, but keep known lowercase tokens
    keep_lower = {"and","of","for","in","on","with","to"}
    words = s.lower().split()
    titled = [w if w in keep_lower else w.capitalize() for w in words]
    # preserve common canonical spellings
    joined = " ".join(titled)
    joined = joined.replace("Mergesort", "Mergesort")
    joined = joined.replace("Merge Sort", "Mergesort")
    joined = joined.replace("Quicksort", "Quicksort")
    joined = joined.replace("Radix Sort", "Radix Sort")
    joined = joined.replace("Heapify", "Heapify")
    return joined

def _heuristic_filter(keywords: Iterable[str], max_out: int = 150) -> List[str]:
    # normalize, drop obvious junk, dedup by lowercase
    cleaned = []
    seen = set()
    for raw in keywords:
        if not raw:
            continue
        p = _norm_phrase(str(raw))
        if not p or _looks_codey(p) or _too_generic(p):
            continue
        # unify whitespace & case for dedup
        key = p.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(p)

    # prefer items containing strong seeds; also boost multiword phrases
    def score(p: str) -> int:
        L = p.lower()
        s = 0
        for k in _KEEP_SEEDS:
            if k in L:
                s += 5
        if "sort" in L: s += 2
        if "complexity" in L or "space" in L: s += 2
        if "partition" in L or "pivot" in L: s += 2
        if len(L.split()) >= 2: s += 1
        return s

    cleaned.sort(key=lambda x: (score(x), len(x)), reverse=True)
    # apply canonical casing
    canon = [_canonical_case(x) for x in cleaned]
    # limit size before sending to LLM
    return canon[:max_out]

# ---- Anthropic call ----------------------------------------------------------

def _call_anthropic_concepts(candidates: List[str]) -> List[str]:
    """
    Sends candidates to Anthropic and asks for a JSON array of canonical concept phrases.
    Falls back to the input list on any error.
    """
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        system_prompt = (
            "You are a strict normalizer of technical concept phrases from a CS text about sorting. "
            "Your job: deduplicate, merge variants, and return only meaningful, canonical concepts "
            "(algorithm names, core operations, properties like stability/in-place, key procedures like "
            "partition/pivot selection, complexity notions). "
            "Drop non-concepts (code artifacts, placeholders like 'first step', 'return true', container/vector/idx noise). "
            "Prefer standard canonical names (e.g., 'Mergesort', 'Quicksort', 'Radix Sort', 'Heapify', "
            "'Divide and Conquer', 'Partition Scheme', 'Auxiliary Space', 'Stable Sort'). "
            "Return a compact JSON array of strings (no objects), each a single canonical concept phrase. "
            "Do not include explanations or comments—JSON array only."
        )

        user_prompt = (
            "Here are candidate phrases extracted from the document. "
            "Clean them per the rules and return a JSON array of canonical concept phrases.\n\n"
            + json.dumps(candidates, ensure_ascii=False, indent=2)
        )

        resp = client.messages.create(
            model="claude-3-5-sonnet-latest",
            max_tokens=2048,
            temperature=0,
            system=system_prompt,
            messages=[{"role":"user","content":user_prompt}],
            response_format={"type":"json"}  # ask for pure JSON
        )

        content = resp.content[0].text if hasattr(resp, "content") else resp.output_text  # handle SDK variations
        parsed = json.loads(content)
        # post-validate: list[str], canonical case normalization pass
        out = []
        seen = set()
        for item in parsed:
            if not isinstance(item, str):
                continue
            c = _canonical_case(_norm_phrase(item))
            low = c.lower()
            if low and low not in seen:
                seen.add(low)
                out.append(c)
        return out

    except Exception as e:
        # Fallback: return candidates (already reasonably cleaned)
        return candidates

# ---- Public function ---------------------------------------------------------

def clean_keywords(keywords: Iterable[str]) -> List[str]:
    """
    1) Heuristically normalize & deduplicate raw keyphrases.
    2) Ask Anthropic to return a canonical list of concept phrases (JSON array).
    3) Final small pass to ensure uniqueness and tidy casing.
    """
    candidates = _heuristic_filter(keywords, max_out=150)
    llm_out = _call_anthropic_concepts(candidates)

    # tiny final polish & cap length
    final = []
    seen = set()
    for p in llm_out:
        c = _canonical_case(_norm_phrase(p))
        low = c.lower()
        if not c or low in seen: 
            continue
        # filter any lingering non-concepts
        if _looks_codey(c) or _too_generic(c):
            continue
        seen.add(low)
        final.append(c)

    # optional: prioritize algorithms & core ops at the top
    def final_score(s: str) -> int:
        L = s.lower()
        sc = 0
        if "sort" in L: sc += 5
        if any(k in L for k in ["quicksort","mergesort","heapsort","radix","insertion","selection","bubble"]): sc += 5
        if any(k in L for k in ["partition","pivot","heapify","divide and conquer","stable","in-place"]): sc += 3
        if any(k in L for k in ["time complexity","auxiliary space","stability"]): sc += 2
        if len(L.split()) <= 4: sc += 1  # concise concepts
        return sc

    final.sort(key=lambda s: (final_score(s), -len(s)), reverse=True)
    return final[:80]  # keep it tight

# ---- Example integration in main --------------------------------------------

def main():
    load_dotenv()
    setup_environment()
    with open("articles/sorting_algorithms.txt", 'r') as file:
        text = file.read()

    phrases = extract_key_phrases(text, plot=True)
    concepts = clean_keywords(phrases)
    print("\n=== Canonical Concepts ===")
    for c in concepts:
        print("-", c)
