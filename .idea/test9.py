import math
import string
import matplotlib.pyplot as plt
import networkx as nx
import nltk
from collections import Counter, defaultdict
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
from dotenv import load_dotenv
import json
import re
import itertools
import ray
from typing import List, Dict, Tuple, Callable, Optional
Adj = Dict[str, Dict[str, int]]


PUNCT = set(string.punctuation)
STOP = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

# ---- your helpers assumed present ----
# - filter_for_tags(tagged)
# - normalize(tagged)

def setup_environment():
    """Download required resources."""
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger_eng')
    nltk.download('wordnet')
    print('Completed resource downloads.')

def filter_for_tags(tagged, tags=('NN', 'NNS', 'NNP', 'JJ')):
    """Keep only nouns/adjectives/etc."""
    return [item for item in tagged if item[1] in tags]

def normalize(tagged):
    """Remove trailing periods etc."""
    return [(item[0].replace('.', ''), item[1]) for item in tagged]

def unique_everseen(iterable, key=None):
    seen = set()
    result = []
    for element in iterable:
        k = element if key is None else key(element)
        if k not in seen:
            seen.add(k)
            result.append(element)
    return result

def _clean_token(w):
    w = w.lower()
    # strip surrounding punctuation
    w = w.strip(string.punctuation)
    # block pure digits or mixed digit-like junk
    if any(ch.isdigit() for ch in w):
        return ""
    return w

def _candidate_tokens(text, *, use_lemmatize=True, remove_stopwords=True, min_len=2):
    """
    Return:
      textlist: original tokens (surface forms) for phrase merging
      cand_tokens: cleaned candidate tokens (filtered by POS, stopwords, etc.)
    """
    # 1) tokenize + tag
    word_tokens = nltk.word_tokenize(text)
    tagged_all = nltk.pos_tag(word_tokens)

    # Keep original sequence for phrase merging (surface forms)
    textlist = [w for (w, _) in tagged_all]

    # 2) POS filter (your rule)
    tagged = filter_for_tags(tagged_all)

    # 3) Normalize (your rule; e.g., strip periods)
    tagged = normalize(tagged)

    # 4) Lowercase + stopword/punct/length filter + optional lemmatization
    cand_tokens = []
    for w, pos in tagged:
        w = _clean_token(w)
        if not w or len(w) < min_len:
            continue
        if remove_stopwords and w in STOP:
            continue
        if all(ch in PUNCT for ch in w):
            continue
        if use_lemmatize:
            # simple noun/adjective lemmatization heuristic
            tag0 = pos[0].lower() if pos else 'n'
            wn_pos = {'j':'a','n':'n','v':'v','r':'r'}.get(tag0, 'n')
            w = LEMMATIZER.lemmatize(w, pos=wn_pos)
        # after lemmatize, re-check length/stopwords just in case
        if len(w) < min_len:
            continue
        if remove_stopwords and w in STOP:
            continue
        cand_tokens.append(w)

    return textlist, cand_tokens

def _build_cooccurrence_graph(cand_tokens, window=4):
    G = nx.Graph()
    for i, u in enumerate(cand_tokens):
        G.add_node(u)
        for j in range(i + 1, min(i + window, len(cand_tokens))):
            v = cand_tokens[j]
            if u == v:
                continue
            if G.has_edge(u, v):
                G[u][v]["weight"] += 1.0
            else:
                G.add_edge(u, v, weight=1.0)
    return G

def _bigram_counts(cand_tokens):
    return Counter(zip(cand_tokens, cand_tokens[1:]))

def extract_key_phrases(
    text,
    plot=True,
    window=4,
    keep_fraction=0.33,
    topk_edges_per_node=3,
    layout_seed=7
):
    """
    TextRank-style extraction with:
      - POS+normalize -> stopword & punctuation filtering -> optional lemmatization
      - Sliding-window co-occurrence graph over the cleaned candidate stream
      - TF-personalized PageRank
      - Phrase merge on surface tokens, scoring by sum(PR) + small PMI boost
      - Plot: top-k strongest edges per kept node
    """
    # Candidates (now stopword-filtered & cleaned)
    textlist, cand_tokens = _candidate_tokens(text)
    if not cand_tokens:
        return set()

    tf = Counter(cand_tokens)
    total = sum(tf.values())
    personalization = {w: tf[w] / total for w in tf}

    G = _build_cooccurrence_graph(cand_tokens, window=window)
    if G.number_of_nodes() == 0:
        return set()

    pr = nx.pagerank(G, weight="weight", personalization=personalization)

    # keep top fraction by PR
    sorted_words = sorted(pr, key=pr.get, reverse=True)
    keep_n = max(1, int(len(sorted_words) * keep_fraction))
    key_words = set(sorted_words[:keep_n])

    # PMI helper (with light smoothing)
    unig = tf
    big = _bigram_counts(cand_tokens)
    N = total
    def pmi(u, v):
        pu = (unig[u] + 1) / (N + len(unig))
        pv = (unig[v] + 1) / (N + len(unig))
        cuv = big.get((u, v), 0) + big.get((v, u), 0)
        pv_uv = (cuv + 1) / (N + len(big))
        return math.log(pv_uv / (pu * pv))

    # merge adjacent selected words in surface sequence
    phrases = []
    i = 0
    while i < len(textlist):
        w = textlist[i].lower().strip(string.punctuation)
        if w in key_words:
            run_surface = [textlist[i]]   # keep original casing for display
            run_clean = [w]
            i += 1
            while i < len(textlist):
                w2c = textlist[i].lower().strip(string.punctuation)
                if w2c in key_words:
                    run_surface.append(textlist[i])
                    run_clean.append(w2c)
                    i += 1
                else:
                    break
            score = sum(pr.get(t, 0.0) for t in run_clean)
            for a, b in zip(run_clean, run_clean[1:]):
                score += 0.15 * max(0.0, pmi(a, b))
            phrases.append((" ".join(run_surface), score))
        else:
            i += 1

    # dedupe phrases, keep best score
    best = defaultdict(float)
    for p, s in phrases:
        if s > best[p]:
            best[p] = s
    ranked_phrases = sorted(best.items(), key=lambda x: x[1], reverse=True)

    # Plot pruned subgraph (top-k edges per node)
    if plot:
        H = G.subgraph(key_words).copy()
        edges_to_keep = set()
        for u in H.nodes():
            nbrs = sorted(
                ((H[u][v].get("weight", 1.0), v) for v in H[u]),
                key=lambda x: (-x[0], str(x[1]))
            )[:topk_edges_per_node]
            for _, v in nbrs:
                a, b = (u, v) if u <= v else (v, u)
                edges_to_keep.add((a, b))
        H2 = nx.Graph()
        H2.add_nodes_from(H.nodes())
        H2.add_edges_from(
            (u, v, H.get_edge_data(u, v) or {"weight": 1.0})
            for (u, v) in edges_to_keep if H.has_edge(u, v)
        )
        H2.remove_nodes_from(list(nx.isolates(H2)))

        if H2.number_of_nodes() > 0:
            node_sizes = [max(300.0, 5000.0 * pr.get(n, 0.0)) for n in H2.nodes()]
            pos = nx.spring_layout(H2, seed=layout_seed, weight="weight")
            plt.figure(figsize=(9, 7))
            nx.draw_networkx_nodes(H2, pos, node_size=node_sizes, alpha=0.9)
            nx.draw_networkx_edges(H2, pos, alpha=0.6)
            nx.draw_networkx_labels(H2, pos, font_size=10)
            plt.title("Keyphrase Graph (stopword-filtered; top-k edges/node)")
            plt.axis("off"); plt.tight_layout(); plt.show()
        else:
            print("[extract_key_phrases] Nothing to plot after pruning.")

    # Return a set of top phrases (you can change the cut as needed)
    top_phrases = {p for p, _ in ranked_phrases[:max(5, int(len(ranked_phrases) * 0.5))]}
    return G, top_phrases

def deduplicate_keywords_with_claude(keywords):
    """
    Given a list of keywords, ask Claude to:
    - Strictly filter out irrelevant, overly generic, or meaningless keywords.
    - Deduplicate similar terms into a canonical form.
    - Return a clean JSON mapping.
    """
    prompt = f"""{HUMAN_PROMPT}
    You are given a list of keywords. Your job is to create a **strict, high-quality keyword list**.

    Rules:
    1. **Filter aggressively**: Remove irrelevant, overly generic, or meaningless keywords.  
    - Examples of keywords to remove: "index", "sorting", "next value", "a", "the", "function", etc.  
    - Keep only strong, meaningful technical terms, algorithms, and unique concepts.
    - It is acceptable for some input keywords to be completely dropped.

    2. **Group and Deduplicate**:  
    - Group similar keywords (case differences, typos, duplicates, word-order changes) under one canonical keyword.
    - Canonical keyword should be lowercase and normalized.
    - If two keywords mean the same concept (e.g., "void quicksort" and "Quicksort"), merge them.

    3. **Output JSON object ONLY**:
    - Key: canonical keyword
    - Value: list of all original variations (as they appeared in input)
    - Omit all dropped keywords entirely; they should not appear in any list.

    4. **No duplicates across lists**:  
    - Each input keyword appears in at most one list or is dropped entirely.
    - The JSON should only include strong, meaningful concepts.

    Here are the keywords:
    {keywords}

    {AI_PROMPT}"""
    
    load_dotenv()
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # Send prompt to Claude
    response = client.messages.create(
        model="claude-opus-4-1-20250805",
        max_tokens=800,
        messages=[{"role": "user", "content": prompt}]
    )

    # Claude's reply is JSON
    return response.content[0].text    

def json_to_dict(json_str):
    """
    Convert a JSON string (Claude's response) into a Python dictionary.
    Cleans up common formatting mistakes before parsing.
    """
    try:
        # Claude sometimes returns extra code fences or explanations; strip them
        cleaned = json_str.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            # Remove language specifiers like ```json
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()
        
        # Parse JSON
        data = json.loads(cleaned)
        
        # Ensure it's a dictionary
        if not isinstance(data, dict):
            raise ValueError("Parsed JSON is not a dictionary.")
        
        return data
    
    except json.JSONDecodeError as e:
        print("❌ JSON parsing error:", e)
        return {}
    except Exception as e:
        print("❌ Unexpected error:", e)
        return {}

def draw_graph(G, title="Graph"):
    """Visualize a graph with edge weights."""
    plt.figure(figsize=(10, 7))
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=800)
    nx.draw_networkx_edges(G, pos, edge_color='gray', width=2)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels={k: f"{v:.2f}" for k, v in edge_labels.items()},
        font_color='red'
    )
    plt.title(title)
    plt.axis('off')
    plt.show()

def consolidate_graph(G, clusters):
    """
    Consolidate nodes in a graph based on clusters.
    - Combine all nodes in each cluster's value list into the canonical key
    - For neighbors shared by multiple nodes in a cluster, use max weight
    - Remove nodes not present as canonical keys in clusters
    - When removing a node, connect its neighbors to each other with max weight
    """
    new_G = nx.Graph()
    
    # Build mapping from original node -> canonical node
    node_map = {}
    
    # First, map all canonical nodes to themselves
    for canonical in clusters.keys():
        node_map[canonical] = canonical
    
    # Then, map all member nodes to their canonical nodes
    for canonical, members in clusters.items():
        for member in members:
            node_map[member] = canonical
    
    print(f"\n--- DEBUG: Cluster analysis ---")
    print(f"Canonical nodes: {list(clusters.keys())}")
    print(f"Total mapped nodes: {len(node_map)}")
    
    # Find nodes that will be removed (not in node_map at all)
    all_nodes = set(G.nodes())
    mapped_nodes = set(node_map.keys())
    nodes_to_remove = all_nodes - mapped_nodes
    
    print(f"\n--- DEBUG: Graph size analysis ---")
    print(f"Original graph: {len(all_nodes)} nodes, {G.number_of_edges()} edges")
    print(f"Mapped nodes: {len(mapped_nodes)} nodes")
    print(f"Nodes to remove: {len(nodes_to_remove)} nodes")
    
    # Add all canonical nodes first
    for canonical in clusters.keys():
        new_G.add_node(canonical)
    
    # Let's analyze what happens with a few high-degree nodes
    print(f"\n--- DEBUG: Analyzing high-degree nodes ---")
    node_degrees = [(node, G.degree(node)) for node in G.nodes()]
    node_degrees.sort(key=lambda x: x[1], reverse=True)
    
    for node, degree in node_degrees[:10]:  # Top 10 highest degree nodes
        status = "MAPPED" if node in node_map else "REMOVED"
        if node in node_map:
            canonical = node_map[node]
            print(f"  {node} (degree={degree}) -> {canonical} [{status}]")
        else:
            # Count how many neighbors will survive
            surviving_neighbors = [n for n in G.neighbors(node) if n in node_map]
            canonical_neighbors = list(set(node_map[n] for n in surviving_neighbors))
            print(f"  {node} (degree={degree}) -> [{status}] -> {len(canonical_neighbors)} canonical neighbors: {canonical_neighbors}")
    
    print(f"\n--- DEBUG: Processing removed nodes ---")
    total_new_edges = 0
    
    # Process removed nodes: connect their neighbors to each other
    for removed_node in nodes_to_remove:
        if removed_node in G:
            neighbors = []
            neighbor_weights = {}
            
            # Get all neighbors that will survive (are mapped to canonical nodes)
            for neighbor in G.neighbors(removed_node):
                if neighbor in node_map:
                    canonical_neighbor = node_map[neighbor]
                    weight = G[removed_node][neighbor].get('weight', 1.0)
                    if canonical_neighbor not in neighbor_weights:
                        neighbors.append(canonical_neighbor)
                        neighbor_weights[canonical_neighbor] = weight
                    else:
                        neighbor_weights[canonical_neighbor] = max(neighbor_weights[canonical_neighbor], weight)
            
            # Remove duplicates while preserving order
            neighbors = list(dict.fromkeys(neighbors))
            
            if len(neighbors) >= 2:
                print(f"Removing '{removed_node}' (degree={G.degree(removed_node)}) -> connecting {len(neighbors)} canonical neighbors")
                
                # Connect all pairs of surviving neighbors
                for i in range(len(neighbors)):
                    for j in range(i + 1, len(neighbors)):
                        u, v = neighbors[i], neighbors[j]
                        if u != v:  # Skip if same canonical node
                            max_weight = max(neighbor_weights[u], neighbor_weights[v])
                            
                            # Add or update edge with maximum weight
                            if new_G.has_edge(u, v):
                                current_weight = new_G[u][v].get('weight', 0)
                                new_weight = max(current_weight, max_weight)
                                new_G[u][v]['weight'] = new_weight
                            else:
                                new_G.add_edge(u, v, weight=max_weight)
                                total_new_edges += 1
    
    print(f"--- Created {total_new_edges} new edges from removed nodes ---")
    
    # Process all edges from the original graph that involve mapped nodes
    print(f"\n--- Processing original edges between mapped nodes ---")
    edges_processed = 0
    for u, v, data in G.edges(data=True):
        weight = data.get('weight', 1.0)
        
        # Check if both nodes are mapped (not removed)
        if u in node_map and v in node_map:
            canonical_u = node_map[u]
            canonical_v = node_map[v]
            
            # Skip self-loops
            if canonical_u == canonical_v:
                continue
            
            # Add or update edge with maximum weight
            if new_G.has_edge(canonical_u, canonical_v):
                current_weight = new_G[canonical_u][canonical_v].get('weight', 0)
                new_weight = max(current_weight, weight)
                new_G[canonical_u][canonical_v]['weight'] = new_weight
            else:
                new_G.add_edge(canonical_u, canonical_v, weight=weight)
            
            edges_processed += 1
    
    print(f"--- Processed {edges_processed} original edges between mapped nodes ---")
    
    print(f"\n--- Final Graph Stats ---")
    print(f"Nodes: {len(new_G.nodes())} (was {len(all_nodes)})")
    print(f"Edges: {new_G.number_of_edges()}")
    
    print(f"\n--- Sample of Final Edges ---")
    edge_list = list(new_G.edges(data=True))
    for u, v, data in edge_list[:20]:  # Show first 20 edges
        print(f"{u} -- {v}, weight={data['weight']}")
    if len(edge_list) > 20:
        print(f"... and {len(edge_list) - 20} more edges")
    
    # Display the graph
    if new_G.number_of_nodes() > 0:
        draw_graph(new_G, title="Consolidated Graph")
    else:
        print("No nodes to display in consolidated graph")
    
    return new_G

def createEdges(
    text: str,
    nodes: List[str],
    *,
    min_support: int = 1,
    make_aliases: bool = True,
    sentence_splitter: Callable[[str], List[str]] = None,
) -> Dict[str, Dict[str, int]]:
    """
    Build a directed adjacency list of edges between chosen nodes based on
    sentence-level co-occurrence. Each co-occurrence increments both (u->v) and (v->u) by 1.

    Args:
        text: The raw passage.
        nodes: List of canonical node strings (e.g., ["quicksort", "mergesort", ...]).
        min_support: Drop edges with total count < min_support.
        make_aliases: If True, generate simple aliases (space<->hyphen, plural).
        sentence_splitter: Optional custom splitter; defaults to a regex splitter.

    Returns:
        adjacency: dict[node][neighbor] = count
    """

    # --- 1) Canonicalize nodes and compile matchers ---
    canonical = {n: n.strip() for n in nodes}
    # Build alias map: node -> list of regex patterns that match the node in text
    alias_patterns: Dict[str, List[re.Pattern]] = {}

    def _aliases_for(base: str) -> List[str]:
        base = base.lower().strip()
        forms = {base}
        if make_aliases:
            forms |= {
                base.replace("-", " "),
                base.replace(" ", "-")
            }
            if not base.endswith("s") and " " not in base:
                # very light pluralization for single tokens
                forms.add(base + "s")
        return sorted(forms)

    for n in nodes:
        pats = []
        for form in _aliases_for(n):
            # Word-boundary exact-ish match; escape to avoid regex surprises
            pats.append(re.compile(rf"\b{re.escape(form)}\b", re.IGNORECASE))
        alias_patterns[n] = pats

    # --- 2) Sentence split ---
    if sentence_splitter is None:
        # Simple regex that splits on . ! ? while keeping abbreviations mostly intact
        sentence_splitter = lambda t: [s for s in re.split(r'(?<=[.!?])\s+', t) if s.strip()]

    sentences = sentence_splitter(text)

    # --- 3) Scan sentences and tally co-occurrences ---
    # counts[(u,v)] accumulates number of sentences where both appeared
    counts: Dict[Tuple[str, str], int] = {}

    def nodes_in_sentence(sent: str) -> List[str]:
        present = []
        lower = sent.lower()
        for node, pats in alias_patterns.items():
            if any(p.search(lower) for p in pats):
                present.append(canonical[node])
        # de-dup while preserving order
        seen = set()
        uniq = []
        for x in present:
            if x not in seen:
                seen.add(x)
                uniq.append(x)
        return uniq

    for sent in sentences:
        present = nodes_in_sentence(sent)
        if len(present) < 2:
            continue
        # All unordered pairs get +1, then we emit both directions
        for u, v in itertools.combinations(present, 2):
            counts[(u, v)] = counts.get((u, v), 0) + 1
            counts[(v, u)] = counts.get((v, u), 0) + 1

    # --- 4) Build adjacency dict, apply min_support prune ---
    adjacency: Dict[str, Dict[str, int]] = {n: {} for n in nodes}
    for (u, v), c in counts.items():
        if u == v:
            continue
        if c >= min_support:
            adjacency.setdefault(u, {})
            adjacency[u][v] = adjacency[u].get(v, 0) + c

    # Remove isolated nodes with empty dicts? (Keep them by default so caller sees all nodes)
    return adjacency


def plot_graph(
    adj: Adj,
    *,
    top_k_out: Optional[int] = None,   # keep only top-k neighbors per node
    min_weight: int = 1,               # drop edges with weight < min_weight
    layout: str = "spring",            # "spring" | "kamada_kawai" | "circular" | "spectral"
    figsize=(14, 10),
    title: Optional[str] = None,
    save_path: Optional[str] = None,   # e.g. "graph.png"
    show_weights: bool = False,        # draw numeric weights on edges
) -> nx.Graph:
    """
    Build and draw an undirected weighted graph from an adjacency dict.
    Directed edges are flattened into a single undirected edge with summed weights.

    Returns the NetworkX Graph in case you want to inspect it further.
    """
    # 1) Build undirected edge weights
    edge_weights: Dict[frozenset, int] = {}

    for u, nbrs in adj.items():
        items = list(nbrs.items())
        if top_k_out is not None:
            items = sorted(items, key=lambda kv: kv[1], reverse=True)[:top_k_out]
        for v, w in items:
            if w is None or w < min_weight or u == v:
                continue
            key = frozenset([u, v])
            edge_weights[key] = edge_weights.get(key, 0) + w

    # 2) Build graph
    G = nx.Graph()
    for key, w in edge_weights.items():
        u, v = tuple(key)
        G.add_edge(u, v, weight=w)

    # 2b) Drop isolates (nodes with no edges)
    isolates = list(nx.isolates(G))
    G.remove_nodes_from(isolates)

    # 3) Layout
    if layout == "spring":
        pos = nx.spring_layout(G, k=0.7, iterations=200, seed=42)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "spectral":
        pos = nx.spectral_layout(G)
    else:
        raise ValueError(f"Unknown layout: {layout}")

    # 4) Edge widths scaled by weight
    weights = [G[u][v]["weight"] for u, v in G.edges()]
    if weights:
        w_min, w_max = min(weights), max(weights)
        if math.isclose(w_min, w_max):
            widths = [3.0 for _ in weights]
        else:
            widths = [1.5 + 4.5 * (w - w_min) / (w_max - w_min) for w in weights]
    else:
        widths = []

    # 5) Draw
    plt.figure(figsize=figsize)
    nx.draw_networkx_nodes(G, pos, node_size=900, linewidths=0.8)
    nx.draw_networkx_labels(G, pos, font_size=8)
    nx.draw_networkx_edges(
        G, pos,
        width=widths,
        edge_color="gray"
    )

    if show_weights and len(G.edges()) > 0:
        edge_labels = {(u, v): G[u][v]["weight"] for u, v in G.edges()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)

    if title:
        plt.title(title)

    plt.axis("off")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.show()

    return G

def chunk_text_sliding_by_sentences(
    text: str,
    target_chars: int = 5000,
    hard_max_chars: int = 8000,
    overlap_sentences: int = 2,
    max_chunks: int = 5,
) -> List[str]:
    """
    Build up to `max_chunks` overlapping chunks. Each chunk is ~target_chars
    (capped by hard_max_chars). Next chunk starts `overlap_sentences` before
    the previous chunk ends.
    """
    sents = [s for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]
    chunks = []
    i = 0
    while i < len(sents) and len(chunks) < max_chunks:
        buf, cur_len, j = [], 0, i
        while j < len(sents):
            s_len = len(sents[j]) + 1
            # stop at target or hard cap (but ensure we don't emit an empty chunk)
            if (cur_len + s_len > target_chars and buf) or (cur_len + s_len > hard_max_chars):
                break
            buf.append(sents[j])
            cur_len += s_len
            j += 1
        if buf:
            chunks.append(" ".join(buf))
        if j >= len(sents) or len(chunks) >= max_chunks:
            break
        # slide with overlap
        i = max(i + 1, j - overlap_sentences)
    return chunks

@ray.remote
def process_chunk_remote(
    chunk_text: str,
    *,
    window: int = 4,
    keep_fraction: float = 0.33,
    topk_edges_per_node: int = 3,
    layout_seed: int = 7,
    dedup_per_chunk: bool = True  # <-- default True here
) -> Dict:
    try:
        setup_environment()
    except Exception:
        pass

    try:
        _G, phrases = extract_key_phrases(
            chunk_text,
            plot=False,
            window=window,
            keep_fraction=keep_fraction,
            topk_edges_per_node=topk_edges_per_node,
            layout_seed=layout_seed
        )
    except Exception as e:
        return {"phrases": [], "keyword_map": {}, "err": f"extract_failed: {e}"}

    phrases = sorted(set(phrases))
    if not dedup_per_chunk:
        # not used in this config, but keep for completeness
        return {"phrases": phrases, "keyword_map": {}, "err": None}

    try:
        dedup_json = deduplicate_keywords_with_claude(phrases)
        kw_map = json_to_dict(dedup_json)
        return {"phrases": [], "keyword_map": kw_map, "err": None}
    except Exception as e:
        return {"phrases": phrases, "keyword_map": {}, "err": f"chunk_dedup_failed: {e}"}

def _merge_keyword_maps(maps: List[Dict[str, List[str]]]) -> Dict[str, List[str]]:
    merged: Dict[str, set] = {}
    for m in maps:
        for canon, variants in m.items():
            merged.setdefault(canon, set()).update(variants)
    return {k: sorted(v) for k, v in merged.items()}

def run_parallel_pipeline(
    text: str,
    *,
    target_chars: int = 5000,
    hard_max_chars: int = 8000,
    overlap_sentences: int = 2,
    max_chunks: int = 5,               # <-- cap at 5 to respect Claude limit
    num_cpus: Optional[int] = None,
    window: int = 4,
    keep_fraction: float = 0.33,
    final_global_dedup: bool = False,  # <-- keep OFF to stay within 5 calls/min
):
    """
    1) make ≤ max_chunks overlapping chunks
    2) parallel Ray map: per-chunk TextRank + Claude dedup
    3) merge per-chunk canonical maps
    4) (optional) one final global dedup call
    5) build co-occurrence graph on full text
    """
    # --- chunk (≤ max_chunks)
    chunks = chunk_text_sliding_by_sentences(
        text,
        target_chars=target_chars,
        hard_max_chars=hard_max_chars,
        overlap_sentences=overlap_sentences,
        max_chunks=max_chunks,
    )
    if not chunks:
        return {}, {}

    # --- init Ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, num_cpus=num_cpus or os.cpu_count())

    # --- map: submit one task per chunk (runs concurrently)
    futures = [
        process_chunk_remote.options(name=f"process_chunk[{i}]").remote(
            c,
            window=window,
            keep_fraction=keep_fraction,
            dedup_per_chunk=True,   # per your requirement
        )
        for i, c in enumerate(chunks)
    ]

    results = ray.get(futures)

    # --- collect errors (non-fatal)
    errs = [r["err"] for r in results if r.get("err")]
    if errs:
        print("[ray] Non-fatal worker errors:")
        for e in errs[:10]:
            print("  -", e)
        if len(errs) > 10:
            print(f"  ... +{len(errs)-10} more")

    # --- merge per-chunk maps
    partial_map = _merge_keyword_maps([r["keyword_map"] for r in results])

    # --- (optional) single global dedup across chunks
    if final_global_dedup:
        seed_terms = sorted(
            set(partial_map.keys())
            | set(itertools.chain.from_iterable(partial_map.values()))
        )
        print(f"[ray] Final global dedup over {len(seed_terms)} terms...")
        dedup_json = deduplicate_keywords_with_claude(seed_terms)
        keyword_map = json_to_dict(dedup_json)
    else:
        keyword_map = partial_map

    # --- build edges over full text using final canonical nodes
    nodes = list(keyword_map.keys())
    adj = createEdges(text, nodes)

    return keyword_map, adj

def main():
    load_dotenv()
    setup_environment()

    with open("articles/sorting_algorithms.txt", "r") as f:
        text = f.read()

    keyword_map, adj = run_parallel_pipeline(
        text,
        target_chars=6000,
        hard_max_chars=9000,
        overlap_sentences=2,   # a little overlap helps boundary concepts
        max_chunks=5,          # <= 5 Claude calls
        num_cpus=None,
        window=4,
        keep_fraction=0.33,
        final_global_dedup=False,   # keep within 5 calls/min
    )

    print(json.dumps(keyword_map, indent=2)[:2000])

    print("PLOTTING GRAPH\n")
    plot_graph(
        adj,
        top_k_out=None,
        min_weight=1,
        layout="spring",
        title="Co-occurrence Graph (parallel per-chunk dedup, ≤5 tasks)",
        save_path=None,
        show_weights=False,
    )

if __name__ == "__main__":
    main()
