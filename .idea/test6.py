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
    Nodes not in any cluster are removed completely.
    """
    new_G = nx.Graph()
    
    # Build mapping from original node -> canonical node
    node_map = {}
    for canonical, members in clusters.items():
        for node in members:
            node_map[node] = canonical

    # Add edges to the new graph
    for u, v, data in G.edges(data=True):
        if u not in node_map or v not in node_map:
            continue  # drop unmapped nodes
        
        u_new, v_new = node_map[u], node_map[v]
        weight = data.get('weight', 1.0)

        # Allow self-loops if merged nodes were connected
        if new_G.has_edge(u_new, v_new):
            current = new_G[u_new][v_new]['weight']
            new_G[u_new][v_new]['weight'] = max(current, weight)
        else:
            new_G.add_edge(u_new, v_new, weight=weight)

    # Ensure all canonical nodes exist
    for canonical in clusters:
        if canonical not in new_G:
            new_G.add_node(canonical)

    print("\n--- DEBUG: Final Edges ---")
    for u, v, w in new_G.edges(data=True):
        print(f"{u} -- {v}, weight={w['weight']}")
    
    draw_graph(new_G, title="Consolidated Graph")
    return new_G

def main():
    load_dotenv()
    setup_environment()
    with open("articles/sorting_algorithms.txt", 'r') as file:
        text = file.read() 
        G, phrases = extract_key_phrases(text, plot=True)
        # print(phrases)
        dedup_phrases = deduplicate_keywords_with_claude(phrases)
        print(dedup_phrases)
        keyword_map = json_to_dict(dedup_phrases)
        print(keyword_map)
        new_G = consolidate_graph(G, keyword_map)

    

if __name__ == "__main__":
    main()

