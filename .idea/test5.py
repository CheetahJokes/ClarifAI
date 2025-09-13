from Levenshtein import distance
import io
import itertools
import networkx as nx
import nltk
import os
import time
import math
import cProfile
import pstats
from contextlib import contextmanager

# ------------- Simple timing utilities -------------
_TIMINGS = {}

@contextmanager
def timed(name):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        _TIMINGS.setdefault(name, []).append(dt)

def print_timing_summary():
    print("\n=== Timing summary ===")
    for k, vs in _TIMINGS.items():
        total = sum(vs)
        n = len(vs)
        avg = total / n
        print(f"{k:30s} total={total:8.3f}s  calls={n:3d}  avg={avg:7.3f}s")
    print("=== End summary ===\n")

# ------------- NLTK setup -------------
def setup_environment():
    """Download required resources (NLTK >= 3.9 names)."""
    needed = [
        ("punkt", "tokenizers/punkt"),
        ("punkt_tab", "tokenizers/punkt_tab"),
        ("averaged_perceptron_tagger_eng", "taggers/averaged_perceptron_tagger_eng"),
    ]
    for res, path in needed:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(res)
    print('Completed resource downloads.')

# ------------- Helpers -------------
def filter_for_tags(tagged, tags=('NN', 'JJ', 'NNP')):
    """Apply syntactic filters based on POS tags."""
    return [item for item in tagged if item[1] in tags]

def normalize(tagged):
    """Return a list of tuples with the first item's periods removed."""
    return [(item[0].replace('.', ''), item[1]) for item in tagged]

def unique_everseen(iterable, key=None):
    """List unique elements in order of appearance."""
    seen = set()
    if key is None:
        key = lambda x: x
    for element in iterable:
        k = key(element)
        if k not in seen:
            seen.add(k)
            yield element

# ------------- Graph builder with progress + ETA -------------
def build_graph(nodes, progress_every_pct=5):
    """
    Return a networkx graph instance with edges weighted by Levenshtein distance.
    Shows progress & ETA for O(n^2) work.
    """
    with timed("graph_build_total"):
        gr = nx.Graph()
        gr.add_nodes_from(nodes)

        n = len(nodes)
        total_edges = n * (n - 1) // 2
        if total_edges == 0:
            return gr

        next_mark = progress_every_pct
        processed = 0
        t0 = time.perf_counter()

        for (a, b) in itertools.combinations(nodes, 2):
            # dist = distance(a, b)
            # If you want PageRank to treat closer nodes as stronger edges, flip to similarity:
            dist = distance(a, b)
            # sim = 1.0 / (1.0 + dist)     # uncomment to use similarity instead
            # gr.add_edge(a, b, weight=sim)
            gr.add_edge(a, b, weight=dist)

            processed += 1
            pct = (processed * 100) / total_edges
            if pct >= next_mark:
                elapsed = time.perf_counter() - t0
                rate = processed / max(elapsed, 1e-9)
                remaining = total_edges - processed
                eta = remaining / max(rate, 1e-9)
                print(f"  Graph build: {pct:5.1f}%  "
                      f"({processed}/{total_edges})  "
                      f"elapsed={elapsed:6.1f}s  ETA={eta:6.1f}s")
                next_mark += progress_every_pct

        return gr

# ------------- Key phrase extraction with profiling -------------
def extract_key_phrases(text):
    """Return a set of key phrases."""
    with timed("tokenize"):
        word_tokens = nltk.word_tokenize(text)  # uses punkt/punkt_tab

    with timed("pos_tag"):
        # Explicit language for NLTK >=3.9 split taggers
        tagged = nltk.pos_tag(word_tokens, lang='eng')

    with timed("filter_normalize"):
        textlist = [x[0] for x in tagged]
        tagged = normalize(filter_for_tags(tagged))
        word_set_list = list(unique_everseen([x[0] for x in tagged]))

    with timed("graph_build_words"):
        graph = build_graph(word_set_list)

    with timed("pagerank_words"):
        calculated_page_rank = nx.pagerank(graph, weight='weight')

    with timed("keyphrase_select"):
        keyphrases = sorted(calculated_page_rank, key=calculated_page_rank.get, reverse=True)
        one_third = len(word_set_list) // 3
        keyphrases = keyphrases[0:one_third + 1]

        modified_key_phrases = set()
        i = 0
        while i < len(textlist):
            w = textlist[i]
            if w in keyphrases:
                phrase_ws = [w]
                i += 1
                while i < len(textlist) and textlist[i] in keyphrases:
                    phrase_ws.append(textlist[i])
                    i += 1
                phrase = ' '.join(phrase_ws)
                modified_key_phrases.add(phrase)
            else:
                i += 1

    return modified_key_phrases

# ------------- Sentence extraction with profiling -------------
def extract_sentences(text, summary_length=100, clean_sentences=False, language='english'):
    """Return a paragraph formatted summary of the source text."""
    with timed("sentence_tokenize"):
        sent_detector = nltk.data.load('tokenizers/punkt/' + language + '.pickle')
        sentence_tokens = sent_detector.tokenize(text.strip())

    with timed("graph_build_sentences"):
        graph = build_graph(sentence_tokens)

    with timed("pagerank_sentences"):
        calculated_page_rank = nx.pagerank(graph, weight='weight')

    with timed("summary_assemble"):
        sentences = sorted(calculated_page_rank, key=calculated_page_rank.get, reverse=True)
        summary = ' '.join(sentences)
        summary_words = summary.split()
        summary_words = summary_words[0:summary_length]
        dot_indices = [idx for idx, word in enumerate(summary_words) if '.' in word]
        if clean_sentences and dot_indices:
            last_dot = max(dot_indices) + 1
            summary = ' '.join(summary_words[0:last_dot])
        else:
            summary = ' '.join(summary_words)
        return summary

def write_files(summary, key_phrases, filename):
    """Write key phrases and summaries to a file."""
    with timed("write_files"):
        os.makedirs('keywords', exist_ok=True)
        os.makedirs('summaries', exist_ok=True)
        print("Generating output to " + 'keywords/' + filename)
        with io.open('keywords/' + filename, 'w') as key_phrase_file:
            for key_phrase in key_phrases:
                key_phrase_file.write(key_phrase + '\n')

        print("Generating output to " + 'summaries/' + filename)
        with io.open('summaries/' + filename, 'w') as summary_file:
            summary_file.write(summary)
        print("-")

def summarize_all():
    articles = sorted(os.listdir("articles"))
    print(f"Found {len(articles)} article(s).")
    for idx, article in enumerate(articles, 1):
        print(f"\n[{idx}/{len(articles)}] Reading articles/{article}")
        with timed("per_article_total"):
            with open(os.path.join('articles', article), 'r') as f:
                text = f.read()
            keyphrases = extract_key_phrases(text)
            summary = extract_sentences(text)
            write_files(summary, keyphrases, article)

if __name__ == "__main__":
    setup_environment()

    # ---- optional: cProfile the whole run ----
    profile_output = "profile_stats.prof"
    print(f"\nProfiling run... (output: {profile_output})")
    pr = cProfile.Profile()
    pr.enable()

    with timed("TOTAL_RUN"):
        summarize_all()

    pr.disable()
    pr.dump_stats(profile_output)

    # Print top hotspots
    print("\n=== cProfile: top 25 by cumulative time ===")
    ps = pstats.Stats(pr).strip_dirs().sort_stats("cumulative")
    ps.print_stats(25)
    print("=== End cProfile ===")

    # Human-readable phase summary
    print_timing_summary()
