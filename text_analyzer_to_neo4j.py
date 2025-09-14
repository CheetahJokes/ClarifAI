# text_analyzer_neo4j.py

import math
import string
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

# ---- Your sink (import if you put it in a module) ----
# from neo4j_sink import Neo4JSink
# For this snippet we assume the class Neo4JSink is importable as above.

Adj = Dict[str, Dict[str, int]]

PUNCT = set(string.punctuation)
LEMMATIZER = WordNetLemmatizer()


def _ensure_stopwords() -> set:
    try:
        return set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords")
        return set(stopwords.words("english"))


STOP = _ensure_stopwords()


class TextAnalyzerToNeo4j:
    """
    Analyze text into (keyword_map, adjacency) and save into Neo4j.
    - keyword_map: {canonical_term: [variants...]}
    - adjacency: directed co-occurrence counts {u: {v: w}}
    """

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.text = None
        self.keyword_map: Dict[str, List[str]] = {}
        self.adjacency: Adj = {}

        load_dotenv()
        self.setup_environment()
        self._load_text()

    def _load_text(self):
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                self.text = f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {self.file_path}")
        except Exception as e:
            raise Exception(f"Error reading file {self.file_path}: {e}")

    @staticmethod
    def setup_environment():
        nltk.download("punkt", quiet=True)
        # NLTK >=3.8 uses this name; if you're on older NLTK, change to 'averaged_perceptron_tagger'
        nltk.download("averaged_perceptron_tagger_eng", quiet=True)
        nltk.download("wordnet", quiet=True)

    @staticmethod
    def filter_for_tags(tagged, tags=("NN", "NNS", "NNP", "JJ")):
        return [item for item in tagged if item[1] in tags]

    @staticmethod
    def normalize(tagged):
        return [(item[0].replace(".", ""), item[1]) for item in tagged]

    @staticmethod
    def unique_everseen(iterable, key=None):
        seen = set()
        result = []
        for element in iterable:
            k = element if key is None else key(element)
            if k not in seen:
                seen.add(k)
                result.append(element)
        return result

    @staticmethod
    def _clean_token(w: str) -> str:
        w = w.lower().strip(string.punctuation)
        if any(ch.isdigit() for ch in w):
            return ""
        return w

    def _candidate_tokens(
        self,
        text: str,
        *,
        use_lemmatize=True,
        remove_stopwords=True,
        min_len=2,
    ):
        word_tokens = nltk.word_tokenize(text)
        tagged_all = nltk.pos_tag(word_tokens, tagset=None, lang="eng")

        textlist = [w for (w, _) in tagged_all]
        tagged = self.filter_for_tags(tagged_all)
        tagged = self.normalize(tagged)

        cand_tokens = []
        for w, pos in tagged:
            w = self._clean_token(w)
            if not w or len(w) < min_len:
                continue
            if remove_stopwords and w in STOP:
                continue
            if all(ch in PUNCT for ch in w):
                continue
            if use_lemmatize:
                tag0 = pos[0].lower() if pos else "n"
                wn_pos = {"j": "a", "n": "n", "v": "v", "r": "r"}.get(tag0, "n")
                w = LEMMATIZER.lemmatize(w, pos=wn_pos)
            if len(w) < min_len:
                continue
            if remove_stopwords and w in STOP:
                continue
            cand_tokens.append(w)

        return textlist, cand_tokens

    def _build_cooccurrence_graph(self, cand_tokens, window=4):
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

    def _bigram_counts(self, cand_tokens):
        return Counter(zip(cand_tokens, cand_tokens[1:]))

    def extract_key_phrases(
        self,
        text: str,
        *,
        window=4,
        keep_fraction=0.33,
    ):
        """
        Returns:
          G: full co-occurrence graph (NetworkX Graph)
          top_phrases: set[str]
        """
        textlist, cand_tokens = self._candidate_tokens(text)
        if not cand_tokens:
            return nx.Graph(), set()

        tf = Counter(cand_tokens)
        total = sum(tf.values())
        personalization = {w: tf[w] / total for w in tf}

        G = self._build_cooccurrence_graph(cand_tokens, window=window)
        if G.number_of_nodes() == 0:
            return G, set()

        pr = nx.pagerank(G, weight="weight", personalization=personalization)

        sorted_words = sorted(pr, key=pr.get, reverse=True)
        keep_n = max(1, int(len(sorted_words) * keep_fraction))
        key_words = set(sorted_words[:keep_n])

        big = self._bigram_counts(cand_tokens)
        N = total

        def pmi(u, v):
            pu = (tf[u] + 1) / (N + len(tf))
            pv = (tf[v] + 1) / (N + len(tf))
            cuv = big.get((u, v), 0) + big.get((v, u), 0)
            pv_uv = (cuv + 1) / (N + len(big))
            return math.log(pv_uv / (pu * pv))

        phrases = []
        i = 0
        while i < len(textlist):
            w = textlist[i].lower().strip(string.punctuation)
            if w in key_words:
                run_surface = [textlist[i]]
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

        best = defaultdict(float)
        for p, s in phrases:
            if s > best[p]:
                best[p] = s
        ranked_phrases = sorted(best.items(), key=lambda x: x[1], reverse=True)
        top_phrases = {p for p, _ in ranked_phrases[:max(5, int(len(ranked_phrases) * 0.5))]}
        return G, top_phrases

    def deduplicate_keywords_with_claude(self, keywords: List[str]) -> str:
        prompt = f"""{HUMAN_PROMPT}
You are given a list of keywords. Create a strict, high-quality keyword list:

Rules:
1) Filter aggressively (drop generic/meaningless).
2) Group & deduplicate to a lowercase canonical form.
3) Output JSON ONLY as: {{"canonical": ["variant1","variant2", ...]}}
4) No duplicates across lists.

Input keywords:
{keywords}
{AI_PROMPT}"""

        client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        resp = client.messages.create(
            model="claude-opus-4-1-20250805",
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text

    @staticmethod
    def json_to_dict(json_str: str) -> Dict[str, List[str]]:
        try:
            cleaned = json_str.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.strip("`")
                if cleaned.lower().startswith("json"):
                    cleaned = cleaned[4:].strip()
            data = json.loads(cleaned)
            if not isinstance(data, dict):
                raise ValueError("Parsed JSON is not a dict.")
            # ensure lists
            out = {}
            for k, v in data.items():
                if isinstance(v, list):
                    out[k] = v
            return out
        except Exception as e:
            print("JSON parse error:", e)
            return {}

    def _merge_keyword_maps(self, maps: List[Dict[str, List[str]]]) -> Dict[str, List[str]]:
        merged: Dict[str, set] = {}
        for m in maps:
            for canon, variants in m.items():
                merged.setdefault(canon, set()).update(variants)
        return {k: sorted(v) for k, v in merged.items()}

    def chunk_text_sliding_by_sentences(
        self,
        target_chars: int = 5000,
        hard_max_chars: int = 8000,
        overlap_sentences: int = 2,
        max_chunks: int = 5,
    ) -> List[str]:
        sents = [s for s in re.split(r"(?<=[.!?])\s+", self.text.strip()) if s.strip()]
        chunks = []
        i = 0
        while i < len(sents) and len(chunks) < max_chunks:
            buf, cur_len, j = [], 0, i
            while j < len(sents):
                s_len = len(sents[j]) + 1
                if (cur_len + s_len > target_chars and buf) or (cur_len + s_len > hard_max_chars):
                    break
                buf.append(sents[j])
                cur_len += s_len
                j += 1
            if buf:
                chunks.append(" ".join(buf))
            if j >= len(sents) or len(chunks) >= max_chunks:
                break
            i = max(i + 1, j - overlap_sentences)
        return chunks

    def create_edges(
        self,
        nodes: List[str],
        *,
        min_support: int = 1,
        make_aliases: bool = True,
        sentence_splitter: Callable[[str], List[str]] = None,
    ) -> Adj:
        canonical = {n: n.strip() for n in nodes}
        alias_patterns: Dict[str, List[re.Pattern]] = {}

        def _aliases_for(base: str) -> List[str]:
            base = base.lower().strip()
            forms = {base}
            if make_aliases:
                forms |= {base.replace("-", " "), base.replace(" ", "-")}
                if not base.endswith("s") and " " not in base:
                    forms.add(base + "s")
            return sorted(forms)

        for n in nodes:
            pats = []
            for form in _aliases_for(n):
                pats.append(re.compile(rf"\b{re.escape(form)}\b", re.IGNORECASE))
            alias_patterns[n] = pats

        if sentence_splitter is None:
            sentence_splitter = lambda t: [s for s in re.split(r"(?<=[.!?])\s+", t) if s.strip()]

        sentences = sentence_splitter(self.text)

        counts: Dict[Tuple[str, str], int] = {}

        def nodes_in_sentence(sent: str) -> List[str]:
            present = []
            lower = sent.lower()
            for node, pats in alias_patterns.items():
                if any(p.search(lower) for p in pats):
                    present.append(canonical[node])
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
            for u, v in itertools.combinations(present, 2):
                counts[(u, v)] = counts.get((u, v), 0) + 1
                counts[(v, u)] = counts.get((v, u), 0) + 1

        adjacency: Adj = {n: {} for n in nodes}
        for (u, v), c in counts.items():
            if u == v:
                continue
            if c >= min_support:
                adjacency.setdefault(u, {})
                adjacency[u][v] = adjacency[u].get(v, 0) + c
        return adjacency

    def _run_sequential_pipeline(
        self,
        chunks: List[str],
        *,
        window: int = 4,
        keep_fraction: float = 0.33,
        final_global_dedup: bool = False,
    ):
        all_phrases = []
        partial_maps = []

        for i, chunk in enumerate(chunks):
            try:
                _G, phrases = self.extract_key_phrases(
                    chunk, window=window, keep_fraction=keep_fraction
                )
                phrases = sorted(set(phrases))
                all_phrases.extend(phrases)

                if os.getenv("ANTHROPIC_API_KEY"):
                    try:
                        dedup_json = self.deduplicate_keywords_with_claude(phrases)
                        kw_map = self.json_to_dict(dedup_json)
                        if kw_map:
                            partial_maps.append(kw_map)
                    except Exception as e:
                        print(f"[chunk {i+1}] Claude dedup failed: {e}")
                        partial_maps.append({p: [p] for p in phrases})
                else:
                    partial_maps.append({p: [p] for p in phrases})
            except Exception as e:
                print(f"[chunk {i+1}] extract error: {e}")

        if partial_maps:
            keyword_map = self._merge_keyword_maps(partial_maps)
        else:
            unique_phrases = sorted(set(all_phrases))
            keyword_map = {p: [p] for p in unique_phrases}

        nodes = list(keyword_map.keys())
        adj = self.create_edges(nodes)

        self.keyword_map = keyword_map
        self.adjacency = adj
        return keyword_map, adj

    def run_parallel_pipeline(
        self,
        *,
        target_chars: int = 5000,
        hard_max_chars: int = 8000,
        overlap_sentences: int = 2,
        max_chunks: int = 5,
        num_cpus: Optional[int] = None,
        window: int = 4,
        keep_fraction: float = 0.33,
        final_global_dedup: bool = False,
    ):
        chunks = self.chunk_text_sliding_by_sentences(
            target_chars=target_chars,
            hard_max_chars=hard_max_chars,
            overlap_sentences=overlap_sentences,
            max_chunks=max_chunks,
        )
        if not chunks:
            return {}, {}

        try:
            if not ray.is_initialized():
                ray.init(
                    ignore_reinit_error=True,
                    num_cpus=min(num_cpus or os.cpu_count() or 2, 4),
                    _temp_dir="/tmp/ray",
                )
        except Exception as e:
            print("Ray init failed:", e)
            return self._run_sequential_pipeline(
                chunks, window=window, keep_fraction=keep_fraction, final_global_dedup=final_global_dedup
            )

        @ray.remote
        def process_chunk_remote(chunk_text: str, window: int, keep_fraction: float):
            import os as _os
            from dotenv import load_dotenv as _ld
            _ld()
            ta = TextAnalyzerToNeo4j.__new__(TextAnalyzerToNeo4j)
            ta.file_path = None
            ta.text = None
            ta.keyword_map = {}
            ta.adjacency = {}
            try:
                ta.setup_environment()
            except Exception:
                pass
            try:
                _G, phrases = ta.extract_key_phrases(
                    chunk_text, window=window, keep_fraction=keep_fraction
                )
            except Exception as e:
                return {"phrases": [], "kw_map": {}, "err": f"extract_failed: {e}"}

            phrases = sorted(set(phrases))
            if not _os.getenv("ANTHROPIC_API_KEY"):
                return {"phrases": phrases, "kw_map": {}, "err": None}

            try:
                dedup_json = ta.deduplicate_keywords_with_claude(phrases)
                kw_map = ta.json_to_dict(dedup_json)
                return {"phrases": [], "kw_map": kw_map, "err": None}
            except Exception as e:
                return {"phrases": phrases, "kw_map": {}, "err": f"claude_failed: {e}"}

        try:
            futures = [
                process_chunk_remote.options(name=f"process_chunk[{i}]").remote(
                    c, window, keep_fraction
                )
                for i, c in enumerate(chunks)
            ]
            results = ray.get(futures)
        except Exception as e:
            print("Ray map failed:", e)
            return self._run_sequential_pipeline(
                chunks, window=window, keep_fraction=keep_fraction, final_global_dedup=final_global_dedup
            )

        partial_map = self._merge_keyword_maps([r["kw_map"] for r in results])
        leftover_phrases = list(
            set(itertools.chain.from_iterable(r["phrases"] for r in results))
        )
        if leftover_phrases:
            for p in leftover_phrases:
                partial_map.setdefault(p, []).append(p)

        if final_global_dedup and os.getenv("ANTHROPIC_API_KEY"):
            seed_terms = sorted(
                set(partial_map.keys())
                | set(itertools.chain.from_iterable(partial_map.values()))
            )
            try:
                dedup_json = self.deduplicate_keywords_with_claude(seed_terms)
                keyword_map = self.json_to_dict(dedup_json) or partial_map
            except Exception as e:
                print("Final dedup failed:", e)
                keyword_map = partial_map
        else:
            keyword_map = partial_map

        nodes = list(keyword_map.keys())
        adj = self.create_edges(nodes)

        self.keyword_map = keyword_map
        self.adjacency = adj
        return keyword_map, adj

    # ---------------- Neo4j integration ----------------

    def save_to_neo4j(self, *, doc_path: str):
        """
        Persist the current keyword_map + adjacency into Neo4j via Neo4JSink.
        """
        if not self.keyword_map or not self.adjacency:
            raise RuntimeError("Nothing to save: run analyze() first to build the graph.")

        from neo4j_sink import Neo4JSink  # adjust import to your file/module name

        sink = Neo4JSink()
        try:
            sink.save_graph(doc_path=doc_path, keyword_map=self.keyword_map, adjacency=self.adjacency)
        finally:
            sink.close()

    # ---------------- Public entrypoint ----------------

    def analyze_and_save(
        self,
        *,
        doc_path: str,
        target_chars: int = 6000,
        hard_max_chars: int = 9000,
        overlap_sentences: int = 2,
        max_chunks: int = 5,
        num_cpus: Optional[int] = None,
        window: int = 4,
        keep_fraction: float = 0.33,
        final_global_dedup: bool = False,
        use_ray: bool = True,
    ):
        """
        Full pipeline → writes to Neo4j. Returns (keyword_map, adjacency).
        """
        if use_ray:
            try:
                keyword_map, adj = self.run_parallel_pipeline(
                    target_chars=target_chars,
                    hard_max_chars=hard_max_chars,
                    overlap_sentences=overlap_sentences,
                    max_chunks=max_chunks,
                    num_cpus=num_cpus,
                    window=window,
                    keep_fraction=keep_fraction,
                    final_global_dedup=final_global_dedup,
                )
            except Exception as e:
                print("Parallel pipeline failed, falling back:", e)
                chunks = self.chunk_text_sliding_by_sentences(
                    target_chars=target_chars,
                    hard_max_chars=hard_max_chars,
                    overlap_sentences=overlap_sentences,
                    max_chunks=max_chunks,
                )
                keyword_map, adj = self._run_sequential_pipeline(
                    chunks, window=window, keep_fraction=keep_fraction, final_global_dedup=final_global_dedup
                )
        else:
            chunks = self.chunk_text_sliding_by_sentences(
                target_chars=target_chars,
                hard_max_chars=hard_max_chars,
                overlap_sentences=overlap_sentences,
                max_chunks=max_chunks,
            )
            keyword_map, adj = self._run_sequential_pipeline(
                chunks, window=window, keep_fraction=keep_fraction, final_global_dedup=final_global_dedup
            )

        # Persist to Neo4j
        self.save_to_neo4j(doc_path=doc_path)

        # Helpful logs (trim long JSON)
        try:
            print("[Saved to Neo4j]")
            print(json.dumps(keyword_map, indent=2)[:2000])
        except Exception:
            pass

        return keyword_map, adj

    @staticmethod
    def shutdown_ray():
        if ray.is_initialized():
            ray.shutdown()


# -------------- Example CLI usage --------------
def main():
    # Point to your input file (used also as Neo4j Document.path)
    in_path = "articles/sorting_algorithms.txt"
    analyzer = TextAnalyzerToNeo4j(in_path)
    try:
        analyzer.analyze_and_save(
            doc_path=in_path,      # this becomes Document {path: ...} in Neo4j
            target_chars=6000,
            hard_max_chars=9000,
            overlap_sentences=2,
            max_chunks=5,
            num_cpus=None,
            window=4,
            keep_fraction=0.33,
            final_global_dedup=False,
            use_ray=True,
        )
    finally:
        analyzer.shutdown_ray()


if __name__ == "__main__":
    main()
