#!/usr/bin/env python3

from flask import Flask, render_template, request, jsonify
import os
import sys
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import tempfile
import traceback
import threading
import atexit
import time
from datetime import datetime
from multiprocessing import get_context
from queue import Empty
import pickle

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from text_analyzer import TextAnalyzer  # child re-imports; parent import ok
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure text_analyzer.py is in the same directory as web_app.py")
    sys.exit(1)

def log_message(msg: str):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")
    sys.stdout.flush()

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

RAY_INIT_TIMEOUT_SECS = 20

# ----------------------------- Workers -----------------------------

def analysis_worker_ray(temp_path, params, progress_q, result_path, ray_address=None):
    try:
        progress_q.put(("status", "Ray worker starting..."))
        import ray, sys as _sys

        if _sys.version_info >= (3, 13):
            progress_q.put(("status", "Python 3.13 detected; Ray may be unstable"))
        progress_q.put(("status", "Initializing Ray..."))

        if ray_address:
            ray.init(address=ray_address, include_dashboard=False, ignore_reinit_error=True, namespace="webapp")
        else:
            ray.init(include_dashboard=False, ignore_reinit_error=True, namespace="webapp")

        progress_q.put(("status", "Creating analyzer..."))
        analyzer = TextAnalyzer(temp_path)

        progress_q.put(("status", "Chunking text...")); progress_q.put(("progress", 30))
        progress_q.put(("status", "Processing chunks (Ray)...")); progress_q.put(("progress", 40))

        keyword_map, adj = analyzer.analyze(
            target_chars=params["target_chars"],
            max_chunks=params["max_chunks"],
            window=params["window"],
            keep_fraction=params["keep_fraction"],
            final_global_dedup=False,
            plot_graph=False,
            use_ray=True,
        )

        with open(result_path, "wb") as f:
            pickle.dump({"keyword_map": keyword_map, "adj": adj}, f)

        progress_q.put(("progress", 100))
        progress_q.put(("status", "Done"))
        progress_q.put(("done", True))

    except Exception as e:
        try:
            progress_q.put(("error", f"{type(e).__name__}: {e}"))
            import traceback as _tb
            progress_q.put(("error_detail", "".join(_tb.format_exc())))
        except Exception:
            pass
    finally:
        try:
            import ray
            if ray.is_initialized():
                ray.shutdown()
        except Exception:
            pass

def analysis_worker_local(temp_path, params, progress_q, result_path):
    try:
        progress_q.put(("status", "Local worker starting (no Ray)..."))
        from text_analyzer import TextAnalyzer
        analyzer = TextAnalyzer(temp_path)

        progress_q.put(("status", "Chunking text...")); progress_q.put(("progress", 30))
        progress_q.put(("status", "Processing chunks (local)...")); progress_q.put(("progress", 40))

        keyword_map, adj = analyzer.analyze(
            target_chars=params["target_chars"],
            max_chunks=params["max_chunks"],
            window=params["window"],
            keep_fraction=params["keep_fraction"],
            final_global_dedup=False,
            plot_graph=False,
            use_ray=False,
        )

        with open(result_path, "wb") as f:
            pickle.dump({"keyword_map": keyword_map, "adj": adj}, f)

        progress_q.put(("progress", 100))
        progress_q.put(("status", "Done (local)"))
        progress_q.put(("done", True))

    except Exception as e:
        try:
            progress_q.put(("error", f"{type(e).__name__}: {e}"))
            import traceback as _tb
            progress_q.put(("error_detail", "".join(_tb.format_exc())))
        except Exception:
            pass

# ----------------------------- Global state -----------------------------

current_analyzer = None
analysis_status = {
    'status': 'idle',
    'progress': 0,
    'message': '',
    'error': None,
    'start_time': None
}

# ----------------------------- Routes -----------------------------

@app.route('/')
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        log_message(f"Template error: {e}")
        return f"Template error: {e}", 500

@app.route('/status')
def get_status():
    global analysis_status
    try:
        log_message(f"Status check: {analysis_status['status']} - {analysis_status.get('message','')}")
        status_copy = analysis_status.copy()
        if analysis_status['start_time']:
            status_copy['elapsed_time'] = time.time() - analysis_status['start_time']
        return jsonify(status_copy)
    except Exception as e:
        log_message(f"Status endpoint error: {e}")
        return jsonify({
            'status': 'error', 'progress': 0,
            'message': 'Status check failed', 'error': str(e), 'start_time': None
        })

@app.route('/upload', methods=['POST'])
def upload_file():
    global current_analyzer, analysis_status

    log_message("=== UPLOAD STARTED ===")
    try:
        analysis_status = {
            'status': 'processing',
            'progress': 5,
            'message': 'Upload received, starting analysis...',
            'error': None,
            'start_time': time.time()
        }

        if 'file' not in request.files:
            analysis_status['status'] = 'error'
            analysis_status['error'] = 'No file provided'
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        log_message(f"File received: {file.filename}")

        if file.filename == '':
            analysis_status['status'] = 'error'
            analysis_status['error'] = 'No file selected'
            return jsonify({'error': 'No file selected'}), 400

        # Params
        target_chars = int(request.form.get('target_chars', 6000))
        max_chunks = int(request.form.get('max_chunks', 5))
        window = int(request.form.get('window', 4))
        keep_fraction = float(request.form.get('keep_fraction', 0.33))
        params = dict(target_chars=target_chars, max_chunks=max_chunks, window=window, keep_fraction=keep_fraction)
        log_message(f"Parameters: {params}")

        # Save uploaded file
        analysis_status['message'] = 'Saving uploaded file...'; analysis_status['progress'] = 10
        file_content = file.read().decode('utf-8', errors='ignore')
        temp_fd, temp_path = tempfile.mkstemp(suffix='.txt', text=True)
        with os.fdopen(temp_fd, 'w', encoding='utf-8', errors='ignore') as tmp_file:
            tmp_file.write(file_content)
        log_message(f"Temp file saved: {temp_path} ({len(file_content)} chars)")

        # Use one multiprocessing context end-to-end
        mp_ctx = get_context("spawn")
        progress_q = mp_ctx.Queue()  # <-- IMPORTANT: Queue from the same context
        result_fd, result_path = tempfile.mkstemp(suffix=".pkl"); os.close(result_fd)

        ray_address = os.getenv("RAY_ADDRESS")
        force_local = os.getenv("FORCE_LOCAL") == "1"

        def start_proc(target, args):
            p = mp_ctx.Process(target=target, args=args)
            p.daemon = False
            p.start()
            return p

        # Start worker (Ray or local) 
        if force_local:
            proc = start_proc(analysis_worker_local, (temp_path, params, progress_q, result_path))
        else:
            proc = start_proc(analysis_worker_ray, (temp_path, params, progress_q, result_path, ray_address))

        def progress_pump():
            nonlocal proc  # we rebind it if we fallback to local
            global analysis_status, current_analyzer
            analysis_status['status'] = 'processing'
            analysis_status['progress'] = 15
            analysis_status['message'] = 'Spawned analysis worker'
            last_msg_time = time.time()
            saw_initializing = False

            try:
                while True:
                    try:
                        msg = progress_q.get(timeout=1.0)
                        kind, payload = msg
                        last_msg_time = time.time()

                        if kind == "status":
                            analysis_status['message'] = payload
                            if "Initializing Ray" in payload:
                                saw_initializing = True

                        elif kind == "progress":
                            analysis_status['progress'] = int(payload)

                        elif kind == "error":
                            analysis_status['status'] = 'error'
                            analysis_status['error'] = payload
                            # optional extra detail
                            try:
                                k2, detail = progress_q.get_nowait()
                                if k2 == "error_detail":
                                    analysis_status['error'] += f"\n{detail}"
                            except Empty:
                                pass
                            break

                        elif kind == "done":
                            with open(result_path, "rb") as f:
                                data = pickle.load(f)
                            class _Holder: pass
                            h = _Holder()
                            h.keyword_map = data.get("keyword_map", {})
                            h.adjacency = data.get("adj", {})
                            current_analyzer = h
                            analysis_status['status'] = 'completed'
                            analysis_status['message'] = 'Analysis completed!'
                            break

                    except Empty:
                        # Child died?
                        if not proc.is_alive():
                            analysis_status['status'] = 'error'
                            analysis_status['error'] = f"Worker exited unexpectedly (exitcode={proc.exitcode})."
                            break

                        # Ray init stalled? Fallback to local
                        if (not force_local) and saw_initializing and (time.time() - last_msg_time) > RAY_INIT_TIMEOUT_SECS:
                            analysis_status['message'] = 'Ray init stalled; falling back to local mode...'
                            try: proc.terminate()
                            except Exception: pass
                            # Drain queue
                            while True:
                                try: progress_q.get_nowait()
                                except Empty: break
                            proc = start_proc(analysis_worker_local, (temp_path, params, progress_q, result_path))
                            saw_initializing = False
                            last_msg_time = time.time()

            except Exception as e:
                analysis_status['status'] = 'error'
                analysis_status['error'] = f"Progress pump failed: {e}"
            finally:
                # Cleanup temp files
                try:
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                except Exception:
                    pass
                try:
                    if os.path.exists(result_path):
                        os.unlink(result_path)
                except Exception:
                    pass

        threading.Thread(target=progress_pump, daemon=True).start()
        return jsonify({'success': True, 'message': 'Analysis started', 'filename': file.filename})

    except Exception as e:
        error_msg = f"Upload failed: {e}"
        log_message(f"UPLOAD ERROR: {error_msg}")
        log_message(f"Full traceback: {traceback.format_exc()}")
        analysis_status['status'] = 'error'
        analysis_status['error'] = error_msg
        return jsonify({'error': error_msg}), 500

@app.route('/results')
def get_results():
    global current_analyzer, analysis_status
    try:
        log_message("Results requested")
        if analysis_status['status'] != 'completed':
            log_message(f"Results requested but status is: {analysis_status['status']}")
            return jsonify({'error': 'Analysis not completed yet'}), 400
        if not current_analyzer or not getattr(current_analyzer, "adjacency", None):
            log_message("No analysis data available")
            return jsonify({'error': 'No analysis data available'}), 400

        log_message("Generating graph visualization...")
        graph_img = generate_graph_image(current_analyzer.adjacency)
        log_message("Graph generated successfully!")
        return jsonify({
            'graph_image': graph_img,
            'keyword_map': current_analyzer.keyword_map,
            'keywords_count': len(current_analyzer.keyword_map),
            'edges_count': sum(len(nbrs) for nbrs in current_analyzer.adjacency.values())
        })
    except Exception as e:
        error_msg = f"Results generation failed: {e}"
        log_message(f"RESULTS ERROR: {error_msg}")
        log_message(f"Full traceback: {traceback.format_exc()}")
        return jsonify({'error': error_msg}), 500

def generate_graph_image(adj):
    try:
        log_message("Creating graph from adjacency data...")
        edge_weights = {}
        for u, nbrs in adj.items():
            for v, w in nbrs.items():
                if w is None or w < 1 or u == v: continue
                key = frozenset([u, v])
                edge_weights[key] = edge_weights.get(key, 0) + w

        G = nx.Graph()
        for key, w in edge_weights.items():
            u, v = tuple(key)
            G.add_edge(u, v, weight=w)

        G.remove_nodes_from(list(nx.isolates(G)))
        log_message(f"Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

        plt.figure(figsize=(14, 10))
        if G.number_of_nodes() == 0:
            plt.text(0.5, 0.5, 'No connections found', ha='center', va='center', fontsize=16)
            plt.axis('off')
        else:
            pos = nx.spring_layout(G, k=0.7, iterations=200, seed=42)
            node_sizes = [max(300, 100 * G.degree(n)) for n in G.nodes()]
            weights = [G[u][v]["weight"] for u, v in G.edges()]
            if weights:
                w_min, w_max = min(weights), max(weights)
                widths = [3.0 if w_min == w_max else 1.5 + 4.5 * (w - w_min) / (w_max - w_min) for w in weights]
            else:
                widths = []
            nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='lightblue', alpha=0.7, linewidths=0.8)
            nx.draw_networkx_edges(G, pos, width=widths, edge_color='gray', alpha=0.6)
            nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
            plt.title("Keyword Co-occurrence Graph", fontsize=16, fontweight='bold'); plt.axis('off')

        buf = io.BytesIO()
        plt.tight_layout(); plt.savefig(buf, format='png', dpi=150, bbox_inches='tight'); buf.seek(0)
        img_str = base64.b64encode(buf.getvalue()).decode()
        plt.close()
        log_message("Graph image generated successfully")
        return img_str
    except Exception as e:
        log_message(f"Graph image generation error: {e}")
        raise

@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy', 'message': 'Server is running'})

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    log_message(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

def cleanup_ray():
    try:
        import ray
        if ray.is_initialized():
            ray.shutdown()
            log_message("Ray shutdown completed")
    except Exception:
        pass

atexit.register(cleanup_ray)

if __name__ == '__main__':
    if not os.path.exists('text_analyzer.py'):
        print("Error: text_analyzer.py not found in current directory"); sys.exit(1)
    if not os.path.exists('templates/index.html'):
        print("Error: templates/index.html not found"); sys.exit(1)

    log_message("Starting Flask application...")
    log_message(f"Templates directory: {os.path.abspath('templates')}")
    log_message(f"Current directory: {os.getcwd()}")

    # Parent never initializes Ray. Child process handles it.
    app.run(debug=False, host='0.0.0.0', port=8001, threaded=True, use_reloader=False)
