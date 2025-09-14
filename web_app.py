#!/usr/bin/env python3

from flask import Flask, render_template, request, jsonify
import os
import sys
import json
import io
import base64
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import networkx as nx
import tempfile
import traceback
import threading
import time
from datetime import datetime

# Add current directory to Python path to ensure import works
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from text_analyzer import TextAnalyzer
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure text_analyzer.py is in the same directory as web_app.py")
    sys.exit(1)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global variables
current_analyzer = None
analysis_status = {
    'status': 'idle',  # idle, processing, completed, error
    'progress': 0,
    'message': '',
    'error': None,
    'start_time': None
}

def log_message(message):
    """Log message with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")
    sys.stdout.flush()  # Force flush to see logs immediately

@app.route('/')
def index():
    """Main page with file upload form."""
    try:
        return render_template('index.html')
    except Exception as e:
        log_message(f"Template error: {str(e)}")
        return f"Template error: {str(e)}", 500

@app.route('/status')
def get_status():
    """Get current analysis status - always returns JSON."""
    global analysis_status
    
    try:
        log_message(f"Status check: {analysis_status['status']} - {analysis_status['message']}")
        
        # Add elapsed time if processing
        status_copy = analysis_status.copy()
        if analysis_status['start_time']:
            status_copy['elapsed_time'] = time.time() - analysis_status['start_time']
        
        return jsonify(status_copy)
    
    except Exception as e:
        log_message(f"Status endpoint error: {e}")
        # Always return JSON even on error
        return jsonify({
            'status': 'error',
            'progress': 0,
            'message': 'Status check failed',
            'error': str(e),
            'start_time': None
        })

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and start analysis."""
    global current_analyzer, analysis_status
    
    log_message("=== UPLOAD STARTED ===")
    
    try:
        # Reset status
        analysis_status = {
            'status': 'processing',
            'progress': 5,
            'message': 'Upload received, starting analysis...',
            'error': None,
            'start_time': time.time()
        }
        
        log_message("Status reset to processing")
        
        if 'file' not in request.files:
            log_message("ERROR: No file in request")
            analysis_status['status'] = 'error'
            analysis_status['error'] = 'No file provided'
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        log_message(f"File received: {file.filename}")
        
        if file.filename == '':
            log_message("ERROR: Empty filename")
            analysis_status['status'] = 'error'
            analysis_status['error'] = 'No file selected'
            return jsonify({'error': 'No file selected'}), 400
        
        # Get analysis parameters from form
        target_chars = int(request.form.get('target_chars', 6000))
        max_chunks = int(request.form.get('max_chunks', 5))
        window = int(request.form.get('window', 4))
        keep_fraction = float(request.form.get('keep_fraction', 0.33))
        
        log_message(f"Parameters: target_chars={target_chars}, max_chunks={max_chunks}, window={window}, keep_fraction={keep_fraction}")
        
        # Save uploaded file temporarily
        analysis_status['message'] = 'Saving uploaded file...'
        analysis_status['progress'] = 10
        
        file_content = file.read().decode('utf-8')
        log_message(f"File content length: {len(file_content)} characters")
        
        temp_fd, temp_path = tempfile.mkstemp(suffix='.txt', text=True)
        log_message(f"Created temp file: {temp_path}")
        
        try:
            with os.fdopen(temp_fd, 'w', encoding='utf-8') as tmp_file:
                tmp_file.write(file_content)
            
            log_message("File saved successfully")
            
            # Start analysis in a separate thread to avoid blocking
            def run_analysis():
                global current_analyzer, analysis_status
                log_message("=== ANALYSIS THREAD STARTED ===")
                
                try:
                    analysis_status['message'] = 'Creating analyzer...'
                    analysis_status['progress'] = 20
                    log_message("Creating TextAnalyzer...")
                    
                    current_analyzer = TextAnalyzer(temp_path)
                    log_message("TextAnalyzer created successfully")
                    
                    analysis_status['message'] = 'Running text analysis...'
                    analysis_status['progress'] = 30
                    log_message("Starting analysis...")
                    
                    keyword_map, adj = current_analyzer.analyze(
                        target_chars=target_chars,
                        max_chunks=max_chunks,
                        window=window,
                        keep_fraction=keep_fraction,
                        final_global_dedup=False,
                        plot_graph=False  # Don't show plot, we'll generate it for web
                    )
                    
                    log_message(f"Analysis completed! Keywords: {len(keyword_map)}, Edges: {sum(len(neighbors) for neighbors in adj.values())}")
                    
                    analysis_status['message'] = 'Analysis completed!'
                    analysis_status['progress'] = 100
                    analysis_status['status'] = 'completed'
                    
                except Exception as e:
                    error_msg = f"Analysis failed: {str(e)}"
                    log_message(f"ANALYSIS ERROR: {error_msg}")
                    log_message(f"Full traceback: {traceback.format_exc()}")
                    
                    analysis_status['status'] = 'error'
                    analysis_status['error'] = error_msg
                    analysis_status['progress'] = 0
                finally:
                    # Clean up temp file
                    try:
                        if os.path.exists(temp_path):
                            os.unlink(temp_path)
                            log_message("Temp file cleaned up")
                        if current_analyzer:
                            current_analyzer.shutdown_ray()
                            log_message("Ray shutdown completed")
                    except Exception as cleanup_error:
                        log_message(f"Cleanup error: {cleanup_error}")
            
            # Start analysis thread
            log_message("Starting analysis thread...")
            analysis_thread = threading.Thread(target=run_analysis)
            analysis_thread.daemon = True
            analysis_thread.start()
            log_message("Analysis thread started")
            
            return jsonify({
                'success': True,
                'message': 'Analysis started',
                'filename': file.filename
            })
            
        except Exception as e:
            # Clean up temp file on error
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            except:
                pass
            raise
            
    except Exception as e:
        error_msg = f"Upload failed: {str(e)}"
        log_message(f"UPLOAD ERROR: {error_msg}")
        log_message(f"Full traceback: {traceback.format_exc()}")
        
        analysis_status['status'] = 'error'
        analysis_status['error'] = error_msg
        
        return jsonify({'error': error_msg}), 500

@app.route('/results')
def get_results():
    """Get analysis results when completed."""
    global current_analyzer, analysis_status
    
    try:
        log_message("Results requested")
        
        if analysis_status['status'] != 'completed':
            log_message(f"Results requested but status is: {analysis_status['status']}")
            return jsonify({'error': 'Analysis not completed yet'}), 400
            
        if not current_analyzer or not current_analyzer.adjacency:
            log_message("No analysis data available")
            return jsonify({'error': 'No analysis data available'}), 400
        
        log_message("Generating graph visualization...")
        
        # Generate graph image
        graph_img = generate_graph_image(current_analyzer.adjacency)
        
        log_message("Graph generated successfully!")
        
        return jsonify({
            'graph_image': graph_img,
            'keyword_map': current_analyzer.keyword_map,
            'keywords_count': len(current_analyzer.keyword_map),
            'edges_count': sum(len(neighbors) for neighbors in current_analyzer.adjacency.values())
        })
        
    except Exception as e:
        error_msg = f"Results generation failed: {str(e)}"
        log_message(f"RESULTS ERROR: {error_msg}")
        log_message(f"Full traceback: {traceback.format_exc()}")
        return jsonify({'error': error_msg}), 500

def generate_graph_image(adj):
    """Generate graph image as base64 string."""
    try:
        log_message("Creating graph from adjacency data...")
        
        # Create the graph
        edge_weights = {}
        for u, nbrs in adj.items():
            for v, w in nbrs.items():
                if w is None or w < 1 or u == v:
                    continue
                key = frozenset([u, v])
                edge_weights[key] = edge_weights.get(key, 0) + w

        G = nx.Graph()
        for key, w in edge_weights.items():
            u, v = tuple(key)
            G.add_edge(u, v, weight=w)

        # Remove isolated nodes
        isolates = list(nx.isolates(G))
        G.remove_nodes_from(isolates)
        
        log_message(f"Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        # Create the visualization
        plt.figure(figsize=(14, 10))
        
        if G.number_of_nodes() == 0:
            plt.text(0.5, 0.5, 'No connections found', ha='center', va='center', fontsize=16)
            plt.axis('off')
        else:
            pos = nx.spring_layout(G, k=0.7, iterations=200, seed=42)
            
            # Node sizes based on degree
            node_sizes = [max(300, 100 * G.degree(node)) for node in G.nodes()]
            
            # Edge widths based on weights
            weights = [G[u][v]["weight"] for u, v in G.edges()]
            if weights:
                w_min, w_max = min(weights), max(weights)
                if w_min == w_max:
                    widths = [3.0 for _ in weights]
                else:
                    widths = [1.5 + 4.5 * (w - w_min) / (w_max - w_min) for w in weights]
            else:
                widths = []

            # Draw the graph
            nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='lightblue', 
                                  alpha=0.7, linewidths=0.8)
            nx.draw_networkx_edges(G, pos, width=widths, edge_color='gray', alpha=0.6)
            nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
            
            plt.title("Keyword Co-occurrence Graph", fontsize=16, fontweight='bold')
            plt.axis('off')
        
        # Convert plot to base64 string
        img_buffer = io.BytesIO()
        plt.tight_layout()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        
        img_str = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()  # Important: close the figure to free memory
        
        log_message("Graph image generated successfully")
        return img_str
        
    except Exception as e:
        log_message(f"Graph image generation error: {str(e)}")
        raise

@app.route('/health')
def health_check():
    """Simple health check endpoint."""
    return jsonify({'status': 'healthy', 'message': 'Server is running'})

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    log_message(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Check if required files exist
    if not os.path.exists('text_analyzer.py'):
        print("Error: text_analyzer.py not found in current directory")
        sys.exit(1)
    
    if not os.path.exists('templates/index.html'):
        print("Error: templates/index.html not found")
        sys.exit(1)
    
    log_message("Starting Flask application...")
    log_message(f"Templates directory: {os.path.abspath('templates')}")
    log_message(f"Current directory: {os.getcwd()}")
    
    app.run(debug=True, host='0.0.0.0', port=8001, threaded=True)
