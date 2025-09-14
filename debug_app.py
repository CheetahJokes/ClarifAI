from flask import Flask, render_template, request, jsonify
import os
import sys
import json
import traceback
import tempfile

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)

@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')

@app.route('/test')
def test_route():
    """Test endpoint to verify server is working."""
    print("TEST ROUTE CALLED!")
    return jsonify({'message': 'Test successful', 'status': 'working'})

@app.route('/upload', methods=['POST'])
def upload_file():
    """Simple upload test without analysis."""
    print("=== UPLOAD ROUTE CALLED ===")
    
    try:
        print("Checking for file in request...")
        if 'file' not in request.files:
            print("ERROR: No file in request")
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        print(f"File received: {file.filename}")
        
        if file.filename == '':
            print("ERROR: Empty filename")
            return jsonify({'error': 'No file selected'}), 400
        
        # Test reading the file
        file_content = file.read().decode('utf-8')
        content_length = len(file_content)
        print(f"File content length: {content_length} characters")
        print(f"First 100 characters: {file_content[:100]}")
        
        # Test importing TextAnalyzer
        print("Testing TextAnalyzer import...")
        try:
            from text_analyzer import TextAnalyzer
            print("✓ TextAnalyzer import successful")
        except Exception as e:
            print(f"✗ TextAnalyzer import failed: {e}")
            return jsonify({'error': f'Import failed: {str(e)}'}), 500
        
        # Test creating a temporary file
        print("Testing temp file creation...")
        temp_fd, temp_path = tempfile.mkstemp(suffix='.txt', text=True)
        try:
            with os.fdopen(temp_fd, 'w', encoding='utf-8') as tmp_file:
                tmp_file.write(file_content)
            print(f"Temp file created: {temp_path}")
            
            # Test TextAnalyzer creation
            print("Testing TextAnalyzer creation...")
            analyzer = TextAnalyzer(temp_path)
            print("✓ TextAnalyzer created successfully")
            
            return jsonify({
                'success': True,
                'message': 'All tests passed!',
                'filename': file.filename,
                'content_length': content_length,
                'temp_path': temp_path
            })
            
        except Exception as e:
            print(f"Error during testing: {e}")
            print(f"Full traceback: {traceback.format_exc()}")
            return jsonify({'error': f'Test failed: {str(e)}'}), 500
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                print("Temp file cleaned up")
    
    except Exception as e:
        print(f"UPLOAD ERROR: {e}")
        print(f"Full traceback: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("=== STARTING DEBUG SERVER ===")
    print(f"Current directory: {os.getcwd()}")
    print(f"Files in directory: {os.listdir('.')}")
    
    # Check if required files exist
    if os.path.exists('text_analyzer.py'):
        print("✓ text_analyzer.py found")
    else:
        print("✗ text_analyzer.py NOT found")
    
    if os.path.exists('templates/index.html'):
        print("✓ templates/index.html found")
    else:
        print("✗ templates/index.html NOT found")
    
    app.run(debug=True, host='0.0.0.0', port=8001)  # Different port to avoid conflicts
