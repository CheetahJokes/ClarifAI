#!/usr/bin/env python3

print("Testing imports...")

try:
    import sys
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"Python path: {sys.path[:3]}...")  # Show first 3 paths
except Exception as e:
    print(f"Error getting Python info: {e}")
try:
    import matplotlib
    print(f"✅ Matplotlib imported: {matplotlib.__version__}")
except ImportError as e:
    print(f"❌ Matplotlib import failed: {e}")

try:
    import networkx
    print(f"✅ NetworkX imported: {networkx.__version__}")
except ImportError as e:
    print(f"❌ NetworkX import failed: {e}")

try:
    import nltk
    print(f"✅ NLTK imported: {nltk.__version__}")
except ImportError as e:
    print(f"❌ NLTK import failed: {e}")

try:
    import ray
    print(f"✅ Ray imported: {ray.__version__}")
    ray.init()
except ImportError as e:
    print(f"❌ Ray import failed: {e}")

try:
    import flask
    print(f"✅ Flask imported successfully: {flask.__version__}")
    app = flask.Flask(__name__)
    app.run()
except ImportError as e:
    print(f"❌ Flask import failed: {e}")

try:
    import anthropic
    print(f"✅ Anthropic imported: {anthropic.__version__}")
except ImportError as e:
    print(f"❌ Anthropic import failed: {e}")

print("\nTest complete!")
