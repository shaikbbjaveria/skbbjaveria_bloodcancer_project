import os
import sys

# Set environment variables for better compatibility
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging
os.environ['PYTHONUNBUFFERED'] = '1'

print("Installing TensorFlow for Python 3.14...")
# Install TensorFlow CPU version that works with Python 3.14
os.system(f"{sys.executable} -m pip install tensorflow-cpu")
try:
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
except ImportError as e:
    print(f"TensorFlow import error: {e}")
    print("Attempting to install regular TensorFlow...")
    os.system(f"{sys.executable} -m pip install tensorflow --no-deps")
    os.system(f"{sys.executable} -m pip install numpy==1.23.5")
    try:
        import tensorflow as tf
        print(f"TensorFlow version: {tf.__version__}")
    except ImportError as e2:
        print(f"Both TensorFlow installations failed: {e2}")
        print("Running without TensorFlow...")
        tf = None

# Import and run the Flask app
from app import app

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    print(f"Starting app on port {port}...")
    app.run(host='0.0.0.0', port=port)
