"""
Neural Image Captioning Studio - Flask Backend with Model Inference
"""
from flask import Flask, render_template, request, jsonify
import os
import time
import random
import base64
import json
import numpy as np
import cv2
import pickle
import traceback

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    HAS_TF = True
except ImportError:
    HAS_TF = False
    print("Warning: TensorFlow not found. Running in simulation mode.")

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Model configuration mapping to filenames in notebook
MODELS_CONFIG = [
    {
        "id": "resnet_lstm",
        "name": "ResNet50 + LSTM",
        "file": "Resnet50_LSTM_model.keras",
        "tokenizer": "tokenizer_en.pkl",
        "icon": "🔵", "color": "#7c6aff", "params": "~33M", "speed": "Fast", "arch": "ResNet50+LSTM",
        "desc": "Deep residual network extracts 2048-dim features, fed into LSTM decoder with 256 units."
    },
    {
        "id": "vgg_gru",
        "name": "VGG + GRU",
        "file": "model_vgg_gru.keras",
        "tokenizer": "tokenizer_en.pkl",
        "icon": "🟣", "color": "#a855f7", "params": "~138M", "speed": "Medium", "arch": "VGG16+GRU",
        "desc": "VGG-16 pretrained on ImageNet extracts 4096-dim features, decoded by Gated Recurrent Unit."
    },
    {
        "id": "cnn_rnn",
        "name": "CNN + SimpleRNN",
        "file": "model.keras",
        "tokenizer": "tokenizer_en.pkl",
        "icon": "🟡", "color": "#ffb743", "params": "~5M", "speed": "Very Fast", "arch": "CNN+RNN",
        "desc": "Lightweight custom CNN architecture with SimpleRNN decoder. Optimized for speed."
    },
    {
        "id": "cnn_gru_attn",
        "name": "CNN + GRU + Attention",
        "file": "CNN_GRU_Attention_model.keras",
        "tokenizer": "tokenizer_en.pkl",
        "icon": "🟠", "color": "#ff5e9e", "params": "~12M", "speed": "Medium", "arch": "CNN+Attention",
        "desc": "Bahdanau attention mechanism over CNN features with GRU decoder for focused captioning."
    },
    {
        "id": "cnn_transformer",
        "name": "CNN + Transformers",
        "file": "CNN_Tranformer_model_en.keras",
        "tokenizer": "tokenizer_en.pkl",
        "icon": "🔴", "color": "#00d4ff", "params": "~20M", "speed": "Slow", "arch": "CNN+Transformer",
        "desc": "CNN encoder with multi-head self-attention transformer decoder. State-of-the-art architecture."
    },
]

class InferenceService:
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.loaded_status = {}
        self.base_dir = os.path.dirname(__file__)

    def load_resources(self):
        if not HAS_TF:
            return
        
        print("\n" + "="*60)
        print("  AI Model Loader")
        print("="*60)
        
        for config in MODELS_CONFIG:
            model_id = config['id']
            model_path = os.path.join(self.base_dir, config['file'])
            tok_path = os.path.join(self.base_dir, config['tokenizer'])

            try:
                # Load Tokenizer
                if os.path.exists(tok_path):
                    with open(tok_path, 'rb') as f:
                        self.tokenizers[model_id] = pickle.load(f)
                else:
                    print(f"  [!] Missing tokenizer for {model_id}: {config['tokenizer']}")
                    self.loaded_status[model_id] = False
                    continue

                # Load Model
                if os.path.exists(model_path):
                    print(f"  [>] Loading {config['name']}...")
                    self.models[model_id] = keras.models.load_model(model_path)
                    self.loaded_status[model_id] = True
                else:
                    print(f"  [!] Missing model file for {model_id}: {config['file']}")
                    self.loaded_status[model_id] = False
            except Exception as e:
                print(f"  [ERROR] Failed to load {model_id}: {str(e)}")
                self.loaded_status[model_id] = False
        print("="*60 + "\n")

    def preprocess_image(self, image_path):
        """Preprocess image to match notebook logic: resize to 224x224 and normalize."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Could not read image file")
        
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Notebook uses standard float32 conversion
        img_array = np.array(img, dtype=np.float32)
        
        # Apply preprocess_input if needed (most models in notebook use resnet50.preprocess_input or similar)
        # For simplicity, we implement the common ResNet50 preprocessing here:
        # Subtract mean and swap channels if necessary, but Keras usually handles this in the layer if part of model.
        # If it's NOT part of the model, we should call the specific preprocess_input.
        # In the notebook it was called explicitly: x = resnet50.preprocess_input(x)
        
        # We will add a check if it's a resnet model
        # For now, let's keep it consistent with the notebook's explicit calls if possible
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array

    def generate_caption(self, model_id, image_path, max_length=35):
        if not self.loaded_status.get(model_id):
            return None, 0.0

        model = self.models[model_id]
        tokenizer = self.tokenizers[model_id]
        
        try:
            image_input = self.preprocess_image(image_path)
            
            # Apply preprocess_input based on model type
            if "resnet" in model_id:
                image_input = tf.keras.applications.resnet50.preprocess_input(image_input)
            elif "vgg" in model_id:
                image_input = tf.keras.applications.vgg16.preprocess_input(image_input)

            in_text = "<start>"
            confidences = []

            for _ in range(max_length):
                sequence = tokenizer.texts_to_sequences([in_text])[0]
                sequence = pad_sequences([sequence], maxlen=max_length, padding='post')
                
                y_pred = model.predict([image_input, sequence], verbose=0)
                
                # Logic from notebook: y_pred = np.argmax(y_pred[0, len(in_text.split())-1])
                # We need the full distribution to get confidence
                idx = len(in_text.split()) - 1
                if idx >= y_pred.shape[1]: break # Sequence too long
                
                pred_dist = y_pred[0, idx]
                predicted_id = np.argmax(pred_dist)
                
                # Track confidence for this word
                confidences.append(float(pred_dist[predicted_id]))

                word = None
                for w, index in tokenizer.word_index.items():
                    if index == predicted_id:
                        word = w
                        break
                
                if word is None:
                    break
                
                in_text += " " + word
                
                if word == "<end>" or len(in_text.split()) > 20:
                    break
            
            # Clean up caption
            final_caption = in_text.replace("<start>", "").replace("<end>", "").strip()
            avg_confidence = float(np.mean(confidences)) if confidences else 0.0
            
            return final_caption, avg_confidence
        except Exception as e:
            print(f"[ERROR] Inference error: {traceback.format_exc()}")
            return None, 0.0

# Initialize Service
service = InferenceService()
service.load_resources()

@app.route('/')
def index():
    return render_template('index.html', models=MODELS_CONFIG)

@app.route('/api/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    f = request.files['image']
    if f.filename == '':
        return jsonify({"error": "Empty filename"}), 400
    
    filename = f"upload_{int(time.time())}_{f.filename}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    f.save(filepath)
    return jsonify({"success": True, "filename": filename, "path": f"/static/uploads/{filename}"})

@app.route('/api/generate', methods=['POST'])
def generate():
    data = request.get_json()
    model_id = data.get('model_id', '')
    image_url = data.get('image', '')
    
    if not image_url:
        return jsonify({"error": "No image URL provided"}), 400

    # Resolve local path from URL
    if image_url.startswith('/static/'):
        local_path = os.path.join(os.path.dirname(__file__), image_url.lstrip('/'))
    else:
        # Handle cases where path might be relative or full
        local_path = os.path.join(os.path.dirname(__file__), image_url.replace('/', os.sep))

    start_time = time.time()
    
    caption = None
    confidence = 0.0
    
    if HAS_TF and service.loaded_status.get(model_id):
        caption, confidence = service.generate_caption(model_id, local_path)
    
    # Fallback to simulation if model not loaded or error occurred
    if caption is None:
        time.sleep(random.uniform(0.5, 1.5))
        mock_captions = {
            "resnet_lstm": "A wide angle shot of a group of hikers walking along a scenic mountain trail",
            "vgg_gru": "A cozy interior of a cafe with sunlight streaming through large windows",
            "cnn_rnn": "A playful dog jumping to catch a colorful frisbee in a green field",
            "cnn_gru_attn": "A close up of a chef carefully plating a gourmet dish in a kitchen",
            "cnn_transformer": "An aerial view of a vibrant city skyline during the golden hour"
        }
        caption = mock_captions.get(model_id, "A high-quality image captured by neural network models.")
        confidence = round(random.uniform(0.65, 0.85), 3)
        simulated = True
    else:
        simulated = False

    time_ms = int((time.time() - start_time) * 1000)
    
    return jsonify({
        "model_id": model_id,
        "caption": caption,
        "confidence": confidence,
        "tokens": len(caption.split()),
        "time_ms": time_ms,
        "simulated": simulated
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("  🧠 Neural Image Captioning Studio (Backend)")
    print("  🌐 Open: http://localhost:5000")
    if not HAS_TF:
        print("  ⚠️ RUNNING WITHOUT TENSORFLOW - Simulation Mode Active")
    print("="*60 + "\n")
    app.run(debug=True, port=5000)
