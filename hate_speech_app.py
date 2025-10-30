from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from torch.nn.functional import sigmoid
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import gc

app = Flask(__name__)
CORS(app)

# ===== HATE SPEECH DETECTION SETUP =====
# Using a smaller, more efficient model
model = None
tokenizer = None

# Using a smaller model that fits in 512MB
MODEL_NAME = "cardiffnlp/twitter-roberta-base-offensive"  # Much smaller model

TOXIC_LABELS = ["offensive", "not_offensive"]  # Simplified labels for smaller model

def load_hate_speech_model():
    """Load a smaller hate speech model"""
    global model, tokenizer
    try:
        print("🔄 Loading hate speech model...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        model.eval()
        print("✅ Hate speech model loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Hate speech model loading failed: {e}")
        return False

def predict_hate_speech(text):
    """Predict hate speech with smaller model"""
    global model, tokenizer
    
    if model is None or tokenizer is None:
        return {"error": "Hate speech model not available"}
    
    try:
        inputs = tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=128
        )
        
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
        
        # Get the prediction (offensive or not)
        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][prediction].item()
        
        is_hate = (prediction == 0)  # 0 = offensive, 1 = not offensive
        
        # Clean up memory
        del inputs, outputs, probabilities
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        return {
            "hate_speech": bool(is_hate),
            "classification": "offensive" if is_hate else "clean",
            "confidence": round(confidence, 3),
            "detected_categories": [{"label": "offensive", "confidence": round(confidence, 3)}] if is_hate else []
        }
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

# ===== ROUTES =====

@app.route('/api/predict', methods=['POST'])
def predict_hate_speech_endpoint():
    """API endpoint for hate speech prediction"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
            
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'Text field is required'}), 400
        
        result = predict_hate_speech(text)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy' if model is not None else 'unhealthy',
        'model': MODEL_NAME,
        'model_loaded': model is not None
    })

@app.route('/', methods=['GET'])
def home():
    """Home endpoint"""
    return jsonify({
        "message": "Hate Speech Detection API",
        "status": "running",
        "model_ready": model is not None,
        "model": MODEL_NAME
    })

# Load model at application startup
print("🚀 Starting Hate Speech Detection API...")
print("📥 Pre-loading hate speech model...")
load_hate_speech_model()
print("✅ Application startup complete!")

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False)