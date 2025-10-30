from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from torch.nn.functional import sigmoid
from transformers import BertTokenizer, BertForSequenceClassification
import os
import gc

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# ===== HATE SPEECH DETECTION SETUP =====
model = None
tokenizer = None

# Define labels for toxic-bert
TOXIC_LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

def load_hate_speech_model():
    """Load the hate speech model"""
    global model, tokenizer
    if model is None or tokenizer is None:
        try:
            print("🔄 Loading hate speech model...")
            tokenizer = BertTokenizer.from_pretrained("unitary/toxic-bert")
            model = BertForSequenceClassification.from_pretrained("unitary/toxic-bert")
            model.eval()
            print("✅ Hate speech model loaded successfully")
        except Exception as e:
            print(f"❌ Hate speech model loading failed: {e}")
            return None, None
    return tokenizer, model

def predict_hate_speech(text):
    """
    Predict if text contains hate speech using toxic-bert model.
    This model is multi-label, so we use sigmoid activation and thresholds.
    """
    tokenizer_local, model_local = load_hate_speech_model()
    
    if tokenizer_local is None or model_local is None:
        return {"error": "Hate speech model not available"}
    
    try:
        inputs = tokenizer_local(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        
        with torch.no_grad():
            outputs = model_local(**inputs)
            probabilities = sigmoid(outputs.logits)
        
        threshold = 0.5
        predictions = probabilities[0].tolist()
        detected_categories = []
        max_confidence = 0.0
        
        for i, label in enumerate(TOXIC_LABELS):
            if predictions[i] > threshold:
                detected_categories.append({
                    "label": label,
                    "confidence": round(predictions[i], 3)
                })
                max_confidence = max(max_confidence, predictions[i])
        
        is_hate = len(detected_categories) > 0
        
        if is_hate:
            primary_category = max(detected_categories, key=lambda x: x['confidence'])
            classification = primary_category['label']
        else:
            classification = "Not classified as any category"
        
        # Clean up
        del inputs, outputs, probabilities
        gc.collect()
        
        return {
            "hate_speech": is_hate,
            "classification": classification,
            "confidence": round(max_confidence, 3) if is_hate else 0.0,
            "detected_categories": detected_categories
        }
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

# ===== ROUTES =====

@app.route('/api/predict', methods=['POST'])
def predict_hate_speech_endpoint():
    """API endpoint for hate speech prediction"""
    data = request.get_json()
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'Text field is required'}), 400
    
    result = predict_hate_speech(text)
    return jsonify(result)

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint for hate speech detection"""
    status = "ready" if model is not None else "loaded"
    return jsonify({
        'status': 'healthy', 
        'model': 'toxic-bert',
        'model_status': status
    })

@app.route('/', methods=['GET'])
def home():
    """Home endpoint with API information"""
    return jsonify({
        "message": "Hate Speech Detection API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "POST /api/predict",
            "health": "GET /api/health"
        },
        "model": {
            "name": "unitary/toxic-bert",
            "type": "multi-label classification",
            "labels": TOXIC_LABELS
        }
    })

# Load model on startup
@app.before_first_request
def load_model_on_startup():
    """Load the model when the app starts"""
    print("🚀 Starting Hate Speech Detection API...")
    load_hate_speech_model()

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)