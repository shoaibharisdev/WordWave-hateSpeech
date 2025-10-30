from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from torch.nn.functional import sigmoid
from transformers import BertTokenizer, BertForSequenceClassification
import os
import gc

app = Flask(__name__)
CORS(app)

# ===== HATE SPEECH DETECTION SETUP =====
# Model will be loaded at startup - no lazy loading
model = None
tokenizer = None

TOXIC_LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

def load_hate_speech_model():
    """Load the hate speech model at startup"""
    global model, tokenizer
    try:
        print("🔄 Loading hate speech model...")
        tokenizer = BertTokenizer.from_pretrained("unitary/toxic-bert")
        model = BertForSequenceClassification.from_pretrained("unitary/toxic-bert")
        model.eval()
        print("✅ Hate speech model loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Hate speech model loading failed: {e}")
        return False

def predict_hate_speech(text):
    """Predict hate speech - model is always loaded"""
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
            classification = "clean"
        
        # Clean up memory
        del inputs, outputs, probabilities
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
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
        'model': 'toxic-bert',
        'model_loaded': model is not None
    })

@app.route('/', methods=['GET'])
def home():
    """Home endpoint"""
    return jsonify({
        "message": "Hate Speech Detection API",
        "status": "running",
        "model_ready": model is not None
    })

# Load model at application startup
print("🚀 Starting Hate Speech Detection API...")
print("📥 Pre-loading hate speech model...")
load_hate_speech_model()
print("✅ Application startup complete!")

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False)