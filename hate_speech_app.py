from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os
from dotenv import load_dotenv

app = Flask(__name__)
CORS(app)

load_dotenv()

# ===== CONFIG =====
HF_API_URL = "https://api-inference.huggingface.co/models/unitary/toxic-bert"
HF_API_TOKEN = os.getenv("HG_TOKEN") 
HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}

TOXIC_LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

# ===== FUNCTIONS =====
def predict_hate_speech(text):
    """Send text to Hugging Face Inference API instead of local PyTorch model"""
    try:
        payload = {"inputs": text}
        response = requests.post(HF_API_URL, headers=HEADERS, json=payload, timeout=20)
        response.raise_for_status()
        predictions = response.json()[0]

        detected_categories = []
        max_confidence = 0.0

        for pred in predictions:
            label = pred["label"].lower()
            score = pred["score"]
            if score > 0.5:
                detected_categories.append({"label": label, "confidence": round(score, 3)})
                max_confidence = max(max_confidence, score)

        is_hate = len(detected_categories) > 0
        classification = (
            max(detected_categories, key=lambda x: x["confidence"])["label"]
            if is_hate
            else "Not classified as any category"
        )

        return {
            "hate_speech": is_hate,
            "classification": classification,
            "confidence": round(max_confidence, 3) if is_hate else 0.0,
            "detected_categories": detected_categories,
        }

    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

# ===== ROUTES =====
@app.route("/api/predict", methods=["POST"])
def predict_hate_speech_endpoint():
    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "Text field is required"}), 400
    result = predict_hate_speech(text)
    return jsonify(result)

@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "model": "unitary/toxic-bert (via Hugging Face API)",
        "requires_pytorch": False,
    })

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Hate Speech Detection API (lightweight mode)",
        "version": "2.0.0",
        "endpoints": {"predict": "POST /api/predict", "health": "GET /api/health"},
        "model": {"name": "unitary/toxic-bert", "type": "multi-label classification", "labels": TOXIC_LABELS},
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
