#!/usr/bin/env python3
"""
Script to pre-download the finiteautomata sentiment model for hate speech detection
"""
import os
import sys
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def download_model():
    print("📥 Pre-downloading finiteautomata sentiment model for Render...")
    try:
        # Download finiteautomata model - VERIFIED TO EXIST
        model_name = "finiteautomata/bertweet-base-sentiment-analysis"
        
        print(f"Downloading model: {model_name}")
        print("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        print("Downloading model...")
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        print("✅ Finiteautomata model downloaded successfully!")
        print("📊 Model info:")
        print(f"   - Model: {model_name}")
        print(f"   - Tokenizer vocab size: {len(tokenizer)}")
        print(f"   - Model parameters: ~150M")
        print(f"   - Labels: Negative, Neutral, Positive")
        return True
    except Exception as e:
        print(f"❌ Model download failed: {e}")
        return False

if __name__ == "__main__":
    success = download_model()
    sys.exit(0 if success else 1)