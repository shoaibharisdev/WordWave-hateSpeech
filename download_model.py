#!/usr/bin/env python3
"""
Script to pre-download the ultra-light hate speech model
"""
import os
import sys
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def download_model():
    print("📥 Pre-downloading ultra-light hate speech model for Render...")
    try:
        # Download ultra-light model
        model_name = "microsoft/DialogRPT-offensive"
        
        print(f"Downloading model: {model_name}")
        print("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        print("Downloading model...")
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        print("✅ Ultra-light model downloaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Model download failed: {e}")
        return False

if __name__ == "__main__":
    success = download_model()
    sys.exit(0 if success else 1)