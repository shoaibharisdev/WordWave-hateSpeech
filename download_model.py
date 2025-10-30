#!/usr/bin/env python3
"""
Script to pre-download the hate speech model during Render build phase
"""
import os
import sys
from transformers import BertTokenizer, BertForSequenceClassification

def download_model():
    print("📥 Pre-downloading hate speech model for Render...")
    try:
        # Download tokenizer
        print("Downloading tokenizer...")
        tokenizer = BertTokenizer.from_pretrained("unitary/toxic-bert")
        
        # Download model
        print("Downloading model...")
        model = BertForSequenceClassification.from_pretrained("unitary/toxic-bert")
        
        print("✅ Model downloaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Model download failed: {e}")
        return False

if __name__ == "__main__":
    success = download_model()
    sys.exit(0 if success else 1)