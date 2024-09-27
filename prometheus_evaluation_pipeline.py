import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load Prometheus model from Hugging Face
prometheus_tokenizer = AutoTokenizer.from_pretrained("prometheus-eval/prometheus-7b-v2.0")
prometheus_model = AutoModelForSequenceClassification.from_pretrained("prometheus-eval/prometheus-7b-v2.0")

def evaluate_with_prometheus(prompt, answer):
    """Evaluate a prompt-answer pair using the Prometheus model."""
    inputs = prometheus_tokenizer(prompt + " " + answer, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = prometheus_model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        score = probabilities[0][1].item()  # Assuming binary classification (class 1 is positive)
    return score
