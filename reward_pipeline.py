import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load pre-trained GPT-2 model for sequence classification
tokenizer = AutoTokenizer.from_pretrained("gpt2")
reward_model = AutoModelForSequenceClassification.from_pretrained("gpt2")

def calculate_reward(prompt, answer):
    """Calculate a reward score for a given prompt-answer pair."""
    inputs = tokenizer(prompt + answer, return_tensors="pt")
    with torch.no_grad():
        outputs = reward_model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        reward_score = probabilities[0][1].item()  # Assuming binary classification (class 1 is positive)
    return reward_score
