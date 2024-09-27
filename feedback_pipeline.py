import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load a pre-trained GPT-2 model for generating feedback
tokenizer = AutoTokenizer.from_pretrained("gpt2")
language_model = AutoModelForCausalLM.from_pretrained("gpt2")

def filter_low_reward(data, threshold=0.7):
    """
    Filter out low-reward answers based on a reward score threshold.
    Returns only rows where the reward_score is less than the threshold.
    """
    return data[data['reward_score'] < threshold]

def generate_feedback_with_ai(prompt):
    """
    Generate feedback for a given prompt using GPT-2 model.
    """
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=100)
    attention_mask = torch.tensor((inputs['input_ids'] != tokenizer.pad_token_id).numpy(), dtype=torch.int64)
    
    # Generate feedback using the GPT-2 model with adjusted parameters
    with torch.no_grad():
        output = language_model.generate(
            inputs.input_ids, 
            attention_mask=attention_mask,  # Pass the manually created attention mask
            max_length=50, 
            num_return_sequences=1,
            temperature=0.7,  # Lower temperature for more focused responses
            top_p=0.9,        # Use nucleus sampling to improve quality
            top_k=50,         # Limit to top 50 token choices for each step
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode the output and return the first sentence
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    return answer.split(".")[0]  # Return the first complete sentence
