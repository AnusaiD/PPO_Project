import os
import uuid
import pandas as pd
from reward_pipeline import calculate_reward
from feedback_pipeline import filter_low_reward, generate_feedback_with_ai
from ppo_train_pipeline import TextEnv, train_ppo_model
from prometheus_evaluation_pipeline import evaluate_with_prometheus

# Paths for data and model storage
DATA_DIR = "./data"
MODELS_DIR = "./models"
DEBUG_DIR = "./debug"

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)

# Helper function to check if a file exists
def file_exists(step_name):
    """Checks if a file for a particular step exists in the debug folder."""
    for filename in os.listdir(DEBUG_DIR):
        if step_name in filename:
            return os.path.join(DEBUG_DIR, filename)
    return None

# Debug function to write intermediate data with UUID
def write_to_debug_folder(data, step_name):
    """Writes data to a debug folder with a unique UUID."""
    unique_id = str(uuid.uuid4())
    filename = os.path.join(DEBUG_DIR, f"{step_name}_{unique_id}.csv")
    data.to_csv(filename, index=False)
    print(f"Debug: Data for {step_name} saved to {filename}.")
    return filename  # Return the filename

# 1. Reward Model Function
def run_reward_model(data):
    print("Running Reward Model...")

    # Check if the reward data already exists
    reward_file = file_exists("reward")
    if reward_file:
        print(f"Found existing reward dataset at: {reward_file}")
        return pd.read_csv(reward_file)
    
    # Apply the reward calculation
    data['reward_score'] = data.apply(lambda row: calculate_reward(row['prompt'], row['answer']), axis=1)
    
    # Save the dataset with reward scores to the debug folder
    reward_filename = write_to_debug_folder(data, "reward")
    print(f"Reward Model completed. Updated dataset saved with reward scores to: {reward_filename}")
    return data

# 2. Feedback Function
def get_feedback(prompt):
    """Generate feedback for low-reward answers using AI (GPT-2)."""
    return generate_feedback_with_ai(prompt)

def run_feedback_function(data):
    print("Running Feedback Function...")

    # Check if feedback data already exists
    feedback_file = file_exists("feedback")
    if feedback_file:
        print(f"Found existing feedback dataset at: {feedback_file}")
        return pd.read_csv(feedback_file)
    
    # Filter low-reward answers
    low_reward_data = filter_low_reward(data)
    
    if low_reward_data.empty:
        raise ValueError("No low-reward data found after filtering. Please check the reward scores.")
    
    # Collect feedback for each low-reward answer
    low_reward_data['correct_answer'] = low_reward_data['prompt'].apply(get_feedback)

    # Save the dataset with feedback to the debug folder
    feedback_filename = write_to_debug_folder(low_reward_data, "feedback")
    print(f"Feedback Function completed. Updated dataset saved with feedback to: {feedback_filename}")
    return low_reward_data

# 3. PPO Training Function
def run_ppo_training(feedback_data):
    print("Running PPO Training Function...")
    
    # Train the PPO model on the feedback dataset
    env = TextEnv(feedback_data)
    train_ppo_model(env, timesteps=10000)
    
    print("PPO Training completed. Model saved.")

# 4. Prometheus Evaluation
def run_prometheus_evaluation(feedback_data):
    print("Running Prometheus Evaluation...")

    # Check if evaluation data already exists
    evaluation_file = file_exists("evaluation")
    if evaluation_file:
        print(f"Found existing evaluation dataset at: {evaluation_file}")
        return pd.read_csv(evaluation_file)
    
    # Evaluate the trained PPO model using Prometheus
    feedback_data['prometheus_score'] = feedback_data.apply(lambda row: evaluate_with_prometheus(row['prompt'], row['answer']), axis=1)
    
    # Calculate the overall Prometheus score
    average_score = feedback_data['prometheus_score'].mean()
    evaluation_filename = write_to_debug_folder(feedback_data, "evaluation")
    print(f"Prometheus Evaluation completed. Average score: {average_score}. Debug data saved to: {evaluation_filename}")
    
    return average_score

# Function that implements the entire pipeline
def run_pipeline(data):
    # Step 1: Reward Model
    reward_data = run_reward_model(data)

    # Step 2: Feedback Function
    feedback_data = run_feedback_function(reward_data)
    
    # Debug: Print feedback_data shape
    print("Feedback data shape:", feedback_data.shape)
    print("Feedback data preview:", feedback_data.head())

    # Step 3: PPO Training
    run_ppo_training(feedback_data)

    # Step 4: Prometheus Evaluation
    final_score = run_prometheus_evaluation(feedback_data)
    
    print("Pipeline completed successfully.")
    print(f"Final Prometheus evaluation score: {final_score}")

# Main function
def main():
    print("Reading dataset...")
    
    # Load the dataset from the data folder
    data = pd.read_csv(f"{DATA_DIR}/dataset.csv")
    
    # Run the pipeline with the dataset
    run_pipeline(data)

if __name__ == "__main__":
    main()
