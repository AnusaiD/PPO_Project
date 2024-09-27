import gym
import pandas as pd

class TextEnv(gym.Env):
    """Custom environment for training the PPO model."""
    def __init__(self, dataset):
        super(TextEnv, self).__init__()

        if len(dataset) == 0:
            raise ValueError("The dataset is empty, cannot initialize environment.")

        self.dataset = dataset
        self.action_space = gym.spaces.Discrete(len(dataset))
        self.observation_space = gym.spaces.Discrete(len(dataset))

    def reset(self):
        """Reset the environment."""
        return self.dataset.sample().prompt

    def step(self, action):
        row = self.dataset.iloc[action]
        prompt, answer, correct_answer = row['prompt'], row['answer'], row['correct_answer']
        reward = 1 if answer == correct_answer else 0
        done = True
        return answer, reward, done, {}
    
def train_ppo_model(env, timesteps=10000):
    """Train the PPO model on the custom environment."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    vec_env = DummyVecEnv([lambda: env])  # Ensure the environment is wrapped
    ppo_model = PPO("MlpPolicy", vec_env, verbose=1)
    ppo_model.learn(total_timesteps=timesteps)
    ppo_model.save("./models/ppo_model")
