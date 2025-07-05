import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from my_needle_pick_env import NeedlePickTrainEnv  # Use relative import if this file is in the same package


def evaluate_model(model_path, num_episodes=10):
    """
    Evaluates a trained model in the environment for a given number of episodes.
    """
    # Create the environment
    def make_env():
        return NeedlePickTrainEnv(render_mode='human')

    # Create a vectorized environment for evaluation
    env = DummyVecEnv([make_env])

    # Load the trained PPO model
    model = PPO.load(model_path)

    total_rewards = []

    for episode in range(num_episodes):
        obs = env.reset()  # Reset the environment to start a new episode
        done = False
        total_reward = 0

        while not done:
            # Take action based on the current observation
            action, _states = model.predict(obs, deterministic=False)
            obs, reward, done, info = env.step(action)  # Step the environment forward
            
            total_reward += reward  # Accumulate the reward for this episode
  
            print(f"Total reward: {total_reward}")

            # Optionally, you can render the environment to visualize the robot's performance
            # env.render()

        total_rewards.append(total_reward)
        print(f"Episode {episode + 1} - Total Reward: {total_reward}")

    # Calculate and print the average reward
    avg_reward = np.mean(total_rewards)
    print(f"Average Reward over {num_episodes} episodes: {avg_reward}")
    return avg_reward


if __name__ == "__main__":
    # Define the path where the trained model is saved
    model_path = "/home/host-20-04/SurRol_venv/SurRoL/surrol/tasks/models/needle_pick_ppo_gpu_45"  # Update this path to your model's location
    
    # Evaluate the trained model for 3 episodes
    evaluate_model(model_path, num_episodes=3)
