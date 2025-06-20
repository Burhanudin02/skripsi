from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from my_needle_pick_env import NeedlePickTrainEnv  # Your environment

def make_env():
    return NeedlePickTrainEnv(render_mode=None)

if __name__ == '__main__':
    num_envs = 12  # Adjust the number of parallel environments you want

    # Create parallel environments
    env = SubprocVecEnv([make_env for _ in range(num_envs)])

    # Create a directory to save the TensorBoard logs
    log_dir = "/home/host-20-04/SurRol_venv/SurRoL/surrol/tasks/logs"

    # Initialize PPO model with CPU
    # model = PPO('MlpPolicy', env, verbose=1, device='cpu', n_steps=1024, batch_size=32, learning_rate=1e-8, clip_range=0.1, tensorboard_log=log_dir)
    model_path = "/home/host-20-04/SurRol_venv/SurRoL/surrol/tasks/models/needle_pick_ppo_gpu_9"
    model = PPO.load(model_path, env, device='cpu')

    # Train the model and use TensorBoard callback
    model.learn(total_timesteps=1000000, progress_bar=True)

    # Save the trained model
    model.save("/home/host-20-04/SurRol_venv/SurRoL/surrol/tasks/models/needle_pick_ppo_gpu_10")

    # If you want to load TensorBoard logs and view them later, open the terminal and run:
    # tensorboard --logdir /home/host-20-04/SurRol_venv/SurRoL/surrol/tasks/logs

#needle_pick_ppo_gpu --> n_steps=1024, batch_size=32, learning_rate=1e-14, clip_range=0.1
#needle_pick_ppo_gpu_2 --> n_steps=2048, batch_size=32, learning_rate=1e-5, clip_range=0.1
#needle_pick_ppo_gpu_3 --> n_steps=2048, batch_size=32, learning_rate=1e-10, clip_range=0.1
#needle_pick_ppo_gpu_4 --> n_steps=1024, batch_size=32, learning_rate=1e-8, clip_range=0.1 --> mulai dari sini versi log tensorboard PPO_3
#needle_pick_ppo_gpu_5 --> n_steps=1024, batch_size=32, learning_rate=1e-8, clip_range=0.1 (improved definition of done)
#needle_pick_ppo_gpu_6 --> n_steps=1024, batch_size=32, learning_rate=1e-8, clip_range=0.1 (improved definition of done + reset() method)
#needle_pick_ppo_gpu_7 --> n_steps=1024, batch_size=32, learning_rate=1e-8, clip_range=0.1 (improved definiton of done + reward shaping)
#needle_pick_ppo_gpu_8 --> n_steps=1024, batch_size=32, learning_rate=1e-8, clip_range=0.1 (timesteps 1.000.000)
#needle_pick_ppo_gpu_9 --> n_steps=1024, batch_size=32, learning_rate=1e-8, clip_range=0.1 (transfer learn from needle_pick_ppo_gpu_8 + 1.000.000 timesteps)