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
    model = PPO('MlpPolicy', env, verbose=1, device='cpu', n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.005, clip_range=0.2, tensorboard_log=log_dir)
    # model_path = "/home/host-20-04/SurRol_venv/SurRoL/surrol/tasks/models/needle_pick_ppo_gpu_9"
    # model = PPO.load(model_path, env, device='cpu')

    # Train the model and use TensorBoard callback
    model.learn(total_timesteps=1000000, progress_bar=True)

    # Save the trained model
    model.save("/home/host-20-04/SurRol_venv/SurRoL/surrol/tasks/models/needle_pick_ppo_gpu_15")

    # addition: otw 2 jt
    model_path = "/home/host-20-04/SurRol_venv/SurRoL/surrol/tasks/models/needle_pick_ppo_gpu_15"
    model = PPO.load(model_path, env, device='cpu')

    model.learn(total_timesteps=1000000, progress_bar=True)
    model.save("/home/host-20-04/SurRol_venv/SurRoL/surrol/tasks/models/needle_pick_ppo_gpu_16")

    # If you want to load TensorBoard logs and view them later, open the terminal and run:
    # tensorboard --logdir /home/host-20-04/SurRol_venv/SurRoL/surrol/tasks/logs

#needle_pick_ppo_gpu --> n_steps=1024, batch_size=32, learning_rate=1e-14, clip_range=0.1
#needle_pick_ppo_gpu_2 --> n_steps=2048, batch_size=32, learning_rate=1e-5, clip_range=0.1
#needle_pick_ppo_gpu_3 --> n_steps=2048, batch_size=32, learning_rate=1e-10, clip_range=0.1
#needle_pick_ppo_gpu_4 --> n_steps=1024, batch_size=32, learning_rate=1e-8, clip_range=0.1 --> mulai dari sini versi log tensorboard PPO_4
#needle_pick_ppo_gpu_5 --> n_steps=1024, batch_size=32, learning_rate=1e-8, clip_range=0.1 (improved definition of done)
#needle_pick_ppo_gpu_6 --> n_steps=1024, batch_size=32, learning_rate=1e-8, clip_range=0.1 (improved definition of done + reset() method)
#needle_pick_ppo_gpu_7 --> n_steps=1024, batch_size=32, learning_rate=1e-8, clip_range=0.1 (improved definiton of done + reward shaping)
#needle_pick_ppo_gpu_8 --> n_steps=1024, batch_size=32, learning_rate=1e-8, clip_range=0.1 (timesteps 1.000.000)
#needle_pick_ppo_gpu_9 --> n_steps=1024, batch_size=32, learning_rate=1e-8, clip_range=0.1 (transfer learn from needle_pick_ppo_gpu_8 + 1.000.000 timesteps)
#needle_pick_ppo_gpu_10 --> n_steps=1024, batch_size=32, learning_rate=1e-8, clip_range=0.1 (transfer learn from needle_pick_ppo_gpu_9 + 1.000.000 timesteps)
#needle_pick_ppo_gpu_11 --> n_steps=1024, batch_size=32, learning_rate=1e-2, clip_range=0.1 (set step() to use TIP_LINK_INDEX, less-sparse reward shaping 1.000.000 ts's)
#needle_pick_ppo_gpu_12 --> n_steps=1024, batch_size=32, learning_rate=1e-2, clip_range=0.1 (transfer learn from needle_pick_ppo_gpu_11 + 1.000.000 timesteps)
#needle_pick_ppo_gpu_13 --> n_steps=1024, batch_size=32, learning_rate=1e-2, clip_range=0.1 (set step() to use EEF_LINK_INDEX, less-sparse reward shaping 1.000.000 ts's)
#needle_pick_ppo_gpu_14 --> n_steps=1024, batch_size=32, learning_rate=1e-2, clip_range=0.1 (transfer learn from needle_pick_ppo_gpu_13 + 1.000.000 timesteps)
#needle_pick_ppo_gpu_15 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.005, clip_range=0.2 (updated less-sparse reward shape, hyp-param from chatgpt 1M t-steps)
#needle_pick_ppo_gpu_16 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.005, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_15 + 1.000.000 timesteps)
#needle_pick_ppo_gpu_17 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.005, clip_range=0.2 (angle based punishment erased, 1.000.000 timesteps)
#needle_pick_ppo_gpu_18 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.005, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_17 + 1.000.000 timesteps)
#needle_pick_ppo_gpu_19 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.005, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_18 + 1.000.000 timesteps)
#needle_pick_ppo_gpu_20 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.005, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_19 + 1.000.000 timesteps)
