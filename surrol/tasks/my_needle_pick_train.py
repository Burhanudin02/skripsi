from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from my_needle_pick_env import NeedlePickTrainEnv  # Your environment

def make_env():
    return NeedlePickTrainEnv(render_mode=None)

# +
if __name__ == '__main__':
    num_envs = 15  # Adjust the number of parallel environments you want

    # Create parallel environments
    env = SubprocVecEnv([make_env for _ in range(num_envs)])

    # Create a directory to save the TensorBoard logs
    log_dir = "/data/skripsi/surrol/tasks/logs"

    # Initialize PPO model with CPU
#     model = PPO('MlpPolicy', env, verbose=1, device='cpu', n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.02, clip_range=0.2, tensorboard_log=log_dir)
#     model_path = "/data/skripsi/surrol/tasks/models/needle_pick_ppo_gpu_70"
#     model = PPO.load(model_path, env, device='cpu')

#     # Train the model and use TensorBoard callback
#     model.learn(total_timesteps=1000000, progress_bar=True)

#     # Save the trained model
#     model.save("/data/skripsi/surrol/tasks/models/needle_pick_ppo_gpu_91")

#     # addition: otw 2 jt
#     model_path = "/data/skripsi/surrol/tasks/models/needle_pick_ppo_gpu_91"
#     model = PPO.load(model_path, env, device='cpu')

#     model.learn(total_timesteps=1000000, progress_bar=True)
#     model.save("/data/skripsi/surrol/tasks/models/needle_pick_ppo_gpu_92")

#     # addition: otw 3 jt
#     model_path = "/data/skripsi/surrol/tasks/models/needle_pick_ppo_gpu_92"
#     model = PPO.load(model_path, env, device='cpu')

#     model.learn(total_timesteps=1000000, progress_bar=True)
#     model.save("/data/skripsi/surrol/tasks/models/needle_pick_ppo_gpu_93")

#     # addition: otw 4 jt
#     model_path = "/data/skripsi/surrol/tasks/models/needle_pick_ppo_gpu_93"
#     model = PPO.load(model_path, env, device='cpu')

#     model.learn(total_timesteps=1000000, progress_bar=True)
#     model.save("/data/skripsi/surrol/tasks/models/needle_pick_ppo_gpu_94")

    # addition: otw 5 jt
    model_path = "/data/skripsi/surrol/tasks/models/needle_pick_ppo_gpu_94"
    model = PPO.load(model_path, env, device='cpu')

    model.learn(total_timesteps=1000000, progress_bar=True)
    model.save("/data/skripsi/surrol/tasks/models/needle_pick_ppo_gpu_95")

#     # addition: otw 6 jt
#     model_path = "/data/skripsi/surrol/tasks/models/needle_pick_ppo_gpu_88"
#     model = PPO.load(model_path, env, device='cpu')

#     model.learn(total_timesteps=1000000, progress_bar=True)
#     model.save("/data/skripsi/surrol/tasks/models/needle_pick_ppo_gpu_89")

#     # addition: otw 7 jt
#     model_path = "/data/skripsi/surrol/tasks/models/needle_pick_ppo_gpu_89"
#     model = PPO.load(model_path, env, device='cpu')

#     model.learn(total_timesteps=1000000, progress_bar=True)
#     model.save("/data/skripsi/surrol/tasks/models/needle_pick_ppo_gpu_90")

    # If you want to load TensorBoard logs and view them later, open the terminal and run:
    # tensorboard --logdir /home/host-20-04/SurRol_venv/SurRoL/surrol/tasks/logs
# -

# needle_pick_ppo_gpu --> n_steps=1024, batch_size=32, learning_rate=1e-14, clip_range=0.1
# needle_pick_ppo_gpu_2 --> n_steps=2048, batch_size=32, learning_rate=1e-5, clip_range=0.1
# needle_pick_ppo_gpu_3 --> n_steps=2048, batch_size=32, learning_rate=1e-10, clip_range=0.1
# needle_pick_ppo_gpu_4 --> n_steps=1024, batch_size=32, learning_rate=1e-8, clip_range=0.1 --> mulai dari sini versi log tensorboard PPO_4
# needle_pick_ppo_gpu_5 --> n_steps=1024, batch_size=32, learning_rate=1e-8, clip_range=0.1 (improved definition of done)
# needle_pick_ppo_gpu_6 --> n_steps=1024, batch_size=32, learning_rate=1e-8, clip_range=0.1 (improved definition of done + reset() method)
# needle_pick_ppo_gpu_7 --> n_steps=1024, batch_size=32, learning_rate=1e-8, clip_range=0.1 (improved definiton of done + reward shaping)
# needle_pick_ppo_gpu_8 --> n_steps=1024, batch_size=32, learning_rate=1e-8, clip_range=0.1 (timesteps 1.000.000)
# needle_pick_ppo_gpu_9 --> n_steps=1024, batch_size=32, learning_rate=1e-8, clip_range=0.1 (transfer learn from needle_pick_ppo_gpu_8 + 1.000.000 timesteps)
# needle_pick_ppo_gpu_10 --> n_steps=1024, batch_size=32, learning_rate=1e-8, clip_range=0.1 (transfer learn from needle_pick_ppo_gpu_9 + 1.000.000 timesteps)
# needle_pick_ppo_gpu_11 --> n_steps=1024, batch_size=32, learning_rate=1e-2, clip_range=0.1 (set step() to use TIP_LINK_INDEX, less-sparse reward shaping 1.000.000 ts's)       --> false logic
# needle_pick_ppo_gpu_12 --> n_steps=1024, batch_size=32, learning_rate=1e-2, clip_range=0.1 (transfer learn from needle_pick_ppo_gpu_11 + 1.000.000 timesteps)                  --> false logic
# needle_pick_ppo_gpu_13 --> n_steps=1024, batch_size=32, learning_rate=1e-2, clip_range=0.1 (set step() to use EEF_LINK_INDEX, less-sparse reward shaping 1.000.000 ts's)
# needle_pick_ppo_gpu_14 --> n_steps=1024, batch_size=32, learning_rate=1e-2, clip_range=0.1 (transfer learn from needle_pick_ppo_gpu_13 + 1.000.000 timesteps)
# needle_pick_ppo_gpu_15 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.005, clip_range=0.2 (updated less-sparse reward shape, hyp-param from chatgpt 1M t-steps)
# needle_pick_ppo_gpu_16 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.005, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_15 + 1.000.000 timesteps)
# needle_pick_ppo_gpu_17 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.005, clip_range=0.2 (angle based punishment erased, 1.000.000 timesteps)
# needle_pick_ppo_gpu_18 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.005, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_17 + 1.000.000 timesteps)
# needle_pick_ppo_gpu_19 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.005, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_18 + 1.000.000 timesteps)
# needle_pick_ppo_gpu_20 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.005, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_19 + 1.000.000 timesteps)  --> overfit
# needle_pick_ppo_gpu_21 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.005, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_20 + 1.000.000 timesteps)  --> overfit
# needle_pick_ppo_gpu_22 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.01, clip_range=0.2 (using sparse reward shaping, 1.000.000 timesteps)                   
# needle_pick_ppo_gpu_23 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.01, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_22 + 1.000.000 timesteps)
# needle_pick_ppo_gpu_24 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.01, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_23 + 1.000.000 timesteps)
# needle_pick_ppo_gpu_25 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.01, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_24 + 1.000.000 timesteps)
# needle_pick_ppo_gpu_26 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.005, clip_range=0.2 (using improved less-sparse reward shaping, 1.000.000 timesteps)    --> good approach
# needle_pick_ppo_gpu_27 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.005, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_26 + 1.000.000 timesteps)    
# needle_pick_ppo_gpu_28 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.005, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_27 + 1.000.000 timesteps)    
# needle_pick_ppo_gpu_29 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.005, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_28 + 1.000.000 timesteps)    
# needle_pick_ppo_gpu_30 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.005, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_29 + 1.000.000 timesteps)  
# needle_pick_ppo_gpu_31 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.005, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_30 + 1.000.000 timesteps)  --> overfit
# needle_pick_ppo_gpu_32 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.005, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_31 + 1.000.000 timesteps)  --> overfit
# needle_pick_ppo_gpu_33 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.005, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_32 + 1.000.000 timesteps)  --> overfit
# needle_pick_ppo_gpu_34 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.005, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_33 + 1.000.000 timesteps)  --> overfit, reward shape not smooth enough
# needle_pick_ppo_gpu_35 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.01, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_25 + 1.000.000 timesteps)  
# needle_pick_ppo_gpu_36 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.01, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_35 + 1.000.000 timesteps)
# needle_pick_ppo_gpu_37 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.01, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_36 + 1.000.000 timesteps)  
# needle_pick_ppo_gpu_38 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.005, clip_range=0.2 (corrected action space, using sparse reward shaping, 1M timesteps) --> ent_coef small 
# needle_pick_ppo_gpu_39 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.005, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_38 + 1.000.000 timesteps)
# needle_pick_ppo_gpu_40 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.005, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_39 + 1.000.000 timesteps)
# needle_pick_ppo_gpu_41 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.005, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_40 + 1.000.000 timesteps)
# needle_pick_ppo_gpu_42 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.005, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_41 + 1.000.000 timesteps)
# needle_pick_ppo_gpu_43 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.005, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_42 + 1.000.000 timesteps)  --> overfit, too exploitative
# needle_pick_ppo_gpu_44 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.005, clip_range=0.2 (corrected action space, using improved less-sparse reward shaping) 
# needle_pick_ppo_gpu_45 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.005, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_44 + 1.000.000 timesteps)  --> traped at local optimum
# needle_pick_ppo_gpu_46 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.005, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_45 + 1.000.000 timesteps)  --> traped at local optimum
# needle_pick_ppo_gpu_47 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.005, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_46 + 1.000.000 timesteps)  --> traped at local optimum
# needle_pick_ppo_gpu_48 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.005, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_47 + 1.000.000 timesteps)  --> traped at local optimum
# needle_pick_ppo_gpu_49 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.005, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_48 + 1.000.000 timesteps)  --> traped at local optimum
# needle_pick_ppo_gpu_50 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.005, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_49 + 1.000.000 timesteps)  --> traped at local optimum
# needle_pick_ppo_gpu_51 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.1, clip_range=0.2 (using sparse reward shaping, 1.000.000 timesteps)
# needle_pick_ppo_gpu_52 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.1, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_51 + 1.000.000 timesteps)
# needle_pick_ppo_gpu_53 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.1, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_52 + 1.000.000 timesteps)
# needle_pick_ppo_gpu_54 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.1, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_53 + 1.000.000 timesteps)
# needle_pick_ppo_gpu_55 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.1, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_54 + 1.000.000 timesteps)    --> underfit, too explorative
# needle_pick_ppo_gpu_56 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.2, clip_range=0.2 (using improved less-sparse reward shaping, 1.000.000 timesteps)
# needle_pick_ppo_gpu_57 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.2, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_56 + 1.000.000 timesteps)
# needle_pick_ppo_gpu_58 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.2, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_57 + 1.000.000 timesteps)
# needle_pick_ppo_gpu_59 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.2, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_58 + 1.000.000 timesteps)    --> underfit, too explorative, too high ent_coef
# needle_pick_ppo_gpu_60 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.005, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_43 + 1.000.000 timesteps)
# needle_pick_ppo_gpu_61 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.005, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_60 + 1.000.000 timesteps)  
# needle_pick_ppo_gpu_62 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.005, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_61 + 1.000.000 timesteps)  --> kinda overfit, try to explore more
# needle_pick_ppo_gpu_63 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.02, clip_range=0.2 (using improved less-sparse reward shaping, 1.000.000 timesteps)  
# needle_pick_ppo_gpu_64 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.02, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_63 + 1.000.000 timesteps)
# needle_pick_ppo_gpu_65 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.02, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_64 + 1.000.000 timesteps)  
# needle_pick_ppo_gpu_66 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.02, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_65 + 1.000.000 timesteps)
# needle_pick_ppo_gpu_67 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.02, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_66 + 1.000.000 timesteps)   --> good at tracking needle
# needle_pick_ppo_gpu_68 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.1, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_55 + 1.000.000 timesteps)
# needle_pick_ppo_gpu_69 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.1, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_68 + 1.000.000 timesteps)
# needle_pick_ppo_gpu_70 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.1, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_69 + 1.000.000 timesteps)    --> seems still try to learn

