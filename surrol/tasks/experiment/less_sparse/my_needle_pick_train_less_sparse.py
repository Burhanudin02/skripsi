# +
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from surrol.tasks.my_needle_pick_env import NeedlePickTrainEnv
import os, re

# DEFINE THE NUMBER OF PARALLEL ENVIRONMENTS YOU WANT!  
num_envs = 10
trajectory_len = 10240

def make_env():
    return NeedlePickTrainEnv(render_mode=None, reward_mode="less_sparse", num_envs=num_envs, traj_len = trajectory_len)

if __name__ == '__main__':
    num_envs = num_envs  # Adjust the number of parallel environments you want

    # Create parallel environments
    env = SubprocVecEnv([make_env for _ in range(num_envs)])

    # Create a directory to save the TensorBoard logs
    log_dir = "/data/skripsi/surrol/tasks/experiment/less_sparse/logs"

    base_path = "/data/skripsi/surrol/tasks/experiment/less_sparse/models/"
    prefix = "needle_pick_ppo_gpu_"

    # Cari semua file/folder dengan prefix
    all_files = os.listdir(base_path)
    
    # Filter hanya yang sesuai pola prefix
    pattern = re.compile(rf"^{prefix}(\d+)$")
    indices = []
    
    for f in all_files:
        fname, _ = os.path.splitext(f)
        match = pattern.match(fname)
        print(f, match)
        if match:
            indices.append(int(match.group(1)))
    
    if indices:
        last_idx = max(indices)
        print(f"Indeks terakhir: {last_idx}")
    else:
        raise FileNotFoundError(f"Tidak ada file dengan prefix {prefix} di {base_path}")
      
    # First initialization of PPO model parameter

#     agent_index = 2     # index model awal yang akan diload untuk re-train
    jumlah_retrain = 19
#     model = PPO.load(f"{base_path}{prefix}{agent_index}", env, device='cuda')

    # model = PPO('MultiInputPolicy', env, verbose=1, device='cuda', n_steps=512, batch_size=64, learning_rate=2.5e-4, ent_coef=0.01, clip_range=0.2, tensorboard_log=log_dir)
    model = PPO('MultiInputPolicy', env, verbose=1, device='cuda', n_steps=trajectory_len, 
                batch_size=256, learning_rate=3e-4, ent_coef=0.005, clip_range=0.2,
                n_epochs=3, tensorboard_log=log_dir
                )
    
    # Train the model and use TensorBoard callback
    model.learn(total_timesteps=1024000, progress_bar=True)

    # Save the trained model
    model.save(f"{base_path}{prefix}{last_idx+1}")

    # Retrain the initialized model
    start_idx = last_idx+1   # model terakhir yang ada
    end_idx = start_idx + jumlah_retrain    # model terakhir yang akan disimpan

    if not start_idx == end_idx:
        for i in range(start_idx, end_idx):
            # Load model sebelumnya
            model_path = f"{base_path}{prefix}{i}"
            model = PPO.load(model_path, env, device='cuda')
        
            # Training
            model.learn(total_timesteps=1024000, progress_bar=True)
        
            # Save dengan nomor urut berikutnya
            save_path = f"{base_path}{prefix}{i+1}"
            model.save(save_path)
        
            print(f"Model {i} -> {i+1} selesai disimpan di {save_path}")

