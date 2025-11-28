import os, re
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from surrol.tasks.my_needle_pick_env import NeedlePickTrainEnv


# ============================================================
# Smoother
# ============================================================
def smooth_curve(values, window=200):
    if len(values) < window:
        return values
    out = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        out.append(np.mean(values[start:i+1]))
    return out


# ============================================================
# Live Training Plot Callback
# ============================================================
class LivePlotCallback(BaseCallback):

    def __init__(self, update_freq=10, verbose=0):
        super().__init__(verbose)

        self.update_freq = update_freq
        self.rewards = []
        self.contacts = []
        self.distances = []
        self.global_step = 0

        # ----- SETUP LIVE PLOTS -----
        plt.ion()
        self.fig, self.axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

        self.ax_reward, self.ax_contact, self.ax_dist = self.axs

        # reward plot
        self.reward_line_raw, = self.ax_reward.plot([], [], label="Reward", alpha=0.4)
        self.reward_line_smooth, = self.ax_reward.plot([], [], label="Smoothed (200)", linewidth=2)
        self.ax_reward.set_title("Reward per Step")
        self.ax_reward.grid(True)
        self.ax_reward.legend()

        # contact plot
        self.contact_line_raw, = self.ax_contact.plot([], [], label="Contact 0/1", drawstyle='steps-post')
        self.ax_contact.set_title("Contact Events")
        self.ax_contact.grid(True)
        self.ax_contact.legend()

        # distance plot
        self.dist_line_raw, = self.ax_dist.plot([], [], label="Distance")
        self.dist_line_smooth, = self.ax_dist.plot([], [], label="Smoothed (200)", linewidth=2)
        self.ax_dist.set_title("Distance to Goal")
        self.ax_dist.grid(True)
        self.ax_dist.legend()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


    # -------------------------------------------
    # Called after each environment step
    # -------------------------------------------
    def _on_step(self):

        info = self.locals["infos"][0]     # vec env → first env
        reward = float(self.locals["rewards"][0])

        # append reward
        self.rewards.append(reward)

        # contact = physics flag
        contact = 1 if info.get("is_gripping", False) else 0
        self.contacts.append(contact)

        # distance to goal
        obs = self.locals["new_obs"]
        ach = obs["achieved_goal"][0][:3]
        des = obs["desired_goal"][0][:3]
        dist = float(np.linalg.norm(ach - des) / 5.0)
        self.distances.append(dist)

        # update live plot
        if self.global_step % self.update_freq == 0:
            self.update_plot()

        self.global_step += 1

        return True


    # -------------------------------------------
    # Redraw live plot
    # -------------------------------------------
    def update_plot(self):

        xs = np.arange(len(self.rewards))

        # reward
        self.reward_line_raw.set_xdata(xs)
        self.reward_line_raw.set_ydata(self.rewards)

        r_smooth = smooth_curve(self.rewards, window=200)
        self.reward_line_smooth.set_xdata(xs)
        self.reward_line_smooth.set_ydata(r_smooth)

        # contact
        self.contact_line_raw.set_xdata(xs)
        self.contact_line_raw.set_ydata(self.contacts)

        # distance
        self.dist_line_raw.set_xdata(xs)
        self.dist_line_raw.set_ydata(self.distances)

        d_smooth = smooth_curve(self.distances, window=200)
        self.dist_line_smooth.set_xdata(xs)
        self.dist_line_smooth.set_ydata(d_smooth)

        # autoscale everything
        for ax in self.axs:
            ax.relim()
            ax.autoscale_view()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


# ============================================================
# Environment Factory
# ============================================================
def make_env():
    return NeedlePickTrainEnv(
        render_mode=None,
        reward_mode="curriculum",
        num_envs=1,
        traj_len=10240
    )


# ============================================================
# TRAIN SCRIPT WITH LIVE PLOT
# ============================================================
if __name__ == '__main__':

    # IMPORTANT: real-time plotting only works with 1 env
    num_envs = 1
    env = DummyVecEnv([make_env])

    log_dir = "/home/host-20-04/SurRol_venv/SurRoL/surrol/tasks/experiment/logs"
    base_path = "/home/host-20-04/SurRol_venv/SurRoL/surrol/tasks/experiment/models/"
    prefix = "needle_pick_ppo_gpu_"

    # -----------------------------------------------------------
    # Create new PPO model from scratch (OLD STYLE)
    # -----------------------------------------------------------
    # model = PPO(
    #     policy="MultiInputPolicy",
    #     env=env,
    #     verbose=1,
    #     device="cuda",
    #     n_steps=1024,           # your trajectory_len
    #     batch_size=256,
    #     learning_rate=3e-4,
    #     ent_coef=0.005,
    #     clip_range=0.2,
    #     n_epochs=3,
    #     tensorboard_log=log_dir
    # )

    agent_index = 9     # index model awal yang akan diload untuk re-train
    jumlah_retrain = 0
    # model = PPO.load(f"{base_path}{prefix}{agent_index}", env, device='cuda')
    model = PPO.load(f"/home/host-20-04/Downloads/needle_pick_ppo_gpu_9.zip", env, device='cuda', tensorboard_log=log_dir)

    # -----------------------------------------------------------
    # Live training diagnostic window
    # -----------------------------------------------------------
    callback = LivePlotCallback(update_freq=10)

    # -----------------------------------------------------------
    # Train with real-time plotting
    # -----------------------------------------------------------
    model.learn(total_timesteps=102400, progress_bar=True, callback=callback)

    # -----------------------------------------------------------
    # Determine next checkpoint number to save
    # -----------------------------------------------------------
    all_files = os.listdir(base_path)
    pattern = re.compile(rf"^{prefix}(\d+)$")
    indices = []

    for f in all_files:
        fname, _ = os.path.splitext(f)
        match = pattern.match(fname)
        if match:
            indices.append(int(match.group(1)))

    # Next model name
    if indices:
        last_idx = max(indices)
    else:
        last_idx = 0

    # Save new checkpoint (model after first training)
    next_idx = last_idx + 1
    save_path = f"{base_path}{prefix}{next_idx}"
    model.save(save_path)
    print(f"Saved → {save_path}")

    # -----------------------------------------------------------
    # Retrain the initialized model
    # -----------------------------------------------------------
    start_idx = next_idx          # <-- FIXED (instead of max(indices)+2)
    end_idx = start_idx + jumlah_retrain

    if jumlah_retrain > 0:
        for i in range(start_idx, end_idx):
            # Load previous checkpoint
            model_path = f"{base_path}{prefix}{i}"
            model = PPO.load(model_path, env, device='cuda')

            # Train
            model.learn(total_timesteps=102400, progress_bar=True, callback=LivePlotCallback(update_freq=10))

            # Save next checkpoint
            next_path = f"{base_path}{prefix}{i+1}"
            model.save(next_path)
            print(f"Model {i} → {i+1} saved to {next_path}")





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
#needle_pick_ppo_gpu_11 --> n_steps=1024, batch_size=32, learning_rate=1e-2, clip_range=0.1 (set step() to use TIP_LINK_INDEX, less-sparse reward shaping 1.000.000 ts's)       --> false logic
#needle_pick_ppo_gpu_12 --> n_steps=1024, batch_size=32, learning_rate=1e-2, clip_range=0.1 (transfer learn from needle_pick_ppo_gpu_11 + 1.000.000 timesteps)                  --> false logic
#needle_pick_ppo_gpu_13 --> n_steps=1024, batch_size=32, learning_rate=1e-2, clip_range=0.1 (set step() to use EEF_LINK_INDEX, less-sparse reward shaping 1.000.000 ts's)
#needle_pick_ppo_gpu_14 --> n_steps=1024, batch_size=32, learning_rate=1e-2, clip_range=0.1 (transfer learn from needle_pick_ppo_gpu_13 + 1.000.000 timesteps)
#needle_pick_ppo_gpu_15 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.005, clip_range=0.2 (updated less-sparse reward shape, hyp-param from chatgpt 1M t-steps)
#needle_pick_ppo_gpu_16 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.005, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_15 + 1.000.000 timesteps)
#needle_pick_ppo_gpu_17 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.005, clip_range=0.2 (angle based punishment erased, 1.000.000 timesteps)
#needle_pick_ppo_gpu_18 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.005, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_17 + 1.000.000 timesteps)
#needle_pick_ppo_gpu_19 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.005, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_18 + 1.000.000 timesteps)
#needle_pick_ppo_gpu_20 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.005, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_19 + 1.000.000 timesteps)  --> overfit
#needle_pick_ppo_gpu_21 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.005, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_20 + 1.000.000 timesteps)  --> overfit
#needle_pick_ppo_gpu_22 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.01, clip_range=0.2 (using sparse reward shaping, 1.000.000 timesteps)                   
#needle_pick_ppo_gpu_23 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.01, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_22 + 1.000.000 timesteps)
#needle_pick_ppo_gpu_24 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.01, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_23 + 1.000.000 timesteps)
#needle_pick_ppo_gpu_25 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.01, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_24 + 1.000.000 timesteps)
#needle_pick_ppo_gpu_26 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.005, clip_range=0.2 (using improved less-sparse reward shaping, 1.000.000 timesteps)    --> good approach
#needle_pick_ppo_gpu_27 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.005, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_26 + 1.000.000 timesteps)    
#needle_pick_ppo_gpu_28 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.005, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_27 + 1.000.000 timesteps)    
#needle_pick_ppo_gpu_29 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.005, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_28 + 1.000.000 timesteps)    
#needle_pick_ppo_gpu_30 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.005, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_29 + 1.000.000 timesteps)  
#needle_pick_ppo_gpu_31 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.005, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_30 + 1.000.000 timesteps)  --> overfit
#needle_pick_ppo_gpu_32 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.005, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_31 + 1.000.000 timesteps)  --> overfit
#needle_pick_ppo_gpu_33 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.005, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_32 + 1.000.000 timesteps)  --> overfit
#needle_pick_ppo_gpu_34 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.005, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_33 + 1.000.000 timesteps)  --> overfit, reward shape not smooth enough
#needle_pick_ppo_gpu_35 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.01, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_25 + 1.000.000 timesteps)  
#needle_pick_ppo_gpu_36 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.01, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_35 + 1.000.000 timesteps)
#needle_pick_ppo_gpu_37 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.01, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_36 + 1.000.000 timesteps)  
#needle_pick_ppo_gpu_38 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.005, clip_range=0.2 (corrected action space, using sparse reward shaping, 1M timesteps) --> ent_coef small 
#needle_pick_ppo_gpu_39 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.005, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_38 + 1.000.000 timesteps)
#needle_pick_ppo_gpu_40 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.005, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_39 + 1.000.000 timesteps)
#needle_pick_ppo_gpu_41 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.005, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_40 + 1.000.000 timesteps)
#needle_pick_ppo_gpu_42 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.005, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_41 + 1.000.000 timesteps)
#needle_pick_ppo_gpu_43 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.005, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_42 + 1.000.000 timesteps)  --> overfit, too exploitative
#needle_pick_ppo_gpu_44 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.005, clip_range=0.2 (corrected action space, using improved less-sparse reward shaping) 
#needle_pick_ppo_gpu_45 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.005, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_44 + 1.000.000 timesteps)  --> traped at local optimum
#needle_pick_ppo_gpu_46 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.005, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_45 + 1.000.000 timesteps)  --> traped at local optimum
#needle_pick_ppo_gpu_47 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.005, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_46 + 1.000.000 timesteps)  --> traped at local optimum
#needle_pick_ppo_gpu_48 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.005, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_47 + 1.000.000 timesteps)  --> traped at local optimum
#needle_pick_ppo_gpu_49 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.005, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_48 + 1.000.000 timesteps)  --> traped at local optimum
#needle_pick_ppo_gpu_50 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.005, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_49 + 1.000.000 timesteps)  --> traped at local optimum
#needle_pick_ppo_gpu_51 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.1, clip_range=0.2 (using sparse reward shaping, 1.000.000 timesteps)
#needle_pick_ppo_gpu_52 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.1, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_51 + 1.000.000 timesteps)
#needle_pick_ppo_gpu_53 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.1, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_52 + 1.000.000 timesteps)
#needle_pick_ppo_gpu_54 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.1, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_53 + 1.000.000 timesteps)
#needle_pick_ppo_gpu_55 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.1, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_54 + 1.000.000 timesteps)    --> underfit, too explorative
#needle_pick_ppo_gpu_56 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.2, clip_range=0.2 (using improved less-sparse reward shaping, 1.000.000 timesteps)
#needle_pick_ppo_gpu_57 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.2, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_56 + 1.000.000 timesteps)
#needle_pick_ppo_gpu_58 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.2, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_57 + 1.000.000 timesteps)
#needle_pick_ppo_gpu_59 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.2, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_58 + 1.000.000 timesteps)    --> underfit, too explorative, too high ent_coef
#needle_pick_ppo_gpu_60 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.005, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_43 + 1.000.000 timesteps)
#needle_pick_ppo_gpu_61 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.005, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_60 + 1.000.000 timesteps)  
#needle_pick_ppo_gpu_62 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.005, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_61 + 1.000.000 timesteps)  --> kinda overfit, try to explore more
#needle_pick_ppo_gpu_63 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.02, clip_range=0.2 (using improved less-sparse reward shaping, 1.000.000 timesteps)  
#needle_pick_ppo_gpu_64 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.02, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_63 + 1.000.000 timesteps)
#needle_pick_ppo_gpu_65 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.02, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_64 + 1.000.000 timesteps)  
#needle_pick_ppo_gpu_66 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.02, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_65 + 1.000.000 timesteps)
#needle_pick_ppo_gpu_67 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.02, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_66 + 1.000.000 timesteps)   --> good at tracking needle
#needle_pick_ppo_gpu_68 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.1, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_55 + 1.000.000 timesteps)
#needle_pick_ppo_gpu_69 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.1, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_68 + 1.000.000 timesteps)
#needle_pick_ppo_gpu_70 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.1, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_69 + 1.000.000 timesteps)    --> seems still try to learn
#needle_pick_ppo_gpu_71 --> n_steps=512, batch_size=64, learning_rate=2.5e-4, ent_coef=0.01, clip_range=0.2 (using curriculum learning reward shaping, 1.000.000 timesteps)     --> */ false logic, trained too long at Stage 1
#needle_pick_ppo_gpu_72 --> n_steps=512, batch_size=64, learning_rate=2.5e-4, ent_coef=0.01, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_71 + 1.000.000 timesteps)         relative_timesteps = timesteps / num_of_parallel_envs
#needle_pick_ppo_gpu_73 --> n_steps=512, batch_size=64, learning_rate=2.5e-4, ent_coef=0.01, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_72 + 1.000.000 timesteps)         forget to set relative_timesteps = timesteps / num_of_parallel_envs
#needle_pick_ppo_gpu_74 --> n_steps=512, batch_size=64, learning_rate=2.5e-4, ent_coef=0.01, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_73 + 1.000.000 timesteps)         relative_timesteps only exceeds 66.666 when using 15 parallel environments
#needle_pick_ppo_gpu_75 --> n_steps=512, batch_size=64, learning_rate=2.5e-4, ent_coef=0.01, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_74 + 1.000.000 timesteps)         training process never achieved Stage 2 /*
#needle_pick_ppo_gpu_76 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.02, clip_range=0.2 (using ajusted stage 1 curricullum learning reward shaping, +1M ts)
#needle_pick_ppo_gpu_77 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.02, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_76 + 1.000.000 timesteps)         
#needle_pick_ppo_gpu_78 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.02, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_77 + 1.000.000 timesteps)         
#needle_pick_ppo_gpu_79 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.02, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_78 + 1.000.000 timesteps)         
#needle_pick_ppo_gpu_80 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.02, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_79 + 1.000.000 timesteps)      
#needle_pick_ppo_gpu_81 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.1, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_70 + 1.000.000 timesteps)
#needle_pick_ppo_gpu_82 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.1, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_81 + 1.000.000 timesteps)
#needle_pick_ppo_gpu_83 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.1, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_82 + 1.000.000 timesteps)
#needle_pick_ppo_gpu_84 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.1, clip_range=0.2 (using adjusted distance threshold sparse reward shaping, +1M ts)
#needle_pick_ppo_gpu_85 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.1, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_84 + 1.000.000 timesteps)
#needle_pick_ppo_gpu_86 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.1, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_85 + 1.000.000 timesteps)
#needle_pick_ppo_gpu_87 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.1, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_86 + 1.000.000 timesteps)
#needle_pick_ppo_gpu_88 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.1, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_87 + 1.000.000 timesteps)
#needle_pick_ppo_gpu_89 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.1, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_88 + 1.000.000 timesteps)
#needle_pick_ppo_gpu_90 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.1, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_89 + 1.000.000 timesteps)
#needle_pick_ppo_gpu_91 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.02, clip_range=0.2 (using adjusted distance threshold less-sparse reward shaping, +1M)
#needle_pick_ppo_gpu_92 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.02, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_91 + 1.000.000 timesteps)
#needle_pick_ppo_gpu_93 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.02, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_92 + 1.000.000 timesteps)
#needle_pick_ppo_gpu_94 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.02, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_93 + 1.000.000 timesteps)
#needle_pick_ppo_gpu_95 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.02, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_94 + 1.000.000 timesteps)
#needle_pick_ppo_gpu_94 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.02, clip_range=0.2 (using adjusted distance threshold less-sparse reward shaping, +1M)
#needle_pick_ppo_gpu_95 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.02, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_94 + 1.000.000 timesteps)
#needle_pick_ppo_gpu_96 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.02, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_95 + 1.000.000 timesteps)
#needle_pick_ppo_gpu_97 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.02, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_96 + 1.000.000 timesteps)
#needle_pick_ppo_gpu_98 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.02, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_97 + 1.000.000 timesteps)
#needle_pick_ppo_gpu_99 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.02, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_98 + 1.000.000 timesteps)
#needle_pick_ppo_gpu_100 --> n_steps=2048, batch_size=64, learning_rate=3e-4, ent_coef=0.02, clip_range=0.2 (transfer learn from needle_pick_ppo_gpu_99 + 1.000.000 timesteps)
