import numpy as np
import csv
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from surrol.tasks.my_needle_pick_env import NeedlePickTrainEnv


# -----------------------------------------------------------
# Moving average smoother
# -----------------------------------------------------------
def smooth_curve(values, window=50):
    if len(values) < window:
        return values.copy()
    out = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        out.append(np.mean(values[start:i+1]))
    return out


# -----------------------------------------------------------
# Contact episodes analysis
# -----------------------------------------------------------
def analyze_grip_episodes(contact_series):
    """Return list of contiguous grip episode lengths (in steps)."""
    durations = []
    current = 0
    for v in contact_series:
        if v:
            current += 1
        else:
            if current > 0:
                durations.append(current)
                current = 0
    if current > 0:
        durations.append(current)
    return durations


# -----------------------------------------------------------
# MAIN EVALUATION WITH REAL-TIME DIAGNOSTIC PLOTS
# -----------------------------------------------------------
def evaluate_model(model_path, num_episodes=10, max_steps=None, csv_path=None):
    if csv_path is None:
        # csv_path = "/home/host-20-04/SurRol_venv/SurRoL/surrol/tasks/test_reward_log/step_rewards_full.csv"
        # csv_path = "/home/host-20-04/SurRol_venv/SurRoL/surrol/tasks/test_reward_log/step_rewards_curriculum.csv"
        csv_path = "/home/host-20-04/SurRol_venv/SurRoL/surrol/tasks/test_reward_log/step_rewards_curriculum_new.csv"
        # csv_path = "/home/host-20-04/SurRol_venv/SurRoL/surrol/tasks/test_reward_log/step_rewards_sparse.csv"
        # csv_path = "/home/host-20-04/SurRol_venv/SurRoL/surrol/tasks/test_reward_log/step_rewards_less_sparse.csv"
        # csv_path = "/home/host-20-04/SurRol_venv/SurRoL/surrol/tasks/test_reward_log/step_rewards_less_sparse_best.csv"     # current best is train number 12
    env = DummyVecEnv([make_env])
    model = PPO.load(model_path)

    total_rewards = []
    step_rewards = []
    contact_values = []
    distance_values = []

    global_step = 0

    # Prepare CSV file header and open for streaming write
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["step", "reward", "contact", "distance"])

    # -----------------------------------------------------------
    # Setup live diagnostic plot (3 subplots)
    # -----------------------------------------------------------
    plt.ion()
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    ax_reward, ax_contact, ax_dist = axs

    # Reward plot handles
    raw_reward_line, = ax_reward.plot([], [], label="Raw Reward", alpha=0.4)
    smooth_reward_line, = ax_reward.plot([], [], label="Reward Smoothed (MA-50)", linewidth=2)

    # Contact Consistency curve (cumulative contact ratio)
    consistency_line, = ax_contact.plot([], [], label="Contact Consistency Score", linewidth=2)

    # Dist-to-goal plot handles
    raw_dist_line, = ax_dist.plot([], [], label="Raw Dist", alpha=0.4)
    smooth_dist_line, = ax_dist.plot([], [], label="Dist Smoothed (MA-50)", linewidth=2)

    # Titles & formatting
    ax_reward.set_title("Reward per Step (Live)")
    ax_contact.set_title("Contact Consistency Score (0–1)")
    ax_dist.set_title("Distance to Goal")
    for ax in axs:
        ax.grid(True)
        ax.legend()

    fig.canvas.draw()
    fig.canvas.flush_events()

    # -----------------------------------------------------------
    # Reset flag for keypress "R"
    # -----------------------------------------------------------
    reset_flag = {"reset": False}

    def on_key_press(event):
        if event.key == 'r':
            print("\n[R] Pressed — scheduling environment reset...")
            reset_flag["reset"] = True

    fig.canvas.mpl_connect('key_press_event', on_key_press)

    # -----------------------------------------------------------
    # Plot updater (called every 10 steps)
    # -----------------------------------------------------------
    SMOOTH_WINDOW = 200

    def update_plot():
        # Reward
        raw_reward_line.set_xdata(np.arange(len(step_rewards)))
        raw_reward_line.set_ydata(step_rewards)
        smooth_reward_line.set_xdata(np.arange(len(step_rewards)))
        smooth_reward_line.set_ydata(smooth_curve(step_rewards, SMOOTH_WINDOW))

        # Contact Consistency Curve
        if len(contact_values) > 0:
            cumulative = np.cumsum(contact_values)
            consistency = cumulative / np.arange(1, len(contact_values) + 1)
            consistency_line.set_xdata(np.arange(len(consistency)))
            consistency_line.set_ydata(consistency)


        # Distance
        raw_dist_line.set_xdata(np.arange(len(distance_values)))
        raw_dist_line.set_ydata(distance_values)
        smooth_dist_line.set_xdata(np.arange(len(distance_values)))
        smooth_dist_line.set_ydata(smooth_curve(distance_values, SMOOTH_WINDOW))

        # Autoscale each subplot
        for ax in axs:
            ax.relim()
            ax.autoscale_view()

        fig.canvas.draw()
        fig.canvas.flush_events()

    # -----------------------------------------------------------
    # RUN EPISODES
    # -----------------------------------------------------------
    try:
        for episode in range(num_episodes):

            obs = env.reset()
            done = False
            total_reward = 0

            print(f"\n=== Episode {episode+1} ===")

            while not done:

                # Hard stop
                if max_steps is not None and global_step >= max_steps:
                    print(f"\nReached max_steps={max_steps}. Ending test.\n")
                    # finalize before returning
                    csv_file.flush()
                    csv_file.close()
                    return finalize_and_exit(step_rewards, contact_values, distance_values, total_rewards, csv_path)

                # R pressed → reset safely
                if reset_flag["reset"]:
                    print("Resetting environment...\n")
                    obs = env.reset()
                    total_reward = 0
                    reset_flag["reset"] = False
                    continue

                # Step env
                action, _ = model.predict(obs, deterministic=False)
                obs, reward, done, info = env.step(action)

                # ----------------------------
                # Diagnostics logging
                # ----------------------------

                # Reward
                r = float(reward[0])
                step_rewards.append(r)
                total_reward += r

                # Contact tracker — from true physics flag (info is list from VecEnv)
                info_dict = info[0] if isinstance(info, (list, tuple)) else info
                contact_bool = bool(info_dict.get("is_gripping", False))
                contact_val = 1 if contact_bool else 0
                contact_values.append(contact_val)

                # Distance to goal (vectorized)
                ach = obs["achieved_goal"][0][:3]
                des = obs["desired_goal"][0][:3]
                dist = np.linalg.norm(ach - des) / 5.0
                distance_values.append(float(dist))

                # Write CSV row (step, reward, contact(0/1), distance)
                csv_writer.writerow([global_step, r, contact_val, float(dist)])

                # Console line
                print(f"Step {global_step:6d} | Reward: {r:+.4f} | Contact: {contact_bool} | Dist: {dist:.4f}")

                # Update plot every 10 steps
                if global_step % 10 == 0:
                    update_plot()

                global_step += 1

            # end episode
            total_rewards.append(total_reward)
            print(f"Episode {episode+1} finished — Total reward: {total_reward}")

    finally:
        # ensure csv file closed even on error/keyboard interrupt
        csv_file.flush()
        csv_file.close()

    # After all episodes
    return finalize_and_exit(step_rewards, contact_values, distance_values, total_rewards, csv_path)


# -----------------------------------------------------------
# Finalize (CSV save + final plot freeze + contact metrics)
# -----------------------------------------------------------
def finalize_and_exit(step_rewards, contact_values, distance_values, total_rewards, csv_path):
    # Already written per-step CSV during run; print final file path
    print(f"\nPer-step CSV saved → {csv_path}")

    # Compute contact metrics
    contact_array = np.array(contact_values, dtype=int)
    mean_contact = float(contact_array.mean()) if contact_array.size > 0 else 0.0
    grip_durations = analyze_grip_episodes(contact_array.tolist())
    interruption_count = 0
    if len(contact_array) > 1:
        # count transitions from 1 -> 0 after being 1
        interruption_count = int(((contact_array[:-1] == 1) & (contact_array[1:] == 0)).sum())

    mean_duration = float(np.mean(grip_durations)) if grip_durations else 0.0
    max_duration = int(max(grip_durations)) if grip_durations else 0

    # Contact stability (final smoothed value)
    stability_curve = smooth_curve(contact_array.tolist(), window=50) if contact_array.size > 0 else []
    final_stability = float(stability_curve[-1]) if stability_curve else 0.0

    # Print a compact report
    print("\n=== Contact Consistency Report ===")
    print(f"Total steps recorded: {len(step_rewards)}")
    print(f"Mean contact (0-1):        {mean_contact:.4f}")
    print(f"Grip interruption count:   {interruption_count}")
    print(f"Grip episodes (durations): {grip_durations}")
    print(f"Mean grip duration:        {mean_duration:.2f} steps")
    print(f"Max grip duration:         {max_duration} steps")
    print(f"Final contact stability:   {final_stability:.4f}")
    print("=================================\n")

    # Show final plot (turn off interactive mode)
    plt.ioff()
    plt.show()

    # Summary reward
    if total_rewards:
        avg = np.mean(total_rewards)
        print("Average reward:", avg)
        return avg
    else:
        print("No episode completed.")
        return None

def make_env():
    return NeedlePickTrainEnv(
        render_mode="human",
        reward_mode="curriculum",
        traj_len=10240
    )
# -----------------------------------------------------------
# RUN MAIN
# -----------------------------------------------------------
if __name__ == "__main__":
    model_path = "/home/host-20-04/Downloads/numpang_data/skripsi/surrol/tasks/experiment/curriculum/models/needle_pick_ppo_gpu_40.zip"
    # model_path = "/home/host-20-04/Downloads/numpang_data/skripsi/surrol/tasks/experiment/sparse/models/needle_pick_ppo_gpu_20.zip"
    # model_path = "/home/host-20-04/Downloads/numpang_data/skripsi/surrol/tasks/experiment/less_sparse/models/needle_pick_ppo_gpu_20.zip"
    evaluate_model(model_path, num_episodes=3, max_steps=10240)
