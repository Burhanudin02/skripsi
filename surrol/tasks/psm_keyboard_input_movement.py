import time
import numpy as np
import pybullet as p
from surrol.tasks.psm_env import PsmEnv  # Inherit PsmEnv properly
import os
from surrol.const import ASSET_DIR_PATH  # gives you asset path

class PsmKeyboardControlEnv(PsmEnv):  # Inherit from PsmEnv correctly
    ACTION_MODE = 'yaw'
    QPOS_PSM = (0, 0, 0.10, 0, 0, 0)  # Initial joint position for PSM
    WORKSPACE_LIMITS = ((-0.5, 0.5), (-0.4, 0.4), (0.05, 0.05))

    def __init__(self, render_mode=None):
        self._step = 0
        super().__init__(render_mode)  # Calls PsmEnv's constructor and _env_setup()

    def _env_setup(self):
        """Set up the PSM environment."""
        super()._env_setup()  # Call the parent class's setup
        
        self.psm1.reset_joint(self.QPOS_PSM)
        self.block_gripper = False

        # 1. Load tray pad
        tray_path = os.path.join(ASSET_DIR_PATH, 'tray/tray_pad.urdf')
        tray_position = [0.5, 0.0, 0.675]  # Same as used in needle_pick.py
        tray_orientation = p.getQuaternionFromEuler([0, 0, 0])
    
        self.tray_id = p.loadURDF(
            tray_path,
            basePosition=tray_position,
            baseOrientation=tray_orientation,
            useFixedBase=True,
            globalScaling=self.SCALING
        )

        # 2. Load the needle on top of the tray
        needle_path = os.path.join(ASSET_DIR_PATH, 'needle/needle_40mm.urdf')
        needle_position = [0.5, 0.0, 0.68]  # Slightly above tray Z
        needle_orientation = p.getQuaternionFromEuler([0, 0, 0])
    
        self.needle_id = p.loadURDF(
            needle_path,
            basePosition=needle_position,
            baseOrientation=needle_orientation,
            useFixedBase=False,
            globalScaling=self.SCALING
        )

    def _sample_goal(self):
        return np.array([0., 0., 0.])

    def compute_reward(self, achieved_goal, desired_goal, info):
        return 0.0

    def _is_success(self, achieved_goal, desired_goal):
        return 0.0


# --- Key mappings for XYZ control ---
KEY_MAP = {
    ord('s'): np.array([0.1, 0.0, 0.0]),
    ord('w'): np.array([-0.1, 0.0, 0.0]),
    ord('d'): np.array([0.0, 0.1, 0.0]),
    ord('a'): np.array([0.0, -0.1, 0.0]),
    ord('e'): np.array([0.0, 0.0, -10.0]),
    ord('q'): np.array([0.0, 0.0, 10.0]),
}
ESC_KEY = 27

def main():
    env = PsmKeyboardControlEnv(render_mode='human')  # Use PSM env instead of ECM
    env.reset()
    print("✅ PSM keyboard control started.")
    print("Controls: W/A/S/D = XY, Q/E = Z, J/K = yaw jaw. ESC to quit.")

    try:
        while True:
            action = np.zeros(5)  # Ensure action is a 5D vector, corresponding to (XYZ movement + jaw control)
            keys = p.getKeyboardEvents()
            action[4] = -1

            for k in keys:
                if keys[k] & p.KEY_IS_DOWN:
                    if k in KEY_MAP:
                        action[:3] += KEY_MAP[k]  # Add to XYZ movement part of action
                    elif k == ord('j'):  # Press 'j' to rotate the jaw
                        action[3] = 1.0  # Jaw open
                    elif k == ord('k'):  # Press 'k' to rotate the jaw
                        action[3] = -1.0  # Jaw close
                    elif k == ord('o'):
                        action[4] = 1  # Open jaw
                    # elif k == ord('p'):
                    #     action[4] = -1  # Close jaw
                    elif k == ESC_KEY:
                        raise KeyboardInterrupt

            # Debugging: Log action shape and limits to ensure proper clipping
            # print(f"Action shape: {action.shape}")
            # print(f"Action space low: {env.action_space.low.shape}, high: {env.action_space.high.shape}")

            if np.any(action):
                # Ensure action is within the allowed bounds
                action = np.clip(action, env.action_space.low, env.action_space.high)

                # Log action after clipping
                # print(f"Clipped action: {action}")

                env.step(action)
                pos = env.psm1.get_current_position()[:3, 3]  # Get the current position of the robot's jaw
                print("Tip Position:", np.round(pos, 4))

                # Get joint positions from observation (similar to my_needle_pick_env.py)
                robot_state = env._get_robot_state(idx=0)
                position = robot_state[:3]
                euler_angles = robot_state[3:6]
                orientation_quat = p.getQuaternionFromEuler(euler_angles)
                psm_joints_angle = env.psm1.inverse_kinematics((position, orientation_quat), env.psm1.EEF_LINK_INDEX)
                print("PSM Joint Angles:", np.round(np.degrees(psm_joints_angle), 4))


            time.sleep(0.05)

    except KeyboardInterrupt:
        print("🔚 Exiting.")
    finally:
        env.close()
        print("✅ Environment closed.")

if __name__ == "__main__":
    main()
