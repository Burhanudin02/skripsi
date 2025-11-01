import time
import numpy as np
import pybullet as p
from surrol.tasks.psm_env import PsmEnv
import os
from surrol.const import ASSET_DIR_PATH
from surrol.utils.pybullet_utils import get_link_pose, wrap_angle

class PsmKeyboardControlEnv(PsmEnv):
    ACTION_MODE = 'yaw'
    SCALING = 5.0 # Match the training environment

    def __init__(self, render_mode=None):
        self._step = 0
        self.needle_id = -1
        self.tray_id = -1
        super().__init__(render_mode)

    def _env_setup(self):
        """Set up the PSM environment."""
        super()._env_setup()
        
        self.has_object = True  # Enable object interaction logic
        self.block_gripper = False

        # Reset the robot pose using IK for consistency with training
        workspace_limits = self.workspace_limits1
        pos = (workspace_limits[0][0], workspace_limits[1][1], (workspace_limits[2][1] + workspace_limits[2][0]) / 2)
        orn = (0.5, 0.5, -0.5, -0.5)
        joint_positions = self.psm1.inverse_kinematics((pos, orn), self.psm1.EEF_LINK_INDEX)
        self.psm1.reset_joint(joint_positions)

        # Load tray (scaling position and size)
        tray_path = os.path.join(ASSET_DIR_PATH, 'tray/tray_pad.urdf')
        tray_position_unscaled = np.array([0.55, 0.0, 0.6751])
        self.tray_id = p.loadURDF(
            tray_path,
            basePosition=tray_position_unscaled * self.SCALING,
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=True,
            globalScaling=self.SCALING
        )

        # Load needle (scaling position and size)
        needle_path = os.path.join(ASSET_DIR_PATH, 'needle/needle_40mm.urdf')
        needle_position_unscaled = np.array([0.55, 0.0, 0.685])
        self.needle_id = p.loadURDF(
            needle_path,
            basePosition=needle_position_unscaled * self.SCALING,
            baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi / 2]),
            useFixedBase=False,
            globalScaling=self.SCALING
        )

        # --- FIX: Set the object ID and the link index for the achieved_goal/grasping logic ---
        self.obj_id = self.needle_id
        self.obj_link1 = 1  # Set the link index to 1 (as in the original needle pick env)
        # --------------------------------------------------------------------------------------

    def _meet_contact_constraint_requirement(self) -> bool:
        # Required for the parent class's grasping logic to work
        return True

    def _sample_goal(self):
        return np.array([0., 0., 0.])
    def compute_reward(self, achieved_goal, desired_goal, info):
        return 0.0
    def _is_success(self, achieved_goal, desired_goal):
        return 0.0
    
    def reset(self):
        """Reset the environment to an initial state."""
        super().reset()


# --- Key map for arrow keys ---
KEY_MAP = {
    p.B3G_DOWN_ARROW: np.array([0.1, 0.0, 0.0]),
    p.B3G_UP_ARROW:   np.array([-0.1, 0.0, 0.0]),
    p.B3G_RIGHT_ARROW:np.array([0.0, 0.1, 0.0]),
    p.B3G_LEFT_ARROW: np.array([0.0, -0.1, 0.0]),
    ord('e'):         np.array([0.0, 0.0, -0.1]),
    ord('q'):         np.array([0.0, 0.0, 0.1]),
}
ESC_KEY = 27

def main():
    env = PsmKeyboardControlEnv(render_mode='human')
    env.reset()
    print("✅ PSM keyboard control started.")
    print("Controls: Arrow Keys=XY, Q/E=Z, J/K=yaw. O=Open Jaw, P=Close Jaw. ESC=quit.")

    try:
        while True:
            action = np.zeros(5)
            keys = p.getKeyboardEvents()
            action[4] = 1 # Default to jaw open

            for k in keys:
                if keys[k] & p.KEY_IS_DOWN:
                    if k in KEY_MAP:
                        action[:3] += KEY_MAP[k]
                    elif k == ord('j'): action[3] = 0.01
                    elif k == ord('k'): action[3] = -0.01
                    elif k == ord('o'): action[4] = 1
                    elif k == ord('p'): action[4] = -1 # Send close jaw command
                    elif k == ord('r'): env.reset()
                    elif k == ESC_KEY: raise KeyboardInterrupt

            env.step(action)

            # Check the grasp status
            if env._contact_constraint is not None:
                print("Grasp Status: ✅ GRASPED (Constraint Active)")
            else:
                print("Grasp Status: ❌ NOT GRASPING")

            # Distance calculation
            gripper_tip_pos, _ = get_link_pose(env.psm1.body, env.psm1.TIP_LINK_INDEX)
            needle_pos, _ = get_link_pose(env.needle_id, -1) 
            distance_scaled = np.linalg.norm(np.array(gripper_tip_pos) - np.array(needle_pos))
            real_world_distance = distance_scaled / env.SCALING
            
            # Get orientations
            needle_pos, needle_orn = get_link_pose(env.needle_id, -1)
            _, gripper_orn = get_link_pose(env.psm1.body, env.psm1.TIP_LINK_INDEX)

            # Convert quaternions to euler angles and extract yaw
            needle_euler = p.getEulerFromQuaternion(needle_orn)
            gripper_euler = p.getEulerFromQuaternion(gripper_orn)

            # Calculate yaw error
            # yaw_error = abs(gripper_euler[2] - needle_euler[2])
            # yaw_error = min(yaw_error, np.pi - yaw_error)
            yaw_error = wrap_angle(gripper_euler[2] - needle_euler[2])
            yaw_error = np.abs(yaw_error)
            
            print(f"Distance (Real World): {real_world_distance:.4f}\n")
            print(f"Yaw Error (rad): {yaw_error:.4f}")

    except KeyboardInterrupt:
        print("🔚 Exiting.")
    finally:
        env.close()
        print("✅ Environment closed.")

if __name__ == "__main__":
    main()