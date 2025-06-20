import time
import numpy as np
import pybullet as p
import cv2
from surrol.tasks.ecm_env import EcmEnv
from surrol.robots.ecm import RENDER_HEIGHT, RENDER_WIDTH
import os
from surrol.const import ASSET_DIR_PATH  # gives you asset path


class EcmKeyboardControlEnv(EcmEnv):
    ACTION_MODE = 'cVc'
    QPOS_ECM = (0, 0, 0.04, 0)
    WORKSPACE_LIMITS = ((-0.5, 0.5), (-0.4, 0.4), (0.05, 0.05))

    def __init__(self, render_mode=None):
        self._step = 0
        super().__init__(render_mode)

    def _env_setup(self):
        super()._env_setup()
        self.use_camera = True
        self.ecm.reset_joint(self.QPOS_ECM)

         # 1. Load tray pad
        tray_path = os.path.join(ASSET_DIR_PATH, 'tray/tray_pad.urdf')
        tray_position = [0.5, 0.0, 0.0]  # same as used in needle_pick.py
        tray_orientation = p.getQuaternionFromEuler([0, 0, 0])
    
        self.tray_id = p.loadURDF(
            tray_path,
            basePosition=tray_position,
            baseOrientation=tray_orientation,
            useFixedBase=True,
            globalScaling=self.SCALING
        )

        # 2. Load needle on top of tray
        needle_path = os.path.join(ASSET_DIR_PATH, 'needle/needle_40mm.urdf')
        needle_position = [0.5, 0.0, 0.005]  # slightly above tray Z
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
    ord('a'): np.array([1.0, 0.0, 0.0]),
    ord('d'): np.array([-1.0, 0.0, 0.0]),
    ord('w'): np.array([0.0, 1.0, 0.0]),
    ord('s'): np.array([0.0, -1.0, 0.0]),
    ord('e'): np.array([0.0, 0.0, -1000.0]),
    ord('q'): np.array([0.0, 0.0, 1000.0]),
}
ESC_KEY = 27

def main():
    env = EcmKeyboardControlEnv(render_mode='human')
    env.reset()
    print("✅ ECM keyboard control started.")
    print("Controls: W/A/S/D = XY, Q/E = Z. ESC to quit.")

    try:
        while True:
            keys = p.getKeyboardEvents()
            action = np.zeros(3)

            for k in keys:
                if keys[k] & p.KEY_IS_DOWN:
                    if k in KEY_MAP:
                        action += KEY_MAP[k]
                    elif k == ESC_KEY:
                        raise KeyboardInterrupt

            if np.any(action):
                env.step(action)
                pos = env.ecm.get_current_position()[:3, 3]
                print("Tip Position:", np.round(pos, 4))

            # ✅ Show ECM endoscopic camera image directly
            if env.use_camera:
                rgb_cam, _ = env.ecm.render_image(RENDER_HEIGHT, RENDER_WIDTH)
                # if rgb_cam is not None:
                #     cv2.imshow("Endoscopic Camera", cv2.cvtColor(rgb_cam, cv2.COLOR_RGB2BGR))
                #     if cv2.waitKey(1) & 0xFF == 27:
                #         raise KeyboardInterrupt

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("🔚 Exiting.")
    finally:
        env.close()
        cv2.destroyAllWindows()
        print("✅ Environment closed.")

if __name__ == "__main__":
    main()
