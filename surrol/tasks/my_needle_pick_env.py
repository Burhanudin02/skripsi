import numpy as np
import pybullet as p
import os
from surrol.utils.pybullet_utils import get_link_pose, step
from surrol.const import ASSET_DIR_PATH
from surrol.tasks.psm_env import PsmEnv
from gym import spaces

class NeedlePickTrainEnv(PsmEnv):
    """
    Custom environment for training a robot to pick and place a needle.
    Inherits from SurRoL's PsmEnv and is GoalEnv-compatible (for HER).
    """

    # Needle workspace limits (meters)
    WORKSPACE_LIMITS = ((0.50, 0.60), (-0.05, 0.05), (0.685, 0.745))
    SCALING = 5.0
    POSE_TRAY = ((0.55, 0, 0.6751), (0, 0, 0))

    TOOL_JOINT_LIMIT = {
        'lower': np.deg2rad([-6.04, -32.26, 8.5, -172.45, -44.78, -43.25]),
        'upper': np.deg2rad([44.03, 9.58, 12.74, 164.87, 48.2, 43.47]),
    }

    def __init__(self, render_mode=None, reward_mode="sparse"):
        self.render_mode = render_mode
        self.reward_mode = reward_mode
        self.enable_logging = False
        
        super().__init__(render_mode=render_mode)

        # Action: dx, dy, dz, d_yaw, gripper_open/close
        self.action_space = spaces.Box(
            low=np.array([-5e-14, -5e-14, -5e-14, -5e-14, -1.0]),
            high=np.array([5e-14, 5e-14, 5e-14, 5e-14, 1.0]),
            dtype=np.float64
        )

        # Observation: dict format for GoalEnv
        self.observation_space = spaces.Dict({
            "observation": spaces.Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32),
            "achieved_goal": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            "desired_goal": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        })

    def _env_setup(self):
        super()._env_setup()
        self.has_object = True
        self._waypoint_goal = True

        # Reset robot pose using IK to a neutral start position
        workspace_limits = self.workspace_limits1
        pos = (
            workspace_limits[0][0],
            workspace_limits[1][1],
            (workspace_limits[2][1] + workspace_limits[2][0]) / 2
        )
        orn = (0.5, 0.5, -0.5, -0.5)
        joint_positions = self.psm1.inverse_kinematics((pos, orn), self.psm1.EEF_LINK_INDEX)
        self.psm1.reset_joint(joint_positions)

        self.block_gripper = False
        self._contact_approx = False

        # Load tray
        tray_id = p.loadURDF(
            os.path.join(ASSET_DIR_PATH, 'tray/tray_pad.urdf'),
            np.array(self.POSE_TRAY[0]) * self.SCALING,
            p.getQuaternionFromEuler(self.POSE_TRAY[1]),
            globalScaling=self.SCALING
        )
        self.obj_ids['fixed'].append(tray_id)

        # Load needle (rigid body)
        yaw = (np.random.rand() - 0.5) * np.pi
        needle_pos = (
            np.mean(self.WORKSPACE_LIMITS[0]) + (np.random.rand() - 0.5) * 0.1,
            np.mean(self.WORKSPACE_LIMITS[1]) + (np.random.rand() - 0.5) * 0.1,
            self.WORKSPACE_LIMITS[2][0] + 0.01
        )
        needle_id = p.loadURDF(
            os.path.join(ASSET_DIR_PATH, 'needle/needle_40mm.urdf'),
            needle_pos,
            p.getQuaternionFromEuler((0, 0, yaw)),
            useFixedBase=False,
            globalScaling=self.SCALING
        )
        p.changeVisualShape(needle_id, -1, specularColor=(80, 80, 80))
        self.obj_ids['rigid'].append(needle_id)
        self.obj_id, self.obj_link1 = needle_id, 1

        self.needle_out_of_bounds = False
        self.goal = self._sample_goal()

    def _sample_goal(self):
        limits = self.WORKSPACE_LIMITS
        return np.array([
            np.random.uniform(limits[0][0], limits[0][1]),
            np.random.uniform(limits[1][0], limits[1][1]),
            np.random.uniform(limits[2][0], limits[2][1])
        ])

    def _get_obs(self):
        robot_state = self._get_robot_state(idx=0)
        object_pos, _ = get_link_pose(self.obj_id, self.obj_link1)
        goal_pos = self.goal

        return {
            "observation": np.concatenate([robot_state, object_pos, goal_pos]).astype(np.float32),
            "achieved_goal": np.array(object_pos, dtype=np.float32),
            "desired_goal": np.array(goal_pos, dtype=np.float32)
        }

    def compute_reward(self, achieved_goal, desired_goal, info):
        # Jarak gripper ke needle (target)
        distance = np.linalg.norm(achieved_goal - desired_goal)
        print(f"Distance to needle: {distance}")

        # Jarak needle ke goal akhir
        needle_to_goal = np.linalg.norm(desired_goal - self.goal)
        print(f"Needle to goal: {needle_to_goal}")

        if self.reward_mode == "less_sparse":
            return self.less_sparse_reward_shape(-distance, distance, needle_to_goal)
        elif self.reward_mode == "curriculum":
            return self.curriculum_learn_reward(info, -distance, distance, achieved_goal, needle_to_goal)

        # Default sparse
        reward = -distance
        if distance < 0.1:
            reward += 2.0
        if needle_to_goal < 0.1:
            reward += 10.0
        if not info.get("joint_valid", True):
            reward -= 0.1

        print(f"Reward: {reward})")
        return reward


    def less_sparse_reward_shape(self, reward, distance, goal_dist):
        reward += (3 - distance) * 0.05 if distance < 3.0 else 0
        if distance < 0.5:
            reward -= distance * 0.001
        if goal_dist < 0.5:
            reward += (1 - goal_dist) * 0.05
        if goal_dist < 0.1:
            reward += (1 - goal_dist) * 0.5
        return reward

    def curriculum_learn_reward(self, info, reward, distance, achieved_goal, goal_dist):
        steps = info.get("timestep", 0)
        jaw_close = self.jaw_action < 0

        if steps < 10000 / 12:
            reward += 0.1 if info.get("joint_valid", True) else -0.1
        elif steps < 250000 / 12:
            reward += 0.1 if info.get("joint_valid", True) else -0.1
            reward += (1 - distance) * 0.2
        elif steps < 500000 / 12:
            reward += (1 - distance) * 0.2
            if 0.91 < distance < 1.0:
                reward += np.exp(-distance)
                if jaw_close and achieved_goal[2] > -0.14:
                    reward += np.exp(-distance)
                else:
                    reward -= np.exp(distance) * 0.001
        elif steps < 1000000 / 12:
            reward += (1 - distance) * 0.2
            if 0.91 < distance < 1.0:
                reward += np.exp(-distance)
                if jaw_close and achieved_goal[2] > -0.14:
                    reward += np.exp(-distance)
                else:
                    reward -= np.exp(distance) * 0.001
            reward += np.exp(-goal_dist)

        return reward

    def _set_action(self, action: np.ndarray):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.jaw_action = action[4]
        super()._set_action(action)

    def reset(self):
        print("Reset...")
        self.timestep = 0
        workspace_limits = self.workspace_limits1
        pos = (
            workspace_limits[0][0],
            workspace_limits[1][1],
            (workspace_limits[2][1] + workspace_limits[2][0]) / 2
        )
        orn = (0.5, 0.5, -0.5, -0.5)
        joint_positions = self.psm1.inverse_kinematics((pos, orn), self.psm1.EEF_LINK_INDEX)
        self.psm1.reset_joint(joint_positions)
        return self._get_obs()

    def step(self, action):
        self.timestep += 1
        self._set_action(action)
        step(1)

        self.needle_out_of_bounds = self.check_needle_out_of_bounds()
        if self.needle_out_of_bounds:
            self.realign_needle()

        obs = self._get_obs()
        # achieved = obs["achieved_goal"]
        # desired = obs["desired_goal"]

        achieved = obs["observation"][:3] # get the position of the robot
        desired = obs["achieved_goal"]

        # Joint valid check
        robot_state = self._get_robot_state(idx=0)
        position = robot_state[:3]
        quat = robot_state[3:7]  # ensure quaternion is returned from _get_robot_state
        joint_angles = self.psm1.inverse_kinematics((position, quat), self.psm1.EEF_LINK_INDEX)

        joint_valid = all(
            self.TOOL_JOINT_LIMIT['lower'][i] <= joint_angles[i] <= self.TOOL_JOINT_LIMIT['upper'][i]
            for i in range(len(joint_angles))
        )

        info = {
            "joint_valid": joint_valid,
            "timestep": self.timestep,
            "is_gripping": self._activated == 0,
            "needle_out_of_bounds": self.needle_out_of_bounds
        }

        reward = self.compute_reward(achieved, desired, info)
        done = self._is_done(desired, obs["desired_goal"])

        return obs, reward, done, info

    def _is_done(self, achieved_goal, desired_goal):
        dist_gripper_to_needle = np.linalg.norm(achieved_goal - desired_goal)
        dist_needle_to_goal = np.linalg.norm(desired_goal - self.goal)
        return dist_gripper_to_needle < 0.1 and dist_needle_to_goal < 0.1

    def check_needle_out_of_bounds(self):
        x, y, _ = get_link_pose(self.obj_id, self.obj_link1)[0]
        limits = self.workspace_limits1
        return not (limits[0][0] <= x <= limits[0][1] and limits[1][0] <= y <= limits[1][1])

    def realign_needle(self):
        if self.needle_out_of_bounds:
            workspace = self.workspace_limits1
            new_pos = np.array([
                workspace[0].mean() + (np.random.rand() - 0.5) * 0.1,
                workspace[1].mean() + (np.random.rand() - 0.5) * 0.1,
                workspace[2][0] + 0.01
            ])
            p.resetBasePositionAndOrientation(self.obj_id, new_pos, p.getQuaternionFromEuler([0, 0, 0]))

    def render(self, mode=None):
        if mode == 'human':
            return np.array([])
