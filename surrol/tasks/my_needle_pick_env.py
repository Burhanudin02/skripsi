import numpy as np
import pybullet as p
import os
from surrol.utils.pybullet_utils import get_link_pose, step, wrap_angle
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

    # TOOL_JOINT_LIMIT = {   # The values are manually observed
    #     'lower': np.deg2rad([-6.04, -32.26, 8.5, -172.45, -44.78, -43.25]),
    #     'upper': np.deg2rad([44.03, 9.58, 12.74, 164.87, 48.2, 43.47]),
    # }

    def __init__(self, render_mode=None, reward_mode="sparse", num_envs=1):
        self.render_mode = render_mode
        self.reward_mode = reward_mode
        self.num_envs = num_envs
        self.was_gripping = False
        self.enable_logging = False
        
        super().__init__(render_mode=render_mode)

        # Action: dx, dy, dz, d_yaw, gripper_open/close
        self.action_space = spaces.Box(
            low=np.array([-5e-2, -5e-2, -5e-2, -5e-2, -1.0]),
            high=np.array([5e-2, 5e-2, 5e-2, 5e-2, 1.0]),
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
        object_pos, object_orn_quat = get_link_pose(self.obj_id, self.obj_link1)
        goal_pos = self.goal

        object_euler = p.getEulerFromQuaternion(object_orn_quat)
        needle_yaw = object_euler[2]
        robot_yaw = robot_state[5]
        self.current_needle_yaw = needle_yaw
        self.current_robot_yaw = robot_yaw

        return {
            "observation": np.concatenate([robot_state, object_pos, goal_pos]).astype(np.float32),
            "achieved_goal": np.array(object_pos, dtype=np.float32),
            "desired_goal": np.array(goal_pos, dtype=np.float32)
        }

    def compute_reward(self, position, achieved_goal, desired_goal, info):
        # # Jarak gripper ke needle (target)
        distance = np.linalg.norm(position - achieved_goal)
        distance = distance/5.0  # Normalize distance by scaling factor
        print(f"Distance to needle: {distance}")

        # Orientation difference (yaw) between gripper and needle
        raw_yaw_diff = self.current_robot_yaw - self.current_needle_yaw
        wrapped_yaw_error = wrap_angle(raw_yaw_diff)
        abs_yaw_error = np.abs(wrapped_yaw_error)
        print(f"Yaw error (rad): {abs_yaw_error}, (deg): {np.rad2deg(abs_yaw_error)}")

        # Grasp logic
        is_gripping_now = info.get("is_gripping", False)
        just_grasped = is_gripping_now and not self.was_gripping

        # Jarak needle ke goal akhir
        needle_to_goal = np.linalg.norm(achieved_goal - desired_goal)
        print(f"Needle to goal: {needle_to_goal}")

        # Check for different reward modes
        if self.reward_mode == "less_sparse":
            return self.less_sparse_reward_shape(-distance, distance, 
                                                 abs_yaw_error, just_grasped, 
                                                 is_gripping_now, needle_to_goal)
        elif self.reward_mode == "curriculum":
            return self.curriculum_learn_reward(info, -distance, distance, 
                                                abs_yaw_error, just_grasped, 
                                                 is_gripping_now, needle_to_goal)

        # Default: sparse
        reward = (1-distance) * 0.01

        if just_grasped:
            print("🎉 Just Grasped! Applying Bonus.")
            reward += 1.0  # Large, one-time bonus for success

        if is_gripping_now:
            reward += (1 - needle_to_goal) * 0.01
        
        print(f"Reward: {reward}")
        return reward

    def less_sparse_reward_shape(self, reward, distance, 
                                 abs_yaw_error, just_grasped, 
                                 is_gripping_now, needle_to_goal):
        reward = reward

        if distance < 0.07:
            reward += (1 - abs_yaw_error) * 0.01
            if just_grasped:
                print("🎉 Just Grasped! Applying Bonus.")
                reward += 1.0  # Large, one-time bonus for success

            if not is_gripping_now:
                # --- STAGE 1: Approach the needle ---
                # Reward for getting closer to the needle
                reward = -distance 

            else:
                # --- STAGE 2: Move the needle to the goal ---
                # Agent is now holding the needle. Reward for moving needle to goal.
                reward += (1 - needle_to_goal) * 0.1
                
                # Constant "holding" bonus to incentivize not dropping the needle
                reward += 0.000001
        
        return reward

    def curriculum_learn_reward(self, info, reward, distance, 
                                abs_yaw_error, just_grasped, 
                                is_gripping_now, needle_to_goal):
        steps = info.get("timestep", 0)
        num_envs = self.num_envs
        reward = reward
        
        # Approaching needle
        if steps < 100000 / num_envs:
            reward = reward
        
        # Aligning orientation
        elif steps < 250000 / num_envs:
            reward += (1 - abs_yaw_error) * 0.01
        
        # Grasping 
        elif steps < 500000 / num_envs:
            if just_grasped:
                print("🎉 Just Grasped! Applying Bonus.")
                reward += 1.0  # Large, one-time bonus for success
                if not is_gripping_now:
                    reward -= 0.5 # Erase half of given bonus

        # Go to final goal        
        elif steps < 1000000 / num_envs:
            if is_gripping_now:
                reward += -needle_to_goal

        return reward

    def _set_action(self, action: np.ndarray):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.jaw_action = action[4]
        super()._set_action(action)

    def reset(self):
        print("Reset...")
        self.timestep = 0
        self.was_gripping = False
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
        super()._step_callback()

        self.needle_out_of_bounds = self.check_needle_out_of_bounds()
        if self.needle_out_of_bounds:
            self.realign_needle()

        obs = self._get_obs()
        achieved = obs["achieved_goal"]
        desired = obs["desired_goal"]

        # Joint valid check
        robot_state = obs["observation"][:7] # Get the robot state from the observation
        position = robot_state[:3]
        quat = robot_state[3:7]  # ensure quaternion is returned from _get_robot_state
        # joint_angles = self.psm1.inverse_kinematics((position, quat), self.psm1.EEF_LINK_INDEX)

        info = {
            "timestep": self.timestep,
            "is_gripping": self._contact_constraint is not None,
            "needle_out_of_bounds": self.needle_out_of_bounds
        }

        reward = self.compute_reward(position, achieved, desired, info)
        done = self._is_done(achieved, desired)

        self.was_gripping = self._contact_constraint is not None

        return obs, reward, done, info

    def _is_done(self, achieved_goal, desired_goal):
        dist = np.linalg.norm(achieved_goal - desired_goal)
        return dist < 0.1

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

