import numpy as np
import pybullet as p
import os
from surrol.utils.pybullet_utils import get_link_pose, step
from surrol.const import ASSET_DIR_PATH
from surrol.tasks.psm_env import PsmEnv
from gym import spaces

class NeedlePickTrainEnv(PsmEnv):
    """
    NeedlePickTrainEnv is a custom environment for training a robot to pick a needle.
    This environment is based on the PsmEnv class and is used to simulate a robotic arm
    interacting with a needle in a 3D workspace.
    """

    # Workspace limits for needle placement and robot movement
    WORKSPACE_LIMITS = ((0.50, 0.60), (-0.05, 0.05), (0.685, 0.745))  # Adjusted workspace limits
    SCALING = 5.0  # Scaling factor for the environment

    # Tray pose: Position and orientation of the tray where the needle is placed
    POSE_TRAY = ((0.55, 0, 0.6751), (0, 0, 0))

    # Joint limits for the robot arm (defined in radians)
    TOOL_JOINT_LIMIT = {
        'lower': np.deg2rad([-91.0, -53.0, 0.0, -260.0, -80.0, -80.0, -20.0]),
        'upper': np.deg2rad([91.0, 53.0, 240.0, 260.0, 80.0, 80.0, 80.0]),
    }

    # Action space: Smaller, more controlled movements
    action_space = spaces.Box(
        low=np.array([-0.00000000000005, -0.00000000000005, -0.00000000000005], dtype=np.float64),
        high=np.array([0.00000000000005, 0.00000000000005, 0.00000000000005], dtype=np.float64),
        dtype=np.float64
    )

    def _env_setup(self):
        """
        Initialize the environment setup, including the robot and object (needle) setup.
        """
        # Call parent class environment setup
        super(NeedlePickTrainEnv, self)._env_setup()

        # Initialize object and robot state
        self.has_object = True
        self._waypoint_goal = True

        # Robot setup: Set the workspace limits and position of the robot
        workspace_limits = self.workspace_limits1
        pos = (
            workspace_limits[0][0],
            workspace_limits[1][1],
            (workspace_limits[2][1] + workspace_limits[2][0]) / 2
        )
        orn = (0.5, 0.5, -0.5, -0.5)
        joint_positions = self.psm1.inverse_kinematics((pos, orn), self.psm1.EEF_LINK_INDEX) # mengatur posisi joints psm dengan inverse kinematic
        self.psm1.reset_joint(joint_positions)
        self.block_gripper = False                  # True untuk mengunci gripper
        self._contact_approx = False

        # Tray setup: Load tray model into the environment
        obj_id = p.loadURDF(
            os.path.join(ASSET_DIR_PATH, 'tray/tray_pad.urdf'),
            np.array(self.POSE_TRAY[0]) * self.SCALING,
            p.getQuaternionFromEuler(self.POSE_TRAY[1]),
            globalScaling=self.SCALING
        )
        self.obj_ids['fixed'].append(obj_id)

        # Needle setup: Load needle model into the environment with random orientation
        yaw = (np.random.rand() - 0.5) * np.pi
        obj_id = p.loadURDF(
            os.path.join(ASSET_DIR_PATH, 'needle/needle_40mm.urdf'),
            (
                workspace_limits[0].mean() + (np.random.rand() - 0.5) * 0.1,
                workspace_limits[1].mean() + (np.random.rand() - 0.5) * 0.1,
                workspace_limits[2][0] + 0.01
            ),
            p.getQuaternionFromEuler((0, 0, yaw)),
            useFixedBase=False,
            globalScaling=self.SCALING
        )
        p.changeVisualShape(obj_id, -1, specularColor=(80, 80, 80))
        self.obj_ids['rigid'].append(obj_id)
        self.obj_id, self.obj_link1 = self.obj_ids['rigid'][0], 1

        self.needle_out_of_bounds = False

        # Set the goal (randomized target position for needle)
        self.goal = self._sample_goal()

    def _sample_goal(self):
        """
        Sample a random goal position (needle target position) within the workspace.
        This goal represents the position where the needle should be placed.
        """
        workspace_limits = self.WORKSPACE_LIMITS
        return np.array([
            np.random.uniform(workspace_limits[0][0], workspace_limits[0][1]),  # X position
            np.random.uniform(workspace_limits[1][0], workspace_limits[1][1]),  # Y position
            np.random.uniform(workspace_limits[2][0], workspace_limits[2][1])   # Z position
        ])

    def _get_obs(self):
        """
        Get the current observation, including the robot's state and the object's (needle) position.
        """
        # Get the robot's state (joint positions, orientations, jaw positions, etc.)
        robot_state = self._get_robot_state(idx=0)

        # Get the object's state (position and orientation of the needle)
        pos, _ = get_link_pose(self.obj_ids['rigid'][0], self.obj_link1)  # Needle position
        object_pos = np.array(pos)

        # Normalize the object position (optional, based on workspace limits)
        normalized_object_pos = (object_pos - np.array(self.WORKSPACE_LIMITS[2][0])) / (self.WORKSPACE_LIMITS[2][1] - self.WORKSPACE_LIMITS[2][0])

        # Combine robot state and object position into a single observation
        return np.concatenate([robot_state, normalized_object_pos])

    def compute_reward(self, achieved_goal, desired_goal, info):
        """
        Compute the reward based on how close the achieved goal (gripper/jaw/tip position)
        is to the desired goal (needle position).
        """
        # Calculate the distance between the robot's gripper/jaw/tip position and the desired needle position
        distance = np.linalg.norm(achieved_goal - desired_goal)
        goal_dist = np.linalg.norm(desired_goal - self.goal)

        # print(f"Distance : {distance}")

        # Reward is the negative distance (the closer to the desired goal, the better)
        reward = -distance

        # Reward shaping: add bonus if the robot is close to the goal
        if distance < 0.1:
            reward += 20  # Bonus if the gripper is close to the desired position

        if goal_dist <0.1:
            reward += 100

        # Penalize if the robot's joints are out of bounds
        if info.get("joint_valid", True) is False:
            reward -= 1
            print('Punished: Joint out of bounds')

        return reward

    def _set_action(self, action: np.ndarray):
        """
        Apply the action to the robot's actuators. The action represents movement in joint space.
        """
        # Clip the action to ensure it stays within valid bounds
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Pass the action to the parent class for processing
        super()._set_action(action)

    def reset(self):
        """
        Reset the environment to the initial state and return the initial observation.
        """
        # Reset simulation (without re-initializing environment setup)
        print("Reset...")

        # Robot setup: Set the workspace limits and position of the robot
        workspace_limits = self.workspace_limits1
        pos = (
            workspace_limits[0][0],
            workspace_limits[1][1],
            (workspace_limits[2][1] + workspace_limits[2][0]) / 2
        )
        orn = (0.5, 0.5, -0.5, -0.5)
        joint_positions = self.psm1.inverse_kinematics((pos, orn), self.psm1.EEF_LINK_INDEX) # mengatur posisi joints psm dengan inverse kinematic
        self.psm1.reset_joint(joint_positions)

        return self._get_obs()

    def check_needle_out_of_bounds(self):
        """
        Check if the needle has moved outside of the tray's boundaries.
        """
        needle_pos, _ = get_link_pose(self.obj_id, self.obj_link1)
        x, y, z = needle_pos

        workspace_limits = self.workspace_limits1
        if not (workspace_limits[0][0] <= x <= workspace_limits[0][1] and
                workspace_limits[1][0] <= y <= workspace_limits[1][1]):
            return True  # Needle is out of bounds
        return False  # Needle is within bounds

    def realign_needle(self):
        """
        Realign the needle if it goes out of bounds by setting it to a new valid position.
        """
        if self.needle_out_of_bounds:
            workspace_limits = self.workspace_limits1
            new_position = np.array([
                workspace_limits[0].mean() + (np.random.rand() - 0.5) * 0.1,
                workspace_limits[1].mean() + (np.random.rand() - 0.5) * 0.1,
                workspace_limits[2][0] + 0.01
            ])
            p.resetBasePositionAndOrientation(self.obj_id, new_position, p.getQuaternionFromEuler([0, 0, 0]))

    def step(self, action: np.ndarray):
        """
        Take a simulation step with the given action. Update the environment's state.
        """
        # Check if the needle is out of bounds
        self.needle_out_of_bounds = self.check_needle_out_of_bounds()

        # Ensure action is large enough to move the robot
        action_magnitude = np.linalg.norm(action)
        if action_magnitude < 1e-4:
            print("Action is too small, robot might not move.")

        # Apply action to robot
        self._set_action(action)
        step(1)  # PyBullet simulation step

        # Get updated observation
        obs = self._get_obs()

        # Get achieved goal (tip position of the robot)
        achieved_goal = obs[:3]  # Gripper/jaw/tip position from observation

        # Realign the needle if it's out of bounds
        if self.needle_out_of_bounds:
            self.realign_needle()

        # The desired goal is the position of the needle
        pos_needle, _ = get_link_pose(self.obj_id, self.obj_link1)
        desired_goal = np.array(pos_needle)

        # Get joint positions from observation
        robot_state = self._get_robot_state(idx=0)

        psm_joints_angle= self.psm1.inverse_kinematics((robot_state[:3], robot_state[3:7]), self.psm1.EEF_LINK_INDEX)
        # print(f"psm joints angle: {psm_joints_angle}")

        joint_valid = True
        # Check if any joint is outside its limits
        for joint_index, joint_pos in enumerate(psm_joints_angle):
            lower_limit = self.TOOL_JOINT_LIMIT['lower'][joint_index]
            upper_limit = self.TOOL_JOINT_LIMIT['upper'][joint_index]

            if joint_pos < lower_limit or joint_pos > upper_limit:
                joint_valid = False
                print(f"Joint {joint_index} out of joint limits! Position: {joint_pos}, Limits: [{lower_limit}, {upper_limit}]")
                break

        # Update info to calculate reward based on joint validity
        info = {"joint_valid": joint_valid}

        # Calculate reward based on achieved goal (robot's tip) and desired goal (needle position)
        reward = self.compute_reward(achieved_goal, desired_goal, info)

        # print(f"Reward: {reward}")

        done = self._is_done(achieved_goal, desired_goal) 

        return obs, reward, done, info

    def _is_done(self, achieved_goal, desired_goal):
        """
        Definisi satu tugas sudah berhasil dikerjakan ada disini
        """
        # di tahap selanjutnya tetapkan definiton of done
        done = np.linalg.norm(achieved_goal - desired_goal) < 0.1 and np.linalg.norm(desired_goal - self.goal) < 0.1 
        return done

    def render(self, mode=None):
        """
        Render the current state of the environment (optional).
        """
        if mode == 'human':
            return np.array([])  # Implement rendering logic if needed

##this is the stable enough version, lol