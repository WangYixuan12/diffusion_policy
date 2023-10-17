import os
from typing import List, Optional

import numpy as np
from gym import spaces
import sapien.core as sapien
import transforms3d
import hydra
from omegaconf import OmegaConf
import h5py
import cv2
from tqdm import tqdm

import sys
sys.path.append('/home/yixuan/diffusion_policy')

from sapien_env.rl_env.base import BaseRLEnv
from sapien_env.gui.teleop_gui_trossen import GUIBase, YX_TABLE_TOP_CAMERAS
from sapien_env.teleop.teleop_robot import TeleopRobot
from sapien_env.sim_env.constructor import add_default_scene_light
from diffusion_policy.model.common.rotation_transformer import RotationTransformer

def transform_action_from_world_to_robot(action : np.ndarray, pose : sapien.Pose):
    # :param action: (7,) np.ndarray in world frame. action[:3] is xyz, action[3:6] is euler angle, action[6] is gripper
    # :param pose: sapien.Pose of the robot base in world frame
    # :return: (7,) np.ndarray in robot frame. action[:3] is xyz, action[3:6] is euler angle, action[6] is gripper
    # transform action from world to robot frame
    action_mat = np.zeros((4,4))
    action_mat[:3,:3] = transforms3d.euler.euler2mat(action[3], action[4], action[5])
    action_mat[:3,3] = action[:3]
    action_mat[3,3] = 1
    action_mat_in_robot = np.matmul(np.linalg.inv(pose.to_transformation_matrix()),action_mat)
    action_robot = np.zeros(7)
    action_robot[:3] = action_mat_in_robot[:3,3]
    action_robot[3:6] = transforms3d.euler.mat2euler(action_mat_in_robot[:3,:3],axes='sxyz')
    action_robot[6] = action[6]
    return action_robot

class SapienEnvWrapper():
    def __init__(self,
                 env : BaseRLEnv,
                 shape_meta: dict,
                 init_state: Optional[np.ndarray]=None,
                 render_obs_keys = ['right_bottom_view'],):
        # :param init_state: (4,4) np.ndarray. The initial pose of the object in world frame
        self.env = env
        self.shape_meta = shape_meta
        self.init_state = init_state
        self.render_obs_keys = render_obs_keys
        self.rotation_transformer = RotationTransformer('rotation_6d', 'euler_angles', to_convention='XYZ')
        
        # setup spaces
        action_shape = shape_meta['action']['shape']
        action_space = spaces.Box(
            low=-1,
            high=1,
            shape=action_shape,
            dtype=np.float32
        )
        self.action_space = action_space
        
        observation_space = spaces.Dict()
        for key, value in shape_meta['obs'].items():
            shape = value['shape']
            min_value, max_value = -1, 1
            if key.endswith('color'):
                min_value, max_value = 0, 1
            elif key.endswith('quat'):
                min_value, max_value = -1, 1
            elif key.endswith('qpos'):
                min_value, max_value = -1, 1
            elif key.endswith('pos'):
                # better range?
                min_value, max_value = -1, 1
            else:
                raise RuntimeError(f"Unsupported type {key}")
            
            this_space = spaces.Box(
                low=min_value,
                high=max_value,
                shape=shape,
                dtype=np.float32
            )
            observation_space[key] = this_space
        self.observation_space = observation_space
        
        # setup sapien rendering
        self.gui = GUIBase(env.scene, env.renderer, headless=True)
        for name, params in YX_TABLE_TOP_CAMERAS.items():
            self.gui.create_camera(**params)
        
        # setup sapien control
        self.teleop = TeleopRobot(env.robot_name)
    
    def seed(self, seed=None):
        self.env.seed(seed)
    
    def render(self):
        rgbs = self.gui.render()
        rgbs_sel = []
        for cam_idx, cam in enumerate(self.gui.cams):
            if cam.name in self.render_obs_keys:
                rgbs_sel.append(rgbs[cam_idx])
        rgbs_sel = np.concatenate(rgbs_sel, axis=1).astype(np.uint8) # concat horizontally
        return rgbs_sel
    
    def step(self, action):
        # :param: action: np.ndarray. [0:3] for translation, [3:9] for 6d rotation, [10] for gripper
        # :return: obs, reward, done, info
        action_euler = self.rotation_transformer.forward(action[3:9])
        action_in_world = np.concatenate([action[:3], action_euler, action[9:]])
        action_in_robot = transform_action_from_world_to_robot(action_in_world, self.env.robot.get_pose())
        
        arm_dof = self.env.arm_dof
        joint_action = np.zeros(arm_dof+1)
        joint_action[:arm_dof] = self.teleop.ik_panda(self.env.robot.get_qpos()[:],action_in_robot)
        joint_action[arm_dof:] = action_in_robot[-1]
        return self.env.step(joint_action)
    
    def reset(self):
        self.env.reset()
        if self.init_state is not None:
            init_pose = sapien.Pose.from_transformation_matrix(self.init_state)
            self.env.manipulated_object.set_pose(init_pose)

def test():
    dataset_dir = '/media/yixuan_2T/diffusion_policy/data/sapien_env/teleop_data/pick_place_soda/2023-10-17-02-09-09-909762'
    env_cfg = OmegaConf.load(f'{dataset_dir}/config.yaml')
    env = hydra.utils.instantiate(env_cfg)
    add_default_scene_light(env.scene, env.renderer)
    shape_meta = {
        'action': {
            'shape': (10,)
        },
        'obs': {
            'front_right_view_color': {
                'shape': (3, 60, 80),
                'type': 'rgb',
            },
        },
    }
    dataset_path = os.path.join(dataset_dir, f'episode_0.hdf5')
    rotation_transformer = RotationTransformer('euler_angles', 'rotation_6d', from_convention='XYZ')
    with h5py.File(dataset_path) as f:
        init_state = f['info']['init_pose'][()]
        wrapper = SapienEnvWrapper(env, shape_meta, init_state=init_state)
        wrapper.reset()
        action_seq = f['cartesian_action'][()]
        T = action_seq.shape[0]
        for t in tqdm(range(T)):
            action = action_seq[t]
            action_rot = rotation_transformer.forward(action[3:6])
            action = np.concatenate([action[:3], action_rot, action[6:]])
            _ = wrapper.step(action)
            img = wrapper.render()
            cv2.imshow('img', img)
            cv2.waitKey(1)
    
if __name__ == '__main__':
    test()
    
