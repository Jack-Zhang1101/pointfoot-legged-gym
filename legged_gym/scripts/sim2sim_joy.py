import math
import numpy as np
import mujoco
import mujoco_viewer
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R


import isaacgym
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.wheellegged_flat.wheellegged_flat_config import BipedCfgWL

import torch
import pygame
#from legged_gym.envs.base.legged_robot_config import adaptor_name

# Define environment observation settings
class env:
    obs = 22
    obs_his = 22 * 11
    num_actions = 6
    # 关节顺序
    joint_names = ["LF_Joint", "LFP_Joint", "LW_Joint", "RF_Joint", "RFP_Joint", "RW_Joint"]

def quat_rotate_inverse(q, v):
    q_w = q[-1]
    q_vec = q[:3]
    a = v * (2.0 * q_w ** 2 - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a - b + c

def quat_rotate_inverse_ori(q, v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
    return a - b + c

def get_obs(data):
    '''Extracts an observation from the mujoco data structure'''
    q = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)
    quat = data.sensor('imu_quat').data[[1, 2, 3, 0]].astype(np.double)
    r = R.from_quat(quat)
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)  # In the base frame
    omega = data.sensor('imu_gyro').data.astype(np.double)
    gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)
    return (q, dq, quat, v, omega, gvec)

def pd_control(target_q, q, kp, target_dq, dq, kd):
    '''Calculates torques from position commands'''
    return (target_q - q) * kp + (target_dq - dq) * kd

def run_mujoco(cfg, combined_model):
    """
    Run the Mujoco simulation using the provided policy and configuration.

    Returns:
        None
    """
    # Load the Mujoco model
    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
    model.opt.timestep = cfg.sim_config.dt
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    viewer = mujoco_viewer.MujocoViewer(model, data)
    joint_ids = [model.joint(name).id for name in env.joint_names]

    target_q = np.zeros((cfg.env.num_actions), dtype=np.double)
    action = np.zeros((cfg.env.num_actions), dtype=np.double)
    obs_history = torch.zeros(1, env.obs_his, dtype=torch.float)

    # Initialize phase tracking variables
    is_small_speed = True
    phase_shift_steps = 0
    # Single variable for stop transition
    stop_target_phase = None

    # Initialize Pygame for joystick control
    pygame.init()
    pygame.joystick.init()
    joystick = None
    if pygame.joystick.get_count() > 0:
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        print(f"Joystick detected: {joystick.get_name()}")
    else:
        print("No joystick detected. Using default sinusoidal command velocities.")

    count_lowlevel = 0
    total_steps = int(cfg.sim_config.sim_duration / cfg.sim_config.dt)
    for step in tqdm(range(total_steps), desc="Simulating..."):
        # Obtain an observation from Mujoco
        q, dq, quat, v, omega, gvec = get_obs(data)
        q = q[-cfg.env.num_actions:]
        dq = dq[-cfg.env.num_actions:]

        current_time = step * cfg.sim_config.dt  # Calculate current time in seconds

        if joystick:
            pygame.event.pump()
            cmd_velocity_x = -joystick.get_axis(1) * 0.5  # Left stick vertical axis for vx, scaled to [-1, 1] m/s
            cmd_velocity_y = -joystick.get_axis(0) * 0.3  # Left stick horizontal axis for vy, scaled to [-1, 1] m/s
            cmd_velocity_yaw = -joystick.get_axis(3) * 0.5  # Right stick horizontal axis for yaw, scaled to [-2, 2] rad/s
        else:
            cmd_velocity_x = 0.0  # Default command
            cmd_velocity_y = 0.0  # Default command
            cmd_velocity_yaw = 0.0

        # Limit velocities
        cmd_velocity_x = np.clip(cmd_velocity_x, -0.5, 0.5)
        cmd_velocity_y = np.clip(cmd_velocity_y, 0, 0)
        cmd_velocity_yaw = np.clip(cmd_velocity_yaw, -0.5, 0.5)


        if count_lowlevel % cfg.sim_config.decimation == 0:
            obs = torch.zeros(1, env.obs, dtype=torch.float)
            projected_gravity = quat_rotate_inverse(quat, np.array([0., 0., -1.]))

            # 组装当前观测
            obs[0, 0:3] = torch.tensor(omega * cfg.normalization.obs_scales.ang_vel, dtype=torch.float)
            obs[0, 3:6] = torch.tensor(projected_gravity, dtype=torch.float)
            dof_list = [0,1,3,4]
            obs[0, 6:10] = torch.tensor((q[dof_list] - cfg.robot_config.default_joint_angles[dof_list]) * cfg.normalization.obs_scales.dof_pos, dtype=torch.float)
            obs[0, 10:16] = torch.tensor(dq * cfg.normalization.obs_scales.dof_vel, dtype=torch.float)
            obs[0, 16:22] = torch.tensor(action, dtype=torch.float)

            # clip
            obs = torch.clip(obs, -cfg.normalization.clip_observations, cfg.normalization.clip_observations)

            # 拼接历史
            obs_history = torch.cat((obs_history[:, env.obs:], obs[:, :]), dim=-1)  # [1, 220]

            # 拼接当前帧和指令
            command = torch.tensor([[cmd_velocity_x, cmd_velocity_y, cmd_velocity_yaw]], dtype=torch.float32)  # [1, 3]
            input_tensor = torch.cat([obs_history, command], dim=1)  # [1, 245]

            # policy 推理
            with torch.no_grad():
                action_tensor = combined_model(input_tensor)
            action[:] = action_tensor[0].cpu().numpy()

            # clip
            action = np.clip(action, -cfg.normalization.clip_actions, cfg.normalization.clip_actions)
            
            
            
            q = data.qpos[joint_ids].astype(np.float32)
            dq = data.qvel[joint_ids].astype(np.float32)

            # 区分腿和轮
            target_q = np.zeros_like(q)
            target_dq = np.zeros_like(dq)

            # 腿部关节（位置控制）
            leg_idx = [0, 1, 3, 4]
            target_q[leg_idx] = action[leg_idx] * cfg.control.action_scale_pos + cfg.robot_config.default_joint_angles[leg_idx]
            target_dq[leg_idx] = 0.0

            # 轮子关节（速度控制）
            wheel_idx = [2, 5]
            target_q[wheel_idx] = 0.0
            target_dq[wheel_idx] = action[wheel_idx] * cfg.control.action_scale_vel

        tau = pd_control(target_q, q, cfg.robot_config.kps, target_dq, dq, cfg.robot_config.kds)
        tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)
        data.ctrl = tau
        mujoco.mj_step(model, data)
        viewer.render()
        count_lowlevel += 1

    viewer.close()
    pygame.quit()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Deployment script for wheellegged_flat.")
    parser.add_argument(
        "--logdir",
        type=str,
        required=False,
        default=f"{LEGGED_GYM_ROOT_DIR}/logs/pointfoot_flat/WL/exported/policies",
        help="Run to load from.",
    )
    parser.add_argument("--terrain", action="store_true", help="terrain or plane")
    args = parser.parse_args()

    class Sim2simCfg(BipedCfgWL):

        class sim_config:
            if args.terrain:
                mujoco_model_path = f"{LEGGED_GYM_ROOT_DIR}/resources/robots/wheel_legged/mjcf/whee_legged_plane.xml"
            else:
                mujoco_model_path = f"{LEGGED_GYM_ROOT_DIR}/resources/robots/wheel_legged/mjcf/whee_legged_plane.xml"

            sim_duration =100.0
            dt = 0.005
            decimation = 4

        class robot_config:
            kps = np.array([20, 20, 0, 20, 20, 0], dtype=np.double) 
            kds = np.array([0.5, 0.5, 0.8, 0.5, 0.5, 0.8], dtype=np.double)

            default_joint_angles = np.array([0., 0., 0., 0., 0., 0.], dtype=np.double) # v8
            tau_limit = np.array([40, 40, 40, 40, 40, 40], dtype=np.double) 

    # Load models
    combined_model = torch.jit.load(args.logdir + '/combined_policy.pt')
    combined_model.eval()

    run_mujoco(Sim2simCfg(), combined_model)
