# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from isaacgym.torch_utils import *
from legged_gym.envs import *
from legged_gym.utils import (
    get_args,
    export_policy_as_jit,
    export_mlp_as_onnx,
    task_registry,
    Logger,
)

import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import copy

class CombinedPolicy(nn.Module):
    """组合策略网络，输入为[batch, 245]，前242为观测历史（11帧），后3为指令"""
    def __init__(self, encoder, actor, obs_dim, obs_history_steps, command_dim):
        super().__init__()
        self.encoder = encoder
        self.actor = actor
        self.obs_dim = obs_dim
        self.obs_history_steps = obs_history_steps
        self.command_dim = command_dim
        self.history_dim = obs_dim * obs_history_steps

    def forward(self, input_tensor):
        # input_tensor: [B, 245]，前242为观测历史（11帧），后3为指令
        # encoder只吃前10帧（220维），obs为最后1帧（22维）
        obs_history = input_tensor[:, :self.obs_dim * (self.obs_history_steps - 1)]  # 10帧
        obs = input_tensor[:, self.obs_dim * (self.obs_history_steps - 1):self.obs_dim * self.obs_history_steps]  # 第11帧
        commands = input_tensor[:, self.obs_dim * self.obs_history_steps:]
        latent = self.encoder(obs_history)
        x = torch.cat([latent, obs, commands], dim=-1)
        actions = self.actor(x)
        return actions

def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.episode_length_s = 30
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 100)

    env_cfg.terrain.num_rows = 10
    env_cfg.terrain.num_cols = 20
    env_cfg.terrain.terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
    env_cfg.terrain.max_init_terrain_level = 4
    env_cfg.terrain.curriculum = True
    env_cfg.noise.add_noise = True
    env_cfg.noise.noise_level = 0.5
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_restitution = False
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.push_interval_s = 3
    env_cfg.domain_rand.randomize_Kp = False
    env_cfg.domain_rand.randomize_Kd = False
    env_cfg.domain_rand.randomize_motor_torque = False
    env_cfg.domain_rand.randomize_default_dof_pos = False
    env_cfg.domain_rand.randomize_action_delay = False

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    # get robot_type
    robot_type = os.getenv("ROBOT_TYPE")
    commands_val = to_torch([1.0, 0.0, 0.0], device=env.device) if robot_type in ["Tita", "WF_TRON1A","WL"]  else to_torch([0.5, 0.0, 0, 0], device=env.device) if robot_type.startswith("PF") else to_torch([1.5, 0.0, 0.0, 0.0, 0.0])
    action_scale = env.cfg.control.action_scale_pos if robot_type in ["Tita", "WF_TRON1A","WL"] else env.cfg.control.action_scale
    obs, obs_history, commands, _ = env.get_observations()

    # 直接加载 torchscript 导出的合并模型
    policy_path = os.path.join(
        LEGGED_GYM_ROOT_DIR,
        "logs",
        args.task,
        train_cfg.runner.experiment_name,
        "exported",
        "policies",
        "combined_policy.pt"
    )
    policy = torch.jit.load(policy_path)
    policy.eval()
    print("Loaded combined_policy.pt from:", policy_path)

    logger = Logger(env.dt)
    robot_index = 5  # which robot is used for logging
    joint_index = 1  # which joint is used for logging
    stop_state_log = 100  # number of steps before plotting states
    stop_rew_log = (
        env.max_episode_length + 1
    )  # number of steps before print average episode rewards
    # camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    # camera_vel = np.array([1.0, 1.0, 0.0])
    # camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0
    est = None
    for i in range(10 * int(env.max_episode_length)):
        # 组装 input_tensor
        print("play.py obs_history:", obs_history[0].cpu().numpy())
        print("play.py obs:", obs[0].cpu().numpy())
        print("play.py command:", commands[0].cpu().numpy())
        input_tensor = torch.cat((obs_history, obs, commands), dim=1)
        print("play.py input_tensor:", input_tensor[0].cpu().numpy())
        with torch.no_grad():
            actions = policy(input_tensor)
        print("play.py action:", actions[0].cpu().numpy())

        env.commands[:, :] = commands_val

        obs, rews, dones, infos, obs_history, commands, _ = env.step(
            actions.detach()
        )
        if False:  # RECORD_FRAMES
            if i % 2:
                filename = os.path.join(
                    LEGGED_GYM_ROOT_DIR,
                    "logs",
                    train_cfg.runner.experiment_name,
                    "exported",
                    "frames",
                    f"{img_idx}.png",
                )
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1
        if False:  # MOVE_CAMERA
            camera_offset = np.array(env_cfg.viewer.pos)
            target_position = np.array(
                env.base_position[robot_index, :].to(device="cpu")
            )
            target_position[2] = 0
            camera_position = target_position + camera_offset
            # env.set_camera(camera_position, target_position)

        if i < stop_state_log:
            logger.log_states(
                {
                    "dof_pos_target": actions[robot_index, joint_index].item() * action_scale,
                    "dof_pos": (
                        env.dof_pos[robot_index, joint_index]
                        - env.raw_default_dof_pos[joint_index]
                    ).item(),
                    "dof_vel": env.dof_vel[robot_index, joint_index].item(),
                    "dof_torque": env.torques[robot_index, joint_index].item(),
                    "command_x": env.commands[robot_index, 0].item(),
                    "command_y": env.commands[robot_index, 1].item(),
                    "command_yaw": env.commands[robot_index, 2].item(),
                    "base_vel_x": env.base_lin_vel[robot_index, 0].item(),
                    "base_vel_y": env.base_lin_vel[robot_index, 1].item(),
                    "base_vel_z": env.base_lin_vel[robot_index, 2].item(),
                    "base_vel_yaw": env.base_ang_vel[robot_index, 2].item(),
                    "power": torch.sum(env.power[robot_index, :]).item(),
                    "contact_forces_z": env.contact_forces[
                        robot_index, env.feet_indices, 2
                    ]
                    .cpu()
                    .numpy(),
                }
            )
            # print(torch.sum(env.power[robot_index, :]).item())
            if est != None:
                logger.log_states(
                    {
                        "est_lin_vel_x": est[robot_index, 0].item()
                        / env.cfg.normalization.obs_scales.lin_vel,
                        "est_lin_vel_y": est[robot_index, 1].item()
                        / env.cfg.normalization.obs_scales.lin_vel,
                        "est_lin_vel_z": est[robot_index, 2].item()
                        / env.cfg.normalization.obs_scales.lin_vel,
                    }
                )
        elif i == stop_state_log:
            logger.plot_states()

        if 0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes > 0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i == stop_rew_log:
            logger.print_rewards()


if __name__ == "__main__":
    args = get_args()
    play(args)
