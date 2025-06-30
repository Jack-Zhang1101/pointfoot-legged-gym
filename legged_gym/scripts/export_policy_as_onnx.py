from legged_gym import LEGGED_GYM_ROOT_DIR
import os, sys

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger, get_load_path, class_to_dict
from legged_gym.algorithm.mlp_encoder import MLP_Encoder
from legged_gym.algorithm.actor_critic import ActorCritic

import numpy as np
import torch
import copy
import torch.nn as nn

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

def export_policy_as_onnx(args, robot_type):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', args.task, robot_type)
    resume_path = get_load_path(log_root, load_run=args.load_run, checkpoint=args.checkpoint)
    loaded_dict = torch.load(resume_path)
    export_path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', args.task, train_cfg.runner.experiment_name, 'exported', 'policies')
    os.makedirs(export_path, exist_ok=True)
    
    # encoder
    encoder_class = eval(train_cfg.runner.encoder_class_name)
    encoder = encoder_class(**class_to_dict(train_cfg)[train_cfg.runner.encoder_class_name]).to(args.rl_device)
    encoder.load_state_dict(loaded_dict['encoder_state_dict'])
    encoder_path = os.path.join(export_path, "encoder.onnx")
    encoder_model = copy.deepcopy(encoder.encoder).to("cpu")
    encoder_model.eval()
    dummy_input = torch.randn(encoder.num_input_dim)
    input_names = ["nn_input"]
    output_names = ["nn_output"]
    
    torch.onnx.export(
        encoder_model,
        dummy_input,
        encoder_path,
        verbose=True,
        input_names=input_names,
        output_names=output_names,
        export_params=True,
        opset_version=13,
    )
    print("Exported encoder as onnx script to: ", encoder_path)

    # actor_critic
    actor_critic_class = eval(train_cfg.runner.policy_class_name)

    actor_input_dim = env_cfg.env.num_observations + encoder.num_output_dim + env_cfg.commands.num_commands
    critic_input_dim = env_cfg.env.num_critic_observations + env_cfg.commands.num_commands + encoder.num_output_dim
    actor_critic = actor_critic_class(
        actor_input_dim, critic_input_dim, env_cfg.env.num_actions, **class_to_dict(train_cfg.policy)
    ).to(args.rl_device)
    actor_critic.load_state_dict(loaded_dict['model_state_dict'])
    print()
    # export policy as an onnx file
    policy_path = os.path.join(export_path, "policy.onnx")
    model = copy.deepcopy(actor_critic.actor).to("cpu")
    model.eval()

    dummy_input = torch.randn(actor_input_dim)
    input_names = ["nn_input"]
    output_names = ["nn_output"]

    torch.onnx.export(
        model,
        dummy_input,
        policy_path,
        verbose=True,
        input_names=input_names,
        output_names=output_names,
        export_params=True,
        opset_version=13,
    )
    print("Exported policy as onnx script to: ", policy_path)

    # 导出合并的模型
    print("\n开始导出合并的ONNX模型...")
    obs_dim = env_cfg.env.num_observations
    obs_history_steps = env_cfg.env.obs_history_length + 1  # 11步
    command_dim = env_cfg.commands.num_commands
    total_dim = obs_dim * obs_history_steps + command_dim  # 245
    print(f"总输入维度: {total_dim}")
    encoder_cpu = copy.deepcopy(encoder).to("cpu")
    actor_cpu = copy.deepcopy(actor_critic.actor).to("cpu")
    combined = CombinedPolicy(encoder_cpu, actor_cpu, obs_dim, obs_history_steps, command_dim)
    combined.eval()
    combined_path = os.path.join(export_path, "combined_policy.onnx")
    dummy_input = torch.randn(1, total_dim)
    torch.onnx.export(
        combined,
        dummy_input,
        combined_path,
        input_names=["input"],
        output_names=["actions"],
        opset_version=13,
        verbose=True,
        export_params=True,
    )
    print("Exported combined policy as onnx script to: ", combined_path)
    print("合并模型导出完成！")

if __name__ == '__main__':
    
    # check ROBOT_TYPE validity
    robot_type = os.getenv("ROBOT_TYPE")
    if not robot_type:
        print("\033[1m\033[31mError: Please set the ROBOT_TYPE using 'export ROBOT_TYPE=<robot_type>'.\033[0m")
        sys.exit(1)

    if not robot_type in ["PF_TRON1A", "PF_P441A", "PF_P441B", "PF_P441C", "PF_P441C2", "SF_TRON1A", "WF_TRON1A","WL"]:
        print("\033[1m\033[31mError: Input ROBOT_TYPE={}".format(robot_type), 
        "is not among valid robot types WF_TRON1A, SF_TRON1A, PF_TRON1A, PF_P441A, PF_P441B, PF_P441C, PF_P441C2.\033[0m")
        sys.exit(1)
    args = get_args()
    export_policy_as_onnx(args, robot_type)
