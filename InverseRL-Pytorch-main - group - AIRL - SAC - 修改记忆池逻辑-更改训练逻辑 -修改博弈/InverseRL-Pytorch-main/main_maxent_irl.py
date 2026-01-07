"""
深度最大熵逆强化学习 (Deep Maximum Entropy IRL) 基线训练脚本
基于 Ziebart et al. (2008) "Maximum Entropy Inverse Reinforcement Learning"
以及 Wulfmeier et al. (2015) "Maximum Entropy Deep Inverse Reinforcement Learning"

不使用ARG和博弈模块，最基础的MaxEnt IRL实现
"""

from agents.algorithm.ppo import PPO
from agents.algorithm.sac import SAC
from agents.algorithm.ddpg import DDPG
from agents.algorithm.td3 import TD3

from utils.utils import RunningMeanStd, Dict, make_transition

from configparser import ConfigParser
from argparse import ArgumentParser

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ============== 深度最大熵IRL ==============
from discriminators.base import Discriminator


class RewardNetwork(nn.Module):
    """
    奖励网络：学习状态到奖励的映射
    r(s) = f_theta(s)
    """
    def __init__(self, state_dim, hidden_dim=256):
        super(RewardNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        return self.network(state)


class DeepMaxEntIRL(Discriminator):
    """
    深度最大熵逆强化学习
    
    核心思想：
    1. 最大化专家轨迹的似然
    2. 同时最大化策略的熵（使得策略尽可能随机地达到高奖励状态）
    
    目标函数：
    max_theta E_expert[r_theta(s)] - log Z(theta)
    
    其中 Z(theta) 是配分函数，通过采样近似
    
    梯度更新：
    grad = E_expert[grad_r(s)] - E_policy[grad_r(s)]
    """
    def __init__(self, writer, device, state_dim, action_dim, args):
        super(DeepMaxEntIRL, self).__init__()
        self.writer = writer
        self.device = device
        self.args = args
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 奖励网络
        self.reward_net = RewardNetwork(state_dim, hidden_dim=args.hidden_dim)
        
        # 优化器
        self.optimizer = optim.Adam(self.reward_net.parameters(), lr=args.lr)
        
        # 用于奖励归一化
        self.reward_mean = 0.0
        self.reward_std = 1.0
        
    def forward(self, state):
        """计算状态的奖励值"""
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state).float()
        state = state.float().to(self.device)
        return self.reward_net(state)
    
    def get_reward(self, state):
        """获取归一化后的奖励"""
        with torch.no_grad():
            if not isinstance(state, torch.Tensor):
                state = torch.tensor(state).float()
            state = state.float().to(self.device)
            reward = self.reward_net(state)
            # 归一化奖励以稳定训练
            reward = (reward - self.reward_mean) / (self.reward_std + 1e-8)
        return reward.detach()
    
    def compute_state_visitation_freq(self, states, gamma=0.99):
        """
        计算状态访问频率（简化版本）
        使用折扣因子加权
        """
        T = len(states)
        weights = torch.tensor([gamma ** t for t in range(T)]).float().to(self.device)
        weights = weights / weights.sum()
        return weights
    
    def train_network(self, writer, n_epi, agent_states, expert_states):
        """
        训练奖励网络
        
        使用最大熵IRL的梯度：
        grad L = E_expert[grad r(s)] - E_agent[grad r(s)]
        
        这等价于最小化：
        L = -E_expert[r(s)] + E_agent[r(s)]
        
        即：让专家状态的奖励高，让agent状态的奖励低
        """
        # 转换为tensor
        if not isinstance(agent_states, torch.Tensor):
            agent_states = torch.tensor(agent_states).float()
        if not isinstance(expert_states, torch.Tensor):
            expert_states = torch.tensor(expert_states).float()
        
        agent_states = agent_states.to(self.device)
        expert_states = expert_states.to(self.device)
        
        # 计算专家状态的奖励
        expert_rewards = self.reward_net(expert_states)
        
        # 计算agent状态的奖励
        agent_rewards = self.reward_net(agent_states)
        
        # MaxEnt IRL 损失：最大化专家奖励，最小化agent奖励
        # L = -E_expert[r(s)] + log(E_agent[exp(r(s))])
        # 简化版本：L = -mean(r_expert) + mean(r_agent)
        
        # 使用importance sampling近似配分函数
        # 更稳定的损失函数
        expert_loss = -expert_rewards.mean()
        
        # 使用log-sum-exp技巧来近似配分函数
        # log Z ≈ log(1/N * sum(exp(r))) = logsumexp(r) - log(N)
        agent_loss = torch.logsumexp(agent_rewards.squeeze(), dim=0) - np.log(len(agent_rewards))
        
        # 总损失
        loss = expert_loss + agent_loss
        
        # 添加L2正则化
        l2_reg = 0.0
        for param in self.reward_net.parameters():
            l2_reg += torch.norm(param)
        loss = loss + self.args.l2_reg * l2_reg
        
        # 更新奖励网络
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.reward_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        
        # 更新奖励统计量用于归一化
        with torch.no_grad():
            all_rewards = torch.cat([expert_rewards, agent_rewards])
            self.reward_mean = all_rewards.mean().item()
            self.reward_std = all_rewards.std().item()
        
        # 记录日志
        if self.writer is not None:
            self.writer.add_scalar("loss/maxent_total_loss", loss.item(), n_epi)
            self.writer.add_scalar("loss/expert_reward_mean", expert_rewards.mean().item(), n_epi)
            self.writer.add_scalar("loss/agent_reward_mean", agent_rewards.mean().item(), n_epi)
            self.writer.add_scalar("loss/reward_diff", 
                                   (expert_rewards.mean() - agent_rewards.mean()).item(), n_epi)
    
    def save(self, filename):
        torch.save(self.reward_net.state_dict(), filename + "_reward_net")
        torch.save(self.optimizer.state_dict(), filename + "_reward_optimizer")
        torch.save({
            'reward_mean': self.reward_mean,
            'reward_std': self.reward_std
        }, filename + "_reward_stats")

    def load(self, filename):
        self.reward_net.load_state_dict(torch.load(filename + "_reward_net"))
        self.optimizer.load_state_dict(torch.load(filename + "_reward_optimizer"))
        stats = torch.load(filename + "_reward_stats")
        self.reward_mean = stats['reward_mean']
        self.reward_std = stats['reward_std']


class DeepMaxEntIRLv2(Discriminator):
    """
    深度最大熵IRL的另一种实现方式
    使用Guided Cost Learning的思想
    
    这个版本更接近原始论文，考虑了轨迹级别的奖励
    """
    def __init__(self, writer, device, state_dim, action_dim, args):
        super(DeepMaxEntIRLv2, self).__init__()
        self.writer = writer
        self.device = device
        self.args = args
        self.gamma = args.gamma
        
        # 奖励网络
        self.reward_net = nn.Sequential(
            nn.Linear(state_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, 1)
        )
        
        self.optimizer = optim.Adam(self.reward_net.parameters(), lr=args.lr)
        self.reward_mean = 0.0
        self.reward_std = 1.0
        
    def forward(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state).float()
        return self.reward_net(state.to(self.device))
    
    def get_reward(self, state):
        with torch.no_grad():
            if not isinstance(state, torch.Tensor):
                state = torch.tensor(state).float()
            reward = self.reward_net(state.to(self.device))
            reward = (reward - self.reward_mean) / (self.reward_std + 1e-8)
        return reward.detach()
    
    def train_network(self, writer, n_epi, agent_states, expert_states):
        """
        使用对比学习的方式训练
        类似于NCE (Noise Contrastive Estimation)
        """
        if not isinstance(agent_states, torch.Tensor):
            agent_states = torch.tensor(agent_states).float()
        if not isinstance(expert_states, torch.Tensor):
            expert_states = torch.tensor(expert_states).float()
        
        agent_states = agent_states.to(self.device)
        expert_states = expert_states.to(self.device)
        
        # 计算奖励
        expert_rewards = self.reward_net(expert_states)
        agent_rewards = self.reward_net(agent_states)
        
        # 使用margin loss：专家奖励应该比agent奖励高
        # max(0, margin - (r_expert - r_agent))
        margin = self.args.margin
        
        # 随机配对计算margin loss
        batch_size = min(len(expert_rewards), len(agent_rewards))
        indices = torch.randperm(batch_size)
        
        expert_r = expert_rewards[:batch_size]
        agent_r = agent_rewards[indices][:batch_size]
        
        margin_loss = F.relu(margin - (expert_r - agent_r)).mean()
        
        # 熵正则化：鼓励奖励分布更平滑
        entropy_reg = -self.args.entropy_weight * (
            F.softmax(expert_rewards.squeeze(), dim=0) * 
            F.log_softmax(expert_rewards.squeeze(), dim=0)
        ).sum()
        
        loss = margin_loss - entropy_reg
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.reward_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        
        # 更新统计量
        with torch.no_grad():
            all_rewards = torch.cat([expert_rewards, agent_rewards])
            self.reward_mean = all_rewards.mean().item()
            self.reward_std = all_rewards.std().item()
        
        if self.writer is not None:
            self.writer.add_scalar("loss/margin_loss", margin_loss.item(), n_epi)
            self.writer.add_scalar("loss/expert_reward", expert_rewards.mean().item(), n_epi)
            self.writer.add_scalar("loss/agent_reward", agent_rewards.mean().item(), n_epi)

    def save(self, filename):
        torch.save(self.reward_net.state_dict(), filename + "_reward_net")
        torch.save(self.optimizer.state_dict(), filename + "_reward_optimizer")

    def load(self, filename):
        self.reward_net.load_state_dict(torch.load(filename + "_reward_net"))
        self.optimizer.load_state_dict(torch.load(filename + "_reward_optimizer"))


# ============== 简化的Agent类 ==============
class AgentMaxEntIRL(nn.Module):
    """用于MaxEnt IRL的Agent"""
    def __init__(self, algorithm, writer, device, state_dim, action_dim, args, demonstrations_location_args): 
        super(AgentMaxEntIRL, self).__init__()
        self.writer = writer
        self.device = device
        self.args = args
        
        from utils.utils import ReplayBuffer
        self.data = ReplayBuffer(
            action_prob_exist=False, 
            max_size=int(self.args.memory_size), 
            state_dim=state_dim, 
            num_action=action_dim
        )
        
        # 加载专家数据
        self.expert_actions = np.load(
            demonstrations_location_args.expert_action_location, allow_pickle=True)
        self.expert_done = np.load(
            demonstrations_location_args.expert_done_location, allow_pickle=True)
        self.expert_state_flat = np.load(
            demonstrations_location_args.expert_state_flat, allow_pickle=True)
        self.expert_next_state_flat = np.load(
            demonstrations_location_args.expert_next_state_flat, allow_pickle=True)
        
        self.brain = algorithm
        
    def get_action(self, x):
        action, log_prob = self.brain.get_action(x)
        return action, log_prob
    
    def put_data(self, transition):
        self.data.put_data(transition)

    def train(self, discriminator, discriminator_batch_size, state_rms, n_epi, batch_size=64):
        """训练MaxEnt IRL"""
        # 采样agent数据
        data = self.data.sample(shuffle=True, batch_size=discriminator_batch_size)
        agent_states = torch.tensor(data['state']).float().to(self.device)
        
        # 采样expert数据
        expert_indices = np.random.choice(
            len(self.expert_state_flat), 
            size=discriminator_batch_size, 
            replace=False
        )
        expert_states = torch.tensor(self.expert_state_flat[expert_indices]).float().to(self.device)
        
        # 训练奖励网络
        discriminator.train_network(self.writer, n_epi, agent_states, expert_states)
        
        # 训练策略网络
        data = self.data.sample(shuffle=True, batch_size=batch_size)
        states = torch.tensor(data['state']).float().to(self.device)
        actions = torch.tensor(data['action']).float().to(self.device)
        next_states = torch.tensor(data['next_state']).float().to(self.device)
        done_masks = torch.tensor(data['done']).float().to(self.device)
        
        # 使用学到的奖励
        rewards = discriminator.get_reward(states)
        
        self.brain.train_network(self.writer, n_epi, states, actions, rewards, next_states, done_masks)


# ============== 环境类 ==============
from envDesign.environment import InteractionEnv
from envDesign.environmentTest import InteractionEnvForTest


# ============== 主函数 ==============
def main():
    os.makedirs('./model_weights_maxent_irl', exist_ok=True)

    # 加载环境
    env = InteractionEnv(
        '/mnt/f/公开数据集/Yandex/人车交互数据/环境数据/env优化/env_state_pedestrain_train.npy',
        '/mnt/f/公开数据集/Yandex/人车交互数据/环境数据/env优化/env_state_vehicle_train.npy',
        '/mnt/f/公开数据集/Yandex/人车交互数据/环境数据/env优化/env_position_pedestrain_train.npy',
        '/mnt/f/公开数据集/Yandex/人车交互数据/环境数据/env优化/env_position_vehicle_train.npy',
        '/mnt/f/公开数据集/Yandex/人车交互数据/环境数据/env优化/env_done_group_train.npy',
        '/mnt/f/公开数据集/Yandex/人车交互数据/环境数据/env优化/env_num_pedestrain_train.npy',
        '/mnt/f/公开数据集/Yandex/人车交互数据/环境数据/env优化/env_num_vehicle_train.npy'
    )
    print('训练环境读取完成')
    
    envTest = InteractionEnvForTest(
        '/mnt/f/公开数据集/Yandex/人车交互数据/环境数据/env优化/env_state_pedestrain_train.npy',
        '/mnt/f/公开数据集/Yandex/人车交互数据/环境数据/env优化/env_state_vehicle_train.npy',
        '/mnt/f/公开数据集/Yandex/人车交互数据/环境数据/env优化/env_position_pedestrain_train.npy',
        '/mnt/f/公开数据集/Yandex/人车交互数据/环境数据/env优化/env_position_vehicle_train.npy',
        '/mnt/f/公开数据集/Yandex/人车交互数据/环境数据/env优化/env_done_group_train.npy',
        '/mnt/f/公开数据集/Yandex/人车交互数据/环境数据/env优化/env_num_pedestrain_train.npy',
        '/mnt/f/公开数据集/Yandex/人车交互数据/环境数据/env优化/env_num_vehicle_train.npy'
    )
    print('测试环境读取完成')

    action_dim = env.action_dim
    state_dim = env.state_dim
    
    # 参数设置
    parser = ArgumentParser('parameters')
    parser.add_argument('--epochs', type=int, default=100001)
    parser.add_argument("--agent", type=str, default='sac')
    parser.add_argument("--irl_version", type=str, default='v1', 
                        help='v1: 标准MaxEnt IRL, v2: Margin-based')
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument('--tensorboard', type=bool, default=True)
    args = parser.parse_args()
    
    # 读取配置
    config_parser = ConfigParser()
    config_parser.read('config.ini')
    
    torch.manual_seed(0)
    np.random.seed(0)
    
    demonstrations_location_args = Dict(config_parser, 'demonstrations_location', True)
    agent_args = Dict(config_parser, args.agent)
    
    # MaxEnt IRL的参数
    class MaxEntIRLArgs:
        lr = 0.0003
        hidden_dim = 256
        gamma = 0.99
        l2_reg = 0.001  # L2正则化系数
        margin = 1.0    # margin loss的margin值
        entropy_weight = 0.01  # 熵正则化权重
        is_airl = False
        batch_size = 64
    
    irl_args = MaxEntIRLArgs()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # TensorBoard
    writer = None
    if args.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter('runs_maxent_irl')
    
    # 创建MaxEnt IRL
    if args.irl_version == 'v1':
        irl = DeepMaxEntIRL(writer, device, state_dim, action_dim, irl_args)
        print('深度最大熵IRL (v1 - 标准版) 构建完成')
    else:
        irl = DeepMaxEntIRLv2(writer, device, state_dim, action_dim, irl_args)
        print('深度最大熵IRL (v2 - Margin版) 构建完成')
    
    # 创建强化学习算法
    max_action = 2
    if args.agent == 'sac':
        algorithm = SAC(device, state_dim, action_dim, agent_args)
    elif args.agent == 'td3':
        algorithm = TD3(state_dim, action_dim, max_action)
    else:
        raise NotImplementedError
    print('强化学习算法构建完成')
    
    # 创建Agent
    agent = AgentMaxEntIRL(algorithm, writer, device, state_dim, action_dim, agent_args, demonstrations_location_args)
    print('智能体构建完成')
    
    if device == 'cuda':
        agent = agent.cuda()
        irl = irl.cuda()
    
    # 训练循环
    score_lst = []
    irl_score_lst = []
    
    for n_epi in range(args.epochs):
        score = 0.0
        irl_score = 0.0
        
        state_flat, all_state = env.reset()
        done = False
        
        while not done:
            # 获取动作
            action, log_prob = agent.get_action(torch.from_numpy(state_flat).float().to(device))
            action = torch.tensor(action).cpu().detach().numpy()[0]
            
            # 执行动作
            next_state_flat, r, done, info, next_all_state = env.step(action)
            
            # 计算IRL奖励
            reward = irl.get_reward(
                torch.tensor(state_flat).unsqueeze(0).float().to(device)
            ).item()
            
            # 存储transition
            transition = make_transition(
                state_flat,
                action,
                np.array([reward]),
                next_state_flat,
                np.array([done]),
                all_state[0],
                next_all_state[0]
            )
            agent.put_data(transition)
            
            state_flat = next_state_flat
            all_state = next_all_state
            
            score += r
            irl_score += reward
            
            # 训练
            if agent.data.data_idx > agent_args.learn_start_size:
                if agent.data.data_idx >= agent_args.memory_size:
                    agent.train(irl, irl_args.batch_size, None, n_epi, agent_args.batch_size)
        
        score_lst.append(score)
        irl_score_lst.append(irl_score)
        
        # 评估
        if n_epi % 10 == 0:
            ls_tra_mae = []
            ls_speed_mae = []
            ls_tra_hd = []
            ls_speed_hd = []
            
            for i in range(envTest.env_state_pedestrain.shape[0]):
                state_flat_test, all_state_test = envTest.reset()
                doneTest = False
                
                while not doneTest:
                    action, _ = agent.get_action(torch.from_numpy(envTest.state).float().to(device))
                    action = torch.tensor(action).cpu().detach().numpy()[0]
                    tra_mae, speed_mae, tra_hd, speed_hd, doneTest = envTest.step(action)
                
                ls_tra_mae.append(tra_mae)
                ls_speed_mae.append(speed_mae)
                ls_tra_hd.append(tra_hd)
                ls_speed_hd.append(speed_hd)
            
            tra_mae = np.mean(np.array(ls_tra_mae), axis=0)
            speed_mae = np.mean(np.array(ls_speed_mae), axis=0)
            tra_hd = np.mean(np.array(ls_tra_hd), axis=0)
            speed_hd = np.mean(np.array(ls_speed_hd), axis=0)
            
            if writer:
                writer.add_scalar("Metric/P_X_MAE", tra_mae[0], n_epi)
                writer.add_scalar("Metric/P_Y_MAE", tra_mae[1], n_epi)
                writer.add_scalar("Metric/P_X_HD", tra_hd[0], n_epi)
                writer.add_scalar("Metric/P_Y_HD", tra_hd[1], n_epi)
                writer.add_scalar("Metric/V_X_MAE", speed_mae[0], n_epi)
                writer.add_scalar("Metric/V_Y_MAE", speed_mae[1], n_epi)
                writer.add_scalar("Metric/V_X_HD", speed_hd[0], n_epi)
                writer.add_scalar("Metric/V_Y_HD", speed_hd[1], n_epi)
            
            print(f'Episode {n_epi}:')
            print(f'  P_X_MAE:{tra_mae[0]:.4f}; P_Y_MAE:{tra_mae[1]:.4f}')
            print(f'  V_X_MAE:{speed_mae[0]:.4f}; V_Y_MAE:{speed_mae[1]:.4f}')
            print(f'  P_X_HD:{tra_hd[0]:.4f}; P_Y_HD:{tra_hd[1]:.4f}')
            print(f'  V_X_HD:{speed_hd[0]:.4f}; V_Y_HD:{speed_hd[1]:.4f}')
            
            # 保存模型
            if (tra_mae[0] < 1.5) and (tra_mae[1] < 1.5) or (n_epi % 1000 == 0):
                save_name = f'./model_weights_maxent_irl/maxent_irl_{n_epi}'
                irl.save(save_name)
                agent.brain.save(save_name)
        
        if writer:
            writer.add_scalar("score/score", score, n_epi)
            writer.add_scalar("score/irl_reward", irl_score, n_epi)
        
        print(f"Episode {n_epi}: score={score:.1f}, irl_score={irl_score:.1f}")
        score_lst = []
        irl_score_lst = []


if __name__ == "__main__":
    main()
