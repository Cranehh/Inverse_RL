"""
基础GAIL基线训练脚本
不使用ARG和博弈模块，最基础的GAIL实现
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

# ============== 基础GAIL判别器 ==============
from discriminators.base import Discriminator


class GAILBase(Discriminator):
    """
    基础版本的GAIL，不使用ARG和GCN模块
    直接使用MLP网络处理state
    """
    def __init__(self, writer, device, state_dim, action_dim, args):
        super(GAILBase, self).__init__()
        self.writer = writer
        self.device = device
        self.args = args
        self.args.is_airl = False  # 确保标记为非AIRL
        
        # 基础MLP网络
        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.lr)

    def forward(self, state_flat):
        if not isinstance(state_flat, torch.Tensor):
            state_flat = torch.tensor(state_flat).float()
        state_flat = state_flat.float().to(self.device)
        return self.network(state_flat)
    
    def get_reward(self, state_flat):
        """GAIL奖励: -log(D(s))"""
        with torch.no_grad():
            if not isinstance(state_flat, torch.Tensor):
                state_flat = torch.tensor(state_flat).float()
            d = self.forward(state_flat)
            reward = -torch.log(d + 1e-8)
        return reward.detach()
    
    def train_network(self, writer, n_epi, agent_state_flat, expert_state_flat):
        """训练判别器"""
        # 转换为tensor
        if not isinstance(agent_state_flat, torch.Tensor):
            agent_state_flat = torch.tensor(agent_state_flat).float()
        if not isinstance(expert_state_flat, torch.Tensor):
            expert_state_flat = torch.tensor(expert_state_flat).float()
            
        expert_preds = self.forward(expert_state_flat)
        expert_loss = self.criterion(
            expert_preds, 
            torch.zeros(expert_preds.shape[0], 1).to(self.device)
        )

        agent_preds = self.forward(agent_state_flat)
        agent_loss = self.criterion(
            agent_preds, 
            torch.ones(agent_preds.shape[0], 1).to(self.device)
        )
        
        loss = expert_loss + agent_loss
        
        expert_acc = ((expert_preds < 0.5).float()).mean()
        learner_acc = ((agent_preds > 0.5).float()).mean()
        
        if self.writer is not None:
            self.writer.add_scalar("loss/discriminator_loss", loss.item(), n_epi)
            self.writer.add_scalar("loss/expert_acc", expert_acc.item(), n_epi)
            self.writer.add_scalar("loss/learner_acc", learner_acc.item(), n_epi)
        
        if (expert_acc > 0.8) and (learner_acc > 0.8):
            return 
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, filename):
        torch.save(self.network.state_dict(), filename + "_gail_network")
        torch.save(self.optimizer.state_dict(), filename + "_gail_optimizer")

    def load(self, filename):
        self.network.load_state_dict(torch.load(filename + "_gail_network"))
        self.optimizer.load_state_dict(torch.load(filename + "_gail_optimizer"))


# ============== 简化的Agent类 ==============
class AgentGAIL(nn.Module):
    """简化的Agent类，专门用于基础GAIL"""
    def __init__(self, algorithm, writer, device, state_dim, action_dim, args, demonstrations_location_args): 
        super(AgentGAIL, self).__init__()
        self.writer = writer
        self.device = device
        self.args = args
        
        # Replay Buffer
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
        """训练GAIL"""
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
        
        # 训练判别器
        discriminator.train_network(self.writer, n_epi, agent_states, expert_states)
        
        # 训练策略网络（使用判别器奖励）
        data = self.data.sample(shuffle=True, batch_size=batch_size)
        states = torch.tensor(data['state']).float().to(self.device)
        actions = torch.tensor(data['action']).float().to(self.device)
        next_states = torch.tensor(data['next_state']).float().to(self.device)
        done_masks = torch.tensor(data['done']).float().to(self.device)
        
        # 使用判别器计算奖励
        rewards = discriminator.get_reward(states)
        
        self.brain.train_network(self.writer, n_epi, states, actions, rewards, next_states, done_masks)


# ============== 环境类（从原代码复制）==============
from envDesign.environment import InteractionEnv
from envDesign.environmentTest import InteractionEnvForTest


# ============== 主函数 ==============
def main():
    os.makedirs('./model_weights_gail_base', exist_ok=True)

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
    
    # GAIL的参数
    class GAILArgs:
        lr = 0.0003
        is_airl = False
        batch_size = 64
    
    discriminator_args = GAILArgs()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # TensorBoard
    writer = None
    if args.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter('runs_gail_base')
    
    # 创建基础GAIL判别器
    discriminator = GAILBase(writer, device, state_dim, action_dim, discriminator_args)
    print('基础GAIL判别器构建完成')
    
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
    agent = AgentGAIL(algorithm, writer, device, state_dim, action_dim, agent_args, demonstrations_location_args)
    print('智能体构建完成')
    
    if device == 'cuda':
        agent = agent.cuda()
        discriminator = discriminator.cuda()
    
    # 训练循环
    score_lst = []
    discriminator_score_lst = []
    
    for n_epi in range(args.epochs):
        score = 0.0
        discriminator_score = 0.0
        
        state_flat, all_state = env.reset()
        done = False
        
        while not done:
            # 获取动作
            action, log_prob = agent.get_action(torch.from_numpy(state_flat).float().to(device))
            action = torch.tensor(action).cpu().detach().numpy()[0]
            
            # 执行动作
            next_state_flat, r, done, info, next_all_state = env.step(action)
            
            # 计算GAIL奖励
            reward = discriminator.get_reward(
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
            discriminator_score += reward
            
            # 训练
            if agent.data.data_idx > agent_args.learn_start_size:
                if agent.data.data_idx >= agent_args.memory_size:
                    agent.train(discriminator, discriminator_args.batch_size, None, n_epi, agent_args.batch_size)
        
        score_lst.append(score)
        discriminator_score_lst.append(discriminator_score)
        
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
                save_name = f'./model_weights_gail_base/gail_base_{n_epi}'
                discriminator.save(save_name)
                agent.brain.save(save_name)
        
        if writer:
            writer.add_scalar("score/score", score, n_epi)
            writer.add_scalar("score/discriminator", discriminator_score, n_epi)
        
        print(f"Episode {n_epi}: score={score:.1f}, disc_score={discriminator_score:.1f}")
        score_lst = []
        discriminator_score_lst = []


if __name__ == "__main__":
    main()
