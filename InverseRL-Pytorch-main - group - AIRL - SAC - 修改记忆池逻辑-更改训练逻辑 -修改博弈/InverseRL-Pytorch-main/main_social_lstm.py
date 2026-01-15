"""
Social LSTM 基线训练脚本
基于 Alahi et al. (2016) "Social LSTM: Human Trajectory Prediction in Crowded Spaces"

Social LSTM通过Social Pooling机制捕捉行人之间的社会交互
用于预测行人在与车辆交互时的轨迹

注意：Social LSTM是一个轨迹预测模型，不是IRL/IL方法
这里将其适配为模仿学习框架，用预测轨迹作为动作输出
"""

from agents.algorithm.sac import SAC
from agents.algorithm.td3 import TD3

from utils.utils import Dict, make_transition

from configparser import ConfigParser
from argparse import ArgumentParser

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

# ============== Social LSTM 核心模块 ==============

class SocialPooling(nn.Module):
    """
    Social Pooling Layer
    将邻近行人的隐藏状态池化到一个社会张量中
    """
    def __init__(self, hidden_dim, grid_size=4, neighborhood_size=2.0):
        super(SocialPooling, self).__init__()
        self.hidden_dim = hidden_dim
        self.grid_size = grid_size  # NxN网格
        self.neighborhood_size = neighborhood_size  # 邻域大小（米）
        
        # 用于嵌入池化后的社会张量
        self.embedding = nn.Linear(hidden_dim * grid_size * grid_size, hidden_dim)
        
    def get_grid_cell(self, pos, center_pos):
        """
        计算位置pos相对于center_pos在网格中的位置
        """
        # 相对位置
        rel_pos = pos - center_pos
        
        # 归一化到[-1, 1]
        rel_pos = rel_pos / self.neighborhood_size
        
        # 映射到网格索引 [0, grid_size-1]
        grid_pos = ((rel_pos + 1) / 2 * self.grid_size).long()
        grid_pos = torch.clamp(grid_pos, 0, self.grid_size - 1)
        
        return grid_pos
    
    def forward(self, hidden_states, positions, num_pedestrians):
        """
        Args:
            hidden_states: [batch_size, max_peds, hidden_dim] 所有行人的隐藏状态
            positions: [batch_size, max_peds, 2] 所有行人的位置
            num_pedestrians: [batch_size] 每个样本中实际的行人数量
        
        Returns:
            social_tensor: [batch_size, hidden_dim] 池化后的社会特征
        """
        batch_size = hidden_states.size(0)
        max_peds = hidden_states.size(1)
        device = hidden_states.device
        
        # 初始化社会张量
        social_tensors = torch.zeros(batch_size, self.grid_size, self.grid_size, self.hidden_dim).to(device)
        
        for b in range(batch_size):
            n_peds = int(num_pedestrians[b].item()) if num_pedestrians is not None else max_peds
            
            # 第一个行人是目标行人
            target_pos = positions[b, 0]
            
            for p in range(1, min(n_peds, max_peds)):
                # 计算其他行人相对于目标行人的网格位置
                other_pos = positions[b, p]
                
                # 检查是否在邻域内
                dist = torch.norm(other_pos - target_pos)
                if dist > self.neighborhood_size * 2:
                    continue
                
                grid_pos = self.get_grid_cell(other_pos, target_pos)
                
                # 累加隐藏状态到对应网格
                social_tensors[b, grid_pos[0], grid_pos[1]] += hidden_states[b, p]
        
        # 展平并嵌入
        social_tensors = social_tensors.view(batch_size, -1)
        social_embedding = self.embedding(social_tensors)
        
        return social_embedding


class SocialLSTMCell(nn.Module):
    """
    带有Social Pooling的LSTM单元
    """
    def __init__(self, input_dim, hidden_dim, grid_size=4, neighborhood_size=2.0):
        super(SocialLSTMCell, self).__init__()
        self.hidden_dim = hidden_dim
        
        # 位置嵌入
        self.input_embedding = nn.Linear(input_dim, hidden_dim)
        
        # Social Pooling
        self.social_pooling = SocialPooling(hidden_dim, grid_size, neighborhood_size)
        
        # LSTM输入包括：位置嵌入 + 社会特征
        self.lstm_cell = nn.LSTMCell(hidden_dim * 2, hidden_dim)
        
    def forward(self, input_pos, hidden, cell, all_hidden, all_positions, num_pedestrians):
        """
        Args:
            input_pos: [batch_size, input_dim] 当前位置/速度
            hidden: [batch_size, hidden_dim] 上一时刻隐藏状态
            cell: [batch_size, hidden_dim] 上一时刻细胞状态
            all_hidden: [batch_size, max_peds, hidden_dim] 所有行人的隐藏状态
            all_positions: [batch_size, max_peds, 2] 所有行人的位置
            num_pedestrians: [batch_size] 行人数量
        """
        # 位置嵌入
        pos_embed = F.relu(self.input_embedding(input_pos))
        
        # 社会池化
        social_embed = self.social_pooling(all_hidden, all_positions, num_pedestrians)
        
        # 拼接
        lstm_input = torch.cat([pos_embed, social_embed], dim=1)
        
        # LSTM更新
        hidden, cell = self.lstm_cell(lstm_input, (hidden, cell))
        
        return hidden, cell


class SocialLSTM(nn.Module):
    """
    完整的Social LSTM模型
    用于预测行人轨迹
    """
    def __init__(self, input_dim=2, hidden_dim=128, output_dim=2, 
                 grid_size=4, neighborhood_size=2.0, pred_length=12):
        super(SocialLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.pred_length = pred_length
        self.output_dim = output_dim
        
        # 输入嵌入（位置或速度）
        self.input_embedding = nn.Linear(input_dim, hidden_dim)
        
        # Social Pooling
        self.social_pooling = SocialPooling(hidden_dim, grid_size, neighborhood_size)
        
        # LSTM层
        self.lstm = nn.LSTMCell(hidden_dim * 2, hidden_dim)
        
        # 输出层（预测位置偏移）
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
    def init_hidden(self, batch_size, device):
        """初始化隐藏状态"""
        h = torch.zeros(batch_size, self.hidden_dim).to(device)
        c = torch.zeros(batch_size, self.hidden_dim).to(device)
        return h, c
    
    def forward(self, obs_traj, obs_traj_all=None, positions_all=None, num_pedestrians=None):
        """
        Args:
            obs_traj: [seq_len, batch_size, input_dim] 观测轨迹
            obs_traj_all: [seq_len, batch_size, max_peds, input_dim] 所有行人的观测轨迹
            positions_all: [seq_len, batch_size, max_peds, 2] 所有行人的位置
            num_pedestrians: [batch_size] 每个样本的行人数量
        
        Returns:
            pred_traj: [pred_len, batch_size, output_dim] 预测轨迹
        """
        seq_len, batch_size, _ = obs_traj.shape
        device = obs_traj.device
        
        # 初始化隐藏状态
        h, c = self.init_hidden(batch_size, device)
        
        # 用于存储所有行人的隐藏状态（用于social pooling）
        max_peds = obs_traj_all.size(2) if obs_traj_all is not None else 1
        all_hidden = torch.zeros(batch_size, max_peds, self.hidden_dim).to(device)
        
        # 编码阶段：处理观测序列
        for t in range(seq_len):
            # 位置嵌入
            pos_embed = F.relu(self.input_embedding(obs_traj[t]))
            
            # 如果有其他行人信息，进行social pooling
            if obs_traj_all is not None and positions_all is not None:
                # 更新所有行人的隐藏状态（简化：只用目标行人的隐藏状态）
                all_hidden[:, 0] = h
                
                # Social pooling
                positions_t = positions_all[t] if positions_all is not None else None
                social_embed = self.social_pooling(all_hidden, positions_t, num_pedestrians)
            else:
                social_embed = torch.zeros(batch_size, self.hidden_dim).to(device)
            
            # 拼接输入
            lstm_input = torch.cat([pos_embed, social_embed], dim=1)
            
            # LSTM更新
            h, c = self.lstm(lstm_input, (h, c))
        
        # 解码阶段：预测未来轨迹
        pred_traj = []
        current_pos = obs_traj[-1]  # 最后一个观测位置
        
        for t in range(self.pred_length):
            # 位置嵌入
            pos_embed = F.relu(self.input_embedding(current_pos))
            
            # Social pooling（简化：使用固定的social特征）
            social_embed = torch.zeros(batch_size, self.hidden_dim).to(device)
            
            # LSTM输入
            lstm_input = torch.cat([pos_embed, social_embed], dim=1)
            
            # LSTM更新
            h, c = self.lstm(lstm_input, (h, c))
            
            # 预测位置偏移
            output = self.output_layer(h)
            pred_traj.append(output)
            
            # 更新当前位置（用于下一步预测）
            current_pos = output
        
        pred_traj = torch.stack(pred_traj, dim=0)
        return pred_traj


class SocialLSTMPredictor(nn.Module):
    """
    Social LSTM的简化版本，适配到当前环境
    直接预测下一步的动作（位置偏移/速度）
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128, 
                 grid_size=4, neighborhood_size=4.0, num_layers=1):
        super(SocialLSTMPredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 状态编码器
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Social Pooling参数
        self.grid_size = grid_size
        self.neighborhood_size = neighborhood_size
        
        # 简化的Social特征提取（从state中提取周围行人信息）
        # 假设state包含目标行人和周围行人的信息
        self.social_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # LSTM
        self.lstm = nn.LSTM(hidden_dim * 2, hidden_dim, num_layers, batch_first=True)
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # 隐藏状态
        self.hidden = None
        
    def init_hidden(self, batch_size, device):
        h = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return (h, c)
    
    def forward(self, state, hidden=None):
        """
        Args:
            state: [batch_size, state_dim] 当前状态
            hidden: LSTM隐藏状态
        
        Returns:
            action: [batch_size, action_dim] 预测的动作
            hidden: 更新后的隐藏状态
        """
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state).float()
        
        batch_size = state.size(0) if len(state.shape) > 1 else 1
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        
        device = state.device
        
        if hidden is None:
            hidden = self.init_hidden(batch_size, device)
        
        # 状态编码
        state_embed = self.state_encoder(state)
        
        # 社会特征编码
        social_embed = self.social_encoder(state)
        
        # 拼接
        lstm_input = torch.cat([state_embed, social_embed], dim=1)
        lstm_input = lstm_input.unsqueeze(1)  # [batch, 1, hidden*2]
        
        # LSTM
        lstm_out, hidden = self.lstm(lstm_input, hidden)
        lstm_out = lstm_out.squeeze(1)  # [batch, hidden]
        
        # 输出动作
        action = self.output_layer(lstm_out)
        
        return action, hidden
    
    def reset_hidden(self):
        self.hidden = None


# ============== Social LSTM Agent ==============

class SocialLSTMAgent(nn.Module):
    """
    使用Social LSTM进行轨迹预测的Agent
    """
    def __init__(self, writer, device, state_dim, action_dim, args, demonstrations_location_args):
        super(SocialLSTMAgent, self).__init__()
        self.writer = writer
        self.device = device
        self.args = args
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Social LSTM模型
        self.model = SocialLSTMPredictor(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=args.hidden_dim,
            grid_size=args.grid_size,
            neighborhood_size=args.neighborhood_size
        )
        
        # 加载专家数据
        self.expert_actions = np.load(
            demonstrations_location_args.expert_action_location, allow_pickle=True)
        self.expert_state_flat = np.load(
            demonstrations_location_args.expert_state_flat, allow_pickle=True)
        self.expert_next_state_flat = np.load(
            demonstrations_location_args.expert_next_state_flat, allow_pickle=True)
        
        # 优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
        
        # 损失函数
        self.criterion = nn.MSELoss()
        
        # 用于存储序列数据
        self.trajectory_buffer = []
        self.max_buffer_size = 10000
        
        # LSTM隐藏状态
        self.hidden = None
        
    def get_action(self, state):
        """
        获取动作
        """
        self.model.eval()
        with torch.no_grad():
            if not isinstance(state, torch.Tensor):
                state = torch.tensor(state).float()
            state = state.to(self.device)
            
            if len(state.shape) == 1:
                state = state.unsqueeze(0)
            
            action, self.hidden = self.model(state, self.hidden)
            
            # 添加少量噪声以增加探索
            noise = torch.randn_like(action) * 0.1
            action = action + noise
            
        return action.cpu().numpy(), None
    
    def reset(self):
        """重置隐藏状态"""
        self.hidden = None
        self.model.reset_hidden()
    
    def store_transition(self, state, action, next_state):
        """存储转移用于训练"""
        self.trajectory_buffer.append({
            'state': state,
            'action': action,
            'next_state': next_state
        })
        
        if len(self.trajectory_buffer) > self.max_buffer_size:
            self.trajectory_buffer.pop(0)
    
    def train_on_expert(self, n_epi, batch_size=64):
        """
        在专家数据上训练（行为克隆）
        """
        self.model.train()
        
        # 采样专家数据
        indices = np.random.choice(len(self.expert_actions), size=batch_size, replace=False)
        
        states = torch.tensor(self.expert_state_flat[indices]).float().to(self.device)
        expert_actions = torch.tensor(self.expert_actions[indices]).float().to(self.device)
        
        # 前向传播（不使用隐藏状态连续性，因为是随机采样）
        pred_actions, _ = self.model(states, None)
        
        # 计算损失
        loss = self.criterion(pred_actions, expert_actions)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
        self.optimizer.step()
        
        if self.writer is not None:
            self.writer.add_scalar("loss/bc_loss", loss.item(), n_epi)
        
        return loss.item()
    
    def train_on_sequences(self, n_epi, seq_length=10, batch_size=32):
        """
        在序列数据上训练，利用LSTM的时序特性
        """
        if len(self.trajectory_buffer) < seq_length * batch_size:
            return 0.0
        
        self.model.train()
        total_loss = 0.0
        
        # 构建序列
        for _ in range(batch_size):
            start_idx = np.random.randint(0, len(self.trajectory_buffer) - seq_length)
            
            states = []
            actions = []
            
            for i in range(seq_length):
                trans = self.trajectory_buffer[start_idx + i]
                states.append(trans['state'])
                actions.append(trans['action'])
            
            states = torch.tensor(np.array(states)).float().to(self.device)
            actions = torch.tensor(np.array(actions)).float().to(self.device)
            
            # 序列预测
            hidden = None
            pred_actions = []
            
            for t in range(seq_length):
                state_t = states[t:t+1]
                pred_action, hidden = self.model(state_t, hidden)
                pred_actions.append(pred_action)
            
            pred_actions = torch.cat(pred_actions, dim=0)
            
            # 计算损失
            loss = self.criterion(pred_actions, actions)
            total_loss += loss.item()
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()
        
        avg_loss = total_loss / batch_size
        
        if self.writer is not None:
            self.writer.add_scalar("loss/seq_loss", avg_loss, n_epi)
        
        return avg_loss
    
    def save(self, filename):
        torch.save(self.model.state_dict(), filename + "_social_lstm_model")
        torch.save(self.optimizer.state_dict(), filename + "_social_lstm_optimizer")
    
    def load(self, filename):
        self.model.load_state_dict(torch.load(filename + "_social_lstm_model"))
        self.optimizer.load_state_dict(torch.load(filename + "_social_lstm_optimizer"))


# ============== 环境类 ==============
from envDesign.environment import InteractionEnv
from envDesign.environmentTest import InteractionEnvForTest


# ============== 主函数 ==============
def main():
    os.makedirs('./model_weights_social_lstm', exist_ok=True)

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
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument('--tensorboard', type=bool, default=True)
    parser.add_argument('--pretrain_epochs', type=int, default=1000, 
                        help='专家数据预训练轮数')
    args = parser.parse_args()
    
    # 读取配置
    config_parser = ConfigParser()
    config_parser.read('config.ini')
    
    torch.manual_seed(0)
    np.random.seed(0)
    
    demonstrations_location_args = Dict(config_parser, 'demonstrations_location', True)
    
    # Social LSTM的参数
    class SocialLSTMArgs:
        lr = 0.001
        hidden_dim = 128
        grid_size = 4
        neighborhood_size = 4.0  # 邻域大小（米）
        batch_size = 64
        seq_length = 10
    
    lstm_args = SocialLSTMArgs()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # TensorBoard
    writer = None
    if args.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter('runs_social_lstm')
    
    # 创建Social LSTM Agent
    agent = SocialLSTMAgent(writer, device, state_dim, action_dim, lstm_args, demonstrations_location_args)
    print('Social LSTM Agent 构建完成')
    
    if device == 'cuda':
        agent = agent.cuda()
    
    # ========== 阶段1: 在专家数据上预训练 ==========
    print("=" * 50)
    print("阶段1: 专家数据预训练")
    print("=" * 50)
    
    for pretrain_epi in range(args.pretrain_epochs):
        loss = agent.train_on_expert(pretrain_epi, batch_size=lstm_args.batch_size)
        
        if pretrain_epi % 100 == 0:
            print(f"Pretrain Episode {pretrain_epi}: loss={loss:.6f}")
    
    print("预训练完成！")
    
    # ========== 阶段2: 在环境中微调 ==========
    print("=" * 50)
    print("阶段2: 环境交互微调")
    print("=" * 50)
    
    score_lst = []
    
    for n_epi in range(args.epochs):
        score = 0.0
        
        state_flat, all_state = env.reset()
        agent.reset()  # 重置LSTM隐藏状态
        done = False
        
        episode_transitions = []
        
        while not done:
            # 获取动作
            action, _ = agent.get_action(torch.from_numpy(state_flat).float().to(device))
            action = action[0] if len(action.shape) > 1 else action
            
            # 执行动作
            next_state_flat, r, done, info, next_all_state = env.step(action)
            
            # 存储转移
            agent.store_transition(state_flat, action, next_state_flat)
            episode_transitions.append({
                'state': state_flat,
                'action': action,
                'next_state': next_state_flat
            })
            
            state_flat = next_state_flat
            all_state = next_all_state
            score += r
        
        score_lst.append(score)
        
        # 训练（混合专家数据和交互数据）
        if n_epi % 5 == 0:
            # 在专家数据上训练
            agent.train_on_expert(n_epi, batch_size=lstm_args.batch_size)
            
            # 在序列数据上训练
            if len(agent.trajectory_buffer) > lstm_args.seq_length * lstm_args.batch_size:
                agent.train_on_sequences(n_epi, seq_length=lstm_args.seq_length, batch_size=32)
        
        # 评估
        if n_epi % 10 == 0:
            ls_tra_mae = []
            ls_speed_mae = []
            ls_tra_hd = []
            ls_speed_hd = []
            
            for i in range(envTest.env_state_pedestrain.shape[0]):
                state_flat_test, all_state_test = envTest.reset()
                agent.reset()
                doneTest = False
                
                while not doneTest:
                    action, _ = agent.get_action(torch.from_numpy(envTest.state).float().to(device))
                    action = action[0] if len(action.shape) > 1 else action
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
                save_name = f'./model_weights_social_lstm/social_lstm_{n_epi}'
                agent.save(save_name)
        
        if writer:
            writer.add_scalar("score/score", score, n_epi)
        
        if n_epi % 10 == 0:
            avg_score = np.mean(score_lst[-10:]) if len(score_lst) >= 10 else np.mean(score_lst)
            print(f"Episode {n_epi}: score={score:.1f}, avg_score={avg_score:.1f}")


if __name__ == "__main__":
    main()
