from discriminators.base import Discriminator
from networks.base import Network
import torch
import torch.nn as nn


class GAILBase(Discriminator):
    """
    基础版本的GAIL，不使用ARG和GCN模块
    直接使用MLP网络处理state-action对
    """
    def __init__(self, writer, device, state_dim, action_dim, args):
        super(GAILBase, self).__init__()
        self.writer = writer
        self.device = device
        self.args = args
        self.last_activation = torch.sigmoid
        
        # 基础MLP网络，输入为state_dim（压平后的状态）
        # 你的state_dim = 120
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
        """
        输入: state_flat [batch_size, state_dim] - 压平后的状态
        输出: 轨迹为真的概率
        """
        # 确保输入是float类型
        if not isinstance(state_flat, torch.Tensor):
            state_flat = torch.tensor(state_flat).float()
        state_flat = state_flat.float().to(self.device)
        
        return self.network(state_flat)
    
    def get_reward(self, state_flat):
        """
        计算奖励值
        GAIL的奖励: -log(D(s))
        """
        with torch.no_grad():
            d = self.forward(state_flat)
            reward = -torch.log(d + 1e-8)
        return reward.detach()
    
    def train_network(self, writer, n_epi, agent_state_flat, expert_state_flat):
        """
        训练判别器
        agent_state_flat: 智能体生成的状态 [batch_size, state_dim]
        expert_state_flat: 专家示范的状态 [batch_size, state_dim]
        """
        # 专家数据标签为0（真），智能体数据标签为1（假）
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
        
        # 计算准确率
        expert_acc = ((expert_preds < 0.5).float()).mean()
        learner_acc = ((agent_preds > 0.5).float()).mean()
        
        if self.writer is not None:
            self.writer.add_scalar("loss/discriminator_loss", loss.item(), n_epi)
            self.writer.add_scalar("loss/expert_acc", expert_acc.item(), n_epi)
            self.writer.add_scalar("loss/learner_acc", learner_acc.item(), n_epi)
        
        # 如果判别器已经很强，跳过训练
        if (expert_acc > 0.8) and (learner_acc > 0.8):
            return 
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, filename):
        torch.save(self.network.state_dict(), filename + "_network")
        torch.save(self.optimizer.state_dict(), filename + "_optimizer")

    def load(self, filename):
        self.network.load_state_dict(torch.load(filename + "_network"))
        self.optimizer.load_state_dict(torch.load(filename + "_optimizer"))
