from discriminators.base import Discriminator
from networks.base import Network
import torch
import torch.nn as nn
from discriminators.config import *
from discriminators.GcnModule import *
cfg = Config('collective')

cfg.device_list="0"
cfg.training_stage=2
cfg.stage1_model_path='result/STAGE1_MODEL.pth'  #PATH OF THE BASE MODEL
cfg.train_backbone=False

cfg.image_size=480, 720
cfg.out_size=57,87
cfg.num_boxes=6
cfg.num_actions=1
cfg.num_activities=1
cfg.num_frames=10
cfg.num_graph=16
cfg.tau_sqrt=True

cfg.batch_size=2
cfg.test_batch_size=8
cfg.train_learning_rate=1e-4
cfg.train_dropout_prob=0.2
cfg.weight_decay=1e-2
cfg.lr_plan={}
cfg.max_epoch=150

cfg.exp_note='Collective_stage2'
class GAIL(Discriminator):
    def __init__(self, writer, device, state_dim, action_dim, args):
        super(GAIL, self).__init__()
        self.writer = writer
        self.device = device
        self.args = args
        self.last_activation = torch.sigmoid
        self.network_pedestrain = GCNnet_collective(cfg)
        self.network_vehicle = GCNnet_collective(cfg)
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.lr)

    def forward(self, all_state):
        ## 输出轨迹为真的概率
        data_pedestrain = (all_state[:, 0], all_state[:, 2],
                           all_state[:, 4])
        data_vehicle = (all_state[:, 1], all_state[:, 3],
                           all_state[:, 5])

        reward_pedestrain = self.network_pedestrain.forward(data_pedestrain)
        reward_vehicle = self.network_vehicle.forward(data_vehicle)
        total_reward = reward_pedestrain + reward_vehicle
        total_reward = self.last_activation(total_reward)
        return total_reward
    def get_reward(self,all_state):
        data_pedestrain = (all_state[:, 0], all_state[:, 2], all_state[:, 4])
        data_vehicle = (all_state[:, 1], all_state[:, 3], all_state[:, 5])

        reward_pedestrain = self.network_pedestrain.forward(data_pedestrain)
        reward_vehicle = self.network_vehicle.forward(data_vehicle)
        total_reward = reward_pedestrain + reward_vehicle
        total_reward = self.last_activation(total_reward)
        return -torch.log(total_reward).detach()
    def train_network(self,writer,n_epi,agent_all_state, expert_all_state):

        expert_preds = self.forward(expert_all_state)
        
        expert_loss = self.criterion(expert_preds,torch.zeros(expert_preds.shape[0],1).to(self.device))

        agent_preds = self.forward(agent_all_state)
        agent_loss = self.criterion(agent_preds,torch.ones(agent_preds.shape[0],1).to(self.device))
        
        loss = expert_loss+agent_loss
        ## 0为真，1为假
        expert_acc = ((expert_preds < 0.5).float()).mean()
        learner_acc = ((agent_preds > 0.5).float()).mean()
        if self.writer != None:
            self.writer.add_scalar("loss/discriminator_loss", loss.item(), n_epi)
        if (expert_acc > 0.8) and (learner_acc > 0.8):
            return 
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()