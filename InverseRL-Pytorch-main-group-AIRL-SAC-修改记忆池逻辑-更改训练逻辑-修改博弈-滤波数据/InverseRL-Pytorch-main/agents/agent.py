import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

from utils.utils import ReplayBuffer, make_one_mini_batch, convert_to_tensor, make_one_mini_batch_airl

class Agent(nn.Module):
    def __init__(self,algorithm, writer, device, state_dim, action_dim, args, demonstrations_location_args, use_in_next_step=False): 
        super(Agent, self).__init__()
        self.writer = writer
        self.device = device
        self.args = args
        if self.args.on_policy == True :
            self.data = ReplayBuffer(action_prob_exist = True, max_size = self.args.traj_length, state_dim = state_dim, num_action = action_dim)
        else :
            self.data = ReplayBuffer(action_prob_exist = False, max_size = int(self.args.memory_size), state_dim = state_dim, num_action = action_dim)
        file_size = 120

        ##读取专家信息，
        ## 现在状态
        # self.experts_state_pedestrain = np.load(
        #     demonstrations_location_args.expert_state_pedestrain_location, allow_pickle=True)
        # self.experts_state_vehicle = np.load(
        #     demonstrations_location_args.expert_state_vehicle_location, allow_pickle=True)
        # self.experts_position_pedestrain = np.load(
        #     demonstrations_location_args.expert_position_pedestrain_location, allow_pickle=True)
        # self.experts_position_vehicle = np.load(
        #     demonstrations_location_args.expert_position_vehicle_location, allow_pickle=True)
        # self.experts_vehicle_num = np.load(demonstrations_location_args.expert_num_vehicle_location,
        #                               allow_pickle=True)
        # self.experts_pedestrain_num = np.load(
        #     demonstrations_location_args.expert_num_pedestrain_location, allow_pickle=True)
        if not use_in_next_step:
            self.expert_actions = np.load(
                demonstrations_location_args.expert_action_location, allow_pickle=True)
            self.expert_done = np.load(
                demonstrations_location_args.expert_done_location, allow_pickle=True)

        # self.experts_next_state_pedestrain = np.load(
        #     demonstrations_location_args.expert_next_state_pedestrain_location, allow_pickle=True)
        # self.experts_next_state_vehicle = np.load(
        #     demonstrations_location_args.expert_next_state_vehicle_location, allow_pickle=True)
        # self.experts_next_position_pedestrain = np.load(
        #     demonstrations_location_args.expert_next_position_pedestrain_location, allow_pickle=True)
        # self.experts_next_position_vehicle = np.load(
        #     demonstrations_location_args.expert_next_position_vehicle_location, allow_pickle=True)
        # self.experts_next_vehicle_num = np.load(demonstrations_location_args.expert_next_num_vehicle_location,
        #                                    allow_pickle=True)
        # self.experts_next_pedestrain_num = np.load(
        #     demonstrations_location_args.expert_next_num_pedestrain_location, allow_pickle=True)

        ##拼接all_state
        # ls = []
        # ls_next = []
        # for i in range(self.experts_state_pedestrain.shape[0]):
        #     ls_single_all_state = []
        #     ls_single_all_state.append(self.experts_state_pedestrain[i])
        #     ls_single_all_state.append(self.experts_state_vehicle[i])
        #     ls_single_all_state.append(self.experts_position_pedestrain[i])
        #     ls_single_all_state.append(self.experts_position_vehicle[i])
        #     ls_single_all_state.append(self.experts_pedestrain_num[i])
        #     ls_single_all_state.append(self.experts_vehicle_num[i])
        #     ls.append(ls_single_all_state)
        #
        #     ls_next_single_all_state = []
        #     ls_next_single_all_state.append(self.experts_next_state_pedestrain[i])
        #     ls_next_single_all_state.append(self.experts_next_state_vehicle[i])
        #     ls_next_single_all_state.append(self.experts_next_position_pedestrain[i])
        #     ls_next_single_all_state.append(self.experts_next_position_vehicle[i])
        #     ls_next_single_all_state.append(self.experts_next_pedestrain_num[i])
        #     ls_next_single_all_state.append(self.experts_next_vehicle_num[i])
        #     ls_next.append(ls_next_single_all_state)
        if not use_in_next_step:
            self.expert_all_state = np.load(
                demonstrations_location_args.expert_all_state_location, allow_pickle=True)
            self.expert_next_all_state = np.load(
                demonstrations_location_args.expert_next_all_state_location, allow_pickle=True)
        ##拼接state_flat
        # self.experts_state_pedestrain = self.experts_state_pedestrain.reshape(self.experts_state_pedestrain.shape[0], -1)
        # self.experts_state_vehicle = self.experts_state_vehicle.reshape(self.experts_state_vehicle.shape[0], -1)
        # self.experts_position_pedestrain = self.experts_position_pedestrain.reshape(self.experts_position_pedestrain.shape[0], -1)
        # self.experts_position_vehicle = self.experts_position_vehicle.reshape(self.experts_position_vehicle.shape[0], -1)
        #
        # self.expert_state_flat = np.concatenate([self.experts_state_pedestrain,
        #                                           self.experts_state_vehicle,
        #                                           self.experts_position_pedestrain,
        #                                           self.experts_position_vehicle], axis = 1)

        # self.experts_next_state_pedestrain = self.experts_next_state_pedestrain.reshape(self.experts_next_state_pedestrain.shape[0],
        #                                                                       -1)
        # self.experts_next_state_vehicle = self.experts_next_state_vehicle.reshape(self.experts_next_state_vehicle.shape[0], -1)
        # self.experts_next_position_pedestrain = self.experts_next_position_pedestrain.reshape(
        #     self.experts_next_position_pedestrain.shape[0], -1)
        # self.experts_next_position_vehicle = self.experts_next_position_vehicle.reshape(self.experts_next_position_vehicle.shape[0],
        #                                                                       -1)
        #
        # self.expert_next_state_flat = np.concatenate([self.experts_next_state_pedestrain,
        #                                                       self.experts_next_state_vehicle,
        #                                                       self.experts_next_position_pedestrain,
        #                                                       self.experts_next_position_vehicle], axis=1)

        if not use_in_next_step:
            self.expert_state_flat = np.load(
                demonstrations_location_args.expert_state_flat, allow_pickle=True)

            self.expert_next_state_flat = np.load(
                demonstrations_location_args.expert_next_state_flat, allow_pickle=True)



        self.brain = algorithm
        
    def get_action(self,x):
        action, log_prob = self.brain.get_action(x)
        return action, log_prob
    
    def put_data(self,transition):
        self.data.put_data(transition)

    def train_only_rl(self, discriminator, discriminator_batch_size, state_rms, n_epi, batch_size = 64):
        data = self.data.sample(shuffle=True, batch_size=batch_size)
        states, actions, rewards, next_states, done_masks, all_state, next_all_state = convert_to_tensor(self.device,
                                                                                                         data['state'],
                                                                                                         data['action'],
                                                                                                         data['reward'],
                                                                                                         data[
                                                                                                             'next_state'],
                                                                                                         data['done'],
                                                                                                         data[
                                                                                                             'all_state'],
                                                                                                         data[
                                                                                                             'next_all_state'])
        done_masks = torch.ones(done_masks.size()).to(self.device)
        self.brain.train_network(self.writer, n_epi, states, actions, rewards, next_states, done_masks)
    def train(self, discriminator, discriminator_batch_size, state_rms, n_epi, batch_size = 64):
        if self.args.on_policy :
            data = self.data.sample(shuffle = False)
            states, actions, rewards, next_states, done_masks, old_log_probs = convert_to_tensor(self.device, data['state'], data['action'], data['reward'], data['next_state'], data['done'], data['log_prob'])
        else :
            data = self.data.sample(shuffle = True, batch_size = discriminator_batch_size)
            states, actions, rewards, next_states, done_masks, all_state, next_all_state = convert_to_tensor(self.device, data['state'], data['action'], data['reward'], data['next_state'], data['done'],data['all_state'], data['next_all_state'])
        if discriminator.name() == 'sqil':
            agent_s,agent_a,agent_next_s,agent_done_mask = make_one_mini_batch(batch_size, states, actions, next_states, done_masks)
            expert_s,expert_a,expert_next_s,expert_done = make_one_mini_batch(batch_size, self.expert_states, self.expert_actions, self.expert_next_states, self.expert_dones) 
            expert_done_mask = (1 - expert_done.float())

            discriminator.train_network(self.brain, n_epi, agent_s, agent_a, agent_next_s, agent_done_mask, expert_s, expert_a, expert_next_s, expert_done_mask)
            return
        if discriminator.args.is_airl == False:
            agent_all_state = make_one_mini_batch(discriminator_batch_size, all_state)
            expert_all_state = make_one_mini_batch(discriminator_batch_size, self.expert_all_state)
            # if self.args.on_policy :
            #     expert_s = np.clip((expert_s - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5)
            discriminator.train_network(self.writer, n_epi, agent_all_state, expert_all_state)
        else:
            for i in range(20):
                agent_all_state, agent_state, agent_action, agent_next_state, agent_done_mask= make_one_mini_batch_airl(discriminator_batch_size, all_state, states, actions, next_states, done_masks)
                expert_all_state, expert_state_flat,expert_action,expert_next_state_flat,expert_done = make_one_mini_batch_airl(discriminator_batch_size, self.expert_all_state, self.expert_state_flat, self.expert_actions, self.expert_next_state_flat, self.expert_done)

                agent_done_mask = torch.ones(agent_done_mask.size()).to(self.device)
                expert_state_flat = torch.tensor(expert_state_flat).float().to(self.device)
                expert_action = torch.tensor(expert_action).float().to(self.device)
                expert_next_state_flat = torch.tensor(expert_next_state_flat).float().to(self.device)
                expert_done = torch.tensor(expert_done).float().to(self.device)

                expert_done_mask = torch.ones(expert_done.size()).to(self.device)
                # expert_done_mask = (1 - expert_done.float())
                # if self.args.on_policy :
                #     expert_s = np.clip((expert_s - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5).float()
                #     expert_next_s = np.clip((expert_next_s - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5).float()

                mu,sigma = self.brain.get_dist(agent_state.float().to(self.device))
                dist = torch.distributions.Normal(mu,sigma)
                agent_log_prob = dist.log_prob(agent_action).sum(-1,keepdim=True).detach()
                mu,sigma = self.brain.get_dist(expert_state_flat.float().to(self.device))
                dist = torch.distributions.Normal(mu,sigma)
                expert_log_prob = dist.log_prob(expert_action).sum(-1,keepdim=True).detach()
                train_flag = False
                if discriminator.judge_training(agent_log_prob, expert_log_prob,self.writer, n_epi, agent_all_state, agent_state,agent_action,agent_next_state,agent_done_mask,expert_all_state, expert_state_flat,expert_action,expert_next_state_flat,expert_done_mask):
                    train_flag = True
                    discriminator.train_network(agent_log_prob, expert_log_prob, self.writer, n_epi, agent_all_state, agent_state, agent_action, agent_next_state,agent_done_mask,expert_all_state, expert_state_flat,expert_action,expert_next_state_flat,expert_done_mask)
            if train_flag:
                self.data.clear()

                # data = self.data.sample(shuffle=True, batch_size=batch_size)
                # states, actions, rewards, next_states, done_masks, all_state, next_all_state = convert_to_tensor(
                #     self.device, data['state'], data['action'], data['reward'], data['next_state'], data['done'],
                #     data['all_state'], data['next_all_state'])
                # for i in range(10):
                #     self.brain.train_network(self.writer, n_epi, states, actions, rewards, next_states, done_masks)
        if self.args.on_policy :
            self.brain.train_network(self.writer, n_epi, states, actions, rewards, next_states, done_masks, old_log_probs)
        else :
            if train_flag:
                return
            else:
                data = self.data.sample(shuffle = True, batch_size = batch_size)
                states, actions, rewards, next_states, done_masks, all_state, next_all_state = convert_to_tensor(self.device, data['state'], data['action'], data['reward'], data['next_state'], data['done'],data['all_state'], data['next_all_state'])
                done_masks = torch.ones(done_masks.size()).to(self.device)
                self.brain.train_network(self.writer, n_epi, states, actions, rewards, next_states, done_masks)