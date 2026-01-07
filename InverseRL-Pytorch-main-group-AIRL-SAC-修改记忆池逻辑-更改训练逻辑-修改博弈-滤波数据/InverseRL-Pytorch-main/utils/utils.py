import numpy as np
import torch

class Dict(dict):
    def __init__(self, config, section_name, location = False):
        super(Dict, self).__init__()
        self.initialize(config, section_name, location)
    def initialize(self, config, section_name, location):
        for key,value in config.items(section_name):
            if location :
                self[key] = value
            else:
                self[key] = eval(value)
    def __getattr__(self, val):
        return self[val]

def make_transition(state, action, reward, next_state, done, all_state,next_all_state, log_prob=None):
    transition = {}
    transition['state'] = state
    transition['action'] = action
    transition['reward'] = reward
    transition['next_state'] = next_state
    transition['all_state'] = all_state
    transition['next_all_state'] = next_all_state
    transition['log_prob'] = log_prob
    transition['done'] = done
    return transition

def make_mini_batch(*value):
    mini_batch_size = value[0]
    full_batch_size = len(value[1])
    full_indices = np.arange(full_batch_size)
    np.random.shuffle(full_indices)
    for i in range(full_batch_size // mini_batch_size):
        indices = full_indices[mini_batch_size * i : mini_batch_size * (i + 1)]
        yield [x[indices] for x in value[1:]]
        
def make_one_mini_batch(*value):
    mini_batch_size = value[0]
    full_batch_size = len(value[1])
    full_indices = np.arange(full_batch_size)
    np.random.shuffle(full_indices)
    indices = full_indices[ : mini_batch_size]
    return value[1][indices]

def make_one_mini_batch_airl(batch_size, all_state, state_flat,action,next_state_flat,done):
    mini_batch_size = batch_size
    full_batch_size = len(all_state)
    full_indices = np.arange(full_batch_size)
    np.random.shuffle(full_indices)
    indices = full_indices[ : mini_batch_size]
    return all_state[indices], state_flat[indices], action[indices], next_state_flat[indices], done[indices]
        
def convert_to_tensor(*value):
    device = value[0]
    tenser_result = [torch.tensor(x).float().to(device) for x in value[1:-2]]
    tenser_result.append(value[-2])
    tenser_result.append(value[-1])
    return tenser_result
    
class RunningMeanStd(object):
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)


## 融合新增数据和原有数据更新总数居的均值与方差
def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count

##存进回放池，done为True为0
class ReplayBuffer():
    def __init__(self, action_prob_exist, max_size, state_dim, num_action):
        self.max_size = max_size
        self.data_idx = 0
        self.state_dim = state_dim
        self.num_action = num_action
        self.action_prob_exist = action_prob_exist
        self.data = {}
        
        self.data['state'] = np.zeros((self.max_size, state_dim))
        self.data['action'] = np.zeros((self.max_size, num_action))
        self.data['reward'] = np.zeros((self.max_size, 1))
        self.data['next_state'] = np.zeros((self.max_size, state_dim))
        self.data['done'] = np.zeros((self.max_size, 1))
        self.data['all_state'] = []
        self.data['next_all_state'] = []
        if self.action_prob_exist :
            self.data['log_prob'] = np.zeros((self.max_size, 1))
    def put_data(self, transition):
        idx = self.data_idx % self.max_size
        self.data['state'][idx] = transition['state']
        self.data['action'][idx] = transition['action']
        self.data['reward'][idx] = transition['reward']
        self.data['next_state'][idx] = transition['next_state']
        self.data['all_state'].append(transition['all_state'])
        self.data['next_all_state'].append(transition['next_all_state'])
        done = transition['done']
        self.data['done'][idx] = 0.0 if done else 1.0
        if self.action_prob_exist :
            self.data['log_prob'][idx] = transition['log_prob']
        
        self.data_idx += 1
    def sample(self, shuffle, batch_size = None):
        if shuffle :
            sample_num = min(self.max_size, self.data_idx)
            rand_idx = np.random.choice(sample_num, batch_size, replace=False)
            sampled_data = {}
            sampled_data['state'] = self.data['state'][rand_idx]
            sampled_data['action'] = self.data['action'][rand_idx]
            sampled_data['reward'] = self.data['reward'][rand_idx]
            sampled_data['next_state'] = self.data['next_state'][rand_idx]
            sampled_data['all_state'] = np.array(self.data['all_state'])[rand_idx]
            sampled_data['next_all_state'] = np.array(self.data['next_all_state'])[rand_idx]
            sampled_data['done'] = self.data['done'][rand_idx]
            if self.action_prob_exist :
                sampled_data['log_prob'] = self.data['log_prob'][rand_idx]
            return sampled_data
        else:
            return self.data
    def size(self):
        return min(self.max_size, self.data_idx)

    def clear(self):
        self.data = {}

        self.data['state'] = np.zeros((self.max_size, self.state_dim))
        self.data['action'] = np.zeros((self.max_size, self.num_action))
        self.data['reward'] = np.zeros((self.max_size, 1))
        self.data['next_state'] = np.zeros((self.max_size, self.state_dim))
        self.data['done'] = np.zeros((self.max_size, 1))
        self.data['all_state'] = []
        self.data['next_all_state'] = []
        if self.action_prob_exist:
            self.data['log_prob'] = np.zeros((self.max_size, 1))
        self.data_idx = 0