import numpy as np
import math


class InteractionEnv:
    def __init__(self, env_state_pedestrain, env_state_vehicle, env_position_pedestrain, env_position_vehicle,
                 env_done_group, env_num_pedestrian, env_num_vehicle):
        self.env_state_pedestrain = np.load(env_state_pedestrain, allow_pickle=True)
        self.env_state_vehicle = np.load(env_state_vehicle, allow_pickle=True)
        self.env_position_pedestrain = np.load(env_position_pedestrain, allow_pickle=True)
        self.env_position_vehicle = np.load(env_position_vehicle, allow_pickle=True)
        self.env_num_pedestrian = np.load(env_num_pedestrian, allow_pickle=True)
        self.env_num_vehicle = np.load(env_num_vehicle, allow_pickle=True)
        self.env_done_group = np.load(env_done_group, allow_pickle=True)

        self.current_scene = 0
        self.current_step = 0
        self.state_dim = 120
        self.action_dim = 2
        self.state = []
        self.done = False
        self.r = 1
        self.info = ()

    def get_state(self):
        ls = []
        ls.append(self.env_state_pedestrain[self.current_scene][self.current_step].copy().reshape(-1))
        ls.append(self.env_state_vehicle[self.current_scene][self.current_step].copy().reshape(-1))
        ls.append(self.env_position_pedestrain[self.current_scene][self.current_step].copy().reshape(-1))
        ls.append(self.env_position_vehicle[self.current_scene][self.current_step].copy().reshape(-1))
        return np.concatenate(ls)

    def update_state(self, action):
        next_vx = self.state[3] + action[0] * 0.2
        next_vy = self.state[4] + action[1] * 0.2
        next_x = self.state[96] + self.state[3] * 0.2 + 0.5 * action[0] * 0.2 ** 2
        next_y = self.state[97] + self.state[4] * 0.2 + 0.5 * action[1] * 0.2 ** 2
        next_yaw_pedestrain = math.atan2(next_vy, next_vx)
        new_state = self.get_state()
        new_state[3] = next_vx
        new_state[4] = next_vy
        new_state[5] = action[0]
        new_state[6] = action[1]
        new_state[96] = next_x
        new_state[97] = next_y
        new_state[7] = next_yaw_pedestrain

        all_state = (self.env_state_pedestrain[self.current_scene][self.current_step].copy(),
                     self.env_state_vehicle[self.current_scene][self.current_step].copy(),
                     self.env_position_pedestrain[self.current_scene][self.current_step].copy(),
                     self.env_position_vehicle[self.current_scene][self.current_step].copy(),
                     self.env_num_pedestrian[self.current_scene][self.current_step].copy(),
                     self.env_num_vehicle[self.current_scene][self.current_step].copy())
        all_state[0][0][3] = next_vx
        all_state[0][0][4] = next_vy
        all_state[0][0][5] = action[0]
        all_state[0][0][6] = action[1]
        all_state[0][0][7] = next_yaw_pedestrain
        all_state[1][0][3] = next_vx
        all_state[1][0][4] = next_vy
        all_state[1][0][5] = action[0]
        all_state[1][0][6] = action[1]
        all_state[1][0][7] = next_yaw_pedestrain
        all_state[2][0][0] = next_x
        all_state[2][0][1] = next_y
        all_state[3][0][0] = next_x
        all_state[3][0][1] = next_y

        return new_state, np.array(all_state).reshape(1, -1)

    def reset(self):
        self.current_step = 0
        self.state = self.get_state()
        self.all_state = (self.env_state_pedestrain[self.current_scene][self.current_step].copy(),
                          self.env_state_vehicle[self.current_scene][self.current_step].copy(),
                          self.env_position_pedestrain[self.current_scene][self.current_step].copy(),
                          self.env_position_vehicle[self.current_scene][self.current_step].copy(),
                          self.env_num_pedestrian[self.current_scene][self.current_step].copy(),
                          self.env_num_vehicle[self.current_scene][self.current_step].copy())
        self.done = self.env_done_group[self.current_scene][self.current_step].copy()
        return self.state, np.array(self.all_state).reshape(1, -1)

    def step(self, action):



        self.current_step += 1
        self.state, self.all_state = self.update_state(action)
        self.info = (self.current_scene, self.current_step)
        self.done = self.env_done_group[self.current_scene][self.current_step]
        if self.done:
            self.current_scene += 1
            self.current_step = 0
            if self.current_scene >= len(self.env_state_pedestrain):
                self.current_scene = 0
        return self.state, self.r, self.done, self.info, self.all_state