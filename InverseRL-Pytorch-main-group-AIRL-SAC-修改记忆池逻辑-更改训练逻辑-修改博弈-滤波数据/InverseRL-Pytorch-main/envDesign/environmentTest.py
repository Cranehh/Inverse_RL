import numpy as np
import math


class InteractionEnvForTest:
    def __init__(self, env_state_pedestrain, env_state_vehicle, env_position_pedestrain, env_position_vehicle,
                 env_done_group, env_num_pedestrian, env_num_vehicle):
        self.env_state_pedestrain = np.load(env_state_pedestrain, allow_pickle=True)
        self.env_state_vehicle = np.load(env_state_vehicle, allow_pickle=True)
        self.env_position_pedestrain = np.load(env_position_pedestrain, allow_pickle=True)
        self.env_position_vehicle = np.load(env_position_vehicle, allow_pickle=True)
        self.env_num_pedestrian = np.load(env_num_pedestrian, allow_pickle=True)
        self.env_num_vehicle = np.load(env_num_vehicle, allow_pickle=True)
        self.env_done_group = np.load(env_done_group, allow_pickle=True)

        self.real_tra = []
        self.predicted_tra = []

        self.real_speed = []
        self.predicted_speed = []

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
        ## 根据action得到的预测状态
        next_vx = self.state[3] + action[0] * 0.2
        next_vy = self.state[4] + action[1] * 0.2
        next_x = self.state[96] + self.state[3] * 0.2 + 0.5 * action[0] * 0.2 ** 2
        next_y = self.state[97] + self.state[4] * 0.2 + 0.5 * action[1] * 0.2 ** 2
        self.predicted_tra.append([next_x, next_y])
        self.predicted_speed.append([next_vx, next_vy])
        next_yaw_pedestrain = math.atan2(next_vy, next_vx)
        new_state = self.get_state()
        self.real_tra.append([new_state[96], new_state[97]])
        self.real_speed.append([new_state[3], new_state[4]])
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

    def calc_metric(self):
        real_tra = np.array(self.real_tra)
        predicted_tra = np.array(self.predicted_tra)
        real_speed = np.array(self.real_speed)
        predicted_speed = np.array(self.predicted_speed)

        tra_mae = np.mean(np.abs(real_tra - predicted_tra), axis=0)
        speed_mae = np.mean(np.abs(real_speed - predicted_speed), axis=0)
        tra_hd = np.zeros((real_tra.shape[0], 2))
        speed_hd = np.zeros((real_speed.shape[0], 2))

        for i in range(real_tra.shape[0]):
            tra_hd[i, 0] = np.min(np.abs(np.asarray(predicted_tra[i, 0] - real_tra[:, 0])))
            tra_hd[i, 1] = np.min(np.abs(np.asarray(predicted_tra[i, 1] - real_tra[:, 1])))

            speed_hd[i, 0] = np.min(np.abs(np.asarray(predicted_speed[i, 0] - real_speed[:, 0])))
            speed_hd[i, 1] = np.min(np.abs(np.asarray(predicted_speed[i, 1] - real_speed[:, 1])))

        tra_hd = np.max(tra_hd, axis=0)
        speed_hd = np.max(speed_hd, axis=0)

        return tra_mae, speed_mae, tra_hd, speed_hd

    def calc_metric_for_analysis(self):
        real_tra = np.array(self.real_tra)
        predicted_tra = np.array(self.predicted_tra)
        real_speed = np.array(self.real_speed)
        predicted_speed = np.array(self.predicted_speed)

        tra_mae = np.mean(np.abs(real_tra - predicted_tra), axis=0)
        speed_mae = np.mean(np.abs(real_speed - predicted_speed), axis=0)
        tra_hd = np.zeros((real_tra.shape[0], 2))
        speed_hd = np.zeros((real_speed.shape[0], 2))

        for i in range(real_tra.shape[0]):
            tra_hd[i, 0] = np.min(np.abs(np.asarray(predicted_tra[i, 0] - real_tra[:, 0])))
            tra_hd[i, 1] = np.min(np.abs(np.asarray(predicted_tra[i, 1] - real_tra[:, 1])))

            speed_hd[i, 0] = np.min(np.abs(np.asarray(predicted_speed[i, 0] - real_speed[:, 0])))
            speed_hd[i, 1] = np.min(np.abs(np.asarray(predicted_speed[i, 1] - real_speed[:, 1])))

        tra_hd = np.max(tra_hd, axis=0)
        speed_hd = np.max(speed_hd, axis=0)

        return tra_mae, speed_mae, tra_hd, speed_hd, real_tra, predicted_tra, real_speed, predicted_speed

    def step(self, action):
        self.current_step += 1
        self.state, self.all_state = self.update_state(action)
        self.info = (self.current_scene, self.current_step)
        self.done = self.env_done_group[self.current_scene][self.current_step]
        if self.done:
            self.current_scene += 1
            self.current_step = 0
            tra_mae, speed_mae, tra_hd, speed_hd = self.calc_metric()
            self.real_tra = []
            self.predicted_tra = []
            self.real_speed = []
            self.predicted_speed = []
            if self.current_scene >= len(self.env_state_pedestrain):
                self.current_scene = 0
            return tra_mae, speed_mae, tra_hd, speed_hd, self.done
        else:

            return 1, 1, 1, 1, self.done

    def step_for_analysis(self, action):
        self.current_step += 1
        self.state, self.all_state = self.update_state(action)
        self.info = (self.current_scene, self.current_step)
        self.done = self.env_done_group[self.current_scene][self.current_step]
        if self.done:
            self.current_scene += 1
            self.current_step = 0
            tra_mae, speed_mae, tra_hd, speed_hd, real_tra, predicted_tra, real_speed, predicted_speed = self.calc_metric_for_analysis()
            self.real_tra = []
            self.predicted_tra = []
            self.real_speed = []
            self.predicted_speed = []
            if self.current_scene >= len(self.env_state_pedestrain):
                self.current_scene = 0
            return tra_mae, speed_mae, tra_hd, speed_hd, self.done, real_tra, predicted_tra, real_speed, predicted_speed
        else:

            return 1, 1, 1, 1, self.done, 1, 1, 1, 1
