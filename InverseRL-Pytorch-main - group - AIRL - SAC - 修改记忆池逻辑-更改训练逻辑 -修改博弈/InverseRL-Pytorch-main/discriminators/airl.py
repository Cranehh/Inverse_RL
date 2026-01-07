from discriminators.base import Discriminator
from networks.discriminator_network import G,H
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

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
cfg.num_activities=2
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
def trajectory_intersection(position_A, velocity_A, position_B, velocity_B):
    # Unpack the input tuples
    x_A0, y_A0 = position_A
    v_Ax, v_Ay = velocity_A
    x_B0, y_B0 = position_B
    v_Bx, v_By = velocity_B

    # Calculate the coefficients for the equations
    # (v_Ax - v_Bx) * t_A = x_B0 - x_A0 + (v_Bx - v_Ax) * t_B
    # (v_Ay - v_By) * t_A = y_B0 - y_A0 + (v_By - v_Ay) * t_B

    # Define variables for matrix calculation
    delta_x = x_B0 - x_A0
    delta_y = y_B0 - y_A0
    det = v_Ax * v_By - v_Ay * v_Bx  # Determinant of the 2x2 system

    if det == 0:
        return None  # No intersection if determinant is zero (parallel trajectories)

    # Calculate the intersection time parameters t_A and t_B
    t_A = (delta_x * v_By - delta_y * v_Bx) / det
    t_B = (delta_x * v_Ay - delta_y * v_Ax) / det

    # Intersection point occurs when both times are positive (if needed for the future trajectory)
    if t_A >= 0 and t_B >= 0:
        # Compute the intersection point
        x_intersection = x_A0 + v_Ax * t_A
        y_intersection = y_A0 + v_Ay * t_A

        if t_A <= t_B:
            md = ((x_B0 + v_Bx * t_A - x_intersection) ** 2 + (y_B0 + v_By * t_A - y_intersection) ** 2) ** 0.5
        else:
            md = ((x_A0 + v_Ax * t_B - x_intersection) ** 2 + (y_A0 + v_Ay * t_B - y_intersection) ** 2) ** 0.5

        walking_distance_A = ((x_A0 - x_intersection) ** 2 + (y_A0 - y_intersection) ** 2) ** 0.5
        walking_distance_B = ((x_B0 - x_intersection) ** 2 + (y_B0 - y_intersection) ** 2) ** 0.5

        return (x_intersection, y_intersection), torch.abs(t_A - t_B), md, walking_distance_A, walking_distance_B

    return None

import math
import math

## v1 和 A都是要发生避让的对象
def rotate_vector_towards_target(v1, v2, max_angle_step_deg=20):
    # Calculate the angle of the vectors
    angle_v1 = math.atan2(v1[1], v1[0])
    angle_v2 = math.atan2(v2[1], v2[0])

    # Calculate the initial angle difference
    angle_diff = angle_v2 - angle_v1

    # Normalize the angle difference to the range [-pi, pi]
    while angle_diff > math.pi:
        angle_diff -= 2 * math.pi
    while angle_diff < -math.pi:
        angle_diff += 2 * math.pi

    # Calculate the rotation step (in radians)
    angle_step = math.radians(max_angle_step_deg)

    # Rotate in the shortest direction towards v2
    if angle_diff > 0:
        angle_v1 += min(angle_step, angle_diff)
    else:
        angle_v1 -= min(angle_step, -angle_diff)

    # Compute the new rotated vector
    return angle_v1

def adjust_velocity_based_on_angle(position_A, position_B, velocity_A, angle_offset_deg, speed_scale):
    # Calculate the relative position vector from A to B
    delta_x = position_A[0] - position_B[0]
    delta_y = position_A[1] - position_B[1]


    # Calculate the new angle by adding the offset
    angle_new = rotate_vector_towards_target((velocity_A[0], velocity_A[1]), (delta_x, delta_y), angle_offset_deg)

    # Calculate the new speed by scaling the original speed
    speed_A_magnitude = torch.sqrt(velocity_A[0]**2 + velocity_A[1]**2)
    new_speed_A = speed_A_magnitude * speed_scale

    # Calculate the new velocity components in the adjusted direction
    new_vx_A = new_speed_A * torch.cos(torch.tensor(angle_new))
    new_vy_A = new_speed_A * torch.sin((torch.tensor(angle_new)))

    return new_vx_A, new_vy_A
class AIRL(Discriminator):
    def __init__(self, writer, device, state_dim, action_dim, args):
        super(AIRL, self).__init__()
        self.writer = writer
        self.device = device
        self.args = args
        
        self.g_p = GCNnet_collective(cfg)
        # self.h_p = H(self.args.layer_num, state_dim, action_dim, self.args.hidden_dim, self.args.activation_function, self.args.last_activation)

        self.g_v = GCNnet_collective(cfg)
        self.h = H(self.args.layer_num, state_dim, action_dim, self.args.hidden_dim, self.args.activation_function,
                   self.args.last_activation)
        self.fc_fusion = nn.Linear(4, 1)
        self.parameter_value = nn.ParameterDict(
            {'game': nn.Parameter(torch.tensor([-0.852760, -1.520647, 0.447699, 0.011182, 0.086358, 0.062003,
                                                -9.118979, -8.760420, 4.538517, 0.668325, -8.695439, -0.695439,
                                                1.401653, 0.089132]), requires_grad=True)})
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)

    def evolutionary_game_utility(self, state_flat, data_pedestrain, data_vehicle):
        theta_c_p = self.parameter_value['game'][0]
        theta_c_v = self.parameter_value['game'][1]
        theta_md_p = self.parameter_value['game'][2]
        theta_md_v = self.parameter_value['game'][3]
        theta_tmd_p = self.parameter_value['game'][4]
        theta_tmd_v = self.parameter_value['game'][5]
        theta_cs_p = self.parameter_value['game'][6]
        theta_cs_v = self.parameter_value['game'][7]
        theta_dtour_p = self.parameter_value['game'][8]
        theta_dtour_v = self.parameter_value['game'][9]
        theta_speed_p = self.parameter_value['game'][10]
        theta_speed_v = self.parameter_value['game'][11]
        theta_group = self.parameter_value['game'][12]
        tw = 0.2
        u = self.parameter_value['game'][13]
        Epp_u_total = []
        Epw_u_total = []
        Evp_total = []
        Evw_total = []
        for i in range(state_flat.shape[0]):
            x = 0.5
            y = 0.5
            pos_pedestrian = state_flat[i, 96:98]
            pos_vehicle = state_flat[i, 110:112]
            vel_pedestrian = state_flat[i, 3:5]
            vel_vehicle = state_flat[i, 59:61]

            ##P-P
            ## 碰撞间接损失
            ppresult = trajectory_intersection(pos_pedestrian, vel_pedestrian, pos_vehicle, vel_vehicle)
            if ppresult is not None:
                MD_PP = ppresult[2]
                TMD_PP = ppresult[1]
                initial_distance_p = ppresult[3]
                initial_distance_v = ppresult[4]
            else:
                MD_PP = 100
                TMD_PP = 100
                initial_distance_p = 0
                initial_distance_v = 0

            Epp_p = (0 +
                     # theta_c_p * np.sqrt((now_state[5]) ** 2 + (now_state[6]) ** 2) +
                     theta_c_p * torch.sqrt(
                        (vel_pedestrian[0] - vel_vehicle[0]) ** 2 + (vel_pedestrian[1] - vel_vehicle[1]) ** 2) +
                     theta_md_p * MD_PP +
                     theta_tmd_p * TMD_PP +
                     theta_cs_p * torch.sqrt(vel_vehicle[0] ** 2 + vel_vehicle[1] ** 2) +
                     theta_dtour_p * 0 +
                     theta_speed_p * 0 +
                     theta_group * torch.tensor(np.int32(data_pedestrain[2][i])).to(self.device))
            Epp_v = (0 +
                     # theta_c_v * np.sqrt((now_state[7]) ** 2 + (now_state[8]) ** 2) +
                     theta_c_v * torch.sqrt(
                        (vel_pedestrian[0] - vel_vehicle[0]) ** 2 + (vel_pedestrian[1] - vel_vehicle[1]) ** 2) +
                     theta_md_v * MD_PP +
                     theta_tmd_v * TMD_PP +
                     theta_cs_v * torch.sqrt(vel_vehicle[0] ** 2 + vel_vehicle[1] ** 2) +
                     theta_dtour_v * 0 +
                     theta_speed_v * 0).reshape((1, -1))[0]

            ##P-Y

            new_velocity_V = adjust_velocity_based_on_angle(pos_vehicle, pos_pedestrian, vel_vehicle, 20, 0.8)
            pyresult = trajectory_intersection(pos_pedestrian, vel_pedestrian, pos_vehicle, new_velocity_V)
            if pyresult is not None:
                MD_PY = pyresult[2]
                TMD_PY = pyresult[1]
                PY_distance_p = pyresult[3]
                PY_distance_v = pyresult[4]
                PY_detour_p = - PY_distance_p + initial_distance_p if PY_distance_p > initial_distance_p else 0
                PY_detour_v = - PY_distance_v + initial_distance_v if PY_distance_v > initial_distance_v else 0
            else:
                MD_PY = 100
                TMD_PY = 100

                PY_distance_p = 0
                PY_distance_v = 0
                PY_detour_p = - PY_distance_p + initial_distance_p if PY_distance_p > initial_distance_p else 0
                PY_detour_v = - PY_distance_v + initial_distance_v if PY_distance_v > initial_distance_v else 0

            Epy_p = (0 +
                     # theta_c_p * np.sqrt((now_state[5]) ** 2 + (now_state[6]) ** 2) +
                     theta_c_p * torch.sqrt(
                        (vel_pedestrian[0] - new_velocity_V[0]) ** 2 + (vel_pedestrian[1] - new_velocity_V[1]) ** 2) +
                     theta_md_p * MD_PY +
                     theta_tmd_p * TMD_PY +
                     theta_cs_p * torch.sqrt(new_velocity_V[0] ** 2 + new_velocity_V[1] ** 2) +
                     theta_dtour_p * (math.exp(PY_detour_p) - 1) +
                     theta_speed_p * 0 +
                     theta_group * torch.tensor(np.int32(data_pedestrain[2][i])).to(self.device))

            Epy_v = (0 +
                     # theta_c_v * np.sqrt((new_velocity_V[0]) ** 2 + (new_velocity_V[1]) ** 2) +
                     theta_c_v * torch.sqrt(
                        (vel_pedestrian[0] - new_velocity_V[0]) ** 2 + (vel_pedestrian[1] - new_velocity_V[1]) ** 2) +
                     theta_md_v * MD_PY +
                     theta_tmd_v * TMD_PY +
                     theta_cs_v * torch.sqrt(new_velocity_V[0] ** 2 + new_velocity_V[1] ** 2) +
                     theta_dtour_v * (math.exp(PY_detour_v) - 1) +
                     theta_speed_v * (math.exp(0.2) - 1)).reshape((1, -1))[0]

            ##Y-P
            new_velocity_P = adjust_velocity_based_on_angle(pos_pedestrian, pos_vehicle, vel_pedestrian, 20, 0.8)
            ypresult = trajectory_intersection(pos_pedestrian, new_velocity_P, pos_vehicle, vel_vehicle)
            if ypresult is not None:
                MD_YP = ypresult[2]
                TMD_YP = ypresult[1]
                YP_distance_p = ypresult[3]
                YP_distance_v = ypresult[4]
                YP_detour_p = - YP_distance_p + initial_distance_p if YP_distance_p > initial_distance_p else 0
                YP_detour_v = - YP_distance_v + initial_distance_v if YP_distance_v > initial_distance_v else 0
            else:
                MD_YP = 100
                TMD_YP = 100
                YP_distance_p = 0
                YP_distance_v = 0
                YP_detour_p = - YP_distance_p + initial_distance_p if YP_distance_p > initial_distance_p else 0
                YP_detour_v = - YP_distance_v + initial_distance_v if YP_distance_v > initial_distance_v else 0

            Eyp_p = (0 +
                     # theta_c_p * np.sqrt((new_velocity_P[0]) ** 2 + (new_velocity_P[1]) ** 2) +
                     theta_c_p * torch.sqrt(
                        (new_velocity_P[0] - vel_vehicle[0]) ** 2 + (new_velocity_P[1] - vel_vehicle[1]) ** 2) +
                     theta_md_p * MD_YP +
                     theta_tmd_p * TMD_YP +
                     theta_cs_p * torch.sqrt(vel_vehicle[0] ** 2 + vel_vehicle[1] ** 2) +
                     theta_dtour_p * (math.exp(YP_detour_p) - 1) +
                     theta_speed_p * (math.exp(0.2) - 1) +
                     theta_group * torch.tensor(np.int32(data_pedestrain[2][i])).to(self.device))
            Eyp_v = (0 +
                     # theta_c_v * np.sqrt((now_state[7]) ** 2 + (now_state[8]) ** 2) +
                     theta_c_v * torch.sqrt(
                        (new_velocity_P[0] - vel_vehicle[0]) ** 2 + (new_velocity_P[1] - vel_vehicle[1]) ** 2) +
                     theta_md_v * MD_YP +
                     theta_tmd_v * TMD_YP +
                     theta_cs_v * torch.sqrt(vel_vehicle[0] ** 2 + vel_vehicle[1] ** 2) +
                     theta_dtour_v * (math.exp(YP_detour_v) - 1) +
                     theta_speed_v * 0).reshape((1, -1))[0]

            ##Y-Y
            yyresult = trajectory_intersection(pos_pedestrian, new_velocity_P, pos_vehicle, new_velocity_V)
            if yyresult is not None:
                MD_YY = yyresult[2]
                TMD_YY = yyresult[1]
                YY_distance_p = yyresult[3]
                YY_distance_v = yyresult[4]
                YY_detour_p = - YY_distance_p + initial_distance_p if YY_distance_p > initial_distance_p else 0
                YY_detour_v = - YY_distance_v + initial_distance_v if YY_distance_v > initial_distance_v else 0
            else:
                MD_YY = 0
                TMD_YY = 0
                YY_distance_p = 0
                YY_distance_v = 0
                YY_detour_p = - YY_distance_p + initial_distance_p if YY_distance_p > initial_distance_p else 0
                YY_detour_v = - YY_distance_v + initial_distance_v if YY_distance_v > initial_distance_v else 0

            Eyy_p = (0 +
                     # theta_c_p * np.sqrt((new_velocity_P[0]) ** 2 + (new_velocity_P[1]) ** 2) +
                     theta_c_p * torch.sqrt(
                        (new_velocity_P[0] - new_velocity_V[0]) ** 2 + (new_velocity_P[1] - new_velocity_V[1]) ** 2) +
                     theta_md_p * MD_YY +
                     theta_tmd_p * TMD_YY +
                     theta_cs_p * torch.sqrt(new_velocity_V[0] ** 2 + new_velocity_V[1] ** 2) +
                     theta_dtour_p * (math.exp(YY_detour_p) - 1) +
                     theta_speed_p * (math.exp(0.2) - 1) +
                     theta_group * torch.tensor(np.int32(data_pedestrain[2][i])).to(self.device))

            Eyy_v = (0 +
                     # theta_c_v * np.sqrt((new_velocity_V[0]) ** 2 + (new_velocity_V[1]) ** 2) +
                     theta_c_v * torch.sqrt(
                        (new_velocity_P[0] - new_velocity_V[0]) ** 2 + (new_velocity_P[1] - new_velocity_V[1]) ** 2) +
                     theta_md_v * MD_YY +
                     theta_tmd_v * TMD_YY +
                     theta_cs_v * torch.sqrt(new_velocity_V[0] ** 2 + new_velocity_V[1] ** 2) +
                     theta_dtour_v * (math.exp(YY_detour_v) - 1) +
                     theta_speed_v * (math.exp(0.2) - 1)).reshape((1, -1))[0]

            Epp = y * Epp_p + (1 - y) * Epy_p
            Epw = y * Eyp_p + (1 - y) * Eyy_p

            Evp = x * Epp_v + (1 - x) * Eyp_v
            Evw = x * Epy_v + (1 - x) * Eyy_v

            Epp_u = (1 - u) * Epp + u * torch.tensor(np.int32(data_pedestrain[2][i])).to(self.device) * \
                    (Epp - Epw).reshape((1, -1))[0]
            Epw_u = (1 - u) * Epw + u * torch.tensor(np.int32(data_vehicle[2][i])).to(self.device) * \
                    (Epp - Epw).reshape((1, -1))[0]

            Epp_u_total.append(Epp_u)
            Epw_u_total.append(Epw_u)
            Evp_total.append(Evp)
            Evw_total.append(Evw)
        # K1 = self.parameter_value['theta_kp'].values[0] * torch.sqrt((state_flat[:, 3] + state_flat[:, 5]) ** 2 + (state_flat[:, 4] + state_flat[:, 6]) ** 2)
        # K2 = self.parameter_value['theta_kv'].values[0] * torch.sqrt((state_flat[:, 59] + state_flat[:, 61]) ** 2 + (state_flat[:, 60] + state_flat[:, 62]) ** 2)
        # C1 = self.parameter_value['theta_cp'].values[0] * torch.sqrt((state_flat[:, 59] - state_flat[:, 3]) ** 2 + (state_flat[:, 60] - state_flat[:, 4]) ** 2)
        # C2 = self.parameter_value['theta_cv'].values[0] * torch.sqrt((state_flat[:, 59] - state_flat[:, 3]) ** 2 + (state_flat[:, 60] - state_flat[:, 4]) ** 2)
        # # D1 = theta_dpx * ((now_state[0]-now_state[2])/(now_state[4]-now_state[6])) + theta_dpy * ((now_state[1]-now_state[3])/(now_state[5]-now_state[7]))
        # # D2 = theta_dvx * ((now_state[0]-now_state[2])/(now_state[4]-now_state[6])) + theta_dvy * ((now_state[1]-now_state[3])/(now_state[5]-now_state[7]))
        # D1 = self.parameter_value['theta_dpx'].values[0] * torch.sqrt(state_flat[:, 3] ** 2 + state_flat[:, 4] ** 2)
        # D2 = self.parameter_value['theta_dvx'].values[0] * torch.sqrt(state_flat[:, 59] ** 2 + state_flat[:, 60] ** 2)
        # W1 = self.parameter_value['theta_wp'].values[0] * tw
        # W2 = self.parameter_value['theta_wv'].values[0] * (tw + torch.sqrt(state_flat[:, 59] ** 2 + state_flat[:, 60] ** 2))
        # x = 0.5
        # y = 0.5
        # u = self.parameter_value['u'].values[0]
        # Epp = K1 - y * C1
        # Epw = K1 - y * D1 + y * W1 - W1
        #
        # Evp = K2 - x * C2
        # Evw = K2 - x * D2 + x * W2 - W2
        #
        # Epp_u = (1 - u) * Epp + u * torch.tensor(np.int32(data_pedestrain[2])).to(self.device) * (Epp - Epw)
        # Epw_u = (1 - u) * Epw + u * torch.tensor(np.int32(data_vehicle[2])).to(self.device) * (Epp - Epw)

        Epp_u_total = torch.cat(Epp_u_total, dim=0)
        Epw_u_total = torch.cat(Epw_u_total, dim=0)
        Evp_total = torch.cat(Evp_total, dim=0)
        Evw_total = torch.cat(Evw_total, dim=0)

        utility_mat = torch.cat([Epp_u_total, Epw_u_total, Evp_total, Evw_total], dim=0)
        return utility_mat.reshape(state_flat.shape[0], 4).to(self.device)
    def get_f(self,all_state, state_flat,action,next_state_flat,done_mask):
        ## 输出轨迹为真的概率
        data_pedestrain = (all_state[:, 0], all_state[:, 2],
                           all_state[:, 4])
        data_vehicle = (all_state[:, 1], all_state[:, 3],
                        all_state[:, 5])

        ##上一版本：g输出的结果只有1个，有两个h函数

        ##将gp，gv分开建模（最后分别得到两个输出），结合博弈层后输出得到一个值，h用一个就行
        like_reward_p = self.g_p.forward(data_pedestrain, type='pedestrian')
        like_reward_v = self.g_v.forward(data_vehicle, type='vehicle')
        like_reward = torch.cat((like_reward_p, like_reward_v), dim=1)
        # 隐去博弈
        game_utility = self.evolutionary_game_utility(state_flat,data_pedestrain,data_vehicle) / 10
        like_reward = like_reward * game_utility
        like_reward = self.fc_fusion(like_reward)
        like_reward = like_reward + done_mask.reshape(done_mask.shape[0],1).float() * (self.args.gamma * self.h(next_state_flat)  - self.h(state_flat))

        # like_reward_p = like_reward_p + done_mask.reshape(done_mask.shape[0],1).float() * (self.args.gamma * self.h_p(next_state_flat)  - self.h_p(state_flat))
        # like_reward_v = self.g_v.forward(data_vehicle) + done_mask.reshape(done_mask.shape[0],1).float() * (self.args.gamma * self.h_v(next_state_flat)  - self.h_v(state_flat))


        return torch.clamp(like_reward, min=-50, max=50)
    def get_d(self,log_prob, all_state, state_flat,action,next_state_flat,done_mask):
        exp_f = torch.exp(self.get_f(all_state, state_flat,action,next_state_flat,done_mask)).cpu()
        ## 确定性策略，所以为1
        d = exp_f / (exp_f + torch.exp(log_prob).cpu())
        return d
    def get_reward(self,log_prob, all_state, state_flat,action,next_state_flat,done):
        done_mask = torch.ones(done.shape).to(self.device)
        # done_mask = torch.ones(done.shape)(1 - done.float()).to(self.device)
        #return (self.get_f(state,action,next_state,done_mask) - log_prob).detach()
        d = (self.get_d(log_prob, all_state, state_flat,action,next_state_flat,done_mask)).detach()
        reward = (torch.log(d + 1e-3) - torch.log((1-d)+1e-3))
        return reward
        

    def forward(self,log_prob, all_state, state_flat,action,next_state_flat,done_mask):
        d = (self.get_d(log_prob, all_state, state_flat,action,next_state_flat,done_mask))
        return d

    def judge_training(self,actor_log_prob, expert_log_prob, writer, n_epi, agent_all_state, agent_state_flat,agent_action,agent_next_state_flat,agent_done_mask,expert_all_state, expert_state_flat,expert_action,expert_next_state_flat,expert_done_mask):
        expert_preds = self.forward(expert_log_prob, expert_all_state, expert_state_flat, expert_action, expert_next_state_flat,
                                    expert_done_mask)
        # expert_loss = self.criterion(expert_preds.to(self.device), torch.ones(expert_preds.shape[0], 1).to(self.device))

        agent_preds = self.forward(actor_log_prob, agent_all_state, agent_state_flat, agent_action, agent_next_state_flat,
                                   agent_done_mask)
        # agent_loss = self.criterion(agent_preds.to(self.device), torch.zeros(agent_preds.shape[0], 1).to(self.device))

        # loss = expert_loss + agent_loss
        expert_acc = ((expert_preds > 0.5).float()).mean()
        learner_acc = ((agent_preds < 0.5).float()).mean()

        if self.writer != None:
            # self.writer.add_scalar("loss/discriminator_loss", loss.item(), n_epi)
            self.writer.add_scalar("loss/expert_acc", expert_acc, n_epi)
            self.writer.add_scalar("loss/learner_acc", learner_acc, n_epi)
        if (expert_acc > 0.8) and (learner_acc > 0.8):
            return False
        else:
            return True

    def train_network(self,actor_log_prob, expert_log_prob, writer, n_epi, agent_all_state, agent_state_flat,agent_action,agent_next_state_flat,agent_done_mask,expert_all_state, expert_state_flat,expert_action,expert_next_state_flat,expert_done_mask):

        ## 需要的有 all_state, state_flat,action,next_state_flat,done_mask
        expert_preds = self.forward(expert_log_prob, expert_all_state, expert_state_flat,expert_action,expert_next_state_flat,expert_done_mask)
        expert_loss = self.criterion(expert_preds.to(self.device),torch.ones(expert_preds.shape[0],1).to(self.device))

        agent_preds = self.forward(actor_log_prob, agent_all_state, agent_state_flat,agent_action,agent_next_state_flat,agent_done_mask)
        agent_loss = self.criterion(agent_preds.to(self.device),torch.zeros(agent_preds.shape[0],1).to(self.device))
        
        loss = expert_loss+agent_loss
        # expert_acc = ((expert_preds > 0.5).float()).mean()
        # learner_acc = ((agent_preds < 0.5).float()).mean()
        
        if self.writer != None:
            self.writer.add_scalar("loss/discriminator_loss", loss.item(), n_epi)
            self.writer.add_scalar("loss/expert_loss", expert_loss.item(), n_epi)
            self.writer.add_scalar("loss/agent_loss", agent_loss.item(), n_epi)
        #if (expert_acc > 0.8) and (learner_acc > 0.8):
        #    return 
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, filename):
        torch.save(self.g_p.state_dict(), filename + "_g_p")
        torch.save(self.g_v.state_dict(), filename + "_g_v")
        torch.save(self.h.state_dict(), filename + "_h")
        torch.save(self.parameter_value.state_dict(), filename + "_parameter_value")
        torch.save(self.optimizer.state_dict(), filename + "_optimizer")

    def load(self, filename):
        self.g_p.load_state_dict(torch.load(filename + "_g_p"))
        self.g_v.load_state_dict(torch.load(filename + "_g_v"))
        self.h.load_state_dict(torch.load(filename + "_h"))
        self.parameter_value.load_state_dict(torch.load(filename + "_parameter_value"))
        self.optimizer.load_state_dict(torch.load(filename + "_optimizer"))


