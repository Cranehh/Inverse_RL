from agents.algorithm.ppo    import PPO
from agents.algorithm.sac    import SAC
from agents.algorithm.ddpg    import DDPG
from agents.algorithm.td3    import TD3
from agents.agent            import Agent

from discriminators.gail     import GAIL
from discriminators.vail     import VAIL
from discriminators.airl     import AIRL
from discriminators.vairl    import VAIRL
from discriminators.eairl    import EAIRL
from discriminators.sqil    import SQIL
from utils.utils             import RunningMeanStd, Dict, make_transition

from configparser            import ConfigParser
from argparse                import ArgumentParser

import os
from envDesign.environment import InteractionEnv
from envDesign.environmentTest import InteractionEnvForTest
import numpy as np

import torch

os.makedirs('./model_weights_without_game', exist_ok=True)

env = InteractionEnv('/mnt/f/公开数据集/Yandex/人车交互数据/环境数据/env优化/env_state_pedestrain_train.npy',
                        '/mnt/f/公开数据集/Yandex/人车交互数据/环境数据/env优化/env_state_vehicle_train.npy',
                        '/mnt/f/公开数据集/Yandex/人车交互数据/环境数据/env优化/env_position_pedestrain_train.npy',
                        '/mnt/f/公开数据集/Yandex/人车交互数据/环境数据/env优化/env_position_vehicle_train.npy',
                        '/mnt/f/公开数据集/Yandex/人车交互数据/环境数据/env优化/env_done_group_train.npy',
                        '/mnt/f/公开数据集/Yandex/人车交互数据/环境数据/env优化/env_num_pedestrain_train.npy',
                        '/mnt/f/公开数据集/Yandex/人车交互数据/环境数据/env优化/env_num_vehicle_train.npy')
print('训练环境读取完成')
envTest = InteractionEnvForTest('/mnt/f/公开数据集/Yandex/人车交互数据/环境数据/env优化/env_state_pedestrain_train.npy',
                        '/mnt/f/公开数据集/Yandex/人车交互数据/环境数据/env优化/env_state_vehicle_train.npy',
                        '/mnt/f/公开数据集/Yandex/人车交互数据/环境数据/env优化/env_position_pedestrain_train.npy',
                        '/mnt/f/公开数据集/Yandex/人车交互数据/环境数据/env优化/env_position_vehicle_train.npy',
                        '/mnt/f/公开数据集/Yandex/人车交互数据/环境数据/env优化/env_done_group_train.npy',
                        '/mnt/f/公开数据集/Yandex/人车交互数据/环境数据/env优化/env_num_pedestrain_train.npy',
                        '/mnt/f/公开数据集/Yandex/人车交互数据/环境数据/env优化/env_num_vehicle_train.npy')

print('测试环境读取完成')
# env = InteractionEnv(f'F:/公开数据集/Yandex/人车交互数据/环境数据/env/env_state.npy',
#                       f'F:/公开数据集/Yandex/人车交互数据/环境数据/env/env_done.npy')

action_dim = env.action_dim
state_dim = env.state_dim
parser = ArgumentParser('parameters')


parser.add_argument('--test', type=bool, default=False, help="True if test, False if train (default: False)")
parser.add_argument('--render', type=bool, default=False, help="(default: False)")
parser.add_argument('--epochs', type=int, default=100001, help='number of epochs, (default: 1001)')
parser.add_argument("--agent", type=str, default = 'sac', help = 'actor training algorithm(default: ppo)')
parser.add_argument("--discriminator", type=str, default = 'airl', help = 'discriminator training algorithm(default: gail)')
parser.add_argument("--save_interval", type=int, default = 100, help = 'save interval')
parser.add_argument("--print_interval", type=int, default = 1, help = 'print interval')
parser.add_argument('--tensorboard', type=bool, default=True, help='use_tensorboard, (default: True)')

args = parser.parse_args()
parser = ConfigParser()
parser.read('config.ini')

torch.manual_seed(0)
np.random.seed(0)

demonstrations_location_args = Dict(parser,'demonstrations_location',True)
agent_args = Dict(parser,args.agent)
discriminator_args = Dict(parser,args.discriminator)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if args.tensorboard:
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter('runs')
else:
    writer = None

if args.discriminator == 'airl':
    discriminator = AIRL(writer, device, state_dim, action_dim, discriminator_args)

elif args.discriminator == 'vairl':
    discriminator = VAIRL(writer, device, state_dim, action_dim, discriminator_args)
elif args.discriminator == 'gail':
    discriminator = GAIL(writer, device, state_dim, action_dim, discriminator_args)
elif args.discriminator == 'vail':
    discriminator = VAIL(writer,device,state_dim, action_dim, discriminator_args)
elif args.discriminator == 'eairl':
    discriminator = EAIRL(writer, device, state_dim, action_dim, discriminator_args)
elif args.discriminator == 'sqil':
    discriminator = SQIL(writer, device, state_dim, action_dim, discriminator_args)
else:
    raise NotImplementedError

print('逆强化学习构建完成')
max_action = 2
if args.agent == 'ppo':
    algorithm = PPO(device, state_dim, action_dim, agent_args)
elif args.agent == 'sac':
    algorithm = SAC(device, state_dim, action_dim, agent_args)
elif args.agent == 'ddpg':
    algorithm = DDPG(state_dim, action_dim, max_action)
elif args.agent == 'td3':
    algorithm = TD3(state_dim, action_dim, max_action)
else:
    raise NotImplementedError
print('强化学习构建完成')
agent = Agent(algorithm, writer, device, state_dim, action_dim, agent_args, demonstrations_location_args)
print('智能体构建完成')
if device == 'cuda':
    agent = agent.cuda()
    discriminator = discriminator.cuda()
    
state_rms = RunningMeanStd(state_dim)

score_lst = []
discriminator_score_lst = []
score = 0.0
discriminator_score = 0
if agent_args.on_policy == True:
    state_lst = []
    state_ = (env.reset())
    state = np.clip((state_ - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5)
    for n_epi in range(args.epochs):
        for t in range(agent_args.traj_length):
            # if args.render:
            #     env.render()
            state_lst.append(state_)
            
            action, log_prob = agent.get_action(torch.from_numpy(state).float().unsqueeze(0).to(device))
            
            next_state_, r, done, info = env.step(action.cpu().numpy())
            next_state = np.clip((next_state_ - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5)
            if discriminator_args.is_airl:
                reward = discriminator.get_reward(\
                                        log_prob,\
                                        torch.tensor(state).unsqueeze(0).float().to(device),action,\
                                        torch.tensor(next_state).unsqueeze(0).float().to(device),\
                                                              torch.tensor(done).view(1,1)\
                                                 ).item()
            else:
                reward = discriminator.get_reward(torch.tensor(state).unsqueeze(0).float().to(device),action).item()

            transition = make_transition(state,\
                                         action,\
                                         np.array([reward/10.0]),\
                                         next_state,\
                                         np.array([done]),\
                                         log_prob.detach().cpu().numpy()\
                                        )
            agent.put_data(transition) 
            score += r
            discriminator_score += reward
            if done:
                state_ = (env.reset())
                state = np.clip((state_ - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5)
                score_lst.append(score)
                discriminator_score_lst.append(discriminator_score)
                if writer != None:
                    writer.add_scalar("score/real", score, n_epi)
                    writer.add_scalar("score/discriminator", discriminator_score, n_epi)
                score = 0
                discriminator_score = 0
            else:
                state = next_state
                state_ = next_state_
        agent.train(discriminator, discriminator_args.batch_size, state_rms, n_epi)
        state_rms.update(np.vstack(state_lst))
        state_lst = []
        if n_epi%args.print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, sum(score_lst)/len(score_lst)))
            score_lst = []
        if (n_epi % args.save_interval == 0 )& (n_epi != 0):
            torch.save(agent.state_dict(), './model_weights/model_'+str(n_epi))
else : #off-policy
    for n_epi in range(args.epochs):
        score = 0.0
        discriminator_score = 0.0
        ##这里得到的状态需要是很多数组，可以返回一个压平的和一个没有压平的,没有压平的用来获得奖励值，压平的用来获得动作值
        state_flat, all_state = env.reset()
        done = False
        while not done:
            # if args.render:
            #     env.render()
            action, log_prob = agent.get_action(torch.from_numpy(state_flat).float().to(device))
            # action = agent.get_action(torch.from_numpy(state_flat).float().to(device))

            action = torch.tensor(action).cpu().detach().numpy()[0]
            log_prob = log_prob.to(device)
            ##状态更新
            next_state_flat, r, done, info, next_all_state = env.step(action)
            if discriminator_args.is_airl:
                reward = discriminator.get_reward( \
                            log_prob,
                            all_state,
                            torch.tensor(state_flat).unsqueeze(0).float().to(device),action,\
                            torch.tensor(next_state_flat).unsqueeze(0).float().to(device),\
                                                  torch.tensor(done).unsqueeze(0),\
                                                using_game=False\
                                                 ).item()
            else:
                reward = discriminator.get_reward(all_state).item()

            transition = make_transition(state_flat,\
                                         action,\
                                         np.array([reward]),\
                                         next_state_flat,\
                                         np.array([done]), \
                                         all_state[0], \
                                         next_all_state[0]
                                        )
            agent.put_data(transition) 

            state_flat = next_state_flat
            all_state = next_all_state

            score += r
            discriminator_score += reward
            
            if agent.data.data_idx > agent_args.learn_start_size:
                agent.train_only_rl(discriminator, discriminator_args.batch_size, state_rms, n_epi, agent_args.batch_size)

                if agent.data.data_idx >= agent_args.memory_size:

                    agent.train(discriminator, discriminator_args.batch_size, state_rms, n_epi, agent_args.batch_size)
        score_lst.append(score)

        # if (agent.data.data_idx > agent_args.learn_start_size)&(n_epi % 200 == 0):
        #     agent.train(discriminator, discriminator_args.batch_size, state_rms, n_epi, agent_args.batch_size)
        #     score_lst.append(score)


        ## 评价指标计算
        if  (n_epi % 10 == 0):
            ls_tra_mae = []
            ls_speed_mae = []
            ls_tra_hd = []
            ls_speed_hd= []
            for i in range(envTest.env_state_pedestrain.shape[0]):
                state_flat_test, all_state_test = envTest.reset()
                doneTest = False
                while not doneTest:
                    action, log_prob = agent.get_action(torch.from_numpy(envTest.state).float().to(device))
                    # action = agent.get_action(torch.from_numpy(state_flat).float().to(device))

                    action = torch.tensor(action).cpu().detach().numpy()[0]

                    tra_mae, speed_mae, tra_hd, speed_hd, doneTest = envTest.step(action)
                ls_tra_mae.append(tra_mae)
                ls_speed_mae.append(speed_mae)
                ls_tra_hd.append(tra_hd)
                ls_speed_hd.append(speed_hd)
            tra_mae = np.array(ls_tra_mae)
            speed_mae = np.array(ls_speed_mae)
            tra_hd = np.array(ls_tra_hd)
            speed_hd = np.array(ls_speed_hd)
            tra_mae = np.mean(tra_mae,axis = 0)
            speed_mae = np.mean(speed_mae,axis = 0)
            tra_hd = np.mean(tra_hd,axis = 0)
            speed_hd = np.mean(speed_hd,axis = 0)
            if args.tensorboard:
                writer.add_scalar("Metric/P_X_MAE", tra_mae[0], n_epi)
                writer.add_scalar("Metric/P_Y_MAE", tra_mae[1], n_epi)
                writer.add_scalar("Metric/P_X_HD", tra_hd[0], n_epi)
                writer.add_scalar("Metric/P_Y_HD", tra_hd[1], n_epi)
                writer.add_scalar("Metric/V_X_MAE", speed_mae[0], n_epi)
                writer.add_scalar("Metric/V_Y_MAE", speed_mae[1], n_epi)
                writer.add_scalar("Metric/V_X_HD", speed_hd[0], n_epi)
                writer.add_scalar("Metric/V_Y_HD", speed_hd[1], n_epi)
            print(f'P_X_MAE:{tra_mae[0]}; P_Y_MAE:{tra_mae[1]}')
            print(f'V_X_MAE:{speed_mae[0]}; V_Y_MAE:{speed_mae[1]}')
            print(f'P_X_HD:{tra_hd[0]}; P_Y_HD:{tra_hd[1]}')
            print(f'V_X_HD:{speed_hd[0]}; V_Y_HD:{speed_hd[1]}')
            ## 模型保存
            # if (tra_mae[0]<1)&(tra_mae[1]<1)&(speed_mae[0]<1)&(speed_mae[1]<1)&(tra_hd[0]<0.5)&(tra_hd[1]<0.5)&(speed_hd[0]<0.5)&(speed_hd[1]<0.5):
            if (tra_mae[0]<1.5)&(tra_mae[1]<1.5) | (n_epi % 1000 == 0):

                discriminator.save(f'./model_weights/{[tra_mae[0],tra_mae[1],speed_mae[0],speed_mae[1],tra_hd[0],tra_hd[1],speed_hd[0],speed_hd[1],n_epi]}')
                agent.brain.save(f'./model_weights/{[tra_mae[0],tra_mae[1],speed_mae[0],speed_mae[1],tra_hd[0],tra_hd[1],speed_hd[0],speed_hd[1],n_epi]}')
            # if (n_epi % 150 == 0):
            #     discriminator.save(f'./model_weights/150')
            #     agent.brain.save(f'./model_weights/150')
        discriminator_score_lst.append(discriminator_score)
        if args.tensorboard:
            writer.add_scalar("score/score", score/score, n_epi)
            writer.add_scalar("score/discriminator", discriminator_score/score, n_epi)

        print("# of episode :{}, avg score : {:.1f},average_discriminator_score:{:.1f}".format(n_epi, sum(score_lst),sum(discriminator_score_lst)/sum(score_lst)))
        score_lst = []
        discriminator_score_lst = []
        # if n_epi%args.save_interval==0 and n_epi!=0:
        #     torch.save(agent.state_dict(),'./model_weights/agent_'+str(n_epi))