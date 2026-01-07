from agents.algorithm.ppo    import PPO
from agents.algorithm.sac    import SAC
from agents.algorithm.ddpg    import DDPG
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

os.makedirs('./model_weights', exist_ok=True)
# envTest = InteractionEnvForTest('/mnt/f/公开数据集/Yandex/人车交互数据/环境数据/env/env_state_pedestrain_test.npy',
#                         '/mnt/f/公开数据集/Yandex/人车交互数据/环境数据/env/env_state_vehicle_test.npy',
#                         '/mnt/f/公开数据集/Yandex/人车交互数据/环境数据/env/env_position_pedestrain_test.npy',
#                         '/mnt/f/公开数据集/Yandex/人车交互数据/环境数据/env/env_position_vehicle_test.npy',
#                         '/mnt/f/公开数据集/Yandex/人车交互数据/环境数据/env/env_done_group_test.npy',
#                         '/mnt/f/公开数据集/Yandex/人车交互数据/环境数据/env/env_num_pedestrain_test.npy',
#                         '/mnt/f/公开数据集/Yandex/人车交互数据/环境数据/env/env_num_vehicle_test.npy')

# print('测试环境读取完成')

action_dim = 2
state_dim = 120
parser = ArgumentParser('parameters')


parser.add_argument('--test', type=bool, default=False, help="True if test, False if train (default: False)")
parser.add_argument('--render', type=bool, default=False, help="(default: False)")
parser.add_argument('--epochs', type=int, default=1001, help='number of epochs, (default: 1001)')
parser.add_argument("--agent", type=str, default = 'ddpg', help = 'actor training algorithm(default: ppo)')
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
    discriminator = VAIL(writer, device, state_dim, action_dim, discriminator_args)
elif args.discriminator == 'eairl':
    discriminator = EAIRL(writer, device, state_dim, action_dim, discriminator_args)
elif args.discriminator == 'sqil':
    discriminator = SQIL(writer, device, state_dim, action_dim, discriminator_args)
else:
    raise NotImplementedError

print('逆强化学习构建完成')
max_action = 5
if args.agent == 'ppo':
    algorithm = PPO(device, state_dim, action_dim, agent_args)
elif args.agent == 'sac':
    algorithm = SAC(device, state_dim, action_dim, agent_args)
elif args.agent == 'ddpg':
    algorithm = DDPG(state_dim, action_dim, max_action)
else:
    raise NotImplementedError
print('强化学习构建完成')

discriminator.load('./model_weights/[9.282341472883024e-10, 0.00028671899095671663, 5.363067381937108e-10, 8.957569568914229e-05, 2.0334047166701083e-09, 0.0005994389921373825, 6.620034053097664e-10, 9.463479937875769e-05]')
algorithm.load('./model_weights/[0.136517436247528, 0.15566204986598317, 0.07434197947542122, 0.08092887078245922, 0.11276829666387325, 0.1427225394673397, 0.04973939469583986, 0.05322640743970273]')
print('参数读取完成')

agent = Agent(algorithm, writer, device, state_dim, action_dim, agent_args, demonstrations_location_args)
print('智能体构建完成')
if device == 'cuda':
    agent = agent.cuda()
    discriminator = discriminator.cuda()

## 奖励值计算




ls_tra_mae = []
ls_speed_mae = []
ls_tra_hd = []
ls_speed_hd= []
for i in range(envTest.env_state_pedestrain.shape[0]):
    state_flat_test, all_state_test = envTest.reset()
    doneTest = False
    while not doneTest:
        action = agent.get_action(torch.from_numpy(envTest.state).float().to(device))
        action = torch.tensor(action).cpu().detach().numpy()
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

print(f'P_X_MAE:{tra_mae[0]}; P_Y_MAE:{tra_mae[1]}')
print(f'V_X_MAE:{speed_mae[0]}; V_Y_MAE:{speed_mae[1]}')
print(f'P_X_HD:{tra_hd[0]}; P_Y_HD:{tra_hd[1]}')
print(f'V_X_HD:{speed_hd[0]}; V_Y_HD:{speed_hd[1]}')