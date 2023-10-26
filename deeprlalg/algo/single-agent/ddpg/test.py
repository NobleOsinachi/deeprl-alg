from ddpg import DDPG
from deeprlalg.user_config import DEFAULT_DATA_DIR
import os
import time


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str)
parser.add_argument('--exp_name', type=str, default='test')
parser.add_argument('--algo', type=str, default='ddpg')
parser.add_argument('--max_steps', type=int, default=1000)
parser.add_argument('--data_dir', type=str)
parser.add_argument('--epochs', '-n', type=int, default=200)
parser.add_argument('--steps_per_epoch', '-spe', type=int, default=4000)


parser.add_argument('--buffer_size', '-bs', type=int, default=1e6)

parser.add_argument('--polyak', '-pk', type=int, default=0.995)
parser.add_argument('--gamma', '-g', type=int, default=0.99)



#actor parameters
parser.add_argument('--pi_lr', '-plr', type=int, default=1e-3)
parser.add_argument('--pi_output_activation', '-p_oact', type=str, default='identity')
parser.add_argument('--pi_activation', '-p_act', type=str, default='tanh')
parser.add_argument('--pi_hidden_sizes', '-p_hs', nargs='+', default=[256, 256])

#q parameters
parser.add_argument('--q_lr', '-qlr', type=int, default=1e-3)
parser.add_argument('--q_output_activation', '-q_oact', type=str, default='identity')
parser.add_argument('--q_activation', '-q_act', type=str, default='tanh')
parser.add_argument('--q_hidden_sizes', '-q_hs', nargs='+', default=[256, 256])

#Training parameters
parser.add_argument('--batch_size', '-b', type=int, default=100) #steps collected per train iteration
parser.add_argument('--start_steps', '-ss', type=int, default=10000)
parser.add_argument('--update_after', '-ua', type=int, default=1000)
parser.add_argument('--update_freq', '-uf', type=int, default=50)
parser.add_argument('--action_noise', '-an', type=float, default=0.1)
parser.add_argument('--num_test_episodes', '-num_test', type=int, default=10)
parser.add_argument('--max_path_length', '-mpl', type=int, default=100)

parser.add_argument('--save_freq', '-sf', type=int, default=1000)
parser.add_argument('--render', '-r', type=bool, default=False)



parser.add_argument('--seed', type=int,  nargs='+', default=[1])

parser.add_argument('--no_gpu', '-ngpu', action='store_true')
parser.add_argument('--which_gpu', '-gpu_id', default=0)
parser.add_argument('--scalar_log_freq', type=int, default=1)

parser.add_argument('--save_params', action='store_true')
parser.add_argument('--action_noise_std', type=float, default=0)

args = parser.parse_args()
seeds = args.seed
params = vars(args)

max_steps = params['max_steps']
data_path = args.data_dir or DEFAULT_DATA_DIR
file_path = "todo_SimpleMultiSBEnv-v0_15-12-2022_17-29-35"
logdir = args.exp_name + '_' + args.env + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
agentdir = os.path.join(data_path, file_path)
logdir = os.path.join(data_path, logdir)
epoch = "epoch_140000"
params['logdir'] = logdir

for i in range(len(seeds)):
    args.seed = seeds[i]
    params['seed'] = seeds[i]
    seedlogdir = os.path.join(logdir, str(params['seed']))
    params['logdir'] = seedlogdir
    seedparams = params.copy()

    seedagentdir = os.path.join(agentdir, str(params['seed']))
    #setting params
    algo = DDPG(params)



    algo.agent.load(seedagentdir, epoch)

    env = algo.env
    agent = algo.agent
    print("Testing sample action...")

    for ep in range(10):
        state = env.reset()
        episode_reward = 0
        done = False
        steps = 0
        while not done:
            action = agent.get_action(state, 0)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward

            if steps % 10 == 0:
                env.render()
                # print("rendering")

            if done or steps == max_steps:
                # print(steps)
                break

            steps = steps + 1
            # print(steps)
    print("Episode Reward: ", episode_reward)
