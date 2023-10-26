from collections import OrderedDict
import pickle
import os
import sys
import time

import gym
import gym_target_defense
from gym import wrappers
import numpy as np
import torch


from deeprlalg.utils.logger import Logger

from deeprlalg.utils import utils
from deeprlalg.utils import pytorch_utils as ptu
from deeprlalg.utils.action_noise_wrapper import ActionNoiseWrapper
from deeprlalg.utils.utils import colorize
from ddpg_agent import DDPGAgent

from ddpg_policies import MLPPolicyDDPG


class DDPG(object):
    def __init__(self, params):

        # Create Logger and save params
        self.params = params
        self.logger = Logger(self.params['logdir'])
        self.logger.save_params(params)

        #Set Random seed and initialize gpu
        seed = self.params['seed']
        torch.manual_seed(seed)
        np.random.seed(seed)

        ptu.init_gpu(
            use_gpu=not self.params['no_gpu'],
            gpu_id=self.params['which_gpu']
        )

        # Create env and set env variables
        self.env = gym.make(self.params['env'])
        self.env.seed(seed)
        self.test_env = gym.make(self.params['env'])

        # Add noise wrapper
        if params['action_noise_std'] > 0:
            self.env = ActionNoiseWrapper(self.env, seed, params['action_noise_std'])

        # Is this env continuous, or self.discrete?
        discrete = isinstance(self.env.action_space, gym.spaces.Discrete)
        # Are the observations images?
        img = len(self.env.observation_space.shape) > 2


        # Observation and action sizes
        ob_dim = self.env.observation_space.shape if img else self.env.observation_space.shape[0]
        ac_dim = self.env.action_space.n if discrete else self.env.action_space.shape[0]


        # Action limit for clamping: critically, assumes all dimensions share the same bound!
        ac_limit = self.env.action_space.high[0]


        # DDPG Actor and Q PARAMS
        q_params = {
            'hidden_sizes' : params['q_hidden_sizes'],
            'activation' : params['q_activation'],
            'output_activation' : params['q_output_activation'],
            'learning_rate' : params['q_lr']
        }

        pi_params = {
            'hidden_sizes' : params['pi_hidden_sizes'],
            'activation' : params['pi_activation'],
            'output_activation' : params['pi_output_activation'],
            'learning_rate' : params['pi_lr'],
            'ac_limit' : ac_limit
        }

        self.params['agent_class'] = DDPGAgent
        self.agent_params = {}
        self.agent_params['pi_params'] = pi_params
        self.agent_params['polyak'] = self.params['polyak']
        self.agent_params['q_params'] = q_params
        self.agent_params['ac_dim'] = ac_dim
        self.agent_params['ob_dim'] = ob_dim
        self.agent_params['gamma'] = self.params['gamma']
        self.agent_params['buffer_size'] = self.params['buffer_size']


        agent_class = self.params['agent_class']
        self.agent = agent_class(self.env, self.agent_params)
        self.actor = self.agent.actor_critic

        ## Training PARAMS
        self.epochs = params['epochs']
        self.steps_per_epoch = params['steps_per_epoch']


    def run(self):

        # init vars at beginning of training
        self.total_envsteps = self.epochs * self.steps_per_epoch
        self.start_time = time.time()
        o, ep_ret, ep_len = self.env.reset(), 0, 0

        train_stats = dict(EpRet=[],
                        EpLen = [])

        print(colorize("==================================", 'green', bold=True))
        print(colorize("Running Experiment", 'green', bold=True))

        for t in range(self.total_envsteps):

            # decide if metrics should be logged
            if self.params['scalar_log_freq'] == -1:
                self.logmetrics = False
            elif t % self.params['scalar_log_freq'] == 0:
                self.logmetrics = True
            else:
                self.logmetrics = False

            low, high = self.env.action_space.low, self.env.action_space.high



            #start
            if t > self.params['start_steps']:
                a = self.agent.get_action(o, self.params['action_noise'])
                a = low + (0.5 * (a + 1.0) * (high - low))

            else:
                a = self.env.action_space.sample()
                # print("sa: ",a)



            s_a = 2.0 * ((a - low) / (high - low)) - 1.0

            next_o, r, d, _ = self.env.step(a)
            ep_ret += r
            ep_len += 1

            d = False if ep_len ==self.params['max_path_length'] else d

            # Store experience to replay buffer
            self.agent.replay_buffer.insert(o, s_a, r, next_o, d)
            o = next_o

            if d or (ep_len == self.params['max_path_length']):
                train_stats['EpRet'].append(ep_ret)
                train_stats['EpLen'].append(ep_len)
                o, ep_ret, ep_len = self.env.reset(), 0, 0



            #update networks
            if t >= self.params['update_after'] and t % self.params['update_freq'] == 0:
                all_logs  = {
                        'QLoss': [],
                        'PiLoss': [],
                        'Q_values': []
                    }
                for _ in range(self.params['update_freq']):
                    batch = self.agent.replay_buffer.sample(self.params['batch_size'])
                    train_log = self.agent.train(batch)
                    all_logs['QLoss'].append(train_log['QLoss'])
                    all_logs['PiLoss'].append(train_log['PiLoss'])
                    all_logs['Q_values'].append(np.mean(train_log['Q_values']))


            if (t+1) % self.steps_per_epoch == 0:
                epoch = (t+1)
                print(colorize("\n\n********** Epoch %i ************"%t, 'cyan', bold=True))


                if self.params['save_params'] and (epoch % self.params['save_freq'] == 0 or epoch == self.epochs):
                    # print("gl: ", self.params['logdir'])
                    self.agent.save(self.params['logdir'], epoch)

                # log/save
                if self.logmetrics:
                    # perform logging
                    self.perform_logging(self.test_env, epoch, self.actor, train_stats, all_logs)

        print(colorize("End of Experiment", 'green', bold=True))
        print(colorize("==================================", 'green', bold=True))




    def perform_logging(self, env, epoch, test_policy, train_stats, all_logs):
        # print("all_log: ", all_logs)

        # last_log = all_logs[-1]

        # collect test trajectories, for logging
        print("\nCollecting data for testing...")

        test_paths = utils.sample_n_trajectories(env, test_policy,
         self.params['num_test_episodes'], self.params['max_path_length'], render=True)


        # save metrics

        train_returns = train_stats["EpRet"]
        test_returns = [test_path["reward"].sum() for test_path in test_paths]

        # episode lengths, for logging
        train_ep_lens = train_stats["EpLen"]
        test_ep_lens = [len(test_path["reward"]) for test_path in test_paths]


        logs = OrderedDict()
        logs["Test_AverageReturn"] = np.mean(test_returns)
        logs["Test_StdReturn"] = np.std(test_returns)
        logs["Test_MaxReturn"] = np.max(test_returns)
        logs["Test_MinReturn"] = np.min(test_returns)
        logs["Test_AverageEpLen"] = np.mean(test_ep_lens)

        logs["Train_AverageReturn"] = np.mean(train_returns)
        logs["Train_StdReturn"] = np.std(train_returns)
        logs["Train_MaxReturn"] = np.max(train_returns)
        logs["Train_MinReturn"] = np.min(train_returns)
        logs["Train_AverageEpLen"] = np.mean(train_ep_lens)

        logs["Epoch"] = epoch
        logs["TimeSinceStart"] = time.time() - self.start_time
        logs["PiLoss"] = np.mean(all_logs['PiLoss'])
        logs["QLoss"] = np.mean(all_logs['QLoss'])
        logs["Q_values"] =np.mean(all_logs["Q_values"])

        # if epoch == 0:
        #     self.initial_return = np.mean(train_returns)
        # logs["Initial_DataCollection_AverageReturn"] = self.initial_return

        # perform the logging
        for key, value in logs.items():
            print('{} : {}'.format(key, value))
            self.logger.log_scalar(value, key, epoch)

        self.logger.dump_tabular(epoch, logs)
        print('Done logging...\n\n')

        self.logger.flush()
