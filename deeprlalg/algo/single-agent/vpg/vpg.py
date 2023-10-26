from collections import OrderedDict
import pickle
import os
import sys
import time

import gym
from gym import wrappers
import numpy as np
import torch

from deeprlalg.utils.logger import Logger

from deeprlalg.utils import utils
from deeprlalg.utils import pytorch_utils as pu
from deeprlalg.utils.action_noise_wrapper import ActionNoiseWrapper
from deeprlalg.utils.utils import colorize
from pg_agent import PGAgent


class VPG(object):
    def __init__(self, params):

        # Create Logger and save params
        self.params = params
        self.logger = Logger(self.params['logdir'])
        self.logger.save_params(params)

        #Set Random seed and initialize gpu
        seed = self.params['seed']
        torch.manual_seed(seed)
        np.random.seed(seed)

        pu.init_gpu(
            use_gpu=not self.params['no_gpu'],
            gpu_id=self.params['which_gpu']
        )

        # Create env and set env variables
        self.env = gym.make(self.params['env'])
        self.env.seed(seed)

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


        # PG Agent PARAMS
        computation_graph_args = {
            'n_layers': params['n_layers'],
            'size': params['size'],
            'learning_rate': params['learning_rate'],
            }

        estimate_advantage_args = {
            'gamma': params['discount'],
            'standardize_advantages': not(params['dont_standardize_advantages']),
            'reward_to_go': params['reward_to_go'],
            'nn_baseline': params['nn_baseline'],
            'gae_lambda': params['gae_lambda'],
        }

        train_args = {
            'num_agent_train_steps_per_iter': params['num_agent_train_steps_per_iter'],
        }

        agent_params = {**computation_graph_args, **estimate_advantage_args, **train_args}

        self.params['agent_class'] = PGAgent
        self.params['agent_params'] = agent_params
        self.params['agent_params']['discrete'] = discrete
        self.params['agent_params']['ac_dim'] = ac_dim
        self.params['agent_params']['ob_dim'] = ob_dim

        #############
        ## AGENT
        #############

        agent_class = self.params['agent_class']
        self.agent = agent_class(self.env, self.params['agent_params'])
        self.actor = self.agent.actor

        ## Training PARAMS
        self.epochs = params['epochs']



    def run(self):

        # init vars at beginning of training
        self.total_envsteps = 0
        self.start_time = time.time()
        print(colorize("==================================", 'green', bold=True))
        print(colorize("Running Experiment", 'green', bold=True))

        for epoch in range(self.epochs):
            print(colorize("\n\n********** Epoch %i ************"%epoch, 'cyan', bold=True))

            # decide if metrics should be logged
            if self.params['scalar_log_freq'] == -1:
                self.logmetrics = False
            elif epoch % self.params['scalar_log_freq'] == 0:
                self.logmetrics = True
            else:
                self.logmetrics = False

            # collect trajectories, to be used for training
            paths, envsteps_this_batch = self.collect_rollouts(
                                            self.actor,
                                            self.params['batch_size'])



            self.total_envsteps += envsteps_this_batch

            #data for training
            ob_batch = np.concatenate([path["observation"] for path in paths])
            ac_batch = np.concatenate([path["action"] for path in paths])
            re_batch =[path["reward"] for path in paths]
            next_ob_batch = np.concatenate([path["next_observation"] for path in paths])
            terminal_batch = np.concatenate([path["terminal"] for path in paths])

            # train agent
            train_log = self.agent.train(ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch)

            # log/save
            if self.logmetrics:
                # perform logging

                self.perform_logging(epoch, self.actor, paths, train_log)

                if self.params['save_params']:
                    self.agent.save('{}/agent_epoch_{}.pt'.format(self.params['logdir'], epoch))

        print(colorize("End of Experiment", 'green', bold=True))
        print(colorize("==================================", 'green', bold=True))


    def collect_rollouts(self, agent_policy, batch_size):

        rollouts, envsteps_this_batch = utils.sample_trajectories(self.env, agent_policy,
                                                batch_size, self.params['ep_len'],
                                                render=False)
        return rollouts, envsteps_this_batch


    def perform_logging(self, epoch, test_policy, paths, all_logs):
        print("all_log: ", all_logs)

        last_log = all_logs

        # collect test trajectories, for logging
        print("\nCollecting data for testing...")

        test_paths, test_envsteps_this_batch = \
        utils.sample_trajectories(self.env, test_policy,
                                self.params['test_batch_size'],
                                self.params['ep_len'],
                                self.params['render'])

        # save metrics
        if self.logmetrics:
            # returns, for logging
            train_returns = [path["reward"].sum() for path in paths]
            test_returns = [test_path["reward"].sum() for test_path in test_paths]

            # episode lengths, for logging
            train_ep_lens = [len(path["reward"]) for path in paths]
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

            logs["Train_EnvstepsSoFar"] = self.total_envsteps
            logs["TimeSinceStart"] = time.time() - self.start_time
            logs.update(last_log)

            if epoch == 0:
                self.initial_return = np.mean(train_returns)
            logs["Initial_DataCollection_AverageReturn"] = self.initial_return

            # perform the logging
            for key, value in logs.items():
                print('{} : {}'.format(key, value))
                self.logger.log_scalar(value, key, epoch)

            self.logger.dump_tabular(epoch, logs)
            print('Done logging...\n\n')

            self.logger.flush()
