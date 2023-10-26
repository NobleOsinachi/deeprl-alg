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
from ppo_agent import PPOAgent

from ppo_policies import MLPPolicyPPO


class PPO(object):
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




        # PPO Actor and Q PARAMS
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
            'discrete':discrete
        }

        self.params['agent_class'] = PPOAgent
        self.agent_params = {}
        self.agent_params['pi_params'] = pi_params
        self.agent_params['clip_ratio'] = self.params['clip_ratio']
        self.agent_params['num_train_pi_iters'] = self.params['num_train_pi_iters']
        self.agent_params['num_train_v_iters'] = self.params['num_train_v_iters']
        self.agent_params['target_kl'] =self.params['target_kl']
        self.agent_params['q_params'] = q_params
        self.agent_params['ac_dim'] = ac_dim
        self.agent_params['ob_dim'] = ob_dim
        self.agent_params['gamma'] = self.params['gamma']
        self.agent_params['lam'] = self.params['lam']
        self.agent_params['buffer_size'] = self.params['steps_per_epoch']


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



        print(colorize("==================================", 'green', bold=True))
        print(colorize("Running Experiment", 'green', bold=True))

        for epoch in range(self.total_envsteps):
            train_stats = dict(EpRet=[],
                            EpLen = [],
                            Values = [])
            for t in range(self.steps_per_epoch):

                # decide if metrics should be logged
                if self.params['scalar_log_freq'] == -1:
                    self.logmetrics = False
                elif t % self.params['scalar_log_freq'] == 0:
                    self.logmetrics = True
                else:
                    self.logmetrics = False

                a, v, logp = self.agent.step(o)

                next_o, r, d, _ = self.env.step(a)
                ep_ret += r
                ep_len += 1



                # Store experience to replay buffer
                self.agent.replay_buffer.insert(o, a, r, v, logp)
                train_stats['Values'].append(v)

                #update the next observation
                o = next_o

                timeout = ep_len == self.params['max_path_length']
                terminal = d or timeout
                epoch_ended = t==self.steps_per_epoch-1

                if terminal or epoch_ended:
                    if epoch_ended and not(terminal):
                        print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                    # if trajectory didn't reach terminal state, bootstrap value target
                    if timeout or epoch_ended:
                        _, v, _ = self.agent.step(o)
                    else:
                        v = 0
                    self.agent.replay_buffer.finish_path(v)
                    if terminal:
                        # only save EpRet / EpLen if trajectory finished
                        train_stats['EpRet'].append(ep_ret)
                        train_stats['EpLen'].append(ep_len)
                    o, ep_ret, ep_len = self.env.reset(), 0, 0


            if self.params['save_params'] and (epoch % self.params['save_freq'] == 0 or epoch == self.epochs):
                self.agent.save(self.params['logdir'], epoch)


            #Train PPO agent
            train_log=self.agent.train(self.agent.replay_buffer.get())


            print(colorize("\n\n********** Epoch %i ************"%epoch, 'cyan', bold=True))

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
         self.params['num_test_episodes'], self.params['max_path_length'])


        # save metrics

        train_returns = train_stats["EpRet"]
        train_values = train_stats["Values"]
        test_returns = [test_path["reward"].sum() for test_path in test_paths]

        # episode lengths, for logging
        train_ep_lens = train_stats["EpLen"]
        test_ep_lens = [len(test_path["reward"]) for test_path in test_paths]


        logs = OrderedDict()
        logs["Test/AverageReturn"] = np.mean(test_returns)
        logs["Test/StdReturn"] = np.std(test_returns)
        logs["Test/MaxReturn"] = np.max(test_returns)
        logs["Test/MinReturn"] = np.min(test_returns)
        logs["Test/AverageEpLen"] = np.mean(test_ep_lens)

        logs["Train_AverageValues"] = np.mean(train_values)
        logs["Train_StdValues"] = np.std(train_values)
        logs["Train_MaxValues"] = np.max(train_values)
        logs["Train_MinValues"] = np.min(train_values)

        logs["Train_AverageReturn"] = np.mean(train_returns)
        logs["Train_StdReturn"] = np.std(train_returns)
        logs["Train_MaxReturn"] = np.max(train_returns)
        logs["Train_MinReturn"] = np.min(train_returns)
        logs["Train_AverageEpLen"] = np.mean(train_ep_lens)

        logs["Epoch"] = epoch
        logs["TimeSinceStart"] = time.time() - self.start_time
        logs.update(all_logs)

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
