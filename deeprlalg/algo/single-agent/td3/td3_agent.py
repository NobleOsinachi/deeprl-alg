from td3_policies import MLPPolicyTD3
import numpy as np
from base_agent import BaseAgent
from copy import deepcopy
from replay_buffer import ReplayBuffer
import torch
import itertools

from deeprlalg.utils import pytorch_utils as ptu
class TD3Agent(BaseAgent):
    def __init__(self, env, agent_params):
        super(TD3Agent, self).__init__()

        # init vars
        self.env = env
        self.agent_params = agent_params
        self.gamma = self.agent_params['gamma']
        self.polyak = self.agent_params['polyak']
        self.ac_dim = self.agent_params['ac_dim']
        self.ob_dim = self.agent_params['ob_dim']
        self.target_noise = self.agent_params['target_noise']
        self.noise_clip = self.agent_params['noise_clip']
        self.policy_delay = self.agent_params['policy_delay']

        pi_params = self.agent_params['pi_params']
        self.ac_limit =  pi_params['ac_limit']
        q_params = self.agent_params['q_params']


        # policy and q function
        self.actor_critic = MLPPolicyTD3(self.ac_dim, self.ob_dim, pi_params, q_params)

        self.actor_critic_targ = deepcopy(self.actor_critic)

        # Freeze target networks since update is only via polyak averaging and not the optimizers
        for p in self.actor_critic_targ.parameters():
            p.requires_grad = False

        # List of parameters for both Q-networks (save this for convenience)
        self.q_params = itertools.chain(self.actor_critic.q1.parameters(), \
                                        self.actor_critic.q2.parameters())


        self.replay_buffer = ReplayBuffer(obs_dim=  self.agent_params['ob_dim'],
                                          act_dim= self.agent_params['ac_dim'],
                                          max_size=int(self.agent_params['buffer_size']))


    def compute_q_loss(self, paths):
        obs, act, rew, next_o, done = paths['obs'], paths['act'], paths['rew'], paths['obs_n'], paths['done']

        obs = ptu.to_gpu(obs)
        act = ptu.to_gpu(act)
        next_o = ptu.to_gpu(next_o)
        rew = ptu.to_gpu(rew)
        done = ptu.to_gpu(done)
        # self.gamma  = ptu.to_gpu(self.gamma )

        q1_values = self.actor_critic.q1(obs, act)
        q2_values = self.actor_critic.q2(obs, act)


        # Bellman Equation for Q function
        with torch.no_grad():
            pi_targ = self.actor_critic_targ.pi(next_o)

            # Target policy smoothing
            epsilon = torch.randn_like(pi_targ) * self.target_noise
            epsilon = torch.clamp(epsilon, -self.noise_clip, self.noise_clip)
            a2 = pi_targ + epsilon
            a2 = torch.clamp(a2, -self.act_limit, self.act_limit)


            q1_pi_targ = self.actor_critic_targ.q1(next_o, a2)
            q2_pi_targ = self.actor_critic_targ.q2(next_o, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            target = rew + self.gamma * (1 - done) * q_pi_targ

        #Mean-squared Bellman error (MSBE) function
        loss_q1 = ((q1_values - target)**2).mean()
        loss_q2 = ((q2_values - target)**2).mean()
        loss_q = loss_q1 + loss_q2


        return loss_q, ptu.to_numpy(q1_values), ptu.to_numpy(q2_values)

    def compute_pi_loss(self, paths):
        obs =  ptu.to_gpu(paths['obs'])
        q1_pi = self.actor_critic.q1(obs, self.actor_critic.pi(obs))
        return -q1_pi.mean()


    def train(self, paths, itr):

        """
            Training a TD3 agent refers to updating its actor and q networks
            using the given trajectories

        """
        # Gradient Descent for Q
        self.actor_critic.q1.optimizer.zero_grad()
        q_loss, q1_values, q2_values = self.compute_q_loss(paths)
        q_loss.backward()
        self.actor_critic.q1.optimizer.step()

        train_log = {}

        if itr % self.policy_delay == 0:

            for p in self.q_params:
                p.requires_grad = False


            # Gradient Descent for Pi
            self.actor_critic.pi.optimizer.zero_grad()
            pi_loss = self.compute_pi_loss(paths)
            pi_loss.backward()
            self.actor_critic.pi.optimizer.step()

            for p in self.q_params:
                p.requires_grad = True

            # Finally, update target networks by polyak averaging.
            with torch.no_grad():
                for p, p_targ in zip(self.actor_critic.parameters(), self.actor_critic_targ.parameters()):
                    # NB: We use an in-place operations "mul_", "add_" to update target
                    # params, as opposed to "mul" and "add", which would make new tensors.
                    p_targ.data.mul_(self.polyak)
                    p_targ.data.add_((1 - self.polyak) * p.data)

            train_log = {
                    'PiLoss': ptu.to_numpy(pi_loss),
                    'Q1_values': q1_values,
                    'Q2_values': q2_values
                }

        train_log['QLoss'] = ptu.to_numpy(q_loss)

        return train_log

    def get_action(self, o, noise_scale):

        a, _ = self.actor_critic.step(o)

        a += noise_scale * np.random.randn(self.ac_dim)
        return np.clip(a, -self.ac_limit, self.ac_limit)


    def save(self, path, epoch):
        self.actor_critic.save_model(path, epoch)
