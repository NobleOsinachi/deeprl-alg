from maddpg_policies import MLPPolicyMADDPG
import numpy as np
from base_agent import BaseAgent
from copy import deepcopy
from replay_buffer import ReplayBuffer
import torch


from deeprlalg.utils import pytorch_utils as ptu
class MADDPGAgent(BaseAgent):
    def __init__(self, env, agent_params):
        super(MADDPGAgent, self).__init__()

        # init vars
        self.env = env
        self.agent_params = agent_params
        self.gamma = self.agent_params['gamma']
        self.polyak = self.agent_params['polyak']
        self.ac_dim = self.agent_params['ac_dim']
        self.ob_dim = self.agent_params['ob_dim']



        pi_params = self.agent_params['pi_params']
        self.ac_limit =  pi_params['ac_limit']
        q_params = self.agent_params['q_params']


        # policy and q function
        self.actor_critic = MLPPolicyMADDPG(self.ac_dim, self.ob_dim, pi_params, q_params)

        self.actor_critic_targ = deepcopy(self.actor_critic)

        for p in self.actor_critic_targ.parameters():
            p.requires_grad = False

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
        q_values = self.actor_critic.q(obs, act)



        # Bellman Equation for Q function
        with torch.no_grad():
            q_pi_targ = self.actor_critic_targ.q(next_o, self.actor_critic.pi(next_o))
            target = rew + self.gamma * (1 - done) * q_pi_targ

        #Mean-squared Bellman error (MSBE) function
        loss = ((q_values - target) ** 2).mean()

        return loss, ptu.to_numpy(q_values)

    def compute_pi_loss(self, paths):
        obs =  ptu.to_gpu(paths['obs'])
        q_pi = self.actor_critic.q(obs, self.actor_critic.pi(obs))
        return -q_pi.mean()


    def train(self, paths):

        """
            Training a MADDPG agent refers to updating its actor and q networks
            using the given trajectories

        """
        # Gradient Descent for Q
        self.actor_critic.q.optimizer.zero_grad()
        q_loss, q_values = self.compute_q_loss(paths)
        q_loss.backward()
        self.actor_critic.q.optimizer.step()

        for p in self.actor_critic.q.parameters():
            p.requires_grad = False

        # Gradient Descent for Pi
        self.actor_critic.pi.optimizer.zero_grad()
        pi_loss = self.compute_pi_loss(paths)
        pi_loss.backward()
        self.actor_critic.pi.optimizer.step()

        for p in self.actor_critic.q.parameters():
            p.requires_grad = True

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.actor_critic.parameters(), self.actor_critic_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

        train_log = {
                'QLoss': ptu.to_numpy(q_loss),
                'PiLoss': ptu.to_numpy(pi_loss),
                'Q_values': q_values
            }


        return train_log

    def get_action(self, o, noise_scale):

        a, _ = self.actor_critic.step(o)

        a += noise_scale * np.random.randn(self.ac_dim)
        return np.clip(a, -self.ac_limit, self.ac_limit)


    def save(self, path, epoch):
        self.actor_critic.save_model(path, epoch)
