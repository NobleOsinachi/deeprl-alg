from ppo_policies import MLPPolicyPPO
import numpy as np
from base_agent import BaseAgent
from copy import deepcopy
from replay_buffer import ReplayBuffer
import torch
import itertools

from deeprlalg.utils import pytorch_utils as ptu

class PPOAgent(BaseAgent):
    def __init__(self, env, agent_params):
        super(PPOAgent, self).__init__()

        # init vars
        self.env = env
        self.agent_params = agent_params
        self.gamma = self.agent_params['gamma']
        self.ac_dim = self.agent_params['ac_dim']
        self.ob_dim = self.agent_params['ob_dim']

        pi_params = self.agent_params['pi_params']

        q_params = self.agent_params['q_params']
        self.clip_ratio = self.agent_params['clip_ratio']
        self.target_kl = self.agent_params['target_kl']
        self.num_train_pi_iters = self.agent_params['num_train_pi_iters']
        self.num_train_v_iters = self.agent_params['num_train_v_iters']

        # policy and q function
        self.actor_critic = MLPPolicyPPO(self.ac_dim, self.ob_dim, pi_params, q_params)

        self.replay_buffer = ReplayBuffer(obs_dim=  self.agent_params['ob_dim'],
                                          act_dim= self.agent_params['ac_dim'],
                                          max_size=int(self.agent_params['buffer_size']),
                                          gamma= self.agent_params['gamma'],
                                           lam= self.agent_params['lam'])


    def compute_pi_loss(self, paths):
        obs, act, adv, logp_old = paths['obs'], paths['act'], paths['adv'], paths['logp']

        obs = ptu.to_gpu(obs)
        act = ptu.to_gpu(act)
        adv = ptu.to_gpu(adv)
        logp_old = ptu.to_gpu(logp_old)

        # Policy loss
        pi, logp = self.actor_critic.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1+self.clip_ratio) | ratio.lt(1-self.clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info


    def compute_v_loss(self, paths):
        obs, ret =  ptu.to_gpu(paths['obs']), ptu.to_gpu(paths['ret'])
        return ((self.actor_critic.v(obs) - ret)**2).mean()


    def train(self, paths):

        """
            Training a PPO agent refers to updating its actor and value networks
            using the given trajectories

        """
        # Gradient Descent for Q

        pi_l_old, pi_info_old = self.compute_pi_loss(paths)
        pi_l_old = pi_l_old.item()
        v_l_old = self.compute_v_loss(paths).item()

        # Train policy with multiple steps of gradient descent
        for i in range(self.num_train_pi_iters):
            self.actor_critic.pi.optimizer.zero_grad()
            loss_pi, pi_info = self.compute_pi_loss(data)
            kl = pi_info['kl'].mean()
            if kl > 1.5 * self.target_kl:
                print('Early stopping at step %d due to reaching max kl.'%i)
                break
            loss_pi.backward()
            self.actor_critic.pi.step()

        pi_end_itr = i

        # Value function learning
        for i in range(self.num_train_v_iters):
            self.actor_critic.v.zero_grad()
            loss_v = self.compute_v_loss(data)
            loss_v.backward()
            self.actor_critic.v.step()

        # Log changes from update
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']

        train_log = {
                'StopIter': i,
                'pi/kl': kl,
                'pi/Entropy': ent,
                'pi/ClipFrac': cf,
                'Loss/pi_loss_old': pi_l_old,
                'Loss/v_loss_old': v_l_old,
                'Loss/LossPi': ptu.to_numpy(loss_pi),
                'Loss/LossV': loss_v.item(),
                'Loss/DeltaLossPi': loss_pi.item()- pi_l_old,
                'Loss/DeltaLossV': loss_v.item() - v_l_old
            }

        return train_log

    def step(self, o):

        a, v, logp = self.actor_critic.step(o)
        return a, v, logp


    def save(self, path, epoch):
        self.actor_critic.save_model(path, epoch)
