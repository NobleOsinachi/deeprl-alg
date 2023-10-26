import abc
import itertools
from typing import Any
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from deeprlalg.utils import pytorch_utils as ptu


class Actor(nn.Module, metaclass=abc.ABCMeta):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob(pi, act)
        return pi, logp_a


class MLPCategoricalActor(Actor):
    def __init__(self, ac_dim, ob_dim, params):
        super().__init__()

        hidden_sizes = params['hidden_sizes']
        activation = params['activation']
        output_activation = params['output_activation']
        learning_rate = params['learning_rate']

        network_sizes = [ob_dim] + list(hidden_sizes) + [ac_dim]

        self.logits_net = ptu.build_mlp(network_sizes, activation, output_activation)
        self.logits_net.to(ptu.device)
        self.optimizer = optim.Adam(self.logits_net.parameters(),
                                    learning_rate)

    def _distribution(self, obs):
        return distributions.Categorical(logits=self.logits_net(obs))

    def _log_prob(self, pi, act):
        return pi.log_prob(act)

    def save(self, filepath):
        save_dict = self.state_dict()
        torch.save(save_dict, filepath)

class MLPGaussianActor(Actor):
    def __init__(self, ac_dim, ob_dim, params):
        super().__init__()

        hidden_sizes = params['hidden_sizes']
        activation = params['activation']
        output_activation = params['output_activation']
        learning_rate = params['learning_rate']

        network_sizes = [ob_dim] + list(hidden_sizes) + [ac_dim]

        self.mean_net = ptu.build_mlp(network_sizes, activation, output_activation)
        self.mean_net.to(ptu.device)
        log_std = -0.5 * np.ones(ac_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.log_std.to(ptu.device)
        self.optimizer = optim.Adam(
            itertools.chain([self.log_std], self.mean_net.parameters()),
            learning_rate
        )

    def _distribution(self, obs):
        mean = self.mean_net(obs)
        pi = distributions.Normal(mean, self.log_std.exp())
        return pi

    def _log_prob(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)

    def save(self, filepath):
        save_dict = {
                'mean': self.state_dict(),
                'logstd': self.log_std
        }

        torch.save(save_dict, filepath)


class MLPCritic(nn.Module):

        def __init__(self, ac_dim, ob_dim, params):
            super().__init__()

            hidden_sizes = params['hidden_sizes']
            activation = params['activation']
            output_activation = params['output_activation']
            learning_rate = params['learning_rate']


            network_sizes = [ob_dim] + list(hidden_sizes) + [1]

            self.v_net = ptu.build_mlp(network_sizes, activation, output_activation)

            self.optimizer = optim.Adam(
                self.v_net.parameters(),
                learning_rate,
            )
            self.v_net.to(ptu.device)


        def forward(self, obs):
            return torch.squeeze(self.v_net(obs), -1) # Critical to ensure v has right shape.

        def save(self, filepath):
            save_dict = self.state_dict()
            torch.save(save_dict, filepath)

class MLPPolicyPPO(nn.Module):
    def __init__(self, ac_dim, ob_dim, pi_params, q_params):
        super().__init__()

        discrete = pi_params['discrete']

        if discrete:
            self.pi = MLPCategoricalActor(ac_dim, ob_dim, pi_params)
        else:
            self.pi = MLPGaussianActor(ac_dim, ob_dim, pi_params)

        self.v = MLPCritic(ac_dim, ob_dim, q_params)

    def step(self, obs):
        obs = ptu.from_numpy(obs)
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob(pi, a)
            v = self.v(obs)
        return ptu.to_numpy(a), ptu.to_numpy(v), ptu.to_numpy(logp_a)

    def act(self, obs):
        obs = ptu.from_numpy(obs)
        with torch.no_grad():
            return ptu.to_numpy(self.pi(obs))
