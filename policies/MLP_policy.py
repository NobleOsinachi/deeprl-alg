import abc
import itertools
from typing import Any
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from deeprlalg.utils import pytorch_util as ptu
from deeprlalg.policies.base_policy import BasePolicy


class Actor(nn.Module, metaclass=abc.ABCMeta):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a



class MLPCategoricalActor(Actor):
    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 learning_rate=1e-4,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        self.logits_net = ptu.build_mlp(
            input_size=ob_dim,
            output_size=ac_dim,
            n_layers=n_layers,
            size=size,
        )
        self.logits_net.to(ptu.device)
        self.optimizer = optim.Adam(self.logits_net.parameters(),
                                    learning_rate)

    def forward(self, obs):
        return distributions.Categorical(self.logits_net(observation))

    def _log_prob(self, pi, act):
        return pi.log_prob(act)

    def save(self, filepath):
        save_dict = self.state_dict()
        torch.save(save_dict, filepath)



class MLPGaussianActor(Actor):
    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 learning_rate=1e-4,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        self.mean_net = ptu.build_mlp(
            input_size=ob_dim,
            output_size=ac_dim,
            n_layers=n_layers, size=size,
        )
        self.mean_net.to(ptu.device)
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))

        # self.logstd = nn.Parameter(
        #     torch.zeros(self.ac_dim, dtype=torch.float32, device=ptu.device)
        # )
        self.logstd.to(ptu.device)
        self.optimizer = optim.Adam(
            itertools.chain([self.logstd], self.mean_net.parameters()),
            self.learning_rate
        )

    def forward(self, obs):
        mean = self.mean_net(observation)
        pi = distributions.Normal(mean, self.logstd.exp())
        return pi

    def _log_prob(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)

    def save(self, filepath):
        save_dict = {
                'mean': self.state_dict(),
                'logstd': self.logstd
        }

        torch.save(save_dict, filepath)


class MLPCritic(nn.Module):

        def __init__(self,
                     ac_dim,
                     ob_dim,
                     n_layers,
                     size,
                     learning_rate=1e-4,
                     **kwargs
                     ):
        super().__init__()

        self.v_net = ptu.build_mlp(
            input_size=ob_dim,
            output_size=1,
            n_layers=n_layers,
            size=size,
        )
        self.optimizer = optim.Adam(
            self.v_net.parameters(),
            learning_rate,
        )
        self.v_net.to(ptu.device)


    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1) # Critical to ensure v has right shape.
