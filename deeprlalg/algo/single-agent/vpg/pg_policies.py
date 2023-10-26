import numpy as np


from deeprlalg.algo.policies.MLP_policy import *
from deeprlalg.utils import pytorch_utils as ptu

from deeprlalg.utils import utils

class MLPPolicyPG(nn.Module):
    def __init__(self, ac_dim, ob_dim,  n_layers, size, discrete,
                learning_rate, nn_baseline, **kwargs):
        super().__init__()
        self.nn_baseline = nn_baseline

        if discrete:
            self.pi = MLPCategoricalActor(ac_dim,
                                ob_dim,
                                n_layers,
                                size,
                                learning_rate)
        else:
            self.pi = MLPGaussianActor(ac_dim,
                                ob_dim,
                                n_layers,
                                size,
                                learning_rate)

        if nn_baseline:
            self.baseline_loss = nn.MSELoss()
            self.baseline = MLPCritic(ac_dim,
                                ob_dim,
                                n_layers,
                                size,
                                learning_rate)


    def step(self, obs):
        with torch.no_grad():
            if len(obs.shape) > 1:
                observation = ptu.from_numpy(obs)
            else:
                observation = ptu.from_numpy(obs[None])


            pi = self.pi(observation)
            action = pi.sample()
            logp_a = self.pi._log_prob(pi, action)


        return ptu.to_numpy(action), ptu.to_numpy(logp_a)


    def sample_action(self, obs):
        action = self.pi(obs).sample()
        return action

    def act(self, obs):
        return self.step(obs)[0]

    def update(self, observations, actions, advantages, q_values=None):
        observations = ptu.from_numpy(observations)
        actions = ptu.from_numpy(actions)
        advantages = ptu.from_numpy(advantages)
        dist = self.pi(observations)
        pred_actions = self.sample_action(observations)
        logprob = self.pi._log_prob(dist, actions)

        loss = -1 *torch.mean( logprob * advantages)
        self.pi.optimizer.zero_grad()
        loss.backward()
        self.pi.optimizer.step()

        if self.nn_baseline:
            pred_values = self.baseline(observations)
            q_values_normalized =  utils.normalize(q_values)
            q_values_normalized = ptu.from_numpy(q_values_normalized)

            baseline_loss = self.baseline_loss(pred_values, q_values)
            self.baseline.optimizer.zero_grad()
            baseline_loss.backward()
            self.baseline.optimizer.step()

        train_log = {
            'Training Loss': ptu.to_numpy(loss),

        }
        if self.nn_baseline:
            train_log['Baseline Training Loss']= ptu.to_numpy(baseline_loss)

        return train_log
