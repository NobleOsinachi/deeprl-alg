import numpy as np


from deeprlalg.algo.policies.MLP_policy import *
from deeprlalg.utils import pytorch_utils as ptu



class MLPActor(nn.Module):

    def __init__(self, ac_dim, ob_dim, params):
        super().__init__()

        hidden_sizes = params['hidden_sizes']
        activation = params['activation']
        output_activation = params['output_activation']
        learning_rate = params['learning_rate']
        action_limit = params['ac_limit']


        network_sizes = [ob_dim] + list(hidden_sizes) + [ac_dim]
        # print("ac_dim: ", ac_dim)
        self.pi = ptu.build_mlp(network_sizes, activation, output_activation)

        self.optimizer = optim.Adam(
            self.pi.parameters(),
            learning_rate,
        )
        self.pi.to(ptu.device)
        self.action_limit = action_limit

    def forward(self, obs):
        return self.action_limit * self.pi(obs)

    def save(self, filepath):
        save_dict = self.state_dict()
        torch.save(save_dict, filepath)




class MLPQFunction(nn.Module):

    def __init__(self, ac_dim, ob_dim, params):
        super().__init__()


        hidden_sizes = params['hidden_sizes']
        activation = params['activation']
        output_activation = params['output_activation']
        learning_rate = params['learning_rate']

        network_sizes = [ob_dim+ ac_dim] + list(hidden_sizes) + [1]

        self.q = ptu.build_mlp(network_sizes, activation, output_activation)

        self.optimizer = optim.Adam(
            self.q.parameters(),
            learning_rate,
        )

        self.q.to(ptu.device)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1)

    def save(self, filepath):
        save_dict = self.state_dict()
        torch.save(save_dict, filepath)




class MLPPolicyDDPG(nn.Module):
    def __init__(self, ac_dim, ob_dim, pi_params, q_params):
        super().__init__()

        self.pi = MLPActor(ac_dim, ob_dim, pi_params)

        self.q = MLPQFunction(ac_dim, ob_dim, q_params)


    def step(self, obs):
        obs = ptu.from_numpy(obs)
        with torch.no_grad():
            return ptu.to_numpy(self.pi(obs)), 0

    def act(self, obs):
        obs = ptu.from_numpy(obs)
        with torch.no_grad():
            return ptu.to_numpy(self.pi(obs))

    def save_model(self, filepath, epoch):
        """
        Save the Actor and Q Model after training is completed.
        """

        print('Saving model to {}'.format(filepath))
        pi_filepath = '{}/pi_agent_epoch_{}.pt'.format(filepath, epoch)
        q_filepath = '{}/q_agent_epoch_{}.pt'.format(filepath, epoch)
        torch.save(self.pi.state_dict(), pi_filepath)
        torch.save(self.q.state_dict(), q_filepath)

    def load_model(self, filepath, epoch, map_location=None):
        """
        Load Actor Model from given paths.
        :param actor_path: The actor path.
        :return: None
        """
        pi_filepath = '{}/pi_agent_{}.pt'.format(filepath, epoch)
        q_filepath = '{}/q_agent_{}.pt'.format(filepath, epoch)

        print('Loading pi model from {}'.format(pi_filepath))
        print('Loading q model from {}'.format(q_filepath))
        if filepath is not None:
            self.pi.load_state_dict(torch.load(pi_filepath, map_location))
            self.pi.to(ptu.device)
            self.q.load_state_dict(torch.load(q_filepath, map_location))
            self.q.to(ptu.device)
