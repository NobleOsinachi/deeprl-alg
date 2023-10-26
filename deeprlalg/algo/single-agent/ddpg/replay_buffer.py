import numpy as np
from deeprlalg.utils.utils import combined_shape
import torch
class ReplayBuffer(object):

    def __init__(self, obs_dim, act_dim, max_size=1000000):

        self.obs_buf = np.zeros(combined_shape(max_size, obs_dim), dtype=np.float32)
        self.obs_n_buf = np.zeros(combined_shape(max_size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(max_size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(max_size, dtype=np.float32)
        self.done_buf = np.zeros(max_size, dtype=np.float32)

        self.ptr, self.size, self.max_size = 0, 0, max_size


    def insert(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs_n_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done

        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample(self, batch_size=64):
        idxs =  np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs_n=self.obs_n_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}
