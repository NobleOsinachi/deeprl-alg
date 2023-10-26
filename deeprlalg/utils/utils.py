import time
import numpy as np
import scipy.signal
color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38
)

def colorize(string, color, bold=False, highlight=False):
    attr = []
    num = color2num[color]
    if highlight: num += 10
    attr.append(str(num))
    if bold: attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)


def Trajectory(obs, actions, rewards, next_obs, terminals, traj_probs):
    return {"observation" : np.array(obs, dtype=np.float32),
            "action" : np.array(actions, dtype=np.float32),
            "reward" : np.array(rewards, dtype=np.float32),
            "next_observation": np.array(next_obs, dtype=np.float32),
            "terminal": np.array(terminals, dtype=np.float32),
            "traj_probs": np.array(traj_probs, dtype=np.float32)}


def sample_trajectory(env, policy, max_path_length, render=False):
    # initialize env for the beginning of a new rollout
    ob = env.reset()

    # init vars
    obs, acs, rewards, next_obs, terminals, \
    traj_probs  =  [], [], [], [], [], []

    steps = 0
    rollout_done = False
    while True:
        if render:
            env.render()
            time.sleep(0.1)
        obs.append(ob)
        ac, logp = policy.step(ob)
        acs.append(ac)
        traj_probs.append(logp)


        # take the action to get next obs and reward
        ob, rew, done, _ = env.step(ac)

        steps += 1
        next_obs.append(ob)
        rewards.append(rew)

        if done or steps == max_path_length:
            rollout_done = True


        terminals.append(rollout_done)

        if rollout_done:
            break
        # print("rollout: ", rollout_done)

    return Trajectory(obs, acs, rewards, next_obs, terminals, traj_probs)

def sample_trajectories(env, policy, min_timesteps_per_batch, max_path_length, render=False):
    """
        Collect rollouts until we have collected min_timesteps_per_batch steps.

    """
    timesteps_this_batch = 0
    paths = []
    while timesteps_this_batch < min_timesteps_per_batch:
        path = sample_trajectory(env, policy, max_path_length, render)
        paths.append(path)
        timesteps_this_batch += get_pathlength(path)


    return paths, timesteps_this_batch

def sample_n_trajectories(env, policy, ntraj, max_path_length, render=False):
    """
        Collect rollouts until we have collected min_timesteps_per_batch steps.

    """
    paths = []
    for i in range(ntraj):
        path = sample_trajectory(env, policy, max_path_length, render)
        paths.append(path)

    return paths

def get_pathlength(path):
    return len(path["reward"])

def normalize(data, mean, std, eps=1e-8):
    return (data-mean)/(std+eps)

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def discount_cumsum(x, discount):

    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]
