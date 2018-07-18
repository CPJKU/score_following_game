from __future__ import print_function

import gym
import numpy as np
import torch
from baselines.common.atari_wrappers import wrap_deepmind, make_atari
from reinforcement_learning.algorithms.a2c.a2c import A2CAgent
from reinforcement_learning.algorithms.models import CombinedModel
from reinforcement_learning.algorithms.networks import PongNet
from reinforcement_learning.envs.wrappers import FrameDifference, PreProcessPong, WrapPyTorch
from reinforcement_learning.envs.multi_modal_subproc_vec_env import MultiModalSubprocVecEnv
from reinforcement_learning.envs.wrappers import MultiModalityWrapper


def make_env(rank, seed=1, env_id='PongNoFrameskip-v4'):
    def _thunk():
        env = MultiModalityWrapper(WrapPyTorch(wrap_deepmind(make_atari(env_id), frame_stack=True)))

        env.seed(seed+rank)
        return env

    return _thunk


def make_env_pong():
    def _thunk():
        return FrameDifference(WrapPyTorch(PreProcessPong(gym.make('PongNoFrameskip-v4'))), keep_raw=True)

    return _thunk


if __name__ == '__main__':
    """ main """

    # fix a seed for reproducing our results
    np.random.seed(4711)

    n_worker = 16

    env = MultiModalSubprocVecEnv([make_env(rank=i, env_id='PongNoFrameskip-v4') for i in range(n_worker)])

    net = PongNet(input_channels=4, n_actions=env.action_space.n)
    optimizer = torch.optim.Adam(params=net.parameters(), lr=1e-4, betas=(0.5, 0.999))
    model = CombinedModel(net=net, optimizer=optimizer, max_grad_norm=0.5)

    use_cuda = False

    if use_cuda:
        model.cuda()

    a2c_agent = A2CAgent(env, model=model, t_max=5, n_worker=n_worker, use_cuda=use_cuda)

    a2c_agent.train(max_updates=10000000)
