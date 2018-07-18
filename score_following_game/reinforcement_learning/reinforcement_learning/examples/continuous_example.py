from __future__ import print_function

import gym
import torch
import numpy as np
from reinforcement_learning.algorithms.networks import MuSigmaNet, ValueNet
from reinforcement_learning.algorithms.models import SeparatedModel
from reinforcement_learning.algorithms.reinforce.continuous_reinforce import ContinuousReinforceAgent
from reinforcement_learning.algorithms.a2c.continuous_a2c import ContinuousA2CAgent
from reinforcement_learning.algorithms.ppo.continuous_ppo import ContinuousPPOAgent
from reinforcement_learning.envs.wrappers import MultiModalityWrapper
from reinforcement_learning.envs.multi_modal_subproc_vec_env import MultiModalSubprocVecEnv
from tensorboardX import SummaryWriter


class NormalizedActions(gym.ActionWrapper):

    def action(self, action):
        action = (action + 1) / 2  # [-1, 1] => [0, 1]
        action *= (self.action_space.high - self.action_space.low)
        action += self.action_space.low
        return action

    def reverse_action(self, action):
        action -= self.action_space.low
        action /= (self.action_space.high - self.action_space.low)
        action = action * 2 - 1
        return action


def make_env(rank, seed=1, env_id='Pendulum-v0'):
    env = gym.make(env_id)
    env.seed(seed + rank)
    return MultiModalityWrapper(NormalizedActions(env))


def get_make_env(rank, seed=1, env_id='Pendulum-v0'):
    def _thunk():
        return make_env(rank, seed, env_id)

    return _thunk


if __name__ == '__main__':
    """
    main
    """

    # add argument parser
    import argparse

    parser = argparse.ArgumentParser(description='Continuous Example.')

    parser.add_argument('--agent', help='reinforcement learning algorithm [reinforce|a2c|ppo].', type=str,
                        default="reinforce")

    parser.add_argument('--env', help='the environment', type=str, default='Pendulum-v0')

    parser.add_argument('--max_updates', help='maximum number of update steps', type=np.int, default=1000)
    parser.add_argument('--n_worker', help='number of parallel workers.', type=np.int, default=8)
    parser.add_argument('--use_cuda', help='use cuda or not.', default=False, action='store_true')

    parser.add_argument('--seed', help='random seed.', type=np.int, default=4711)

    args = parser.parse_args()

    if args.agent == 'reinforce':
        env = make_env(1, seed=args.seed, env_id=args.env)
    else:
        env = MultiModalSubprocVecEnv([get_make_env(rank=i, seed=args.seed, env_id=args.env)
                                       for i in range(args.n_worker)])

    # fix a seed for reproducing our results
    np.random.seed(args.seed)

    print('Algorithm: {}'.format(args.agent))
    print('Environment: {}'.format(args.env))

    # get dimensions of observations
    in_shape = env.observation_space.spaces[0].shape[0]
    print("Observation Space:", in_shape)

    policy_net = MuSigmaNet(input_shape=in_shape, hidden_size=64)
    value_net = ValueNet(input_shape=in_shape, hidden_size=128)

    policy_optimizer = torch.optim.Adam(params=policy_net.parameters(), lr=1e-3)
    value_optimizer = torch.optim.Adam(params=value_net.parameters(), lr=2e-5)
    optimizer = [policy_optimizer, value_optimizer]

    model = SeparatedModel(policy_net=policy_net, value_net=value_net, optimizer=optimizer, max_grad_norm=0.5,
                           entropy_coef=0.001)

    if args.use_cuda:
        model.cuda()

    writer = SummaryWriter()

    if args.agent == 'reinforce':
        agent = ContinuousReinforceAgent(env=env, model=model, no_baseline=False, gamma=0.95, use_cuda=args.use_cuda)
    elif args.agent == 'a2c':
        agent = ContinuousA2CAgent(env=env, model=model, n_worker=args.n_worker, t_max=15,
                                   gamma=0.95, use_cuda=args.use_cuda)
    elif args.agent == 'ppo':
        agent = ContinuousPPOAgent(env=env, model=model, n_worker=args.n_worker, t_max=15,
                                   gamma=0.95, use_cuda=args.use_cuda)
    else:
        raise NotImplementedError('Invalid Algorithm')

    agent.train(max_updates=args.max_updates)

    writer.close()


