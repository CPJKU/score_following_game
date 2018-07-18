from __future__ import print_function

import numpy as np
import torch
from reinforcement_learning.algorithms.models import SeparatedModel
from reinforcement_learning.algorithms.networks import ConvPolicyNet, ConvValueNet
from reinforcement_learning.algorithms.reinforce.reinforce import ReinforceAgent
from reinforcement_learning.envs.pixel_cartpole import PxlsCartPoleEnv
from reinforcement_learning.envs.wrappers import FrameDifference, MultiModalityWrapper
from tensorboardX import SummaryWriter

if __name__ == '__main__':
    """
    main
    """

    # fix a seed for reproducing our results
    np.random.seed(4711)

    verbose = True

    max_steps = 1000  # maximum time steps per episode
    max_epochs = 500  # overall maximum number of epochs

    # select gym environment
    env = MultiModalityWrapper(FrameDifference(PxlsCartPoleEnv(), keep_raw=True))

    # get action space
    n_actions = env.action_space.n
    print("Action Space:", n_actions, env.action_space.__class__)

    # get dimensions of observations
    in_shape = env.observation_space.spaces[0].shape[0]
    print("Observation Space:", in_shape)

    use_cuda = False

    actor = ConvPolicyNet(n_actions=env.action_space.n, input_channels=2)
    critic = ConvValueNet(input_channels=2)

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-4)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=2e-4)

    optimizer = [actor_optimizer, critic_optimizer]

    model = SeparatedModel(policy_net=actor, value_net=critic, optimizer=optimizer)

    if use_cuda:
        model.cuda()

    writer = SummaryWriter()
    agent = ReinforceAgent(env=env, model=model, no_baseline=False, gamma=0.97, use_cuda=use_cuda)

    agent.train(max_updates=max_epochs)

    writer.close()
