import torch
import math
import numpy as np
from torch.distributions import Normal
from reinforcement_learning.algorithms.a2c.a2c import A2CAgent


class ContinuousA2CAgent(A2CAgent):

    def __init__(self, env, model, t_max=5, n_worker=1, gamma=0.99, use_cuda=torch.cuda.is_available(),
                 distribution=Normal):
        A2CAgent.__init__(self, env, model, t_max, n_worker, gamma, use_cuda)

        self.action_tensor = torch.FloatTensor
        self.distribution = distribution

    def _calc_entropy(self, policy):
        # TODO update pytorch such that distribution.entropy is usable
        # distr = self.distribution(**policy)
        # return distr.entropy()

        return (0.5 + 0.5 * math.log(2 * math.pi) + torch.log(policy['std'])).mean()

    def _get_log_probs(self, policy, actions):
        # create a distribution from the parameters, policy has to be a dictionary
        distr = self.distribution(**policy)
        return distr.log_prob(actions)

    def _sample_action(self, policy):

        # transform variables to tensors policy has to be a dictionary
        distribution_params = {key: policy[key].data for key in policy}

        # create a distribution from the parameters
        distr = self.distribution(**distribution_params)

        # draw a random action by sampling from the policy prediction
        actions = distr.sample()

        # expand as continuous environments need each action inside of an array
        return actions, np.expand_dims(actions.view(-1).cpu().numpy(), 1)
