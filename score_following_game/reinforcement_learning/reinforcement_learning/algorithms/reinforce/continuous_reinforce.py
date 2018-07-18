import torch
import numpy as np
from torch.distributions import Normal
from reinforcement_learning.algorithms.reinforce.reinforce import ReinforceAgent


class ContinuousReinforceAgent(ReinforceAgent):

    def __init__(self, env, model, no_baseline=False, gamma=0.99, use_cuda=torch.cuda.is_available(), max_steps=1e8,
                 distribution=Normal):

        ReinforceAgent.__init__(self, env, model, no_baseline, gamma, use_cuda, max_steps)
        self.action_dtype = np.float32
        self.distribution = distribution

    def _sample_action(self, state):

        # return of forward pass has to be a dictionary
        distribution_params = self.model.forward_policy(state)

        # transform variables to tensors
        distribution_params = {key: distribution_params[key].data for key in distribution_params}

        # create a distribution from the parameters
        distr = self.distribution(**distribution_params)

        # draw a random action by sampling from the policy prediction [0][0] to unpack the values
        return np.asarray([distr.sample()[0][0]])

    def _get_log_probs(self, policy, actions):

        # create a distribution from the parameters, policy has to be a dictionary
        distr = self.distribution(**policy)

        return distr.log_prob(actions)
