import torch

import torch.nn as nn

from score_following_game.reinforcement_learning.torch_extentions.distributions.adapted_categorical import AdaptedCategorical


class Model(nn.Module):
    """
    Template for creating models used in the reinforcement learning algorithms
    """
    def __init__(self, net, optimizer, max_grad_norm=0.5, value_coef=0.5,
                 entropy_coef=0.01, distribution=AdaptedCategorical):
        super(Model, self).__init__()

        self.net = net
        self.optimizer = optimizer
        self.max_grad_norm = max_grad_norm
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.distribution = distribution

    def forward(self, x):
        return self.net.forward(**x)

    def forward_policy(self, x):
        return self.net.forward(**x)['policy']

    def forward_value(self, x):
        return self.net.forward(**x)['value']

    def update(self, losses):

        self.optimizer.zero_grad()
        (losses['policy_loss'] +
         self.value_coef * losses['value_loss'] -
         self.entropy_coef * losses['dist_entropy']).backward()

        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)

        self.optimizer.step()

    def set_train_mode(self):
        self.net.train()

    def set_eval_mode(self):
        self.net.eval()

    def get_learn_rate(self):
        # return learn rate of optimizer for logging
        return self.optimizer.param_groups[0]['lr']

    def get_named_params(self):
        return self.net.named_parameters()

    def save_network(self, name):
        torch.save(self.net.state_dict(), name+'.pt')

    def calc_entropy(self, policy):
        return self.distribution(**policy).entropy().sum(-1).mean(dim=0, keepdim=True)

    def get_log_probs(self, policy, actions):
        return self.distribution(**policy).log_prob(actions).sum(-1, keepdim=True)

    def sample_action(self, policy, deterministic=False):

        distr = self.distribution(**policy)

        if deterministic:
            # in case of discrete actions the mean will be the argmax decision
            actions = distr.mean
        else:
            actions = distr.sample()

        return actions, actions.cpu().detach().numpy()



