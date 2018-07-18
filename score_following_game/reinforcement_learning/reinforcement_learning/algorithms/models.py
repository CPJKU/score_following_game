import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Template for creating models used in the reinforcement learning algorithms
    """
    def __init__(self):
        super().__init__()
        self.net = None
        self.optimizer = None

    def forward(self, x):
        return self.net.forward(x)

    def forward_policy(self, x):
        return self.net.forward(x)[0]

    def forward_value(self, x):
        return self.net.forward(x)[1]

    def update(self, losses):
        pass

    def get_learn_rate(self):
        # return learn rate of optimizer for logging
        return self.optimizer.param_groups[0]['lr']

    def get_named_params(self):
        return self.net.named_parameters()

    def save_network(self, name):
        torch.save(self.net.state_dict(), name)


class SeparatedModel(Model):

    def __init__(self, policy_net, value_net, optimizer, entropy_coef=0.01, max_grad_norm=None):
        super().__init__()
        self.policy_net = policy_net
        self.value_net = value_net
        self.optimizer = optimizer
        self.max_grad_norm = max_grad_norm
        self.entropy_coef = entropy_coef

    def forward(self, x):
        return self.policy_net.forward(x), self.value_net.forward(x)

    def forward_policy(self, x):
        return self.policy_net.forward(x)

    def forward_value(self, x):
        return self.value_net.forward(x)

    def update(self, losses):

        for opt in self.optimizer:
            opt.zero_grad()

        if len(losses) == 3:
            # if the entropy is given
            (losses[0] - losses[2] * self.entropy_coef).backward()
        else:
            # policy loss
            losses[0].backward()

        if len(losses) > 1:
            # value loss
            losses[1].backward()

        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm(self.policy_net.parameters(), self.max_grad_norm)
            torch.nn.utils.clip_grad_norm(self.value_net.parameters(), self.max_grad_norm)

        for opt in self.optimizer:
            opt.step()

    def get_learn_rate(self):
        # only the learn rate of the first (policy) optimizer
        return self.optimizer[0].param_groups[0]['lr']

    def get_named_params(self):
        return self.policy_net.named_parameters(), self.value_net.parameters()

    def save_network(self, name):
        torch.save(self.policy_net.state_dict(), name+"_policy_net")
        torch.save(self.value_net.state_dict(), name+"_value_net")


class CombinedModel(Model):

    def __init__(self, net, optimizer, value_coef=0.5, entropy_coef=0.01, max_grad_norm=None):
        super().__init__()
        self.net = net
        self.optimizer = optimizer
        self.max_grad_norm = max_grad_norm
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

    def update(self, losses):
        self.optimizer.zero_grad()

        if len(losses) == 3:
            (losses[0] + losses[1]*self.value_coef - losses[2]*self.entropy_coef).backward()
        elif len(losses) == 2:
            (losses[0] + losses[1] * self.value_coef).backward()
        else:
            losses[0].backward()

        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm(self.net.parameters(), self.max_grad_norm)

        self.optimizer.step()
