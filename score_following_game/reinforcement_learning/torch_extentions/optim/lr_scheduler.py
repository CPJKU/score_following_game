import copy

import numpy as np

from torch.optim.optimizer import Optimizer


class RefinementLRScheduler(object):
    def __init__(self, optimizer, model=None, n_refinement_steps=2, patience=5, learn_rate_multiplier=0.1, high_is_better=False):

        self.n_refinement_steps = n_refinement_steps
        self.patience = patience
        self.learn_rate_multiplier = learn_rate_multiplier
        self.high_is_better = high_is_better

        self.best_score = -np.inf if self.high_is_better else np.inf
        self.last_improvement = 0

        self.model = model
        self.best_model = None
        self.stop_learning = False

        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))

        self.optimizer = optimizer

    def get_lr(self):
        lrs = list(map(lambda group: group['lr'], self.optimizer.param_groups))
        return [lr * self.learn_rate_multiplier for lr in lrs]

    def step(self, score):

        no_improvement = (self.high_is_better and score < self.best_score) or \
                         (not self.high_is_better and score > self.best_score)
        if no_improvement:
            self.last_improvement += 1
        else:
            self.last_improvement = 0
            self.best_score = score
            if self.model is not None:
                self.best_model = copy.deepcopy(self.model.net)

        print('Impatience Level {} of {}'.format(self.last_improvement, self.patience))

        if self.last_improvement == self.patience:
            self.last_improvement = 0
            self.n_refinement_steps -= 1

            if self.model is not None:
                print('Reset model')
                # self.model.net = copy.deepcopy(self.best_model.net)
                self.model.net.load_state_dict(self.best_model.state_dict())
                self.model.model_params = self.model.net.parameters()

            # update lrs for all parameter groups
            if self.n_refinement_steps >= 0:
                for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
                    print('Refine learning rate from {} to {}'.format(param_group['lr'], lr))
                    param_group['lr'] = lr

            else:
                print('No more refinement steps. Setting learning rate to 0')
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = 0.0
                self.stop_learning = True

    def learning_stopped(self):
        return self.stop_learning
