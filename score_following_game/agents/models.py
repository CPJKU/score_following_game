
import torch
from torch.autograd import Variable

from reinforcement_learning.algorithms.models import Model


class SFModel(Model):

    def __init__(self, net, optimizer, max_grad_norm=0.5, value_coef=0.5, entropy_coef=0.01):
        super(SFModel, self).__init__()
        self.net = net
        self.optimizer = optimizer
        self.max_grad_norm = max_grad_norm
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

    def update(self, losses):
        self.optimizer.zero_grad()

        # if not all 'losses' are specified (policy, value and entropy) 0's are added
        while len(losses) < 3:
            losses.append(0)

        (losses[0] + losses[1]*self.value_coef - losses[2]*self.entropy_coef).backward()

        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm(self.net.parameters(), self.max_grad_norm)

        self.optimizer.step()


class BaselineModel:

    def __init__(self, net, optimizer, max_grad_norm=None, use_cuda=torch.cuda.is_available()):
        super(BaselineModel, self).__init__()
        self.net = net
        self.optimizer = optimizer
        self.max_grad_norm = max_grad_norm
        self.use_cuda = use_cuda

    def update(self, loss):
        self.optimizer.zero_grad()

        loss.backward()

        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm(self.net.parameters(), self.max_grad_norm)

        self.optimizer.step()

    def predict_action(self, state):
        spec, sheet = state

        spec = torch.from_numpy(spec).float()
        sheet = torch.from_numpy(sheet).float()

        if self.use_cuda:
            spec = spec.cuda()
            sheet = sheet.cuda()

        action = self.net.forward([Variable(spec.unsqueeze(0)), Variable(sheet.unsqueeze(0))])

        return action

    def perform_action(self, state):
        return self.predict_action(state).data.cpu().numpy()[0]
