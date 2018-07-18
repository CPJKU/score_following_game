
import numpy as np

import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.distributions import Normal


class Agent(object):

    def perform_action(self, state):
        pass


class RandomAgent(Agent):

    def __init__(self, ac_space):
        self.ac_space = ac_space

    def perform_action(self, state):
        return np.random.choice(self.ac_space.n)


class TrainedAgent(Agent):

    def __init__(self, net, use_cuda=torch.cuda.is_available()):

        self.net = net
        self.use_cuda = use_cuda

        if self.use_cuda:
            self.net.cuda()

    def perform_action(self, state):

        observations = []
        for i, obs in enumerate(state):
            observations.append(torch.from_numpy(obs).float().unsqueeze(0))

        if self.use_cuda:
            observations = [obs.cuda() for obs in observations]

        policy, _ = self.net([Variable(obs) for obs in observations])

        probabilities = F.softmax(policy, dim=-1)

        return probabilities.multinomial().data[0].cpu().numpy()[0]


class TrainedContinuousAgent(Agent):

    def __init__(self, net, distr=Normal, use_cuda=torch.cuda.is_available()):

        self.net = net
        self.use_cuda = use_cuda

        if self.use_cuda:
            self.net.cuda()

        self.distribution = distr

    def perform_action(self, state):

        spec, sheet = state

        spec = torch.from_numpy(spec).float()
        sheet = torch.from_numpy(sheet).float()

        if self.use_cuda:
            spec = spec.cuda()
            sheet = sheet.cuda()

        policy, _ = self.net([Variable(spec.unsqueeze(0)), Variable(sheet.unsqueeze(0))])

        # transform variables to tensors policy has to be a dictionary
        distribution_params = {key: policy[key].data for key in policy}

        # create a distribution from the parameters
        distr = self.distribution(**distribution_params)

        # draw a random action by sampling from the policy prediction
        actions = distr.sample()

        # expand as continuous environments need each action inside of an array
        return actions.view(-1).cpu().numpy()


class OptimalAgent(Agent):

    def __init__(self, rl_pool):
        super(OptimalAgent, self).__init__()
        self.rl_pool = rl_pool
        self.optimal_actions = []

    def set_optimal_actions(self):

        song_onsets = np.asarray(self.rl_pool.onsets[self.rl_pool.sheet_id], dtype=np.int32)
        song_spec = self.rl_pool.spectrograms[self.rl_pool.sheet_id]
        interpol_fcn = self.rl_pool.interpolationFunctions[self.rl_pool.sheet_id]

        interpolated_coords = interpol_fcn(range(song_onsets[0], song_onsets[-1]))

        current_idx = 0
        dist = []
        for i in range(1, len(interpolated_coords)):

            if interpolated_coords[i - 1] == song_onsets[current_idx]:
                current_idx += 1

            dist = np.append(dist, (interpolated_coords[i] - interpolated_coords[i - 1]))

        self.optimal_actions = np.concatenate(
            (np.zeros(int(song_onsets[0] + 1)), dist, np.zeros(int(song_spec.shape[1] - song_onsets[-1]))))

    def perform_action(self, state):
        super(OptimalAgent, self).perform_action(state)

        # reward, timestep, current_speed = state
        current_speed = self.rl_pool.sheet_speed

        timestep = self.rl_pool.curr_spec_frame+1

        if timestep < len(self.optimal_actions):
            optimal_action = self.optimal_actions[timestep] - current_speed
        else:
            # set speed to 0 if the last known action is reached
            optimal_action = - current_speed
        return [optimal_action]


class TrainedLSTMAgent(Agent):

    def __init__(self, net, use_cuda=torch.cuda.is_available()):

        self.net = net
        self.use_cuda = use_cuda

        # initialize lstm hidden state
        n_worker = 1
        self.cell_states = torch.zeros(self.net.lstm_layers, n_worker, self.net.lstm_size)
        self.hidden_states = torch.zeros(self.net.lstm_layers, n_worker, self.net.lstm_size)

        if self.use_cuda:
            self.net.cuda()
            self.cell_states = self.cell_states.cuda()
            self.hidden_states = self.hidden_states.cuda()

    def perform_action(self, state):

        observations = []
        for i, obs in enumerate(state):
            observations.append(torch.from_numpy(obs).float().unsqueeze(0))

        if self.use_cuda:
            observations = [obs.cuda() for obs in observations]

        net_input = []
        for obs in observations:
            net_input.append((Variable(obs), (Variable(self.hidden_states), Variable(self.cell_states))))

        policy, _, (hidden_states, cell_states) = self.net(net_input)

        # update lstm hidden state
        self.hidden_states = hidden_states.data
        self.cell_states = cell_states.data

        probabilities = F.softmax(policy, dim=-1)

        return probabilities.multinomial().data[0].cpu().numpy()[0]

    def predict_value(self, state):

        observations = []
        for i, obs in enumerate(state):
            observations.append(torch.from_numpy(obs).float().unsqueeze(0))

        if self.use_cuda:
            observations = [obs.cuda() for obs in observations]

        net_input = []
        for obs in observations:
            net_input.append((Variable(obs), (Variable(self.hidden_states), Variable(self.cell_states))))

        _, value, _ = self.net(net_input)

        return value.data.cpu().numpy()[0, 0]


class TrainedLSTMContinuousAgent(Agent):

    def __init__(self, net, distr=Normal, use_cuda=torch.cuda.is_available()):

        self.net = net
        self.use_cuda = use_cuda
        self.distribution = distr

        # initialize lstm hidden state
        n_worker = 1
        self.cell_states = torch.zeros(self.net.lstm_layers, n_worker, self.net.lstm_size)
        self.hidden_states = torch.zeros(self.net.lstm_layers, n_worker, self.net.lstm_size)

        if self.use_cuda:
            self.net.cuda()
            self.cell_states = self.cell_states.cuda()
            self.hidden_states = self.hidden_states.cuda()

    def perform_action(self, state):

        spec, sheet = state

        spec = torch.from_numpy(spec).float()
        sheet = torch.from_numpy(sheet).float()

        if self.use_cuda:
            spec = spec.cuda()
            sheet = sheet.cuda()

        # policy, _ = self.net([Variable(spec.unsqueeze(0)), Variable(sheet.unsqueeze(0))])
        input = []
        for obs in [spec.unsqueeze(0), sheet.unsqueeze(0)]:
            input.append((Variable(obs), (Variable(self.hidden_states), Variable(self.cell_states))))

        policy, value, (hidden_states, cell_states) = self.net(input)

        # update lstm hidden state
        self.hidden_states = hidden_states.data
        self.cell_states = cell_states.data

        # transform variables to tensors policy has to be a dictionary
        distribution_params = {key: policy[key].data for key in policy}

        # create a distribution from the parameters
        distr = self.distribution(**distribution_params)

        # draw a random action by sampling from the policy prediction
        actions = distr.sample()

        # expand as continuous environments need each action inside of an array
        return actions.view(-1).cpu().numpy()
