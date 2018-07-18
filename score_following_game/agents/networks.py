
import importlib
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_network(network_name, n_actions, spec_channels, sheet_channels):
    """
    Compile network by name

    :param network_name:
    :return:
    """
    package = importlib.import_module("score_following_game.agents.networks")
    constructor = getattr(package, network_name)
    network = constructor(n_actions, spec_channels, sheet_channels)
    return network


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        nn.init.orthogonal(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)


def weights_init_selu(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        f_in = m.weight.numel()
        sdev = np.sqrt(1. / f_in)
        nn.init.normal(m.weight.data, 0.0, sdev)
        if m.bias is not None:
            m.bias.data.fill_(0)


def num_flat_features(x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


class ScoreFollowingNet(nn.Module):
    def __init__(self,  n_actions, spec_channels, sheet_channels):
        super(ScoreFollowingNet, self).__init__()

        self.spec_conv1 = nn.Conv2d(spec_channels, out_channels=16, kernel_size=(4, 4), stride=2)
        self.spec_conv2 = nn.Conv2d(self.spec_conv1.out_channels, out_channels=32, kernel_size=(3, 3), stride=2)
        self.spec_conv3 = nn.Conv2d(self.spec_conv2.out_channels, out_channels=64, kernel_size=(3, 3), stride=1)

        self.sheet_conv1 = nn.Conv2d(sheet_channels, out_channels=16, kernel_size=(4, 8), stride=2)
        self.sheet_conv2 = nn.Conv2d(self.sheet_conv1.out_channels, out_channels=32, kernel_size=(3, 3), stride=(1, 2))
        self.sheet_conv3 = nn.Conv2d(self.sheet_conv2.out_channels, out_channels=32, kernel_size=(3, 3), stride=(1, 2))
        self.sheet_conv4 = nn.Conv2d(self.sheet_conv3.out_channels, out_channels=32, kernel_size=(4, 4), stride=2)

        self.concat_fc = nn.Linear(self.spec_conv3.out_channels * 6 * 2 + self.sheet_conv4.out_channels * 6 * 7, 256)

        self.policy_fc = nn.Linear(self.concat_fc.out_features, 256)
        self.policy_out = nn.Linear(self.policy_fc.out_features, n_actions)

        self.critic_fc = nn.Linear(self.concat_fc.out_features, 256)
        self.critic_out = nn.Linear(self.critic_fc.out_features, 1)

        self.apply(weights_init)

        # scaling for orthogonal initialization
        # self.conv1.weight.data.mul_(math.sqrt(2))
        # self.conv2.weight.data.mul_(math.sqrt(2))
        # self.conv3.weight.data.mul_(math.sqrt(2))
        # self.fc1.weight.data.mul_(math.sqrt(2))

    def forward(self, x):

        spec_x = F.elu(self.spec_conv1(x[0]))
        spec_x = F.elu(self.spec_conv2(spec_x))
        spec_x = F.elu(self.spec_conv3(spec_x))

        spec_x = spec_x.view(-1, num_flat_features(spec_x)) # flatten

        sheet_x = F.elu(self.sheet_conv1(x[1]))
        sheet_x = F.elu(self.sheet_conv2(sheet_x))
        sheet_x = F.elu(self.sheet_conv3(sheet_x))
        sheet_x = F.elu(self.sheet_conv4(sheet_x))

        sheet_x = sheet_x.view(-1, num_flat_features(sheet_x)) # flatten

        cat_x = torch.cat((spec_x, sheet_x), dim=1)

        cat_x = F.elu(self.concat_fc(cat_x))

        policy = F.elu(self.policy_fc(cat_x))
        policy = self.policy_out(policy)

        critic = F.elu(self.critic_fc(cat_x))
        critic = self.critic_out(critic)

        # if we use softmax and the logarithmic softmax PyTorch recommends to use the log_softmax function
        # therefore I return the linear output of the policy and apply the softmax/log softmax where it is necessary
        return policy, critic


class ScoreFollowingNetDo(nn.Module):
    def __init__(self,  n_actions, spec_channels, sheet_channels):
        super(ScoreFollowingNetDo, self).__init__()

        self.spec_conv1 = nn.Conv2d(spec_channels, out_channels=16, kernel_size=(4, 4), stride=2)
        self.spec_conv2 = nn.Conv2d(self.spec_conv1.out_channels, out_channels=32, kernel_size=(3, 3), stride=2)
        self.spec_conv3 = nn.Conv2d(self.spec_conv2.out_channels, out_channels=64, kernel_size=(3, 3), stride=1)
        self.spec_do3 = nn.Dropout(p=0.2)

        self.sheet_conv1 = nn.Conv2d(sheet_channels, out_channels=16, kernel_size=(4, 8), stride=2)
        self.sheet_conv2 = nn.Conv2d(self.sheet_conv1.out_channels, out_channels=32, kernel_size=(3, 3), stride=(1, 2))
        self.sheet_conv3 = nn.Conv2d(self.sheet_conv2.out_channels, out_channels=32, kernel_size=(3, 3), stride=(1, 2))
        self.sheet_conv4 = nn.Conv2d(self.sheet_conv3.out_channels, out_channels=32, kernel_size=(4, 4), stride=2)
        self.sheet_do4 = nn.Dropout(p=0.2)

        self.concat_fc = nn.Linear(self.spec_conv3.out_channels * 6 * 2 + self.sheet_conv4.out_channels * 6 * 7, 256)
        self.concat_do = nn.Dropout(p=0.2)

        self.policy_fc = nn.Linear(self.concat_fc.out_features, 256)
        self.policy_do = nn.Dropout(p=0.2)
        self.policy_out = nn.Linear(self.policy_fc.out_features, n_actions)

        self.critic_fc = nn.Linear(self.concat_fc.out_features, 256)
        self.critic_do = nn.Dropout(p=0.2)
        self.critic_out = nn.Linear(self.critic_fc.out_features, 1)

        self.apply(weights_init)

        # scaling for orthogonal initialization
        # self.conv1.weight.data.mul_(math.sqrt(2))
        # self.conv2.weight.data.mul_(math.sqrt(2))
        # self.conv3.weight.data.mul_(math.sqrt(2))
        # self.fc1.weight.data.mul_(math.sqrt(2))

    def forward(self, x):

        spec_x = F.elu(self.spec_conv1(x[0]))
        spec_x = F.elu(self.spec_conv2(spec_x))
        spec_x = F.elu(self.spec_conv3(spec_x))
        spec_x = self.spec_do3(spec_x)

        spec_x = spec_x.view(-1, num_flat_features(spec_x)) # flatten

        sheet_x = F.elu(self.sheet_conv1(x[1]))
        sheet_x = F.elu(self.sheet_conv2(sheet_x))
        sheet_x = F.elu(self.sheet_conv3(sheet_x))
        sheet_x = F.elu(self.sheet_conv4(sheet_x))
        sheet_x = self.sheet_do4(sheet_x)

        sheet_x = sheet_x.view(-1, num_flat_features(sheet_x)) # flatten

        cat_x = torch.cat((spec_x, sheet_x), dim=1)

        cat_x = F.elu(self.concat_fc(cat_x))
        cat_x = self.concat_do(cat_x)

        policy = F.elu(self.policy_fc(cat_x))
        policy = self.policy_do(policy)
        policy = self.policy_out(policy)

        critic = F.elu(self.critic_fc(cat_x))
        critic = self.critic_do(critic)
        critic = self.critic_out(critic)

        # if we use softmax and the logarithmic softmax PyTorch recommends to use the log_softmax function
        # therefore I return the linear output of the policy and apply the softmax/log softmax where it is necessary
        return policy, critic


class ScoreFollowingNetBN(nn.Module):
    def __init__(self,  n_actions, spec_channels, sheet_channels):
        super(ScoreFollowingNetBN, self).__init__()

        self.spec_conv1 = nn.Conv2d(spec_channels, out_channels=16, kernel_size=(4, 4), stride=2)
        self.spec_conv1_bn = nn.BatchNorm2d(self.spec_conv1.out_channels)
        self.spec_conv2 = nn.Conv2d(self.spec_conv1.out_channels, out_channels=32, kernel_size=(3, 3), stride=2)
        self.spec_conv2_bn = nn.BatchNorm2d(self.spec_conv2.out_channels)
        self.spec_conv3 = nn.Conv2d(self.spec_conv2.out_channels, out_channels=64, kernel_size=(3, 3), stride=1)
        self.spec_conv3_bn = nn.BatchNorm2d(self.spec_conv3.out_channels)

        self.sheet_conv1 = nn.Conv2d(sheet_channels, out_channels=16, kernel_size=(4, 8), stride=2)
        self.sheet_conv1_bn = nn.BatchNorm2d(self.sheet_conv1.out_channels)
        self.sheet_conv2 = nn.Conv2d(self.sheet_conv1.out_channels, out_channels=32, kernel_size=(3, 3), stride=(1, 2))
        self.sheet_conv2_bn = nn.BatchNorm2d(self.sheet_conv2.out_channels)
        self.sheet_conv3 = nn.Conv2d(self.sheet_conv2.out_channels, out_channels=32, kernel_size=(3, 3), stride=(1, 2))
        self.sheet_conv3_bn = nn.BatchNorm2d(self.sheet_conv3.out_channels)
        self.sheet_conv4 = nn.Conv2d(self.sheet_conv3.out_channels, out_channels=32, kernel_size=(4, 4), stride=2)
        self.sheet_conv4_bn = nn.BatchNorm2d(self.sheet_conv4.out_channels)

        self.concat_fc = nn.Linear(self.spec_conv3.out_channels * 6 * 2 + self.sheet_conv4.out_channels * 6 * 7, 256)

        self.policy_fc = nn.Linear(self.concat_fc.out_features, 256)
        self.policy_fc_bn = nn.BatchNorm1d(256)
        self.policy_out = nn.Linear(self.policy_fc.out_features, n_actions)

        self.critic_fc = nn.Linear(self.concat_fc.out_features, 256)
        self.critic_fc_bn = nn.BatchNorm1d(256)
        self.critic_out = nn.Linear(self.critic_fc.out_features, 1)

        self.apply(weights_init)

        # scaling for orthogonal initialization
        # self.conv1.weight.data.mul_(math.sqrt(2))
        # self.conv2.weight.data.mul_(math.sqrt(2))
        # self.conv3.weight.data.mul_(math.sqrt(2))
        # self.fc1.weight.data.mul_(math.sqrt(2))

    def forward(self, x):

        spec_x = F.elu(self.spec_conv1_bn(self.spec_conv1(x[0])))
        spec_x = F.elu(self.spec_conv2_bn(self.spec_conv2(spec_x)))
        spec_x = F.elu(self.spec_conv3_bn(self.spec_conv3(spec_x)))

        spec_x = spec_x.view(-1, num_flat_features(spec_x)) # flatten

        sheet_x = F.elu(self.sheet_conv1_bn(self.sheet_conv1(x[1])))
        sheet_x = F.elu(self.sheet_conv2_bn(self.sheet_conv2(sheet_x)))
        sheet_x = F.elu(self.sheet_conv3_bn(self.sheet_conv3(sheet_x)))
        sheet_x = F.elu(self.sheet_conv4_bn(self.sheet_conv4(sheet_x)))

        sheet_x = sheet_x.view(-1, num_flat_features(sheet_x)) # flatten

        cat_x = torch.cat((spec_x, sheet_x), dim=1)

        cat_x = F.elu(self.concat_fc(cat_x))

        policy = F.elu(self.policy_fc_bn(self.policy_fc(cat_x)))
        policy = self.policy_out(policy)

        critic = F.elu(self.critic_fc_bn(self.critic_fc(cat_x)))
        critic = self.critic_out(critic)

        # if we use softmax and the logarithmic softmax PyTorch recommends to use the log_softmax function
        # therefore I return the linear output of the policy and apply the softmax/log softmax where it is necessary
        return policy, critic


class ScoreFollowingNetSELU(nn.Module):
    def __init__(self,  n_actions, spec_channels, sheet_channels):
        super(ScoreFollowingNetSELU, self).__init__()

        self.spec_conv1 = nn.Conv2d(spec_channels, out_channels=16, kernel_size=(4, 4), stride=2)
        self.spec_conv2 = nn.Conv2d(self.spec_conv1.out_channels, out_channels=32, kernel_size=(3, 3), stride=2)
        self.spec_conv3 = nn.Conv2d(self.spec_conv2.out_channels, out_channels=64, kernel_size=(3, 3), stride=1)

        self.sheet_conv1 = nn.Conv2d(sheet_channels, out_channels=16, kernel_size=(4, 8), stride=2)
        self.sheet_conv2 = nn.Conv2d(self.sheet_conv1.out_channels, out_channels=32, kernel_size=(3, 3), stride=(1, 2))
        self.sheet_conv3 = nn.Conv2d(self.sheet_conv2.out_channels, out_channels=32, kernel_size=(3, 3), stride=(1, 2))
        self.sheet_conv4 = nn.Conv2d(self.sheet_conv3.out_channels, out_channels=32, kernel_size=(4, 4), stride=2)

        self.concat_fc = nn.Linear(self.spec_conv3.out_channels * 6 * 2 + self.sheet_conv4.out_channels * 6 * 7, 256)

        self.policy_fc = nn.Linear(self.concat_fc.out_features, 256)
        self.policy_out = nn.Linear(self.policy_fc.out_features, n_actions)

        self.critic_fc = nn.Linear(self.concat_fc.out_features, 256)
        self.critic_out = nn.Linear(self.critic_fc.out_features, 1)

        self.apply(weights_init_selu)

    def forward(self, x):

        spec_x = F.selu(self.spec_conv1(x[0]))
        spec_x = F.selu(self.spec_conv2(spec_x))
        spec_x = F.selu(self.spec_conv3(spec_x))

        spec_x = spec_x.view(-1, num_flat_features(spec_x)) # flatten

        sheet_x = F.selu(self.sheet_conv1(x[1]))
        sheet_x = F.selu(self.sheet_conv2(sheet_x))
        sheet_x = F.selu(self.sheet_conv3(sheet_x))
        sheet_x = F.selu(self.sheet_conv4(sheet_x))

        sheet_x = sheet_x.view(-1, num_flat_features(sheet_x)) # flatten

        cat_x = torch.cat((spec_x, sheet_x), dim=1)

        cat_x = F.selu(self.concat_fc(cat_x))

        policy = F.selu(self.policy_fc(cat_x))
        policy = self.policy_out(policy)

        critic = F.selu(self.critic_fc(cat_x))
        critic = self.critic_out(critic)

        # if we use softmax and the logarithmic softmax PyTorch recommends to use the log_softmax function
        # therefore I return the linear output of the policy and apply the softmax/log softmax where it is necessary
        return policy, critic


class ScoreFollowingNetSELU01(nn.Module):
    def __init__(self,  n_actions, spec_channels, sheet_channels):
        super(ScoreFollowingNetSELU01, self).__init__()

        # use batch norm layer for input standardization
        self.spec_bn = nn.BatchNorm2d(2, affine=False, momentum=0.01)
        self.sheet_bn = nn.BatchNorm2d(2, affine=False, momentum=0.01)

        self.spec_conv1 = nn.Conv2d(spec_channels, out_channels=16, kernel_size=(4, 4), stride=2)
        self.spec_conv2 = nn.Conv2d(self.spec_conv1.out_channels, out_channels=32, kernel_size=(3, 3), stride=2)
        self.spec_conv3 = nn.Conv2d(self.spec_conv2.out_channels, out_channels=64, kernel_size=(3, 3), stride=1)

        self.sheet_conv1 = nn.Conv2d(sheet_channels, out_channels=16, kernel_size=(4, 8), stride=2)
        self.sheet_conv2 = nn.Conv2d(self.sheet_conv1.out_channels, out_channels=32, kernel_size=(3, 3), stride=(1, 2))
        self.sheet_conv3 = nn.Conv2d(self.sheet_conv2.out_channels, out_channels=32, kernel_size=(3, 3), stride=(1, 2))
        self.sheet_conv4 = nn.Conv2d(self.sheet_conv3.out_channels, out_channels=32, kernel_size=(4, 4), stride=2)

        self.concat_fc = nn.Linear(self.spec_conv3.out_channels * 6 * 2 + self.sheet_conv4.out_channels * 6 * 7, 256)

        self.policy_fc = nn.Linear(self.concat_fc.out_features, 256)
        self.policy_out = nn.Linear(self.policy_fc.out_features, n_actions)

        self.critic_fc = nn.Linear(self.concat_fc.out_features, 256)
        self.critic_out = nn.Linear(self.critic_fc.out_features, 1)

        self.apply(weights_init_selu)

    def forward(self, x):

        # normalize data
        x[0] = self.spec_bn(x[0])
        x[1] = self.sheet_bn(x[1])

        spec_x = F.selu(self.spec_conv1(x[0]))
        spec_x = F.selu(self.spec_conv2(spec_x))
        spec_x = F.selu(self.spec_conv3(spec_x))

        spec_x = spec_x.view(-1, num_flat_features(spec_x)) # flatten

        sheet_x = F.selu(self.sheet_conv1(x[1]))
        sheet_x = F.selu(self.sheet_conv2(sheet_x))
        sheet_x = F.selu(self.sheet_conv3(sheet_x))
        sheet_x = F.selu(self.sheet_conv4(sheet_x))

        sheet_x = sheet_x.view(-1, num_flat_features(sheet_x)) # flatten

        cat_x = torch.cat((spec_x, sheet_x), dim=1)

        cat_x = F.selu(self.concat_fc(cat_x))

        policy = F.selu(self.policy_fc(cat_x))
        policy = self.policy_out(policy)

        critic = F.selu(self.critic_fc(cat_x))
        critic = self.critic_out(critic)

        # if we use softmax and the logarithmic softmax PyTorch recommends to use the log_softmax function
        # therefore I return the linear output of the policy and apply the softmax/log softmax where it is necessary
        return policy, critic


class ScoreFollowingNetLSTM(nn.Module):
    def __init__(self,  n_actions, spec_channels, sheet_channels):
        super(ScoreFollowingNetLSTM, self).__init__()

        self.spec_conv1 = nn.Conv2d(spec_channels, out_channels=16, kernel_size=(4, 4), stride=2)
        self.spec_conv2 = nn.Conv2d(self.spec_conv1.out_channels, out_channels=32, kernel_size=(3, 3), stride=2)
        self.spec_conv3 = nn.Conv2d(self.spec_conv2.out_channels, out_channels=64, kernel_size=(3, 3), stride=1)

        self.sheet_conv1 = nn.Conv2d(sheet_channels, out_channels=16, kernel_size=(4, 8), stride=2)
        self.sheet_conv2 = nn.Conv2d(self.sheet_conv1.out_channels, out_channels=32, kernel_size=(3, 3), stride=(1, 2))
        self.sheet_conv3 = nn.Conv2d(self.sheet_conv2.out_channels, out_channels=32, kernel_size=(3, 3), stride=(1, 2))
        self.sheet_conv4 = nn.Conv2d(self.sheet_conv3.out_channels, out_channels=32, kernel_size=(4, 4), stride=2)

        self.concat_fc = nn.Linear(self.spec_conv3.out_channels * 6 * 2 + self.sheet_conv4.out_channels * 6 * 7, 256)

        self.lstm_size = 256
        self.lstm_layers = 1
        self.lstm = nn.LSTM(256, self.lstm_size, num_layers=self.lstm_layers)

        self.policy_fc = nn.Linear(self.concat_fc.out_features, 256)
        self.policy_out = nn.Linear(self.policy_fc.out_features, n_actions)

        self.critic_fc = nn.Linear(self.concat_fc.out_features, 256)
        self.critic_out = nn.Linear(self.critic_fc.out_features, 1)

        self.apply(weights_init)

    def forward(self, x):

        spec_x, (hidden_state, cell_state) = x[0]
        sheet_x, _ = x[1]

        spec_x = F.elu(self.spec_conv1(spec_x))
        spec_x = F.elu(self.spec_conv2(spec_x))
        spec_x = F.elu(self.spec_conv3(spec_x))

        spec_x = spec_x.view(-1, num_flat_features(spec_x)) # flatten

        sheet_x = F.elu(self.sheet_conv1(sheet_x))
        sheet_x = F.elu(self.sheet_conv2(sheet_x))
        sheet_x = F.elu(self.sheet_conv3(sheet_x))
        sheet_x = F.elu(self.sheet_conv4(sheet_x))

        sheet_x = sheet_x.view(-1, num_flat_features(sheet_x)) # flatten

        cat_x = torch.cat((spec_x, sheet_x), dim=1)

        cat_x = F.elu(self.concat_fc(cat_x))

        cat_x, (hidden_state_n, cell_state_n) = self.lstm(cat_x.unsqueeze(0), (hidden_state, cell_state))

        # only one time step
        cat_x = cat_x[0]

        policy = F.elu(self.policy_fc(cat_x))
        policy = self.policy_out(policy)

        critic = F.elu(self.critic_fc(cat_x))
        critic = self.critic_out(critic)

        # if we use softmax and the logarithmic softmax PyTorch recommends to use the log_softmax function
        # therefore I return the linear output of the policy and apply the softmax/log softmax where it is necessary
        return policy, critic, (hidden_state_n, cell_state_n)


class ScoreFollowingNetContinuous(nn.Module):
    def __init__(self, n_actions, spec_channels, sheet_channels):
        super(ScoreFollowingNetContinuous, self).__init__()

        self.spec_conv1 = nn.Conv2d(spec_channels, out_channels=16, kernel_size=(4, 4), stride=2)
        self.spec_conv2 = nn.Conv2d(self.spec_conv1.out_channels, out_channels=32, kernel_size=(3, 3), stride=2)
        self.spec_conv3 = nn.Conv2d(self.spec_conv2.out_channels, out_channels=64, kernel_size=(3, 3), stride=1)

        self.sheet_conv1 = nn.Conv2d(sheet_channels, out_channels=16, kernel_size=(4, 8), stride=2)
        self.sheet_conv2 = nn.Conv2d(self.sheet_conv1.out_channels, out_channels=32, kernel_size=(3, 3),
                                     stride=(1, 2))
        self.sheet_conv3 = nn.Conv2d(self.sheet_conv2.out_channels, out_channels=32, kernel_size=(3, 3),
                                     stride=(1, 2))
        self.sheet_conv4 = nn.Conv2d(self.sheet_conv2.out_channels, out_channels=32, kernel_size=(4, 4), stride=2)

        self.concat_fc = nn.Linear(self.spec_conv3.out_channels * 6 * 2 + self.sheet_conv4.out_channels * 6 * 7,
                                   256)

        self.mu_fc = nn.Linear(self.concat_fc.out_features, 256)
        self.mu_out = nn.Linear(self.mu_fc.out_features, 1)

        self.sigma_fc = nn.Linear(self.concat_fc.out_features, 256)
        self.sigma_out = nn.Linear(self.sigma_fc.out_features, 1)

        self.critic_fc = nn.Linear(self.concat_fc.out_features, 256)
        self.critic_out = nn.Linear(self.critic_fc.out_features, 1)

        self.apply(weights_init)

        # scaling for orthogonal initialization
        # self.conv1.weight.data.mul_(math.sqrt(2))
        # self.conv2.weight.data.mul_(math.sqrt(2))
        # self.conv3.weight.data.mul_(math.sqrt(2))
        # self.fc1.weight.data.mul_(math.sqrt(2))

    def forward(self, x):
        spec_x = F.elu(self.spec_conv1(x[0]))
        spec_x = F.elu(self.spec_conv2(spec_x))
        spec_x = F.elu(self.spec_conv3(spec_x))

        spec_x = spec_x.view(-1, num_flat_features(spec_x))  # flatten

        sheet_x = F.elu(self.sheet_conv1(x[1]))
        sheet_x = F.elu(self.sheet_conv2(sheet_x))
        sheet_x = F.elu(self.sheet_conv3(sheet_x))
        sheet_x = F.elu(self.sheet_conv4(sheet_x))

        sheet_x = sheet_x.view(-1, num_flat_features(sheet_x))  # flatten

        cat_x = torch.cat((spec_x, sheet_x), dim=1)

        cat_x = F.elu(self.concat_fc(cat_x))

        critic = F.elu(self.critic_fc(cat_x))
        critic = self.critic_out(critic)

        mu = F.relu(self.mu_fc(cat_x))
        sigma = F.relu(self.sigma_fc(cat_x))

        mu = self.mu_out(mu)
        sigma = torch.exp(self.sigma_out(sigma)) + 1e-3

        return {'mean': mu, 'std': sigma}, critic


class ScoreFollowingNetLSTMContinuous(nn.Module):
    def __init__(self, n_actions, spec_channels, sheet_channels):
        super(ScoreFollowingNetLSTMContinuous, self).__init__()

        self.spec_conv1 = nn.Conv2d(spec_channels, out_channels=16, kernel_size=(4, 4), stride=2)
        self.spec_conv2 = nn.Conv2d(self.spec_conv1.out_channels, out_channels=32, kernel_size=(3, 3), stride=2)
        self.spec_conv3 = nn.Conv2d(self.spec_conv2.out_channels, out_channels=64, kernel_size=(3, 3), stride=1)

        self.sheet_conv1 = nn.Conv2d(sheet_channels, out_channels=16, kernel_size=(4, 8), stride=2)
        self.sheet_conv2 = nn.Conv2d(self.sheet_conv1.out_channels, out_channels=32, kernel_size=(3, 3),
                                     stride=(1, 2))
        self.sheet_conv3 = nn.Conv2d(self.sheet_conv2.out_channels, out_channels=32, kernel_size=(3, 3),
                                     stride=(1, 2))
        self.sheet_conv4 = nn.Conv2d(self.sheet_conv2.out_channels, out_channels=32, kernel_size=(4, 4), stride=2)

        self.concat_fc = nn.Linear(self.spec_conv3.out_channels * 6 * 2 + self.sheet_conv4.out_channels * 6 * 7, 256)

        self.lstm_size = 256
        self.lstm_layers = 1
        self.lstm = nn.LSTM(256, self.lstm_size, num_layers=self.lstm_layers)

        self.mu_fc = nn.Linear(self.concat_fc.out_features, 256)
        self.mu_out = nn.Linear(self.mu_fc.out_features, 1)

        self.sigma_fc = nn.Linear(self.concat_fc.out_features, 256)
        self.sigma_out = nn.Linear(self.sigma_fc.out_features, 1)

        self.critic_fc = nn.Linear(self.concat_fc.out_features, 256)
        self.critic_out = nn.Linear(self.critic_fc.out_features, 1)

        self.apply(weights_init)

    def forward(self, x):
        spec_x, (hidden_state, cell_state) = x[0]
        sheet_x, _ = x[1]

        spec_x = F.elu(self.spec_conv1(spec_x))
        spec_x = F.elu(self.spec_conv2(spec_x))
        spec_x = F.elu(self.spec_conv3(spec_x))

        spec_x = spec_x.view(-1, num_flat_features(spec_x))  # flatten

        sheet_x = F.elu(self.sheet_conv1(sheet_x))
        sheet_x = F.elu(self.sheet_conv2(sheet_x))
        sheet_x = F.elu(self.sheet_conv3(sheet_x))
        sheet_x = F.elu(self.sheet_conv4(sheet_x))

        sheet_x = sheet_x.view(-1, num_flat_features(sheet_x))  # flatten

        cat_x = torch.cat((spec_x, sheet_x), dim=1)

        cat_x = F.elu(self.concat_fc(cat_x))

        cat_x, (hidden_state_n, cell_state_n) = self.lstm(cat_x.unsqueeze(0), (hidden_state, cell_state))

        # only one time step
        cat_x = cat_x[0]

        critic = F.elu(self.critic_fc(cat_x))
        critic = self.critic_out(critic)

        mu = F.relu(self.mu_fc(cat_x))
        sigma = F.relu(self.sigma_fc(cat_x))

        mu = self.mu_out(mu)
        sigma = torch.exp(self.sigma_out(sigma)) + 1e-3

        return {'mean': mu, 'std': sigma}, critic, (hidden_state_n, cell_state_n)


class ScoreFollowingNetMSMD(nn.Module):
    def __init__(self,  n_actions, spec_channels, sheet_channels):
        super(ScoreFollowingNetMSMD, self).__init__()

        self.spec_conv1 = nn.Conv2d(spec_channels, out_channels=16, kernel_size=(4, 4), stride=2)
        self.spec_conv2 = nn.Conv2d(self.spec_conv1.out_channels, out_channels=32, kernel_size=(3, 3), stride=2)
        self.spec_conv3 = nn.Conv2d(self.spec_conv2.out_channels, out_channels=64, kernel_size=(3, 3), stride=1)

        self.sheet_conv1 = nn.Conv2d(sheet_channels, out_channels=16, kernel_size=(4, 8), stride=2)
        self.sheet_conv2 = nn.Conv2d(self.sheet_conv1.out_channels, out_channels=32, kernel_size=(3, 3), stride=(1, 2))
        self.sheet_conv3 = nn.Conv2d(self.sheet_conv2.out_channels, out_channels=32, kernel_size=(3, 3), stride=(1, 2))
        self.sheet_conv4 = nn.Conv2d(self.sheet_conv3.out_channels, out_channels=32, kernel_size=(4, 4), stride=2)

        self.concat_fc = nn.Linear(768 + 2304, 256)

        self.policy_fc = nn.Linear(self.concat_fc.out_features, 256)
        self.policy_out = nn.Linear(self.policy_fc.out_features, n_actions)

        self.critic_fc = nn.Linear(self.concat_fc.out_features, 256)
        self.critic_out = nn.Linear(self.critic_fc.out_features, 1)

        self.apply(weights_init)

        # scaling for orthogonal initialization
        # self.conv1.weight.data.mul_(math.sqrt(2))
        # self.conv2.weight.data.mul_(math.sqrt(2))
        # self.conv3.weight.data.mul_(math.sqrt(2))
        # self.fc1.weight.data.mul_(math.sqrt(2))

    def forward(self, x):

        spec_x = F.elu(self.spec_conv1(x[0]))
        spec_x = F.elu(self.spec_conv2(spec_x))
        spec_x = F.elu(self.spec_conv3(spec_x))

        spec_x = spec_x.view(-1, num_flat_features(spec_x)) # flatten

        sheet_x = F.elu(self.sheet_conv1(x[1]))
        sheet_x = F.elu(self.sheet_conv2(sheet_x))
        sheet_x = F.elu(self.sheet_conv3(sheet_x))
        sheet_x = F.elu(self.sheet_conv4(sheet_x))

        sheet_x = sheet_x.view(-1, num_flat_features(sheet_x)) # flatten

        cat_x = torch.cat((spec_x, sheet_x), dim=1)

        cat_x = F.elu(self.concat_fc(cat_x))

        policy = F.elu(self.policy_fc(cat_x))
        policy = self.policy_out(policy)

        critic = F.elu(self.critic_fc(cat_x))
        critic = self.critic_out(critic)

        # if we use softmax and the logarithmic softmax PyTorch recommends to use the log_softmax function
        # therefore I return the linear output of the policy and apply the softmax/log softmax where it is necessary
        return policy, critic


class ScoreFollowingNetMSMDLC(nn.Module):
    def __init__(self,  n_actions, spec_channels, sheet_channels):
        super(ScoreFollowingNetMSMDLC, self).__init__()

        self.spec_conv1 = nn.Conv2d(spec_channels, out_channels=16, kernel_size=(4, 4), stride=2)
        self.spec_conv2 = nn.Conv2d(self.spec_conv1.out_channels, out_channels=32, kernel_size=(3, 3), stride=2)
        self.spec_conv3 = nn.Conv2d(self.spec_conv2.out_channels, out_channels=64, kernel_size=(3, 3), stride=1)

        self.sheet_conv1 = nn.Conv2d(sheet_channels, out_channels=16, kernel_size=(4, 8), stride=2)
        self.sheet_conv2 = nn.Conv2d(self.sheet_conv1.out_channels, out_channels=32, kernel_size=(3, 3), stride=(1, 2))
        self.sheet_conv3 = nn.Conv2d(self.sheet_conv2.out_channels, out_channels=32, kernel_size=(3, 3), stride=(1, 2))
        self.sheet_conv4 = nn.Conv2d(self.sheet_conv3.out_channels, out_channels=32, kernel_size=(4, 4), stride=2)
        self.sheet_conv5 = nn.Conv2d(self.sheet_conv4.out_channels, out_channels=32, kernel_size=(4, 4), stride=2)

        self.concat_fc = nn.Linear(768 + 1152, 256)

        self.policy_fc = nn.Linear(self.concat_fc.out_features, 256)
        self.policy_out = nn.Linear(self.policy_fc.out_features, n_actions)

        self.critic_fc = nn.Linear(self.concat_fc.out_features, 256)
        self.critic_out = nn.Linear(self.critic_fc.out_features, 1)

        self.apply(weights_init)

        # scaling for orthogonal initialization
        # self.conv1.weight.data.mul_(math.sqrt(2))
        # self.conv2.weight.data.mul_(math.sqrt(2))
        # self.conv3.weight.data.mul_(math.sqrt(2))
        # self.fc1.weight.data.mul_(math.sqrt(2))

    def forward(self, x):

        spec_x = F.elu(self.spec_conv1(x[0]))
        spec_x = F.elu(self.spec_conv2(spec_x))
        spec_x = F.elu(self.spec_conv3(spec_x))

        spec_x = spec_x.view(-1, num_flat_features(spec_x))  # flatten

        sheet_x = F.elu(self.sheet_conv1(x[1]))
        sheet_x = F.elu(self.sheet_conv2(sheet_x))
        sheet_x = F.elu(self.sheet_conv3(sheet_x))
        sheet_x = F.elu(self.sheet_conv4(sheet_x))
        sheet_x = F.elu(self.sheet_conv5(sheet_x))

        sheet_x = sheet_x.view(-1, num_flat_features(sheet_x))  # flatten

        # print(sheet_x.shape)

        cat_x = torch.cat((spec_x, sheet_x), dim=1)

        cat_x = F.elu(self.concat_fc(cat_x))

        policy = F.elu(self.policy_fc(cat_x))
        policy = self.policy_out(policy)

        critic = F.elu(self.critic_fc(cat_x))
        critic = self.critic_out(critic)

        # if we use softmax and the logarithmic softmax PyTorch recommends to use the log_softmax function
        # therefore I return the linear output of the policy and apply the softmax/log softmax where it is necessary
        return policy, critic


class ScoreFollowingNetMSMDLCH(nn.Module):
    def __init__(self,  n_actions, spec_channels, sheet_channels):
        super(ScoreFollowingNetMSMDLCH, self).__init__()

        self.spec_conv1 = nn.Conv2d(spec_channels, out_channels=16, kernel_size=(4, 4), stride=2)
        self.spec_conv2 = nn.Conv2d(self.spec_conv1.out_channels, out_channels=32, kernel_size=(3, 3), stride=2)
        self.spec_conv3 = nn.Conv2d(self.spec_conv2.out_channels, out_channels=64, kernel_size=(3, 3), stride=1)

        self.sheet_conv1 = nn.Conv2d(sheet_channels, out_channels=16, kernel_size=(4, 8), stride=2)
        self.sheet_conv2 = nn.Conv2d(self.sheet_conv1.out_channels, out_channels=32, kernel_size=(3, 3), stride=(1, 2))
        self.sheet_conv3 = nn.Conv2d(self.sheet_conv2.out_channels, out_channels=32, kernel_size=(3, 3), stride=(1, 2))
        self.sheet_conv4 = nn.Conv2d(self.sheet_conv3.out_channels, out_channels=32, kernel_size=(4, 4), stride=2)
        self.sheet_conv5 = nn.Conv2d(self.sheet_conv4.out_channels, out_channels=32, kernel_size=(4, 4), stride=2)

        self.concat_fc = nn.Linear(768 + 1344, 256)

        self.policy_fc = nn.Linear(self.concat_fc.out_features, 256)
        self.policy_out = nn.Linear(self.policy_fc.out_features, n_actions)

        self.critic_fc = nn.Linear(self.concat_fc.out_features, 256)
        self.critic_out = nn.Linear(self.critic_fc.out_features, 1)

        self.apply(weights_init)

        # scaling for orthogonal initialization
        # self.conv1.weight.data.mul_(math.sqrt(2))
        # self.conv2.weight.data.mul_(math.sqrt(2))
        # self.conv3.weight.data.mul_(math.sqrt(2))
        # self.fc1.weight.data.mul_(math.sqrt(2))

    def forward(self, x):

        spec_x = F.elu(self.spec_conv1(x[0]))
        spec_x = F.elu(self.spec_conv2(spec_x))
        spec_x = F.elu(self.spec_conv3(spec_x))

        spec_x = spec_x.view(-1, num_flat_features(spec_x))  # flatten

        sheet_x = F.elu(self.sheet_conv1(x[1]))
        sheet_x = F.elu(self.sheet_conv2(sheet_x))
        sheet_x = F.elu(self.sheet_conv3(sheet_x))
        sheet_x = F.elu(self.sheet_conv4(sheet_x))
        sheet_x = F.elu(self.sheet_conv5(sheet_x))

        sheet_x = sheet_x.view(-1, num_flat_features(sheet_x))  # flatten

        cat_x = torch.cat((spec_x, sheet_x), dim=1)

        cat_x = F.elu(self.concat_fc(cat_x))

        policy = F.elu(self.policy_fc(cat_x))
        policy = self.policy_out(policy)

        critic = F.elu(self.critic_fc(cat_x))
        critic = self.critic_out(critic)

        # if we use softmax and the logarithmic softmax PyTorch recommends to use the log_softmax function
        # therefore I return the linear output of the policy and apply the softmax/log softmax where it is necessary
        return policy, critic


class ScoreFollowingNetMSMDLCHS(nn.Module):
    def __init__(self,  n_actions, spec_channels, sheet_channels):
        super(ScoreFollowingNetMSMDLCHS, self).__init__()

        self.spec_conv1 = nn.Conv2d(spec_channels, out_channels=16, kernel_size=(4, 4), stride=2)
        self.spec_conv2 = nn.Conv2d(self.spec_conv1.out_channels, out_channels=32, kernel_size=(3, 3), stride=2)
        self.spec_conv3 = nn.Conv2d(self.spec_conv2.out_channels, out_channels=32, kernel_size=(3, 3), stride=2)
        self.spec_conv4 = nn.Conv2d(self.spec_conv3.out_channels, out_channels=64, kernel_size=(3, 3), stride=1)

        self.sheet_conv1 = nn.Conv2d(sheet_channels, out_channels=16, kernel_size=(4, 8), stride=2)
        self.sheet_conv2 = nn.Conv2d(self.sheet_conv1.out_channels, out_channels=32, kernel_size=(3, 3), stride=(1, 2))
        self.sheet_conv3 = nn.Conv2d(self.sheet_conv2.out_channels, out_channels=32, kernel_size=(3, 3), stride=(1, 2))
        self.sheet_conv4 = nn.Conv2d(self.sheet_conv3.out_channels, out_channels=32, kernel_size=(4, 4), stride=2)
        self.sheet_conv5 = nn.Conv2d(self.sheet_conv4.out_channels, out_channels=32, kernel_size=(4, 4), stride=2)

        self.concat_fc = nn.Linear(768 + 1344, 256)

        self.policy_fc = nn.Linear(self.concat_fc.out_features, 256)
        self.policy_out = nn.Linear(self.policy_fc.out_features, n_actions)

        self.critic_fc = nn.Linear(self.concat_fc.out_features, 256)
        self.critic_out = nn.Linear(self.critic_fc.out_features, 1)

        self.apply(weights_init)

        # scaling for orthogonal initialization
        # self.conv1.weight.data.mul_(math.sqrt(2))
        # self.conv2.weight.data.mul_(math.sqrt(2))
        # self.conv3.weight.data.mul_(math.sqrt(2))
        # self.fc1.weight.data.mul_(math.sqrt(2))

    def forward(self, x):

        spec_x = F.elu(self.spec_conv1(x[0]))
        spec_x = F.elu(self.spec_conv2(spec_x))
        spec_x = F.elu(self.spec_conv3(spec_x))
        spec_x = F.elu(self.spec_conv4(spec_x))

        spec_x = spec_x.view(-1, num_flat_features(spec_x))  # flatten

        sheet_x = F.elu(self.sheet_conv1(x[1]))
        sheet_x = F.elu(self.sheet_conv2(sheet_x))
        sheet_x = F.elu(self.sheet_conv3(sheet_x))
        sheet_x = F.elu(self.sheet_conv4(sheet_x))
        sheet_x = F.elu(self.sheet_conv5(sheet_x))

        sheet_x = sheet_x.view(-1, num_flat_features(sheet_x))  # flatten

        cat_x = torch.cat((spec_x, sheet_x), dim=1)

        cat_x = F.elu(self.concat_fc(cat_x))

        policy = F.elu(self.policy_fc(cat_x))
        policy = self.policy_out(policy)

        critic = F.elu(self.critic_fc(cat_x))
        critic = self.critic_out(critic)

        # if we use softmax and the logarithmic softmax PyTorch recommends to use the log_softmax function
        # therefore I return the linear output of the policy and apply the softmax/log softmax where it is necessary
        return policy, critic


class ScoreFollowingNetMSMDLCHSContinuous(nn.Module):
    def __init__(self,  n_actions, spec_channels, sheet_channels):
        super(ScoreFollowingNetMSMDLCHSContinuous, self).__init__()

        self.spec_conv1 = nn.Conv2d(spec_channels, out_channels=16, kernel_size=(4, 4), stride=2)
        self.spec_conv2 = nn.Conv2d(self.spec_conv1.out_channels, out_channels=32, kernel_size=(3, 3), stride=2)
        self.spec_conv3 = nn.Conv2d(self.spec_conv2.out_channels, out_channels=32, kernel_size=(3, 3), stride=2)
        self.spec_conv4 = nn.Conv2d(self.spec_conv3.out_channels, out_channels=64, kernel_size=(3, 3), stride=1)

        self.sheet_conv1 = nn.Conv2d(sheet_channels, out_channels=16, kernel_size=(4, 8), stride=2)
        self.sheet_conv2 = nn.Conv2d(self.sheet_conv1.out_channels, out_channels=32, kernel_size=(3, 3), stride=(1, 2))
        self.sheet_conv3 = nn.Conv2d(self.sheet_conv2.out_channels, out_channels=32, kernel_size=(3, 3), stride=(1, 2))
        self.sheet_conv4 = nn.Conv2d(self.sheet_conv3.out_channels, out_channels=32, kernel_size=(4, 4), stride=2)
        self.sheet_conv5 = nn.Conv2d(self.sheet_conv4.out_channels, out_channels=32, kernel_size=(4, 4), stride=2)

        self.concat_fc = nn.Linear(768 + 1344, 256)


        self.mu_fc = nn.Linear(self.concat_fc.out_features, 256)
        self.mu_out = nn.Linear(self.mu_fc.out_features, 1)

        self.sigma_fc = nn.Linear(self.concat_fc.out_features, 256)
        self.sigma_out = nn.Linear(self.sigma_fc.out_features, 1)


        self.critic_fc = nn.Linear(self.concat_fc.out_features, 256)
        self.critic_out = nn.Linear(self.critic_fc.out_features, 1)

        self.apply(weights_init)

        # scaling for orthogonal initialization
        # self.conv1.weight.data.mul_(math.sqrt(2))
        # self.conv2.weight.data.mul_(math.sqrt(2))
        # self.conv3.weight.data.mul_(math.sqrt(2))
        # self.fc1.weight.data.mul_(math.sqrt(2))

    def forward(self, x):

        spec_x = F.elu(self.spec_conv1(x[0]))
        spec_x = F.elu(self.spec_conv2(spec_x))
        spec_x = F.elu(self.spec_conv3(spec_x))
        spec_x = F.elu(self.spec_conv4(spec_x))

        spec_x = spec_x.view(-1, num_flat_features(spec_x))  # flatten

        sheet_x = F.elu(self.sheet_conv1(x[1]))
        sheet_x = F.elu(self.sheet_conv2(sheet_x))
        sheet_x = F.elu(self.sheet_conv3(sheet_x))
        sheet_x = F.elu(self.sheet_conv4(sheet_x))
        sheet_x = F.elu(self.sheet_conv5(sheet_x))

        sheet_x = sheet_x.view(-1, num_flat_features(sheet_x))  # flatten

        cat_x = torch.cat((spec_x, sheet_x), dim=1)

        cat_x = F.elu(self.concat_fc(cat_x))

        critic = F.elu(self.critic_fc(cat_x))
        critic = self.critic_out(critic)

        mu = F.relu(self.mu_fc(cat_x))
        sigma = F.relu(self.sigma_fc(cat_x))

        mu = self.mu_out(mu)
        sigma = torch.exp(self.sigma_out(sigma)) + 1e-3

        return {'mean': mu, 'std': sigma}, critic


class ScoreFollowingNetMSMDLCHSdo(nn.Module):
    def __init__(self,  n_actions, spec_channels, sheet_channels):
        super(ScoreFollowingNetMSMDLCHSdo, self).__init__()

        self.spec_conv1 = nn.Conv2d(spec_channels, out_channels=16, kernel_size=(4, 4), stride=2)
        self.spec_conv2 = nn.Conv2d(self.spec_conv1.out_channels, out_channels=32, kernel_size=(3, 3), stride=2)
        self.spec_conv3 = nn.Conv2d(self.spec_conv2.out_channels, out_channels=32, kernel_size=(3, 3), stride=2)
        self.spec_do3 = nn.Dropout(p=0.2)
        self.spec_conv4 = nn.Conv2d(self.spec_conv3.out_channels, out_channels=64, kernel_size=(3, 3), stride=1)
        self.spec_do4 = nn.Dropout(p=0.2)

        self.sheet_conv1 = nn.Conv2d(sheet_channels, out_channels=16, kernel_size=(4, 8), stride=2)
        self.sheet_conv2 = nn.Conv2d(self.sheet_conv1.out_channels, out_channels=32, kernel_size=(3, 3), stride=(1, 2))
        self.sheet_conv3 = nn.Conv2d(self.sheet_conv2.out_channels, out_channels=32, kernel_size=(3, 3), stride=(1, 2))
        self.sheet_do3 = nn.Dropout(p=0.2)
        self.sheet_conv4 = nn.Conv2d(self.sheet_conv3.out_channels, out_channels=32, kernel_size=(4, 4), stride=2)
        self.sheet_conv5 = nn.Conv2d(self.sheet_conv4.out_channels, out_channels=32, kernel_size=(4, 4), stride=2)
        self.sheet_do5 = nn.Dropout(p=0.2)

        self.concat_fc = nn.Linear(768 + 1344, 256)

        self.policy_fc = nn.Linear(self.concat_fc.out_features, 256)
        self.policy_do = nn.Dropout(p=0.2)
        self.policy_out = nn.Linear(self.policy_fc.out_features, n_actions)

        self.critic_fc = nn.Linear(self.concat_fc.out_features, 256)
        self.critic_do = nn.Dropout(p=0.2)
        self.critic_out = nn.Linear(self.critic_fc.out_features, 1)

        self.apply(weights_init)

        # scaling for orthogonal initialization
        # self.conv1.weight.data.mul_(math.sqrt(2))
        # self.conv2.weight.data.mul_(math.sqrt(2))
        # self.conv3.weight.data.mul_(math.sqrt(2))
        # self.fc1.weight.data.mul_(math.sqrt(2))

    def forward(self, x):

        spec_x = F.elu(self.spec_conv1(x[0]))
        spec_x = F.elu(self.spec_conv2(spec_x))
        spec_x = F.elu(self.spec_conv3(spec_x))
        spec_x = self.spec_do3(spec_x)
        spec_x = F.elu(self.spec_conv4(spec_x))
        spec_x = self.spec_do4(spec_x)

        spec_x = spec_x.view(-1, num_flat_features(spec_x))  # flatten

        sheet_x = F.elu(self.sheet_conv1(x[1]))
        sheet_x = F.elu(self.sheet_conv2(sheet_x))
        sheet_x = F.elu(self.sheet_conv3(sheet_x))
        sheet_x = self.sheet_do3(sheet_x)
        sheet_x = F.elu(self.sheet_conv4(sheet_x))
        sheet_x = F.elu(self.sheet_conv5(sheet_x))
        sheet_x = self.sheet_do5(sheet_x)

        sheet_x = sheet_x.view(-1, num_flat_features(sheet_x))  # flatten

        cat_x = torch.cat((spec_x, sheet_x), dim=1)

        cat_x = F.elu(self.concat_fc(cat_x))

        policy = F.elu(self.policy_fc(cat_x))
        policy = self.policy_do(policy)
        policy = self.policy_out(policy)

        critic = F.elu(self.critic_fc(cat_x))
        critic = self.critic_do(critic)
        critic = self.critic_out(critic)

        # if we use softmax and the logarithmic softmax PyTorch recommends to use the log_softmax function
        # therefore I return the linear output of the policy and apply the softmax/log softmax where it is necessary
        return policy, critic


class ScoreFollowingNetMSMDLCHSA(nn.Module):
    def __init__(self,  n_actions, spec_channels, sheet_channels):
        super(ScoreFollowingNetMSMDLCHSA, self).__init__()

        self.spec_conv1 = nn.Conv2d(spec_channels, out_channels=16, kernel_size=(4, 4), stride=2)
        self.spec_conv2 = nn.Conv2d(self.spec_conv1.out_channels, out_channels=32, kernel_size=(3, 3), stride=2)
        self.spec_conv3 = nn.Conv2d(self.spec_conv2.out_channels, out_channels=32, kernel_size=(3, 3), stride=2)
        self.spec_conv4 = nn.Conv2d(self.spec_conv3.out_channels, out_channels=64, kernel_size=(3, 3), stride=1)
        self.spec_conv5 = nn.Conv2d(self.spec_conv4.out_channels, out_channels=96, kernel_size=(3, 3), stride=1)

        self.sheet_conv1 = nn.Conv2d(sheet_channels, out_channels=16, kernel_size=(4, 8), stride=2)
        self.sheet_conv2 = nn.Conv2d(self.sheet_conv1.out_channels, out_channels=32, kernel_size=(3, 3), stride=(1, 2))
        self.sheet_conv3 = nn.Conv2d(self.sheet_conv2.out_channels, out_channels=32, kernel_size=(3, 3), stride=(1, 2))
        self.sheet_conv4 = nn.Conv2d(self.sheet_conv3.out_channels, out_channels=32, kernel_size=(4, 4), stride=2)
        self.sheet_conv5 = nn.Conv2d(self.sheet_conv4.out_channels, out_channels=32, kernel_size=(4, 4), stride=2)

        self.concat_fc = nn.Linear(768 + 1344, 256)

        self.policy_fc = nn.Linear(self.concat_fc.out_features, 256)
        self.policy_out = nn.Linear(self.policy_fc.out_features, n_actions)

        self.critic_fc = nn.Linear(self.concat_fc.out_features, 256)
        self.critic_out = nn.Linear(self.critic_fc.out_features, 1)

        self.apply(weights_init)

        # scaling for orthogonal initialization
        # self.conv1.weight.data.mul_(math.sqrt(2))
        # self.conv2.weight.data.mul_(math.sqrt(2))
        # self.conv3.weight.data.mul_(math.sqrt(2))
        # self.fc1.weight.data.mul_(math.sqrt(2))

    def forward(self, x):

        spec_x = F.elu(self.spec_conv1(x[0]))
        spec_x = F.elu(self.spec_conv2(spec_x))
        spec_x = F.elu(self.spec_conv3(spec_x))
        spec_x = F.elu(self.spec_conv4(spec_x))
        spec_x = F.elu(self.spec_conv5(spec_x))

        spec_x = spec_x.view(-1, num_flat_features(spec_x))  # flatten

        sheet_x = F.elu(self.sheet_conv1(x[1]))
        sheet_x = F.elu(self.sheet_conv2(sheet_x))
        sheet_x = F.elu(self.sheet_conv3(sheet_x))
        sheet_x = F.elu(self.sheet_conv4(sheet_x))
        sheet_x = F.elu(self.sheet_conv5(sheet_x))

        sheet_x = sheet_x.view(-1, num_flat_features(sheet_x))  # flatten

        cat_x = torch.cat((spec_x, sheet_x), dim=1)

        cat_x = F.elu(self.concat_fc(cat_x))

        policy = F.elu(self.policy_fc(cat_x))
        policy = self.policy_out(policy)

        critic = F.elu(self.critic_fc(cat_x))
        critic = self.critic_out(critic)

        # if we use softmax and the logarithmic softmax PyTorch recommends to use the log_softmax function
        # therefore I return the linear output of the policy and apply the softmax/log softmax where it is necessary
        return policy, critic


class ScoreFollowingNetMSMDLCHSADeepDo(nn.Module):
    def __init__(self,  n_actions, spec_channels, sheet_channels):
        super(ScoreFollowingNetMSMDLCHSADeepDo, self).__init__()

        # spec part
        self.spec_conv1 = nn.Conv2d(spec_channels, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.spec_conv2 = nn.Conv2d(self.spec_conv1.out_channels, out_channels=32, kernel_size=3, stride=1, padding=0)

        self.spec_conv3 = nn.Conv2d(self.spec_conv2.out_channels, out_channels=64, kernel_size=3, stride=2, padding=0)
        self.spec_conv4 = nn.Conv2d(self.spec_conv3.out_channels, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.spec_do4 = nn.Dropout(p=0.2)

        self.spec_conv5 = nn.Conv2d(self.spec_conv4.out_channels, out_channels=64, kernel_size=3, stride=2, padding=0)

        self.spec_conv6 = nn.Conv2d(self.spec_conv5.out_channels, out_channels=96, kernel_size=3, stride=2, padding=0)
        self.spec_conv7 = nn.Conv2d(self.spec_conv6.out_channels, out_channels=96, kernel_size=3, stride=1, padding=0)
        self.spec_conv8 = nn.Conv2d(self.spec_conv7.out_channels, out_channels=96, kernel_size=1, stride=1, padding=0)
        self.spec_do8 = nn.Dropout(p=0.2)

        self.spec_fc = nn.Linear(1440, 512)

        # sheet part
        self.sheet_conv1 = nn.Conv2d(sheet_channels, out_channels=32, kernel_size=5, stride=(1, 2), padding=0)
        self.sheet_conv2 = nn.Conv2d(self.sheet_conv1.out_channels, out_channels=32, kernel_size=3, stride=1, padding=0)

        self.sheet_conv3 = nn.Conv2d(self.sheet_conv2.out_channels, out_channels=64, kernel_size=3, stride=2, padding=0)
        self.sheet_conv4 = nn.Conv2d(self.sheet_conv3.out_channels, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.sheet_do4 = nn.Dropout(p=0.2)

        self.sheet_conv5 = nn.Conv2d(self.sheet_conv4.out_channels, out_channels=64, kernel_size=3, stride=2, padding=0)

        self.sheet_conv6 = nn.Conv2d(self.sheet_conv5.out_channels, out_channels=64, kernel_size=3, stride=2, padding=0)
        self.sheet_do6 = nn.Dropout(p=0.2)

        self.sheet_conv7 = nn.Conv2d(self.sheet_conv6.out_channels, out_channels=96, kernel_size=3, stride=2, padding=0)
        self.sheet_conv8 = nn.Conv2d(self.sheet_conv7.out_channels, out_channels=96, kernel_size=1, stride=1, padding=0)
        self.sheet_do8 = nn.Dropout(p=0.2)

        self.sheet_fc = nn.Linear(1728, 512)

        # multi-modal part
        self.concat_fc = nn.Linear(512 + 512, 512)

        self.policy_fc = nn.Linear(self.concat_fc.out_features, 256)
        self.policy_fc_do = nn.Dropout(p=0.2)
        self.policy_out = nn.Linear(self.policy_fc.out_features, n_actions)

        self.critic_fc = nn.Linear(self.concat_fc.out_features, 256)
        self.critic_fc_do = nn.Dropout(p=0.2)
        self.critic_out = nn.Linear(self.critic_fc.out_features, 1)

        self.apply(weights_init)

        # scaling for orthogonal initialization
        # self.conv1.weight.data.mul_(math.sqrt(2))
        # self.conv2.weight.data.mul_(math.sqrt(2))
        # self.conv3.weight.data.mul_(math.sqrt(2))
        # self.fc1.weight.data.mul_(math.sqrt(2))

    def forward(self, x):

        spec_x = F.elu(self.spec_conv1(x[0]))
        spec_x = F.elu(self.spec_conv2(spec_x))
        spec_x = F.elu(self.spec_conv3(spec_x))
        spec_x = F.elu(self.spec_conv4(spec_x))
        spec_x = self.spec_do4(spec_x)
        spec_x = F.elu(self.spec_conv5(spec_x))
        spec_x = F.elu(self.spec_conv6(spec_x))
        spec_x = F.elu(self.spec_conv7(spec_x))
        spec_x = F.elu(self.spec_conv8(spec_x))
        spec_x = self.spec_do8(spec_x)

        spec_x = spec_x.view(-1, num_flat_features(spec_x))  # flatten
        spec_x = F.elu(self.spec_fc(spec_x))

        sheet_x = F.elu(self.sheet_conv1(x[1]))
        sheet_x = F.elu(self.sheet_conv2(sheet_x))
        sheet_x = F.elu(self.sheet_conv3(sheet_x))
        sheet_x = F.elu(self.sheet_conv4(sheet_x))
        sheet_x = self.sheet_do4(sheet_x)
        sheet_x = F.elu(self.sheet_conv5(sheet_x))
        sheet_x = F.elu(self.sheet_conv6(sheet_x))
        sheet_x = self.sheet_do6(sheet_x)
        sheet_x = F.elu(self.sheet_conv7(sheet_x))
        sheet_x = F.elu(self.sheet_conv8(sheet_x))
        sheet_x = self.sheet_do8(sheet_x)

        sheet_x = sheet_x.view(-1, num_flat_features(sheet_x))  # flatten
        sheet_x = F.elu(self.sheet_fc(sheet_x))

        cat_x = torch.cat((spec_x, sheet_x), dim=1)

        cat_x = F.elu(self.concat_fc(cat_x))

        policy = F.elu(self.policy_fc(cat_x))
        policy = self.policy_fc_do(policy)
        policy = self.policy_out(policy)

        critic = F.elu(self.critic_fc(cat_x))
        critic = self.critic_fc_do(critic)
        critic = self.critic_out(critic)

        # if we use softmax and the logarithmic softmax PyTorch recommends to use the log_softmax function
        # therefore I return the linear output of the policy and apply the softmax/log softmax where it is necessary
        return policy, critic


class ScoreFollowingNetMSMDLCHSDeep(nn.Module):
    def __init__(self,  n_actions, spec_channels, sheet_channels):
        super(ScoreFollowingNetMSMDLCHSDeep, self).__init__()

        # spec part
        self.spec_conv1 = nn.Conv2d(spec_channels, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.spec_conv2 = nn.Conv2d(self.spec_conv1.out_channels, out_channels=32, kernel_size=3, stride=1, padding=0)

        self.spec_conv3 = nn.Conv2d(self.spec_conv2.out_channels, out_channels=64, kernel_size=3, stride=2, padding=0)
        self.spec_conv4 = nn.Conv2d(self.spec_conv3.out_channels, out_channels=64, kernel_size=3, stride=1, padding=0)

        self.spec_conv5 = nn.Conv2d(self.spec_conv4.out_channels, out_channels=64, kernel_size=3, stride=2, padding=0)

        self.spec_conv6 = nn.Conv2d(self.spec_conv5.out_channels, out_channels=96, kernel_size=3, stride=2, padding=0)
        self.spec_conv7 = nn.Conv2d(self.spec_conv6.out_channels, out_channels=96, kernel_size=1, stride=1, padding=0)

        self.spec_fc = nn.Linear(2016, 512)

        # sheet part
        self.sheet_conv1 = nn.Conv2d(sheet_channels, out_channels=32, kernel_size=5, stride=(1, 2), padding=0)
        self.sheet_conv2 = nn.Conv2d(self.sheet_conv1.out_channels, out_channels=32, kernel_size=3, stride=1, padding=0)

        self.sheet_conv3 = nn.Conv2d(self.sheet_conv2.out_channels, out_channels=64, kernel_size=3, stride=2, padding=0)
        self.sheet_conv4 = nn.Conv2d(self.sheet_conv3.out_channels, out_channels=64, kernel_size=3, stride=1, padding=0)

        self.sheet_conv5 = nn.Conv2d(self.sheet_conv4.out_channels, out_channels=64, kernel_size=3, stride=2, padding=0)

        self.sheet_conv6 = nn.Conv2d(self.sheet_conv5.out_channels, out_channels=64, kernel_size=3, stride=2, padding=0)

        self.sheet_conv7 = nn.Conv2d(self.sheet_conv6.out_channels, out_channels=96, kernel_size=3, stride=2, padding=0)
        self.sheet_conv8 = nn.Conv2d(self.sheet_conv7.out_channels, out_channels=96, kernel_size=1, stride=1, padding=0)

        self.sheet_fc = nn.Linear(1728, 512)

        # multi-modal part
        self.concat_fc = nn.Linear(512 + 512, 512)

        self.policy_fc = nn.Linear(self.concat_fc.out_features, 256)
        self.policy_out = nn.Linear(self.policy_fc.out_features, n_actions)

        self.critic_fc = nn.Linear(self.concat_fc.out_features, 256)
        self.critic_out = nn.Linear(self.critic_fc.out_features, 1)

        self.apply(weights_init)

        # scaling for orthogonal initialization
        # self.conv1.weight.data.mul_(math.sqrt(2))
        # self.conv2.weight.data.mul_(math.sqrt(2))
        # self.conv3.weight.data.mul_(math.sqrt(2))
        # self.fc1.weight.data.mul_(math.sqrt(2))

    def forward(self, x):

        spec_x = F.elu(self.spec_conv1(x[0]))
        spec_x = F.elu(self.spec_conv2(spec_x))
        spec_x = F.elu(self.spec_conv3(spec_x))
        spec_x = F.elu(self.spec_conv4(spec_x))
        spec_x = F.elu(self.spec_conv5(spec_x))
        spec_x = F.elu(self.spec_conv6(spec_x))
        spec_x = F.elu(self.spec_conv7(spec_x))

        spec_x = spec_x.view(-1, num_flat_features(spec_x))  # flatten
        spec_x = F.elu(self.spec_fc(spec_x))

        sheet_x = F.elu(self.sheet_conv1(x[1]))
        sheet_x = F.elu(self.sheet_conv2(sheet_x))
        sheet_x = F.elu(self.sheet_conv3(sheet_x))
        sheet_x = F.elu(self.sheet_conv4(sheet_x))
        sheet_x = F.elu(self.sheet_conv5(sheet_x))
        sheet_x = F.elu(self.sheet_conv6(sheet_x))
        sheet_x = F.elu(self.sheet_conv7(sheet_x))
        sheet_x = F.elu(self.sheet_conv8(sheet_x))

        sheet_x = sheet_x.view(-1, num_flat_features(sheet_x))  # flatten
        sheet_x = F.elu(self.sheet_fc(sheet_x))

        cat_x = torch.cat((spec_x, sheet_x), dim=1)

        cat_x = F.elu(self.concat_fc(cat_x))

        policy = F.elu(self.policy_fc(cat_x))
        policy = self.policy_out(policy)

        critic = F.elu(self.critic_fc(cat_x))
        critic = self.critic_out(critic)

        # if we use softmax and the logarithmic softmax PyTorch recommends to use the log_softmax function
        # therefore I return the linear output of the policy and apply the softmax/log softmax where it is necessary
        return policy, critic


class ScoreFollowingNetMSMDLCHSDeepRELU(nn.Module):
    def __init__(self,  n_actions, spec_channels, sheet_channels):
        super(ScoreFollowingNetMSMDLCHSDeepRELU, self).__init__()

        # spec part
        self.spec_conv1 = nn.Conv2d(spec_channels, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.spec_conv2 = nn.Conv2d(self.spec_conv1.out_channels, out_channels=32, kernel_size=3, stride=1, padding=0)

        self.spec_conv3 = nn.Conv2d(self.spec_conv2.out_channels, out_channels=64, kernel_size=3, stride=2, padding=0)
        self.spec_conv4 = nn.Conv2d(self.spec_conv3.out_channels, out_channels=64, kernel_size=3, stride=1, padding=0)

        self.spec_conv5 = nn.Conv2d(self.spec_conv4.out_channels, out_channels=64, kernel_size=3, stride=2, padding=0)

        self.spec_conv6 = nn.Conv2d(self.spec_conv5.out_channels, out_channels=96, kernel_size=3, stride=2, padding=0)
        self.spec_conv7 = nn.Conv2d(self.spec_conv6.out_channels, out_channels=96, kernel_size=1, stride=1, padding=0)

        self.spec_fc = nn.Linear(2016, 512)

        # sheet part
        self.sheet_conv1 = nn.Conv2d(sheet_channels, out_channels=32, kernel_size=5, stride=(1, 2), padding=0)
        self.sheet_conv2 = nn.Conv2d(self.sheet_conv1.out_channels, out_channels=32, kernel_size=3, stride=1, padding=0)

        self.sheet_conv3 = nn.Conv2d(self.sheet_conv2.out_channels, out_channels=64, kernel_size=3, stride=2, padding=0)
        self.sheet_conv4 = nn.Conv2d(self.sheet_conv3.out_channels, out_channels=64, kernel_size=3, stride=1, padding=0)

        self.sheet_conv5 = nn.Conv2d(self.sheet_conv4.out_channels, out_channels=64, kernel_size=3, stride=2, padding=0)

        self.sheet_conv6 = nn.Conv2d(self.sheet_conv5.out_channels, out_channels=64, kernel_size=3, stride=2, padding=0)

        self.sheet_conv7 = nn.Conv2d(self.sheet_conv6.out_channels, out_channels=96, kernel_size=3, stride=2, padding=0)
        self.sheet_conv8 = nn.Conv2d(self.sheet_conv7.out_channels, out_channels=96, kernel_size=1, stride=1, padding=0)

        self.sheet_fc = nn.Linear(1728, 512)

        # multi-modal part
        self.concat_fc = nn.Linear(512 + 512, 512)

        self.policy_fc = nn.Linear(self.concat_fc.out_features, 256)
        self.policy_out = nn.Linear(self.policy_fc.out_features, n_actions)

        self.critic_fc = nn.Linear(self.concat_fc.out_features, 256)
        self.critic_out = nn.Linear(self.critic_fc.out_features, 1)

        self.apply(weights_init)

        # scaling for orthogonal initialization
        # self.conv1.weight.data.mul_(math.sqrt(2))
        # self.conv2.weight.data.mul_(math.sqrt(2))
        # self.conv3.weight.data.mul_(math.sqrt(2))
        # self.fc1.weight.data.mul_(math.sqrt(2))

    def forward(self, x):

        spec_x = F.relu(self.spec_conv1(x[0]))
        spec_x = F.relu(self.spec_conv2(spec_x))
        spec_x = F.relu(self.spec_conv3(spec_x))
        spec_x = F.relu(self.spec_conv4(spec_x))
        spec_x = F.relu(self.spec_conv5(spec_x))
        spec_x = F.relu(self.spec_conv6(spec_x))
        spec_x = F.relu(self.spec_conv7(spec_x))

        spec_x = spec_x.view(-1, num_flat_features(spec_x))  # flatten
        spec_x = F.relu(self.spec_fc(spec_x))

        sheet_x = F.relu(self.sheet_conv1(x[1]))
        sheet_x = F.relu(self.sheet_conv2(sheet_x))
        sheet_x = F.relu(self.sheet_conv3(sheet_x))
        sheet_x = F.relu(self.sheet_conv4(sheet_x))
        sheet_x = F.relu(self.sheet_conv5(sheet_x))
        sheet_x = F.relu(self.sheet_conv6(sheet_x))
        sheet_x = F.relu(self.sheet_conv7(sheet_x))
        sheet_x = F.relu(self.sheet_conv8(sheet_x))

        sheet_x = sheet_x.view(-1, num_flat_features(sheet_x))  # flatten
        sheet_x = F.relu(self.sheet_fc(sheet_x))

        cat_x = torch.cat((spec_x, sheet_x), dim=1)

        cat_x = F.relu(self.concat_fc(cat_x))

        policy = F.relu(self.policy_fc(cat_x))
        policy = self.policy_out(policy)

        critic = F.relu(self.critic_fc(cat_x))
        critic = self.critic_out(critic)

        # if we use softmax and the logarithmic softmax PyTorch recommends to use the log_softmax function
        # therefore I return the linear output of the policy and apply the softmax/log softmax where it is necessary
        return policy, critic


class ScoreFollowingNetMSMDLCHSDeepSELU(nn.Module):
    def __init__(self,  n_actions, spec_channels, sheet_channels):
        super(ScoreFollowingNetMSMDLCHSDeepSELU, self).__init__()

        # spec part
        self.spec_conv1 = nn.Conv2d(spec_channels, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.spec_conv2 = nn.Conv2d(self.spec_conv1.out_channels, out_channels=32, kernel_size=3, stride=1, padding=0)

        self.spec_conv3 = nn.Conv2d(self.spec_conv2.out_channels, out_channels=64, kernel_size=3, stride=2, padding=0)
        self.spec_conv4 = nn.Conv2d(self.spec_conv3.out_channels, out_channels=64, kernel_size=3, stride=1, padding=0)

        self.spec_conv5 = nn.Conv2d(self.spec_conv4.out_channels, out_channels=64, kernel_size=3, stride=2, padding=0)

        self.spec_conv6 = nn.Conv2d(self.spec_conv5.out_channels, out_channels=96, kernel_size=3, stride=2, padding=0)
        self.spec_conv7 = nn.Conv2d(self.spec_conv6.out_channels, out_channels=96, kernel_size=1, stride=1, padding=0)

        self.spec_fc = nn.Linear(2016, 512)

        # sheet part
        self.sheet_conv1 = nn.Conv2d(sheet_channels, out_channels=32, kernel_size=5, stride=(1, 2), padding=0)
        self.sheet_conv2 = nn.Conv2d(self.sheet_conv1.out_channels, out_channels=32, kernel_size=3, stride=1, padding=0)

        self.sheet_conv3 = nn.Conv2d(self.sheet_conv2.out_channels, out_channels=64, kernel_size=3, stride=2, padding=0)
        self.sheet_conv4 = nn.Conv2d(self.sheet_conv3.out_channels, out_channels=64, kernel_size=3, stride=1, padding=0)

        self.sheet_conv5 = nn.Conv2d(self.sheet_conv4.out_channels, out_channels=64, kernel_size=3, stride=2, padding=0)

        self.sheet_conv6 = nn.Conv2d(self.sheet_conv5.out_channels, out_channels=64, kernel_size=3, stride=2, padding=0)

        self.sheet_conv7 = nn.Conv2d(self.sheet_conv6.out_channels, out_channels=96, kernel_size=3, stride=2, padding=0)
        self.sheet_conv8 = nn.Conv2d(self.sheet_conv7.out_channels, out_channels=96, kernel_size=1, stride=1, padding=0)

        self.sheet_fc = nn.Linear(1728, 512)

        # multi-modal part
        self.concat_fc = nn.Linear(512 + 512, 512)

        self.policy_fc = nn.Linear(self.concat_fc.out_features, 256)
        self.policy_out = nn.Linear(self.policy_fc.out_features, n_actions)

        self.critic_fc = nn.Linear(self.concat_fc.out_features, 256)
        self.critic_out = nn.Linear(self.critic_fc.out_features, 1)

        self.apply(weights_init_selu)

    def forward(self, x):

        spec_x = F.selu(self.spec_conv1(x[0]))
        spec_x = F.selu(self.spec_conv2(spec_x))
        spec_x = F.selu(self.spec_conv3(spec_x))
        spec_x = F.selu(self.spec_conv4(spec_x))
        spec_x = F.selu(self.spec_conv5(spec_x))
        spec_x = F.selu(self.spec_conv6(spec_x))
        spec_x = F.selu(self.spec_conv7(spec_x))

        spec_x = spec_x.view(-1, num_flat_features(spec_x))  # flatten
        spec_x = F.selu(self.spec_fc(spec_x))

        sheet_x = F.selu(self.sheet_conv1(x[1]))
        sheet_x = F.selu(self.sheet_conv2(sheet_x))
        sheet_x = F.selu(self.sheet_conv3(sheet_x))
        sheet_x = F.selu(self.sheet_conv4(sheet_x))
        sheet_x = F.selu(self.sheet_conv5(sheet_x))
        sheet_x = F.selu(self.sheet_conv6(sheet_x))
        sheet_x = F.selu(self.sheet_conv7(sheet_x))
        sheet_x = F.selu(self.sheet_conv8(sheet_x))

        sheet_x = sheet_x.view(-1, num_flat_features(sheet_x))  # flatten
        sheet_x = F.selu(self.sheet_fc(sheet_x))

        cat_x = torch.cat((spec_x, sheet_x), dim=1)

        cat_x = F.selu(self.concat_fc(cat_x))

        policy = F.selu(self.policy_fc(cat_x))
        policy = self.policy_out(policy)

        critic = F.selu(self.critic_fc(cat_x))
        critic = self.critic_out(critic)

        # if we use softmax and the logarithmic softmax PyTorch recommends to use the log_softmax function
        # therefore I return the linear output of the policy and apply the softmax/log softmax where it is necessary
        return policy, critic


class ScoreFollowingNetMSMDLCHSDeepSELU01(nn.Module):
    def __init__(self,  n_actions, spec_channels, sheet_channels):
        super(ScoreFollowingNetMSMDLCHSDeepSELU01, self).__init__()

        # use batch norm layer for input standardization
        self.spec_bn = nn.BatchNorm2d(2, affine=False, momentum=0.01)
        self.sheet_bn = nn.BatchNorm2d(2, affine=False, momentum=0.01)

        # spec part
        self.spec_conv1 = nn.Conv2d(spec_channels, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.spec_conv2 = nn.Conv2d(self.spec_conv1.out_channels, out_channels=32, kernel_size=3, stride=1, padding=0)

        self.spec_conv3 = nn.Conv2d(self.spec_conv2.out_channels, out_channels=64, kernel_size=3, stride=2, padding=0)
        self.spec_conv4 = nn.Conv2d(self.spec_conv3.out_channels, out_channels=64, kernel_size=3, stride=1, padding=0)

        self.spec_conv5 = nn.Conv2d(self.spec_conv4.out_channels, out_channels=64, kernel_size=3, stride=2, padding=0)

        self.spec_conv6 = nn.Conv2d(self.spec_conv5.out_channels, out_channels=96, kernel_size=3, stride=2, padding=0)
        self.spec_conv7 = nn.Conv2d(self.spec_conv6.out_channels, out_channels=96, kernel_size=1, stride=1, padding=0)

        self.spec_fc = nn.Linear(2016, 512)

        # sheet part
        self.sheet_conv1 = nn.Conv2d(sheet_channels, out_channels=32, kernel_size=5, stride=(1, 2), padding=0)
        self.sheet_conv2 = nn.Conv2d(self.sheet_conv1.out_channels, out_channels=32, kernel_size=3, stride=1, padding=0)

        self.sheet_conv3 = nn.Conv2d(self.sheet_conv2.out_channels, out_channels=64, kernel_size=3, stride=2, padding=0)
        self.sheet_conv4 = nn.Conv2d(self.sheet_conv3.out_channels, out_channels=64, kernel_size=3, stride=1, padding=0)

        self.sheet_conv5 = nn.Conv2d(self.sheet_conv4.out_channels, out_channels=64, kernel_size=3, stride=2, padding=0)

        self.sheet_conv6 = nn.Conv2d(self.sheet_conv5.out_channels, out_channels=64, kernel_size=3, stride=2, padding=0)

        self.sheet_conv7 = nn.Conv2d(self.sheet_conv6.out_channels, out_channels=96, kernel_size=3, stride=2, padding=0)
        self.sheet_conv8 = nn.Conv2d(self.sheet_conv7.out_channels, out_channels=96, kernel_size=1, stride=1, padding=0)

        self.sheet_fc = nn.Linear(1728, 512)

        # multi-modal part
        self.concat_fc = nn.Linear(512 + 512, 512)

        self.policy_fc = nn.Linear(self.concat_fc.out_features, 256)
        self.policy_out = nn.Linear(self.policy_fc.out_features, n_actions)

        self.critic_fc = nn.Linear(self.concat_fc.out_features, 256)
        self.critic_out = nn.Linear(self.critic_fc.out_features, 1)

        self.apply(weights_init_selu)

    def forward(self, x):

        # normalize data
        x[0] = self.spec_bn(x[0])
        x[1] = self.sheet_bn(x[1])

        spec_x = F.selu(self.spec_conv1(x[0]))
        spec_x = F.selu(self.spec_conv2(spec_x))
        spec_x = F.selu(self.spec_conv3(spec_x))
        spec_x = F.selu(self.spec_conv4(spec_x))
        spec_x = F.selu(self.spec_conv5(spec_x))
        spec_x = F.selu(self.spec_conv6(spec_x))
        spec_x = F.selu(self.spec_conv7(spec_x))

        spec_x = spec_x.view(-1, num_flat_features(spec_x))  # flatten
        spec_x = F.selu(self.spec_fc(spec_x))

        sheet_x = F.selu(self.sheet_conv1(x[1]))
        sheet_x = F.selu(self.sheet_conv2(sheet_x))
        sheet_x = F.selu(self.sheet_conv3(sheet_x))
        sheet_x = F.selu(self.sheet_conv4(sheet_x))
        sheet_x = F.selu(self.sheet_conv5(sheet_x))
        sheet_x = F.selu(self.sheet_conv6(sheet_x))
        sheet_x = F.selu(self.sheet_conv7(sheet_x))
        sheet_x = F.selu(self.sheet_conv8(sheet_x))

        sheet_x = sheet_x.view(-1, num_flat_features(sheet_x))  # flatten
        sheet_x = F.selu(self.sheet_fc(sheet_x))

        cat_x = torch.cat((spec_x, sheet_x), dim=1)

        cat_x = F.selu(self.concat_fc(cat_x))

        policy = F.selu(self.policy_fc(cat_x))
        policy = self.policy_out(policy)

        critic = F.selu(self.critic_fc(cat_x))
        critic = self.critic_out(critic)

        # if we use softmax and the logarithmic softmax PyTorch recommends to use the log_softmax function
        # therefore I return the linear output of the policy and apply the softmax/log softmax where it is necessary
        return policy, critic


class ScoreFollowingNetMSMDLCHSDeepLSTM(nn.Module):
    def __init__(self,  n_actions, spec_channels, sheet_channels):
        super(ScoreFollowingNetMSMDLCHSDeepLSTM, self).__init__()

        # spec part
        self.spec_conv1 = nn.Conv2d(spec_channels, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.spec_conv2 = nn.Conv2d(self.spec_conv1.out_channels, out_channels=32, kernel_size=3, stride=1, padding=0)

        self.spec_conv3 = nn.Conv2d(self.spec_conv2.out_channels, out_channels=64, kernel_size=3, stride=2, padding=0)
        self.spec_conv4 = nn.Conv2d(self.spec_conv3.out_channels, out_channels=64, kernel_size=3, stride=1, padding=0)

        self.spec_conv5 = nn.Conv2d(self.spec_conv4.out_channels, out_channels=64, kernel_size=3, stride=2, padding=0)

        self.spec_conv6 = nn.Conv2d(self.spec_conv5.out_channels, out_channels=96, kernel_size=3, stride=2, padding=0)
        self.spec_conv7 = nn.Conv2d(self.spec_conv6.out_channels, out_channels=96, kernel_size=1, stride=1, padding=0)

        self.spec_fc = nn.Linear(2016, 512)

        # sheet part
        self.sheet_conv1 = nn.Conv2d(sheet_channels, out_channels=32, kernel_size=5, stride=(1, 2), padding=0)
        self.sheet_conv2 = nn.Conv2d(self.sheet_conv1.out_channels, out_channels=32, kernel_size=3, stride=1, padding=0)

        self.sheet_conv3 = nn.Conv2d(self.sheet_conv2.out_channels, out_channels=64, kernel_size=3, stride=2, padding=0)
        self.sheet_conv4 = nn.Conv2d(self.sheet_conv3.out_channels, out_channels=64, kernel_size=3, stride=1, padding=0)

        self.sheet_conv5 = nn.Conv2d(self.sheet_conv4.out_channels, out_channels=64, kernel_size=3, stride=2, padding=0)

        self.sheet_conv6 = nn.Conv2d(self.sheet_conv5.out_channels, out_channels=64, kernel_size=3, stride=2, padding=0)

        self.sheet_conv7 = nn.Conv2d(self.sheet_conv6.out_channels, out_channels=96, kernel_size=3, stride=2, padding=0)
        self.sheet_conv8 = nn.Conv2d(self.sheet_conv7.out_channels, out_channels=96, kernel_size=1, stride=1, padding=0)

        self.sheet_fc = nn.Linear(1728, 512)

        # multi-modal part
        self.concat_fc = nn.Linear(512 + 512, 512)

        self.lstm_size = 256
        self.lstm_layers = 1
        self.lstm = nn.LSTM(self.concat_fc.out_features, self.lstm_size, num_layers=self.lstm_layers)

        self.policy_fc = nn.Linear(self.lstm_size, 256)
        self.policy_out = nn.Linear(self.policy_fc.out_features, n_actions)

        self.critic_fc = nn.Linear(self.lstm_size, 256)
        self.critic_out = nn.Linear(self.critic_fc.out_features, 1)

        self.apply(weights_init)

        # scaling for orthogonal initialization
        # self.conv1.weight.data.mul_(math.sqrt(2))
        # self.conv2.weight.data.mul_(math.sqrt(2))
        # self.conv3.weight.data.mul_(math.sqrt(2))
        # self.fc1.weight.data.mul_(math.sqrt(2))

    def forward(self, x):
        spec_x, (hidden_state, cell_state) = x[0]
        sheet_x, _ = x[1]

        spec_x = F.elu(self.spec_conv1(spec_x))
        spec_x = F.elu(self.spec_conv2(spec_x))
        spec_x = F.elu(self.spec_conv3(spec_x))
        spec_x = F.elu(self.spec_conv4(spec_x))
        spec_x = F.elu(self.spec_conv5(spec_x))
        spec_x = F.elu(self.spec_conv6(spec_x))
        spec_x = F.elu(self.spec_conv7(spec_x))

        spec_x = spec_x.view(-1, num_flat_features(spec_x))  # flatten
        spec_x = F.elu(self.spec_fc(spec_x))

        sheet_x = F.elu(self.sheet_conv1(sheet_x))
        sheet_x = F.elu(self.sheet_conv2(sheet_x))
        sheet_x = F.elu(self.sheet_conv3(sheet_x))
        sheet_x = F.elu(self.sheet_conv4(sheet_x))
        sheet_x = F.elu(self.sheet_conv5(sheet_x))
        sheet_x = F.elu(self.sheet_conv6(sheet_x))
        sheet_x = F.elu(self.sheet_conv7(sheet_x))
        sheet_x = F.elu(self.sheet_conv8(sheet_x))

        sheet_x = sheet_x.view(-1, num_flat_features(sheet_x))  # flatten
        sheet_x = F.elu(self.sheet_fc(sheet_x))

        cat_x = torch.cat((spec_x, sheet_x), dim=1)

        cat_x = F.elu(self.concat_fc(cat_x))

        cat_x, (hidden_state_n, cell_state_n) = self.lstm(cat_x.unsqueeze(0), (hidden_state, cell_state))

        # only one time step
        cat_x = cat_x[0]

        policy = F.elu(self.policy_fc(cat_x))
        policy = self.policy_out(policy)

        critic = F.elu(self.critic_fc(cat_x))
        critic = self.critic_out(critic)

        # if we use softmax and the logarithmic softmax PyTorch recommends to use the log_softmax function
        # therefore I return the linear output of the policy and apply the softmax/log softmax where it is necessary
        return policy, critic, (hidden_state_n, cell_state_n)


class ScoreFollowingNetMSMDLCHSDeepDo(nn.Module):
    def __init__(self,  n_actions, spec_channels, sheet_channels):
        super(ScoreFollowingNetMSMDLCHSDeepDo, self).__init__()

        # spec part
        self.spec_conv1 = nn.Conv2d(spec_channels, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.spec_conv2 = nn.Conv2d(self.spec_conv1.out_channels, out_channels=32, kernel_size=3, stride=1, padding=0)

        self.spec_conv3 = nn.Conv2d(self.spec_conv2.out_channels, out_channels=64, kernel_size=3, stride=2, padding=0)
        self.spec_conv4 = nn.Conv2d(self.spec_conv3.out_channels, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.spec_do4 = nn.Dropout(p=0.2)

        self.spec_conv5 = nn.Conv2d(self.spec_conv4.out_channels, out_channels=64, kernel_size=3, stride=2, padding=0)

        self.spec_conv6 = nn.Conv2d(self.spec_conv5.out_channels, out_channels=96, kernel_size=3, stride=2, padding=0)
        self.spec_conv7 = nn.Conv2d(self.spec_conv6.out_channels, out_channels=96, kernel_size=1, stride=1, padding=0)
        self.spec_do7 = nn.Dropout(p=0.2)

        self.spec_fc = nn.Linear(2016, 512)

        # sheet part
        self.sheet_conv1 = nn.Conv2d(sheet_channels, out_channels=32, kernel_size=5, stride=(1, 2), padding=0)
        self.sheet_conv2 = nn.Conv2d(self.sheet_conv1.out_channels, out_channels=32, kernel_size=3, stride=1, padding=0)

        self.sheet_conv3 = nn.Conv2d(self.sheet_conv2.out_channels, out_channels=64, kernel_size=3, stride=2, padding=0)
        self.sheet_conv4 = nn.Conv2d(self.sheet_conv3.out_channels, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.sheet_do4 = nn.Dropout(p=0.2)

        self.sheet_conv5 = nn.Conv2d(self.sheet_conv4.out_channels, out_channels=64, kernel_size=3, stride=2, padding=0)

        self.sheet_conv6 = nn.Conv2d(self.sheet_conv5.out_channels, out_channels=64, kernel_size=3, stride=2, padding=0)
        self.sheet_do6 = nn.Dropout(p=0.2)

        self.sheet_conv7 = nn.Conv2d(self.sheet_conv6.out_channels, out_channels=96, kernel_size=3, stride=2, padding=0)
        self.sheet_conv8 = nn.Conv2d(self.sheet_conv7.out_channels, out_channels=96, kernel_size=1, stride=1, padding=0)
        self.sheet_do8 = nn.Dropout(p=0.2)

        self.sheet_fc = nn.Linear(1728, 512)

        # multi-modal part
        self.concat_fc = nn.Linear(512 + 512, 512)

        self.policy_fc = nn.Linear(self.concat_fc.out_features, 256)
        self.policy_fc_do = nn.Dropout(p=0.2)
        self.policy_out = nn.Linear(self.policy_fc.out_features, n_actions)

        self.critic_fc = nn.Linear(self.concat_fc.out_features, 256)
        self.critic_fc_do = nn.Dropout(p=0.2)
        self.critic_out = nn.Linear(self.critic_fc.out_features, 1)

        self.apply(weights_init)

        # scaling for orthogonal initialization
        # self.conv1.weight.data.mul_(math.sqrt(2))
        # self.conv2.weight.data.mul_(math.sqrt(2))
        # self.conv3.weight.data.mul_(math.sqrt(2))
        # self.fc1.weight.data.mul_(math.sqrt(2))

    def forward(self, x):

        spec_x = F.elu(self.spec_conv1(x[0]))
        spec_x = F.elu(self.spec_conv2(spec_x))
        spec_x = F.elu(self.spec_conv3(spec_x))
        spec_x = F.elu(self.spec_conv4(spec_x))
        spec_x = self.spec_do4(spec_x)
        spec_x = F.elu(self.spec_conv5(spec_x))
        spec_x = F.elu(self.spec_conv6(spec_x))
        spec_x = F.elu(self.spec_conv7(spec_x))
        spec_x = self.spec_do7(spec_x)

        spec_x = spec_x.view(-1, num_flat_features(spec_x))  # flatten
        spec_x = F.elu(self.spec_fc(spec_x))

        sheet_x = F.elu(self.sheet_conv1(x[1]))
        sheet_x = F.elu(self.sheet_conv2(sheet_x))
        sheet_x = F.elu(self.sheet_conv3(sheet_x))
        sheet_x = F.elu(self.sheet_conv4(sheet_x))
        sheet_x = self.sheet_do4(sheet_x)
        sheet_x = F.elu(self.sheet_conv5(sheet_x))
        sheet_x = F.elu(self.sheet_conv6(sheet_x))
        sheet_x = self.sheet_do6(sheet_x)
        sheet_x = F.elu(self.sheet_conv7(sheet_x))
        sheet_x = F.elu(self.sheet_conv8(sheet_x))
        sheet_x = self.sheet_do8(sheet_x)

        sheet_x = sheet_x.view(-1, num_flat_features(sheet_x))  # flatten
        sheet_x = F.elu(self.sheet_fc(sheet_x))

        cat_x = torch.cat((spec_x, sheet_x), dim=1)

        cat_x = F.elu(self.concat_fc(cat_x))

        policy = F.elu(self.policy_fc(cat_x))
        policy = self.policy_fc_do(policy)
        policy = self.policy_out(policy)

        critic = F.elu(self.critic_fc(cat_x))
        critic = self.critic_fc_do(critic)
        critic = self.critic_out(critic)

        # if we use softmax and the logarithmic softmax PyTorch recommends to use the log_softmax function
        # therefore I return the linear output of the policy and apply the softmax/log softmax where it is necessary
        return policy, critic


class ScoreFollowingNetMSMDLCHSDeepContinuous(nn.Module):
    def __init__(self,  n_actions, spec_channels, sheet_channels):
        super(ScoreFollowingNetMSMDLCHSDeepContinuous, self).__init__()

        # spec part
        self.spec_conv1 = nn.Conv2d(spec_channels, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.spec_conv2 = nn.Conv2d(self.spec_conv1.out_channels, out_channels=32, kernel_size=3, stride=1, padding=0)

        self.spec_conv3 = nn.Conv2d(self.spec_conv2.out_channels, out_channels=64, kernel_size=3, stride=2, padding=0)
        self.spec_conv4 = nn.Conv2d(self.spec_conv3.out_channels, out_channels=64, kernel_size=3, stride=1, padding=0)

        self.spec_conv5 = nn.Conv2d(self.spec_conv4.out_channels, out_channels=64, kernel_size=3, stride=2, padding=0)

        self.spec_conv6 = nn.Conv2d(self.spec_conv5.out_channels, out_channels=96, kernel_size=3, stride=2, padding=0)
        self.spec_conv7 = nn.Conv2d(self.spec_conv6.out_channels, out_channels=96, kernel_size=1, stride=1, padding=0)

        self.spec_fc = nn.Linear(2016, 512)

        # sheet part
        self.sheet_conv1 = nn.Conv2d(sheet_channels, out_channels=32, kernel_size=5, stride=(1, 2), padding=0)
        self.sheet_conv2 = nn.Conv2d(self.sheet_conv1.out_channels, out_channels=32, kernel_size=3, stride=1, padding=0)

        self.sheet_conv3 = nn.Conv2d(self.sheet_conv2.out_channels, out_channels=64, kernel_size=3, stride=2, padding=0)
        self.sheet_conv4 = nn.Conv2d(self.sheet_conv3.out_channels, out_channels=64, kernel_size=3, stride=1, padding=0)

        self.sheet_conv5 = nn.Conv2d(self.sheet_conv4.out_channels, out_channels=64, kernel_size=3, stride=2, padding=0)

        self.sheet_conv6 = nn.Conv2d(self.sheet_conv5.out_channels, out_channels=64, kernel_size=3, stride=2, padding=0)

        self.sheet_conv7 = nn.Conv2d(self.sheet_conv6.out_channels, out_channels=96, kernel_size=3, stride=2, padding=0)
        self.sheet_conv8 = nn.Conv2d(self.sheet_conv7.out_channels, out_channels=96, kernel_size=1, stride=1, padding=0)

        self.sheet_fc = nn.Linear(1728, 512)

        # multi-modal part
        self.concat_fc = nn.Linear(512 + 512, 512)

        self.mu_fc = nn.Linear(self.concat_fc.out_features, 256)
        self.mu_out = nn.Linear(self.mu_fc.out_features, 1)

        self.sigma_fc = nn.Linear(self.concat_fc.out_features, 256)
        self.sigma_out = nn.Linear(self.sigma_fc.out_features, 1)

        self.critic_fc = nn.Linear(self.concat_fc.out_features, 256)
        self.critic_out = nn.Linear(self.critic_fc.out_features, 1)

        self.apply(weights_init)

        # scaling for orthogonal initialization
        # self.conv1.weight.data.mul_(math.sqrt(2))
        # self.conv2.weight.data.mul_(math.sqrt(2))
        # self.conv3.weight.data.mul_(math.sqrt(2))
        # self.fc1.weight.data.mul_(math.sqrt(2))

    def forward(self, x):

        spec_x = F.elu(self.spec_conv1(x[0]))
        spec_x = F.elu(self.spec_conv2(spec_x))
        spec_x = F.elu(self.spec_conv3(spec_x))
        spec_x = F.elu(self.spec_conv4(spec_x))
        spec_x = F.elu(self.spec_conv5(spec_x))
        spec_x = F.elu(self.spec_conv6(spec_x))
        spec_x = F.elu(self.spec_conv7(spec_x))

        spec_x = spec_x.view(-1, num_flat_features(spec_x))  # flatten
        spec_x = F.elu(self.spec_fc(spec_x))

        sheet_x = F.elu(self.sheet_conv1(x[1]))
        sheet_x = F.elu(self.sheet_conv2(sheet_x))
        sheet_x = F.elu(self.sheet_conv3(sheet_x))
        sheet_x = F.elu(self.sheet_conv4(sheet_x))
        sheet_x = F.elu(self.sheet_conv5(sheet_x))
        sheet_x = F.elu(self.sheet_conv6(sheet_x))
        sheet_x = F.elu(self.sheet_conv7(sheet_x))
        sheet_x = F.elu(self.sheet_conv8(sheet_x))

        sheet_x = sheet_x.view(-1, num_flat_features(sheet_x))  # flatten
        sheet_x = F.elu(self.sheet_fc(sheet_x))

        cat_x = torch.cat((spec_x, sheet_x), dim=1)

        cat_x = F.elu(self.concat_fc(cat_x))

        mu = F.elu(self.mu_fc(cat_x))
        mu = self.mu_out(mu)

        sigma = F.elu(self.sigma_fc(cat_x))
        sigma = self.sigma_out(sigma)
        sigma = torch.exp(sigma) + 1e-3

        critic = F.elu(self.critic_fc(cat_x))
        critic = self.critic_out(critic)

        return {'mean': mu, 'std': sigma}, critic


class ScoreFollowingNetMSMDLCHSDeepLSTMContinuous(nn.Module):
    def __init__(self,  n_actions, spec_channels, sheet_channels):
        super(ScoreFollowingNetMSMDLCHSDeepLSTMContinuous, self).__init__()

        # spec part
        self.spec_conv1 = nn.Conv2d(spec_channels, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.spec_conv2 = nn.Conv2d(self.spec_conv1.out_channels, out_channels=32, kernel_size=3, stride=1, padding=0)

        self.spec_conv3 = nn.Conv2d(self.spec_conv2.out_channels, out_channels=64, kernel_size=3, stride=2, padding=0)
        self.spec_conv4 = nn.Conv2d(self.spec_conv3.out_channels, out_channels=64, kernel_size=3, stride=1, padding=0)

        self.spec_conv5 = nn.Conv2d(self.spec_conv4.out_channels, out_channels=64, kernel_size=3, stride=2, padding=0)

        self.spec_conv6 = nn.Conv2d(self.spec_conv5.out_channels, out_channels=96, kernel_size=3, stride=2, padding=0)
        self.spec_conv7 = nn.Conv2d(self.spec_conv6.out_channels, out_channels=96, kernel_size=1, stride=1, padding=0)

        self.spec_fc = nn.Linear(2016, 512)

        # sheet part
        self.sheet_conv1 = nn.Conv2d(sheet_channels, out_channels=32, kernel_size=5, stride=(1, 2), padding=0)
        self.sheet_conv2 = nn.Conv2d(self.sheet_conv1.out_channels, out_channels=32, kernel_size=3, stride=1, padding=0)

        self.sheet_conv3 = nn.Conv2d(self.sheet_conv2.out_channels, out_channels=64, kernel_size=3, stride=2, padding=0)
        self.sheet_conv4 = nn.Conv2d(self.sheet_conv3.out_channels, out_channels=64, kernel_size=3, stride=1, padding=0)

        self.sheet_conv5 = nn.Conv2d(self.sheet_conv4.out_channels, out_channels=64, kernel_size=3, stride=2, padding=0)

        self.sheet_conv6 = nn.Conv2d(self.sheet_conv5.out_channels, out_channels=64, kernel_size=3, stride=2, padding=0)

        self.sheet_conv7 = nn.Conv2d(self.sheet_conv6.out_channels, out_channels=96, kernel_size=3, stride=2, padding=0)
        self.sheet_conv8 = nn.Conv2d(self.sheet_conv7.out_channels, out_channels=96, kernel_size=1, stride=1, padding=0)

        self.sheet_fc = nn.Linear(1728, 512)

        # multi-modal part
        self.concat_fc = nn.Linear(512 + 512, 512)

        self.lstm_size = 256
        self.lstm_layers = 1
        self.lstm = nn.LSTM(self.concat_fc.out_features, self.lstm_size, num_layers=self.lstm_layers)

        self.mu_fc = nn.Linear(self.lstm_size, 256)
        self.mu_out = nn.Linear(self.mu_fc.out_features, 1)

        self.sigma_fc = nn.Linear(self.lstm_size, 256)
        self.sigma_out = nn.Linear(self.sigma_fc.out_features, 1)

        self.critic_fc = nn.Linear(self.lstm_size, 256)
        self.critic_out = nn.Linear(self.critic_fc.out_features, 1)

        self.apply(weights_init)

    def forward(self, x):
        spec_x, (hidden_state, cell_state) = x[0]
        sheet_x, _ = x[1]

        spec_x = F.elu(self.spec_conv1(spec_x))
        spec_x = F.elu(self.spec_conv2(spec_x))
        spec_x = F.elu(self.spec_conv3(spec_x))
        spec_x = F.elu(self.spec_conv4(spec_x))
        spec_x = F.elu(self.spec_conv5(spec_x))
        spec_x = F.elu(self.spec_conv6(spec_x))
        spec_x = F.elu(self.spec_conv7(spec_x))

        spec_x = spec_x.view(-1, num_flat_features(spec_x))  # flatten
        spec_x = F.elu(self.spec_fc(spec_x))

        sheet_x = F.elu(self.sheet_conv1(sheet_x))
        sheet_x = F.elu(self.sheet_conv2(sheet_x))
        sheet_x = F.elu(self.sheet_conv3(sheet_x))
        sheet_x = F.elu(self.sheet_conv4(sheet_x))
        sheet_x = F.elu(self.sheet_conv5(sheet_x))
        sheet_x = F.elu(self.sheet_conv6(sheet_x))
        sheet_x = F.elu(self.sheet_conv7(sheet_x))
        sheet_x = F.elu(self.sheet_conv8(sheet_x))

        sheet_x = sheet_x.view(-1, num_flat_features(sheet_x))  # flatten
        sheet_x = F.elu(self.sheet_fc(sheet_x))

        cat_x = torch.cat((spec_x, sheet_x), dim=1)

        cat_x = F.elu(self.concat_fc(cat_x))

        cat_x, (hidden_state_n, cell_state_n) = self.lstm(cat_x.unsqueeze(0), (hidden_state, cell_state))

        # only one time step
        cat_x = cat_x[0]

        critic = F.elu(self.critic_fc(cat_x))
        critic = self.critic_out(critic)

        mu = F.relu(self.mu_fc(cat_x))
        sigma = F.relu(self.sigma_fc(cat_x))

        mu = self.mu_out(mu)
        sigma = torch.exp(self.sigma_out(sigma)) + 1e-3

        return {'mean': mu, 'std': sigma}, critic, (hidden_state_n, cell_state_n)


class ScoreFollowingNetMSMDLCHSDeepDoContinuous(nn.Module):
    def __init__(self,  n_actions, spec_channels, sheet_channels):
        super(ScoreFollowingNetMSMDLCHSDeepDoContinuous, self).__init__()

        # spec part
        self.spec_conv1 = nn.Conv2d(spec_channels, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.spec_conv2 = nn.Conv2d(self.spec_conv1.out_channels, out_channels=32, kernel_size=3, stride=1, padding=0)

        self.spec_conv3 = nn.Conv2d(self.spec_conv2.out_channels, out_channels=64, kernel_size=3, stride=2, padding=0)
        self.spec_conv4 = nn.Conv2d(self.spec_conv3.out_channels, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.spec_do4 = nn.Dropout(p=0.2)

        self.spec_conv5 = nn.Conv2d(self.spec_conv4.out_channels, out_channels=64, kernel_size=3, stride=2, padding=0)

        self.spec_conv6 = nn.Conv2d(self.spec_conv5.out_channels, out_channels=96, kernel_size=3, stride=2, padding=0)
        self.spec_conv7 = nn.Conv2d(self.spec_conv6.out_channels, out_channels=96, kernel_size=1, stride=1, padding=0)
        self.spec_do7 = nn.Dropout(p=0.2)

        self.spec_fc = nn.Linear(2016, 512)

        # sheet part
        self.sheet_conv1 = nn.Conv2d(sheet_channels, out_channels=32, kernel_size=5, stride=(1, 2), padding=0)
        self.sheet_conv2 = nn.Conv2d(self.sheet_conv1.out_channels, out_channels=32, kernel_size=3, stride=1, padding=0)

        self.sheet_conv3 = nn.Conv2d(self.sheet_conv2.out_channels, out_channels=64, kernel_size=3, stride=2, padding=0)
        self.sheet_conv4 = nn.Conv2d(self.sheet_conv3.out_channels, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.sheet_do4 = nn.Dropout(p=0.2)

        self.sheet_conv5 = nn.Conv2d(self.sheet_conv4.out_channels, out_channels=64, kernel_size=3, stride=2, padding=0)

        self.sheet_conv6 = nn.Conv2d(self.sheet_conv5.out_channels, out_channels=64, kernel_size=3, stride=2, padding=0)
        self.sheet_do6 = nn.Dropout(p=0.2)

        self.sheet_conv7 = nn.Conv2d(self.sheet_conv6.out_channels, out_channels=96, kernel_size=3, stride=2, padding=0)
        self.sheet_conv8 = nn.Conv2d(self.sheet_conv7.out_channels, out_channels=96, kernel_size=1, stride=1, padding=0)
        self.sheet_do8 = nn.Dropout(p=0.2)

        self.sheet_fc = nn.Linear(1728, 512)

        # multi-modal part
        self.concat_fc = nn.Linear(512 + 512, 512)

        self.mu_fc = nn.Linear(self.concat_fc.out_features, 256)
        self.mu_fc_do = nn.Dropout(p=0.2)
        self.mu_out = nn.Linear(self.mu_fc.out_features, 1)

        self.sigma_fc = nn.Linear(self.concat_fc.out_features, 256)
        self.sigma_fc_do = nn.Dropout(p=0.2)
        self.sigma_out = nn.Linear(self.sigma_fc.out_features, 1)

        self.critic_fc = nn.Linear(self.concat_fc.out_features, 256)
        self.critic_fc_do = nn.Dropout(p=0.2)
        self.critic_out = nn.Linear(self.critic_fc.out_features, 1)

        self.apply(weights_init)

        # scaling for orthogonal initialization
        # self.conv1.weight.data.mul_(math.sqrt(2))
        # self.conv2.weight.data.mul_(math.sqrt(2))
        # self.conv3.weight.data.mul_(math.sqrt(2))
        # self.fc1.weight.data.mul_(math.sqrt(2))

    def forward(self, x):

        spec_x = F.elu(self.spec_conv1(x[0]))
        spec_x = F.elu(self.spec_conv2(spec_x))
        spec_x = F.elu(self.spec_conv3(spec_x))
        spec_x = F.elu(self.spec_conv4(spec_x))
        spec_x = self.spec_do4(spec_x)
        spec_x = F.elu(self.spec_conv5(spec_x))
        spec_x = F.elu(self.spec_conv6(spec_x))
        spec_x = F.elu(self.spec_conv7(spec_x))
        spec_x = self.spec_do7(spec_x)

        spec_x = spec_x.view(-1, num_flat_features(spec_x))  # flatten
        spec_x = F.elu(self.spec_fc(spec_x))

        sheet_x = F.elu(self.sheet_conv1(x[1]))
        sheet_x = F.elu(self.sheet_conv2(sheet_x))
        sheet_x = F.elu(self.sheet_conv3(sheet_x))
        sheet_x = F.elu(self.sheet_conv4(sheet_x))
        sheet_x = self.sheet_do4(sheet_x)
        sheet_x = F.elu(self.sheet_conv5(sheet_x))
        sheet_x = F.elu(self.sheet_conv6(sheet_x))
        sheet_x = self.sheet_do6(sheet_x)
        sheet_x = F.elu(self.sheet_conv7(sheet_x))
        sheet_x = F.elu(self.sheet_conv8(sheet_x))
        sheet_x = self.sheet_do8(sheet_x)

        sheet_x = sheet_x.view(-1, num_flat_features(sheet_x))  # flatten
        sheet_x = F.elu(self.sheet_fc(sheet_x))

        cat_x = torch.cat((spec_x, sheet_x), dim=1)

        cat_x = F.elu(self.concat_fc(cat_x))

        mu = F.elu(self.mu_fc(cat_x))
        mu = self.mu_fc_do(mu)
        mu = self.mu_out(mu)

        sigma = F.elu(self.sigma_fc(cat_x))
        sigma = self.sigma_fc_do(sigma)
        sigma = self.sigma_out(sigma)
        sigma = torch.exp(sigma) + 1e-3

        critic = F.elu(self.critic_fc(cat_x))
        critic = self.critic_fc_do(critic)
        critic = self.critic_out(critic)

        return {'mean': mu, 'std': sigma}, critic


class BaselineNet(nn.Module):
    def __init__(self, n_actions, spec_channels, sheet_channels):
        super(BaselineNet, self).__init__()

        self.spec_conv1 = nn.Conv2d(spec_channels, out_channels=16, kernel_size=(4, 4), stride=2)
        self.spec_conv2 = nn.Conv2d(self.spec_conv1.out_channels, out_channels=32, kernel_size=(3, 3), stride=2)
        self.spec_conv3 = nn.Conv2d(self.spec_conv2.out_channels, out_channels=64, kernel_size=(3, 3), stride=1)

        self.sheet_conv1 = nn.Conv2d(sheet_channels, out_channels=16, kernel_size=(4, 8), stride=2)
        self.sheet_conv2 = nn.Conv2d(self.sheet_conv1.out_channels, out_channels=32, kernel_size=(3, 3), stride=(1, 2))
        self.sheet_conv3 = nn.Conv2d(self.sheet_conv2.out_channels, out_channels=32, kernel_size=(3, 3), stride=(1, 2))
        self.sheet_conv4 = nn.Conv2d(self.sheet_conv2.out_channels, out_channels=32, kernel_size=(4, 4), stride=2)

        self.concat_fc = nn.Linear(self.spec_conv3.out_channels * 6 * 2 + self.sheet_conv4.out_channels * 6 * 7, 256)

        self.policy_fc = nn.Linear(self.concat_fc.out_features, 256)
        self.policy_out = nn.Linear(self.policy_fc.out_features, n_actions)

        self.apply(weights_init)

        # scaling for orthogonal initialization
        # self.conv1.weight.data.mul_(math.sqrt(2))
        # self.conv2.weight.data.mul_(math.sqrt(2))
        # self.conv3.weight.data.mul_(math.sqrt(2))
        # self.fc1.weight.data.mul_(math.sqrt(2))

    def forward(self, x):

        spec_x = F.elu(self.spec_conv1(x[0]))
        spec_x = F.elu(self.spec_conv2(spec_x))
        spec_x = F.elu(self.spec_conv3(spec_x))

        spec_x = spec_x.view(-1, num_flat_features(spec_x)) # flatten

        sheet_x = F.elu(self.sheet_conv1(x[1]))
        sheet_x = F.elu(self.sheet_conv2(sheet_x))
        sheet_x = F.elu(self.sheet_conv3(sheet_x))
        sheet_x = F.elu(self.sheet_conv4(sheet_x))

        sheet_x = sheet_x.view(-1, num_flat_features(sheet_x)) # flatten

        cat_x = torch.cat((spec_x, sheet_x), dim=1)

        cat_x = F.elu(self.concat_fc(cat_x))

        policy = F.elu(self.policy_fc(cat_x))
        policy = self.policy_out(policy)

        return policy
