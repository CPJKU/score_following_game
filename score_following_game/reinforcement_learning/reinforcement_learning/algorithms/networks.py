import torch
import math
import torch.nn as nn
import torch.nn.functional as F


class PolicyNet(nn.Module):

    def __init__(self, input_shape, n_actions, hidden_size=32):
        super().__init__()
        self.fc1 = nn.Linear(input_shape, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, n_actions)

        self.apply(weights_init)

    def forward(self, x):
        x = F.tanh(self.fc1(x[0]))
        x = F.tanh(self.fc2(x))
        x = self.fc3(x)
        return x


class ValueNet(nn.Module):
    def __init__(self, input_shape, hidden_size=32):
        super().__init__()
        self.fc1 = nn.Linear(input_shape, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

        self.apply(weights_init)

    def forward(self, x):
        x = F.elu(self.fc1(x[0]))
        x = F.elu(self.fc2(x))
        x = self.fc3(x)
        return x


class A2CFFNet(nn.Module):

    def __init__(self,  input_shape, n_actions):
        super().__init__()

        self.fc1 = nn.Linear(input_shape, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 32)

        self.policy_fc1 = nn.Linear(32, 16)
        self.critic_fc1 = nn.Linear(32, 16)

        self.policy_out = nn.Linear(16, n_actions)
        self.critic_out = nn.Linear(16, 1)

        self.apply(weights_init)

        # scaling for orthogonal initialization
        # self.conv1.weight.data.mul_(math.sqrt(2))
        # self.conv2.weight.data.mul_(math.sqrt(2))
        # self.conv3.weight.data.mul_(math.sqrt(2))
        # self.fc1.weight.data.mul_(math.sqrt(2))

    def forward(self, x):

        x = F.elu(self.fc1(x[0]))
        x = F.elu(self.fc2(x))
        x = F.elu(self.fc3(x))

        policy = F.tanh(self.policy_fc1(x))
        critic = F.relu(self.critic_fc1(x))

        # policy = F.softmax(self.policy_out(x), dim=-1)
        policy = self.policy_out(policy)
        critic = self.critic_out(critic)

        # if we use softmax and the logarithmic softmax PyTorch recommends  to use the log_softmax function
        # therefore I return the linear output of the policy and apply the softmax/log softmax where it is necessary
        return policy, critic


class ConvPolicyNet(nn.Module):

    def __init__(self, n_actions, input_channels=1):
        super().__init__()

        self.conv1 = nn.Conv2d(input_channels, 16, (8, 8), 4)
        self.conv2 = nn.Conv2d(16, 32, (4, 4), 2)
        self.fc1 = nn.Linear(32 * 18 * 22, 128)

        self.policy_out = nn.Linear(128, n_actions)

        # self.apply(weights_init)

    def forward(self, x):

        x = F.relu(self.conv1(x[0]))
        x = F.relu(self.conv2(x))
        x = x.view(-1, num_flat_features(x))  # flatten x
        x = F.relu(self.fc1(x))
        policy = self.policy_out(x)

        return policy


class ConvValueNet(nn.Module):

    def __init__(self, input_channels=1):
        super().__init__()

        self.conv1 = nn.Conv2d(input_channels, 16, (8, 8), 4)
        self.conv2 = nn.Conv2d(16, 32, (4, 4), 2)
        self.fc1 = nn.Linear(32 * 18 * 22, 128)

        self.value_out = nn.Linear(128, 1)

        # self.apply(weights_init)

    def forward(self, x):

        x = F.relu(self.conv1(x[0]))
        x = F.relu(self.conv2(x))
        x = x.view(-1, num_flat_features(x))  # flatten x
        x = F.relu(self.fc1(x))
        value = self.value_out(x)

        return value


class PongNet(nn.Module):

    def __init__(self,  n_actions, input_channels=1):
        super().__init__()

        self.conv1 = nn.Conv2d(input_channels, 32, (8, 8), 4)
        self.conv2 = nn.Conv2d(32, 64, (4, 4), 2)
        self.conv3 = nn.Conv2d(64, 64, (3, 3), 1)
        # self.fc1 = nn.Linear(64 * 6 * 6, 512) # if 80x80
        self.fc1 = nn.Linear(64 * 7 * 7, 512) # if 84x84

        self.policy_out = nn.Linear(512, n_actions)
        self.critic_out = nn.Linear(512, 1)

        self.apply(weights_init)

        # scaling for orthogonal initialization
        # self.conv1.weight.data.mul_(math.sqrt(2))
        # self.conv2.weight.data.mul_(math.sqrt(2))
        # self.conv3.weight.data.mul_(math.sqrt(2))
        # self.fc1.weight.data.mul_(math.sqrt(2))

    def forward(self, x):

        x = F.relu(self.conv1(x[0]/255.0))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, num_flat_features(x))    # flatten x
        x = F.relu(self.fc1(x))
        # policy = F.softmax(self.policy_out(x), dim=-1)
        policy = self.policy_out(x)
        critic = self.critic_out(x)

        # if we use softmax and the logarithmic softmax PyTorch recommends  to use the log_softmax function
        # therefore I return the linear output of the policy and apply the softmax/log softmax where it is necessary
        return policy, critic


class MuSigmaNet(nn.Module):
    def __init__(self, input_shape, hidden_size=64):
        super().__init__()
        self.fc1 = nn.Linear(input_shape, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        self.mu_out = nn.Linear(hidden_size, 1)
        self.sigma_out = nn.Linear(hidden_size, 1)

        self.apply(weights_init)

    def forward(self, x):
        x = F.elu(self.fc1(x[0]))
        x = F.elu(self.fc2(x))
        # mu = F.relu(self.fc1(x))
        # sigma = F.relu(self.fc2(x))

        # mu = self.mu_out(mu)
        # sigma = torch.exp(self.sigma_out(sigma))
        mu = self.mu_out(x)
        sigma = F.softplus(self.sigma_out(x)) + 1e-4
        return {'mean': mu, 'std': sigma}


class BetaDistrNet(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.fc1 = nn.Linear(input_shape, 128)
        # self.fc2 = nn.Linear(input_shape, 64)

        self.alpha_out = nn.Linear(128, 1)
        self.beta_out = nn.Linear(128, 1)

        self.apply(weights_init)

    def forward(self, x):
        x = F.relu(self.fc1(x[0]))

        alpha = F.softplus(self.alpha_out(x)) + 1
        beta = F.softplus(self.beta_out(x)) + 1

        return {'concentration1': alpha, 'concentration2': beta}


class ContinuousValueNet(nn.Module):
    def __init__(self, input_shape):
        super().__init__()

        self.fc1 = nn.Linear(input_shape, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 100)

        self.mu_sigma_fc1 = nn.Linear(100, 16)
        self.value_fc1 = nn.Linear(100, 16)

        self.mu_out = nn.Linear(16, 1)
        self.sigma_out = nn.Linear(16, 1)
        self.value_out = nn.Linear(16, 1)

        self.apply(weights_init)

    def forward(self, x):

        x = F.elu(self.fc1(x[0]))
        x = F.elu(self.fc2(x))
        x = F.elu(self.fc3(x))

        mu_sigma = F.relu(self.mu_sigma_fc1(x))
        mu = self.mu_out(mu_sigma)
        # sigma = torch.exp(self.sigma_out(mu_sigma))
        sigma = F.softplus(self.sigma_out(mu_sigma)) + 1e-4

        value = F.relu(self.value_fc1(x))
        value = self.value_out(value)

        return {'mean': mu, 'std': sigma}, value


class ContinuousPixelValueNet(nn.Module):
    def __init__(self,  input_channels=1):
        super().__init__()

        self.conv1 = nn.Conv2d(input_channels, 32, (8, 8), 4)
        self.conv2 = nn.Conv2d(32, 64, (4, 4), 2)
        self.conv3 = nn.Conv2d(64, 64, (3, 3), 1)

        self.fc1 = nn.Linear(64 * 8 *8, 128)


        self.mu_sigma_fc1 = nn.Linear(128, 32)
        self.value_fc1 = nn.Linear(128, 32)

        self.mu_out = nn.Linear(32, 1)
        self.sigma_out = nn.Linear(32, 1)
        self.value_out = nn.Linear(32, 1)

        self.apply(weights_init)

    def forward(self, x):

        x = F.relu(self.conv1(x[0]/255.0))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(-1, num_flat_features(x))  # flatten x

        x = F.relu(self.fc1(x))

        mu_sigma = F.relu(self.mu_sigma_fc1(x))
        mu = self.mu_out(mu_sigma)
        # sigma = torch.exp(self.sigma_out(mu_sigma))
        sigma = F.softplus(self.sigma_out(mu_sigma)) + 1e-4

        value = F.relu(self.value_fc1(x))
        value = self.value_out(value)

        return {'mean': mu, 'std': sigma}, value


# helper functions

def num_flat_features(x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


def weights_xavier_uniform(layer):
    torch.nn.init.xavier_uniform(layer.weight)
    layer.bias.data.zero_()


def weights_xavier_normal(layer):
    torch.nn.init.xavier_normal(layer.weight)
    layer.bias.data.zero_()


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        nn.init.orthogonal(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)



