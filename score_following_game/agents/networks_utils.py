import importlib
import torch.nn as nn
import torch


def get_network(file, network_name, n_actions, shapes=dict(), activation=None):
    """
    Compile network by name

    :param network_name:
    :return:
    """
    package = importlib.import_module("score_following_game.agents.{}".format(file))
    constructor = getattr(package, network_name)

    if activation is None:
        network = constructor(n_actions, **shapes)
    else:
        network = constructor(n_actions, activation=activation, **shapes)

    return network


def weights_init(m):
    if type(m) == torch.nn.modules.linear.Linear or type(m) == torch.nn.modules.conv.Conv2d:
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)


def num_flat_features(x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features

