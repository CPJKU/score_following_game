
import numpy as np
import torch.optim as optim

from ast import literal_eval as make_tuple


def cast_optim_params(optim_params):
    """

    :param optim_params:
    :return:
    """
    for k in optim_params.keys():

        # check if argument is a parameter tuple
        if isinstance(optim_params[k], str) and '(' in optim_params[k]:
            optim_params[k] = make_tuple(optim_params[k])
        else:
            try:
                optim_params[k] = np.float(optim_params[k])
            except:
                pass

    return optim_params


def get_optimizer(optimizer_name, params, **kwargs):
    """
    Compile pytorch optimizer

    :param optimizer_name:
    :param params:
    :param kwargs:
    :return:
    """
    constructor = getattr(optim, optimizer_name)
    optimizer = constructor(params, **kwargs)
    return optimizer
