
from __future__ import print_function

import matplotlib.pyplot as plt
import seaborn as sns

import os
import yaml
import pickle as pkl
import numpy as np
from tqdm import tqdm

import theano
import theano.tensor as T

import lasagne
from lasagne.layers import InputLayer, DenseLayer, FlattenLayer, ConcatLayer
from lasagne.nonlinearities import elu, identity, softmax

from lasagne_wrapper.utils import print_net_architecture

from score_following_game.data_processing.data_pool import get_data_pool, load_game_config
from score_following_game.environment.score_following_env import ScoreFollowingEnv
from score_following_game.environment.env_wrappers import ResizeSizeObservations, SpecDifference, PrepareForNet, SheetDifference, InvertSheet, ConvertToFloat

from lasagne.layers import dnn
Conv2DLayer = dnn.Conv2DDNNLayer
MaxPool2DLayer = dnn.MaxPool2DDNNLayer
batch_norm = dnn.batch_norm_dnn


def batch_norm(net):
    return net


def get_linear(start_at, ini_lr, decrease_epochs):
    """ linear learn rate schedule"""

    def update(lr, epoch):
        if epoch < start_at:
            return np.float32(lr)
        else:
            k = ini_lr / decrease_epochs
            return np.float32(np.max([0.0, lr - k]))

    return update


def _create_networks(n_actions, spec_shape, sheet_shape, show_nets=False):
    """ Build policy network """

    l_in_spec = InputLayer(shape=[None, ] + spec_shape)
    l_in_sheet = InputLayer(shape=[None, ] + sheet_shape)

    net_spec = l_in_spec
    net_spec = Conv2DLayer(net_spec, num_filters=16, filter_size=4, stride=2, nonlinearity=elu)
    net_spec = Conv2DLayer(net_spec, num_filters=32, filter_size=3, stride=2, nonlinearity=elu)
    net_spec = Conv2DLayer(net_spec, num_filters=64, filter_size=3, stride=1, nonlinearity=elu)
    net_spec = FlattenLayer(net_spec)

    net_sheet = l_in_sheet
    net_sheet = Conv2DLayer(net_sheet, num_filters=16, filter_size=(4, 8), stride=2, nonlinearity=elu)
    net_sheet = Conv2DLayer(net_sheet, num_filters=32, filter_size=3, stride=(1, 2), nonlinearity=elu)
    net_sheet = Conv2DLayer(net_sheet, num_filters=32, filter_size=3, stride=(1, 2), nonlinearity=elu)
    net_sheet = Conv2DLayer(net_sheet, num_filters=32, filter_size=4, stride=2, nonlinearity=elu)
    net_sheet = FlattenLayer(net_sheet)

    net = ConcatLayer((net_spec, net_sheet), axis=1)
    net = DenseLayer(net, num_units=256, nonlinearity=elu)

    policy_net = DenseLayer(net, num_units=256, nonlinearity=elu)
    policy_net = DenseLayer(policy_net, num_units=n_actions, nonlinearity=softmax)

    value_net = DenseLayer(net, num_units=256, nonlinearity=elu)
    value_net = DenseLayer(value_net, num_units=1, nonlinearity=identity)

    if show_nets:
        print_net_architecture(policy_net, tag="Policy Network", detailed=False)
        print_net_architecture(value_net, tag="Value Network", detailed=False)

    return policy_net, value_net


def create_networks(n_actions, spec_shape, sheet_shape, show_nets=False):
    """ Build policy network """
    """ Build policy network """

    l_in_spec = InputLayer(shape=[None, ] + spec_shape)
    l_in_sheet = InputLayer(shape=[None, ] + sheet_shape)

    net_spec = l_in_spec
    net_spec = Conv2DLayer(net_spec, num_filters=16, filter_size=3, stride=1, nonlinearity=elu)
    net_spec = Conv2DLayer(net_spec, num_filters=32, filter_size=3, stride=2, nonlinearity=elu)
    net_spec = Conv2DLayer(net_spec, num_filters=64, filter_size=3, stride=2, nonlinearity=elu)
    net_spec = Conv2DLayer(net_spec, num_filters=128, filter_size=3, stride=1, nonlinearity=elu)
    net_spec = FlattenLayer(net_spec)

    net_sheet = l_in_sheet
    net_sheet = Conv2DLayer(net_sheet, num_filters=16, filter_size=3, stride=1, nonlinearity=elu)
    net_sheet = Conv2DLayer(net_sheet, num_filters=32, filter_size=(4, 8), stride=2, nonlinearity=elu)
    net_sheet = Conv2DLayer(net_sheet, num_filters=32, filter_size=3, stride=(1, 2), nonlinearity=elu)
    net_sheet = Conv2DLayer(net_sheet, num_filters=64, filter_size=3, stride=(1, 2), nonlinearity=elu)
    net_sheet = Conv2DLayer(net_sheet, num_filters=128, filter_size=4, stride=2, nonlinearity=elu)
    net_sheet = FlattenLayer(net_sheet)

    net = ConcatLayer((net_spec, net_sheet), axis=1)
    net = DenseLayer(net, num_units=256, nonlinearity=elu)

    policy_net = DenseLayer(net, num_units=256, nonlinearity=elu)
    policy_net = DenseLayer(policy_net, num_units=n_actions, nonlinearity=softmax)

    value_net = DenseLayer(net, num_units=256, nonlinearity=elu)
    value_net = DenseLayer(value_net, num_units=1, nonlinearity=identity)

    if show_nets:
        print_net_architecture(policy_net, tag="Policy Network", detailed=False)
        print_net_architecture(value_net, tag="Value Network", detailed=False)

    return policy_net, value_net


def get_input_vars(net):
    """ get input variables """
    input_vars = []
    for l in lasagne.layers.get_all_layers(net):
        if isinstance(l, InputLayer):
            input_vars.append(l.input_var)
    return input_vars


def policy_gradient(net, lr):
    """ computation of policy gradient """

    # init place holders
    actions = T.matrix("actions")
    advantages = T.vector("advantages")

    # get input variables
    input_vars = get_input_vars(net)

    # get network output (action probabilities)
    probabilities = lasagne.layers.get_output(net, deterministic=False)

    # sum of probabilities of "good actions"
    good_probabilities = T.sum(probabilities * actions, axis=1)

    # weight log probabilities of actions with advantages
    eligibility = T.log(good_probabilities) * advantages

    # compile loss
    loss = -T.mean(eligibility)

    # get params to optimize
    params = lasagne.layers.get_all_params(net, trainable=True)

    # define update rule
    updates = lasagne.updates.adam(loss, params, learning_rate=lr)

    # compile theano functions
    compute_pl_probs = theano.function(input_vars, probabilities)
    train_step = theano.function(input_vars + [advantages, actions], loss, updates=updates)

    return compute_pl_probs, train_step


def value_gradient(net, lr):
    """ computation of value gradient """

    # init place holders
    newvals = T.matrix("newvals")

    # get input variables
    input_vars = get_input_vars(net)

    # get network output
    calculated = lasagne.layers.get_output(net, deterministic=False)

    # minimize difference between network output and measured ...
    diffs = calculated - newvals
    loss = T.mean(diffs**2)

    # get params to optimize
    params = lasagne.layers.get_all_params(net, trainable=True)

    # define update rule
    updates = lasagne.updates.adam(loss, params, learning_rate=lr)

    # compile theano functions
    compute_value = theano.function(input_vars, calculated)
    train_step = theano.function(input_vars + [newvals], loss, updates=updates)

    return compute_value, train_step


# helper vector for faster discounting
gamma = 0.97
gamma_vec = np.ones(10000, dtype=np.float32)
for t in range(10000):
    gamma_vec[t] = gamma**t


def run_episode(env, policy_grad, value_grad, n_actions, max_steps,
                std_adv=True, discounting=True, use_baseline=True,
                batch_size=None):
    """ run episode """

    # extract outputs
    compute_pl_probs, pl_updates = policy_grad
    compute_value, vl_updates = value_grad

    # reset environment and get initial observation
    observation = env.reset()

    # init total reward
    total_reward = 0

    # initialize state arrays
    spec_states, sheet_states = [], []
    actions = []
    transitions = []

    # generate episode
    for _ in range(max_steps):

        # prepare observation
        spec, sheet = observation

        # compute policy prediction
        action_probabilities = compute_pl_probs(spec, sheet)[0]

        # draw a random action by sampling from the policy prediction
        # print()
        # print(spec.min(), spec.max(), spec.mean())
        # print(sheet.min(), sheet.max(), sheet.mean())
        # print(np.around(action_probabilities, 3))
        action = np.random.choice([0, 1, 2], p=action_probabilities)

        # record state
        spec_states.append(spec)
        sheet_states.append(sheet)

        # record one hot encoded action
        action_blank = np.zeros(n_actions)
        action_blank[action] = 1
        actions.append(action_blank)

        # take the action in the environment
        old_observation = observation
        observation, reward, done, info = env.step(action)

        # record the transition
        transitions.append((old_observation, action, reward))
        total_reward += reward

        # check if environment reported terminal state
        if done:
            break

    # prepare rewards
    rewards = np.array([r for _, _, r in transitions], dtype=np.float32)

    # iterate collected transitions
    discounted_returns = np.zeros((len(transitions), ), dtype=np.float32)
    for t, trans in enumerate(transitions):

        # fast discounting
        if discounting:
            n_steps = len(rewards[t:])
            Gt = np.sum(rewards[t:] * gamma_vec[:n_steps])

        # no discounting
        else:
            Gt = np.sum(rewards[t:])

        discounted_returns[t] = Gt

    # convert to float32
    discounted_returns = discounted_returns.astype(np.float32)

    # prepare network input
    spec_states = np.vstack(spec_states)
    sheet_states = np.vstack(sheet_states)
    actions = np.asarray(actions, dtype=np.float32)

    # compute state values
    values = compute_value(spec_states, sheet_states)[0, 0] if use_baseline else 0.0

    # compute advantages
    advantages = discounted_returns - values

    # standardize advantages to reduce variance
    if std_adv:
        advantages -= np.mean(advantages)
        advantages /= np.std(advantages)

    # prepare discounted returns
    discounted_returns = np.expand_dims(discounted_returns, axis=1)

    # update network parameters
    if batch_size is None:
        vl_loss = vl_updates(spec_states, sheet_states, discounted_returns) if use_baseline else 0.0
        pl_loss = pl_updates(spec_states, sheet_states, advantages, actions)
    else:
        vl_loss, pl_loss = 0, 0
        n_batches = int(np.ceil(float(spec_states.shape[0]) / batch_size))
        for ib in range(n_batches):
            i0 = ib * batch_size
            i1 = min(i0 + batch_size, spec_states.shape[0])
            vl_loss += vl_updates(spec_states[i0:i1], sheet_states[i0:i1], discounted_returns[i0:i1]) if use_baseline else 0.0
            pl_loss += pl_updates(spec_states[i0:i1], sheet_states[i0:i1], advantages[i0:i1], actions[i0:i1])
        vl_loss /= n_batches
        pl_loss /= n_batches

    return total_reward, pl_loss, vl_loss


def make_env(rl_pool, config, render_mode=None):

    # initialize environment
    env = ScoreFollowingEnv(rl_pool, config, render_mode=render_mode)

    # apply wrapper to env
    env = ResizeSizeObservations(env, config["spec_factor"], config["sheet_factor"])
    env = ConvertToFloat(env)
    env = InvertSheet(env)
    env = SpecDifference(env)
    env = SheetDifference(env)
    env = PrepareForNet(env)

    return env


if __name__ == '__main__':
    """ main """

    # add argument parser
    import argparse
    parser = argparse.ArgumentParser(description='Train multi-modality model.')
    parser.add_argument('--lrpl', help='learn rate of policy network.', type=np.float32, default=0.0001)
    parser.add_argument('--lrvl', help='learn rate of value network.', type=np.float32, default=0.001)
    parser.add_argument('--std_adv', help='standardize advantages.', action='store_true')
    parser.add_argument('--discounting', help='use discounting.', action='store_true')
    parser.add_argument('--dump_params', help='dump network parameters.', action='store_true')
    parser.add_argument('--use_baseline', help='dump network parameters.', action='store_true')
    parser.add_argument('--data_set', help='path to dataset.', type=str, default="test_sample_clip")
    parser.add_argument('--resume', help='resume training from last point.', action='store_true')
    parser.add_argument('--game_config', help='path to game config file.', type=str, default=None)
    parser.add_argument('--batch_size', help='batch size used for updates.', type=int, default=None)
    parser.add_argument('--log_root', help='tensorboard log directory.', type=str,
                        default="/home/matthias/experiments/score_following_game")
    args = parser.parse_args()

    # load game config
    with open(args.game_config, "rb") as fp:
        config = yaml.load(fp)

    # string for learning configuration
    values = (args.lrpl, args.lrvl, args.std_adv, args.discounting, args.use_baseline, os.path.basename(args.data_set))
    setting = "lrpl(%.5f)_lrvl(%.5f)_std_adv(%d)_discounting(%d)_bl(%d)_%s" % values

    params_file = os.path.join(args.log_root, "params_%s.pkl" % setting)
    log_plot_file = os.path.join(args.log_root, "log_%s.png" % setting)
    log_file = os.path.join(args.log_root, "log_%s.pkl" % setting)

    epoch_length = 200   # episodes
    max_steps = 100000   # maximum time steps per episode
    max_epochs = 99999   # maximum number of epochs

    # init learning rate schedule
    lr_pl = np.float32(args.lrpl)
    lr_vl = np.float32(args.lrvl)
    # lr_pl_update = get_linear(max_epochs // 2, lr_pl, max_epochs - max_epochs // 2)
    # lr_vl_update = get_linear(max_epochs // 2, lr_vl, max_epochs - max_epochs // 2)
    # lr_pl_update = get_constant()
    # lr_vl_update = get_constant()

    # # todo: remove this
    # lr_pl = np.float32(0.1 * lr_pl)
    # lr_vl = np.float32(0.1 * lr_vl)

    # load game config
    config = load_game_config(args.game_config)

    # init data pool
    rl_pool = get_data_pool(config, directory=args.data_set)

    # initialize environment
    env = make_env(rl_pool, config, render_mode=None)

    # get action space
    n_actions = 3

    # compile networks
    pl_net, vl_net = create_networks(n_actions, config["spec_shape"], config["sheet_shape"])

    # init pg
    # lr_pl_theano = theano.shared(lr_pl)
    policy_grad = policy_gradient(pl_net, lr=lr_pl)

    # init vg
    # lr_vl_theano = theano.shared(lr_vl)
    value_grad = value_gradient(vl_net, lr=lr_vl)

    # resume training
    if args.resume:

        with open(params_file, "rb") as fp:
            params = pkl.load(fp)
        lasagne.layers.set_all_param_values([pl_net, vl_net], params)

        with open(log_file, "rb") as fp:
            rewards, pl_losses, vl_losses, max_trck_ratio, trck_ratios = pkl.load(fp)

        last_episode = len(pl_losses)

    # init everything
    else:
        rewards = np.zeros((epoch_length, 0))
        pl_losses = []
        vl_losses = []
        trck_ratios = []
        max_trck_ratio = -np.inf
        last_episode = 0

    # iterate episodes
    for epoch in range(last_episode, max_epochs):

        # # update learning rates
        # lr_pl = lr_pl_update(lr_pl, epoch)
        # lr_pl_theano.set_value(lr_pl)
        # lr_vl = lr_vl_update(lr_pl, epoch)
        # lr_vl_theano.set_value(lr_vl)

        r = []
        pl_l, vl_l = 0.0, 0.0
        for iteration in tqdm(range(epoch_length), ncols=100):

            # run episode
            reward, pl_loss, vl_loss = run_episode(env, policy_grad, value_grad, n_actions, max_steps,
                                                   std_adv=args.std_adv, discounting=args.discounting,
                                                   use_baseline=args.use_baseline, batch_size=args.batch_size)
            r.append(reward)
            pl_l += pl_loss
            vl_l += vl_loss

        # book keeping
        new_col = np.array(r).reshape((epoch_length, 1))
        rewards = np.hstack((rewards, new_col))
        pl_losses.append(pl_l / epoch_length)
        vl_losses.append(vl_l / epoch_length)
        rwd = np.mean(new_col)

        print("Epoch %04d, Avg-Reward: %.1f, pl_loss: %.5f, vl_loss: %.5f" % (epoch, rwd, pl_losses[-1], vl_losses[-1]))

        # TODO: evaluate agent on validation set
        from score_following_game.evaluation.evaluation import Evaluator
        from score_following_game.first_try_test import Agent
        from score_following_game.data_processing.data_pool import get_data_pools

        # init agent
        params = lasagne.layers.get_all_param_values([pl_net, vl_net])
        agent = Agent(model_params=params,
                      spec_shape=config["spec_shape"],
                      sheet_shape=config["sheet_shape"])

        evaluation_pools = get_data_pools(config, directory=args.data_set.replace("train", "valid"))

        evaluator = Evaluator(make_env, evaluation_pools, config, render_mode=None)
        stats = evaluator.evaluate(agent, log_writer=None, log_step=0)

        # dump network parameters
        if args.dump_params and (stats["global_tracking_ratio"] > max_trck_ratio):
            max_trck_ratio = stats["global_tracking_ratio"]
            params = lasagne.layers.get_all_param_values([pl_net, vl_net])

            with open(params_file, "wb") as fp:
                pkl.dump(params, fp, -1)

            with open(log_file, "wb") as fp:
                log_data = [rewards, pl_losses, vl_losses, max_trck_ratio, trck_ratios]
                pkl.dump(log_data, fp, -1)

        if epoch < 1:
            continue

        print('global_tracking_ratio   %.2f' % stats["global_tracking_ratio"])
        print('tracked_until_end_ratio %.2f' % stats["tracked_until_end_ratio"])
        trck_ratios.append(stats["global_tracking_ratio"])

        # plot log
        plt.figure('Monitor', figsize=(10, 10))
        plt.clf()

        plt.subplot(2, 2, 1)
        sns.tsplot(rewards)
        plt.title("Reward")
        plt.grid('on')

        plt.subplot(2, 2, 2)
        plt.plot(trck_ratios)
        plt.title("tracking ratio")
        plt.grid('on')

        plt.subplot(2, 2, 3)
        plt.plot(pl_losses, '-')
        plt.title("Policy Loss")
        plt.grid('on')

        plt.subplot(2, 2, 4)
        plt.plot(vl_losses, '-')
        plt.title("Value Loss")
        plt.grid('on')

        plt.savefig(log_plot_file)
