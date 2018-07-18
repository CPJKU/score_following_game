from __future__ import print_function

# this is required as we included our reinforcment_learning package
# as a submodule for convenience
from score_following_game.utils import init_rl_imports
init_rl_imports()

import os
import sys
import copy
import yaml
import torch
import getpass
import numpy as np
from time import gmtime, strftime
from tensorboardX import SummaryWriter

from reinforcement_learning.algorithms.reinforce.reinforce import ReinforceAgent
from reinforcement_learning.algorithms.a2c.a2c import A2CAgent
from reinforcement_learning.algorithms.ppo.ppo import PPOAgent
from reinforcement_learning.algorithms.a2c.a2c_recurrent import A2CAgent as A2CLSTMAgent

from reinforcement_learning.algorithms.reinforce.continuous_reinforce import ContinuousReinforceAgent
from reinforcement_learning.algorithms.a2c.continuous_a2c import ContinuousA2CAgent
from reinforcement_learning.algorithms.ppo.continuous_ppo import ContinuousPPOAgent
from reinforcement_learning.algorithms.a2c.a2c_recurrent import ContinuousA2CAgent as A2CLSTMContinuousAgent


from reinforcement_learning.torch_extentions.optim.lr_scheduler import RefinementLRScheduler

from score_following_game.agents.models import SFModel
from score_following_game.agents.optim_utils import get_optimizer, cast_optim_params
from score_following_game.agents.networks import get_network
from score_following_game.data_processing.data_pool import get_data_pool, get_data_pools, load_game_config
from score_following_game.environment.env_wrappers import ResizeSizeObservations, SpecDifference, SheetDifference, \
    InvertSheet, ConvertToFloat
from score_following_game.environment.multi_modal_subproc_vec_env import MultiModalSubprocVecEnv
from score_following_game.environment.score_following_env import ScoreFollowingEnv
from score_following_game.evaluation.evaluation import Evaluator


def get_make_env(rl_pool, config, render_mode=None):
    def _thunk():
        return make_env(rl_pool, config, render_mode=render_mode)

    return _thunk


def make_env(rl_pool, config, render_mode=None):

    # initialize environment
    env = ScoreFollowingEnv(rl_pool, config, render_mode=render_mode)

    # apply wrapper to env
    env = ResizeSizeObservations(env, config["spec_factor"], config["sheet_factor"])
    env = ConvertToFloat(env)
    env = InvertSheet(env)
    env = SpecDifference(env)
    env = SheetDifference(env)

    return env


if __name__ == '__main__':
    """ main """

    # add argument parser
    import argparse

    parser = argparse.ArgumentParser(description='Train multi-modality model.')

    parser.add_argument('--agent', help='reinforcement learning algorithm [reinforce|a2c|ppo].', type=str, default="a2c")
    parser.add_argument('--net', help='network architecture to optimize.', type=str, default="ScoreFollowingNet")

    parser.add_argument('--optim', help='optimizer.', type=str, default="Adam")
    parser.add_argument('--optim_params', help='optimizer parameters.', type=yaml.load,
                        default="{lr: 1e-4, betas: '(0.5, 0.999)'}")
    parser.add_argument('--max_grad_norm', help='maximum length of gradient vectors.', type=np.float, default=None)

    parser.add_argument('--max_updates', help='maximum number of update steps.', type=np.int, default=np.int(1e9))
    parser.add_argument('--patience', help='max number of evaluations without improvement.', type=np.int, default=100)
    parser.add_argument('--max_refinements', help='max number of learning rate refinements.', type=np.int, default=1)
    parser.add_argument('--lr_multiplier', help='after patience expires multiply the learning rate with this factor.',
                        type=np.float32, default=0.1)
    parser.add_argument('--eval_interval', help='', type=np.int, default=5000)
    parser.add_argument('--log_interval', help='log train progress after every k updates.', type=np.int, default=100)
    parser.add_argument('--dump_interval', help='dump model parameters after every k updates.', type=np.int,
                        default=np.int(100000))
    parser.add_argument('--eval_trials', help='number of evaluation trials to run.', type=int, default=5)

    parser.add_argument('--n_worker', help='number of parallel workers.', type=np.int, default=16)
    parser.add_argument('--t_max', help='maximum number of time steps/horizon.', type=np.int, default=15)
    parser.add_argument('--value_coef', help='influence of value loss (critic).', type=np.float, default=0.5)
    parser.add_argument('--entropy_coef', help='influence of entropy regularization.', type=np.float, default=0.05)
    parser.add_argument('--discounting', help='discount factor.', type=np.float, default=0.9)

    parser.add_argument('--gae_lambda', help='lambda for generalized advantage estimation.', type=np.float, default=0.95)
    parser.add_argument('--ppo_epsilon', help='clipping parameter for policy changes.', type=np.float, default=0.2)
    parser.add_argument('--ppo_epochs', help='number of epochs for surrogate objective optimization.',
                        type=np.int, default=4)
    parser.add_argument('--batch_size', help='batch size for surrogate objective optimization',
                        type=np.int, default=32)
    parser.add_argument('--prioritized_sampling', help='sample difficult train songs more often.', action='store_true')
    parser.add_argument('--log_root', help='tensorboard log directory.', type=str, default="runs")
    parser.add_argument('--param_root', help='dump network parameters to this folder.', type=str, default="params")
    parser.add_argument('--train_set', help='path to train dataset.', type=str, default="test_sample")
    parser.add_argument('--eval_set', help='path to evaluation dataset.', type=str, default="test_sample")
    parser.add_argument('--game_config', help='path to game config file.', type=str,
                        default='game_configs/nottingham.yaml')

    parser.add_argument('--no_log', help='no tensorboard log.', action='store_true')

    parser.add_argument('--seed', help='random seed.', type=np.int, default=4711)

    parser.add_argument('--ini_params', help='path to initial parameters.', type=str, default=None)

    args = parser.parse_args()

    # we also fix a seed for reproducing our results
    np.random.seed(args.seed)

    # compile unique result folder
    time_stamp = strftime("%Y%m%d_%H%M%S", gmtime())
    tr_set = os.path.basename(args.train_set)
    config_name = os.path.basename(args.game_config).split(".yaml")[0]
    user = getpass.getuser()
    exp_dir = args.agent + "-" + args.net + "-" + tr_set + "-" + config_name + "_" + time_stamp + "-" + user

    # create model parameter directory
    dump_dir = os.path.join(args.param_root, exp_dir)
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)

    # initialize tensorboard logger
    log_writer = None
    if not args.no_log:
        log_dir = os.path.join(args.log_root, exp_dir)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_writer = SummaryWriter(log_dir=log_dir)

        # log run settings
        text = ""
        arguments = np.sort([arg for arg in vars(args)])
        for arg in arguments:
            text += "**{}:** {}<br>".format(arg, getattr(args, arg))
            # log_writer.add_text(arg, "{}".format(getattr(args, arg)))
        log_writer.add_text("run_config", text)
        log_writer.add_text("cmd", " ".join(sys.argv))

    # cast optimizer parameters to float
    args.optim_params = cast_optim_params(args.optim_params)

    # load game config
    config = load_game_config(args.game_config)

    # initialize data pool
    rl_pool = get_data_pool(config, directory=args.train_set)

    # initialize environment
    if args.agent == 'reinforce':
        env = make_env(rl_pool, config, render_mode=None)
    else:
        env = MultiModalSubprocVecEnv([get_make_env(rl_pool, config, render_mode=None) for i in range(args.n_worker)])

    # compile network architecture
    net = get_network(args.net, env.action_space.n,
                      spec_channels=config['spec_shape'][0],
                      sheet_channels=config['sheet_shape'][0])

    # load initial parameters
    if args.ini_params:
        net.load_state_dict(torch.load(args.ini_params))

    # initialize optimizer
    optimizer = get_optimizer(args.optim, net.parameters(), **args.optim_params)

    # initialize model
    model = SFModel(net, optimizer, max_grad_norm=args.max_grad_norm,
                    value_coef=args.value_coef, entropy_coef=args.entropy_coef)

    # initialize refinement scheduler
    lr_scheduler = RefinementLRScheduler(optimizer=optimizer, model=model, n_refinement_steps=args.max_refinements,
                                         patience=args.patience, learn_rate_multiplier=args.lr_multiplier,
                                         high_is_better=True)

    # use cuda if available
    if torch.cuda.is_available():
        model.cuda()

    # initialize model evaluation
    evaluation_pools = get_data_pools(config, directory=args.eval_set)
    evaluator = Evaluator(make_env, evaluation_pools, config=config, trials=args.eval_trials, render_mode=None)

    if args.agent == 'reinforce':

        if config['continuous']:
            agent = ContinuousReinforceAgent(env=env, model=model, gamma=args.discounting)
        else:
            agent = ReinforceAgent(env=env, model=model, gamma=args.discounting)

    elif args.agent == 'a2c':

        if config['continuous']:
            agent = ContinuousA2CAgent(env=env, model=model, t_max=args.t_max,
                                       n_worker=args.n_worker, gamma=args.discounting)
        else:
            agent = A2CAgent(env=env, model=model, t_max=args.t_max,
                             n_worker=args.n_worker, gamma=args.discounting)

    # TODO: merge this with a2c
    elif args.agent == 'a2c_lstm':

        if config['continuous']:
            agent = A2CLSTMContinuousAgent(env=env, model=model, t_max=args.t_max,
                                           n_worker=args.n_worker, gamma=args.discounting)
        else:
            agent = A2CLSTMAgent(env=env, model=model, t_max=args.t_max,
                                 n_worker=args.n_worker, gamma=args.discounting)

    elif args.agent == 'ppo':

        if config['continuous']:
            agent = ContinuousPPOAgent(env=env, model=model, t_max=args.t_max, n_worker=args.n_worker,
                                       gamma=args.discounting, gae_lambda=args.gae_lambda, ppo_epoch=args.ppo_epochs,
                                       epsilon=args.ppo_epsilon, batch_size=args.batch_size)

        else:
            agent = PPOAgent(env=env, model=model, t_max=args.t_max, n_worker=args.n_worker, gamma=args.discounting,
                             gae_lambda=args.gae_lambda, ppo_epoch=args.ppo_epochs, epsilon=args.ppo_epsilon,
                             batch_size=args.batch_size)
    else:
        raise NotImplementedError('Invalid Algorithm')

    agent.train(max_updates=args.max_updates, log_writer=log_writer, log_interval=args.log_interval,
                evaluator=evaluator, eval_interval=args.eval_interval, lr_scheduler=lr_scheduler,
                score_name='global_tracking_ratio', high_is_better=True, dump_interval=args.dump_interval,
                dump_dir=dump_dir)

    if not args.no_log:
        log_writer.close()
