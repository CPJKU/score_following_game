
import os
import sys
import yaml

import numpy as np

from torch.utils.tensorboard import SummaryWriter


def setup_parser():
    import argparse

    parser = argparse.ArgumentParser(description='Train Score Following agent.')

    # general parameters
    parser.add_argument('--train_set', help='path to train dataset.', type=str, default=None)
    parser.add_argument('--eval_set', help='path to evaluation dataset.', type=str, default=None)
    parser.add_argument('--real_perf', help='Performances are real audio recordings.',
                        choices=[None, 'wav'], type=str, default=None)
    parser.add_argument('--game_config', help='path to game config file.', type=str,
                        default='game_configs/midi_config.yaml')
    parser.add_argument('--use_cuda', help='if set use gpu instead of cpu.', action='store_true')
    parser.add_argument('--seed', help='random seed.', type=np.int, default=4711)

    # agent parameters
    parser.add_argument('--agent', help='reinforcement learning algorithm [reinforce|a2c|ppo].',
                        choices=['reinforce', 'a2c', 'ppo'], type=str, default="a2c")
    parser.add_argument('--net', help='network architecture to optimize.', type=str)
    parser.add_argument('--ini_params', help='path to initial parameters.', type=str, default=None)
    parser.add_argument('--n_worker', help='number of parallel workers.', type=np.int, default=8)
    parser.add_argument('--t_max', help='maximum number of time steps/horizon.', type=np.int, default=15)
    parser.add_argument('--value_coef', help='influence of value loss (critic).', type=np.float, default=0.5)
    parser.add_argument('--entropy_coef', help='influence of entropy regularization.', type=np.float, default=0.05)
    parser.add_argument('--discounting', help='discount factor.', type=np.float, default=0.9)
    parser.add_argument('--gae_lambda', help='lambda for generalized advantage estimation.', type=np.float,
                        default=0.95)
    parser.add_argument('--gae', help='use generalized advantage estimation for a2c.', default=False,
                        action='store_true')
    parser.add_argument('--ppo_epsilon', help='clipping parameter for policy changes.', type=np.float, default=0.2)
    parser.add_argument('--ppo_epochs', help='number of epochs for surrogate objective optimization.',
                        type=np.int, default=4)
    parser.add_argument('--batch_size', help='batch size for surrogate objective optimization',
                        type=np.int, default=32)
    parser.add_argument('--clip_value', help='clip value loss.', default=False, action='store_true')

    parser.add_argument('--optim', help='optimizer.', type=str, default="Adam")
    parser.add_argument('--optim_params', help='optimizer parameters.', type=yaml.load,
                        default="{lr: 1e-4, betas: '(0.9, 0.999)'}")
    parser.add_argument('--max_grad_norm', help='maximum length of gradient vectors.', type=np.float, default=0.5)

    parser.add_argument('--max_updates', help='maximum number of update steps.', type=np.int, default=np.int(1e9))
    parser.add_argument('--patience', help='max number of evaluations without improvement.', type=np.int, default=50)
    parser.add_argument('--max_refinements', help='max number of learning rate refinements.', type=np.int, default=2)
    parser.add_argument('--lr_multiplier', help='after patience expires multiply the learning rate with this factor.',
                        type=np.float32, default=0.1)

    # evaluation
    parser.add_argument('--eval_interval', help='', type=np.int, default=5000)
    parser.add_argument('--eval_trials', help='number of evaluation trials to run.', type=int, default=1)
    parser.add_argument('--eval_score_name', help='name of the evaluation score used for model selection.', type=str,
                        default="global_tracking_ratio")
    parser.add_argument('--low_is_better', help='indicate if a low score is better than a high one for evaluation.',
                        default=False, action='store_true')

    # logging
    parser.add_argument('--no_log', help='no tensorboard log.', action='store_true')
    parser.add_argument('--log_root', help='tensorboard log directory.', type=str, default="runs")
    parser.add_argument('--log_interval', help='log train progress after every k updates.', type=np.int, default=100)
    parser.add_argument('--param_root', help='dump network parameters to this folder.', type=str, default="params")
    parser.add_argument('--dump_interval', help='dump model parameters after every k updates.', type=np.int,
                        default=np.int(100000))
    parser.add_argument('--log_gradients', help='log gradients.', default=False, action='store_true')

    return parser


def setup_evaluation_parser():

    import argparse

    parser = argparse.ArgumentParser(description='Evaluate a trained RL agent.')
    parser.add_argument('--data_set', help='path to data set.', type=str, default=None)
    parser.add_argument('--game_config', help='path to game config file.', type=str, default=None)
    parser.add_argument('--net', help='network architecture to optimize.', type=str, default=None)
    parser.add_argument('--params', help='path to parameter dump.', type=str, default=None)
    parser.add_argument('--piece', help='select song for testing.', type=str, default=None)
    parser.add_argument('--real_perf', help='Performances are real audio recordings.',
                        choices=[None, 'wav'], type=str, default=None)
    parser.add_argument('--trials', help='number of trials to run.', type=int, default=1)
    parser.add_argument('--eval_embedding', help='evaluate the learned embeddings.', default=False, action='store_true')
    return parser


def setup_agent(args):
    from score_following_game.reinforcement_learning.algorithms.agent import get_agent

    params = {
        'observation_space': args.env.observation_space.spaces,
        'model': args.model,
        'gamma': args.discounting,
        'use_cuda': args.use_cuda,
        'log_writer': args.log_writer, 'log_interval': args.log_interval,
        'evaluator': args.evaluator, 'eval_interval': args.eval_interval,
        'lr_scheduler': args.lr_scheduler, 'score_name': args.eval_score_name, 'high_is_better': not args.low_is_better,
        'dump_interval': args.dump_interval, 'dump_dir': args.dump_dir,
        'n_actions': args.n_actions,
    }

    if hasattr(args, 'distribution'):
        params['distribution'] = args.distribution

    if args.agent == 'reinforce':
        params['batch_size'] = args.batch_size

    elif args.agent == 'a2c':
        params['n_worker'] = args.n_worker
        params['t_max'] = args.t_max
        params['gae_lambda'] = args.gae_lambda
        params['gae'] = args.gae

    elif args.agent == 'ppo':
        params['n_worker'] = args.n_worker
        params['t_max'] = args.t_max
        params['gae_lambda'] = args.gae_lambda
        params['ppo_epoch'] = args.ppo_epochs
        params['epsilon'] = args.ppo_epsilon
        params['batch_size'] = args.batch_size
        params['clip_value'] = args.clip_value

    return get_agent(args.agent, **params)


def setup_logger(args):

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    log_writer = SummaryWriter(log_dir=args.log_dir)

    log_writer.log_gradients = args.log_gradients

    # log run settings
    text = ""
    arguments = np.sort([arg for arg in vars(args)])
    for arg in arguments:
        text += "**{}:** {}<br>".format(arg, getattr(args, arg))

    log_writer.add_text("run_config", text)
    log_writer.add_text("cmd", " ".join(sys.argv))

    return log_writer


def initialize_trained_agent(model, use_cuda=True, deterministic=False, distribution=None):
    from score_following_game.reinforcement_learning.algorithms.agent import TrainedAgent
    from score_following_game.reinforcement_learning.torch_extentions.distributions.adapted_categorical import AdaptedCategorical

    if distribution is None:
        distribution = AdaptedCategorical

    return TrainedAgent(model, use_cuda=use_cuda, deterministic=deterministic, distribution=distribution)


def make_env_tismir(rl_pool, config, render_mode=None):
    from score_following_game.environment.score_following_env import ScoreFollowingEnv
    from score_following_game.environment.env_wrappers import ConvertToFloatWrapper, ResizeSizeWrapper, InvertWrapper, \
        DifferenceWrapper

    # initialize environment
    env = ScoreFollowingEnv(rl_pool, config, render_mode=render_mode)
    env = ResizeSizeWrapper(env, key='score', factor=config['score_factor'], dim=config['score_dim'])
    env = ResizeSizeWrapper(env, key='perf', factor=config['perf_factor'], dim=config['perf_dim'])
    env = ConvertToFloatWrapper(env, key='score')
    env = InvertWrapper(env, key='score')

    if config['spec_shape'][0] > 1:
        env = DifferenceWrapper(env, key='perf')
    if config['sheet_shape'][0] > 1:
        env = DifferenceWrapper(env, key='score')

    return env


def get_make_env(rl_pool, config, make_env_fnc, render_mode=None):
    def _thunk():
        return make_env_fnc(rl_pool, config, render_mode=render_mode)

    return _thunk
