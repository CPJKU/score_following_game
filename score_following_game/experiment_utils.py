
import yaml

import numpy as np

from score_following_game.agents.a2c import A2CAgent
from score_following_game.agents.ppo import PPOAgent
from score_following_game.agents.reinforce import ReinforceAgent


def load_game_config(config_file):
    """Load game config from YAML file."""
    with open(config_file, 'rb') as fp:
        config = yaml.load(fp, Loader=yaml.FullLoader)
    return config


def setup_parser():
    import argparse

    parser = argparse.ArgumentParser(description='Train Score Following agent.')

    # general parameters
    parser.add_argument('--train_set', help='path to train dataset.', type=str, default=None)
    parser.add_argument('--eval_set', help='path to evaluation dataset.', type=str, default=None)
    parser.add_argument('--game_config', help='path to game config file.', type=str,
                        default='game_configs/midi_config.yaml')
    parser.add_argument('--use_cuda', help='if set use gpu instead of cpu.', action='store_true')
    parser.add_argument('--seed', help='random seed.', type=np.int, default=4711)
    parser.add_argument('--split_data', help='split piece per page.', default=False, action='store_true')

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

    parser.add_argument('--lr', help='learning rate', type=np.float, default=1e-4)
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
    parser.add_argument('--dump_root', help='dump network parameters to this folder.', type=str, default="params")
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
    parser.add_argument('--trials', help='number of trials to run.', type=int, default=1)
    parser.add_argument('--eval_embedding', help='evaluate the learned embeddings.', default=False, action='store_true')
    parser.add_argument('--split_data', help='split piece per page.', default=False, action='store_true')
    parser.add_argument('--seed', help='random seed.', type=np.int, default=4711)
    return parser


def setup_agent(args):

    params = {
        'model': args.model,
        'gamma': args.discounting,
        'logger': args.logger,
        'evaluator': args.evaluator,
        'lr_scheduler': args.lr_scheduler,
    }

    if args.model == 'reinforce':
        params['batch_size'] = args.batch_size
        agent_type = ReinforceAgent

    elif args.model == 'a2c':
        params['observation_space'] = args.env.observation_space.spaces
        params['n_worker'] = args.n_worker
        params['t_max'] = args.t_max
        params['gae_lambda'] = args.gae_lambda
        params['gae'] = args.gae
        agent_type = A2CAgent

    elif args.model == 'ppo':
        params['observation_space'] = args.env.observation_space.spaces
        params['n_worker'] = args.n_worker
        params['t_max'] = args.t_max
        params['gae_lambda'] = args.gae_lambda
        params['ppo_epoch'] = args.ppo_epochs
        params['epsilon'] = args.ppo_epsilon
        params['batch_size'] = args.batch_size
        agent_type = PPOAgent
    else:
        raise NotImplementedError('Invalid Algorithm')

    return agent_type(**params)


def make_env_tismir(rl_pool, config, seed, rank=0, render_mode=None):
    from score_following_game.environment.score_following_env import ScoreFollowingEnv
    from score_following_game.environment.env_wrappers import ConvertToFloatWrapper, ResizeSizeWrapper, InvertWrapper, \
        DifferenceWrapper

    # initialize environment
    env = ScoreFollowingEnv(rl_pool, config, render_mode=render_mode)
    env.seed(seed + rank)
    env.action_space.seed(seed + rank)

    env = ResizeSizeWrapper(env, key='score', factor=config['score_factor'], dim=config['score_dim'])
    env = ResizeSizeWrapper(env, key='perf', factor=config['perf_factor'], dim=config['perf_dim'])
    env = ConvertToFloatWrapper(env, key='score')
    env = InvertWrapper(env, key='score')

    if config['spec_shape'][0] > 1:
        env = DifferenceWrapper(env, key='perf')
    if config['sheet_shape'][0] > 1:
        env = DifferenceWrapper(env, key='score')

    return env


def get_make_env(rl_pool, config, make_env_fnc, seed, rank=0, render_mode=None):
    def _thunk():
        return make_env_fnc(rl_pool, config, seed=seed, rank=rank, render_mode=render_mode)

    return _thunk
