
import random
import torch

import numpy as np

from baselines.common.vec_env.shmem_vec_env import ShmemVecEnv
from score_following_game.agents.networks import get_network
from score_following_game.data_processing.song import get_data_pools
from score_following_game.data_processing.song import create_shared_songs_pool, load_songs
from score_following_game.evaluation.evaluation import PerformanceEvaluator as Evaluator
from score_following_game.experiment_utils import setup_parser, setup_agent, make_env_tismir,\
    get_make_env, load_game_config
from score_following_game.agents.lr_scheduler import RefinementLRScheduler
from score_following_game.agents.models import Model
from score_following_game.logger import Logger


if __name__ == '__main__':
    """ main """

    parser = setup_parser()
    args = parser.parse_args()

    device = torch.device("cuda" if args.use_cuda else "cpu")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    torch.backends.cudnn.benchmark = True

    # initialize logger
    logger = Logger.setup_logger(args)
    args.logger = logger

    # load game config
    config = load_game_config(args.game_config)

    # initialize songs and data pools
    songs = load_songs(config=config, directory=args.train_set, split=args.split_data)
    print(f'Loading {len(songs)} training songs')
    shared_songs_pool = create_shared_songs_pool(songs)

    if args.model == 'reinforce':
        env = get_make_env(shared_songs_pool, config, make_env_tismir, seed=args.seed, rank=0)()
    else:
        env = ShmemVecEnv([get_make_env(shared_songs_pool, config, make_env_tismir, seed=args.seed, rank=i)
                           for i in range(args.n_worker)])

    # compile network architecture
    net = get_network(args.net, env.action_space.n, shapes=dict(perf_shape=config['spec_shape'],
                                                                score_shape=config['sheet_shape']))

    # load initial parameters
    if args.ini_params:
        net.load_state_dict(torch.load(args.ini_params))

    # initialize optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999))

    # initialize model
    model = Model(net, optimizer, max_grad_norm=args.max_grad_norm, value_coef=args.value_coef,
                  entropy_coef=args.entropy_coef, device=device)

    # initialize refinement scheduler
    lr_scheduler = RefinementLRScheduler(optimizer=optimizer, model=model, n_refinement_steps=args.max_refinements,
                                         patience=args.patience, learn_rate_multiplier=args.lr_multiplier,
                                         high_is_better=not args.low_is_better)

    model.to(device)

    # initialize model evaluation
    evaluation_pools = get_data_pools(config, directory=args.eval_set)

    evaluator = Evaluator(make_env_tismir, evaluation_pools, config=config, logger=logger, trials=args.eval_trials,
                          render_mode=None, eval_interval=args.eval_interval, score_name=args.eval_score_name,
                          high_is_better=not args.low_is_better, seed=args.seed)

    args.model = model
    args.env = env
    args.lr_scheduler = lr_scheduler
    args.evaluator = evaluator
    agent = setup_agent(args=args)

    max_updates = args.max_updates * args.t_max
    agent.train(env, max_updates)

    # clean up
    env.close()
    logger.close()
