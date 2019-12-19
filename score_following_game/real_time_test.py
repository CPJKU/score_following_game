
import os
import time
import torch
import tqdm

import numpy as np

from score_following_game.agents.networks_utils import get_network
from score_following_game.data_processing.data_pools import get_single_song_pool
from score_following_game.data_processing.utils import load_game_config, spectrogram_processor
from score_following_game.experiment_utils import initialize_trained_agent, make_env_tismir, setup_evaluation_parser
from score_following_game.reinforcement_learning.algorithms.models import Model


if __name__ == "__main__":

    parser = setup_evaluation_parser()
    args = parser.parse_args()

    exp_name = os.path.basename(os.path.split(args.params)[0])

    if args.net is None:
        args.net = exp_name.split('-')[1]

    if args.game_config is None:
        args.game_config = 'game_configs/{}.yaml'.format(exp_name.split("-")[3].rsplit("_", 2)[0])

    config = load_game_config(args.game_config)

    pool = get_single_song_pool(dict(config=config, song_name=args.piece,
                                     directory=args.data_set, real_perf=args.real_perf))

    # initialize environment
    env = make_env_tismir(pool, config, render_mode=None)

    # compile network architecture
    n_actions = len(config["actions"])
    net = get_network("networks_sheet_spec", args.net, n_actions=n_actions,
                      shapes=dict(perf_shape=config['spec_shape'], score_shape=config['sheet_shape']))

    # load network parameters
    net.load_state_dict(torch.load(args.params))

    # set model to evaluation mode
    net.eval()

    # create agent
    use_cuda = torch.cuda.is_available()
    model = Model(net, optimizer=None)

    agent = initialize_trained_agent(model, use_cuda=use_cuda, deterministic=True)

    times_per_step = []

    processor = spectrogram_processor(config['spectrogram_params'])

    for _ in tqdm.tqdm(range(1000)):
        observation = env.reset()
        done = False

        while not done:
            # choose action
            start = time.time()

            # simulate frame processing
            processor(np.random.randn(1102))

            # choose action
            action = agent.select_action(observation)

            # perform step and observe
            observation, reward, done, info = env.step(action)

            end = time.time()
            times_per_step.append(end - start)

    print('Average time per step: {} seconds'.format(np.mean(times_per_step)))
