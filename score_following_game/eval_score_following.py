
from __future__ import print_function

# this is required as we included our reinforcment_learning package
# as a submodule for convenience
from score_following_game.utils import init_rl_imports
init_rl_imports()

import torch
import numpy as np

from score_following_game.sf_experiment import make_env as make_eval_env
from score_following_game.data_processing.data_pool import load_game_config
from score_following_game.evaluation.evaluation import Evaluator, print_formatted_stats
from score_following_game.data_processing.data_pool import get_data_pools
from score_following_game.agents.networks import get_network

from reinforcement_learning.agents.agents import TrainedAgent, TrainedContinuousAgent
from reinforcement_learning.agents.agents import TrainedLSTMAgent, TrainedLSTMContinuousAgent


# render mode for the environment ('human', 'computer')
render_mode = 'computer'


def initialize_agent(net, continuous, rnn, use_cuda=True):

    if continuous:
        if rnn:
            agent = TrainedLSTMContinuousAgent(net, use_cuda=use_cuda)
        else:
            agent = TrainedContinuousAgent(net, use_cuda=use_cuda)
    else:
        if rnn:
            agent = TrainedLSTMAgent(net, use_cuda=use_cuda)
        else:
            agent = TrainedAgent(net, use_cuda=use_cuda)

    return agent


if __name__ == "__main__":
    """ main """

    # add argument parser
    import argparse

    parser = argparse.ArgumentParser(description='Train multi-modality model.')
    parser.add_argument('--net', help='network architecture to optimize.', type=str, default=None)
    parser.add_argument('--params', help='path to parameter dump.', type=str)
    parser.add_argument('--data_set', help='path to data set.', type=str, default="test_sample_clip")
    parser.add_argument('--game_config', help='path to game config file.', type=str, default=None)
    parser.add_argument('--trials', help='number of trials to run.', type=int, default=1)
    args = parser.parse_args()

    # parse parameter string
    if args.net is None:
        args.net = args.params.split("-")[1]

    if args.game_config is None:
        args.game_config = "game_configs/%s.yaml" % args.params.split("-")[3].rsplit("_", 2)[0]

    # load game config
    config = load_game_config(args.game_config)

    # set agent type ('human', 'optimal') optimal currently not supported
    agent_type = render_mode

    # compile network architecture
    n_actions = len(config["actions"])
    net = get_network(args.net, n_actions=n_actions,
                      spec_channels=config['spec_shape'][0],
                      sheet_channels=config['sheet_shape'][0])

    # load network parameters
    net.load_state_dict(torch.load(args.params))

    # set model to evaluation mode
    net.eval()

    # create agent
    use_cuda = torch.cuda.is_available()
    has_rnn = "LSTM" in args.net
    agent = initialize_agent(net, continuous=config['continuous'], rnn=has_rnn, use_cuda=True)

    # initialize evaluation pools
    evaluation_pools = get_data_pools(config, directory=args.data_set)

    # initialize evaluator
    evaluator = Evaluator(make_eval_env, evaluation_pools, config, render_mode=None)

    # set verbosity level
    verbose = args.trials == 1

    # evaluate on all pieces
    mean_stats = None
    for i_trial in range(args.trials):
        stats = evaluator.evaluate(agent, log_writer=None, log_step=0, verbose=verbose)
        print_formatted_stats(stats)

        if mean_stats is None:
            mean_stats = dict()
            for key in stats.keys():
                if key != "evaluation_data":
                    mean_stats[key] = []

        for key in mean_stats.keys():
            mean_stats[key].append(stats[key])

    print("-" * 50)
    for key in mean_stats.keys():
        mean_stats[key] = np.mean(mean_stats[key])
    print_formatted_stats(mean_stats)
