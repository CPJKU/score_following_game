
import os
import pickle
import torch

from score_following_game.agents.networks_utils import get_network
from score_following_game.data_processing.data_pools import get_data_pools
from score_following_game.data_processing.utils import load_game_config
from score_following_game.evaluation.evaluation import EmbeddingEvaluator, PerformanceEvaluator, print_formatted_stats
from score_following_game.experiment_utils import initialize_trained_agent, setup_evaluation_parser, make_env_tismir
from score_following_game.reinforcement_learning.algorithms.models import Model

# render mode for the environment ('human', 'computer')
render_mode = 'computer'

if __name__ == "__main__":
    """ main """

    parser = setup_evaluation_parser()

    args = parser.parse_args()

    # parse parameter string
    exp_name = os.path.basename(os.path.split(args.params)[0])
    if args.net is None:
        args.net = exp_name.split('-')[1]

    if args.game_config is None:
        args.game_config = 'game_configs/{}.yaml'.format(exp_name.split("-")[3].rsplit("_", 2)[0])

    # load game config
    config = load_game_config(args.game_config)

    # set agent type ('human', 'optimal') optimal currently not supported
    agent_type = render_mode

    # compile network architecture
    n_actions = len(config["actions"])
    net = get_network('networks_sheet_spec', args.net, n_actions=n_actions,
                      shapes=dict(perf_shape=config['spec_shape'], score_shape=config['sheet_shape']))

    # load network parameters
    net.load_state_dict(torch.load(args.params, map_location=lambda storage, loc: storage))

    # set model to evaluation mode
    net.eval()

    # create agent
    use_cuda = torch.cuda.is_available()

    model = Model(net, optimizer=None)

    agent = initialize_trained_agent(model, use_cuda=use_cuda, deterministic=False)

    # initialize evaluation pools
    evaluation_pools = get_data_pools(config, directory=args.data_set, real_perf=args.real_perf)

    # set verbosity level
    verbose = args.trials == 1

    if args.eval_embedding:
        evaluator = EmbeddingEvaluator(make_env_tismir, evaluation_pools, config, render_mode=None)

        # evaluate on all pieces
        res = evaluator.evaluate(agent, log_writer=None, log_step=0, verbose=verbose)
        with open('eval_elu.pickle', 'wb') as f:
            pickle.dump(res, f)

    else:
        evaluator = PerformanceEvaluator(make_env_tismir, evaluation_pools, config, render_mode=None)
        printing = print_formatted_stats

        # evaluate on all pieces
        mean_stats = None
        for i_trial in range(args.trials):
            stats = evaluator.evaluate(agent, log_writer=None, log_step=0, verbose=verbose)
            printing(stats)

            if mean_stats is None:
                mean_stats = dict()
                for key in stats.keys():
                    if key != "evaluation_data":
                        mean_stats[key] = []

            for key in mean_stats.keys():
                mean_stats[key].append(stats[key])

        print("-" * 50)
        printing(mean_stats)

