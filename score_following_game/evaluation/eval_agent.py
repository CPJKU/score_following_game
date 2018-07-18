from __future__ import print_function

import pickle
import torch
import yaml
from score_following_game.data_processing.data_pool import get_data_pools
from score_following_game.environment.env_wrappers import ResizeSizeObservations, SpecDifference, SheetDifference, \
    InvertSheet, ConvertToFloat

from score_following_game.environment.score_following_env import ScoreFollowingEnv
from score_following_game.evaluation.evaluation import Evaluator
from torch.autograd import Variable
import torch.nn.functional as F
from reinforcement_learning.algorithms.util.networks import ScoreFollowingNet
from score_following_game.agents.optimal_agent import OptimalAgent
import datetime


def make_env(rl_pool, config, render_mode=None, continuous=True, onset_window=2):

    def _thunk():

        # initialize environment
        env = ScoreFollowingEnv(rl_pool, render_mode=render_mode, continuous=continuous, reward_window=onset_window)

        # apply wrapper to env
        env = ResizeSizeObservations(env, config["spec_factor"], config["sheet_factor"])
        env = ConvertToFloat(env)
        env = InvertSheet(env)
        env = SpecDifference(env)
        env = SheetDifference(env)

        return env

    return _thunk


class EvaluationAgent(object):

    def __init__(self, path_to_net_params, n_actions, spec_shape, sheet_shape, model):

        # create networks
        self.model = model(n_actions, spec_shape, sheet_shape)

        # load network parameters
        self.model.load_state_dict(torch.load(path_to_net_params))

        if torch.cuda.is_available():
            self.model.cuda()

    def perform_action(self, state):

        spec, sheet = state

        spec = torch.from_numpy(spec).float()
        sheet = torch.from_numpy(sheet).float()

        if torch.cuda.is_available():
            spec = spec.cuda()
            sheet = sheet.cuda()

        policy, _ = self.model([Variable(spec.unsqueeze(0)), Variable(sheet.unsqueeze(0))])

        probabilities = F.softmax(policy, dim=-1)

        return probabilities.multinomial().data[0].cpu().numpy()[0]


if __name__ == '__main__':
    """
    main
    """

    # add argument parser
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate Agent')
    parser.add_argument('--game_config', help='path to game config file.', type=str,
                        default='/home/florianh/RL/score_following_game/score_following_game/game_configs/nottingham.yaml')
    parser.add_argument('--data_set', help='path to dataset.', type=str,
                        default="/home/florianh/RL/nottingham_trvate/Nottingham_train")
    parser.add_argument('--model_params', help='path to the parameters of the agent to evaluate', type=str,
                        default="/home/florianh/RL/reinforcement_learning/model_update_35000000.pt")
    parser.add_argument('--store_to', help='path where evaluation results should be stored', type=str,
                        default="/home/florianh/RL/result_logs")
    parser.add_argument('--store_name', help='name of the file to store', type=str,
                        default='eval'+str(datetime.datetime.now()))

    args = parser.parse_args()

    # load game config
    with open(args.game_config, "rb") as fp:
        config = yaml.load(fp)

    eval_pools = get_data_pools(directory=args.data_set, config_file=args.game_config, version="right_most")

    # temporary environment to get action_space
    env = make_env(eval_pools[0], config, render_mode=None)()
    evaluation_pools = [get_data_pools(directory=args.data_set, config_file=args.game_config, version="right_most")[0]]

    evaluator = Evaluator(make_env, evaluation_pools, config=config, render_mode='computer')

    # agent = EvaluationAgent(path_to_net_params=args.model_params, n_actions=env.action_space.n,
    #                         spec_shape=config["spec_shape"][0], sheet_shape=config["sheet_shape"][0], model=ScoreFollowingNet)

    agent = OptimalAgent(evaluation_pools[0])
    agent.set_optimal_actions()

    ae_mean, ae_median, ae_std, tr_mean, tue_ratio = evaluator.evaluate(agent)
    print('mean: {} median: {} std: {} tr_mean: {} tue_ration: {}'.format(ae_mean, ae_median, ae_std, tr_mean, tue_ratio))
    pickle.dump({'ae_mean': ae_mean, 'ae_median': ae_median,
                 'ae_std': ae_std, 'tr_mean': tr_mean, 'tue_ratio': tue_ratio}, open(args.store_name, 'wb'))