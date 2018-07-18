
import yaml
import logging
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from data_processing.data_pool import get_data_pool

import pickle as pkl
import theano
import lasagne
from first_try import create_networks, get_input_vars, make_env

# render mode for the environment ('human', 'computer')
render_mode = 'computer'


class Agent(object):

    def __init__(self, model_params, spec_shape, sheet_shape):

        # create networks
        pl_net, vl_net = create_networks(n_actions=3, spec_shape=spec_shape, sheet_shape=sheet_shape)

        # load network parameters
        if isinstance(model_params, str):
            with open(model_params, "rb") as fp:
                params = pkl.load(fp)
        else:
            params = model_params
        lasagne.layers.set_all_param_values([pl_net, vl_net], params)

        # get input variables
        input_vars = get_input_vars(pl_net)

        # compile prediction function
        probabilities = lasagne.layers.get_output(pl_net, deterministic=True)
        self.policy = theano.function(input_vars, probabilities)

    def perform_action(self, state):
        spec, sheet = state
        action_probabilities = self.policy(spec, sheet)[0]
        # print([("%.2f" % ap) for ap in action_probabilities])
        action = np.random.choice([0, 1, 2], p=action_probabilities)
        return action


if __name__ == "__main__":
    """ main """

    # add argument parser
    import argparse

    parser = argparse.ArgumentParser(description='Train multi-modality model.')
    parser.add_argument('--params', help='path to parameter dump.', type=str, default="test_sample_clip")
    parser.add_argument('--data_set', help='path to data set.', type=str, default="test_sample_clip")
    parser.add_argument('--game_config', help='path to game config file.', type=str, default=None)
    args = parser.parse_args()

    # load game config
    with open(args.game_config, "rb") as fp:
        config = yaml.load(fp)

    # initialize data pool
    rl_pool = get_data_pool(directory=args.data_set, config_file=args.game_config, version="right_most")

    # set agent type ('human', 'optimal') optimal currently not supported
    agent_type = render_mode

    # initialize environment
    env = make_env(rl_pool, config, render_mode=render_mode)

    # initialize logger and deactivate console output
    logger = logging.getLogger('score_following_game')
    logger.propagate = False

    # create agent
    agent = Agent(model_params=args.params, spec_shape=config["spec_shape"], sheet_shape=config["sheet_shape"])

    # iterate episodes
    time_steps = []
    alignment_errors = []
    tempo_curve = []
    for i_episode in range(1):

        # get observations
        episode_reward = 0
        observation = env.reset()

        reward = 0
        done = False
        timestep = 0

        # setup the logger for the current episode and print a first information line
        log_file_handler = logging.FileHandler('logs/'+rl_pool.get_current_song_name()+'.log')
        logger.addHandler(log_file_handler)
        logger.setLevel(logging.INFO)
        logger.info(rl_pool.get_current_song_name()+' Episode: '+str(i_episode)+' AgentType: ' + agent_type)

        while True:
            timestep += 1

            # choose action
            action = agent.perform_action(observation)

            # perform step and observe
            observation, reward, done, info = env.step(action)
            episode_reward += reward

            # collect some stats
            alignment_errors.append(rl_pool.tracking_error())
            tempo_curve.append(rl_pool.sheet_speed)

            step_msg = '\ttimestep: %8d \taction: %+5f \tpixel_speed: %+5f' \
                       ' \tnew_reward: %+5d \tcumulated_reward: %+8d' \
                       % (timestep, action, rl_pool.sheet_speed, reward, episode_reward)
            logger.info(step_msg)

            if done:
                time_steps.append(timestep)
                done_msg = "Episode finished after %d time steps (avg %.1f)" \
                           % (timestep, np.mean(time_steps))

                print(done_msg)
                logger.info(done_msg)
                break

        episode_msg = "Episode reward %d\n" % episode_reward
        print(episode_msg)

        logger.info(episode_msg)

        # remove the file handler for the currently played song
        logger.removeHandler(log_file_handler)

    plt.figure("Results")
    plt.clf()
    plt.subplot(2, 2, 1)
    plt.plot(alignment_errors, '-')
    plt.subplot(2, 2, 2)
    plt.plot(tempo_curve, '-')
    plt.show(block=True)
