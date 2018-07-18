
import matplotlib.pyplot as plt

from score_following_game.data_processing.data_pool import get_data_pool, load_game_config
from score_following_game.agents.optimal_agent import OptimalAgent
from score_following_game.sf_experiment import make_env

# render mode for the environment ('human', 'computer')
render_mode = 'computer'


if __name__ == "__main__":
    """ main """

    # add argument parser
    import argparse

    parser = argparse.ArgumentParser(description='Train multi-modality model.')
    parser.add_argument('--data_set', help='path to data set.', type=str, default="test_sample")
    parser.add_argument('--game_config', help='path to game config file.', type=str, default="game_configs/nottingham_continuous.yaml")
    parser.add_argument('--song', help='select song for testing.', type=str, default=None)
    args = parser.parse_args()

    # load game config
    config = load_game_config(args.game_config)

    # initialize data pool
    rl_pool = get_data_pool(config, directory=args.data_set, song_name=args.song)

    # set agent type ('human', 'optimal') optimal currently not supported
    agent_type = render_mode

    # initialize environment
    env = make_env(rl_pool, config, render_mode=render_mode)

    # create agent
    agent = OptimalAgent(rl_pool)

    # iterate episodes
    time_steps = []
    alignment_errors = []
    tempo_curve = []
    action_sequence = []

    # get observations
    episode_reward = 0
    observation = env.reset()
    print("Tracking song:", rl_pool.get_current_song_name())

    reward = 0
    done = False

    agent.set_optimal_actions()

    while True:

        # choose action
        action = agent.perform_action(observation)

        # perform step and observe
        observation, reward, done, info = env.step(action)
        episode_reward += reward
        print(observation[0].shape, observation[1].shape, reward)

        # collect some stats
        alignment_errors.append(rl_pool.tracking_error())
        tempo_curve.append(rl_pool.sheet_speed)
        action_sequence.append(action)

        if done:
            break

    print("Episode reward %d\n" % episode_reward)

    plt.figure("tempo_curve")
    plt.clf()

    ax = plt.subplot(2, 1, 1)
    plt.plot(range(len(tempo_curve)), tempo_curve)
    plt.xlim([-5, len(tempo_curve) + 5])
    plt.grid("on")
    plt.ylabel("optimal speed", fontsize=18)
    plt.tick_params(labelsize=16)
    ax.get_yaxis().set_label_coords(-0.1, 0.5)

    ax = plt.subplot(2, 1, 2)
    plt.plot(range(len(action_sequence)), action_sequence)
    plt.xlim([-5, len(action_sequence) + 5])
    plt.grid("on")
    plt.xlabel("time step t", fontsize=18)
    plt.ylabel("optimal action $A_t$", fontsize=18)
    plt.tick_params(labelsize=16)
    ax.get_yaxis().set_label_coords(-0.1, 0.5)

    plt.show(block=True)
