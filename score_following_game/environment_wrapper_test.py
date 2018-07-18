
import matplotlib.pyplot as plt

from score_following_game.agents.optimal_agent import OptimalAgent
from score_following_game.data_processing.data_pool import get_data_pool, load_game_config
from score_following_game.environment.score_following_env import ScoreFollowingEnv
from score_following_game.environment.env_wrappers import ResizeSizeObservations, SpecDifference, PrepareForNet,\
    SheetDifference, InvertSheet, ConvertToFloat

if __name__ == "__main__":
    """ main """

    # add argument parser
    import argparse

    parser = argparse.ArgumentParser(description='Test environment wrappers.')
    parser.add_argument('--data_set', help='path to train dataset.', type=str, default="test_sample")
    parser.add_argument('--game_config', help='path to game config file.', type=str,
                        default='game_configs/nottingham_continuous.yaml')
    args = parser.parse_args()

    # load game config
    config = load_game_config(args.game_config)

    # load data pool
    rl_pool = get_data_pool(config, directory=args.data_set)

    # initialize environment
    env = ScoreFollowingEnv(rl_pool, config, render_mode=None)

    # apply wrapper to env
    env = ResizeSizeObservations(env, spec_factor=1.0, sheet_factor=1.0)
    env = ConvertToFloat(env)
    # env = InvertSheet(env)
    env = SpecDifference(env)
    env = SheetDifference(env)
    env = PrepareForNet(env)

    # create agent
    agent = OptimalAgent(rl_pool)

    # iterate episodes
    time_steps = []
    for i_episode in range(10):

        # get observations
        observation = env.reset()
        episode_reward = 0
        done = False
        timestep = 0

        # initialize optimal agent
        agent.set_optimal_actions()

        # iterate until end-of-song
        while True:
            timestep += 1

            # choose action
            action = agent.perform_action(observation)

            # perform step and observe
            observation, reward, done, info = env.step(action)

            # visualize observation space
            spectrogram, sheet_img = observation
            print("spec: ", spectrogram.shape)
            print("sheet:", sheet_img.shape)

            max_stack = max(spectrogram.shape[1], sheet_img.shape[1])

            plt.figure("Observation Space")
            plt.clf()

            for i, spec in enumerate(spectrogram[0]):
                plt.subplot(max_stack, 2, (2 * i + 1))
                plt.imshow(spec, cmap="viridis", origin="lower")
                plt.xlabel(spec.shape[1])
                plt.ylabel(spec.shape[0])
                plt.colorbar()

            for i, sheet in enumerate(sheet_img[0]):
                plt.subplot(max_stack, 2, (2 * i + 2))
                plt.imshow(sheet, cmap="gray", interpolation=None)
                plt.xlabel(sheet.shape[1])
                plt.ylabel(sheet.shape[0])
                plt.colorbar()

            plt.suptitle("Step: %d, Reward: %.2f" % (timestep, reward))

            plt.draw()
            plt.pause(0.001)

            if done:
                print("done!")
                break
