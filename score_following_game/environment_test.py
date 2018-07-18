
import numpy as np
import sys
import signal

from pynput import keyboard

from score_following_game.agents.human_agent import HumanAgent
from score_following_game.agents.optimal_agent import OptimalAgent
from score_following_game.data_processing.data_pool import get_data_pool, load_game_config
from score_following_game.environment.score_following_env import ScoreFollowingEnv

# render mode for the environment ('human', 'computer')
render_mode = 'computer'


def on_press(key):
    global render_mode

    try:
        if key.char == 'r':
            render_mode = 'human' if render_mode == 'computer' else 'computer'

    except AttributeError:
        pass


if __name__ == "__main__":
    """ main """

    # add argument parser
    import argparse

    parser = argparse.ArgumentParser(description='Test environment.')
    parser.add_argument('--data_set', help='path to dataset.', type=str, default="test_sample")
    parser.add_argument('--game_config', help='path to game config file.', type=str,
                        default="game_configs/nottingham.yaml")
    parser.add_argument('--agent_type', help='agent type (optimal or human).', type=str, default='human')
    args = parser.parse_args()

    # load game config
    config = load_game_config(args.game_config)

    # initialize data pool
    rl_pool = get_data_pool(config, directory=args.data_set)

    # initialize environment
    env = ScoreFollowingEnv(rl_pool, config, render_mode=render_mode)

    def signal_handler(signal, frame):
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    with keyboard.Listener(on_press=on_press) as listener:

        # create Agent
        if args.agent_type == 'human':
            agent = HumanAgent()
        else:
            agent = OptimalAgent(rl_pool)

        # iterate episodes
        time_steps = []
        for i_episode in range(1):

            # get observations
            episode_reward = 0
            observation = env.reset()

            if args.agent_type == 'optimal':
                agent.set_optimal_actions()

            reward = 0
            done = False
            timestep = 0

            while True:
                timestep += 1

                # choose action
                action = agent.perform_action(observation)

                # perform step and observe
                observation, reward, done, info = env.step(action)
                episode_reward += reward

                if done:
                    time_steps.append(timestep)
                    done_msg = "Episode finished after %d time steps (avg %.1f)" \
                               % (timestep, np.mean(time_steps))

                    print(done_msg)
                    break

            episode_msg = "Episode reward %d\n" % episode_reward
            print(episode_msg)
