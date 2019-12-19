
import copy
import cv2
import os
import torch

import matplotlib.cm as cm
import numpy as np

from matplotlib.colors import LinearSegmentedColormap, Normalize
from score_following_game.agents.human_agent import HumanAgent
from score_following_game.agents.optimal_agent import OptimalAgent
from score_following_game.agents.networks_utils import get_network
from score_following_game.data_processing.data_pools import get_single_song_pool
from score_following_game.data_processing.utils import load_game_config
from score_following_game.environment.score_following_env import ScoreFollowingEnv
from score_following_game.environment.render_utils import prepare_sheet_for_render, prepare_spec_for_render
from score_following_game.experiment_utils import initialize_trained_agent, get_make_env, make_env_tismir,\
    setup_evaluation_parser
from score_following_game.integrated_gradients import IntegratedGradients, prepare_grad_for_render
from score_following_game.reinforcement_learning.algorithms.models import Model
from score_following_game.utils import render_video, get_opencv_bar


# render mode for the environment ('human', 'computer', 'video')
# render_mode = 'computer'
render_mode = 'video'
mux_audio = True

if __name__ == "__main__":
    """ main """

    parser = setup_evaluation_parser()
    parser.add_argument('--agent_type', help='which agent to test [rl|optimal|human].',
                        choices=['rl', 'optimal', 'human'], type=str, default="rl")
    parser.add_argument('--plot_stats', help='plot additional statistics.', action='store_true', default=False)
    parser.add_argument('--plot_ig', help='plot integrated gradients.', action='store_true', default=False)

    args = parser.parse_args()


    if args.agent_type == 'rl':
        exp_name = os.path.basename(os.path.split(args.params)[0])

        if args.net is None:
            args.net = exp_name.split('-')[1]

        if args.game_config is None:
            args.game_config = 'game_configs/{}.yaml'.format(exp_name.split("-")[3].rsplit("_", 2)[0])

    config = load_game_config(args.game_config)

    if args.agent_type == 'optimal':
        # the action space for the optimal agent needs to be continuous
        config['actions'] = []

    pool = get_single_song_pool(
        dict(config=config, song_name=args.piece, directory=args.data_set, real_perf=args.real_perf))

    observation_images = []

    # initialize environment
    env = make_env_tismir(pool, config, render_mode='human' if args.agent_type == 'human' else 'video')

    if args.agent_type == 'human' or args.agent_type == 'optimal':

        agent = HumanAgent(pool) if args.agent_type == 'human' else OptimalAgent(pool)
        alignment_errors, action_sequence, observation_images, episode_reward = agent.play_episode(env, render_mode)

    else:

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

        agent = initialize_trained_agent(model, use_cuda=use_cuda, deterministic=False)

        observation_images = []

        # get observations
        episode_reward = 0
        observation = env.reset()

        reward = 0
        done = False

        if args.plot_ig:
            IG = IntegratedGradients(net, 'cuda', steps=20)
            plain_env = get_make_env(copy.deepcopy(pool), config, make_env_fnc=make_env_tismir, render_mode=render_mode)()

            while not isinstance(plain_env, ScoreFollowingEnv):
                plain_env = plain_env.env

            plain_env.reset()

        colors = ['#a50026', '#d73027', '#f46d43', '#fdae61', '#fee090', '#000000', '#e0f3f8', '#abd9e9', '#74add1',
                  '#4575b4', '#313695']
        colors = list(reversed(colors))
        cmap = LinearSegmentedColormap.from_list('cmap', colors)
        norm = Normalize(vmin=-1, vmax=1)

        pos_grads = []
        neg_grads = []
        abs_grads = []
        values = []
        tempo_curve = []

        while True:
            # choose action
            action = agent.select_action(observation)
            # perform step and observe
            observation, reward, done, info = env.step(action)
            episode_reward += reward

            if env.obs_image is not None:

                bar_img = env.obs_image

                if args.plot_ig or args.plot_stats:
                    observation_tensor = agent.prepare_state(observation)
                    model_return = model(observation_tensor)

                    bar_height = env.obs_image.shape[0]
                    spacer = 255 * np.ones((bar_height, 5, 3), np.uint8)

                    if args.plot_ig:
                        obs_org, r, d, _ = plain_env.step(action)

                        # invert and create grayscale score
                        org_score = 1 - obs_org['score'][0]
                        org_score = np.uint8(cm.gray(org_score) * 255)

                        # create grayscale perf
                        org_perf = np.uint8(cm.gray(obs_org['perf'][0]) * 255)

                        # get gradients
                        guided_score_grads, guided_perf_grads = IG.generate_gradients([observation_tensor['perf'],
                                                                                       observation_tensor['score']])
                        # prepare saliency map for score and delta-score
                        grads_score = guided_score_grads[0]
                        grads_score = prepare_grad_for_render(grads_score, (config['score_shape'][2], config['score_shape'][1]), norm, cmap)

                        # prepare saliency map for performance and delta-performance
                        grads_perf = guided_perf_grads[0]
                        grads_perf = prepare_grad_for_render(grads_perf, (config['perf_shape'][2], config['perf_shape'][1]),
                                                             norm, cmap)

                        # add score gradients to score
                        added_image_score = cv2.addWeighted(grads_score, 1.0, org_score[:, :, :-1], 0.4, 0)
                        added_image_score = prepare_sheet_for_render(added_image_score, plain_env.resz_x, plain_env.resz_y, transform_to_bgr=False)

                        # add performance gradients to performance
                        added_image_perf = cv2.addWeighted(grads_perf, 1.0, org_perf[:, :, :-1], 0.4, 0)
                        added_image_perf = prepare_spec_for_render(added_image_perf, plain_env.resz_spec, transform_to_bgr=False)

                        org_img = copy.copy(env.obs_image)
                        env.obs_image[0:added_image_score.shape[0], 0:added_image_score.shape[1], :] = added_image_score

                        c0 = env.obs_image.shape[1] // 2 - added_image_perf.shape[1] // 2
                        c1 = c0 + added_image_perf.shape[1]
                        env.obs_image[added_image_score.shape[0]:, c0:c1, :] = added_image_perf

                        # bar_img = np.concatenate((e.obs_image, spacer, spacer, spacer, value_bgr, spacer, speed_bgr, spacer, error_bgr, spacer, score_bgr, spacer, pot_score_bgr), axis=1)
                        bar_img = np.concatenate((bar_img, spacer, org_img), axis=1)

                    if args.plot_stats:

                        value = model_return['value'].detach().cpu().item()
                        values.append(value)
                        tempo_curve.append(pool.sheet_speed)

                        # get value function bar plot
                        value_bgr = get_opencv_bar(value, bar_heigth=bar_height, max_value=25,
                                                   color=(255, 255, 0), title="value")

                        # get pixel speed bar plot
                        speed_bgr = get_opencv_bar(pool.sheet_speed, bar_heigth=bar_height, min_value=-15, max_value=15,
                                                   color=(255, 0, 255), title="speed")

                        # get tracking error bar
                        error_bgr = get_opencv_bar(np.abs(pool.tracking_error()),
                                                   bar_heigth=bar_height, max_value=config['score_shape'][-1] // 2,
                                                   color=(0, 0, 255), title="error")

                        # get score progress bar
                        score_bgr = get_opencv_bar(episode_reward, bar_heigth=bar_height, max_value=pool.get_current_song_timesteps(),
                                                   color=(0, 255, 255), title="reward")

                        # get potential score progress bar
                        pot_score_bgr = get_opencv_bar(len(tempo_curve), bar_heigth=bar_height,
                                                       max_value=pool.get_current_song_timesteps(),
                                                       color=(0, 255, 255), title="max")

                        bar_img = np.concatenate((bar_img, spacer, value_bgr, spacer, speed_bgr, spacer, error_bgr, spacer,
                                                  score_bgr, spacer, pot_score_bgr), axis=1)

                if render_mode == 'video':
                    observation_images.append(bar_img)
                else:
                    cv2.imshow("Stats Plot", bar_img)
                    cv2.waitKey(1)

            if done:
                break

    # write video
    if args.agent_type != 'human' and render_mode == 'video':
        render_video(observation_images, pool, fps=config['spectrogram_params']['fps'], mux_audio=mux_audio,
                     real_perf=args.real_perf)
