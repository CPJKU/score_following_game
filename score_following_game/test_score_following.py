
# python3 test_score_following.py --params ~/mounts/home@rechenknecht5/shared/score_following_game/params/a2c-ScoreFollowingNetMSMDLCHSDeepSELU-msmd_all2_train-mutopia_lchs1_20180416_164355-matthias/best_model.pt --data_set ~/mounts/home@rechenknecht5/shared/datasets/score_following_game/msmd_all2/msmd_all2_train --game_config game_configs/mutopia_lchs1.yaml

import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from score_following_game.sf_experiment import get_make_env
from score_following_game.data_processing.data_pool import get_data_pool, load_game_config
from score_following_game.agents.networks import get_network
from score_following_game.eval_score_following import initialize_agent


# render mode for the environment ('human', 'computer')
render_mode = 'computer'


def get_opencv_bar(value, bar_heigth=500, max_value=11, color=(255, 255, 0), title=None):

    value_coord = bar_heigth - int(float(bar_heigth - 20) * value / max_value) + 20
    value_coord = np.clip(value_coord, 0, bar_heigth - 1)

    bar_img_bgr = np.zeros((bar_heigth, 100, 3), np.uint8)
    cv2.line(bar_img_bgr, (0, value_coord), (bar_img_bgr.shape[1] - 1, value_coord), color, 5)

    # write current speed to observation image
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    text = "%.2f" % value
    text_size = cv2.getTextSize(text, fontFace=font_face, fontScale=0.6, thickness=1)[0]
    text_org = (100 - text_size[0], value_coord - 6)
    cv2.putText(bar_img_bgr, text, text_org, fontFace=font_face, fontScale=0.6, color=color, thickness=1)

    if title is not None:
        text_size = cv2.getTextSize(title, fontFace=font_face, fontScale=0.6, thickness=1)[0]
        text_org = (100 // 2 - text_size[0] // 2, 20)
        cv2.putText(bar_img_bgr, title, text_org, fontFace=font_face, fontScale=0.6, color=color, thickness=1)

    return bar_img_bgr


if __name__ == "__main__":
    """ main """

    # add argument parser
    import argparse

    parser = argparse.ArgumentParser(description='Train multi-modality model.')
    parser.add_argument('--net', help='network architecture to optimize.', type=str, default=None)
    parser.add_argument('--params', help='path to parameter dump.', type=str)
    parser.add_argument('--data_set', help='path to data set.', type=str, default="test_sample")
    parser.add_argument('--game_config', help='path to game config file.', type=str, default=None)
    parser.add_argument('--dump_images', help='dump tracking images to disc.', action='store_true')
    parser.add_argument('--piece', help='select piece for testing.', type=str, default=None)
    args = parser.parse_args()

    # parse parameter string
    if args.net is None:
        args.net = args.params.split("-")[1]

    if args.game_config is None:
        args.game_config = "game_configs/%s.yaml" % args.params.split("-")[3].rsplit("_", 2)[0]

    # clean up image folder
    if args.dump_images:
        os.system("rm video_frames/*.png")

    # load game config
    config = load_game_config(args.game_config)

    # initialize data pool
    rl_pool = get_data_pool(config, directory=args.data_set, song_name=args.piece)

    # set agent type ('human', 'optimal') optimal currently not supported
    agent_type = render_mode

    # initialize environment
    env = get_make_env(rl_pool, config, render_mode=render_mode)()

    # compile network architecture
    net = get_network(args.net, env.action_space.n,
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

    # iterate episodes
    time_steps = []
    alignment_errors = []
    tempo_curve = []
    action_sequence = []

    # init network (required for faster processing later on)
    spec_shape, sheet_shape = env.observation_space.spaces
    dummy_obs = (np.zeros(spec_shape.shape), np.zeros(sheet_shape.shape))
    action = agent.perform_action(dummy_obs)

    # get observations
    episode_reward = 0
    observation = env.reset()
    print("Tracking piece:", rl_pool.get_current_song_name())

    reward = 0
    done = False
    step = 0

    while True:
        step += 1

        # choose action
        action = agent.perform_action(observation)

        # perform step and observe
        observation, reward, done, info = env.step(action)
        episode_reward += reward

        # visualize state value
        # ---------------------
        from torch.autograd import Variable

        # get dimensions of state rendition
        e = env
        while not hasattr(e, "rl_pool"):
            e = e.env

        if e.obs_image is not None:

            bar_heigth = e.obs_image.shape[0]

            spec, sheet = observation
            spec = torch.from_numpy(spec).float()
            sheet = torch.from_numpy(sheet).float()
            if use_cuda:
                spec = spec.cuda()
                sheet = sheet.cuda()
            net_out = net([Variable(spec.unsqueeze(0)), Variable(sheet.unsqueeze(0))])
            value = net_out[1]
            value = value.data.cpu()[0, 0]

            # get value function bar plot
            value_bgr = get_opencv_bar(value, bar_heigth=bar_heigth, max_value=25,
                                       color=(255, 255, 0), title="value")

            # get pixel speed bar plot
            speed_bgr = get_opencv_bar(rl_pool.sheet_speed, bar_heigth=bar_heigth, max_value=15,
                                       color=(255, 0, 255), title="speed")

            # get tracking error bar
            error_bgr = get_opencv_bar(np.abs(rl_pool.tracking_error()),
                                       bar_heigth=bar_heigth, max_value=config['sheet_context'] // 2,
                                       color=(0, 0, 255), title="error")

            # get score progress bar
            score_bgr = get_opencv_bar(episode_reward, bar_heigth=bar_heigth, max_value=rl_pool.get_current_song_timesteps(),
                                       color=(0, 255, 255), title="reward")

            # get potential score progress bar
            pot_score_bgr = get_opencv_bar(len(tempo_curve), bar_heigth=bar_heigth,
                                           max_value=rl_pool.get_current_song_timesteps(),
                                           color=(0, 255, 255), title="max")

            spacer = 255 * np.ones((bar_heigth, 5, 3), np.uint8)
            bar_img = np.concatenate((e.obs_image, spacer, spacer, spacer, value_bgr, spacer, speed_bgr, spacer, error_bgr, spacer, score_bgr, spacer, pot_score_bgr), axis=1)
            cv2.imshow("Stats Plot", bar_img)
            cv2.waitKey(1)

            if args.dump_images:
                cv2.imwrite("video_frames/%.06d.png" % step, bar_img)

            # -------------------

        # collect some stats
        alignment_errors.append(rl_pool.tracking_error())
        tempo_curve.append(rl_pool.sheet_speed)
        action_sequence.append(action)

        if done:
            break

    print("Total reward %d\n" % episode_reward)

    plt.figure("tempo_curve")
    plt.clf()

    ax = plt.subplot(2, 1, 1)
    plt.plot(range(len(tempo_curve)), tempo_curve)
    plt.xlim([-5, len(tempo_curve) + 5])
    plt.grid("on")
    plt.ylabel("pixel speed", fontsize=18)
    plt.tick_params(labelsize=16)
    ax.get_yaxis().set_label_coords(-0.1, 0.5)

    ax = plt.subplot(2, 1, 2)
    plt.plot(range(len(action_sequence)), action_sequence)
    plt.xlim([-5, len(action_sequence) + 5])
    plt.grid("on")
    plt.xlabel("time step t", fontsize=18)
    plt.ylabel("continuous action $A_t$", fontsize=18)
    plt.tick_params(labelsize=16)
    ax.get_yaxis().set_label_coords(-0.1, 0.5)

    plt.show(block=True)
