
import logging
import matplotlib.pyplot as plt
import numpy as np
import time

import cv2
from gym import Env
from gym import spaces
from gym.utils import seeding

from score_following_game.environment.audio_thread import AudioThread


logger = logging.getLogger(__name__)


class ScoreFollowingEnv(Env):

    metadata = {
        'render.modes': {'human': 'human', 'computer': 'computer'},
    }

    def __init__(self, rl_pool, config, render_mode=None):

        self.rl_pool = rl_pool
        self.actions = config["actions"]
        self.render_mode = render_mode
        self.continuous = config["continuous"]
        self.reward_window = config["reward_window"]

        # distance of tracker to true score position to fail the episode
        self.score_dist_threshold = self.rl_pool.sheet_context // 3

        self.interpolationFunction = None
        self.spectrogram_positions = []
        self.interpolated_coords = []

        # path to the audio file (for playing the audio in the background)
        self.path_to_audio = ""

        # flag that determines if the environment is executed for the first time or n
        self.first_execution = True

        self.spectrogram = None
        self.sheet_img = None

        self._seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

        self.step_id = 0
        self.frame_id = 0
        self.last_reward = None
        self.cum_reward = None
        self.time_stamp = time.time()
        self.step_times = np.zeros(25)

        # setup observation space
        sheet_shape = (1, self.rl_pool.staff_height, self.rl_pool.sheet_context)
        spec_shape = (1, self.rl_pool.frequency_bins, self.rl_pool.spec_context)

        # IMPORTANT keep ordering first insert spec then sheet
        self.observation_space = spaces.Tuple((spaces.Box(0, 255, spec_shape, dtype=np.float32),
                                               spaces.Box(0, 255, sheet_shape, dtype=np.float32)))

        # Another posibility would be to use a dictionary but this would remove the ordering
        # self.observation_space = spaces.Dict({"spec": spaces.Box(0, 255, spec_shape),
        #                                       "sheet": spaces.Box(0, 255, sheet_shape)})

        if self.continuous:
            # self.action_space = spaces.Box(-np.infty, np.infty, shape=(1,), dtype=np.float32)
            self.action_space = spaces.Discrete(1)
        else:
            self.action_space = spaces.Discrete(len(self.actions))

        self.reward_range = (0, 1)

        self.obs_image = None

    def step(self, action):

        if self.continuous:
            # clip continuous action between -1000 and 1000 TODO
            # print(action)
            # if np.isnan(action):
            #     action = 1000
            # speed_update = min(max(-1000, action),1000)

            # action is passed as array to stick to OpenAI continuous environment conventions
            speed_update = action[0]
            # speed_update = action
        else:
            assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
            speed_update = self.actions[action]

        self.rl_pool.update_sheet_speed(speed_update)

        # get current frame from "pace-maker"
        if self.render_mode:
            self.curr_frame = self.audioThread.get_current_spec_position()
            while self.prev_frame == self.curr_frame:
                self.render(mode=self.render_mode)
                self.curr_frame = self.audioThread.get_current_spec_position()
            self.prev_frame = self.curr_frame

        else:
            self.curr_frame += 1

        self.rl_pool.step(self.curr_frame)
        self.spectrogram, self.sheet_img = self.rl_pool.observe()

        self.spectrogram = np.expand_dims(self.spectrogram, 0)
        self.sheet_img = np.expand_dims(self.sheet_img, 0)

        self.state = self.spectrogram, self.sheet_img

        # check if score follower lost its target
        abs_err = int(np.abs(self.rl_pool.tracking_error()))
        target_lost = abs_err > self.score_dist_threshold

        # check if score follower reached end of song
        end_of_song = self.rl_pool.last_onset_reached()

        # compute reward
        if self.reward_window < 0:
            # reward independent of the onset
            reward = float(self.score_dist_threshold) - abs_err
            reward /= self.score_dist_threshold

        else:
            # calculate a reward > 0 only if we are inside a given window around an onset
            reward = 0.0
            if self.rl_pool.in_onset_range(self.reward_window):
                reward = float(self.score_dist_threshold) - abs_err
                reward /= self.score_dist_threshold

        # punish negative pixel speed
        if self.continuous:
            reward += np.clip(self.rl_pool.sheet_speed, -np.inf, 0)

        # end of score following
        done = False
        if target_lost or end_of_song:
            done = True
            if self.render_mode:
                self.audioThread.end_stream()

        # no reward if target is lost
        if target_lost:
            reward = 0.0

        # check if env is still used even if done
        if not done:
            pass
        elif self.steps_beyond_done is None:
            self.steps_beyond_done = 0
        else:
            if self.steps_beyond_done == 0:
                logger.warning(
                    "You are calling 'step()' even though this environment has already returned done = True."
                    " You should always call 'reset()' once you receive 'done = True'"
                    " -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1

        # compute time required for step
        self.step_times[1:] = self.step_times[0:-1]
        self.step_times[0] = time.time() - self.time_stamp
        self.time_stamp = time.time()

        self.last_reward = reward
        self.step_id += 1
        self.cum_reward += reward

        return self.state, reward, done, {}

    def reset(self):

        self.steps_beyond_done = None
        self.step_id = 0
        self.cum_reward = 0
        self.first_execution = True
        self.curr_frame = 0
        self.prev_frame = -1

        # reset data pool
        self.rl_pool.reset()
        self.path_to_audio = self.rl_pool.get_current_audio_file()

        # reset audio thread
        if self.render_mode:
            self.audioThread = AudioThread(self.path_to_audio, self.rl_pool.spectrogram_params['fps'])
            self.audioThread.start()
            self.curr_frame = self.audioThread.get_current_spec_position()

        # get first observation
        self.rl_pool.step(self.curr_frame)
        self.spectrogram, self.sheet_img = self.rl_pool.observe()

        self.spectrogram = np.expand_dims(self.spectrogram, 0)
        self.sheet_img = np.expand_dims(self.sheet_img, 0)

        self.state = self.spectrogram, self.sheet_img

        return self.state

    def render(self, mode='computer', close=False):

        if close:
            # TODO Is there anything we have to close?
            return

        # resize factor
        resz_spec = 4
        resz_imag = float(resz_spec) / 2 * float(self.spectrogram[0].shape[0]) / self.sheet_img[0].shape[0]

        # prepare image
        w, h = int(self.sheet_img[0].shape[1] * resz_imag), int(self.sheet_img[0].shape[0] * resz_imag)
        sheet_rgb = cv2.resize(self.sheet_img[0], (w, h))
        sheet_rgb = cv2.cvtColor(sheet_rgb, cv2.COLOR_GRAY2BGR)
        sheet_rgb = sheet_rgb.astype(np.uint8)

        # highlight image center
        sheet_center = sheet_rgb.shape[1] // 2
        cv2.line(sheet_rgb, (sheet_center, 25), (sheet_center, sheet_rgb.shape[0] - 25), (255, 0, 255), 3)

        font_face = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize("Agent", fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, thickness=1)[0]
        text_org = (sheet_center - text_size[0] // 2, sheet_rgb.shape[0] - 7)
        cv2.putText(sheet_rgb, "Agent", text_org, fontFace=font_face, fontScale=0.6, color=(255, 0, 255), thickness=2)

        # hide tracking lines if it is rendered for humans
        if self.metadata['render.modes']['human'] != mode:

            # visualize tracker position (true position within the score)
            true_position = int(sheet_center - (resz_imag * self.rl_pool.tracking_error()))

            font_face = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize("Target", fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, thickness=1)[0]
            text_org = (true_position - text_size[0] // 2, text_size[1] + 1)
            cv2.putText(sheet_rgb, "Target", text_org, fontFace=font_face, fontScale=0.6, color=(255, 0, 0), thickness=2)

            cv2.line(sheet_rgb, (true_position, 25), (true_position, sheet_rgb.shape[0] - 25), (255, 0, 0), 3)

            # visualize boundaries
            l_boundary = int(sheet_center - self.score_dist_threshold * resz_imag)
            r_boundary = int(sheet_center + self.score_dist_threshold * resz_imag)
            cv2.line(sheet_rgb, (l_boundary, 0), (l_boundary, sheet_rgb.shape[0] - 1), (0, 0, 255), 1)
            cv2.line(sheet_rgb, (r_boundary, 0), (r_boundary, sheet_rgb.shape[0] - 1), (0, 0, 255), 1)

        # prepare spectrogram
        spec = self.spectrogram[0][::-1, :]

        spec = cv2.resize(spec, (self.spectrogram[0].shape[1] * resz_spec, self.spectrogram[0].shape[0] * resz_spec))
        spec = plt.cm.viridis(spec)[:, :, 0:3]
        spec = (spec * 255).astype(np.uint8)
        spec_rgb = cv2.cvtColor(spec, cv2.COLOR_RGB2BGR)

        # TODO This line showed the position within the spectrogram which was no changed to rightmost, so the line would be on the border
        # cv2.line(img=spec_rgb,
        #          pt1=(int(resz_spec * self.rl_pool.spec_offset), 0),
        #          pt2=(int(resz_spec * self.rl_pool.spec_offset), spec_rgb.shape[0]),
        #          color=(0, 0, 255), thickness=1)

        # prepare observation visualization
        cols = sheet_rgb.shape[1]
        rows = sheet_rgb.shape[0] + spec_rgb.shape[0]
        obs_image = np.zeros((rows, cols, 3), dtype=np.uint8)

        # write sheet to observation image
        obs_image[0:sheet_rgb.shape[0], 0:sheet_rgb.shape[1], :] = sheet_rgb

        # write spec to observation image
        c0 = obs_image.shape[1] // 2 - spec_rgb.shape[1] // 2
        c1 = c0 + spec_rgb.shape[1]
        obs_image[sheet_rgb.shape[0]:, c0:c1, :] = spec_rgb

        # write current speed to observation image
        font_face = cv2.FONT_HERSHEY_SIMPLEX
        text = "pixel speed: " + str(self.rl_pool.sheet_speed)
        text_size = cv2.getTextSize(text, fontFace=font_face, fontScale=0.6, thickness=1)[0]
        text_org = ((obs_image.shape[1] - text_size[0] - 5), (sheet_rgb.shape[0] + text_size[1] + 3))
        cv2.putText(obs_image, text, text_org, fontFace=font_face, fontScale=0.6, color=(255, 255, 255), thickness=1)

        # write reward to observation image
        to_print = np.round(self.last_reward, 2)
        text = "last reward: " + str(to_print)
        text_size = cv2.getTextSize(text, fontFace=font_face, fontScale=0.6, thickness=1)[0]
        text_org = ((obs_image.shape[1] - text_size[0] - 5), (sheet_rgb.shape[0] + 3 * text_size[1] + 3))
        cv2.putText(obs_image, text, text_org, fontFace=font_face, fontScale=0.6, color=(255, 255, 255), thickness=1)

        # write cumulative reward (score) to observation image
        to_print = np.round(self.cum_reward, 2)
        text = "score: " + str(to_print)
        text_size = cv2.getTextSize(text, fontFace=font_face, fontScale=0.6, thickness=1)[0]
        text_org = ((obs_image.shape[1] - text_size[0] - 5), (sheet_rgb.shape[0] + 5 * text_size[1] + 3))
        cv2.putText(obs_image, text, text_org, fontFace=font_face, fontScale=0.6, color=(255, 255, 255), thickness=1)

        # preserve this for access from outside
        self.obs_image = obs_image

        # TODO think about proper render mode
        if self.render_mode is not None:
            cv2.imshow("Score Following", obs_image)
            cv2.waitKey(1)

    def close(self):
        # TODO implement
        pass

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
