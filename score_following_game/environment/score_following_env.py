
import cv2
import logging
import time

import numpy as np

from gym import Env, spaces
from gym.utils import seeding
from score_following_game.environment.audio_thread import AudioThread
from score_following_game.environment.render_utils import write_text, prepare_sheet_for_render, prepare_spec_for_render
from score_following_game.environment.reward import Reward

logger = logging.getLogger(__name__)

AGENT_COLOR = (0, 102, 204)
TARGET_COLOR = (255, 0, 0)
TEXT_COLOR = (255, 255, 255)
BORDER_COLOR = (0, 0, 255)


class ScoreFollowingEnv(Env):
    metadata = {
        'render.modes': {'human': 'human',
                         'computer': 'computer',
                         'video': 'video'},
    }

    def __init__(self, rl_pool, config, render_mode=None):

        self.rl_pool = rl_pool
        self.actions = config["actions"]
        self.render_mode = render_mode

        # distance of tracker to true score position to fail the episode
        self.score_dist_threshold = self.rl_pool.score_shape[2] // 3

        self.interpolationFunction = None
        self.spectrogram_positions = []
        self.interpolated_coords = []
        self.spec_representation = config['spec_representation']

        self.text_position = 0

        # path to the audio file (for playing the audio in the background)
        self.path_to_audio = ""

        # flag that determines if the environment is executed for the first time or n
        self.first_execution = True

        self.performance = None
        self.score = None

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

        self.last_action = None

        # setup observation space
        self.observation_space = spaces.Dict({'perf': spaces.Box(0, 255, self.rl_pool.perf_shape, dtype=np.float32),
                                              'score': spaces.Box(0, 255, self.rl_pool.score_shape, dtype=np.float32)})

        if len(config['actions']) == 0:
            self.action_space = spaces.Box(low=-128, high=128, shape=(1,), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(len(self.actions))
        self.reward_range = (-1, 1)
        self.obs_image = None
        self.prev_reward = 0.0
        self.debug_info = {'song_history': self.rl_pool.get_song_history()}

        self.reward = Reward(config['reward_name'], threshold=self.score_dist_threshold, pool=self.rl_pool,
                             params=config['reward_params'])

        # resize factors for rendering
        self.resz_spec = 4
        self.resz_imag = float(self.resz_spec) / 2 * float(self.rl_pool.perf_shape[1]) / self.rl_pool.score_shape[1]
        self.resz_x, self.resz_y = self.resz_imag, self.resz_imag
        self.text_position = 0

    def step(self, action):

        if len(self.actions) > 0:
            # assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
            # decode action if specific action space is given
            action = self.actions[action]
        else:
            action = action[0]

        self.rl_pool.update_position(action)

        self.last_action = action

        # get current frame from "pace-maker"
        if self.render_mode == 'computer' or self.render_mode == 'human':
            self.curr_frame = self.audioThread.get_current_spec_position()

            while self.prev_frame == self.curr_frame:
                self.render(mode=self.render_mode)
                self.curr_frame = self.audioThread.get_current_spec_position()

            self.prev_frame = self.curr_frame

        elif self.render_mode == 'video':
            self.render(mode=self.render_mode)
            self.curr_frame += 1

        else:
            self.curr_frame += 1

        self.performance, self.score = self.rl_pool.step(self.curr_frame)

        self.state = dict(
            perf=self.performance,
            score=self.score
        )

        # check if score follower lost its target
        abs_err = np.abs(self.rl_pool.tracking_error())
        target_lost = abs_err > self.score_dist_threshold

        # check if score follower reached end of song
        end_of_song = self.rl_pool.last_onset_reached()

        reward = self.reward.get_reward(abs_err)

        # end of score following
        done = False
        if target_lost or end_of_song:
            done = True
            if self.render_mode == 'computer' or self.render_mode == 'human':
                self.audioThread.end_stream()

        # no reward if target is lost
        if target_lost:
            reward = np.float32(0.0)

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

        self.last_action = None

        # reset data pool
        self.rl_pool.reset()

        # reset audio thread
        if self.render_mode == 'computer' or self.render_mode == 'human':
            # write midi to wav
            import soundfile as sf
            fn_audio = self.rl_pool.get_current_song_name()[0] + '.wav'
            perf_audio, fs = self.rl_pool.get_current_perf_audio_file()
            sf.write(fn_audio, perf_audio, fs)

            self.path_to_audio = fn_audio
            self.audioThread = AudioThread(self.path_to_audio, self.rl_pool.spectrogram_params['fps'])
            self.audioThread.start()
            self.curr_frame = self.audioThread.get_current_spec_position()

        # get first observation
        self.performance, self.score = self.rl_pool.step(self.curr_frame)

        self.state = dict(
            perf=self.performance,
            score=self.score
        )

        return self.state

    def render(self, mode='computer', close=False):

        if close:
            return

        perf = self.prepare_perf_for_render()

        score = self.prepare_score_for_render()

        # highlight image center
        score_center = score.shape[1] // 2
        cv2.line(score, (score_center, 25), (score_center, score.shape[0] - 25), AGENT_COLOR, 2)

        font_face = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize("Agent", fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, thickness=1)[0]
        text_org = (score_center - text_size[0] // 2, score.shape[0] - 7)
        cv2.putText(score, "Agent", text_org, fontFace=font_face, fontScale=0.6, color=AGENT_COLOR, thickness=2)

        # hide tracking lines if it is rendered for humans
        if self.metadata['render.modes']['human'] != mode:
            # visualize tracker position (true position within the score)
            true_position = int(score_center - (self.resz_x * self.rl_pool.tracking_error()))

            font_face = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize("Target", fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, thickness=1)[0]
            text_org = (true_position - text_size[0] // 2, text_size[1] + 1)
            cv2.putText(score, "Target", text_org, fontFace=font_face, fontScale=0.6, color=TARGET_COLOR, thickness=2)

            cv2.line(score, (true_position, 25), (true_position, score.shape[0] - 25), TARGET_COLOR, 3)

            # visualize boundaries
            l_boundary = int(score_center - self.score_dist_threshold * self.resz_x)
            r_boundary = int(score_center + self.score_dist_threshold * self.resz_x)
            cv2.line(score, (l_boundary, 0), (l_boundary, score.shape[0] - 1), BORDER_COLOR, 1)
            cv2.line(score, (r_boundary, 0), (r_boundary, score.shape[0] - 1), BORDER_COLOR, 1)

        # prepare observation visualization
        cols = score.shape[1]
        rows = score.shape[0] + perf.shape[0]
        obs_image = np.zeros((rows, cols, 3), dtype=np.uint8)

        # write sheet to observation image
        obs_image[0:score.shape[0], 0:score.shape[1], :] = score

        # write spec to observation image
        c0 = obs_image.shape[1] // 2 - perf.shape[1] // 2
        c1 = c0 + perf.shape[1]
        obs_image[score.shape[0]:, c0:c1, :] = perf

        # draw black line to separate score and performance
        obs_image[score.shape[0], c0:c1, :] = 0

        # write text to the observation image
        self._write_text(obs_image=obs_image, pos=self.text_position, color=TEXT_COLOR)

        # preserve this for access from outside
        self.obs_image = obs_image

        # show image
        if self.render_mode == 'computer' or self.render_mode == 'human':
            cv2.imshow("Score Following", self.obs_image)
            cv2.waitKey(1)

    def close(self):
        pass

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _write_text(self, obs_image, pos, color):

        # write reward to observation image
        write_text('reward: {:6.2f}'.format(self.last_reward if self.last_reward is not None else 0),
                   pos, obs_image, color=color)

        # write cumulative reward (score) to observation image
        write_text("score: {:6.2f}".format(self.cum_reward if self.cum_reward is not None else 0),
                   pos + 2, obs_image, color=color)

        # write last action
        write_text("action: {:+6.1f}".format(self.last_action), pos + 4, obs_image, color=color)

    def prepare_score_for_render(self):
        return prepare_sheet_for_render(self.score, resz_x=self.resz_imag, resz_y=self.resz_imag)

    def prepare_perf_for_render(self):
        return prepare_spec_for_render(self.performance, resz_spec=self.resz_spec)




