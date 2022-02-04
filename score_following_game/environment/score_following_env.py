
import cv2

import numpy as np

from gym import Env, spaces
from gym.utils import seeding
from score_following_game.environment.render_utils import write_text, prepare_sheet_for_render, prepare_spec_for_render


AGENT_COLOR = (0, 102, 204)
TARGET_COLOR = (255, 0, 0)
TEXT_COLOR = (255, 255, 255)
BORDER_COLOR = (0, 0, 255)


class ScoreFollowingEnv(Env):

    def __init__(self, song_pool, config, seed=None, render_mode=None):

        self.song_pool = song_pool

        # parse config
        self.score_excerpt_shape = tuple(config['score_shape'])
        self.perf_excerpt_shape = tuple(config['perf_shape'])
        self.spectrogram_params = config['spectrogram_params']

        self.actions = config["actions"]
        self.render_mode = render_mode

        # distance of tracker to true score position to fail the episode
        self.score_dist_threshold = self.score_excerpt_shape[2] // 3

        self.seed(seed)

        self.curr_song, self.curr_frame = None, None
        self.sheet_speed = None
        self.state, self.score, self.performance = None, None, None

        self.last_reward, self.cum_reward, self.last_action = None, None, None
        self.est_score_position, self.true_score_position = None, None

        # setup observation space
        self.observation_space = spaces.Dict({'perf': spaces.Box(0, 255, self.perf_excerpt_shape, dtype=np.float32),
                                              'score': spaces.Box(0, 255, self.score_excerpt_shape, dtype=np.float32)})

        if len(config['actions']) == 0:
            self.action_space = spaces.Box(low=-128, high=128, shape=(1,), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(len(self.actions))

        self.reward_range = (-1, 1)
        self.obs_image = None
        self.prev_reward = 0.0

        # resize factors for rendering
        self.resz_spec = 4
        self.resz_score = float(self.resz_spec) / 2 * float(self.perf_excerpt_shape[1]) / self.score_excerpt_shape[1]

    def step(self, action):

        if len(self.actions) > 0:
            # decode action if specific action space is given
            action = self.actions[action]
        else:
            action = action[0]

        # perform action, i.e., update sheet speed
        self.sheet_speed += action

        self.last_action = action

        # get current frame from "pace-maker"
        if self.render_mode == 'video':
            self.render(mode=self.render_mode)

        self.curr_frame += 1

        # update estimated score position
        self.est_score_position = self.clip_coord(self.est_score_position + self.sheet_speed, self.curr_song.score)

        # get true score position from annotations
        self.true_score_position = self.curr_song.get_true_score_position(self.curr_frame)

        # get current excerpts
        self.performance, self.score = self.curr_song.get_excerpts(self.curr_frame, self.est_score_position)

        # self.performance, self.score = self.get_excerpts()

        self.state = {'perf': self.performance, 'score': self.score}

        # check if score follower lost its target
        abs_err = np.abs(self.tracking_error)
        target_lost = abs_err > self.score_dist_threshold

        reward = np.float32((self.score_dist_threshold - abs_err)/self.score_dist_threshold)

        # end of score following
        done = target_lost or self.end_of_song

        # no reward if target is lost
        if target_lost:
            reward = np.float32(0.0)

        self.last_reward = reward
        self.cum_reward += reward

        return self.state, reward, done, {}

    def reset(self):

        self.cum_reward = 0
        self.last_action = None

        # rand_song_id = np.random.randint(self.song_pool.get_length())
        rand_song_id = self.np_random.randint(self.song_pool.get_length())
        self.curr_song = self.song_pool.get_song(rand_song_id)

        # reset the current performance frame and set the padding for the performance
        self.sheet_speed = 0

        self.est_score_position = self.clip_coord(int(self.curr_song.coords[0]),
                                                  self.curr_song.score)
        self.true_score_position = int(self.curr_song.coords[0])

        # start song at first onset
        self.curr_frame = self.curr_song.first_onset

        # get first observation
        self.performance, self.score = self.curr_song.get_excerpts(self.curr_frame, self.est_score_position)

        self.state = {'perf': self.performance, 'score': self.score}

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

        # visualize tracker position (true position within the score)
        true_position = int(score_center - (self.resz_score * self.tracking_error))

        font_face = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize("Target", fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, thickness=1)[0]
        text_org = (true_position - text_size[0] // 2, text_size[1] + 1)
        cv2.putText(score, "Target", text_org, fontFace=font_face, fontScale=0.6, color=TARGET_COLOR, thickness=2)

        cv2.line(score, (true_position, 25), (true_position, score.shape[0] - 25), TARGET_COLOR, 3)

        # visualize boundaries
        l_boundary = int(score_center - self.score_dist_threshold * self.resz_score)
        r_boundary = int(score_center + self.score_dist_threshold * self.resz_score)
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
        self._write_text(obs_image=obs_image)

        # preserve this for access from outside
        self.obs_image = obs_image

    @property
    def tracking_error(self):
        """Compute distance between score and performance position."""

        # error should go negative when estimate is behind
        return self.est_score_position - self.true_score_position

    @property
    def last_onset_reached(self):
        return self.curr_frame >= self.curr_song.last_onset

    @property
    def end_of_song(self):
        return self.curr_frame >= self.curr_song.num_of_frames

    @property
    def max_timesteps(self):
        return self.curr_song.num_of_frames

    def clip_coord(self, coord, sheet):
        """
        Clip coordinate to be within sheet bounds
        """

        coord = np.max([coord, self.score_excerpt_shape[2] // 2])
        coord = np.min([coord, sheet.shape[2] - self.score_excerpt_shape[2] // 2 - 1])

        return coord

    def close(self):
        pass

    def seed(self, seed=None):
        # Seed the random number generator
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _write_text(self, obs_image, pos=0, color=TEXT_COLOR):

        # write reward to observation image
        write_text('reward: {:6.2f}'.format(self.last_reward if self.last_reward is not None else 0),
                   pos, obs_image, color=color)

        # write cumulative reward (score) to observation image
        write_text("score: {:6.2f}".format(self.cum_reward if self.cum_reward is not None else 0),
                   pos + 2, obs_image, color=color)

        # write last action
        write_text("action: {:+6.1f}".format(self.last_action), pos + 4, obs_image, color=color)

    def prepare_score_for_render(self):
        return prepare_sheet_for_render(self.score, resz_x=self.resz_score, resz_y=self.resz_score)

    def prepare_perf_for_render(self):
        return prepare_spec_for_render(self.performance, resz_spec=self.resz_spec)
