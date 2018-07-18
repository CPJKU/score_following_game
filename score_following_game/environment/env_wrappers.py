import cv2
import gym
import numpy as np
from gym import spaces


class ConvertToFloat(gym.ObservationWrapper):

    def __init__(self, env):
        """
        Invert color of sheet image
        """
        gym.ObservationWrapper.__init__(self, env)

    def observation(self, observation):

        # unfold observation vector
        spectrogram, sheet_img = observation
        sheet_img = sheet_img.astype(np.float32)
        sheet_img /= 255.0
        return spectrogram, sheet_img


class InvertSheet(gym.ObservationWrapper):

    def __init__(self, env):
        """
        Invert color of sheet image
        """
        gym.ObservationWrapper.__init__(self, env)

    def observation(self, observation):

        # unfold observation vector
        spectrogram, sheet_img = observation
        sheet_img = 1.0 - sheet_img
        return spectrogram, sheet_img


class PrepareForNet(gym.ObservationWrapper):

    def __init__(self, env):
        """
        Prepare observations for neural networks
        """
        gym.ObservationWrapper.__init__(self, env)

    def observation(self, observation):

        # unfold observation vector
        spectrogram, sheet_img = observation

        while spectrogram.ndim < 4:
            spectrogram = spectrogram[np.newaxis]

        while sheet_img.ndim < 4:
            sheet_img = sheet_img[np.newaxis]

        return spectrogram, sheet_img


class SpecDifference(gym.ObservationWrapper):

    def __init__(self, env, keep_raw=True):
        """
        Returns original and delta spectrogram
        """
        gym.ObservationWrapper.__init__(self, env)
        self.prev_spec = None
        self.keep_raw = keep_raw

        # spec_shape = list(self.observation_space.spaces['spec'].shape)
        spec_space, sheet_space = self.observation_space.spaces
        spec_shape = list(spec_space.shape)

        # adapt the observation space
        spec_shape[0] = 2 if self.keep_raw else 1

        # IMPORTANT keep ordering first spec then sheet
        self.observation_space = spaces.Tuple((spaces.Box(0, 255, spec_shape, dtype=np.float32), sheet_space))
        # self.observation_space = spaces.Dict({"spec": spaces.Box(0, 255, spec_shape),
        #                                       "sheet": self.observation_space.spaces['sheet']})

    def observation(self, observation):

        # unfold observation vector
        spectrogram, sheet_img = observation

        if self.prev_spec is None:
            self.prev_spec = spectrogram

        spectrogram_diff = spectrogram - self.prev_spec
        self.prev_spec = spectrogram

        if self.keep_raw:
            spectrogram_diff = np.vstack((spectrogram, spectrogram_diff))

        return spectrogram_diff, sheet_img


class SheetDifference(gym.ObservationWrapper):

    def __init__(self, env, keep_raw=True):
        """
        Returns original and delta sheet image
        """
        gym.ObservationWrapper.__init__(self, env)
        self.prev_sheet = None
        self.keep_raw = keep_raw

        # adapt the observation space
        # sheet_shape = list(self.observation_space.spaces['sheet'].shape)
        spec_space, sheet_space = self.observation_space.spaces
        sheet_shape = list(sheet_space.shape)

        # adapt the observation space
        sheet_shape[0] = 2 if self.keep_raw else 1
        # self.observation_space = spaces.Dict({"spec": self.observation_space.spaces['spec'],
        #                                       "sheet": spaces.Box(0, 255, sheet_shape)})

        # IMPORTANT keep ordering first spec then sheet
        self.observation_space = spaces.Tuple((spec_space, spaces.Box(0, 255, sheet_shape, dtype=np.float32)))

    def observation(self, observation):

        # unfold observation vector
        spectrogram, sheet_img = observation

        if self.prev_sheet is None:
            self.prev_sheet = sheet_img

        sheet_diff = sheet_img - self.prev_sheet
        self.prev_sheet = sheet_img

        if self.keep_raw:
            sheet_diff = np.vstack((sheet_img, sheet_diff))

        return spectrogram, sheet_diff


class ResizeSizeObservations(gym.ObservationWrapper):

    def __init__(self, env, spec_factor=1.0, sheet_factor=1.0):
        """
        Scale size of observations (sheet, spec) by certain factor
        """
        gym.ObservationWrapper.__init__(self, env)
        self.spec_factor = spec_factor
        self.sheet_factor = sheet_factor

        # sheet_shape = list(self.observation_space.spaces['sheet'].shape)
        # spec_shape = list(self.observation_space.spaces['spec'].shape)

        spec_space, sheet_space = self.observation_space.spaces
        sheet_shape = list(sheet_space.shape)
        spec_shape = list(spec_space.shape)

        sheet_shape[1] *= sheet_factor
        sheet_shape[2] *= sheet_factor

        spec_shape[1] *= spec_factor
        spec_shape[2] *= spec_factor

        spec_shape = np.asarray(spec_shape, dtype=np.int)
        sheet_shape = np.asarray(sheet_shape, dtype=np.int)

        # IMPORTANT keep ordering first insert spec then sheet
        # self.observation_space = spaces.Dict({"spec": spaces.Box(0, 255, spec_shape),
        #                                       "sheet": spaces.Box(0, 255, sheet_shape)})

        # IMPORTANT keep ordering first spec then sheet
        self.observation_space = spaces.Tuple((spaces.Box(0, 255, spec_shape, dtype=np.float32),
                                               spaces.Box(0, 255, sheet_shape, dtype=np.float32)))

    def observation(self, observation):

        # unfold observation vector
        spectrogram, sheet_img = observation

        # resize spectrogram
        if self.spec_factor != 1.0:
            s = spectrogram.shape
            s = [int(d * self.spec_factor) for d in s[1:]]
            spectrogram = cv2.resize(spectrogram.transpose(1,2,0), (s[1], s[0]))
            spectrogram = np.expand_dims(spectrogram, 0)

        # resize sheet image
        if self.sheet_factor != 1.0:
            s = sheet_img.shape
            s = [int(d * self.sheet_factor) for d in s[1:]]
            sheet_img = cv2.resize(sheet_img.transpose(1, 2, 0), (s[1], s[0]))
            sheet_img = np.expand_dims(sheet_img, 0)

        return spectrogram, sheet_img


class ScaleRewardEnv(gym.RewardWrapper):

    def __init__(self, env, factor):
        """
        Scale reward to smaller range (should help learning)
        """
        gym.RewardWrapper.__init__(self, env)
        self.factor = factor

    def reward(self, reward):
        return reward * self.factor
