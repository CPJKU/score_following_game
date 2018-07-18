import gym
import numpy as np
from gym.spaces.box import Box
from gym import spaces

class FrameDifference(gym.ObservationWrapper):

    def __init__(self, env, keep_raw=True):
        """
        Returns original and delta to previous observation
        """
        gym.ObservationWrapper.__init__(self, env)
        self.prev_obs = None
        self.keep_raw = keep_raw

        # adapt the observation space
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0,0,0],
            self.observation_space.high[0,0,0],
            [2*obs_shape[0] if self.keep_raw else 1, obs_shape[1], obs_shape[2]]  # Channel, Width, Height
        )

    def observation(self, observation):
        """
        :param observation: has to be at least a vector (scalars don't work)
        :return:
        """

        if self.prev_obs is None:
            self.prev_obs = observation

        obs_diff = observation - self.prev_obs
        self.prev_obs = observation

        if self.keep_raw:
            # obs_diff = np.vstack((observation[np.newaxis],
            #                       obs_diff[np.newaxis]))
            obs_diff = np.vstack((observation,
                                  obs_diff))

        return obs_diff.astype(np.float32)


class PreProcessPong(gym.ObservationWrapper):

    def __init__(self, env):
        """
        Returns a cropped and down sampled image where the background is erased
        Adapted from http://karpathy.github.io/2016/05/31/rl/
        """
        gym.ObservationWrapper.__init__(self, env)

        self.observation_space = Box(
            self.observation_space.low[0,0,0],
            self.observation_space.high[0,0,0],
            [80, 80, 1]  # Channel, Width, Height
        )

    def observation(self, observation):

        I = observation[35:195]  # crop
        I = I[::2, ::2, 0]  # down sample by factor of 2
        I[I == 144] = 0  # erase background (background type 1)
        I[I == 109] = 0  # erase background (background type 2)

        I[I != 0] = 1  # everything else (paddles, ball) just set to 1

        return I.astype(np.float32)[..., np.newaxis]


# Wrapper adapted from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/envs.py
class WrapPyTorch(gym.ObservationWrapper):
    """
    Wrap an environment from W,H,Channel to Channel,W,H
    """
    def __init__(self, env=None):
        super(WrapPyTorch, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0,0,0],
            self.observation_space.high[0,0,0],
            [obs_shape[2], obs_shape[1], obs_shape[0]]
        )

    def observation(self, observation):

        observation = np.asarray(observation)
        return observation.transpose(2, 0, 1)


class MultiModalityWrapper(gym.ObservationWrapper):
    """
        Wraps the observation of an environment from a single observation to an observation array with one entry.
        This allows us to use a generalized (multi-modal) version of the reinforcement algorithms
    """

    def __init__(self, env=None):
        gym.ObservationWrapper.__init__(self, env)

        self.observation_space = spaces.Tuple([self.observation_space])

    def observation(self, observation):
        return [observation]
