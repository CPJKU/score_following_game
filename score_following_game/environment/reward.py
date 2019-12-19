import importlib
import numpy as np


class Reward:

    def __init__(self, reward_name, threshold, pool, params):

        self.reward_name = reward_name
        self.params = params
        self.threshold = threshold
        self.pool = pool

        package = importlib.import_module("score_following_game.environment.reward")
        self.reward_fnc = getattr(package, reward_name)

    def get_reward(self, abs_error):
        return self.reward_fnc(abs_error, self.threshold, self.pool, **self.params)


def triangle_reward(abs_error, threshold, pool, only_on_onset=False):

    reward = 0.0
    if not only_on_onset or pool.reached_onset_in_score():
        """ as implemented for ISMIR 2018 and TISMIR 2019 """
        reward = threshold - abs_error
        reward /= threshold
    return np.float32(reward)




