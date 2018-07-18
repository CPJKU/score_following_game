from __future__ import print_function

# this is required as we included our reinforcment_learning package
# as a submodule for convenience
from score_following_game.utils import init_rl_imports
init_rl_imports()


import numpy as np
from reinforcement_learning.agents.agents import Agent


class OptimalAgent(Agent):

    def __init__(self, rl_pool):
        super(OptimalAgent, self).__init__()
        self.rl_pool = rl_pool
        self.optimal_actions = []

    def set_optimal_actions(self):

        song_onsets = np.asarray(self.rl_pool.onsets[self.rl_pool.sheet_id], dtype=np.int32)
        song_spec = self.rl_pool.spectrograms[self.rl_pool.sheet_id]
        interpol_fcn = self.rl_pool.interpolationFunctions[self.rl_pool.sheet_id]

        interpolated_coords = interpol_fcn(range(song_onsets[0], song_onsets[-1]))

        current_idx = 0
        dist = []
        for i in range(1, len(interpolated_coords)):

            if interpolated_coords[i - 1] == song_onsets[current_idx]:
                current_idx += 1

            dist = np.append(dist, (interpolated_coords[i] - interpolated_coords[i - 1]))

        self.optimal_actions = np.concatenate(
            (np.zeros(int(song_onsets[0] + 1)), dist, np.zeros(int(song_spec.shape[1] - song_onsets[-1]))))

    def perform_action(self, state):
        super(OptimalAgent, self).perform_action(state)

        # reward, timestep, current_speed = state
        current_speed = self.rl_pool.sheet_speed

        timestep = self.rl_pool.curr_spec_frame+1

        if timestep < len(self.optimal_actions):
            optimal_action = self.optimal_actions[timestep] - current_speed
        else:
            # set speed to 0 if the last known action is reached
            optimal_action = - current_speed
        return [optimal_action]