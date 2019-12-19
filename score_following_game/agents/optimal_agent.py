
class OptimalAgent:
    def __init__(self, rl_pool,  use_sheet_speed=True, scale_factor=1.):
        self.rl_pool = rl_pool

        self.use_sheet_speed = use_sheet_speed
        self.scale_factor = scale_factor

    def select_action(self, state):

        if self.use_sheet_speed:
            current_speed = self.rl_pool.sheet_speed
        else:
            current_speed = 0

        timestep = self.rl_pool.curr_perf_frame

        if not self.rl_pool.last_onset_reached():

            optimal_action = self.rl_pool.curr_song.get_true_score_position(timestep + 1) \
                             - self.rl_pool.est_score_position - current_speed

        else:
            # set action to 0 if the last known action is reached
            optimal_action = - current_speed

        optimal_action = optimal_action/self.scale_factor

        return [optimal_action]

    def play_episode(self, env, render_mode):

        alignment_errors = []
        action_sequence = []
        observation_images = []

        # get observations
        episode_reward = 0
        observation = env.reset()

        done = False

        e = env
        while not hasattr(e, 'rl_pool'):
            e = e.env

        while not done:
            # choose action
            action = self.select_action(observation)

            # perform step and observe
            observation, reward, done, info = env.step(action)

            episode_reward += reward

            # collect some stats
            alignment_errors.append(self.rl_pool.tracking_error())
            action_sequence.append(action)

            # collect all observations
            if render_mode == 'video':
                observation_images.append(e.obs_image)

        return alignment_errors, action_sequence, observation_images, episode_reward
