
class OptimalAgent:
    def __init__(self, rl_pool,  use_sheet_speed=True, scale_factor=1.):
        self.rl_pool = rl_pool
        self.env = None

        self.use_sheet_speed = use_sheet_speed
        self.scale_factor = scale_factor

    def select_action(self, state):

        if self.use_sheet_speed:
            current_speed = self.env.sheet_speed
        else:
            current_speed = 0

        timestep = self.env.curr_frame

        if not self.env.last_onset_reached:
            optimal_action = self.env.curr_song.get_true_score_position(timestep + 1) \
                             - self.env.est_score_position - current_speed

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

        self.env = env.unwrapped

        while not done:
            # choose action
            action = self.select_action(observation)

            # perform step and observe
            observation, reward, done, info = env.step(action)

            episode_reward += reward

            # collect some stats
            alignment_errors.append(self.env.tracking_error)
            action_sequence.append(action)

            # collect all observations
            if render_mode == 'video':
                observation_images.append(self.env.obs_image)

        return alignment_errors, action_sequence, observation_images, episode_reward
