
import copy
import numpy as np

PXL2CM = 0.035277778


def print_formatted_stats(stats):
    """
    Print formatted results for result tables
    """

    print("& {:.2f} & {:.2f} & {:.2f} & {:.2f} \\\\" .format(np.mean(stats['tracked_until_end_ratio']),
                                                             np.mean(stats['global_tracking_ratio']),
                                                             np.mean(stats['alignment_errors_mean'])*PXL2CM,
                                                             np.mean(stats['alignment_errors_std'])*PXL2CM))


def compute_alignment_stats(evaluation_data):
    """
    Compute alignment stats
    """
    alignment_errors = []
    tracking_ratios = []
    tracked_until_end = 0
    tracked_onsets = 0
    total_onsets = 0

    for date_entry in evaluation_data:
        alignment_errors += date_entry['alignment_errors']
        tracking_ratios.append(date_entry['onsets_tracked'] / float(date_entry['total_onsets']))

        if date_entry['onsets_tracked'] == date_entry['total_onsets']:
            tracked_until_end += 1

        tracked_onsets += date_entry['onsets_tracked']
        total_onsets += date_entry['total_onsets']

    alignment_errors = np.asarray(alignment_errors)
    abs_alignment_errors = np.abs(alignment_errors)
    tracking_ratios = np.asarray(tracking_ratios)

    ae_mean, ae_median, ae_std = -1, -1, -1
    if len(abs_alignment_errors) > 0:
        ae_mean = abs_alignment_errors.mean()
        ae_median = np.median(abs_alignment_errors)
        ae_std = abs_alignment_errors.std()

    tracking_ratios_mean = tracking_ratios.mean()
    tracked_to_end_ratio = tracked_until_end / float(len(evaluation_data))
    global_tracking_ratio = float(tracked_onsets) / total_onsets

    stats = dict()
    stats['alignment_errors_mean'] = ae_mean
    stats['alignment_errors_median'] = ae_median
    stats['alignment_errors_std'] = ae_std
    stats['tracking_ratios_mean'] = tracking_ratios_mean
    stats['global_tracking_ratio'] = global_tracking_ratio
    stats['tracked_until_end_ratio'] = tracked_to_end_ratio

    return stats


class Evaluator:

    def __init__(self, make_env, evaluation_pools, config, logger, seed, trials=1,
                 render_mode=None, eval_interval=5000):
        self.make_env = make_env
        self.evaluation_pools = evaluation_pools
        self.config = config
        self.render_mode = render_mode
        self.trials = trials
        self.eval_interval = eval_interval
        self.logger = logger
        self.seed = seed

    def _eval_pool(self, agent, pool, verbose):

        env = self.make_env(pool, self.config, seed=self.seed, render_mode=self.render_mode)

        alignment_errors = []

        # get observations
        episode_reward = 0
        observation = env.reset()

        if verbose:
            print(env.unwrapped.curr_song.song_name)

        onset_list = env.unwrapped.curr_song.perf_onsets
        while True:

            # choose action
            action = agent.select_action(observation)

            # perform step and observe
            observation, reward, done, info = env.step(action)
            episode_reward += reward

            # keep alignment errors, only store tracking error if an onset occurs
            if env.unwrapped.curr_frame in onset_list:
                alignment_errors.append(env.unwrapped.tracking_error)

            if done:
                break

        # compute number of tracked onsets
        onsets_tracked = np.sum(onset_list <= env.unwrapped.curr_frame)

        song_data = {'alignment_errors': alignment_errors, 'onsets_tracked': onsets_tracked,
                     'total_onsets': len(onset_list)}

        return song_data

    # def evaluate(self, agent, log_writer, log_step=0, verbose=False):
    def evaluate(self, agent, step=0, verbose=False):
        raise NotImplementedError


class PerformanceEvaluator(Evaluator):

    def __init__(self, make_env, evaluation_pools, config, seed, logger=None, trials=1, render_mode=None,
                 eval_interval=5000, score_name=None, high_is_better=False):
        Evaluator.__init__(self, make_env, evaluation_pools, config, logger, seed, trials, render_mode, eval_interval)

        self.score_name = score_name
        self.high_is_better = high_is_better
        self.best_score = -np.inf if self.high_is_better else np.inf

    def evaluate(self, agent, step=0, verbose=False):
        mean_stats = None
        score = None

        if step % self.eval_interval == 0 and step > 0:
            agent.set_eval_mode()
            for _ in range(self.trials):
                evaluation_data = []
                for pool in self.evaluation_pools:

                    song_data = self._eval_pool(agent, pool, verbose)
                    evaluation_data.append(song_data)

                    if verbose:
                        song_stats = compute_alignment_stats([song_data])
                        string = "tracking ratio: %.2f" % song_stats['global_tracking_ratio']
                        if song_stats['global_tracking_ratio'] == 1.0:
                            string += " +"
                        print(string)

                # compute alignment stats
                stats = compute_alignment_stats(evaluation_data)

                stats['evaluation_data'] = evaluation_data

                if mean_stats is None:
                    mean_stats = dict()
                    for key in stats.keys():
                        if key != "evaluation_data":
                            mean_stats[key] = []

                for key in mean_stats.keys():
                    mean_stats[key].append(stats[key])

            for key in mean_stats.keys():
                mean_stats[key] = np.mean(mean_stats[key])

            if self.logger is not None:
                self.logger.log_scalars(mean_stats, "eval", int(step / self.eval_interval))

            if self.score_name is not None:
                score = mean_stats[self.score_name]

                improvement = (self.high_is_better and score >= self.best_score) or \
                              (not self.high_is_better and score <= self.best_score)

                if improvement:
                    self.best_score = score
                    if self.logger is not None:
                        self.logger.store_model(agent, "best_model")

        return mean_stats, score


class EmbeddingEvaluator(Evaluator):
    def __init__(self, make_env, evaluation_pools, config, seed, logger=None, trials=1, render_mode=None,
                 eval_interval=5000):
        Evaluator.__init__(self, make_env, evaluation_pools, config, logger, seed, trials, render_mode, eval_interval)

        self.embedding = None

    def store_embedding(self, module, input_, output_):
        self.embedding = input_[0]

    def register_hook(self, net):
        embedding_layer = net._modules.get('policy_fc')
        embedding_layer.register_forward_hook(self.store_embedding)

    def _eval_pool(self, agent, pool, verbose):

        self.register_hook(agent.model.net)

        env = self.make_env(pool, self.config, render_mode=self.render_mode)

        plain_env = self.make_env(copy.deepcopy(pool), self.config, render_mode=self.render_mode).unwrapped

        plain_env.reset()

        if verbose:
            print(plain_env.curr_song.song_name)

        # get observations
        observation = env.reset()

        return_dicts = {'state': [],
                        'value': [],
                        'embedding': [],
                        'onsets_in_state': [],
                        'target_lost': [],
                        'song_name': [],
                        'tracking_error': [],
                        'speed': []}

        song_onsets = plain_env.curr_song.perf_onsets
        while True:

            # choose action
            action = agent.select_action(observation)

            # perform step and observe
            observation, reward, done, info = env.step(action)

            cur_perf_frame = plain_env.curr_frame
            in_len = plain_env.perf_excerpt_shape[-1]
            onsets_in_input = len(list(filter(lambda o: cur_perf_frame-in_len <= o <= cur_perf_frame, song_onsets)))

            # perform a step in the plain env to get the original observation
            obs_org, r, d, _ = plain_env.step(action)

            return_dicts['state'].append(obs_org)
            return_dicts['value'].append(agent.predict_value(observation))
            return_dicts['embedding'].append(self.embedding.cpu().data.numpy())
            return_dicts['onsets_in_state'].append(onsets_in_input)
            return_dicts['target_lost'].append(done)
            return_dicts['song_name'].append(plain_env.curr_song.song_name)
            return_dicts['tracking_error'].append(plain_env.tracking_error)
            return_dicts['speed'].append(plain_env.sheet_speed)

            if done:
                break

        tue = np.sum(song_onsets <= plain_env.curr_frame) == len(song_onsets)
        return_dicts['tue'] = [tue for _ in range(len(return_dicts['state']))]

        return return_dicts

    def evaluate(self, agent, step=0, verbose=False):

        return_dicts = {'state': [],
                        'value': [],
                        'embedding': [],
                        'onsets_in_state': [],
                        'tue': [],
                        'target_lost': [],
                        'song_name': [],
                        'tracking_error': [],
                        'speed': []}

        for _ in range(self.trials):
            for pool in self.evaluation_pools:

                res = self._eval_pool(agent, pool, verbose)

                return_dicts['state'].extend(res['state'])
                return_dicts['value'].extend(res['value'])
                return_dicts['embedding'].extend(res['embedding'])
                return_dicts['onsets_in_state'].extend(res['onsets_in_state'])
                return_dicts['tue'].extend(res['tue'])
                return_dicts['target_lost'].extend(res['target_lost'])
                return_dicts['song_name'].extend(res['song_name'])
                return_dicts['tracking_error'].extend(res['tracking_error'])
                return_dicts['speed'].extend(res['speed'])

        return return_dicts
