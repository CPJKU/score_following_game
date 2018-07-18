from __future__ import print_function

import numpy as np


def print_formatted_stats(stats):
    """
    Print formatted results for result tables
    """
    print("& %.2f & %.2f & %.2f & %.2f \\\\" % (stats['tracked_until_end_ratio'], stats['global_tracking_ratio'], stats['alignment_errors_mean'], stats['alignment_errors_std']))


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

    def __init__(self, make_env, evaluation_pools, config, trials=1, render_mode=None):

        self.make_env = make_env
        self.evaluation_pools = evaluation_pools
        self.config = config
        self.render_mode = render_mode
        self.trials = trials

    def evaluate(self, agent, log_writer=None, log_step=0, verbose=False):

        mean_stats = None
        for _ in range(self.trials):
            evaluation_data = []
            for pool in self.evaluation_pools:

                if verbose:
                    print(pool.get_current_song_name().ljust(60), end=" ")

                env = self.make_env(pool, self.config, render_mode=self.render_mode)

                alignment_errors = []

                # get observations
                episode_reward = 0
                observation = env.reset()

                onset_list = pool.get_current_song_onsets()

                while True:

                    # choose action
                    action = agent.perform_action(observation)

                    # perform step and observe
                    observation, reward, done, info = env.step(action)
                    episode_reward += reward

                    # keep alignment errors
                    if pool.curr_spec_frame in onset_list:
                        alignment_errors.append(pool.tracking_error())

                    if done:
                        break

                # compute number of tracked onsets
                onsets_tracked = np.sum(onset_list <= pool.curr_spec_frame)

                song_data = {'alignment_errors': alignment_errors, 'onsets_tracked': onsets_tracked,
                             'total_onsets': len(onset_list)}
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

        if log_writer is not None:
            log_writer.add_scalar('eval/alignment_errors_mean', mean_stats['alignment_errors_mean'], log_step)
            log_writer.add_scalar('eval/alignment_errors_median', mean_stats['alignment_errors_median'], log_step)
            log_writer.add_scalar('eval/alignment_errors_std', mean_stats['alignment_errors_std'], log_step)

            log_writer.add_scalar('eval/tracking_ratios_mean', mean_stats['tracking_ratios_mean'], log_step)
            log_writer.add_scalar('eval/global_tracking_ratio', mean_stats['global_tracking_ratio'], log_step)
            log_writer.add_scalar('eval/tracked_until_end_ratio', mean_stats['tracked_until_end_ratio'], log_step)

        return mean_stats
