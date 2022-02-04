
import torch

import numpy as np

from score_following_game.agents.models import dict_obs_to_tensor
from score_following_game.agents.buffer import RolloutBuffer


class A2CAgent:

    def __init__(self, observation_space, model, logger, t_max=5, n_worker=1, gamma=0.99, gae=False, gae_lambda=1.0,
                 evaluator=None, lr_scheduler=None):

        self.observation_space = observation_space
        self.model = model
        self.logger = logger

        self.evaluator = evaluator

        self.lr_scheduler = lr_scheduler
        self.gamma = gamma

        self.t_max = t_max
        self.n_worker = n_worker

        self.buffer = RolloutBuffer(obs_space=observation_space, size=t_max, device=self.model.device,
                                    gamma=gamma, n_proc=n_worker, lam=gae_lambda if gae else 1.0)

    def perform_update(self):

        self.model.set_train_mode()

        data = self.buffer.get()
        obs, act, adv, logp_old, ret = data['obs'], data['act'], data['adv'], data['logp'], data['ret']

        model_returns = self.model(dict_obs_to_tensor(obs, self.model.device))

        policy = model_returns['policy']
        values = model_returns['value'].view(-1)

        log_probs = self.model.log_probs(policy, act)

        dist_entropy = self.model.entropy(policy)

        value_loss = (ret - values).pow(2).mean(dim=0)
        policy_loss = -(adv * log_probs).mean(dim=0)

        losses = dict(policy_loss=policy_loss, value_loss=value_loss, dist_entropy=dist_entropy)

        self.model.update(losses)

        # logging
        log_dict = {
            'policy_loss': policy_loss.detach(),
            'value_loss': value_loss.detach(),
            'entropy': dist_entropy,
            'learn_rate': self.model.get_learn_rate()
        }

        return log_dict

    def train(self, env, max_steps):

        step_cnt = 0
        update_cnt = 0
        t_step = 0

        # reward bookkeeping
        episode_rewards = np.zeros(self.n_worker)
        final_rewards = np.zeros(self.n_worker)

        observation = env.reset()

        while step_cnt < max_steps:

            self.model.set_eval_mode()
            with torch.no_grad():
                # get policy, value and possibly further networks specific returns for the current state
                model_returns = self.model(dict_obs_to_tensor(observation, self.model.device))

            policy = model_returns['policy']
            value = model_returns['value'].cpu().numpy()
            action_tensor, np_actions = self.model.sample_action(policy)

            # primarily used for ppo
            log_probs = self.model.log_probs(policy, action_tensor).cpu().numpy()

            next_observation, reward, done, _ = env.step(np_actions)
            episode_rewards += reward
            self.buffer.store(observation, np_actions, reward, value, log_probs)

            # Update obs
            observation = next_observation

            t_step += 1
            step_cnt += self.n_worker

            epoch_ended = t_step == self.t_max

            if epoch_ended:
                with torch.no_grad():
                    value = self.model(dict_obs_to_tensor(observation, self.model.device))['value'].cpu().numpy()

            for proc_idx in range(self.n_worker):
                terminal = done[proc_idx]

                if terminal or epoch_ended:
                    # if trajectory didn't reach terminal state, bootstrap value target
                    if epoch_ended:
                        v = value[proc_idx]
                    else:
                        v = 0
                    self.buffer.finish_path(proc_idx, v)
                    if terminal:
                        # bookkeeping of rewards
                        final_rewards[proc_idx] = episode_rewards[proc_idx]

                        # no env reset necessary, handled implicitly by subroc_vec_env
                        episode_rewards[proc_idx] = 0

            if epoch_ended:
                log_dict = self.perform_update()
                log_dict['steps'] = step_cnt
                log_dict['avg_reward'] = np.mean(final_rewards)
                log_dict['median_reward'] = np.median(final_rewards)

                update_cnt += 1

                # logging
                self.logger.log(self.model, log_dict, "training", update_cnt)

                # evaluation
                if self.evaluator is not None:
                    stats, score = self.evaluator.evaluate(self.model, update_cnt)

                    if score is not None and self.lr_scheduler is not None:
                        self.lr_scheduler.step(score)
                        if self.model.get_learn_rate() == 0:
                            print('Training stopped')

                t_step = 0

            # stop training if learn rate scheduler stopped
            if self.lr_scheduler is not None and self.lr_scheduler.learning_stopped():
                break
