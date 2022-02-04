import torch

import numpy as np

from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


class ReinforceAgent:

    def __init__(self, model, logger, no_baseline=False, batch_size=1, gamma=0.99, evaluator=None, lr_scheduler=None):

        self.model = model
        self.logger = logger
        self.evaluator = evaluator
        self.lr_scheduler = lr_scheduler

        self.gamma = gamma
        self.no_baseline = no_baseline
        self.batch_size = batch_size

    def perform_update(self, states, rewards, actions):

        self.model.set_train_mode()
        rewards = np.asarray(rewards, dtype=np.float32)

        actions = np.asarray(actions)
        T = len(actions)
        mean_policy_loss = 0
        mean_bl_loss = 0

        Gts = np.zeros((T, 1), dtype=np.float32)

        # iterate collected transitions
        for t in range(T):
            Gts[t] = np.sum(rewards[t:] * [self.gamma ** (i - t) for i in range(t, T)])

        sampler = BatchSampler(SubsetRandomSampler(list(range(T))), self.batch_size, drop_last=False)

        for k in states.keys():
            states[k] = np.asarray(states[k])

        i = 0
        for indices in sampler:

            Gt = torch.from_numpy(Gts[indices]).to(self.model.device)
            At = torch.LongTensor(actions[indices]).to(self.model.device)

            St = {}

            for k in states.keys():
                St[k] = torch.from_numpy(states[k][indices]).to(self.model.device)

            model_returns = self.model(St)

            policy = model_returns['policy']

            baseline = 0 if self.no_baseline else model_returns['value']

            delta = Gt - baseline

            eligibility = self.model.log_probs(policy, At) * delta.data

            policy_loss = -eligibility.mean(dim=0)
            bl_loss = (delta ** 2).mean(dim=0)

            if self.no_baseline:
                bl_loss = 0

            self.model.update({'policy_loss': policy_loss, 'value_loss': bl_loss, 'dist_entropy': 0})

            mean_bl_loss += bl_loss
            mean_policy_loss += policy_loss

            i += 1

        mean_policy_loss /= i
        mean_bl_loss /= i

        # logging
        log_dict = {
            'policy_loss': mean_policy_loss.detach(),
            'learn_rate': self.model.get_learn_rate()
        }

        if not self.no_baseline:
            log_dict['value_loss'] = mean_bl_loss.detach()

        return log_dict

    def train(self, env, max_steps):

        observation_space = env.observation_space.spaces

        step_cnt = 0
        update_cnt = 0

        rewards = []
        actions = []
        episode_reward = 0
        states = {k: [] for k in observation_space}

        state = env.reset()
        done = False

        while step_cnt < max_steps:
            self.model.set_eval_mode()

            if done:
                # update
                log_dict = self.perform_update(states, rewards, actions)

                log_dict['avg_reward'] = episode_reward
                log_dict['steps'] = step_cnt

                # logging
                self.logger.log(self.model, log_dict, "training", update_cnt)

                # evaluation
                if self.evaluator is not None:
                    stats, score = self.evaluator.evaluate(self.model, update_cnt)

                    if score is not None and self.lr_scheduler is not None:
                        self.lr_scheduler.step(score)
                        if self.model.get_learn_rate() == 0:
                            print('Training stopped')

                update_cnt += 1

                # clean up
                states = {k: [] for k in observation_space}
                rewards = []
                actions = []
                episode_reward = 0

                state = env.reset()
                done = False

            else:

                state_tensor = {}
                for k in state.keys():

                    # create state tensor
                    state_tensor[k] = torch.from_numpy(state[k]).float().unsqueeze(0).to(self.model.device)

                    # store state for update
                    states[k].append(state[k])

                model_returns = self.model(state_tensor)

                action = self.model.sample_action(model_returns['policy'])[1][0]
                actions.append(action)

                state, reward, done, _ = env.step(action)
                rewards.append(reward)
                episode_reward += reward

            step_cnt += 1

            # stop training if learn rate scheduler stopped
            if self.lr_scheduler is not None and self.lr_scheduler.learning_stopped():
                break
