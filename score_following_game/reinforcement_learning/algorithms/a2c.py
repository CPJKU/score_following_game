
import torch

import numpy as np

from collections import OrderedDict
from score_following_game.reinforcement_learning.algorithms.agent import Agent
from score_following_game.reinforcement_learning.torch_extentions.distributions.adapted_categorical import AdaptedCategorical


class A2CAgent(Agent):

    def __init__(self, observation_space, model, n_actions=1, t_max=5, n_worker=1, gamma=0.99, distribution=AdaptedCategorical,
                 use_cuda=torch.cuda.is_available(), gae=False, gae_lambda=0.95, log_writer=None, log_interval=10,
                 evaluator=None, eval_interval=5000, lr_scheduler=None, score_name=None, high_is_better=False,
                 dump_interval=100000, dump_dir=None, buffer=None):

        Agent.__init__(self, observation_space=observation_space, model=model,  n_actions=n_actions, gamma=gamma,
                       distribution=distribution, use_cuda=use_cuda, log_writer=log_writer, log_interval=log_interval,
                       evaluator=evaluator, eval_interval=eval_interval, lr_scheduler=lr_scheduler, score_name=score_name,
                       high_is_better=high_is_better, dump_interval=dump_interval, dump_dir=dump_dir, buffer=buffer)

        self.t_max = t_max
        self.n_worker = n_worker

        self.observations = OrderedDict()
        for obs_key in self.observation_space:

            obs_shape = self.observation_space[obs_key].shape
            self.observations[obs_key] = torch.zeros(self.t_max + 1, self.n_worker,
                                                     *[int(x) for x in list(obs_shape)]).to(self.device)

        self.rewards = []
        self.value_predictions = torch.zeros(self.t_max + 1, self.n_worker, 1).to(self.device)
        self.returns = torch.zeros(self.t_max + 1, self.n_worker, 1).to(self.device)

        self.actions = self.action_tensor(self.t_max, self.n_worker, self.n_actions).to(self.device)
        self.masks = torch.ones(self.t_max, self.n_worker, 1).to(self.device)

        # we will only store the log probs of the chose actions
        self.old_log_probs = torch.zeros(self.t_max, self.n_worker, 1).to(self.device)

        # reward bookkeeping
        self.episode_rewards = torch.zeros([self.n_worker, 1]).to(self.device)
        self.final_rewards = torch.zeros([self.n_worker, 1]).to(self.device)

        self.step = 0
        self.first_obs = False
        self.gae = gae
        self.gae_lambda = gae_lambda

    def on_done(self, state_tensor_list, np_masks):

        # If done then clean the current observation
        for key in state_tensor_list:
            pt_masks = torch.from_numpy(np_masks.reshape(np_masks.shape[0], *[1 for _ in range(
                len(state_tensor_list[key].shape[1:]))])).to(self.device)
            state_tensor_list[key] *= pt_masks

    def prepare_model_input(self, step):

        model_in = OrderedDict()

        for obs_key in self.observations:
            model_in[obs_key] = self.observations[obs_key][step]

        return model_in

    def store_step_states(self):
        # save latest state
        for obs_key in self.observations:
            self.observations[obs_key][0].copy_(self.observations[obs_key][-1])

    def select_action(self, state, train=True):
        super().select_action(state, train)

        # activate training mode
        self.model.set_train_mode()

        if len(state) == 4:
            observation, reward, done, _ = state
            reward = np.asarray([reward])
            reward = torch.from_numpy(reward).transpose(1, 0).to(self.device)

            self.episode_rewards += reward

            self.rewards.append(reward)

            np_masks = np.array([0.0 if done_ else 1.0 for done_ in done], dtype=np.float32)

            # create a list of state tensors and be sure that they are of type float
            state_tensor_list = OrderedDict()

            for obs_key in observation:
                state_tensor_list[obs_key] = torch.from_numpy(observation[obs_key]).view(self.observations[obs_key]
                                                                                         .shape[1:]).to(self.device)

            self.on_done(state_tensor_list, np_masks)

            # store observations
            for obs_key in state_tensor_list:
                self.observations[obs_key][self.step].copy_(state_tensor_list[obs_key])

            if train and self.step == self.t_max:
                self.first_obs = False
                self.perform_update()
                self.step = 0
                self.rewards = []
                self.store_step_states()

            self.masks[self.step].copy_(torch.from_numpy(np_masks).unsqueeze(1))

            # bookkeeping of rewards
            self.final_rewards *= self.masks[self.step]
            self.final_rewards += (1 - self.masks[self.step]) * self.episode_rewards
            self.episode_rewards *= self.masks[self.step]

        else:
            observation = state
            for key in observation:
                self.observations[key][0].copy_(torch.from_numpy(observation[key]))

        with torch.no_grad():
            # get policy, value and possibly further networks specific returns for the current state
            model_returns = self.model(self.prepare_model_input(self.step))

        policy = model_returns['policy']
        value = model_returns['value']

        action_tensor, np_actions = self.model.sample_action(policy)

        # primarily used for ppo
        log_probs = self.model.get_log_probs(policy, action_tensor).data

        # self.actions[self.step].copy_(action_tensor.view(-1, 1))

        if self.n_actions == 1 and action_tensor.dim() < 2:
            # otherwise we get wrong dimension
            # TODO find better solution
            action_tensor = action_tensor.unsqueeze(-1)

        self.actions[self.step].copy_(action_tensor)
        self.value_predictions[self.step].copy_(value.data)

        self.old_log_probs[self.step].copy_(log_probs)

        if train:
            self.step += 1

        return np_actions, False

    def prepare_single_forward_pass(self):

        model_in = OrderedDict()

        for obs_key in self.observations:
            obs = self.observations[obs_key]
            model_in[obs_key] = obs[:-1].view(-1, *obs.size()[2:])

        return model_in

    def perform_update(self):
        super().perform_update()

        model_returns = self.model(self.prepare_single_forward_pass())

        policy = model_returns['policy']
        values = model_returns['value']

        if self.gae:
            with torch.no_grad():
                self.value_predictions[-1] = self.model.forward_value(self.prepare_model_input(-1))

            gae = 0

            for step in reversed(range(self.t_max)):
                delta = self.rewards[step] + self.gamma * self.value_predictions[step + 1] * self.masks[step] \
                        - self.value_predictions[step]
                gae = delta + self.gamma * self.gae_lambda * self.masks[step] * gae
                self.returns[step] = gae + self.value_predictions[step]

        else:
            # calculate returns
            with torch.no_grad():
                self.returns[-1] = self.model.forward_value(self.prepare_model_input(-1))

            for step in reversed(range(self.t_max)):
                self.returns[step] = self.returns[step + 1] * self.gamma * self.masks[step] + self.rewards[step]

        advantages = self.returns[:-1].view(-1).unsqueeze(1) - values

        log_probs = self.model.get_log_probs(policy, self.actions.view(self.n_worker * self.t_max, -1)).view(-1, 1)

        dist_entropy = self.model.calc_entropy(policy)

        value_loss = advantages.pow(2).mean(dim=0)

        policy_loss = -(advantages.data * log_probs).mean(dim=0)

        losses = dict(policy_loss=policy_loss, value_loss=value_loss,
                      dist_entropy=dist_entropy)

        self.model.update(losses)

        # logging
        self.log_dict = {
            'policy_loss': policy_loss.detach(),
            'value_loss': value_loss.detach(),
            'entropy': dist_entropy,
            'avg_reward': self.final_rewards.mean(),
            'median_reward': self.final_rewards.median()
        }

