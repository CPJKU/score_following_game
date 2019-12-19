import torch

from collections import OrderedDict
from score_following_game.reinforcement_learning.algorithms.a2c import A2CAgent
from score_following_game.reinforcement_learning.algorithms.agent import Agent
from score_following_game.reinforcement_learning.torch_extentions.distributions.adapted_categorical import AdaptedCategorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


class PPOAgent(A2CAgent):

    def __init__(self, observation_space, model, n_actions=1, t_max=5, n_worker=1, gamma=0.99, gae_lambda=0.95, ppo_epoch=4,
                 epsilon=0.2, clip_value=False, batch_size=32, distribution=AdaptedCategorical, use_cuda=torch.cuda.is_available(),
                 log_writer=None, log_interval=10, evaluator=None, eval_interval=5000, lr_scheduler=None,
                 score_name=None, high_is_better=False, dump_interval=100000, dump_dir=None, buffer=None):

        A2CAgent.__init__(self, observation_space=observation_space, model=model, n_actions=n_actions, t_max=t_max, n_worker=n_worker,
                          gamma=gamma, gae_lambda=gae_lambda, gae=True, distribution=distribution, use_cuda=use_cuda,
                          log_writer=log_writer, log_interval=log_interval, evaluator=evaluator, eval_interval=eval_interval,
                          lr_scheduler=lr_scheduler, score_name=score_name, high_is_better=high_is_better,
                          dump_interval=dump_interval, dump_dir=dump_dir, buffer=buffer)

        self.ppo_epoch = ppo_epoch
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.clip_value = clip_value

        self.alpha = 1

    def perform_update(self):
        Agent.perform_update(self)

        with torch.no_grad():
            self.value_predictions[-1] = self.model.forward_value(self.prepare_model_input(-1))

        gae = 0

        for step in reversed(range(self.t_max)):
            delta = self.rewards[step] + self.gamma * self.value_predictions[step + 1] * self.masks[step] \
                    - self.value_predictions[step]
            gae = delta + self.gamma * self.gae_lambda * self.masks[step] * gae
            self.returns[step] = gae + self.value_predictions[step]

        advantages = self.returns[:-1] - self.value_predictions[:-1]

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        value_loss_epoch = 0
        policy_loss_epoch = 0
        dist_entropy_epoch = 0
        explained_variance_epoch = 0

        n_updates = 0

        clip = self.epsilon * self.alpha

        for _ in range(self.ppo_epoch):
            sampler = BatchSampler(SubsetRandomSampler(list(range(self.n_worker * self.t_max))),
                                   self.batch_size, drop_last=False)

            for indices in sampler:
                actions_batch = self.actions.view(self.n_worker * self.t_max, -1)[indices]
                return_batch = self.returns[:-1].view(-1, 1)[indices]
                old_log_probs_batch = self.old_log_probs.view(-1, *self.old_log_probs.size()[2:])[indices]

                model_returns = self.model(self.prepare_batch_input(indices))

                policy = model_returns['policy']
                values = model_returns['value']

                action_log_probabilities = self.model.get_log_probs(policy, actions_batch)

                ratio = torch.exp(action_log_probabilities - old_log_probs_batch)

                advantage_target = advantages.view(-1, 1)[indices]

                surr1 = ratio * advantage_target
                surr2 = ratio.clamp(1.0 - clip, 1.0 + clip) * advantage_target

                policy_loss = -torch.min(surr1, surr2).mean(dim=0)
                dist_entropy = self.model.calc_entropy(policy)

                # clip value loss according to
                # https://github.com/openai/baselines/tree/master/baselines/ppo2
                if self.clip_value:
                    value_preds_batch = self.value_predictions[:-1].view(-1, 1)[indices]
                    value_pred_clipped = value_preds_batch + \
                                         (values - value_preds_batch).clamp(-clip, clip)
                    value_losses = (return_batch - values).pow(2)
                    value_losses_clipped = (return_batch - value_pred_clipped).pow(2)
                    value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean(dim=0)
                else:
                    value_loss = 0.5*(return_batch - values).pow(2).mean(dim=0)

                losses = dict(policy_loss=policy_loss, value_loss=value_loss,
                              dist_entropy=dist_entropy)

                self.model.update(losses)

                value_loss_epoch += value_loss.item()
                policy_loss_epoch += policy_loss.item()
                dist_entropy_epoch += dist_entropy.item()
                explained_variance_epoch += ((1 - (return_batch-values.detach()).var())
                                             / return_batch.var()).item()
                n_updates += 1

        value_loss_epoch /= n_updates
        policy_loss_epoch /= n_updates
        dist_entropy_epoch /= n_updates
        explained_variance_epoch /= n_updates

        # logging
        self.log_dict = {
            'policy_loss': policy_loss_epoch,
            'value_loss': value_loss_epoch,
            'entropy': dist_entropy_epoch,
            'explained_var': explained_variance_epoch,
            'avg_reward': self.final_rewards.mean(),
            'median_reward': self.final_rewards.median(),
            'ppo_epsilon': clip
        }

    def prepare_batch_input(self, indices):

        states_batch = OrderedDict()

        for obs_key in self.observations:
            obs = self.observations[obs_key]
            states_batch[obs_key] = obs[:-1].view(-1, *obs.size()[2:])[indices]

        return states_batch
