
import torch

from score_following_game.agents.a2c import A2CAgent, dict_obs_to_tensor
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


class PPOAgent(A2CAgent):

    def __init__(self, observation_space, model, logger, t_max=5, n_worker=1, gamma=0.99, gae_lambda=0.95, ppo_epoch=4,
                 epsilon=0.2, batch_size=32, evaluator=None, lr_scheduler=None):

        A2CAgent.__init__(self, observation_space=observation_space, model=model,
                          t_max=t_max, n_worker=n_worker, gamma=gamma, gae_lambda=gae_lambda, gae=True,
                          logger=logger, evaluator=evaluator, lr_scheduler=lr_scheduler)

        self.ppo_epoch = ppo_epoch
        self.epsilon = epsilon
        self.batch_size = batch_size

    def perform_update(self):

        self.model.set_train_mode()

        data = self.buffer.get()
        obs, act, adv, logp_old, ret = data['obs'], data['act'], data['adv'], data['logp'], data['ret']

        obs = dict_obs_to_tensor(obs, self.model.device)

        value_loss_epoch = 0
        policy_loss_epoch = 0
        dist_entropy_epoch = 0

        n_updates = 0

        clip = self.epsilon

        for _ in range(self.ppo_epoch):
            sampler = BatchSampler(SubsetRandomSampler(list(range(self.n_worker * self.t_max))),
                                   self.batch_size, drop_last=False)

            for indices in sampler:
                actions_batch = act[indices]
                return_batch = ret[indices]

                old_log_probs_batch = logp_old[indices]

                states_batch = {}

                for obs_key in obs:
                    states_batch[obs_key] = obs[obs_key][indices]

                model_returns = self.model(states_batch)

                policy = model_returns['policy']
                values = model_returns['value'].view(-1)

                action_log_probabilities = self.model.log_probs(policy, actions_batch)

                ratio = torch.exp(action_log_probabilities - old_log_probs_batch)

                advantage_target = adv[indices]

                surr1 = ratio * advantage_target
                surr2 = ratio.clamp(1.0 - clip, 1.0 + clip) * advantage_target

                policy_loss = -torch.min(surr1, surr2).mean(dim=0)
                dist_entropy = self.model.entropy(policy)

                value_loss = (return_batch - values).pow(2).mean()

                losses = dict(policy_loss=policy_loss, value_loss=value_loss,
                              dist_entropy=dist_entropy)

                self.model.update(losses)

                value_loss_epoch += value_loss.detach()
                policy_loss_epoch += policy_loss.detach()
                dist_entropy_epoch += dist_entropy.detach()

                n_updates += 1

        value_loss_epoch /= n_updates
        policy_loss_epoch /= n_updates
        dist_entropy_epoch /= n_updates

        # logging
        log_dict = {
            'policy_loss': policy_loss_epoch.item(),
            'value_loss': value_loss_epoch.item(),
            'entropy': dist_entropy_epoch.item(),
            'learn_rate': self.model.get_learn_rate()
        }

        return log_dict
