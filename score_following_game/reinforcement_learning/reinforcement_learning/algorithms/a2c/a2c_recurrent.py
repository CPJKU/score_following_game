import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import os
import math
from torch.distributions import Normal
import time
import sys


class A2CAgent(object):

    def __init__(self, env, model, t_max=5, n_worker=1, gamma=0.99, use_cuda=torch.cuda.is_available()):

        self.env = env

        self.model = model

        self.t_max = t_max
        self.n_worker = n_worker
        self.gamma = gamma

        self.use_cuda = use_cuda
        self.action_tensor = torch.LongTensor

    # adapted from https://github.com/ethancaballero/pytorch-a2c-ppo/blob/master/main.py
    def train(self, max_updates=5000, log_writer=None, log_interval=100, evaluator=None, eval_interval=5000,
              lr_scheduler=None, score_name=None, high_is_better=False, dump_interval=100000, dump_dir=None):

        # activate training mode
        self.model.net.train()

        # get the shape of all observations and create a list containing a tensor for all of them
        observation_shapes = [space.shape for space in self.env.observation_space.spaces]
        observations = [torch.zeros(self.t_max + 1, self.n_worker, *[int(x) for x in list(obs_shape)]) for obs_shape in observation_shapes]

        cell_states = torch.zeros(self.t_max + 1, self.model.net.lstm_layers, self.n_worker,  self.model.net.lstm_size)
        hidden_states = torch.zeros(self.t_max + 1, self.model.net.lstm_layers, self.n_worker,  self.model.net.lstm_size)

        # store first observation
        first_observations = self.env.reset()
        for i, obs in enumerate(first_observations):
            observations[i][0].copy_(torch.from_numpy(obs).float())

        rewards = torch.zeros(self.t_max, self.n_worker, 1)
        value_predictions = torch.zeros(self.t_max + 1, self.n_worker, 1)
        returns = torch.zeros(self.t_max + 1, self.n_worker, 1)

        actions = self.action_tensor(self.t_max, self.n_worker)
        masks = torch.zeros(self.t_max, self.n_worker, 1)

        # reward bookkeeping
        episode_rewards = torch.zeros([self.n_worker, 1])
        final_rewards = torch.zeros([self.n_worker, 1])

        # best score for model evaluation
        best_score = -np.inf if high_is_better else np.inf

        # some timing and logging
        steps = 0
        now = after = time.time()
        step_times = np.ones(11, dtype=np.float32)

        if self.use_cuda:
            observations = [obs.cuda() for obs in observations]
            cell_states = cell_states.cuda()
            hidden_states = hidden_states.cuda()
            rewards = rewards.cuda()
            value_predictions = value_predictions.cuda()
            returns = returns.cuda()
            actions = actions.cuda()
            masks = masks.cuda()

        for i in range(1, max_updates+1):

            # estimate updates per second (running avg)
            step_times[0:-1] = step_times[1::]
            step_times[-1] = time.time() - after
            ups = 1.0 / step_times.mean()
            after = time.time()
            print("update %d @ %.1fups" % (np.mod(i, log_interval), ups), end="\r")
            sys.stdout.flush()

            for step in range(self.t_max):
                steps += 1

                policy, value, (hidden_state, cell_state) = self.model([(Variable(obs[step]),
                                                                         (Variable(hidden_states[step]),
                                                                         Variable(cell_states[step]))) for obs in observations])

                actions[step], np_actions = self._sample_action(policy)

                state, reward, done, _ = self.env.step(np_actions)

                # create a list of state tensors and be sure that they are of type float
                # change shape to the observation shape if necessary (Pendulum-v0)
                state_tensor_list = [torch.from_numpy(s).float().cuda().view(observations[idx].shape[1:])
                                     if self.use_cuda else torch.from_numpy(s).float().view(observations[idx].shape[1:])
                                     for idx, s in enumerate(state)]

                reward = torch.from_numpy(reward).float().view(rewards.shape[1:])
                episode_rewards += reward

                np_masks = np.array([0.0 if done_ else 1.0 for done_ in done])

                # If done then clean the current observation
                for j in range(len(state_tensor_list)):

                    # build mask for current part of observation
                    pt_masks = torch.from_numpy(np_masks.reshape(np_masks.shape[0], *[1 for _ in range(
                        len(state_tensor_list[j].shape[1:]))])).float()
                    if self.use_cuda:
                        pt_masks = pt_masks.cuda()

                    state_tensor_list[j] *= pt_masks

                # mask hidden/cell state
                lstm_mask = torch.from_numpy(np_masks).unsqueeze(1).unsqueeze(0).float()

                if self.use_cuda:
                    lstm_mask = lstm_mask.cuda()

                hidden_state = hidden_state.data * lstm_mask
                cell_state = cell_state.data * lstm_mask

                # store hidden/cell states
                hidden_states[step+1].copy_(hidden_state)
                cell_states[step+1].copy_(cell_state)

                # store observations, values, rewards and masks
                for j, obs in enumerate(state_tensor_list):
                    observations[j][step + 1].copy_(obs)

                value_predictions[step].copy_(value.data)
                rewards[step].copy_(reward)
                masks[step].copy_(torch.from_numpy(np_masks).unsqueeze(1))

                # bookkeeping of rewards
                final_rewards *= masks[step].cpu()
                final_rewards += (1 - masks[step].cpu()) * episode_rewards
                episode_rewards *= masks[step].cpu()

            # calculate returns
            returns[-1] = self.model.forward_value([(Variable(obs[-1]),
                                                    (Variable(hidden_states[-1]),
                                                     Variable(cell_states[-1]))) for obs in observations]).data
            for step in reversed(range(self.t_max)):
                returns[step] = returns[step + 1] * self.gamma * masks[step] + rewards[step]

            # reshape to do a single forward pass for all steps
            # policy, values, (_, _) = self.model([(Variable(obs[:-1].view(-1, *obs.size()[2:])),
            #                                      (Variable(hidden_states[:-1].view(-1, self.model.net.lstm_size)),
            #                                       Variable(cell_states[:-1].view(-1, self.model.net.lstm_size)))) for obs in observations])

            # policy, values, (_, _) = self.model([(Variable(obs[:-1].view(-1, *obs.size()[2:])),
            #                                      (Variable(hidden_states[:-1].view(hidden_states[:-1].shape[1],
            #                                                                        hidden_states[:-1].shape[0]*hidden_states[:-1].shape[2],
            #                                                                        -1)),
            #                                       Variable(cell_states[:-1].view(cell_states[:-1].shape[1],
            #                                                                      cell_states[:-1].shape[0] *
            #                                                                      cell_states[:-1].shape[2],
            #                                                                      -1)))) for obs in observations])

            # reshape for lstm
            hidden_reshaped = hidden_states[:-1].transpose(0, 1).contiguous().view(hidden_states[:-1].shape[1],
                                                                                   hidden_states[:-1].shape[0] *
                                                                                   hidden_states[:-1].shape[2],
                                                                                   -1)

            cell_reshaped = cell_states[:-1].transpose(0, 1).contiguous().view(cell_states[:-1].shape[1],
                                                                               cell_states[:-1].shape[0] *
                                                                               cell_states[:-1].shape[2],
                                                                               -1)

            policy, values, (_, _) = self.model([(Variable(obs[:-1].view(-1, *obs.size()[2:])),
                                                 (Variable(hidden_reshaped), Variable(cell_reshaped))) for obs in observations])

            log_probs = self._get_log_probs(policy, Variable(actions.view(-1).unsqueeze(1)))

            dist_entropy = self._calc_entropy(policy)

            advantages = Variable(returns[:-1].view(-1).unsqueeze(1)) - values

            value_loss = advantages.pow(2).mean()

            action_loss = -(Variable(advantages.data) * log_probs).mean()

            self.model.update([action_loss, value_loss, dist_entropy])

            # save latest state
            for j, obs in enumerate(observations):
                observations[j][0].copy_(obs[-1])

            hidden_states[0].copy_(hidden_states[-1])
            cell_states[0].copy_(cell_states[-1])

            # logging
            if i % log_interval == 0:
                print("Updates {} ({:.1f}s),  mean/median reward {:.1f}/{:.1f}, entropy {:.5f}, value loss {:.5f},"
                      " policy loss {:.5f}".format(i, time.time() - now, final_rewards.mean(), final_rewards.median(),
                                                   -dist_entropy.data[0], value_loss.data[0], action_loss.data[0]))
                now = time.time()

            if log_writer is not None and i % log_interval == 0:
                log_writer.add_scalar('training/avg_reward', final_rewards.mean(), int(i/log_interval))
                log_writer.add_scalar('training/policy_loss', action_loss, int(i/log_interval))
                log_writer.add_scalar('training/value_loss', value_loss, int(i/log_interval))
                log_writer.add_scalar('training/entropy', -dist_entropy, int(i/log_interval))
                log_writer.add_scalar('training/learn_rate', self.model.optimizer.param_groups[0]['lr'],
                                      int(i/log_interval))
                log_writer.add_scalar('training/steps', steps, int(i / log_interval))

            if evaluator is not None and i % eval_interval == 0:
                from reinforcement_learning.agents.agents import TrainedLSTMAgent
                lstm_agent = TrainedLSTMAgent(self.model.net)
                stats = evaluator.evaluate(lstm_agent, log_writer, int(i / eval_interval))

                if score_name is not None:
                    if lr_scheduler is not None:
                        lr_scheduler.step(stats[score_name])
                        if self.model.optimizer.param_groups[0]['lr'] == 0:
                            print('Training stopped')
                            break

                    improvement = (high_is_better and stats[score_name] >= best_score) or \
                                  (not high_is_better and stats[score_name] <= best_score)

                    if improvement:
                        print('New best model at update {}'.format(i))
                        self.store_model('best_model.pt', dump_dir)
                        best_score = stats[score_name]

            if i % dump_interval == 0:
                print('Saved model at update {}'.format(i))
                self.store_model('model_update_{}.pt'.format(i), dump_dir)

    def _calc_entropy(self, policy):
        probabilities = F.softmax(policy, dim=-1)
        log_probs = F.log_softmax(policy, dim=-1)
        return -(log_probs * probabilities).sum(-1).mean()

    def _get_log_probs(self, policy, actions):
        log_probs = F.log_softmax(policy, dim=-1)
        return log_probs.gather(1, actions)

    def _sample_action(self, policy):

        probabilities = F.softmax(policy, dim=-1)

        actions = probabilities.multinomial().data.cpu()

        return actions, actions.squeeze(1).cpu().numpy()

    def perform_action(self, state):

        state = [Variable(torch.from_numpy(s).cuda().float().unsqueeze(0)) if self.use_cuda
                 else Variable(torch.from_numpy(s).float().unsqueeze(0)) for s in state]

        policy = self.model.forward_policy(state)

        # 1 because we want to get the numpy array and 0 because we have to unpack the value
        return self._sample_action(policy)[1][0]

    def store_model(self, name, store_dir=None):

        if store_dir is not None:
            model_path = os.path.join(store_dir, name)
        else:
            model_path = name

        self.model.save_network(model_path)


class ContinuousA2CAgent(A2CAgent):

    def __init__(self, env, model, t_max=5, n_worker=1, gamma=0.99, use_cuda=torch.cuda.is_available(),
                 distribution=Normal):
        A2CAgent.__init__(self, env, model, t_max, n_worker, gamma, use_cuda)

        self.action_tensor = torch.FloatTensor
        self.distribution = distribution

    def _calc_entropy(self, policy):
        # TODO update pytorch such that distribution.entropy is usable
        # distr = self.distribution(**policy)
        # return distr.entropy()

        return (0.5 + 0.5 * math.log(2 * math.pi) + torch.log(policy['std'])).mean()

    def _get_log_probs(self, policy, actions):
        # create a distribution from the parameters, policy has to be a dictionary
        distr = self.distribution(**policy)
        return distr.log_prob(actions)

    def _sample_action(self, policy):

        # transform variables to tensors policy has to be a dictionary
        distribution_params = {key: policy[key].data for key in policy}

        # create a distribution from the parameters
        distr = self.distribution(**distribution_params)

        # draw a random action by sampling from the policy prediction
        actions = distr.sample()

        # expand as continuous environments need each action inside of an array
        return actions, np.expand_dims(actions.view(-1).cpu().numpy(), 1)
