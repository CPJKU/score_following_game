import numpy as np
import time
import os
import torch
from torch.autograd import Variable
import torch.nn.functional as F


class ReinforceAgent(object):

    def __init__(self, env, model, no_baseline=False, gamma=0.99, use_cuda=torch.cuda.is_available(), max_steps=1e8):

        self.env = env
        self.model = model
        self.no_baseline = no_baseline
        self.gamma = gamma
        self.use_cuda = use_cuda
        self.action_dtype = np.long
        self.max_steps = max_steps

    def perform_action(self, state):

        if self.use_cuda:
            state = [Variable(torch.FloatTensor(s).unsqueeze(0).cuda()) for s in state]
        else:
            state = [Variable(torch.FloatTensor(s).unsqueeze(0)) for s in state]

        return self._sample_action(state)

    def update(self, episode):

        states = episode['states']
        rewards = episode['rewards']
        Gts = np.zeros(len(states), dtype=np.float32)

        # iterate collected transitions
        for t in range(len(states)):
            Gts[t] = np.sum(rewards[t:] * [self.gamma ** (i - t) for i in range(t, len(states))])

        Gts = torch.from_numpy(Gts).unsqueeze(1)
        actions = torch.from_numpy(np.asarray(episode['actions'], dtype=self.action_dtype))
        states_list = [torch.zeros((len(states), *s.shape)) for s in states[0]]

        for i, state in enumerate(states):
            for j, s in enumerate(state):
                states_list[j][i] = torch.FloatTensor(s)

        if self.use_cuda:
            actions = actions.cuda()
            states = [Variable(state).cuda() for state in states_list]
            Gts = Gts.cuda()
        else:
            states = [Variable(state) for state in states_list]

        actions = Variable(actions)
        Gts = Variable(Gts)

        policy, baseline = self.model(states)

        if self.no_baseline:
            baseline = 0

        deltas = Gts - baseline

        eligibility = self._get_log_probs(policy, actions.view(-1, 1)) * Variable(deltas.data)

        pl_loss = -eligibility.mean()
        bl_loss = (deltas ** 2).mean()

        if self.no_baseline:
            self.model.update([pl_loss])
        else:
            self.model.update([pl_loss, bl_loss])

        return pl_loss.cpu().data[0], bl_loss.cpu().data[0]


    def _sample_action(self, state):
        action_probabilities = F.softmax(self.model.forward_policy(state), dim=-1)

        # draw a random action by sampling from the policy prediction
        action = action_probabilities.multinomial().squeeze(0).cpu().data.numpy()[0]
        return action

    def _get_log_probs(self, policy, actions):
        action_probabilities = F.softmax(policy, dim=-1)
        good_probabilities = action_probabilities.gather(1, actions)
        return torch.log(good_probabilities)

    def train(self, max_updates=5000, log_writer=None, log_interval=100, evaluator=None, eval_interval=5000,
              lr_scheduler=None, score_name=None, high_is_better=False, dump_interval=100000, dump_dir=None):

        # activate training mode
        self.model.net.train()

        # best score for model evaluation
        best_score = -np.inf if high_is_better else np.inf

        # iterate episode
        rewards = np.zeros(max_updates)
        for epoch in range(1, max_updates+1):

            episode = self.generate_episode(max_steps=self.max_steps)
            policy_loss, value_loss = self.update(episode)

            # book keeping
            rewards[epoch-1] = episode['total_reward']

            # logging
            if epoch % log_interval == 0:
                if not self.no_baseline:
                    print("\nepoch: {:08d}, reward: {:.1f}, value loss {:.5f}, policy loss {:.5f}"
                          .format(epoch, rewards[epoch-1], value_loss, policy_loss))
                else:
                    print("\nepoch: {:08d}, reward: {:.1f}, policy loss {:.5f}"
                          .format(epoch, rewards[epoch-1],  policy_loss))

            if log_writer is not None and epoch % log_interval == 0:
                log_writer.add_scalar('training/avg_reward', rewards[epoch-1], int(epoch / log_interval))
                log_writer.add_scalar('training/policy_loss', policy_loss, int(epoch/log_interval))
                if not self.no_baseline:
                    log_writer.add_scalar('training/value_loss', value_loss, int(epoch/log_interval))
                log_writer.add_scalar('training/learn_rate', self.model.get_learn_rate(),
                                      int(epoch / log_interval))

            # evaluation
            if evaluator is not None and epoch % eval_interval == 0:
                self.model.net.eval()
                stats = evaluator.evaluate(self, log_writer, int(epoch / eval_interval))
                self.model.net.train()

                if score_name is not None:
                    if lr_scheduler is not None:
                        lr_scheduler.step(stats[score_name])
                        if self.model.optimizer.param_groups[0]['lr'] == 0:
                            print('Training stopped')
                            break

                    improvement = (high_is_better and stats[score_name] >= best_score) or \
                                  (not high_is_better and stats[score_name] <= best_score)

                    if improvement:
                        print('New best model at update {}'.format(epoch))
                        self.store_model('best_model.pt', dump_dir)
                        best_score = stats[score_name]

            if epoch % dump_interval == 0:
                print('Saved model at update {}'.format(epoch))
                self.store_model('model_update_{}.pt'.format(epoch), dump_dir)

    def generate_episode(self, max_steps, render=False):

        # reset environment and get initial observation
        observation = self.env.reset()

        # init total reward
        total_reward = 0

        # initialize state arrays
        states = [[] for _ in range(len(observation))]
        actions = []
        rewards = []

        # generate episode
        step_cnt = 0
        while step_cnt < max_steps:

            # some visualizations
            if render:
                self.env.render()
                time.sleep(0.001)

            action = self.perform_action(observation)

            # record state
            for i, obs in enumerate(observation):
                states[i].append(obs)

            # record action
            actions.append(action)

            # if step_cnt == 199:
            #     print()
            observation, reward, done, info = self.env.step(action)

            rewards.append(reward)
            total_reward += reward

            step_cnt += 1

            # check if environment reported terminal state
            if done:
                break

        actions = np.asarray(actions, dtype=np.float32)
        rewards = np.asarray(rewards, dtype=np.float32)

        states = [[states[j][i] for j in range(len(states))] for i in range(len(states[0]))]

        return {'states': states, 'actions': actions, 'rewards': rewards, 'total_reward': total_reward}

    def store_model(self, name, store_dir=None):

        if store_dir is not None:
            model_path = os.path.join(store_dir, name)
        else:
            model_path = name

        self.model.save_network(model_path)

