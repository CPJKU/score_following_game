import torch

import torch.nn as nn

from torch.distributions.utils import logits_to_probs


def dict_obs_to_tensor(obs, device):

    tensor_obs = {}
    for space in obs.keys():
        tensor_obs[space] = torch.as_tensor(obs[space], dtype=torch.float32, device=device).contiguous()
        if tensor_obs[space].dim() == 3:
            # add batch dim if necessary
            tensor_obs[space] = tensor_obs[space].unsqueeze(0)

    return tensor_obs


class Model(nn.Module):
    """
    Template for creating models used in the reinforcement learning algorithms
    """
    def __init__(self, net, optimizer, max_grad_norm=0.5, value_coef=0.5, entropy_coef=0.01,
                 device=torch.device("cuda")):
        super(Model, self).__init__()

        self.net = net
        self.optimizer = optimizer
        self.max_grad_norm = max_grad_norm
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.device = device

    def forward(self, x):
        return self.net.forward(**x)

    def update(self, losses):

        self.optimizer.zero_grad()
        (losses['policy_loss'] +
         self.value_coef * losses['value_loss'] -
         self.entropy_coef * losses['dist_entropy']).backward()

        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)

        self.optimizer.step()

    def set_train_mode(self):
        self.net.train()

    def set_eval_mode(self):
        self.net.eval()

    def get_learn_rate(self):
        # return learn rate of optimizer for logging
        return self.optimizer.param_groups[0]['lr']

    def save_network(self, name):
        torch.save(self.net.state_dict(), name+'.pt')

    def select_action(self, state, deterministic=False):

        with torch.no_grad():
            policy = self.forward(dict_obs_to_tensor(state, self.device))['policy']

        return self.sample_action(policy, deterministic=deterministic)[1][0]

    def predict_value(self, state):

        with torch.no_grad():
            value = self.forward(dict_obs_to_tensor(state, self.device))['value'].cpu().numpy()[0, 0]

        return value

    @staticmethod
    def entropy(policy):
        """
        adapted from
        https://pytorch.org/docs/stable/distributions.html#torch.distributions.categorical.Categorical.log_prob
        """
        logits = policy - policy.logsumexp(dim=-1, keepdim=True)
        probs = logits_to_probs(logits)

        min_real = torch.finfo(logits.dtype).min
        logits = torch.clamp(logits, min=min_real)
        p_log_p = logits * probs
        return -p_log_p.sum(-1).mean()

    @staticmethod
    def log_probs(policy, actions):
        """
        adapted from
        https://pytorch.org/docs/stable/distributions.html#torch.distributions.categorical.Categorical.entropy
        """

        logits = policy - policy.logsumexp(dim=-1, keepdim=True)

        value = actions.long().unsqueeze(-1)
        value, log_pmf = torch.broadcast_tensors(value, logits)
        value = value[..., :1]

        return log_pmf.gather(-1, value).squeeze(-1)

    @staticmethod
    def sample_action(policy, deterministic=False):

        if deterministic:
            actions = policy.argmax(-1)
        else:
            """
            adapted from
            https://pytorch.org/docs/stable/distributions.html#torch.distributions.categorical.Categorical.sample
            """
            # normalize logits
            logits = policy - policy.logsumexp(dim=-1, keepdim=True)
            probs = logits_to_probs(logits)

            num_events = probs.size()[-1]

            probs_2d = probs.reshape(-1, num_events)
            actions = torch.multinomial(probs_2d, 1, True).T.reshape(probs.shape[0])

        return actions, actions.cpu().detach().numpy()
