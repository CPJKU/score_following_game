import torch

from torch.distributions import Categorical


class AdaptedCategorical(Categorical):

    def __init__(self, probs=None, logits=None, validate_args=None):
        super().__init__(probs, logits, validate_args)

    def entropy(self):
        return super().entropy().unsqueeze(-1)

    def log_prob(self, value):
        # TODO ugly workaround for shape mismatch
        return super().log_prob(value.squeeze(-1)).unsqueeze(-1)

    def sample(self, sample_shape=torch.Size()):
        return super().sample(sample_shape)

    @property
    def mean(self):
        # define the mean as the deterministic argmax choice
        return self.probs.argmax(dim=-1)
