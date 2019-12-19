
import torch
import torch.nn as nn
import torch.nn.functional as F

from score_following_game.agents.networks_utils import weights_init, num_flat_features


class ScoreFollowingNetMSMDLCHSDeepDoLight(nn.Module):

    def __init__(self,  n_actions, perf_shape, score_shape):
        super(ScoreFollowingNetMSMDLCHSDeepDoLight, self).__init__()

        # spec part
        self.spec_conv1 = nn.Conv2d(perf_shape[0], out_channels=16, kernel_size=3, stride=1, padding=0)
        self.spec_conv2 = nn.Conv2d(self.spec_conv1.out_channels, out_channels=16, kernel_size=3, stride=1, padding=0)

        self.spec_conv3 = nn.Conv2d(self.spec_conv2.out_channels, out_channels=32, kernel_size=3, stride=2, padding=0)
        self.spec_conv4 = nn.Conv2d(self.spec_conv3.out_channels, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.spec_do4 = nn.Dropout(p=0.2)

        self.spec_conv5 = nn.Conv2d(self.spec_conv4.out_channels, out_channels=64, kernel_size=3, stride=2, padding=0)

        self.spec_conv6 = nn.Conv2d(self.spec_conv5.out_channels, out_channels=96, kernel_size=3, stride=2, padding=0)
        self.spec_conv7 = nn.Conv2d(self.spec_conv6.out_channels, out_channels=96, kernel_size=1, stride=1, padding=0)
        self.spec_do7 = nn.Dropout(p=0.2)

        self.spec_fc = nn.Linear(2016, 512)

        # sheet part
        self.sheet_conv1 = nn.Conv2d(score_shape[0], out_channels=16, kernel_size=5, stride=(1, 2), padding=0)
        self.sheet_conv2 = nn.Conv2d(self.sheet_conv1.out_channels, out_channels=16, kernel_size=3, stride=1, padding=0)

        self.sheet_conv3 = nn.Conv2d(self.sheet_conv2.out_channels, out_channels=32, kernel_size=3, stride=2, padding=0)
        self.sheet_conv4 = nn.Conv2d(self.sheet_conv3.out_channels, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.sheet_do4 = nn.Dropout(p=0.2)

        self.sheet_conv5 = nn.Conv2d(self.sheet_conv4.out_channels, out_channels=32, kernel_size=3, stride=2, padding=0)

        self.sheet_conv6 = nn.Conv2d(self.sheet_conv5.out_channels, out_channels=64, kernel_size=3, stride=2, padding=0)
        self.sheet_do6 = nn.Dropout(p=0.2)

        self.sheet_conv7 = nn.Conv2d(self.sheet_conv6.out_channels, out_channels=96, kernel_size=3, stride=2, padding=0)
        self.sheet_conv8 = nn.Conv2d(self.sheet_conv7.out_channels, out_channels=96, kernel_size=1, stride=1, padding=0)
        self.sheet_do8 = nn.Dropout(p=0.2)

        self.sheet_fc = nn.Linear(1728, 512)

        # multi-modal part
        self.concat_fc = nn.Linear(512 + 512, 512)

        self.policy_fc = nn.Linear(self.concat_fc.out_features, 256)
        self.policy_fc_do = nn.Dropout(p=0.2)
        self.policy_out = nn.Linear(self.policy_fc.out_features, n_actions)

        self.critic_fc = nn.Linear(self.concat_fc.out_features, 256)
        self.critic_fc_do = nn.Dropout(p=0.2)
        self.critic_out = nn.Linear(self.critic_fc.out_features, 1)

        self.apply(weights_init)

    def forward(self, perf, score):

        spec_x = F.elu(self.spec_conv1(perf))
        spec_x = F.elu(self.spec_conv2(spec_x))
        spec_x = F.elu(self.spec_conv3(spec_x))
        spec_x = F.elu(self.spec_conv4(spec_x))
        spec_x = self.spec_do4(spec_x)
        spec_x = F.elu(self.spec_conv5(spec_x))
        spec_x = F.elu(self.spec_conv6(spec_x))
        spec_x = F.elu(self.spec_conv7(spec_x))
        spec_x = self.spec_do7(spec_x)

        spec_x = spec_x.view(-1, num_flat_features(spec_x))  # flatten
        spec_x = F.elu(self.spec_fc(spec_x))

        sheet_x = F.elu(self.sheet_conv1(score))
        sheet_x = F.elu(self.sheet_conv2(sheet_x))
        sheet_x = F.elu(self.sheet_conv3(sheet_x))
        sheet_x = F.elu(self.sheet_conv4(sheet_x))
        sheet_x = self.sheet_do4(sheet_x)
        sheet_x = F.elu(self.sheet_conv5(sheet_x))
        sheet_x = F.elu(self.sheet_conv6(sheet_x))
        sheet_x = self.sheet_do6(sheet_x)
        sheet_x = F.elu(self.sheet_conv7(sheet_x))
        sheet_x = F.elu(self.sheet_conv8(sheet_x))
        sheet_x = self.sheet_do8(sheet_x)

        sheet_x = sheet_x.view(-1, num_flat_features(sheet_x))  # flatten
        sheet_x = F.elu(self.sheet_fc(sheet_x))

        cat_x = torch.cat((spec_x, sheet_x), dim=1)

        cat_x = F.elu(self.concat_fc(cat_x))

        policy = F.elu(self.policy_fc(cat_x))
        policy = self.policy_fc_do(policy)
        policy = self.policy_out(policy)

        critic = F.elu(self.critic_fc(cat_x))
        critic = self.critic_fc_do(critic)
        critic = self.critic_out(critic)

        # if we use softmax and the logarithmic softmax PyTorch recommends to use the log_softmax function
        # therefore we return the linear output of the policy and apply the softmax/log softmax where it is necessary
        return {'policy': {'logits': policy}, 'value': critic}


class ScoreFollowingNetNottinghamLS(nn.Module):

    def __init__(self,  n_actions, perf_shape, score_shape):
        super(ScoreFollowingNetNottinghamLS, self).__init__()

        self.spec_conv1 = nn.Conv2d(perf_shape[0], out_channels=16, kernel_size=3, stride=2)
        self.spec_conv2 = nn.Conv2d(self.spec_conv1.out_channels, out_channels=32, kernel_size=3, stride=2)
        self.spec_conv3 = nn.Conv2d(self.spec_conv2.out_channels, out_channels=32, kernel_size=3, stride=2)
        self.spec_conv4 = nn.Conv2d(self.spec_conv3.out_channels, out_channels=64, kernel_size=3, stride=1)

        self.sheet_conv1 = nn.Conv2d(score_shape[0], out_channels=16, kernel_size=5, stride=(1, 2))
        self.sheet_conv2 = nn.Conv2d(self.sheet_conv1.out_channels, out_channels=32, kernel_size=3, stride=2)
        self.sheet_conv3 = nn.Conv2d(self.sheet_conv2.out_channels, out_channels=32, kernel_size=3, stride=2)
        self.sheet_conv4 = nn.Conv2d(self.sheet_conv3.out_channels, out_channels=64, kernel_size=3, stride=(1, 2))

        self.concat_fc = nn.Linear(self.spec_conv4.out_channels * 6 * 2 + self.sheet_conv4.out_channels * 6 * 8, 256)

        self.policy_fc = nn.Linear(self.concat_fc.out_features, 256)
        self.policy_out = nn.Linear(self.policy_fc.out_features, n_actions)

        self.critic_fc = nn.Linear(self.concat_fc.out_features, 256)
        self.critic_out = nn.Linear(self.critic_fc.out_features, 1)

        self.apply(weights_init)

    def forward(self, perf, score):

        spec_x = F.elu(self.spec_conv1(perf))
        spec_x = F.elu(self.spec_conv2(spec_x))
        spec_x = F.elu(self.spec_conv3(spec_x))
        spec_x = F.elu(self.spec_conv4(spec_x))

        spec_x = spec_x.view(-1, num_flat_features(spec_x)) # flatten

        sheet_x = F.elu(self.sheet_conv1(score))
        sheet_x = F.elu(self.sheet_conv2(sheet_x))
        sheet_x = F.elu(self.sheet_conv3(sheet_x))
        sheet_x = F.elu(self.sheet_conv4(sheet_x))

        sheet_x = sheet_x.view(-1, num_flat_features(sheet_x)) # flatten

        cat_x = torch.cat((spec_x, sheet_x), dim=1)

        cat_x = F.elu(self.concat_fc(cat_x))

        policy = F.elu(self.policy_fc(cat_x))
        policy = self.policy_out(policy)

        critic = F.elu(self.critic_fc(cat_x))
        critic = self.critic_out(critic)

        # if we use softmax and the logarithmic softmax PyTorch recommends to use the log_softmax function
        # therefore we return the linear output of the policy and apply the softmax/log softmax where it is necessary
        return {'policy': {'logits': policy}, 'value': critic}
