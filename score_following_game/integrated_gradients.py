"""
adapted for own purposes from
https://github.com/utkuozbulak/pytorch-cnn-visualizations/blob/master/src/integrated_gradients.py
"""

import cv2
import torch

import numpy as np
import torch.nn.functional as F


def prepare_grad_for_render(gradient, resize_shape, norm, cmap):

    saliency = gradient / (np.abs(gradient).max() + 1e-8)
    saliency = cv2.resize(saliency, resize_shape)
    saliency = np.uint8(cmap(norm(saliency)) * 255)
    saliency = cv2.cvtColor(saliency, cv2.COLOR_RGBA2BGR)

    return saliency


class IntegratedGradients:

    def __init__(self, model, device, steps=100):
        self.model = model
        self.gradients = None
        self.score_grads = None
        self.perf_grads = None
        self.device = device
        self.steps = steps

        # Put model in evaluation mode
        self.model.eval()
        self.hook_layers()

    def store_score_grads(self, module, grad_in, grad_out):
        self.score_grads = grad_in[0]

    def store_perf_grads(self, module, grad_in, grad_out):
        self.perf_grads = grad_in[0]

    def hook_layers(self):

        # Register hook to the first score/perf layer
        first_score_layer = self.model._modules.get('sheet_conv1')
        first_score_layer.register_backward_hook(self.store_score_grads)

        first_perf_layer = self.model._modules.get('spec_conv1')
        first_perf_layer.register_backward_hook(self.store_perf_grads)

    def get_scaled(self, inp, baseline):
        return [baseline + ((float(i)/self.steps)*(inp-baseline)) for i in range(0, self.steps+1)]


    def generate_gradients(self, input_image, target_class=None):
        # Forward pass

        perf, perf_diff = input_image[0][0]
        score, score_diff = input_image[1][0]

        perf_baseline = 0*perf.detach()
        perf_diff_baseline = 0*perf_diff.detach()
        score_baseline = 0*score.detach()
        score_diff_baseline = 0*score_diff.detach()

        p = torch.cat((perf.unsqueeze(0), perf_diff.unsqueeze(0)), dim=0).unsqueeze(0)
        s = torch.cat((score.unsqueeze(0), score_diff.unsqueeze(0)), dim=0).unsqueeze(0)

        m_out = F.softmax(self.model(perf=p, score=s)['policy']['logits'], dim=-1)
        target_class = np.ones((m_out.size()[0], 1)) * torch.argmax(m_out).item()

        scaled_perf = self.get_scaled(perf, perf_baseline)
        scaled_perf_diff = self.get_scaled(perf_diff, perf_diff_baseline)

        scaled_score = self.get_scaled(score, score_baseline)
        scaled_score_diff = self.get_scaled(score_diff, score_diff_baseline)

        p_batch = torch.cat((scaled_perf[0].unsqueeze(0), scaled_perf_diff[0].unsqueeze(0)), dim=0).unsqueeze(0)
        s_batch = torch.cat((scaled_score[0].unsqueeze(0), scaled_score_diff[0].unsqueeze(0)), dim=0).unsqueeze(0)

        for i in range(1, self.steps+1):

            p = torch.cat((scaled_perf[i].unsqueeze(0), scaled_perf_diff[i].unsqueeze(0)), dim=0).unsqueeze(0)
            s = torch.cat((scaled_score[i].unsqueeze(0), scaled_score_diff[i].unsqueeze(0)), dim=0).unsqueeze(0)

            p_batch = torch.cat((p_batch, p))
            s_batch = torch.cat((s_batch, s))

        p_batch.requires_grad_()
        s_batch.requires_grad_()
        model_output = F.softmax(self.model(perf=p_batch, score=s_batch)['policy']['logits'], dim=-1)

        # Zero gradients
        self.model.zero_grad()

        out = torch.zeros(model_output.shape).to(self.device)
        out[range(model_output.shape[0]), target_class] = 1

        # Backward pass
        model_output.backward(gradient=out.data)

        score_grads = self.score_grads.cpu().data.numpy()
        perf_grads = self.perf_grads.cpu().data.numpy()


        score_g = (score.cpu().data.numpy() - score_baseline.cpu().data.numpy())*np.average(score_grads[:-1, 0], axis=0)
        score_diff_g = (score_diff.cpu().data.numpy() - score_diff_baseline.cpu().data.numpy())*np.average(score_grads[:-1, 1], axis=0)

        perf_g = (perf.cpu().data.numpy() - perf_baseline.cpu().data.numpy())*np.average(perf_grads[:-1, 0], axis=0)
        perf_diff_g = (perf_diff.cpu().data.numpy() - perf_diff_baseline.cpu().data.numpy())*np.average(perf_grads[:-1, 1], axis=0)

        return [score_g, score_diff_g], [perf_g, perf_diff_g]

