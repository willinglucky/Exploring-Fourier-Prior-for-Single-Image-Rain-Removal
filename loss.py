import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np

def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')

class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * l1_loss(
            pred, target, weight, reduction=self.reduction)

# gt : ground truth
# preds_first_stage : the output of first stage
# preds_second_stage : the output of second stage

cri_l1 = L1Loss()
gt_fft = torch.fft.fft2(gt, dim=(-2, -1))
gt_amp = torch.abs(gt_fft)

l1 = cri_l1(preds_first_state, gt_first_stage)
l2 = cri_l1(preds_second_state, gt)
label_fft = torch.stack((gt_fft.real, gt_fft.imag), -1)
# pred_fft1 = torch.fft.fft2(preds[0],  dim=(-2, -1))
pred_fft = torch.fft.fft2(preds_second_stage, dim=(-2, -1))
pred_fft = torch.stack((pred_fft.real, pred_fft.imag), -1)
# f1 = self.cri_l1(pred_fft1, gt_fft)
l_fft = cri_l1(pred_fft, label_fft)
l_amp = cri_l1(amp_first_state, gt_amp)