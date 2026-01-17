import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt as edt

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: [N, C, ...]  (logits)
        targets: [N, ...]   (ground truth, long类型)
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)  # p = softmax概率
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss

def compute_sdf(img_gt, out_shape):
    """
    计算 Signed Distance Map
    img_gt: [H, W] ground truth mask (0,1)
    out_shape: (H, W)
    """
    posmask = img_gt.astype(np.bool_)
    negmask = ~posmask

    pos_edt = edt(posmask)
    neg_edt = edt(negmask)
    boundary_map = neg_edt - pos_edt
    return boundary_map

class BoundaryLoss(nn.Module):
    def __init__(self):
        super(BoundaryLoss, self).__init__()

    def forward(self, inputs, targets):
        """
        inputs: [N, C, H, W]  (logits)
        targets: [N, H, W]    (ground truth)
        """
        n, c, h, w = inputs.shape
        probs = F.softmax(inputs, dim=1)

        # 转 one-hot
        targets_one_hot = F.one_hot(targets, num_classes=c).permute(0, 3, 1, 2).float()

        # SDF map
        sdf = torch.zeros_like(probs)
        for b in range(n):
            for cls in range(1, c):  # 忽略背景
                gt_cls = targets_one_hot[b, cls].cpu().numpy().astype(np.uint8)
                if gt_cls.max() == 0:  # 没有该类别
                    continue
                sdf[b, cls] = torch.from_numpy(compute_sdf(gt_cls, (h, w))).to(inputs.device)

        # Boundary loss = <p, sdf>
        boundary_loss = torch.mean(probs * sdf)
        return boundary_loss

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, smooth=1e-6):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, inputs, targets):
        """
        inputs: [N, C, H, W]  (logits, 需要softmax)
        targets: [N, H, W]    (ground truth, long类型)
        """
        num_classes = inputs.size(1)
        # One-hot
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()

        probs = F.softmax(inputs, dim=1)
        dims = (0, 2, 3)

        tp = torch.sum(probs * targets_one_hot, dims)
        fp = torch.sum(probs * (1 - targets_one_hot), dims)
        fn = torch.sum((1 - probs) * targets_one_hot, dims)

        tversky = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        return 1 - tversky.mean()
