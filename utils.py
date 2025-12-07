import torch
import torch.nn.functional as F

def dice_loss(pred, target, smooth=1):
    pred = F.softmax(pred, dim=1)
    pred_fg = pred[:, 1, :, :]
    target_fg = (target == 1).float()

    intersection = (pred_fg * target_fg).sum()
    union = pred_fg.sum() + target_fg.sum()

    return 1 - ((2. * intersection + smooth) / (union + smooth))
