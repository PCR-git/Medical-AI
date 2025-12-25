
import torch
import torch.nn as nn
import torch.nn.functional as F

# Dice similarity coefficient
def dsc(y_true, y_pred):
    """ Dice Similarity Coefficient (DSC) component. """
    smooth = 1e-6 
    y_true_f = torch.flatten(y_true)
    y_pred_f = torch.flatten(y_pred)
    
    mask = y_true_f * y_pred_f
    intersection = torch.sum(mask)
    score = (2. * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)
    return score

# Dice loss
def dice_loss(y_true, y_pred):
    """ Dice Loss combined with Binary Cross-Entropy (BCE). """
    dice_l = (1 - dsc(y_true, y_pred))
    # Using the functional API for BCE
    bce_l = F.binary_cross_entropy(y_pred, y_true, reduction='mean') 
    return dice_l + bce_l

# Gradient difference loss
class GDLoss(nn.Module):
    """Calculates the Gradient Difference Loss (GDL)."""
    def __init__(self, alpha=1, beta=1):
        super(GDLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, y_hat, y):
        # 1. Vertical Gradient Difference (Height: H)
        grad_y_hat_v = torch.abs(y_hat[:, :, 1:, :] - y_hat[:, :, :-1, :])
        grad_y_v = torch.abs(y[:, :, 1:, :] - y[:, :, :-1, :])
        diff_v = torch.abs(grad_y_hat_v - grad_y_v) ** self.alpha

        # 2. Horizontal Gradient Difference (Width: W)
        grad_y_hat_h = torch.abs(y_hat[:, :, :, 1:] - y_hat[:, :, :, :-1])
        grad_y_h = torch.abs(y[:, :, :, 1:] - y[:, :, :, :-1])
        diff_h = torch.abs(grad_y_hat_h - grad_y_h) ** self.alpha

        # Sum the differences and apply outer power (beta)
        gdl_loss = (diff_v.mean() + diff_h.mean()) ** self.beta
        
        return gdl_loss

