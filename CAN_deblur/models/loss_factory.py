import torch.nn as nn
import torch.nn.functional as F


class BCELoss2D(nn.Module):
    def __init__(self, cfg=None):
        super(BCELoss2D, self).__init__()
    
    def forward(self, outputs_fake, outputs_real, targets_fake, targets_real):
        fake_loss = F.binary_cross_entropy(outputs_fake, targets_fake)
        real_loss = F.binary_cross_entropy(outputs_real, targets_real)
        final_loss = (fake_loss + real_loss)
        return final_loss


class BCE_L1Loss2G(nn.Module):
    def __init__(self, l1scaling, cfg=None):
        super(BCE_L1Loss2G, self).__init__()
        self.l1scaling = l1scaling
    
    def forward(self, outputs, targets, fake_outputs, real_target):
        final_loss = self.l1scaling * F.l1_loss(outputs, targets)\
            + F.binary_cross_entropy(fake_outputs, real_target)
        return final_loss


class BCE_L2Loss2G(nn.Module):
    def __init__(self, l1scaling, cfg=None):
        super(BCE_L2Loss2G, self).__init__()
        self.l1scaling = l1scaling
    
    def forward(self, outputs, targets, fake_outputs, real_target):
        final_loss =  F.mse_loss(outputs, targets)\
            +self.l1scaling * F.binary_cross_entropy(fake_outputs, real_target)
        return final_loss


class GL2Only(nn.Module):
    def __init__(self, l1scaling, cfg=None):
        super(GL2Only, self).__init__()
        self.l1scaling = l1scaling
    
    def forward(self, outputs, targets):
        final_loss =  F.mse_loss(outputs, targets)
        return final_loss


class MultiBCE_L2Loss2G(nn.Module):
    def __init__(self, l2scaling, cfg=None):
        super(MultiBCE_L2Loss2G, self).__init__()
        self.l2scaling = l2scaling
    
    def forward(self, outputs, targets, fake_outputs, real_target):
        final_loss = (F.mse_loss(outputs[0], targets[0])\
            + F.mse_loss(outputs[1], targets[1])+F.mse_loss(outputs[2], targets[2]))\
            + self.l2scaling * F.binary_cross_entropy(fake_outputs, real_target)
        return final_loss


LOSS_FUNC_LUT = {
        'bce_d': BCELoss2D,
        'bce_l1_g': BCE_L1Loss2G,
        'MultiBCE_L2_g': MultiBCE_L2Loss2G,
        'bce_l2_g': BCE_L2Loss2G,
        'GL2Only': GL2Only,
    }


def get_loss_func(loss_name, **kwargs):    
    try:
        loss_func_class = LOSS_FUNC_LUT.get(loss_name)   
    except:
        kwargs['logger'].error("loss tpye error, {} not exist".format(loss_name))
    if 'scaling' in kwargs:
        loss_func = loss_func_class(kwargs['scaling'])
    else:
        loss_func = loss_func_class()
    return loss_func