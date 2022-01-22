

# -- torch imports --
import torch
from torch import nn as nn
from torch import optim as optim
from einops import rearrange,repeat

# -- external models --
from .pacnet import SepConvNet2D

def get_nn_model(cfg,nn_arch):
    if nn_arch == "sepnn":
        return get_sepnn_model(cfg)
    else:
        raise ValueError(f"Uknown nn architecture [{nn_arch}]")

def get_sepnn_model(cfg):

    # -- init model --
    model = SepConvNet2D(4)
    model = model.to(cfg.gpuid,non_blocking=True)
    loss_fxn_base = nn.MSELoss()
    loss_fxn_base = loss_fxn_base.to(cfg.gpuid,non_blocking=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    argdict = {}

    # -- create closure for loss --
    def wrap_loss_fxn(denoised,gt_img_01,denoised_frames,step):
        # print(denoised.min().item(),denoised.max().item())
        # print(gt_img.min().item(),gt_img.max().item())
        # print("denoised.shape: " ,denoised.shape)
        # print("gt_img.shape: " ,gt_img.shape)
        gt_img = gt_img_01 - 0.5#gt_img.mean()
        loss = loss_fxn_base(denoised,gt_img)
        return loss
    loss_fxn = wrap_loss_fxn

    # -- create empty scheduler --
    def scheduler_fxn(epoch,argdict):
        pass

    # -- wrap call function for interface --
    forward_fxn = model.forward
    def wrap_forward(dyn_noisy,noise_level):


        # -- add weights to ftrs --
        in_weights = (dyn_noisy - dyn_noisy[:, 0:1, ...]) ** 2
        b, n, f, v, h = in_weights.shape
        in_weights = in_weights.view(b, n, f // 3, 3, v, h).mean(2)
        inputs = torch.cat((dyn_noisy, in_weights), 2)

        # -- forward pass --
        res = forward_fxn(inputs)
        deno = dyn_noisy[:,0] - res

        return deno

    model.forward = wrap_forward

    return model,loss_fxn,optimizer,scheduler_fxn,argdict


