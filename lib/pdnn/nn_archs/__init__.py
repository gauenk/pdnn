

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
    model = SepConvNet2D()
    model = model.to(cfg.gpuid,non_blocking=True)
    loss_fxn_base = nn.MSELoss()
    loss_fxn_base = loss_fxn_base.to(cfg.gpuid,non_blocking=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    argdict = {}

    # -- create closure for loss --
    def wrap_loss_fxn(denoised,gt_img,denoised_frames,step):
        gt_img_nmlz = gt_img - 0.5#gt_img.mean()
        loss_basic,loss_anneal = loss_fxn_base(denoised_frames,denoised,gt_img_nmlz,step)
        return loss_basic + loss_anneal
    loss_fxn = wrap_loss_fxn

    # -- create empty scheduler --
    def scheduler_fxn(epoch,argdict):
        pass

    # -- wrap call function for interface --
    forward_fxn = model.forward
    def wrap_forward(dyn_noisy,noise_level):
        deno = forward_fxn(dyn_noisy)
        return deno
    model.forward = wrap_forward

    return model,loss_fxn,optimizer,scheduler_fxn,argdict


