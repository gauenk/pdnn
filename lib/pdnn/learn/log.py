
import numpy as np
from easydict import EasyDict as edict
from einops import rearrange,repeat

from pyutils import images_to_psnrs

import torch
import torchvision

center_crop = torchvision.transforms.functional.center_crop

def print_train_log_info(info,nbatches):
    msg = f"Train @ [{info['batch_iter']}/{nbatches}]"

    image_psnrs = info['image_psnrs']
    mean = image_psnrs.mean()
    msg += f" [PSNR(i)]: %2.2f" % (mean)

    print(msg)


def get_train_log_info(cfg,model,loss,pdeno,pnoisy,pclean):

    # -- init info --
    info = {}

    # -- create ref --
    nimages,npatches,nneigh,pt,c,ph,pw = pclean.shape
    ref = rearrange(pclean[:,:,0,0],'b p c h w -> (b p) c h w')

    # -- image psnrs --
    image_psnrs = images_to_psnrs(pdeno,ref-0.5)
    info['image_psnrs'] = image_psnrs

    # -- patch subset search quality --
    # patch_subset = psubset_quality(model,pnoisy,pclean)
    # info['patch_subset'] = patch_subset

    return info


def get_test_log_info(cfg,model,loss,pdeno,pnoisy,pclean):

    # -- init info --
    info = {}

    # -- create ref --
    nimages,npatches,nneigh,pt,c,ph,pw = pclean.shape
    ref = rearrange(pclean[:,:,0,0],'b p c h w -> (b p) c h w')

    # -- image psnrs --
    image_psnrs = images_to_psnrs(pdeno,ref-0.5)
    info['image_psnrs'] = image_psnrs

    # -- patch subset search quality --
    # patch_subset = psubset_quality(model,pnoisy,pclean)
    # info['patch_subset'] = patch_subset

    return info
