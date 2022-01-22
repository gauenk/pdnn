

# -- linalg --
import numpy as np
import torch as th

# -- vision --
import torchvision.transforms as tvt

# -- package import --
from pdnn.sobel import apply_sobel_filter

# -- sim search import --
import sim_search
from sim_search import fill_patches,exec_sim_search,get_patches

def exec_patch_search(img,sigma,srch_inds,nsearch,ps,ps_model=13,nneigh=15,**kwargs):
    """
    Search for topk patches from the given image
    """

    # -- l2 search --
    inds = exec_sim_search(img,srch_inds,nsearch,sigma,ps,**kwargs) # use "ps = 7"

    # -- get patches at inds --
    patches = get_patches(noisy,inds,ps_model,**kwargs) # grab patches, "ps = 13"

    # -- denoise patches --
    deno = denoise_patches(patches,sigma,ps_model,nneigh)

    # -- standard search --
    delta = th.mean((deno[...,[0],:] - deno)**2,(-2,-1))
    inds = th.argsort(delta,1)

    # -- gather inds from original search inds --
    sorted_inds = th.gather(srch_inds,inds,1)

    return sorted_inds

