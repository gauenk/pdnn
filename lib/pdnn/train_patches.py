
# -- linalg --
import numpy as np
import torch as th
from einops import rearrange

# -- vision --
import torchvision.transforms as tvt

# -- package import --
from pdnn.sobel import apply_sobel_filter

# -- sim search import --
import sim_search
from sim_search import fill_patches,exec_sim_search,get_patches

# -- wrap this to protect our "sim_search" dep --
def imgs2patches(noisy,clean,sigma,ps,npatches,nsearch):
    pnoisy,pclean = sim_search.imgs2patches(noisy,clean,sigma,ps,
                                            npatches,nsearch)
    return pnoisy,pclean

def io_patches(pnoisy,pclean,sigma,ps,nneigh):

    # -- init shells --
    device = pnoisy.device
    B,np,ns,t,c,h,w = pnoisy.shape
    inputs = th.zeros((B,np,nneigh,t,c,h,w),device=device)
    targets = th.zeros((B,np,t,c,h,w),device=device)
    npatches = np
    nsearch = ns

    # -- we want to stratify _where_ we get our "nneigh" values across the top K --
    for p in range(npatches):
        start = th.randperm(nsearch-nneigh)[0]
        index = slice(start,start+nneigh)
        inputs[:,p,:] = pnoisy[:,p,index]
        targets[:,p] = pclean[:,p,start]

    # -- final shaping for net --
    inputs = rearrange(inputs,'b np nn t c h w -> (b np) nn (t c) h w')
    targets = rearrange(targets,'b np t c h w -> (b np) (t c) h w')

    # -- final shaping for net --
    # inputs = rearrange(pnoisy,'b np nn t c h w -> (b np) nn (t c) h w')
    # targets = rearrange(pclean[:,:,0],'b np t c h w -> (b np) (t c) h w')

    return inputs,targets


