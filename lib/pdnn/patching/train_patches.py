
# -- linalg --
import numpy as np
import torch as th

# -- vision --
import torchvision.transforms as tvt

# -- package import --
from pdnn.sobel import apply_sobel_filter

# -- sim search import --
from vnlb.gpu.sim_search import fill_patches

# -- local import --
from .l2_search import exec_l2_search

def get_train_io(noisy,clean,sigma,ps,npatches,nneigh):

    # -- hyper-params --
    pt = 2
    c = 3

    # -- select candidate inds --
    srch_inds = select_patch_inds(clean,npatches,ps)

    # -- return neighbors of "search indices" via l2 search --
    inds = exec_l2_search(noisy,srch_inds,nneigh,sigma,ps,pt)

    # -- create patches from inds --
    pnoisy = construct_patches(noisy,inds,ps,pt)
    pclean = construct_patches(clean,inds,ps,pt)

    return pnoisy,pclean


def select_patch_inds(img,npatches,ps):


    # -- shapes and init --
    device = img.device
    B,T,C,H,W = img.shape
    inds = th.zeros((B,npatches,3),dtype=th.long,device=device)
    rcrop = tvt.RandomCrop((128,128))

    # -- for each elem ... --
    for b in range(B):

        # -- compute edges --
        edges = apply_sobel_filter(img[b])
        edges = rcrop(edges)

        # -- sample indices prop. to edge weight --
        prec90 = th.quantile(edges.ravel(),0.99)
        mask = edges > prec90
        index = th.nonzero(mask)
        order = th.randperm(index.shape[0])
        inds[b,...] = index[order[:npatches]]

    return inds


def construct_patches(img,inds,ps,pt):

    # -- init patches --
    device = inds.device
    B,N,k = inds.shape
    B,T,c,H,W = img.shape
    patches = th.zeros((B,N,k,pt,c,ps,ps),dtype=th.float,device=device)

    # -- parameter --
    cs_ptr = th.cuda.default_stream().cuda_stream

    # -- fill each batch --
    for b in range(B):
        fill_patches(patches[b],img[b],inds[b],cs_ptr)

    return patches
