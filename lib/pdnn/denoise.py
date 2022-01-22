"""

Apply a trained model to denoise a subset of patches

"""

# -- python --
import math
from pathlib import Path
from easydict import EasyDict as edict

# -- linalg --
import torch as th
from einops import rearrange,repeat

# -- vision --
import torchvision.transforms.functional as tvf

# -- package --
from pdnn import load_sigma_model

def expand_patches(patches,pt=2,c=3):
    if patches.ndim == 3:
        D = patches.shape[-1]
        ps = int(math.sqrt(D//(pt*c)))
        patches = rearrange(patches,'b n (t c h w) -> b n t c h w',t=pt,c=c,h=ps)
    return patches

def pad_patches(patches,ps_tgt):
    if ps_tgt is None: return patches
    h,w = patches.shape[-2:]
    th,tw = ps_tgt,ps_tgt
    assert (th >= h) and (tw >= w)
    pad_h = int((th - h)//2)
    pad_w = int((tw - w)//2)
    patches = tvf.pad(patches,[pad_w,pad_h,pad_w,pad_h])
    return patches

def ocrop(pdeno_n,ps,in_ps):
    if ps == in_ps:
        return pdeno_n
    else:
        psh = ps//2
        return pdeno_m[...,psh:-psh,psh:-psh]

def denoise_patches(patches,sigma,ps,nneigh=15,ps_model=None):

    # -- hyper params --
    pt,c = 2,3

    # -- load model using sigma --
    model = load_sigma_model(sigma,patches.device)
    model.eval()

    # -- format patches --
    patches = expand_patches(patches,pt,c)
    device = patches.device
    B,N,T,C,in_ps,in_ps = patches.shape

    # -- pad patches --
    patches = pad_patches(patches,ps)

    # -- cat "t" across color --
    patches = rearrange(patches,'b n t c h w -> b n (t c) h w')

    # -- create denoised shell --
    pdeno = th.zeros((B,N,T,C,ps,ps),dtype=th.float,device=device)

    # -- apply a denoising method --
    # method = "v1"
    method = "v2"
    apply_deno_method(method,model,patches,sigma,pdeno,nneigh,in_ps,ps)

    # -- free up memory --
    del model
    th.cuda.empty_cache()

    return pdeno

def apply_deno_method(method,model,patches,sigma,pdeno,nneigh,in_ps,ps):
    if method == "v1":
        deno_method_1(model,patches,sigma,pdeno,nneigh,in_ps,ps)
    elif method == "v2":
        deno_method_2(model,patches,sigma,pdeno,nneigh,in_ps,ps)
    else:
        raise ValueError(f"Uknown denoising method [{method}]")


# --------------------------------------
#
#    Mechanisms to apply our denoiser
#
# --------------------------------------


def deno_method_1(model,patches,sigma,pdeno,nneigh):

    # -- unpack shape --
    B,N,T,C,ps,ps = pdeno.shape

    # -- denoise patches --
    nstart = 0
    for n in range(N):
        index = th.remainder(th.arange(nstart,nstart+nneigh),N)
        patches_n = patches[:,index]
        with th.no_grad():
            pdeno_n = model(patches_n-0.5,sigma)+0.5
        pdeno_n = rearrange(pdeno_n,'b (t c) h w -> b t c h w',t=2)
        pdeno[:,nstart] = ocrop(pdeno_n,in_ps,ps)

def deno_method_2(model,patches,sigma,pdeno,nneigh,in_ps,ps):

    # -- unpack shape --
    B,N,t,c,ps,ps = pdeno.shape

    # -- denoise patches --
    nstart = 0
    for n in range(N):

        # -- find top "nneigh" for index "n" --
        patch_n = patches[:,[n]]
        delta = th.mean((patch_n - patches)**2,(-3,-2,-1))
        order = th.argsort(delta,1)[:,:nneigh]

        # -- gather inputs --
        aug_order = repeat(order,'b n -> b n (t c) h w',t=t,c=c,h=ps,w=ps)
        inputs = th.gather(patches,1,aug_order)

        # -- denoise --
        with th.no_grad():
            pdeno_n = model(inputs-0.5,sigma)+0.5
        pdeno_n = rearrange(pdeno_n,'b (t c) h w -> b t c h w',t=2)
        pdeno[:,nstart] = ocrop(pdeno_n,in_ps,ps)


