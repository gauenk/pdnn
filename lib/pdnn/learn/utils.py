# -- python imports --
from easydict import EasyDict as edict

# -- numpy imports --
import numpy as np

# -- pytorch imports --
import torch
import torchvision.transforms.functional as tvF

# # -- project imports --
# from align import compute_epe,compute_aligned_psnr

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

def optional(pydict,key,default):
    if pydict is None: return default
    if not(key in pydict): return default
    else: return pydict[key]

def print_dict_ndarray_0_midpix(dict_ndarray,mid_pix):
    print("-"*50)
    for name,ndarray in dict_ndarray.items():
        print(name,ndarray[0,mid_pix])

def remove_center_frame(frames):
    nframes = frames.shape[0]
    nc_frames =torch.cat([frames[:nframes//2],frames[nframes//2+1:]],dim=0)
    return nc_frames

# def remove_center_frame(burst):
#     nframes = burst.shape[0]
#     ref = nframes//2
#     left,right = burst[:ref],burst[ref+1:]
#     if torch.is_tensor(burst):
#         burst = torch.cat([left,right],dim=0)
#     else:
#         burst = np.concatenate([left,right],axis=0)
#     return burst

def compute_pair_flow_acc(guess,gt):
    nimages,npix,nframes_m1,two = guess.shape
    guess = guess.cpu()
    gt = gt.cpu()
    flow_acc = torch.zeros(nframes_m1,nimages)
    for t in range(nframes_m1):
        guess_t = guess[:,:,t,:].type(torch.long)
        gt_t = gt[:,:,t,:].type(torch.long)
        both = torch.all(guess_t == gt_t,dim=-1)
        ncorrect = torch.sum(both,dim=1).type(torch.float)
        acc = 100 * ncorrect / npix
        flow_acc[t,:] = acc
    return flow_acc

def center_crop_frames(frames,csize=30):
    csize = 30
    cc_frames = edict()
    for name,burst in frames.items():
        cc_frames[name] = tvF.center_crop(aligned_of,(csize,csize))
    return cc_frames
