
# -- python imports --
import sys
import numpy as np
from einops import repeat,rearrange

# -- pytorch imports --
import torch

# -- project imports --
from easydict import EasyDict as edict
from datasets.wrap_image_data import dict_to_device

# -- project imports --
from pdnn.utils.images import print_tensor_stats,save_image
from pdnn.train_patches import io_patches,imgs2patches

# -- local imports --
from .log import get_train_log_info,print_train_log_info,get_test_log_info

def train_loop(cfg,model,loss_fxn,optim,data_loader):

    tr_info = []
    # nbatches = len(data_loader)
    # nbatches = 500
    data_iter = iter(data_loader)
    # nbatches = min(1000,len(data_iter))
    nbatches = 100
    # for batch_iter,sample in enumerate(data_loader):
    for batch_iter in range(nbatches):

        # -- sample from iterator --
        sample = next(data_iter)

        # -- unpack sample --
        device = f'cuda:{cfg.gpuid}'
        dict_to_device(sample,device)
        dyn_noisy = sample['dyn_noisy'] # dynamics and noise
        noisy = dyn_noisy # alias
        dyn_clean = sample['dyn_clean'] # dynamics and no noise
        clean = dyn_clean # alias
        static_noisy = sample['static_noisy'] # no dynamics and noise
        static_clean = sample['static_clean'] # no dynamics and no noise
        # flow_gt = sample['ref_flow']
        image_index = sample['index']

        # -- shape info --
        T,B,C,H,W = dyn_noisy.shape
        nframes = T
        isize = edict({'h':H,'w':W})
        ref_t = nframes//2

        # -- create sim images --
        # flow_gt = flow_shaping(flow_gt,H*W)
        gt_info = {'clean':dyn_clean,#'flow':flow_gt,
                   'static_noisy':static_noisy,'isize':isize}

        # -- get patches --
        pnoisy,pclean = imgs2patches(noisy,clean,cfg.noise_level,cfg.ps,
                                     cfg.npatches,cfg.nsearch)

        # -- create inputs and outputs --
        inputs,targets = io_patches(pnoisy,pclean,cfg.noise_level,cfg.ps,cfg.nneigh)

        # -- reset gradient --
        model.zero_grad()
        optim.zero_grad()

        # -- forward pass --
        output = model(inputs,cfg.noise_level) #
        if isinstance(output,tuple): pdeno,denoised_frames = output
        else: pdeno,denoised_frames = output,None

        # -- compute loss --
        loss = loss_fxn(pdeno,targets,denoised_frames,cfg.global_step)

        # print("-"*20)
        # print_tensor_stats("denoised",denoised)
        # print_tensor_stats("sim_0",sims[[0]])
        # print_tensor_stats("sim_1",sims[[1]])
        # print_tensor_stats("dyn_noisy",dyn_noisy)
        # print_tensor_stats("aligned",aligned)
        # print_tensor_stats("dyn_clean",dyn_clean)

        # -- backward --
        loss.backward()
        optim.step()

        # -- log --
        if batch_iter % cfg.train_log_interval == 0:
            info = get_train_log_info(cfg,model,loss,pdeno,pnoisy,pclean)
            info['global_iter'] = cfg.global_step
            info['batch_iter'] = batch_iter
            info['mode'] = 'train'
            info['loss'] = loss.item()
            info['epoch'] = cfg.epoch
            print_train_log_info(info,nbatches)

            # -- save example to inspect --
            pdeno = pdeno.detach()
            with torch.no_grad():
                inputs = torch.clip(inputs,0,1)
                targets = torch.clip(targets,0,1)
                pdeno = torch.clip(pdeno,0,1)
                # save_image(f"inputs_{batch_iter}.png",pnoisy)
                # save_image(f"target_{batch_iter}.png",pclean)
                # save_image(f"pdeno_{batch_iter}.png",pdeno)
                # save_patches(f"inputs_{batch_iter}.png",pnoisy)
                # save_patches(f"target_{batch_iter}.png",pclean)
                # save_patches(f"pdeno_{batch_iter}.png",pdeno)


            tr_info.append(info)

        # -- update global step --
        cfg.global_step += 1

        # -- print update --
        sys.stdout.flush()

    return tr_info

def test_loop(cfg,model,test_loader,loss_fxn):

    model = model.to(cfg.device)
    test_iter = iter(test_loader)
    # nbatches = min(500,len(test_iter))
    nbatches = 20
    # nbatches = 2
    te_info = []

    with torch.no_grad():
        for batch_iter in range(nbatches):

            # -- load data --
            device = f'cuda:{cfg.gpuid}'
            sample = next(test_iter)
            dict_to_device(sample,device)

            # -- unpack --
            dyn_noisy = sample['dyn_noisy']
            dyn_clean = sample['dyn_clean']
            noisy,clean = dyn_noisy,dyn_clean
            static_noisy = sample['static_noisy']
            flow_gt = sample['ref_flow']
            nframes = dyn_clean.shape[0]
            T,B,C,H,W = dyn_noisy.shape
            isize = edict({'h':H,'w':W})

            #
            # -- modify inputs as needed --
            #

            # -- get patches --
            pnoisy,pclean = imgs2patches(noisy,clean,cfg.noise_level,cfg.ps,
                                         cfg.npatches,cfg.nsearch)

            # -- create inputs and outputs --
            inputs,targets = io_patches(pnoisy,pclean,cfg.noise_level,cfg.ps,cfg.nneigh)

            #
            # -- denoise image --
            #

            output = model(inputs,cfg.noise_level)
            if isinstance(output,tuple): pdeno,denoised_frames = output
            else: pdeno,denoised_frames = output,None

            # -- compute gt loss --
            loss = loss_fxn(pdeno,targets,denoised_frames,cfg.global_step)

            # -- log info --
            info = get_test_log_info(cfg,model,loss,pdeno,pnoisy,pclean)
            info['global_iter'] = cfg.global_step
            info['batch_iter'] = batch_iter
            info['mode'] = 'test'
            info['loss'] = loss.item()
            info['epoch'] = cfg.epoch
            te_info.append(info)

            # -- print to screen --
            if batch_iter % cfg.test_log_interval == 0:
                psnr = info['image_psnrs'].mean().item()
                print("[%d/%d] Test PSNR: %2.2f" % (batch_iter+1,nbatches,psnr))

            # -- print update --
            sys.stdout.flush()

    # -- print final update --
    print("[%d/%d] Test PSNR: %2.2f" % (batch_iter+1,nbatches,psnr))
    sys.stdout.flush()


    return te_info

