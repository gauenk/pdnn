
# -- python imports --
import time,sys,os,copy
import numpy as np
import pandas as pd
from pathlib import Path
from einops import rearrange,repeat
from easydict import EasyDict as edict

# -- multiprocessing --
from multiprocessing import current_process

# -- pytorch imports --
import torch

# -- caching --
import cache_io

# -- dataset --
from datasets import load_dataset

# -- pack imports --
from pdnn.nn_archs import get_nn_model

# -- [local] package imports --
from .trte import train_loop,test_loop
from .results_format import append_result_to_dict
from .common import save_model_checkpoint,resume_training,load_checkpoint
from .msgs import *
from .utils import *

def exec_learn(cfg):

    # -- init exp! --
    cfg = init_config(cfg)
    print("Running Exp: [PDNNN] Training/Testing Model [PID: %s]" % cfg.pid)
    print(cfg)

    # -- set default device --
    torch.cuda.set_device(cfg.gpuid)

    # -- create results record to save --
    dims={'batch_results':None,'batch_to_record':None,
          'record_results':{'default':0},
          'stack':{'default':0},'cat':{'default':0}}
    record = cache_io.ExpRecord(dims)

    # -- set random seed --
    set_seed(optional(cfg,'random_seed',123))

    # -- get neural netowrk --
    model,loss_fxn,optim,sched_fxn,argdict = get_nn_model(cfg,cfg.nn_arch)

    # -- load dataset --
    data,loaders = load_dataset(cfg)

    # -- check if exists --
    start_epoch,results = resume_training(cfg, model, optim, argdict)
    print(f"Starting from epoch [{start_epoch}]")

    result_te = test_loop(cfg,model,loaders.te,loss_fxn)

    # -- iterate over images --
    start_time = time.perf_counter()
    for epoch in range(start_epoch,cfg.nepochs):
        print("-"*25)
        print(f"Epoch [{epoch}]")
        print("-"*25)
        cfg.epoch = epoch
        sched_fxn(epoch,argdict)
        result_tr = train_loop(cfg,model,loss_fxn,optim,loaders.tr)
        append_result_to_dict(results,result_tr)
        if epoch % cfg.save_interval == 0:
            save_model_checkpoint(cfg, model, optim, results, argdict)
        if epoch % cfg.test_interval == 0:
            result_te = test_loop(cfg,model,loaders.te,loss_fxn)
            append_result_to_dict(results,result_te)
    result_te = test_loop(cfg,model,loaders.te,loss_fxn)
    save_model_checkpoint(cfg, model, optim, results, argdict)
    append_result_to_dict(results,result_te)
    runtime = time.perf_counter() - start_time

    # -- format results --
    results['runtime'] = np.array([runtime])

    return results


def load_model(cfg,epoch=-1):

    # -- init --
    cfg = init_config(cfg)

    # -- get neural netowrk --
    model,loss_fxn,optim,sched_fxn,argdict = get_nn_model(cfg,cfg.nn_arch)

    # -- check if exists --
    if epoch == -1: ckpt_fn = 'ckpt.pth'
    else: ckpt_fn = 'ckpt_%d.pth' % epoch
    load_checkpoint(cfg, model, optim, argdict, ckpt_fn)

    return model

def init_config(cfg):

    # -- copy --
    cfg = copy.deepcopy(cfg)

    # -- set gpuid --
    if not('gpuid' in cfg):
        cfg.gpuid = 0
    if not('device' in cfg):
        cfg.device = 'cuda:%d' % cfg.gpuid

    # -- set frame size --
    assert not('frame_size' in cfg)
    # cfg.frame_size = [480,854]
    cfg.frame_size = [128,128]

    # -- set pid, if not already set --
    cproc = current_process()
    if not('pid' in cfg): cfg.pid = cproc.pid

    # -- set log info --
    cfg.train_log_interval = 20
    cfg.test_log_interval = 10
    cfg.test_interval = 10
    cfg.save_interval = 5

    # -- train params --
    cfg.nepochs = 60
    cfg.nsearch = 100 # num of patches to select perms from

    # -- reset sys.out if subprocess --
    cproc = current_process()
    if not(cfg.pid == cproc.pid):
        proc_dir = Path("./.proc_running")
        if not(proc_dir.exists()): proc_dir.mkdir()
        printfn = proc_dir / f"{os.getpid()}.txt"
        orig_stdout = sys.stdout
        f = open(printfn, 'w')
        sys.stdout = f

    # -- start iters --
    cfg.epoch = -1
    cfg.global_step = -1

    # -- return the copy --
    return cfg
