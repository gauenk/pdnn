
import torch
import numpy as np
import pandas as pd


def listdict_to_numpy(adict):
    for key,val in adict.items():
        print(key,type(val))
        if isinstance(val,list):
            print(type(val[0]))
            if isinstance(val[0],list):
                adict[key] = np.array(val)
            elif isinstance(val[0],np.ndarray):
                adict[key] = np.stack(val)
            elif torch.is_tensor(val[0]):
                adict[key] = torch.stack(val).numpy()
            else:
                adict[key] = np.array(val)
    return adict

def append_result_to_dict(records,epoch_result):
    for rdict in epoch_result:
        for key,val in rdict.items():
            if isinstance(val,list) and len(val) > 0:
                if isinstance(val[0],list):
                    val = np.array(val)
                elif isinstance(val[0],np.ndarray):
                    val = np.stack(val)
                elif torch.is_tensor(val[0]):
                    val = torch.stack(val).numpy()
                else:
                    val = np.array(val)
            elif torch.is_tensor(val):
                    val = val.numpy()
            if key in records: records[key].append(val)
            else: records[key] = [val]

def format_fields(mgrouped,index):

    # -- list keys --
    print(list(mgrouped.keys()))

    # -- get reference shapes --
    psnrs = mgrouped['psnrs']
    psnrs = np.stack(psnrs,axis=0)
    nmethods,nframes,batchsize = psnrs.shape

    # -- psnrs --
    print("psnrs.shape: ",psnrs.shape)
    psnrs = rearrange(psnrs,'m t i -> (m i) t')
    print("psnrs.shape: ",psnrs.shape)
    mgrouped['psnrs'] = psnrs

    # -- methods --
    methods = np.array(mgrouped['methods'])
    methods = repeat(methods,'m -> (m i)',i=batchsize)
    # rmethods = np.repeat(methods,batchsize).reshape(nmethods,batchsize)
    # tmethods = np.tile(rmethods,nframes).reshape(nmethods,nframes,batchsize)
    # methods = tmethods.ravel()
    print("methods.shape: ",methods.shape)
    mgrouped['methods'] = methods

    # -- runtimes --
    runtimes = np.array(mgrouped['runtimes'])
    runtimes = repeat(runtimes,'m -> (m i)',i=batchsize)
    # rruntimes = np.repeat(runtimes,batchsize).reshape(nmethods,batchsize)
    # truntimes = np.tile(rruntimes,nframes).reshape(nmethods,nframes,batchsize)
    # runtimes = truntimes.ravel()
    print("runtimes.shape: ",runtimes.shape)
    mgrouped['runtimes'] = runtimes

    # -- index --
    index = repeat(index,'i 1 -> (m i)',m=nmethods)
    print("index.shape: ",index.shape)
    mgrouped['image_index'] = index

    # -- test --
    df = pd.DataFrame().append(mgrouped,ignore_index=True)
    return df
