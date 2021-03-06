
# -- linalg --
import torch as th
import numpy as np

# -- patch-based deep neural network --
import pdnn

# -- caching --
import cache_io

#
# -- (1) Init Experiment Cache  --
#

"""
New goal: we want to train using the cascade of nearest neighbors
not just the topK but the "topK+shift" as well.

I suspect out model doesn't correctly handle the "non-iid"
aspect of the images well

maybe we can add a mechanism to address this..

thinking...

maybe each patch can retrieve its topk neighbors in the space
and then we denoise each with their topk??
here k = 15 from pacnet....

testing with this yields...

bad results.



"""

verbose = False
cache_root = ".cache_io"
cache_name = "v2"
cache = cache_io.ExpCache(cache_root,cache_name)
# cache.clear() # optionally reset values


#
# -- (2) Load An Meshgrid of Python Dicts: each describe an experiment --
#

exps = {"noise_level":[30.,50.],
        "ps": [13],
        "npatches":[30],
        "nneigh":[15],
        "batch_size": [4],
        "nn_arch":["sepnn"],
        "dataset":["davis"],
        # --  cache info for each exp --
        "cache_root":[cache_root]
}
experiments = cache_io.mesh_pydicts(exps)

"""

is an entry in the uuid_database created
when a uuid is created?

the goal is to re-read the uuid when a config has been
passed through

so an entry in the uuid database does not mean the "results"
are computed.

it merely means the assignment exists

Do we use a uuid existing in the database as an indication the
result has been computed? E.g. "results = None" only if there is
no uuid?

"""


# -- (3) [Execute or Load] each Experiment --
nexps = len(experiments)
for exp_num,config in enumerate(experiments):

    # -- info --
    if verbose:
        print("-="*25+"-")
        print(f"Running exeriment number {exp_num+1}/{nexps}")
        print("-="*25+"-")
        print(config)

    # -- optionally, load from cache --
    results = cache.load_exp(config) # possibly load result
    uuid = cache.get_uuid(config) # assing ID to each Dict in Meshgrid

    # -- exec if no results --
    if results is None: # check if no result
        # -- append info to cache --
        config.uuid = uuid
        results = pdnn.exec_learn(config)
        cache.save_exp(uuid,config,results) # save to cache

print("Training models are complete!")

# -- (4) print results! --
records = cache.load_flat_records(experiments)
print("\n\n\n\n")
print("Available Fields to Inspect:")
print(list(records.columns))

# -- (5) load a model by specifying a config  --
config = experiments[0]
config.uuid = cache.get_uuid(config)
model_a = pdnn.load_model(config,epoch=-1) # most recent
model_b = pdnn.load_model(config,epoch=3) # @ specific epoch
delta = 0
for param_a,param_b in zip(model_a.parameters(),model_b.parameters()):
    data_a,data_b = param_a.data,param_b.data
    delta += th.mean((data_a-data_b)**2)
print("The two models are different, with [delta = %2.2e] " % delta)

# -- (6) inspect results by noise level --
print("\n\n\n\n")
print("Inspect by noise level")
print("\n\n")
for std,nrecord in records.groupby("noise_level"):

    # -- banner --
    print("-"*30)
    print("--- Results @ [std = %d] ---" % int(std))
    print("-"*30)

    # -- print results from last few global steps --
    giters = np.unique(nrecord["global_iter"].to_numpy())
    gs_perc90 = np.quantile(giters,.9)
    info = nrecord[nrecord["global_iter"] >= gs_perc90]

    # -- split train and test --
    train_info = nrecord[nrecord["mode"] == "train"]
    test_info = nrecord[nrecord["mode"] == "test"]

    # -- print psnrs info --
    tr_psnrs = np.stack(train_info['image_psnrs'].to_numpy())
    te_psnrs = np.stack(test_info['image_psnrs'].to_numpy())
    tr_psnrs_mean,tr_psnrs_std = tr_psnrs.mean(),tr_psnrs.std()
    te_psnrs_mean,te_psnrs_std = te_psnrs.mean(),te_psnrs.std()
    print("[PSNRS.tr]: %2.2f +/- %2.2f" % (tr_psnrs_mean,tr_psnrs_std))
    print("[PSNRS.te]: %2.2f +/- %2.2f" % (te_psnrs_mean,te_psnrs_std))

    # -- print patch subset info --
    # tr_psub = np.stack(train_info['patch_subset'].to_numpy())
    # te_psub = np.stack(test_info['patch_subset'].to_numpy())
    # tr_psub_mean,tr_psub_std = tr_psub.mean(),tr_psub.std()
    # te_psub_mean,te_psub_std = te_psub.mean(),te_psub.std()
    # print("%2.2f +/- %2.2f" % (tr_psub_mean,tr_psub_std))
    # print("%2.2f +/- %2.2f" % (te_psub_mean,te_psub_std))

