

# -- python --
import numpy as np

# -- patch-based deep neural network --
import pdnn

# -- caching --
import cache_io

# -- load the data --
import datasets

#
# -- (1) Init Experiment Cache  --
#

verbose = False
cache_root = ".cache_io"
cache_name = "example"
cache = cache_io.ExpCache(cache_root,cache_name)
# cache.clear() # optionally reset values

#
# -- (2) Load An Meshgrid of Python Dicts: each describe an experiment --
#

exps = {"noise_level":[10.,25.,50.],
        "ps": [13],
        "npatches":[2],
        "nneigh":[15],
        "batch_size": [4],
        "nepochs": [3],
        "nn_arch":["sepnn"],
        "dataset":["davis"]}
experiments = cache_io.mesh_pydicts(exps)

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
        config.cache_root = cache_root

        results = pdnn.exec_learn(config)
        cache.save_exp(uuid,config,results) # save to cache

# -- (4) print results! --
records = cache.load_flat_records(experiments)
print("Available Fields to Inspect:")
print(list(records.columns))

# -- (5) inspect results by noise level --
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

