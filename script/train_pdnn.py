

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
cache.clear() # optionally reset values

#
# -- (2) Load An Meshgrid of Python Dicts: each describe an experiment --
#

exps = {"noise_level":[10.,25.,50.],
        "batch_size": [10],
        "nepochs": [10],
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
print(records.columns)
print(records[['accuracy','precision','dataset','nn_arch','noise_level']])

