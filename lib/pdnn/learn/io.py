
# -- python --
from pathlib import Path
from easydict import EasyDict as edict

# -- cache --
import cache_io

# -- package --
from pdnn.utils import config_from_uuid
from .loop import load_model

def get_sigma_uuid(sigma):
    if int(sigma) == 10 or int(sigma*255.) == 10:
        return "e7c0822f-e5f8-436f-9c8f-cae89433befb"
    elif int(sigma) == 30 or int(sigma*255.) == 30:
        return "eb47647a-7ac3-44e9-ad37-c33e047fd7eb"
    else:
        raise KeyError(f"uknown sigma model [{sigma}]")

def load_sigma_model(sigma,device,epoch=-1):

    # -- create init dict --
    uuid = get_sigma_uuid(sigma)
    cfg = config_from_uuid(uuid)
    # cfg = edict()
    # cfg.cache_root = Path("/home/gauenk/Documents/packages/pdnn/.cache_io/")
    # cfg.nn_arch = "sepnn"
    # cfg.device = device
    # cfg.gpuid = 0 # maybe a bad idea
    # print(cfg)

    # -- load model --
    model = load_model(cfg,epoch)
    model = model.to(device)

    return model
