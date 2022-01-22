
from pathlib import Path
import cache_io


def config_from_uuid(uuid):

    # -- load cache --
    path = Path("/home/gauenk/Documents/packages/pdnn/")
    cache_root = path / ".cache_io"
    cache_name = "example"
    cache = cache_io.ExpCache(cache_root,cache_name)

    # -- get config --
    config = cache.get_config_from_uuid(uuid)
    assert config != -1,"invalid uuid. check your .cache_io folder :D"
    config.uuid = uuid
    config.cache_root = str(cache_root)
    return config

