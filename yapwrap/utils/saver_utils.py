import torch
def recursive_naming(config):
    for k,v in config.items():
        if isinstance(v, dict):
            config[k] = recursive_naming(v)
        if not isinstance(v, (str, int, float, bool, dict, type(None))):
            if isinstance(v, (list, tuple, torch.Tensor)):
                config[k] = str(v)
            else:
                config[k] = v.__name__
    return config
