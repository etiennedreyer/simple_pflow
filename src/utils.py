import torch

transform_funcs = {
    "minmax": lambda x, d: (x - d["min"]) / (d["max"] - d["min"]),
    "minmax_inv": lambda x, d: x * (d["max"] - d["min"]) + d["min"],
    "minmax_sym": lambda x, d: 2 * (x - d["min"]) / (d["max"] - d["min"]) - 1,
    "minmax_sym_inv": lambda x, d: ((x + 1) / 2) * (d["max"] - d["min"]) + d["min"],
    "log": lambda x, d: (torch.log(x + d.get("offset", 0)) + d.get("shift", 0)) / d.get("norm", 1),
    "log_inv": lambda x, d: torch.exp(x * d.get("norm", 1) - d.get("shift", 0)) - d.get("offset", 0),
    "standard": lambda x, d: (x - d["mean"]) / d["std"],
    "standard_inv": lambda x, d: x * d["std"] + d["mean"],
    "none": lambda x, _: x,
    "none_inv": lambda x, _: x
}

def transform(x, var, cfg, inverse=False):

    if var not in cfg:
        raise ValueError(f"Variable {var} not configured")

    if cfg[var]["type"] not in transform_funcs:
        raise NotImplementedError(
            f"Transform {cfg[var]['type']} not implemented"
        )

    key = cfg[var]["type"]
    if inverse:
        key += "_inv"

    return transform_funcs[key](x, cfg[var])

def get_max_N_safe(indices, N_max=None):

    '''
    Checks whether external maximum is suffiently big if provided
    Otherwise returns maximum index + 1
    '''
    N_max_indices = indices.max().item() + 1
    if N_max is not None:
        assert N_max >= N_max_indices, \
            f"max number in batch ({N_max_indices}) > N_max argument ({N_max})"
        return N_max
    return N_max_indices
