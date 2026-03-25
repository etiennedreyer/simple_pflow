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

def unflatten(flat: torch.Tensor, idx_0: torch.Tensor, 
              idx_1=None, max_len=None):
    B = idx_0.max().item() + 1
    d = flat.device

    if idx_1 is None:
        ### Count entries in each event
        counts = torch.bincount(idx_0, minlength=B) # (B,)

        ### Create index for dim=1
        cumsum = counts.cumsum(dim=0) # (B,)
        offsets = torch.cat((idx_0.new_zeros(1), cumsum[:-1])) # (B,)
        idx_1 = torch.arange(len(flat), device=d) - offsets[idx_0] # (N,)

    ### Compute max length for padding
    max_len_batch = idx_1.max().item() + 1
    if max_len is None:
        max_len = max_len_batch
    else:
        assert max_len >= max_len_batch, \
            f"max_len in batch ({max_len_batch}) > max_len argument ({max_len})"

    ### Placeholder for output
    unflat = torch.full((B, max_len), float('nan'), 
                        dtype=flat.dtype, device=d) # (B, max_len)

    ### Scatter into unflat tensor
    unflat[idx_0, idx_1] = flat

    return unflat, idx_1