# ---------------------------------------------------------------
# Copyright (c) 2024 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License
# ---------------------------------------------------------------


import itertools


def get_param_group(params, lr):
    return {"params": list(params), "lr": lr}


def get_full_names(module, prefix):
    if hasattr(module, "named_parameters"):
        return {f"{prefix}.{name}" for name, _ in module.named_parameters()}
    return {prefix}


def process_parameters(param_defs, current_params):
    params = itertools.chain(
        *(
            mod.parameters() if hasattr(mod, "parameters") else [mod]
            for _, mod in param_defs
            if mod is not None
        )
    )
    param_names = set(
        itertools.chain(
            *(
                get_full_names(mod, name)
                if hasattr(mod, "parameters")
                else [name]
                for name, mod in param_defs
            )
        )
    )
    current_params -= param_names

    return params, current_params
