# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
from torch import distributed as dist
from torch.distributed.tensor import DTensor


@torch.no_grad()
def clip_grad_norm_mixed(
    parameters: torch.Tensor | list[torch.Tensor],
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: bool | None = None,
    pp_mesh=None,
) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return torch.tensor(0.0)

    dtensor_grads = [grad for grad in grads if isinstance(grad, DTensor)]
    local_grads = [grad for grad in grads if not isinstance(grad, DTensor)]

    if dtensor_grads and local_grads:
        dtensor_norm = torch.nn.utils.get_total_norm(
            dtensor_grads, norm_type, error_if_nonfinite, foreach
        )
        if isinstance(dtensor_norm, DTensor):
            dtensor_norm = dtensor_norm.full_tensor()
        local_norm = torch.nn.utils.get_total_norm(
            local_grads, norm_type, error_if_nonfinite, foreach
        )
        local_norm = local_norm.to(dtensor_norm.device)
        if math.isinf(norm_type):
            total_norm = torch.maximum(dtensor_norm, local_norm)
        else:
            total_norm = dtensor_norm**norm_type + local_norm**norm_type
            total_norm **= 1.0 / norm_type
        foreach = False
    elif dtensor_grads:
        total_norm = torch.nn.utils.get_total_norm(
            dtensor_grads, norm_type, error_if_nonfinite, foreach
        )
    else:
        total_norm = torch.nn.utils.get_total_norm(
            local_grads, norm_type, error_if_nonfinite, foreach
        )

    if isinstance(total_norm, DTensor):
        total_norm = total_norm.full_tensor()

    if pp_mesh is not None:
        if math.isinf(norm_type):
            dist.all_reduce(total_norm, op=dist.ReduceOp.MAX, group=pp_mesh.get_group())
        else:
            total_norm **= norm_type
            dist.all_reduce(total_norm, op=dist.ReduceOp.SUM, group=pp_mesh.get_group())
            total_norm **= 1.0 / norm_type

    torch.nn.utils.clip_grads_with_norm_(parameters, max_norm, total_norm, foreach)
    return total_norm
