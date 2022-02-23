"""Redefine the interfaces similar to lazy_model.py in lazy_tensor_core."""
# pylint: disable=invalid-name, protected-access, c-extension-no-member, too-many-nested-blocks
import torch

from mnm import distributed as dist
import _TORCHMNMC

REDUCE_SUM = "sum"
REDUCE_MUL = "mul"
REDUCE_AND = "and"
REDUCE_OR = "or"
REDUCE_MIN = "min"
REDUCE_MAX = "max"


def all_reduce(reduce_type, inputs, scale=1.0, groups=None):
    """Performs an inplace reduce operation on the input tensor(s).

    Args:
      reduce_type (string): One of ``ltm.REDUCE_SUM``, ``ltm.REDUCE_MUL``,
      ``ltm.REDUCE_AND``, ``ltm.REDUCE_OR``, ``ltm.REDUCE_MIN`` and ``ltm.REDUCE_MAX``.
      inputs: Either a single `torch.Tensor` or a list of `torch.Tensor` to
        perform the all reduce op to.
      scale (float): A default scaling value to be applied after the reduce.
        Default: 1.0
      groups (list, optional): A list of list, representing the replica groups for
        the `all_reduce()` operation. Example: `[[0, 1, 2, 3], [4, 5, 6, 7]]`
          defines two groups, one with the `[0, 1, 2, 3]` replicas and one with
          the `[4, 5, 6, 7]` replicas. If `None` there will be only one group with
          all the replicas in it.
    Returns:
      If a single `torch.Tensor` is passed, the return value is a `torch.Tensor`
      holding the reduced value (across the replicas). If a list/tuple is passed,
      this function performs an inplace all-reduce op on the input tensors, and
      returns the list/tuple itself.
    """

    dctx = dist.get_context()
    if groups is None:
        groups = [list(range(0, dctx.size))]

    if isinstance(inputs, torch.Tensor):
        token = _TORCHMNMC._mnm_create_token(inputs.device.type)
        result = _TORCHMNMC._ltc_all_reduce(reduce_type, inputs, token, scale, groups)
        results = [result[0]]
    else:
        token = _TORCHMNMC._mnm_create_token(inputs[0].device.type)
        _ = _TORCHMNMC._ltc_all_reduce_inplace(reduce_type, inputs, token, scale, groups)
        results = inputs

    return results[0] if isinstance(inputs, torch.Tensor) else results


def all_gather(value, dim=0, groups=None):
    """Performs an all-gather operation along a given dimension.

    Args:
        value (torch.Tensor): The input tensor.
        dim (int): The gather dimension. Default: 0
        groups (list, optional): A list of list, representing the replica groups for
        the all_gather() operation.
        Example: [[0, 1, 2, 3], [4, 5, 6, 7]] defines two groups, one with the [0, 1, 2, 3]
        replicas and one with the [4, 5, 6, 7] replicas. If None there will be only one group
        with all the replicas in it.

    Returns:
        A tensor which has, in the dim dimension, all the values from the participating replicas.
    """
    dctx = dist.get_context()
    if groups is None:
        groups = [list(range(0, dctx.size))]
    result = _TORCHMNMC._mnm_all_gather(value, dim, groups)
    return result


def reduce_gradients(optimizer, groups=None):
    """Reduces all the gradients handled by an optimizer.

    Args:
      optimizer (:class:`torch.Optimizer`): The `torch.Optimizer` instance
        containing the gradients to be reduced.
      groups (list, optional): A list of list, representing the replica groups for
        the `all_reduce()` operation. Example: `[[0, 1, 2, 3], [4, 5, 6, 7]]`
          defines two groups, one with the `[0, 1, 2, 3]` replicas and one with
          the `[4, 5, 6, 7]` replicas. If `None` there will be only one group with
          all the replicas in it.
    """
    dctx = dist.get_context()
    world_size = dctx.size
    if world_size > 1:
        for param_group in optimizer.__getstate__()["param_groups"]:
            for group, params in param_group.items():
                if group == "params":
                    for p in params:
                        if isinstance(p, torch.Tensor) and p.grad is not None:
                            p.grad = all_reduce(
                                REDUCE_SUM, p.grad, scale=1.0 / world_size, groups=groups
                            )
