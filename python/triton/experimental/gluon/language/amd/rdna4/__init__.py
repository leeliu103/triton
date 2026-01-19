from triton._C.libtriton.gluon_ir import get_global_load_tr_layouts

from ..._core import _unwrap_if_constexpr, builtin, tensor, distributed_type
from ..._layouts import BlockedLayout, DistributedLinearLayout
from ..cdna3 import buffer_load, buffer_store
from .._ops import _wmma

__all__ = ["buffer_load", "buffer_store", "wmma", "global_load_transpose"]


@builtin
def global_load_transpose(ptr, blocked_layout, _semantic=None):
    blocked_layout = _unwrap_if_constexpr(blocked_layout)

    assert isinstance(blocked_layout, BlockedLayout), "blocked_layout must be a BlockedLayout"
    assert isinstance(ptr, tensor), "ptr must be a tensor"
    ptr_ty = ptr.type
    assert isinstance(ptr_ty, distributed_type), "ptr must be a distributed_type"
    ptr_rank = len(ptr_ty.shape)
    assert ptr_rank == 2, "global_load_transpose requires a 2D tensor"
    assert isinstance(ptr_ty.layout, DistributedLinearLayout) or isinstance(ptr_ty.layout, BlockedLayout), \
        "ptr layout must be a DistributedLinearLayout or BlockedLayout"

    ptr_scalar = ptr_ty.scalar
    assert hasattr(ptr_scalar, "is_ptr") and ptr_scalar.is_ptr(), "ptr must be a tensor of pointers"
    elem_ty = ptr_scalar.element_ty
    assert elem_ty.primitive_bitwidth == 16, "global_load_transpose supports 16-bit element types"

    addr_layout, data_layout = get_global_load_tr_layouts(blocked_layout, list(ptr_ty.shape))

    ptr_in = ptr
    if ptr_ty.layout != addr_layout:
        ptr_in = _semantic.convert_layout(ptr, addr_layout)

    data_ty = distributed_type(elem_ty, ptr_ty.shape, data_layout)
    handle = _semantic.builder.create_global_load_transpose(data_ty.to_ir(_semantic.builder), ptr_in.handle)
    return tensor(handle, data_ty)


@builtin
def wmma(a, b, acc, _semantic=None):
    """
    Computes matrix-multiplication of a * b + acc using AMD WMMA instruction.

    Args:
        a (tensor): The operand a to be multiplied.
        b (tensor): The operand b to be multiplied.
        acc (tensor): The accumulator tensor.
    """
    return _wmma(2, a, b, acc, _semantic)
