from xdsl.dialects import memref
from xdsl.dialects.builtin import (
    IndexType,  # Re-imported for explicit alignment
    IntegerAttr,
    MemRefType,
)
from xdsl.dialects.builtin import (
    i8 as i8_type,
)
from xdsl.dialects.emitc import (
    lower_subview_to_affine_loops,
)
from xdsl.ir import Block  # Added Attribute for type checking
from xdsl.printer import Printer


def test_convert_memref_subview_to_emitc():
    i8 = i8_type
    idx_type = IndexType()

    source_shape = [16, 8]
    source_memref_type = MemRefType(i8, source_shape)

    static_offsets = [0, 0]
    static_sizes = [8, 4]
    static_strides = [1, 1]

    insertion_block = Block(arg_types=[source_memref_type])

    temp_block = Block()
    alignment_attr = IntegerAttr(0, idx_type)
    alloc_op = memref.AllocOp.get(
        return_type=source_memref_type, dynamic_sizes=[], alignment=alignment_attr
    )
    temp_block.add_op(alloc_op)
    source_memref_ssa = alloc_op.results[0]

    subview_op = memref.SubviewOp.from_static_parameters(
        source_memref_ssa,
        source_memref_type,
        static_offsets,
        static_sizes,
        static_strides,
    )
    temp_block.add_op(subview_op)

    Printer().print_block(temp_block)

    lower_subview_to_affine_loops(
        subview_op=subview_op,
        insertion_block=insertion_block,
    )
    Printer().print_block(insertion_block)


test_convert_memref_subview_to_emitc()
