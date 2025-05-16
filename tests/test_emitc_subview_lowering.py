import io
from collections.abc import Sequence  # Added Sequence
from typing import cast

import pytest

from xdsl.dialects import arith, memref
from xdsl.dialects.builtin import (
    DenseIntElementsAttr,
    IndexType,  # Re-imported for explicit alignment
    IntegerAttr,
    MemRefType,
    UnrealizedConversionCastOp,
)
from xdsl.dialects.builtin import (
    i8 as i8_type,
)
from xdsl.dialects.emitc import (
    EmitC_ArrayType,
    EmitC_AssignOp,
    EmitC_ForOp,
    lower_subview_to_affine_loops,
)
from xdsl.ir import Attribute, Block  # Added Attribute for type checking
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

    copy_ops = lower_subview_to_affine_loops(
        subview_op=subview_op,
        insertion_block=insertion_block,
    )
    insertion_block.add_ops(copy_ops)

    output_buffer = io.StringIO()
    printer = Printer(stream=output_buffer)
    for op_index, op_in_block in enumerate(insertion_block.ops):
        op_in_block.print(printer)
        if op_index < len(insertion_block.ops) - 1:
            printer.print("\n")
    output_mlir = output_buffer.getvalue()

    print(f"Generated EmitC ops in insertion_block:\n{output_mlir}")

    assert "emitc.for" in output_mlir
    assert output_mlir.count("emitc.for") == len(static_sizes)
    assert "emitc.assign" in output_mlir
    assert "emitc.subscript" in output_mlir
    assert "emitc.add" in output_mlir
    assert "emitc.mul" in output_mlir

    cast_ops_in_block = [
        op for op in insertion_block.ops if isinstance(op, UnrealizedConversionCastOp)
    ]
    assert len(cast_ops_in_block) == 1
    found_cast_op = cast_ops_in_block[0]
    assert found_cast_op.operands[0] == source_memref_ssa
    cast_result_type = found_cast_op.results[0].type

    assert isinstance(cast_result_type, EmitC_ArrayType)

    cast_type_shape_attr = cast_result_type.shape
    if _is_dense_int_elements_attr(cast_type_shape_attr):
        # Cast to DenseIntElementsAttr to help type checker with .data attribute
        dense_cast_attr = cast(DenseIntElementsAttr, cast_type_shape_attr)
        # Further cast .data to Sequence[IntegerAttr] for iteration
        data_iterable_cast = cast(Sequence[IntegerAttr], dense_cast_attr.data)
        cast_type_shape = [d.data for d in data_iterable_cast]

        source_type_shape_attr = source_memref_type.shape
        if _is_dense_int_elements_attr(source_type_shape_attr):
            dense_source_attr = cast(DenseIntElementsAttr, source_type_shape_attr)
            data_iterable_source = cast(Sequence[IntegerAttr], dense_source_attr.data)
            source_type_shape = [d.data for d in data_iterable_source]
            assert cast_type_shape == source_type_shape, (
                f"Cast result shape {cast_type_shape} does not match source shape {source_type_shape}"
            )
        else:
            pytest.fail("Source memref shape is not DenseIntElementsAttr")
    else:
        pytest.fail("Casted EmitC_ArrayType shape is not DenseIntElementsAttr")

    expected_constant_values = set(
        static_offsets + static_sizes + static_strides + [0, 1]
    )
    actual_constant_values: set[int] = set()
    for op in insertion_block.ops:
        if isinstance(op, arith.ConstantOp) and isinstance(op.value, IntegerAttr):
            actual_constant_values.add(op.value.value.data)

    assert expected_constant_values == actual_constant_values, (
        f"Expected constant values {expected_constant_values}, got {actual_constant_values}"
    )

    for_ops_in_block = [op for op in insertion_block.ops if isinstance(op, EmitC_ForOp)]
    assert len(for_ops_in_block) == 1
    outer_for_op = for_ops_in_block[0]

    outer_loop_bound_ssa = outer_for_op.operands[1]
    assert isinstance(outer_loop_bound_ssa.owner, arith.ConstantOp)
    assert outer_loop_bound_ssa.owner in insertion_block.ops
    assert isinstance(outer_loop_bound_ssa.owner.value, IntegerAttr)
    assert outer_loop_bound_ssa.owner.value.value.data == static_sizes[0]

    inner_for_op = None
    for op_in_loop_body in outer_for_op.regions[0].blocks[0].ops:
        if isinstance(op_in_loop_body, EmitC_ForOp):
            inner_for_op = op_in_loop_body
            break
    assert inner_for_op is not None

    inner_loop_bound_ssa = inner_for_op.operands[1]  # type: ignore[union-attr]
    assert isinstance(inner_loop_bound_ssa.owner, arith.ConstantOp)
    assert inner_loop_bound_ssa.owner in insertion_block.ops
    assert isinstance(inner_loop_bound_ssa.owner.value, IntegerAttr)
    assert inner_loop_bound_ssa.owner.value.value.data == static_sizes[1]

    assign_op_found = False
    for op_in_inner_loop_body in inner_for_op.regions[0].blocks[0].ops:  # type: ignore[union-attr]
        if isinstance(op_in_inner_loop_body, EmitC_AssignOp):
            assign_op_found = True
            break
    assert assign_op_found

    print("Test test_convert_memref_subview_to_emitc passed basic structural checks.")


test_convert_memref_subview_to_emitc()
