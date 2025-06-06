from xdsl.context import Context
from xdsl.dialects import memref
from xdsl.dialects.arith import Arith
from xdsl.dialects.builtin import DenseArrayBase, IntegerType, TensorType, f64, i64
from xdsl.dialects.stencil import IndexAttr
from xdsl.dialects.tensor import ExpandShapeOp, ExtractSliceOp, InsertSliceOp, Tensor
from xdsl.dialects.test import TestOp
from xdsl.parser import Parser
from xdsl.utils.test_value import create_ssa_value


def test_extract_slice_static():
    input_t = TensorType(f64, [10, 20, 30])
    input_v = TestOp(result_types=[input_t]).res[0]

    extract_slice = ExtractSliceOp.from_static_parameters(input_v, [1, 2, 3], [4, 5, 6])

    assert extract_slice.source is input_v
    assert extract_slice.static_offsets == DenseArrayBase.from_list(i64, [1, 2, 3])
    assert extract_slice.static_sizes == DenseArrayBase.from_list(i64, [4, 5, 6])
    assert extract_slice.static_strides == DenseArrayBase.from_list(i64, [1, 1, 1])
    assert extract_slice.offsets == ()
    assert extract_slice.sizes == ()
    assert extract_slice.strides == ()
    assert extract_slice.result.type == TensorType(f64, [4, 5, 6])

    extract_slice = ExtractSliceOp.from_static_parameters(
        input_v, [1, 2, 3], [4, 5, 6], [8, 9, 10]
    )

    assert extract_slice.source is input_v
    assert extract_slice.static_offsets == DenseArrayBase.from_list(i64, [1, 2, 3])
    assert extract_slice.static_sizes == DenseArrayBase.from_list(i64, [4, 5, 6])
    assert extract_slice.static_strides == DenseArrayBase.from_list(i64, [8, 9, 10])
    assert extract_slice.offsets == ()
    assert extract_slice.sizes == ()
    assert extract_slice.strides == ()
    assert extract_slice.result.type == TensorType(f64, [4, 5, 6])


def test_insert_slice_static():
    source_t = TensorType(f64, [10, 20])
    source_v = TestOp(result_types=[source_t]).res[0]
    dest_t = TensorType(f64, [10, 20, 30])
    dest_v = TestOp(result_types=[dest_t]).res[0]

    insert_slice = InsertSliceOp.from_static_parameters(
        source_v, dest_v, [1, 2], [4, 5]
    )

    assert insert_slice.source is source_v
    assert insert_slice.dest is dest_v
    assert insert_slice.static_offsets == DenseArrayBase.from_list(i64, [1, 2])
    assert insert_slice.static_sizes == DenseArrayBase.from_list(i64, [4, 5])
    assert insert_slice.static_strides == DenseArrayBase.from_list(i64, [1, 1])
    assert insert_slice.offsets == ()
    assert insert_slice.sizes == ()
    assert insert_slice.strides == ()
    assert insert_slice.result.type == dest_t

    insert_slice = InsertSliceOp.from_static_parameters(
        source_v, dest_v, [1, 2], [4, 5], [8, 9]
    )

    assert insert_slice.source is source_v
    assert insert_slice.dest is dest_v
    assert insert_slice.static_offsets == DenseArrayBase.from_list(i64, [1, 2])
    assert insert_slice.static_sizes == DenseArrayBase.from_list(i64, [4, 5])
    assert insert_slice.static_strides == DenseArrayBase.from_list(i64, [8, 9])
    assert insert_slice.offsets == ()
    assert insert_slice.sizes == ()
    assert insert_slice.strides == ()
    assert insert_slice.result.type == dest_t


def test_insert_slice_dynamic():
    source_t = TensorType(f64, [10, 20])
    source_v = create_ssa_value(source_t)
    dest_t = TensorType(f64, [10, 20, 30])
    dest_v = create_ssa_value(dest_t)
    offset1 = create_ssa_value(IndexAttr.get(3))
    offset2 = create_ssa_value(IndexAttr.get(15))
    stride1 = create_ssa_value(IndexAttr.get(2))
    stride2 = create_ssa_value(IndexAttr.get(5))

    insert_slice = InsertSliceOp.get(
        source=source_v,
        dest=dest_v,
        static_sizes=[1, 2],
        offsets=[offset1, offset2],
        strides=[stride1, stride2],
    )

    assert insert_slice.static_offsets == DenseArrayBase.from_list(
        i64, 2 * [memref.SubviewOp.DYNAMIC_INDEX]
    )
    assert insert_slice.static_strides == DenseArrayBase.from_list(
        i64, 2 * [memref.SubviewOp.DYNAMIC_INDEX]
    )


def test_expand_shape_parse():
    MODULE_CTX = """
    %src = tensor.empty() : tensor<1x5xi32>
    %c0 = arith.constant 0 : index
    %dim = tensor.dim %src, %c0 : tensor<1x5xi32>
    %expanded = tensor.expand_shape %src [[0 : i64], [1 : i64, 2 : i64, 3 : i64]] output_shape [%dim, 1, 1, 5] : tensor<1x5xi32> into tensor<1x1x1x5xi32>
    """

    ctx = Context()
    ctx.load_dialect(Tensor)
    ctx.load_dialect(Arith)

    module_op = Parser(ctx, MODULE_CTX).parse_module()

    expand_shape_op = module_op.body.block.ops.last
    assert isinstance(expand_shape_op, ExpandShapeOp)

    assert expand_shape_op.src._name == "src"
    assert expand_shape_op.src.type == TensorType(IntegerType(32), [1, 5])
    assert expand_shape_op.result.type == TensorType(IntegerType(32), [1, 1, 1, 5])

    for line in MODULE_CTX.splitlines():
        assert line.strip() in str(module_op), f"{line.strip()} not in {str(module_op)}"
