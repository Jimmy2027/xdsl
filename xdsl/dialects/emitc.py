"""
Dialect to generate C/C++ from MLIR.

The EmitC dialect allows to convert operations from other MLIR dialects to EmitC ops.
Those can be translated to C/C++ via the Cpp emitter.

See external [documentation](https://mlir.llvm.org/docs/Dialects/EmitC/).
"""

from collections.abc import Iterable
from typing import cast

from xdsl.dialects import arith  # For arith.ConstantOp
from xdsl.dialects.builtin import (
    ArrayAttr,
    BFloat16Type,
    ContainerType,
    DenseArrayBase,
    Float16Type,
    Float32Type,
    Float64Type,
    IndexType,
    IntAttr,
    IntegerType,
    MemRefType,
    ShapedType,
    TensorType,
    TupleType,
    UnrealizedConversionCastOp,
)
from xdsl.dialects.memref import SubviewOp
from xdsl.ir import (
    Attribute,
    AttributeCovT,
    Block,
    Dialect,
    OpResult,
    ParametrizedAttribute,
    Region,
    SSAValue,
    TypeAttribute,  # Keep original name for class definitions
)

# Use this alias for xdsl.ir.TypeAttribute in type hints within functions
from xdsl.ir import TypeAttribute as IRTypeAttribute
from xdsl.irdl import (
    AnyAttr,
    AnyOf,
    BaseAttr,
    IRDLOperation,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    opt_prop_def,
    prop_def,
    region_def,
    result_def,
)
from xdsl.parser import AttrParser
from xdsl.printer import Printer
from xdsl.utils.exceptions import VerifyException


@irdl_attr_definition
class EmitC_ArrayType(
    ParametrizedAttribute, TypeAttribute, ShapedType, ContainerType[AttributeCovT]
):
    """EmitC array type"""

    name = "emitc.array"

    shape: ArrayAttr[IntAttr]
    element_type: AttributeCovT

    def __init__(
        self,
        shape: Iterable[int | IntAttr],
        element_type: AttributeCovT,
    ):
        shape = ArrayAttr(
            [IntAttr(dim) if isinstance(dim, int) else dim for dim in shape]
        )
        super().__init__(shape, element_type)

    def verify(self) -> None:
        if not self.shape.data:
            raise VerifyException("EmitC array shape must not be empty")

        for dim_attr in self.shape.data:
            if dim_attr.data < 0:
                raise VerifyException(
                    "EmitC array dimensions must have non-negative size"
                )

        element_type = self.get_element_type()

        if isinstance(element_type, EmitC_ArrayType):
            raise VerifyException(
                "EmitC array element type cannot be another EmitC_ArrayType."
            )

        # Check that the element type is a supported EmitC type.
        if not self._is_valid_element_type(element_type):
            raise VerifyException(
                f"EmitC array element type '{element_type}' is not a supported EmitC type."
            )

    def get_num_dims(self) -> int:
        return len(self.shape.data)

    def get_shape(self) -> tuple[int, ...]:
        return tuple(i.data for i in self.shape.data)

    def get_element_type(self) -> AttributeCovT:
        return self.element_type

    @classmethod
    def parse_parameters(cls, parser: AttrParser):
        with parser.in_angle_brackets():
            shape, type = parser.parse_ranked_shape()
            return ArrayAttr(IntAttr(dim) for dim in shape), type

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            printer.print_list(
                self.shape, lambda dim: printer.print_string(f"{dim.data}"), "x"
            )
            printer.print_string("x")
            printer.print_attribute(self.element_type)

    def _is_valid_element_type(self, element_type: Attribute) -> bool:
        """
        Check if the element type is valid for EmitC_ArrayType.
        See external [documentation](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/EmitC/IR/EmitCTypes.td#L77).
        """
        return is_integer_index_or_opaque_type(element_type) or is_supported_float_type(
            element_type
        )


@irdl_attr_definition
class EmitC_LValueType(ParametrizedAttribute, TypeAttribute):
    """
    EmitC lvalue type.
    Values of this type can be assigned to and their address can be taken.
    See [tablegen definition](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/EmitC/IR/EmitCTypes.td#L87)
    """

    name = "emitc.lvalue"
    value_type: TypeAttribute

    def verify(self) -> None:
        """
        Verify the LValueType.
        See [MLIR implementation](https://github.com/llvm/llvm-project/blob/main/mlir/lib/Dialect/EmitC/IR/EmitC.cpp#L1095)
        """
        # Check that the wrapped type is valid. This especially forbids nested lvalue types.
        if not is_supported_emitc_type(self.value_type):
            raise VerifyException(
                f"!emitc.lvalue must wrap supported emitc type, but got {self.value_type}"
            )
        if isinstance(self.value_type, EmitC_ArrayType):
            raise VerifyException("!emitc.lvalue cannot wrap !emitc.array type")


@irdl_attr_definition
class EmitC_OpaqueType(ParametrizedAttribute, TypeAttribute):
    """EmitC opaque type"""

    name = "emitc.opaque"
    value: ParameterDef[StringAttr]

    def __init__(self, value: StringAttr):
        super().__init__([value])

    def verify(self) -> None:
        if not self.value.data:
            raise VerifyException("expected non empty string in !emitc.opaque type")
        if self.value.data[-1] == "*":
            raise VerifyException(
                "pointer not allowed as outer type with !emitc.opaque, use !emitc.ptr instead"
            )

    @classmethod
    def parse_parameter(cls, parser: AttrParser):
        with parser.in_angle_brackets():
            value = parser.parse_str_literal()
            if not value:
                raise parser.raise_error(
                    "expected non empty string in !emitc.opaque type"
                )
            if value[-1] == "*":
                raise parser.raise_error(
                    "pointer not allowed as outer type with !emitc.opaque, use !emitc.ptr instead"
                )
            return StringAttr(value)


@irdl_attr_definition
class EmitC_PointerType(ParametrizedAttribute, TypeAttribute):
    """EmitC pointer type"""

    name = "emitc.ptr"
    pointee_type: ParameterDef[TypeAttribute]

    def __init__(self, pointee_type: TypeAttribute):
        super().__init__([pointee_type])

    def verify(self) -> None:
        if isinstance(self.pointee_type, EmitC_LValueType):
            raise VerifyException("pointers to lvalues are not allowed")

    @classmethod
    def parse_parameter(cls, parser: AttrParser):
        with parser.in_angle_brackets():
            type = parser.parse_type()
            if isinstance(type, EmitC_LValueType):
                raise parser.raise_error("pointers to lvalues are not allowed")
            return type


@irdl_attr_definition
class EmitC_PtrDiffT(ParametrizedAttribute, TypeAttribute):
    """EmitC signed pointer diff type"""

    name = "emitc.ptrdiff_t"


@irdl_attr_definition
class EmitC_SignedSizeT(ParametrizedAttribute, TypeAttribute):
    """EmitC signed size type"""

    name = "emitc.ssize_t"


@irdl_attr_definition
class EmitC_SizeT(ParametrizedAttribute, TypeAttribute):
    """EmitC unsigned size type"""

    name = "emitc.size_t"


@irdl_attr_definition
class EmitC_OpaqueAttr(ParametrizedAttribute):
    """An opaque attribute"""

    name = "emitc.opaque"
    value: ParameterDef[StringAttr]


def is_pointer_wide_type(type_attr: Attribute) -> bool:
    """Check if a type is a pointer-wide type."""
    return isinstance(type_attr, EmitC_SignedSizeT | EmitC_SizeT | EmitC_PtrDiffT)


_SUPPORTED_BITWIDTHS = (1, 8, 16, 32, 64)


def _is_supported_integer_type(type_attr: Attribute) -> bool:
    """
    Check if an IntegerType is supported by EmitC.
    See external [documentation](https://github.com/llvm/llvm-project/blob/main/mlir/lib/Dialect/EmitC/IR/EmitC.cpp#L96).
    """
    return (
        isinstance(type_attr, IntegerType)
        and type_attr.width.data in _SUPPORTED_BITWIDTHS
    )


def is_supported_float_type(type_attr: Attribute) -> bool:
    """
    Check if a type is a supported floating-point type in EmitC.
    See external [documentation](https://github.com/llvm/llvm-project/blob/main/mlir/lib/Dialect/EmitC/IR/EmitC.cpp#L117)
    """
    match type_attr:
        case Float16Type() | BFloat16Type() | Float32Type() | Float64Type():
            return True
        case _:
            return True

    if is_supported_float_type(type_attr):
        return True

    if isinstance(type_attr, IntegerType):
        # is_supported_integer_type handles the width check
        if _is_supported_integer_type(type_attr):
            return True

    if isinstance(type_attr, IndexType):
        return True

    if isinstance(type_attr, EmitC_OpaqueType):
        return True

    if is_pointer_wide_type(type_attr):
        return True

    if isinstance(type_attr, EmitC_PointerType):
        return is_supported_emitc_type(type_attr.pointee_type)

    if isinstance(type_attr, EmitC_ArrayType):
        elem_type: Attribute = type_attr.get_element_type()
        return not isinstance(elem_type, EmitC_ArrayType) and is_supported_emitc_type(
            elem_type
        )

    if isinstance(type_attr, IndexType) or is_pointer_wide_type(type_attr):
        return True

    if isinstance(type_attr, IntegerType):
        return is_supported_integer_type(type_attr)

    if isinstance(type_attr, AnyFloat):
        return is_supported_float_type(type_attr)

    if isinstance(type_attr, TensorType):
        elem_type = type_attr.get_element_type()
        if isinstance(elem_type, EmitC_ArrayType):
            return False


def is_integer_index_or_opaque_type(
    type_attr: Attribute,
) -> bool:
    """
    Check if a type is an integer, index, or opaque type.

    The emitC opaque type is not implemented yet so this function currently checks
    only for integer and index types.
    See external [documentation](https://github.com/llvm/llvm-project/blob/main/mlir/lib/Dialect/EmitC/IR/EmitC.cpp#L112).
    """
    return _is_supported_integer_type(type_attr) or isinstance(type_attr, IndexType)


def is_supported_emitc_type(type_attr: Attribute) -> bool:
    """
    Check if a type is supported by EmitC.
    See [MLIR implementation](https://github.com/llvm/llvm-project/blob/main/mlir/lib/Dialect/EmitC/IR/EmitC.cpp#L62).
    """
    match type_attr:
        case IntegerType():
            return _is_supported_integer_type(type_attr)
        case IndexType():
            return True
        case EmitC_ArrayType():
            elem_type = cast(Attribute, type_attr.get_element_type())
            return not isinstance(
                elem_type, EmitC_ArrayType
            ) and is_supported_emitc_type(elem_type)
        case Float16Type() | BFloat16Type() | Float32Type() | Float64Type():
            return True
        case TensorType():
            elem_type = cast(Attribute, type_attr.get_element_type())
            if isinstance(elem_type, EmitC_ArrayType):
                return False
            return is_supported_emitc_type(elem_type)
        case TupleType():
            return all(
                not isinstance(t, EmitC_ArrayType) and is_supported_emitc_type(t)
                for t in type_attr.types
            )
        case _:
            return False


def is_supported_emitc_type(type_attr: Attribute) -> bool:
    """
    Check if a type is supported by EmitC.
    See [MLIR implementation](https://github.com/llvm/llvm-project/blob/main/mlir/lib/Dialect/EmitC/IR/EmitC.cpp#L62).
    """
    match type_attr:
        case IntegerType():
            return _is_supported_integer_type(type_attr)
        case IndexType():
            return True
        case EmitC_ArrayType():
            elem_type = cast(Attribute, type_attr.get_element_type())
            return not isa(elem_type, EmitC_ArrayType) and is_supported_emitc_type(
                elem_type
            )
        case Float16Type() | BFloat16Type() | Float32Type() | Float64Type():
            return True
        case TensorType():
            elem_type = cast(Attribute, type_attr.get_element_type())
            if isinstance(elem_type, EmitC_ArrayType):
                return False
            return is_supported_emitc_type(elem_type)
        # TODO: Tuple type is not implemented yet, but it should be a valid emitc type.
        # case TupleType():
        #     return all(
        #         not isinstance(t, EmitC_ArrayType) and is_supported_emitc_type(t)
        #         for t in type_attr.types
        #     )
        case _:
            return False


@irdl_op_definition
class EmitC_AddOp(IRDLOperation):
    """Addition operation"""

    name = "emitc.add"
    assembly_format = """operands attr-dict `:` functional-type(operands, results)"""
    lhs = operand_def(AnyAttr())
    rhs = operand_def(AnyAttr())
    v1 = result_def(AnyAttr())


@irdl_op_definition
class EmitC_ApplyOp(IRDLOperation):
    """Apply operation"""

    name = "emitc.apply"
    assembly_format = """
        $applicableOperator `(` $operand `)` attr-dict `:` functional-type($operand, results)
      """
    applicableOperator = prop_def(AnyAttr())
    operand = operand_def(AnyOf((AnyAttr(), BaseAttr(EmitC_LValueType))))
    result = result_def(AnyAttr())


@irdl_op_definition
class EmitC_AssignOp(IRDLOperation):
    """Assign operation"""

    name = "emitc.assign"
    var = prop_def(AnyAttr())
    value = operand_def(AnyAttr())


@irdl_op_definition
class EmitC_BitwiseAndOp(IRDLOperation):
    name = "emitc.bitwise_and"


@irdl_op_definition
class EmitC_BitwiseLeftShiftOp(IRDLOperation):
    name = "emitc.bitwise_left_shift"


@irdl_op_definition
class EmitC_BitwiseNotOp(IRDLOperation):
    name = "emitc.bitwise_not"


@irdl_op_definition
class EmitC_BitwiseOrOp(IRDLOperation):
    name = "emitc.bitwise_or"


@irdl_op_definition
class EmitC_BitwiseRightShiftOp(IRDLOperation):
    name = "emitc.bitwise_right_shift"


@irdl_op_definition
class EmitC_BitwiseXorOp(IRDLOperation):
    name = "emitc.bitwise_xor"


@irdl_op_definition
class EmitC_CallOp(IRDLOperation):
    name = "emitc.call"


@irdl_op_definition
class EmitC_CallOpaqueOp(IRDLOperation):
    name = "emitc.call_opaque"


@irdl_op_definition
class EmitC_CastOp(IRDLOperation):
    name = "emitc.cast"


@irdl_op_definition
class EmitC_CmpOp(IRDLOperation):
    name = "emitc.cmp"


@irdl_op_definition
class EmitC_ConditionalOp(IRDLOperation):
    name = "emitc.conditional"


@irdl_op_definition
class EmitC_ConstantOp(IRDLOperation):
    name = "emitc.constant"


@irdl_op_definition
class EmitC_DeclareFuncOp(IRDLOperation):
    name = "emitc.declare_func"


@irdl_op_definition
class EmitC_DivOp(IRDLOperation):
    name = "emitc.div"


@irdl_op_definition
class EmitC_ExpressionOp(IRDLOperation):
    name = "emitc.expression"


@irdl_op_definition
class EmitC_FileOp(IRDLOperation):
    name = "emitc.file"


@irdl_op_definition
class EmitC_ForOp(IRDLOperation):
    name = "emitc.for"
    lower_bound = operand_def(AnyAttr())
    upper_bound = operand_def(AnyAttr())
    step = operand_def(AnyAttr())
    body: Region = region_def("single_block")


@irdl_op_definition
class EmitC_FuncOp(IRDLOperation):
    name = "emitc.func"
    sym_name: SymbolNameAttr = prop_def(SymbolNameAttr)
    function_type = prop_def(AnyAttr())
    body: Region = region_def("single_block")


@irdl_op_definition
class EmitC_GetGlobalOp(IRDLOperation):
    name = "emitc.get_global"


@irdl_op_definition
class EmitC_GlobalOp(IRDLOperation):
    name = "emitc.global"
    sym_name: SymbolNameAttr = prop_def(SymbolNameAttr)
    type: EmitC_ArrayType = prop_def(EmitC_ArrayType)
    initial_value: Attribute | None = opt_prop_def(Attribute)


@irdl_op_definition
class EmitC_IfOp(IRDLOperation):
    name = "emitc.if"


@irdl_op_definition
class EmitC_IncludeOp(IRDLOperation):
    name = "emitc.include"


@irdl_op_definition
class EmitC_LiteralOp(IRDLOperation):
    name = "emitc.literal"


@irdl_op_definition
class EmitC_LoadOp(IRDLOperation):
    name = "emitc.load"


@irdl_op_definition
class EmitC_LogicalAndOp(IRDLOperation):
    name = "emitc.logical_and"


@irdl_op_definition
class EmitC_LogicalNotOp(IRDLOperation):
    name = "emitc.logical_not"


@irdl_op_definition
class EmitC_LogicalOrOp(IRDLOperation):
    name = "emitc.logical_or"


@irdl_op_definition
class EmitC_MemberOfPtrOp(IRDLOperation):
    name = "emitc.member_of_ptr"


@irdl_op_definition
class EmitC_MemberOp(IRDLOperation):
    name = "emitc.member"


@irdl_op_definition
class EmitC_MulOp(IRDLOperation):
    name = "emitc.mul"


@irdl_op_definition
class EmitC_RemOp(IRDLOperation):
    name = "emitc.rem"


@irdl_op_definition
class EmitC_ReturnOp(IRDLOperation):
    name = "emitc.return"


@irdl_op_definition
class EmitC_SubOp(IRDLOperation):
    name = "emitc.sub"


@irdl_op_definition
class EmitC_SubscriptOp(IRDLOperation):
    name = "emitc.subscript"


@irdl_op_definition
class EmitC_SwitchOp(IRDLOperation):
    name = "emitc.switch"


@irdl_op_definition
class EmitC_UnaryMinusOp(IRDLOperation):
    name = "emitc.unary_minus"


@irdl_op_definition
class EmitC_UnaryPlusOp(IRDLOperation):
    name = "emitc.unary_plus"


@irdl_op_definition
class EmitC_VariableOp(IRDLOperation):
    name = "emitc.variable"


@irdl_op_definition
class EmitC_VerbatimOp(IRDLOperation):
    name = "emitc.verbatim"


@irdl_op_definition
class EmitC_YieldOp(IRDLOperation):
    name = "emitc.yield"


def _generate_emitc_nested_loops_for_copy(
    source_base: SSAValue,
    dest_base: SSAValue,
    element_type: IRTypeAttribute,
    offsets: list[SSAValue],
    sizes: list[SSAValue],
    strides: list[SSAValue],
    iv_type: IRTypeAttribute,
    const_zero_ssa: SSAValue,
    const_one_ssa: SSAValue,
) -> list[IRDLOperation]:
    """
    Generates a list of emitc operations to perform a copy equivalent to a
    memref.subview's data movement.

    Args:
        source_base: SSAValue for the source memref (expected to be an emitc type).
        dest_base: SSAValue for the destination memref (expected to be an emitc type).
        element_type: The scalar element TypeAttribute of the memref.
        offsets: List of SSAValues for the subview offsets.
        sizes: List of SSAValues for the subview sizes (loop bounds).
        strides: List of SSAValues for the subview strides.
        iv_type: The TypeAttribute for loop induction variables.
        const_zero_ssa: SSAValue for the constant 0 of iv_type.
        const_one_ssa: SSAValue for the constant 1 of iv_type.
    """
    rank = len(sizes)
    if rank == 0:
        return []

    iv_ssas: list[SSAValue | OpResult] = [source_base] * rank

    def build_loops_recursive(current_dim: int) -> list[IRDLOperation]:
        if current_dim == rank:
            ops_for_assignment_body: list[IRDLOperation] = []
            source_indices_ssas_local: list[SSAValue | OpResult] = [source_base] * rank
            for k_idx in range(rank):
                ivk_mul_stridek_op = arith.MuliOp(
                    iv_ssas[k_idx],
                    strides[k_idx],
                    result_type=iv_type,
                )
                ops_for_assignment_body.append(ivk_mul_stridek_op)
                src_idx_k_op = arith.AddiOp(
                    offsets[k_idx],
                    ivk_mul_stridek_op.results[0],
                    result_type=iv_type,
                )
                ops_for_assignment_body.append(src_idx_k_op)
                source_indices_ssas_local[k_idx] = src_idx_k_op.results[0]
            current_rhs_val_ssa: SSAValue | OpResult = source_base
            for k_idx in range(rank):
                res_type_rhs: IRTypeAttribute
                if k_idx == rank - 1:
                    res_type_rhs = element_type
                else:
                    res_type_rhs = EmitC_PointerType(element_type)
                subscript_op = EmitC_SubscriptOp.build(
                    operands=[current_rhs_val_ssa, source_indices_ssas_local[k_idx]],
                    result_types=[[res_type_rhs]],
                )
                ops_for_assignment_body.append(subscript_op)
                current_rhs_val_ssa = subscript_op.results[0]
            current_lhs_lvalue_ssa: SSAValue | OpResult = dest_base
            lvalue_of_element_type = EmitC_LValueType(element_type)
            for k_idx in range(rank):
                res_type_lhs: IRTypeAttribute
                if k_idx == rank - 1:
                    res_type_lhs = lvalue_of_element_type
                else:
                    res_type_lhs = EmitC_PointerType(element_type)
                subscript_op = EmitC_SubscriptOp.build(
                    operands=[current_lhs_lvalue_ssa, iv_ssas[k_idx]],
                    result_types=[[res_type_lhs]],
                )
                ops_for_assignment_body.append(subscript_op)
                current_lhs_lvalue_ssa = subscript_op.results[0]
            assign_op = EmitC_AssignOp.build(
                operands=[current_rhs_val_ssa],
                properties={"var": current_lhs_lvalue_ssa.type},
                result_types=[],
            )
            ops_for_assignment_body.append(assign_op)
            return ops_for_assignment_body

        for_op = EmitC_ForOp.build(
            operands=[const_zero_ssa, sizes[current_dim], const_one_ssa],
            regions=[Region([Block(arg_types=[iv_type])])],  # type: ignore
        )
        iv_ssas[current_dim] = for_op.regions[0].blocks[0].args[0]
        inner_ops = build_loops_recursive(current_dim + 1)
        for_op.regions[0].blocks[0].add_ops(inner_ops)
        return [for_op]

    final_ops = build_loops_recursive(0)
    return final_ops


def generate_emitc_subview_copy_ops(
    subview_op: SubviewOp,
    dest_emitc_array_ssa: SSAValue,  # SSAValue of EmitC_ArrayType for the destination
    insertion_block: Block,
) -> list[IRDLOperation]:
    """
    Generates emitc operations to perform a copy equivalent to a memref.subview.
    Prerequisite ops (casts, constants) are inserted into insertion_block.
    Returns the list of loop ops, which should also be added to insertion_block by the caller.
    """
    source_memref_val = subview_op.source
    if not isinstance(source_memref_val.type, MemRefType):
        raise TypeError("Subview source is not a MemRefType")
    source_memref_type: MemRefType = source_memref_val.type

    # 1. Cast source memref to EmitC_ArrayType
    emitc_source_type = EmitC_ArrayType(
        source_memref_type.get_shape(), source_memref_type.element_type
    )
    cast_source_op = UnrealizedConversionCastOp.build(
        operands=[source_memref_val], result_types=[[emitc_source_type]]
    )
    insertion_block.add_op(cast_source_op)
    emitc_source_ssa = SSAValue.get(cast_source_op.results[0])

    element_type = source_memref_type.element_type
    iv_type = IndexType()

    # 2. Create constants for 0 and 1
    const_zero_op = arith.ConstantOp.from_int_and_width(0, iv_type)
    const_one_op = arith.ConstantOp.from_int_and_width(1, iv_type)
    insertion_block.add_ops([const_zero_op, const_one_op])
    const_zero_ssa = SSAValue.get(const_zero_op.results[0])
    const_one_ssa = SSAValue.get(const_one_op.results[0])

    # 3. Extract offsets, sizes, strides from SubviewOp and create constants
    def attr_to_ssas(attr: Attribute, op_name: str) -> list[SSAValue]:
        if not isinstance(attr, DenseArrayBase):
            raise TypeError(
                f"Subview {op_name} is not a DenseIntElementsAttr for static extraction. Dynamic not supported here."
            )
        ssas: list[SSAValue] = []

        for val in attr.iter_values():
            const_op = arith.ConstantOp.from_int_and_width(val, iv_type)
            insertion_block.add_op(const_op)
            ssas.append(SSAValue.get(const_op.results[0]))
        return ssas

    offsets_ssas = attr_to_ssas(subview_op.static_offsets, "static_offsets")
    sizes_ssas = attr_to_ssas(subview_op.static_sizes, "static_sizes")
    strides_ssas = attr_to_ssas(subview_op.static_strides, "static_strides")

    # 4. Call the loop generation logic
    return _generate_emitc_nested_loops_for_copy(
        source_base=emitc_source_ssa,
        dest_base=dest_emitc_array_ssa,
        element_type=element_type,
        offsets=offsets_ssas,
        sizes=sizes_ssas,
        strides=strides_ssas,
        iv_type=iv_type,
        const_zero_ssa=const_zero_ssa,
        const_one_ssa=const_one_ssa,
    )


EmitC = Dialect(
    "emitc",
    [
        EmitC_AddOp,
        EmitC_ApplyOp,
        EmitC_AssignOp,
        EmitC_BitwiseAndOp,
        EmitC_BitwiseLeftShiftOp,
        EmitC_BitwiseNotOp,
        EmitC_BitwiseOrOp,
        EmitC_BitwiseRightShiftOp,
        EmitC_BitwiseXorOp,
        EmitC_CallOp,
        EmitC_CallOpaqueOp,
        EmitC_CastOp,
        EmitC_CmpOp,
        EmitC_ConditionalOp,
        EmitC_ConstantOp,
        EmitC_DeclareFuncOp,
        EmitC_DivOp,
        EmitC_ExpressionOp,
        EmitC_FileOp,
        EmitC_ForOp,
        EmitC_FuncOp,
        EmitC_GetGlobalOp,
        EmitC_GlobalOp,
        EmitC_IfOp,
        EmitC_IncludeOp,
        EmitC_LiteralOp,
        EmitC_LoadOp,
        EmitC_LogicalAndOp,
        EmitC_LogicalNotOp,
        EmitC_LogicalOrOp,
        EmitC_MemberOfPtrOp,
        EmitC_MemberOp,
        EmitC_MulOp,
        EmitC_RemOp,
        EmitC_ReturnOp,
        EmitC_SubOp,
        EmitC_SubscriptOp,
        EmitC_SwitchOp,
        EmitC_UnaryMinusOp,
        EmitC_UnaryPlusOp,
        EmitC_VariableOp,
        EmitC_VerbatimOp,
        EmitC_YieldOp,
    ],
    [
        EmitC_ArrayType,
        EmitC_LValueType,
        EmitC_LValueType,
        EmitC_OpaqueType,
        EmitC_PointerType,
        EmitC_PtrDiffT,
        EmitC_SignedSizeT,
        EmitC_SizeT,
        EmitC_OpaqueAttr,
    ],
)
