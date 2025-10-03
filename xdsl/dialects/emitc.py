"""
Dialect to generate C/C++ from MLIR.

The EmitC dialect allows to convert operations from other MLIR dialects to EmitC ops.
Those can be translated to C/C++ via the Cpp emitter.

See external [documentation](https://mlir.llvm.org/docs/Dialects/EmitC/).
"""

import abc
from collections.abc import Iterable, Mapping, Sequence
from enum import StrEnum
from typing import Generic, Literal

from typing_extensions import TypeVar

from xdsl.dialects.builtin import (
    ArrayAttr,
    BFloat16Type,
    ContainerType,
    DenseIntOrFPElementsAttr,
    FlatSymbolRefAttrConstr,
    Float16Type,
    Float32Type,
    Float64Type,
    FloatAttr,
    IndexType,
    IntAttr,
    IntegerAttr,
    IntegerType,
    ShapedType,
    StaticShapeArrayConstr,
    StringAttr,
    SymbolNameConstraint,
    TensorType,
    TupleType,
    UnitAttr,
)

# Use this alias for xdsl.ir.TypeAttribute in type hints within functions
from xdsl.ir import (
    Attribute,
    Block,
    Dialect,
    ParametrizedAttribute,
    Region,
    SSAValue,
    TypeAttribute,
)
from xdsl.irdl import (
    AnyAttr,
    AnyOf,
    AttrConstraint,
    BaseAttr,
    ConstraintContext,
    EqAttrConstraint,
    IntConstraint,
    IRDLOperation,
    ParsePropInAttrDict,
    irdl_attr_definition,
    irdl_op_definition,
    irdl_to_attr_constraint,
    operand_def,
    opt_operand_def,
    opt_prop_def,
    param_def,
    prop_def,
    region_def,
    result_def,
    traits_def,
    var_operand_def,
    var_region_def,
    var_result_def,
)
from xdsl.parser import AttrParser, Parser
from xdsl.printer import Printer
from xdsl.traits import IsTerminator
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.hints import isa


@irdl_attr_definition
class EmitC_OpaqueType(ParametrizedAttribute, TypeAttribute):
    """EmitC opaque type"""

    name = "emitc.opaque"
    value: StringAttr

    def verify(self) -> None:
        if not self.value.data:
            raise VerifyException("expected non empty string in !emitc.opaque type")
        if self.value.data[-1] == "*":
            raise VerifyException(
                "pointer not allowed as outer type with !emitc.opaque, use !emitc.ptr instead"
            )


@irdl_attr_definition
class EmitC_PtrDiffT(ParametrizedAttribute, TypeAttribute):
    """
    EmitC signed pointer diff type.
    Signed data type as wide as platform-specific pointer types. In particular, it is as wide as emitc.size_t.
    It corresponds to ptrdiff_t found in <stddef.h>.
    """

    name = "emitc.ptrdiff_t"


@irdl_attr_definition
class EmitC_SignedSizeT(ParametrizedAttribute, TypeAttribute):
    """
    EmitC signed size type.
    Data type representing all values of emitc.size_t, plus -1. It corresponds to ssize_t found in <sys/types.h>.
    Use of this type causes the code to be non-C99 compliant.
    """

    name = "emitc.ssize_t"


@irdl_attr_definition
class EmitC_SizeT(ParametrizedAttribute, TypeAttribute):
    """
    EmitC unsigned size type.
    Unsigned data type as wide as platform-specific pointer types. It corresponds to size_t found in <stddef.h>.
    """

    name = "emitc.size_t"


EmitCIntegerType = IntegerType[Literal[1, 8, 16, 32, 64]]
"""
Type for integer types supported by EmitC.
See external [documentation](https://github.com/llvm/llvm-project/blob/main/mlir/lib/Dialect/EmitC/IR/EmitC.cpp#L96).
"""

EmitCIntegerTypeConstr = irdl_to_attr_constraint(EmitCIntegerType)
"""
Constraint for integer types supported by EmitC.
See external [documentation](https://github.com/llvm/llvm-project/blob/main/mlir/lib/Dialect/EmitC/IR/EmitC.cpp#L96).
"""

EmitCFloatType = Float16Type | BFloat16Type | Float32Type | Float64Type
EmitCFloatTypeConstr = irdl_to_attr_constraint(EmitCFloatType)
"""
Supported floating-point type in EmitC.
See external [documentation](https://github.com/llvm/llvm-project/blob/main/mlir/lib/Dialect/EmitC/IR/EmitC.cpp#L117)
"""

EmitCPointerWideType = EmitC_PtrDiffT | EmitC_SignedSizeT | EmitC_SizeT
EmitCPointerWideTypeConstr = irdl_to_attr_constraint(EmitCPointerWideType)
"""
Constraint for pointer-wide types supported by EmitC.
These types have the same width as platform-specific pointer types.
"""

EmitCIntegerIndexOpaqueType = EmitCIntegerType | IndexType | EmitC_OpaqueType
EmitCIntegerIndexOpaqueTypeConstr = irdl_to_attr_constraint(EmitCIntegerIndexOpaqueType)
"""
Constraint for integer, index, or opaque types supported by EmitC.
"""

EmitCArrayElementType = (
    EmitCIntegerIndexOpaqueType | EmitCFloatType | EmitCPointerWideType
)
EmitCArrayElementTypeConstr = irdl_to_attr_constraint(EmitCArrayElementType)
"""
Constraint for valid element types in EmitC arrays.
"""


@irdl_attr_definition
class EmitC_OpaqueAttr(ParametrizedAttribute):
    """
    An opaque attribute of which the value gets emitted as is.
    """

    name = "emitc.opaque"
    value: StringAttr


EmitCArrayElementTypeCovT = TypeVar(
    "EmitCArrayElementTypeCovT",
    bound=EmitCArrayElementType,
    covariant=True,
    default=EmitCArrayElementType,
)


@irdl_attr_definition
class EmitC_ArrayType(
    ParametrizedAttribute,
    TypeAttribute,
    ShapedType,
    ContainerType[EmitCArrayElementTypeCovT],
    Generic[EmitCArrayElementTypeCovT],
):
    """EmitC array type"""

    name = "emitc.array"

    shape: ArrayAttr[IntAttr] = param_def(StaticShapeArrayConstr)
    element_type: EmitCArrayElementTypeCovT

    def __init__(
        self,
        shape: Iterable[int | IntAttr],
        element_type: EmitCArrayElementType,
    ):
        shape = ArrayAttr(
            [IntAttr(dim) if isinstance(dim, int) else dim for dim in shape]
        )
        super().__init__(shape, element_type)

    def verify(self) -> None:
        if not self.shape.data:
            raise VerifyException("EmitC array shape must not be empty")

        if isinstance(self.element_type, EmitC_ArrayType):
            raise VerifyException("nested EmitC arrays are not allowed")

        for dim_attr in self.shape.data:
            if dim_attr.data < 0:
                raise VerifyException(
                    "EmitC array dimensions must have non-negative size"
                )

    def get_num_dims(self) -> int:
        return len(self.shape.data)

    def get_shape(self) -> tuple[int, ...]:
        return tuple(i.data for i in self.shape.data)

    def get_element_type(self) -> EmitCArrayElementTypeCovT:
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


class EmitCTypeConstraint(AttrConstraint):
    """
    Check if a type is supported by EmitC.
    See [MLIR implementation](https://github.com/llvm/llvm-project/blob/main/mlir/lib/Dialect/EmitC/IR/EmitC.cpp#L62).
    """

    def verify(self, attr: Attribute, constraint_context: ConstraintContext) -> None:
        if isa(attr, TensorType):
            # EmitC only supports tensors with static shapes
            if not attr.has_static_shape():
                raise VerifyException(f"Type {attr} is not a supported EmitC type")
            elem_type = attr.get_element_type()
            if isinstance(elem_type, EmitC_ArrayType):
                raise VerifyException("EmitC type cannot be a tensor of EmitC arrays")
            self.verify(elem_type, constraint_context)
            return

        if isa(attr, EmitC_ArrayType):
            elem_type = attr.get_element_type()
            self.verify(elem_type, constraint_context)
            return

        if isinstance(attr, EmitC_PointerType):
            self.verify(attr.pointee_type, constraint_context)
            return

        if isinstance(attr, TupleType):
            for t in attr.types:
                if isinstance(t, EmitC_ArrayType):
                    raise VerifyException(
                        "EmitC type cannot be a tuple of EmitC arrays"
                    )
                self.verify(t, constraint_context)
            return

        EmitCArrayElementTypeConstr.verify(attr, constraint_context)

    def mapping_type_vars(
        self, type_var_mapping: Mapping[TypeVar, AttrConstraint | IntConstraint]
    ) -> AttrConstraint:
        # No type variables to map in this constraint
        return self


EmitCTypeConstr = EmitCTypeConstraint()


@irdl_attr_definition
class EmitC_LValueType(ParametrizedAttribute, TypeAttribute):
    """
    EmitC lvalue type.
    Values of this type can be assigned to and their address can be taken.
    See [tablegen definition](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/EmitC/IR/EmitCTypes.td#L87)
    """

    name = "emitc.lvalue"
    value_type: Attribute = param_def(EmitCTypeConstr)

    def verify(self) -> None:
        """
        Verify the LValueType.
        See [MLIR implementation](https://github.com/llvm/llvm-project/blob/main/mlir/lib/Dialect/EmitC/IR/EmitC.cpp#L1095)
        """
        if isinstance(self.value_type, EmitC_ArrayType):
            raise VerifyException("!emitc.lvalue cannot wrap !emitc.array type")


@irdl_attr_definition
class EmitC_PointerType(ParametrizedAttribute, TypeAttribute):
    """EmitC pointer type"""

    name = "emitc.ptr"
    pointee_type: TypeAttribute

    def verify(self) -> None:
        if isinstance(self.pointee_type, EmitC_LValueType):
            raise VerifyException("pointers to lvalues are not allowed")


class EmitC_BinaryOperation(IRDLOperation, abc.ABC):
    """Base class for EmitC binary operations."""

    lhs = operand_def(EmitCTypeConstr)
    rhs = operand_def(EmitCTypeConstr)
    result = result_def(EmitCTypeConstr)

    assembly_format = "operands attr-dict `:` functional-type(operands, results)"

    def __init__(
        self,
        lhs: SSAValue,
        rhs: SSAValue,
        result_type: Attribute,
    ):
        super().__init__(
            operands=[lhs, rhs],
            result_types=[result_type],
        )


@irdl_op_definition
class EmitC_AddOp(EmitC_BinaryOperation):
    """
    Addition operation.

    With the `emitc.add` operation the arithmetic operator + (addition) can
    be applied. Supports pointer arithmetic where one operand is a pointer
    and the other is an integer or opaque type.

    Example:

    ```mlir
    // Custom form of the addition operation.
    %0 = emitc.add %arg0, %arg1 : (i32, i32) -> i32
    %1 = emitc.add %arg2, %arg3 : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
    ```
    ```c++
    // Code emitted for the operations above.
    int32_t v5 = v1 + v2;
    float* v6 = v3 + v4;
    ```
    """

    name = "emitc.add"

    def verify_(self) -> None:
        lhs_type = self.lhs.type
        rhs_type = self.rhs.type

        if isa(lhs_type, EmitC_PointerType) and isa(rhs_type, EmitC_PointerType):
            raise VerifyException(
                "emitc.add requires that at most one operand is a pointer"
            )

        if (
            isa(lhs_type, EmitC_PointerType)
            and not isa(rhs_type, IntegerType | EmitC_OpaqueType)
        ) or (
            isa(rhs_type, EmitC_PointerType)
            and not isa(lhs_type, IntegerType | EmitC_OpaqueType)
        ):
            raise VerifyException(
                "emitc.add requires that one operand is an integer or of opaque "
                "type if the other is a pointer"
            )


@irdl_op_definition
class EmitC_ApplyOp(IRDLOperation):
    """Apply operation"""

    name = "emitc.apply"

    assembly_format = """
        $applicableOperator `(` $operand `)` attr-dict `:` functional-type($operand, results)
      """

    applicableOperator = prop_def(StringAttr)

    operand = operand_def(AnyAttr())

    result = result_def(EmitCTypeConstr)

    def verify_(self) -> None:
        applicable_operator = self.applicableOperator.data

        # Applicable operator must not be empty
        if not applicable_operator:
            raise VerifyException("applicable operator must not be empty")

        if applicable_operator not in ("&", "*"):
            raise VerifyException("applicable operator is illegal")

        operand_type = self.operand.type
        result_type = self.result.type

        if applicable_operator == "&":
            if not isinstance(operand_type, EmitC_LValueType):
                raise VerifyException(
                    "operand type must be an lvalue when applying `&`"
                )
            if not isinstance(result_type, EmitC_PointerType):
                raise VerifyException("result type must be a pointer when applying `&`")
        else:  # applicable_operator == "*"
            if not isinstance(operand_type, EmitC_PointerType):
                raise VerifyException(
                    "operand type must be a pointer when applying `*`"
                )

    def has_side_effects(self) -> bool:
        """Return True if the operation has side effects."""
        return self.applicableOperator.data == "*"


@irdl_op_definition
class EmitC_AssignOp(IRDLOperation):
    """Assign operation"""

    name = "emitc.assign"

    # Use custom parse/print to match MLIR format that omits !emitc.lvalue<> wrapper
    assembly_format = None

    var = operand_def(AnyAttr())
    value = operand_def(AnyAttr())

    @classmethod
    def parse(cls, parser: Parser) -> "EmitC_AssignOp":
        # Parse: $value : type($value) to $var : < type >
        value_operand = parser.parse_unresolved_operand()
        parser.parse_punctuation(":")
        value_type = parser.parse_type()
        parser.parse_keyword("to")
        var_operand = parser.parse_unresolved_operand()
        parser.parse_punctuation(":")

        # Custom format: '<' type(result) '>' like emitc.load
        if parser.parse_optional_punctuation("<") is not None:
            var_value_type = parser.parse_type()
            parser.parse_punctuation(">")
            var_type = EmitC_LValueType(var_value_type)
        else:
            # Generic fallback: parse var type directly
            var_type = parser.parse_type()

        attributes = parser.parse_optional_attr_dict()

        # Resolve operands - IMPORTANT: var comes first in MLIR definition!
        (value,) = parser.resolve_operands([value_operand], [value_type], parser.pos)
        (var,) = parser.resolve_operands([var_operand], [var_type], parser.pos)

        return cls.build(operands=[var, value], attributes=attributes)

    def print(self, printer: Printer) -> None:
        printer.print_string(" ")
        printer.print_operand(self.value)
        printer.print_string(" : ")
        printer.print_attribute(self.value.type)
        printer.print_string(" to ")
        printer.print_operand(self.var)
        printer.print_string(" : <")
        # Extract the value type from lvalue type
        if isinstance(self.var.type, EmitC_LValueType):
            printer.print_attribute(self.var.type.value_type)
        else:
            printer.print_attribute(self.var.type)
        printer.print_string(">")
        printer.print_op_attributes(self.attributes)


@irdl_op_definition
class EmitC_BitwiseAndOp(IRDLOperation):
    """Bitwise and operation"""

    name = "emitc.bitwise_and"

    assembly_format = """operands attr-dict `:` functional-type(operands, results)"""

    lhs = operand_def(AnyAttr())

    rhs = operand_def(AnyAttr())

    v2 = result_def(AnyAttr())


@irdl_op_definition
class EmitC_BitwiseLeftShiftOp(IRDLOperation):
    """Bitwise left shift operation"""

    name = "emitc.bitwise_left_shift"

    assembly_format = """operands attr-dict `:` functional-type(operands, results)"""

    lhs = operand_def(AnyAttr())

    rhs = operand_def(AnyAttr())

    v3 = result_def(AnyAttr())


@irdl_op_definition
class EmitC_BitwiseNotOp(IRDLOperation):
    """Bitwise not operation"""

    name = "emitc.bitwise_not"

    assembly_format = """operands attr-dict `:` functional-type(operands, results)"""

    v4 = operand_def(AnyAttr())

    v5 = result_def(AnyAttr())


@irdl_op_definition
class EmitC_BitwiseOrOp(IRDLOperation):
    """Bitwise or operation"""

    name = "emitc.bitwise_or"

    assembly_format = """operands attr-dict `:` functional-type(operands, results)"""

    lhs = operand_def(AnyAttr())

    rhs = operand_def(AnyAttr())

    v6 = result_def(AnyAttr())


@irdl_op_definition
class EmitC_BitwiseRightShiftOp(IRDLOperation):
    """Bitwise right shift operation"""

    name = "emitc.bitwise_right_shift"

    assembly_format = """operands attr-dict `:` functional-type(operands, results)"""

    lhs = operand_def(AnyAttr())

    rhs = operand_def(AnyAttr())

    v7 = result_def(AnyAttr())


@irdl_op_definition
class EmitC_BitwiseXorOp(IRDLOperation):
    """Bitwise xor operation"""

    name = "emitc.bitwise_xor"

    assembly_format = """operands attr-dict `:` functional-type(operands, results)"""

    lhs = operand_def(AnyAttr())

    rhs = operand_def(AnyAttr())

    v8 = result_def(AnyAttr())


@irdl_op_definition
class EmitC_CallOp(IRDLOperation):
    """Call operation"""

    name = "emitc.call"

    # assembly_format = """
    #     $callee `(` $operands `)` attr-dict `:` functional-type($operands, results)
    #   """

    callee = prop_def(AnyAttr())

    call_operands = var_operand_def(AnyAttr())

    arg_attrs = opt_prop_def(AnyAttr())

    res_attrs = opt_prop_def(AnyAttr())

    v9 = var_result_def(AnyAttr())


@irdl_op_definition
class EmitC_CallOpaqueOp(IRDLOperation):
    """
    The `emitc.call_opaque` operation represents a C++ function call. The callee can be an arbitrary non-empty string.
    The call allows specifying order of operands and attributes in the call as follows:

        - integer value of index type refers to an operand;
        - attribute which will get lowered to constant value in call;
    """

    name = "emitc.call_opaque"

    callee = prop_def(StringAttr)
    args = opt_prop_def(ArrayAttr)
    template_args = opt_prop_def(ArrayAttr)
    # The SSA‐value operands of the call
    call_args = var_operand_def()
    res = var_result_def()

    irdl_options = [ParsePropInAttrDict()]
    assembly_format = (
        "$callee `(` $call_args `)` attr-dict `:` functional-type(operands, results)"
    )

    def __init__(
        self,
        callee: StringAttr | str,
        call_args: Sequence[SSAValue],
        result_types: Sequence[Attribute],
        args: ArrayAttr[Attribute] | None = None,
        template_args: ArrayAttr[Attribute] | None = None,
        attributes: dict[str, Attribute] | None = None,
    ):
        if isinstance(callee, str):
            callee = StringAttr(callee)
        super().__init__(
            properties={
                "callee": callee,
                "args": args,
                "template_args": template_args,
            },
            operands=[call_args],
            result_types=[result_types],
            attributes=attributes,
        )

    def verify_(self) -> None:
        if not self.callee.data:
            raise VerifyException("callee must not be empty")

        if self.args is not None:
            for arg in self.args.data:
                if isa(arg, IntegerAttr[IndexType]):
                    index = arg.value.data
                    if not (0 <= index < len(self.call_args)):
                        raise VerifyException("index argument is out of range")
                elif isinstance(arg, ArrayAttr):
                    # see https://github.com/llvm/llvm-project/blob/2eb733b5a6ab17a3ae812bb55c1c7c64569cadcd/mlir/lib/Dialect/EmitC/IR/EmitC.cpp#L342
                    # This part is referenced as a FIXME there.
                    raise VerifyException("array argument has no type")

        if self.template_args is not None:
            for t_arg in self.template_args.data:
                if not isa(
                    t_arg,
                    TypeAttribute | IntegerAttr | FloatAttr | EmitC_OpaqueAttr,
                ):
                    raise VerifyException("template argument has invalid type")

        for res_type in self.res.types:
            if isinstance(res_type, EmitC_ArrayType):
                raise VerifyException("cannot return array type")


@irdl_op_definition
class EmitC_CastOp(IRDLOperation):
    """Cast operation"""

    name = "emitc.cast"

    assembly_format = """$source attr-dict `:` type($source) `to` type($dest)"""

    source = operand_def(AnyAttr())

    dest = result_def(AnyAttr())


class CmpPredicate(StrEnum):
    """Comparison predicate for emitc.cmp operation"""

    eq = "eq"
    ne = "ne"
    lt = "lt"
    le = "le"
    gt = "gt"
    ge = "ge"
    three_way = "three_way"


CMP_PREDICATE_MAP = {
    CmpPredicate.eq: 0,
    CmpPredicate.ne: 1,
    CmpPredicate.lt: 2,
    CmpPredicate.le: 3,
    CmpPredicate.gt: 4,
    CmpPredicate.ge: 5,
    CmpPredicate.three_way: 6,
}

CMP_PREDICATE_REVERSE_MAP = {i: p for p, i in CMP_PREDICATE_MAP.items()}


@irdl_op_definition
class EmitC_CmpOp(EmitC_BinaryOperation):
    """
    Comparison operation.

    With the `emitc.cmp` operation the comparison operators ==, !=, <, <=, >, >=, <=>
    can be applied.

    Example:

    ```mlir
    // Custom form of the cmp operation.
    %0 = emitc.cmp eq, %arg0, %arg1 : (i32, i32) -> i1
    %1 = emitc.cmp lt, %arg2, %arg3 :
        (
          !emitc.opaque<"std::valarray<float>">,
          !emitc.opaque<"std::valarray<float>">
        ) -> !emitc.opaque<"std::valarray<bool>">
    ```
    ```c++
    // Code emitted for the operations above.
    bool v5 = v1 == v2;
    std::valarray<bool> v6 = v3 < v4;
    ```
    """

    name = "emitc.cmp"

    predicate = prop_def(IntegerAttr)

    # Override the assembly_format from parent to use custom parsing
    assembly_format = None

    def __init__(
        self,
        predicate: CmpPredicate | str | int,
        lhs: SSAValue,
        rhs: SSAValue,
        result_type: Attribute,
    ):
        if isinstance(predicate, CmpPredicate):
            predicate_value = CMP_PREDICATE_MAP[predicate]
        elif isinstance(predicate, str):
            predicate_value = CMP_PREDICATE_MAP[CmpPredicate(predicate)]
        else:
            predicate_value = predicate

        super().__init__(lhs, rhs, result_type)
        self.properties["predicate"] = IntegerAttr.from_int_and_width(
            predicate_value, 64
        )

    @classmethod
    def parse(cls, parser: Parser):
        predicate_str = parser.parse_identifier()
        parser.parse_punctuation(",")
        operand1 = parser.parse_unresolved_operand()
        parser.parse_punctuation(",")
        operand2 = parser.parse_unresolved_operand()
        parser.parse_punctuation(":")
        parser.parse_punctuation("(")
        operand1_type = parser.parse_type()
        parser.parse_punctuation(",")
        operand2_type = parser.parse_type()
        parser.parse_punctuation(")")
        parser.parse_punctuation("->")
        result_type = parser.parse_type()

        (operand1, operand2) = parser.resolve_operands(
            [operand1, operand2], [operand1_type, operand2_type], parser.pos
        )

        return cls(predicate_str, operand1, operand2, result_type)

    def print(self, printer: Printer):
        printer.print_string(" ")
        printer.print_string(CMP_PREDICATE_REVERSE_MAP[self.predicate.value.data])
        printer.print_string(", ")
        printer.print_operand(self.lhs)
        printer.print_string(", ")
        printer.print_operand(self.rhs)
        printer.print_string(" : (")
        printer.print_attribute(self.lhs.type)
        printer.print_string(", ")
        printer.print_attribute(self.rhs.type)
        printer.print_string(") -> ")
        printer.print_attribute(self.result.type)


@irdl_op_definition
class EmitC_ConditionalOp(IRDLOperation):
    """Conditional (ternary) operation"""

    name = "emitc.conditional"

    # assembly_format = """operands attr-dict `:` type($result)"""

    condition = operand_def(EqAttrConstraint(IntegerType(1)))

    true_value = operand_def(AnyAttr())

    false_value = operand_def(AnyAttr())

    result = result_def(AnyAttr())


@irdl_op_definition
class EmitC_ConstantOp(IRDLOperation):
    """Constant operation"""

    name = "emitc.constant"

    value = prop_def(AnyAttr())

    result = result_def(AnyAttr())


@irdl_op_definition
class EmitC_DeclareFuncOp(IRDLOperation):
    """An operation to declare a function"""

    name = "emitc.declare_func"

    assembly_format = """
        $sym_name attr-dict
      """

    sym_name = prop_def(AnyAttr())


@irdl_op_definition
class EmitC_DivOp(IRDLOperation):
    """Division operation"""

    name = "emitc.div"

    assembly_format = """operands attr-dict `:` functional-type(operands, results)"""

    v13 = operand_def(AnyAttr())

    v14 = operand_def(AnyAttr())

    v15 = result_def(AnyAttr())


@irdl_op_definition
class EmitC_ExpressionOp(IRDLOperation):
    """
    Expression operation.

    The `emitc.expression` operation returns a single SSA value which is yielded by
    its single-basic-block region. The operation can take variadic operands as defs.
    As the operation is to be emitted as a C expression, the operations within
    its body must form a single Def-Use tree of emitc ops whose result is
    yielded by a terminating `emitc.yield`.

    When specified, the optional `do_not_inline` indicates that the expression is
    to be emitted as the rhs of an EmitC SSA value definition. Otherwise, the
    expression may be emitted inline, i.e. directly at its use.
    """

    name = "emitc.expression"

    # Use custom parse/print to precisely match MLIR syntax expectations.
    assembly_format = None

    defs = var_operand_def(AnyAttr())
    do_not_inline = opt_prop_def(EqAttrConstraint(UnitAttr()))
    result = result_def(EmitCTypeConstr)
    region = region_def("single_block")

    def __init__(
        self,
        defs: Sequence[SSAValue],
        result_type: Attribute,
        region: Region,
        do_not_inline: UnitAttr | None = None,
        attributes: dict[str, Attribute] | None = None,
    ):
        properties: dict[str, Attribute] = {}
        if do_not_inline is not None:
            properties["do_not_inline"] = do_not_inline

        super().__init__(
            operands=[defs],
            result_types=[result_type],
            regions=[region],
            properties=properties,
            attributes=attributes,
        )

        # Ensure the region has block arguments corresponding to operands.
        # If a block exists and has no args yet, insert args matching defs.
        if region.blocks:
            block = region.blocks.first
            assert block is not None
            if len(block.args) == 0 and defs:
                for idx, operand in enumerate(defs):
                    block.insert_arg(operand.type, idx)

    def create_body(self) -> Block:
        """
        Create a body block for the expression with block arguments corresponding to operands.
        Similar to MLIR's createBody() method.
        """
        assert not self.region.blocks, "expression already has a body"
        # Create a block with argument types matching the operands
        arg_types = [operand.type for operand in self.defs]
        block = Block(arg_types=arg_types)
        self.region.add_block(block)
        return block

    def verify_(self) -> None:
        """
        Verify the ExpressionOp.
        Based on MLIR implementation: https://github.com/llvm/llvm-project/blob/main/mlir/lib/Dialect/EmitC/IR/EmitC.cpp
        """
        # Verify operand types are EmitC types or lvalues
        for operand in self.defs:
            operand_type = operand.type
            if isinstance(operand_type, EmitC_LValueType):
                continue
            try:
                EmitCTypeConstr.verify(operand_type, ConstraintContext())
            except VerifyException:
                raise VerifyException(
                    f"operand type {operand_type} is not a valid EmitC type"
                )

        region = self.region

        if not region.blocks:
            raise VerifyException(
                "expression must have a region with at least one block"
            )

        # Single-block region is already enforced by the region def, use first.
        body = region.blocks.first
        assert body is not None

        # If block arguments are present, they must match operands in count and type.
        # Otherwise, zero block arguments is allowed (region can capture defs directly).
        if len(body.args) not in (0, len(self.defs)):
            raise VerifyException(
                "number of block arguments must be zero or match number of operands"
            )
        if len(body.args) == len(self.defs):
            for arg, operand in zip(body.args, self.defs):
                if arg.type != operand.type:
                    raise VerifyException(
                        "block argument types must match operand types"
                    )

        ops_list = list(body.ops)
        if not ops_list:
            raise VerifyException("must yield a value at termination")

        terminator = ops_list[-1]
        if not isinstance(terminator, EmitC_YieldOp):
            raise VerifyException("must yield a value at termination")

        yield_result = terminator.result
        if yield_result is None:
            raise VerifyException("must yield a value at termination")

        # Check that yielded value is defined within the expression
        # For now, allow block arguments (they come from operands)
        # TODO: More precise verification of block arguments vs operations
        root_op = yield_result.owner
        if root_op is not None:
            # Value is defined by an operation - check it's within the expression
            if root_op.parent_op() != self:
                raise VerifyException("yielded value not defined within expression")

        # Check type compatibility
        if self.result.type != yield_result.type:
            raise VerifyException("requires yielded type to match return type")

        # Check that all operations in the body (except terminator) have exactly one result and one use
        for op in ops_list[:-1]:  # Exclude terminator
            # Ensure all ops are from the EmitC dialect (match MLIR constraint)
            if not op.name.startswith("emitc."):
                raise VerifyException(
                    "expression body must consist of emitc.* operations"
                )
            if len(op.results) != 1:
                raise VerifyException("requires exactly one result for each operation")

            result = op.results[0]
            uses_count = sum(1 for _ in result.uses)
            if uses_count != 1:
                raise VerifyException("requires exactly one use for each operation")

    @classmethod
    def parse(cls, parser: Parser):
        # Parse optional defs (operands) separated by commas
        unresolved_defs: list[Parser.UnresolvedOperand] = []
        first_operand = parser.parse_optional_unresolved_operand()
        if first_operand is not None:
            unresolved_defs.append(first_operand)
            while parser.parse_optional_punctuation(",") is not None:
                unresolved_defs.append(parser.parse_unresolved_operand())

        # Optional 'noinline'
        do_not_inline = None
        if parser.parse_optional_keyword("noinline") is not None:
            do_not_inline = UnitAttr()

        # Optional attributes dict
        attributes = parser.parse_optional_attr_dict()

        # Parse function type: (input_types) -> result_type
        parser.parse_punctuation(":")
        parser.parse_punctuation("(")
        input_types: list[Attribute] = []
        if parser.parse_optional_punctuation(")") is None:
            # At least one type
            input_types.append(parser.parse_type())
            while parser.parse_optional_punctuation(",") is not None:
                input_types.append(parser.parse_type())
            parser.parse_punctuation(")")
        parser.parse_punctuation("->")
        result_type = parser.parse_type()

        # Resolve operands against parsed input types
        defs: Sequence[SSAValue] = []
        if unresolved_defs:
            defs = parser.resolve_operands(unresolved_defs, input_types, parser.pos)

        # Parse region as provided (allow labeled entry block, as in tests)
        region = parser.parse_region()

        return cls(defs, result_type, region, do_not_inline, attributes)

    def print(self, printer: Printer) -> None:
        # Print operands (defs)
        if len(self.defs) != 0:
            printer.print_string(" ")
            printer.print_list(self.defs, printer.print_operand)

        # Optional noinline keyword
        if self.do_not_inline is not None:
            printer.print_string(" noinline")

        # Attributes dict (if any)
        printer.print_op_attributes(self.attributes)

        # Function type
        printer.print_string(" : ")
        printer.print_function_type(self.operand_types, self.result_types)

        # Print region body without caret-labeled block header and without args.
        # Remap block arguments to the corresponding operand names for printing.
        printer.print_string(" ")
        body_region = self.region
        with printer.in_braces():
            entry = body_region.blocks.first
            if entry is None:
                printer._print_new_line()
                return
            # Prepare SSA name remapping in a fresh scope
            printer._print_new_line()
            printer.enter_scope()
            try:
                # Allocate or reuse names for defs without emitting text
                def get_or_allocate_name(val: SSAValue) -> str:
                    if val in printer._ssa_values:
                        return printer._ssa_values[val]
                    if val.name_hint:
                        curr_ind = printer.ssa_names.get(val.name_hint, 0)
                        suffix = f"_{curr_ind}" if curr_ind != 0 else ""
                        name = f"{val.name_hint}{suffix}"
                        printer._ssa_values[val] = name
                        printer.ssa_names[val.name_hint] = curr_ind + 1
                        return name
                    name = printer._get_new_valid_name_id()
                    printer._ssa_values[val] = name
                    return name

                # Map block args to defs' names
                for arg, operand in zip(entry.args, self.defs):
                    name = get_or_allocate_name(operand)
                    printer._ssa_values[arg] = name

                # Print inner ops
                with printer.indented():
                    for op in entry.ops:
                        printer._print_new_line()
                        printer.print_op(op)
            finally:
                printer.exit_scope()


@irdl_op_definition
class EmitC_FileOp(IRDLOperation):
    """A file container operation"""

    name = "emitc.file"

    assembly_format = """$id attr-dict-with-keyword $bodyRegion"""

    id = prop_def(AnyAttr())

    bodyRegion = region_def("single_block")


@irdl_op_definition
class EmitC_ForOp(IRDLOperation):
    """For operation"""

    name = "emitc.for"

    lowerBound = operand_def(AnyAttr())

    upperBound = operand_def(AnyAttr())

    step = operand_def(AnyAttr())

    region = region_def("single_block")


@irdl_op_definition
class EmitC_FuncOp(IRDLOperation):
    """An operation with a name containing a single `SSACFG` region"""

    name = "emitc.func"

    sym_name = prop_def(StringAttr)

    function_type = prop_def(AnyAttr())

    specifiers = opt_prop_def(AnyAttr())

    arg_attrs = opt_prop_def(AnyAttr())

    res_attrs = opt_prop_def(AnyAttr())

    body = region_def()


@irdl_op_definition
class EmitC_GetGlobalOp(IRDLOperation):
    """Obtain access to a global variable"""

    name = "emitc.get_global"

    name_attr = prop_def(FlatSymbolRefAttrConstr, prop_name="name")

    assembly_format = "$name `:` type($result) attr-dict"

    result = result_def(AnyOf((BaseAttr(EmitC_ArrayType), BaseAttr(EmitC_LValueType))))


@irdl_op_definition
class EmitC_GlobalOp(IRDLOperation):
    """A global variable"""

    name = "emitc.global"

    sym_name = prop_def(SymbolNameConstraint())

    type = prop_def(TypeAttribute)

    initial_value = opt_prop_def(AnyAttr())

    extern_specifier = opt_prop_def(EqAttrConstraint(UnitAttr()))

    static_specifier = opt_prop_def(EqAttrConstraint(UnitAttr()))

    const_specifier = opt_prop_def(EqAttrConstraint(UnitAttr()))

    @classmethod
    def parse(cls, parser: Parser) -> "EmitC_GlobalOp":
        specifiers = {}

        # Parse optional specifiers
        while True:
            if parser.parse_optional_keyword("extern"):
                specifiers["extern_specifier"] = UnitAttr()
            elif parser.parse_optional_keyword("static"):
                specifiers["static_specifier"] = UnitAttr()
            elif parser.parse_optional_keyword("const"):
                specifiers["const_specifier"] = UnitAttr()
            else:
                break

        # Parse symbol name
        sym_name_attr = parser.parse_symbol_name()

        # Parse type
        parser.parse_punctuation(":")
        type_attr = parser.parse_attribute()
        if not isinstance(type_attr, TypeAttribute):
            parser.raise_error("Expected type attribute")

        # Parse optional initial value
        initial_value: Attribute | None = None
        if parser.parse_optional_punctuation("="):
            if isinstance(type_attr, IntegerType):
                # Parse integer and create IntegerAttr with matching type
                value = parser.parse_integer()
                initial_value = IntegerAttr(value, type_attr)
            elif isinstance(type_attr, Float32Type):
                # Parse float and create FloatAttr with f32 type
                value = parser.parse_float()
                initial_value = FloatAttr(value, Float32Type())
            elif isinstance(type_attr, Float64Type):
                # Parse float and create FloatAttr with f64 type
                value = parser.parse_float()
                initial_value = FloatAttr(value, Float64Type())
            elif isinstance(type_attr, EmitC_ArrayType):
                # For EmitC array types, check if we have a dense attribute
                if parser.parse_optional_keyword("dense"):
                    # We've consumed 'dense', now parse as a dense attribute with array type context
                    tensor_type = TensorType(
                        type_attr.element_type, type_attr.get_shape()
                    )
                    initial_value = parser.parse_dense_int_or_fp_elements_attr(
                        tensor_type
                    )
                else:
                    # For other array initializers, parse as generic attribute
                    initial_value = parser.parse_attribute()
            else:
                # For other types, parse as generic attribute
                initial_value = parser.parse_attribute()

        return cls.build(
            properties={
                "sym_name": sym_name_attr,
                "type": type_attr,
                "initial_value": initial_value,
                **specifiers,
            }
        )

    def print(self, printer: Printer) -> None:
        # Print specifiers with proper spacing
        if self.extern_specifier is not None:
            printer.print_string(" extern")
        if self.static_specifier is not None:
            printer.print_string(" static")
        if self.const_specifier is not None:
            printer.print_string(" const")

        printer.print_string(" ")
        printer.print_symbol_name(self.sym_name.data)
        printer.print_string(" : ")
        printer.print_attribute(self.type)

        if self.initial_value is not None:
            printer.print_string(" = ")
            # Special handling for constants to avoid type annotation
            if isinstance(self.initial_value, IntegerAttr):
                printer.print_string(str(self.initial_value.value.data))
            elif isinstance(self.initial_value, FloatAttr):
                printer.print_string(str(float(self.initial_value.value.data)))
            elif isinstance(
                self.initial_value, DenseIntOrFPElementsAttr
            ) and isinstance(self.type, EmitC_ArrayType):
                # For dense attributes with array types, print without tensor type annotation
                # to match MLIR syntax expectations for emitc.global
                self.initial_value.print_without_type(printer)
            else:
                printer.print_attribute(self.initial_value)

    def verify_(self) -> None:
        # Check for invalid specifier combinations
        if self.extern_specifier is not None and self.static_specifier is not None:
            raise VerifyException("cannot have both static and extern specifiers")


@irdl_op_definition
class EmitC_IfOp(IRDLOperation):
    """If-then-else operation"""

    name = "emitc.if"

    condition = operand_def(EqAttrConstraint(IntegerType(1)))

    thenRegion = region_def("single_block")

    elseRegion = region_def()


@irdl_op_definition
class EmitC_IncludeOp(IRDLOperation):
    """Include operation"""

    name = "emitc.include"

    include = prop_def(AnyAttr())

    is_standard_include = prop_def(EqAttrConstraint(UnitAttr()))


@irdl_op_definition
class EmitC_LiteralOp(IRDLOperation):
    """Literal operation"""

    name = "emitc.literal"

    assembly_format = """$value attr-dict `:` type($result)"""

    value = prop_def(BaseAttr(StringAttr))

    result = result_def(AnyAttr())


@irdl_op_definition
class EmitC_LoadOp(IRDLOperation):
    """Load an lvalue into an SSA value."""

    name = "emitc.load"

    operand = operand_def(AnyAttr())

    result = result_def(AnyAttr())

    # Use custom parse/print to match MLIR custom format exactly.
    # MLIR format prints the result type inside angle brackets and omits the
    # lvalue wrapper from the operand type, e.g.: `load %arg : <i32>`.
    # Keep accepting the generic form as well for compatibility.
    assembly_format = None

    @classmethod
    def parse(cls, parser: Parser) -> "EmitC_LoadOp":
        # Parse operand
        unresolved_operand = parser.parse_unresolved_operand()

        # Optional attributes
        attributes = parser.parse_optional_attr_dict()

        # Parse ':'
        parser.parse_punctuation(":")

        # Custom form: '<' type(result) '>'
        if parser.parse_optional_punctuation("<") is not None:
            result_type = parser.parse_type()
            parser.parse_punctuation(">")
            # Operand must be an lvalue of result_type
            operand_type: Attribute = EmitC_LValueType(result_type)
            (operand,) = parser.resolve_operands(
                [unresolved_operand], [operand_type], parser.pos
            )
            return cls.build(
                operands=[operand], result_types=[result_type], attributes=attributes
            )

        # Generic fallback: parse operand type directly and derive result type.
        operand_type = parser.parse_type()
        if not isinstance(operand_type, EmitC_LValueType):
            parser.raise_error(
                "expected !emitc.lvalue<...> operand type for emitc.load"
            )
        result_type = operand_type.value_type
        (operand,) = parser.resolve_operands(
            [unresolved_operand], [operand_type], parser.pos
        )
        return cls.build(
            operands=[operand], result_types=[result_type], attributes=attributes
        )

    def print(self, printer: Printer) -> None:
        # Print operand
        printer.print_string(" ")
        printer.print_operand(self.operand)

        # Attributes dict (if any)
        printer.print_op_attributes(self.attributes)

        # Print custom result type in angle brackets
        printer.print_string(" : <")
        printer.print_attribute(self.result.type)
        printer.print_string(">")

    def verify_(self) -> None:
        # Operand must be an lvalue, and the result type must match the lvalue's element type.
        op_type = self.operand.type
        if not isinstance(op_type, EmitC_LValueType):
            raise VerifyException("emitc.load operand must be an !emitc.lvalue<...>")
        if self.result.type != op_type.value_type:
            raise VerifyException(
                "emitc.load requires result type to match lvalue element type"
            )


@irdl_op_definition
class EmitC_LogicalAndOp(IRDLOperation):
    """Logical and operation"""

    name = "emitc.logical_and"

    assembly_format = """operands attr-dict `:` type(operands)"""

    lhs = operand_def(AnyAttr())

    rhs = operand_def(AnyAttr())

    v16 = result_def(EqAttrConstraint(IntegerType(1)))


@irdl_op_definition
class EmitC_LogicalNotOp(IRDLOperation):
    """Logical not operation"""

    name = "emitc.logical_not"

    assembly_format = """operands attr-dict `:` type(operands)"""

    v17 = operand_def(AnyAttr())

    v18 = result_def(EqAttrConstraint(IntegerType(1)))


@irdl_op_definition
class EmitC_LogicalOrOp(IRDLOperation):
    """Logical or operation"""

    name = "emitc.logical_or"

    assembly_format = """operands attr-dict `:` type(operands)"""

    lhs = operand_def(AnyAttr())

    rhs = operand_def(AnyAttr())

    v19 = result_def(EqAttrConstraint(IntegerType(1)))


@irdl_op_definition
class EmitC_MemberOfPtrOp(IRDLOperation):
    """Member of pointer operation"""

    name = "emitc.member_of_ptr"

    member = prop_def(AnyAttr())

    operand = operand_def(AnyAttr())

    v20 = result_def(AnyAttr())


@irdl_op_definition
class EmitC_MemberOp(IRDLOperation):
    """Member operation"""

    name = "emitc.member"

    member = prop_def(AnyAttr())

    operand = operand_def(AnyAttr())

    v21 = result_def(AnyAttr())


@irdl_op_definition
class EmitC_MulOp(IRDLOperation):
    """Multiplication operation"""

    name = "emitc.mul"

    assembly_format = """operands attr-dict `:` functional-type(operands, results)"""

    v22 = operand_def(AnyAttr())

    v23 = operand_def(AnyAttr())

    v24 = result_def(AnyAttr())


@irdl_op_definition
class EmitC_RemOp(IRDLOperation):
    """Remainder operation"""

    name = "emitc.rem"

    assembly_format = """operands attr-dict `:` functional-type(operands, results)"""

    v25 = operand_def(AnyAttr())

    v26 = operand_def(AnyAttr())

    v27 = result_def(AnyAttr())


@irdl_op_definition
class EmitC_ReturnOp(IRDLOperation):
    """Function return operation"""

    name = "emitc.return"

    assembly_format = """attr-dict ($operand^ `:` type($operand))?"""

    operand = opt_operand_def(AnyAttr())


@irdl_op_definition
class EmitC_SubOp(IRDLOperation):
    """Subtraction operation"""

    name = "emitc.sub"

    assembly_format = """operands attr-dict `:` functional-type(operands, results)"""

    lhs = operand_def(AnyAttr())

    rhs = operand_def(AnyAttr())

    v28 = result_def(AnyAttr())


@irdl_op_definition
class EmitC_SubscriptOp(IRDLOperation):
    """Subscript operation"""

    name = "emitc.subscript"

    assembly_format = (
        """$value `[` $indices `]` attr-dict `:` functional-type(operands, results)"""
    )

    value = operand_def(AnyAttr())

    indices = var_operand_def(AnyAttr())

    result = result_def(BaseAttr(EmitC_LValueType))


@irdl_op_definition
class EmitC_SwitchOp(IRDLOperation):
    """Switch operation"""

    name = "emitc.switch"

    arg = operand_def(AnyAttr())

    cases = prop_def(AnyAttr())

    defaultRegion = region_def("single_block")

    caseRegions = var_region_def()


@irdl_op_definition
class EmitC_UnaryMinusOp(IRDLOperation):
    """Unary minus operation"""

    name = "emitc.unary_minus"

    assembly_format = """operands attr-dict `:` functional-type(operands, results)"""

    v29 = operand_def(AnyAttr())

    v30 = result_def(AnyAttr())


@irdl_op_definition
class EmitC_UnaryPlusOp(IRDLOperation):
    """Unary plus operation"""

    name = "emitc.unary_plus"

    assembly_format = """operands attr-dict `:` functional-type(operands, results)"""

    v31 = operand_def(AnyAttr())

    v32 = result_def(AnyAttr())


@irdl_op_definition
class EmitC_VariableOp(IRDLOperation):
    """Variable operation"""

    name = "emitc.variable"

    value = prop_def(AnyAttr())
    result = result_def(AnyOf((BaseAttr(EmitC_ArrayType), BaseAttr(EmitC_LValueType))))


@irdl_op_definition
class EmitC_VerbatimOp(IRDLOperation):
    """Verbatim operation"""

    name = "emitc.verbatim"

    assembly_format = """$value (`args` $fmtArgs^ `:` type($fmtArgs))? attr-dict"""

    value = prop_def(BaseAttr(StringAttr))

    fmtArgs = var_operand_def(AnyAttr())


@irdl_op_definition
class EmitC_YieldOp(IRDLOperation):
    """Block termination operation"""

    name = "emitc.yield"

    assembly_format = """ attr-dict ($result^ `:` type($result))? """

    result = opt_operand_def(AnyAttr())
    traits = traits_def(IsTerminator())


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
        EmitC_OpaqueType,
        EmitC_PointerType,
        EmitC_PtrDiffT,
        EmitC_SignedSizeT,
        EmitC_SizeT,
        EmitC_OpaqueAttr,
    ],
)
