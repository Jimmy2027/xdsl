from collections.abc import Iterable
from typing import cast

from xdsl.dialects.builtin import (
    ArrayAttr,
    IntAttr,
    IntegerType,
    StringAttr,
    SymbolNameAttr,
    UnitAttr,
)
from xdsl.ir import (
    Attribute,
    Dialect,
    ParametrizedAttribute,
    TypeAttribute,
)
from xdsl.irdl import (
    AnyAttr,
    AnyOf,
    BaseAttr,
    EqAttrConstraint,
    IRDLOperation,
    ParameterDef,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    opt_operand_def,
    opt_prop_def,
    prop_def,
    region_def,
    result_def,
    var_operand_def,
    var_region_def,
    var_result_def,
)
from xdsl.printer import Printer


@irdl_attr_definition
class EmitC_ArrayType(ParametrizedAttribute, TypeAttribute):
    """EmitC array type"""

    name = "emitc.array"

    shape: ParameterDef[ArrayAttr[IntAttr]]
    element_type: ParameterDef[Attribute]

    def __init__(
        self,
        element_type: Attribute,
        shape: ArrayAttr[IntAttr] | Iterable[int | IntAttr],
    ):
        s: ArrayAttr[IntAttr]
        if isinstance(shape, ArrayAttr):
            # Temporary cast until Pyright is fixed to not infer ArrayAttr[int] as a
            # possible value for shape
            s = cast(ArrayAttr[IntAttr], shape)
        else:
            s = ArrayAttr(
                [IntAttr(dim) if isinstance(dim, int) else dim for dim in shape]
            )
        super().__init__(
            (
                s,
                element_type,
            )
        )

    def get_shape(self) -> tuple[int, ...]:
        """Get the shape of the array type."""
        return tuple(i.data for i in self.shape.data)

    def print_parameters(self, printer: Printer) -> None:
        printer.print(
            "<",
            "x".join([*[str(e.data) for e in self.shape], *[str(self.element_type)]]),
        )
        printer.print(">")


@irdl_attr_definition
class EmitC_LValueType(ParametrizedAttribute, TypeAttribute):
    """EmitC lvalue type"""

    name = "emitc.lvalue"


@irdl_attr_definition
class EmitC_OpaqueType(ParametrizedAttribute, TypeAttribute):
    """EmitC opaque type"""

    name = "emitc.opaque"
    value: ParameterDef[StringAttr]


@irdl_attr_definition
class EmitC_PointerType(ParametrizedAttribute, TypeAttribute):
    """EmitC pointer type"""

    name = "emitc.ptr"
    cast_target_type: ParameterDef[Attribute]


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

    callee = prop_def(AnyAttr())

    operands = var_operand_def(AnyAttr())

    arg_attrs = opt_prop_def(AnyAttr())

    res_attrs = opt_prop_def(AnyAttr())

    v9 = var_result_def(AnyAttr())


@irdl_op_definition
class EmitC_CallOpaqueOp(IRDLOperation):
    """Opaque call operation"""

    name = "emitc.call_opaque"

    call_args = var_operand_def(AnyAttr())

    callee = prop_def(AnyAttr())

    args = prop_def(AnyAttr())

    template_args = prop_def(AnyAttr())

    v10 = var_result_def(AnyAttr())


@irdl_op_definition
class EmitC_CastOp(IRDLOperation):
    """Cast operation"""

    name = "emitc.cast"

    assembly_format = """$source attr-dict `:` type($source) `to` type($dest)"""

    source = operand_def(AnyAttr())

    dest = result_def(AnyAttr())


@irdl_op_definition
class EmitC_CmpOp(IRDLOperation):
    """Comparison operation"""

    name = "emitc.cmp"

    assembly_format = (
        """$predicate `,` operands attr-dict `:` functional-type(operands, results)"""
    )

    predicate = prop_def(AnyAttr())

    lhs = operand_def(AnyAttr())

    rhs = operand_def(AnyAttr())

    v11 = result_def(AnyAttr())


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

    value = prop_def(AnyOf([BaseAttr(EmitC_OpaqueAttr), AnyAttr()]))

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

    v13 = operand_def(AnyOf((AnyAttr(), AnyAttr())))

    v14 = operand_def(AnyOf((AnyAttr(), AnyAttr())))

    v15 = result_def(AnyOf((AnyAttr(), AnyAttr())))


@irdl_op_definition
class EmitC_ExpressionOp(IRDLOperation):
    """Expression operation"""

    name = "emitc.expression"

    assembly_format = (
        """attr-dict (`noinline` $do_not_inline^)? `:` type($result) $region"""
    )

    do_not_inline = prop_def(EqAttrConstraint(UnitAttr()))

    result = result_def(AnyAttr())

    region = region_def("single_block")


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

    sym_name = prop_def(BaseAttr(SymbolNameAttr))

    function_type = prop_def(AnyAttr())

    specifiers = opt_prop_def(AnyAttr())

    arg_attrs = opt_prop_def(AnyAttr())

    res_attrs = opt_prop_def(AnyAttr())

    body = region_def()


@irdl_op_definition
class EmitC_GetGlobalOp(IRDLOperation):
    """Obtain access to a global variable"""

    name = prop_def(AnyAttr())

    # assembly_format = """$name `:` type($result) attr-dict"""

    result = result_def(AnyOf((BaseAttr(EmitC_ArrayType), BaseAttr(EmitC_LValueType))))


@irdl_op_definition
class EmitC_GlobalOp(IRDLOperation):
    """A global variable"""

    name = "emitc.global"

    sym_name = prop_def(BaseAttr(SymbolNameAttr))

    type = prop_def(AnyAttr())

    initial_value = opt_prop_def(AnyOf([BaseAttr(EmitC_OpaqueAttr), AnyAttr()]))

    extern_specifier = prop_def(EqAttrConstraint(UnitAttr()))

    static_specifier = prop_def(EqAttrConstraint(UnitAttr()))

    const_specifier = prop_def(EqAttrConstraint(UnitAttr()))


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

    # assembly_format = """$operand attr-dict `:` type($operand)"""

    operand = prop_def(AnyAttr())

    result = result_def(AnyAttr())


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

    v22 = operand_def(AnyOf((AnyAttr(), AnyAttr())))

    v23 = operand_def(AnyOf((AnyAttr(), AnyAttr())))

    v24 = result_def(AnyOf((AnyAttr(), AnyAttr())))


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

    value = prop_def(AnyAttr())

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

    value = prop_def(AnyOf([BaseAttr(EmitC_OpaqueAttr), AnyAttr()]))


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


EmitC_Dialect = Dialect(
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
