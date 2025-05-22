import pytest

from xdsl.dialects.builtin import MemRefType, StringAttr, i32
from xdsl.dialects.emitc import (
    EmitC_ArrayType,
    EmitC_LValueType,
    EmitC_OpaqueType,
    EmitC_PointerType,
)
from xdsl.utils.exceptions import VerifyException


def test_emitc_array_negative_dimension():
    with pytest.raises(
        VerifyException, match="EmitC array dimensions must have non-negative size"
    ):
        EmitC_ArrayType([-1], i32)


def test_emitc_lvalue_wraps_unsupported_type():
    with pytest.raises(VerifyException, match="must wrap supported emitc type"):
        EmitC_LValueType(MemRefType(element_type=i32, shape=[1]))


def test_emitc_lvalue_cannot_wrap_array():
    arr = EmitC_ArrayType([2], i32)
    with pytest.raises(VerifyException, match="cannot wrap !emitc.array type"):
        EmitC_LValueType(arr)


def test_emitc_opaque_type_empty():
    with pytest.raises(
        VerifyException, match="expected non empty string in !emitc.opaque type"
    ):
        EmitC_OpaqueType(StringAttr("")).verify()


def test_emitc_opaque_type_pointer():
    with pytest.raises(
        VerifyException,
        match="pointer not allowed as outer type with !emitc.opaque, use !emitc.ptr instead",
    ):
        EmitC_OpaqueType(StringAttr("foo*")).verify()


def test_emitc_pointer_type_to_lvalue():
    lval = EmitC_LValueType(i32)
    with pytest.raises(VerifyException, match="pointers to lvalues are not allowed"):
        EmitC_PointerType(lval).verify()
