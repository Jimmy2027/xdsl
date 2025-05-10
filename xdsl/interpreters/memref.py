from math import prod
from typing import Any, cast

import numpy as np

from xdsl.dialects import builtin, memref
from xdsl.interpreter import (
    Interpreter,
    InterpreterFunctions,
    PythonValues,
    impl,
    register_impls,
)
from xdsl.interpreters.builtin import xtype_for_el_type
from xdsl.interpreters.shaped_array import ShapedArray
from xdsl.interpreters.utils.ptr import TypedPtr
from xdsl.traits import SymbolTable
from xdsl.utils.hints import isa


@register_impls
class MemRefFunctions(InterpreterFunctions):
    @impl(memref.AllocOp)
    def run_alloc(
        self, interpreter: Interpreter, op: memref.AllocOp, args: PythonValues
    ) -> PythonValues:
        memref_type = op.memref.type

        shape = memref_type.get_shape()
        size = prod(shape)
        xtype = xtype_for_el_type(
            memref_type.get_element_type(), interpreter.index_bitwidth
        )

        shaped_array = ShapedArray(TypedPtr[Any].zeros(size, xtype=xtype), list(shape))
        return (shaped_array,)

    @impl(memref.DeallocOp)
    def run_dealloc(
        self, interpreter: Interpreter, op: memref.DeallocOp, args: PythonValues
    ) -> PythonValues:
        return ()

    @impl(memref.StoreOp)
    def run_store(
        self, interpreter: Interpreter, op: memref.StoreOp, args: PythonValues
    ) -> PythonValues:
        value, memref, *indices = args

        memref = cast(ShapedArray[Any], memref)

        indices = tuple(indices)
        memref.store(indices, value)

        return ()

    @impl(memref.LoadOp)
    def run_load(
        self, interpreter: Interpreter, op: memref.LoadOp, args: tuple[Any, ...]
    ):
        shaped_array, *indices = args

        shaped_array = cast(ShapedArray[Any], shaped_array)

        indices = tuple(indices)
        value = shaped_array.load(indices)

        return (value,)

    @impl(memref.GetGlobalOp)
    def run_get_global(
        self, interpreter: Interpreter, op: memref.GetGlobalOp, args: PythonValues
    ) -> PythonValues:
        mem = SymbolTable.lookup_symbol(op, op.name_)
        assert isinstance(mem, memref.GlobalOp)
        initial_value = mem.initial_value
        if not isa(initial_value, builtin.DenseIntOrFPElementsAttr):
            raise NotImplementedError(
                "MemRefs that are not dense int or float arrays are not implemented"
            )
        data = initial_value.get_values()
        shape = initial_value.get_shape()
        assert shape is not None
        xtype = xtype_for_el_type(
            initial_value.get_element_type(), interpreter.index_bitwidth
        )
        shaped_array = ShapedArray(TypedPtr[Any].new(data, xtype=xtype), list(shape))
        return (shaped_array,)

    @impl(memref.SubviewOp)
    def run_subview(
        self, interpreter: Interpreter, op: memref.SubviewOp, args: PythonValues
    ) -> PythonValues:
        source, *dynamic_indices = args

        source = cast(ShapedArray[Any], source)

        static_offsets = op.static_offsets.get_values()
        static_sizes = op.static_sizes.get_values()
        static_strides = op.static_strides.get_values()

        offsets = [
            dynamic_indices.pop(0)
            if offset == memref.SubviewOp.DYNAMIC_INDEX
            else offset
            for offset in static_offsets
        ]
        sizes = [
            dynamic_indices.pop(0) if size == memref.SubviewOp.DYNAMIC_INDEX else size
            for size in static_sizes
        ]
        strides = [
            dynamic_indices.pop(0)
            if stride == memref.SubviewOp.DYNAMIC_INDEX
            else stride
            for stride in static_strides
        ]

        # Build a tuple of slice objects per dimension.
        slice_tuple = tuple(
            slice(off, off + sz * st, st)
            for off, sz, st in zip(offsets, sizes, strides)
        )

        # Apply slicing.
        source_data: list[int] = source.data
        source_array = np.array(source_data).reshape(source.shape)
        res_array = source_array[slice_tuple]
        shaped_array = ShapedArray(
            TypedPtr[Any].new(
                res_array.flatten().tolist(),
                xtype=xtype_for_el_type(
                    source.element_type, interpreter.index_bitwidth
                ),
            ),
            list(res_array.shape),
        )
        return (shaped_array,)
