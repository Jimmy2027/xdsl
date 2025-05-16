from xdsl.context import Context
from xdsl.dialects import memref
from xdsl.dialects.builtin import (
    ModuleOp,
)
from xdsl.dialects.builtin import (
    i8 as i8_type,
)
from xdsl.dialects.emitc import (
    lower_subview_to_affine_loops,
)
from xdsl.ir import (
    Block,
    Region,
)
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.printer import Printer


class SubviewRewrite(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.SubviewOp, rewriter: PatternRewriter):
        insertion_block = Block()
        lower_subview_to_affine_loops(
            subview_op=op,
            insertion_block=insertion_block,
        )
        new_ops = [
            insertion_block.detach_op(block_op) for block_op in insertion_block.ops
        ]

        rewriter.insert_op_before_matched_op(new_ops)
        rewriter.erase_matched_op()


class ConvertSubview(ModulePass):
    """
    Pass to apply CxRMatmulToDriver pattern.
    """

    name = "subview_conversion"

    def apply(self, ctx, module):
        pattern = SubviewRewrite()
        PatternRewriteWalker(pattern).rewrite_module(module)


def test_convert_memref_subview_to_emitc():
    i8 = i8_type

    source_shape = [16, 8]

    static_offsets = [0, 0]
    static_sizes = [8, 4]
    static_strides = [1, 1]

    temp_block = Block()
    region = Region(temp_block)
    module = ModuleOp(region)
    ctx = Context(True)

    alloc_op = memref.AllocOp.get(return_type=i8, dynamic_sizes=[], shape=source_shape)
    temp_block.add_op(alloc_op)
    source_memref_ssa = alloc_op.results[0]

    subview_op = memref.SubviewOp.from_static_parameters(
        source_memref_ssa,
        alloc_op.memref.type,
        static_offsets,
        static_sizes,
        static_strides,
    )
    temp_block.add_op(subview_op)

    ConvertSubview().apply(
        ctx=ctx,
        module=module,
    )

    print("\n")
    Printer().print_op(module)
    module.verify()


test_convert_memref_subview_to_emitc()
