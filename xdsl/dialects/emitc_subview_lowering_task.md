# Task: Lowering memref.subview to emitc dialect

## Goal
Implement a helper function in the emitc dialect (in xdsl) that lowers a `memref.subview` operation to C code that performs a nested loop copy, as shown in the example below.

## Input Example (MLIR):
```
%3 = memref.subview %arg1[0, 0] [8, 4] [1, 1] : memref<16x8xi8> to memref<8x4xi8, strided<[8, 1]>>
```

## Output Example (C):
```
for (int i = 0; i < TILE_K; ++i)
    for (int j = 0; j < TILE_N; ++j)
        S_tile[i][j] = S[depth + i][col + j];
```

## Requirements
- The helper function should be implemented in the xdsl emitc dialect, following the correct xdsl syntax and conventions.
- The function should take the relevant parameters from the memref.subview op (source memref, offsets, sizes, strides, destination memref).
- The function should generate the equivalent nested loop copy as emitc dialect operations.
- The function should be reusable for different subview shapes and offsets.
- The function should be placed in the appropriate location in the xdsl emitc dialect codebase.
- Finish the test implementation in `/home/hendrik/src/xdsl/tests/test_emitc_subview_lowering.py`
- Finish the implementation of the helper function in `/home/hendrik/src/xdsl/xdsl/dialects/emitc.py`
- The memref subview takes as input a source memref. To convert that to emitc an unrealized conversion cast is needed to an emitc array type.


## References
- See `/home/hendrik/src/xdsl/xdsl/dialects/emitc.py` for dialect definitions and conventions.
- See `/home/hendrik/ssrc/S-SIT-MLIR/cxr_mlir/dialects/cxr_emitc.py` for usage patterns and integration points.
- Follow the conventions and style of the xdsl package.

## Acceptance Criteria
- The helper function is implemented and available in the emitc dialect.
- The function is tested or demonstrated to lower a memref.subview to the correct C code pattern via emitc ops.
- The code follows xdsl and emitc dialect conventions.
