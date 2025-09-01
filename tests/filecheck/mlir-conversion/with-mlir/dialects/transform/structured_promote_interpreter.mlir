// RUN: xdsl-opt %s --print-op-generic | mlir-opt --transform-interpreter -split-input-file | filecheck %s

// Basic promotion of all operands with default settings.
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = "transform.structured.match"(%arg1) <{"ops" = ["linalg.matmul"]}> : (!transform.any_op) -> !transform.any_op
    %1 = "transform.structured.promote"(%0) <{operands_to_promote = [0 : i64, 1 : i64, 2 : i64], use_full_tiles_by_default}> : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @promote_subview_matmul(
func.func @promote_subview_matmul(%arg0: memref<?x?xf32, strided<[?, 1], offset: ?>>,
                             %arg1: memref<?x?xf32, strided<[?, 1], offset: ?>>,
                             %arg2: memref<?x?xf32, strided<[?, 1], offset: ?>>) {
  %c2000 = arith.constant 2000 : index
  %c3000 = arith.constant 3000 : index
  %c4000 = arith.constant 4000 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = "memref.dim"(%arg0, %c0) : (memref<?x?xf32, strided<[?, 1], offset: ?>>, index) -> index
  %1 = "memref.dim"(%arg0, %c1) : (memref<?x?xf32, strided<[?, 1], offset: ?>>, index) -> index
  %2 = "memref.dim"(%arg1, %c1) : (memref<?x?xf32, strided<[?, 1], offset: ?>>, index) -> index
  scf.for %arg3 = %c0 to %0 step %c2000 {
    scf.for %arg4 = %c0 to %2 step %c3000 {
      scf.for %arg5 = %c0 to %1 step %c4000 {
        %3 = memref.subview %arg0[%arg3, %arg5][%c2000, %c4000][%c1, %c1] :
             memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
        %4 = memref.subview %arg1[%arg5, %arg4][%c4000, %c3000][%c1, %c1] :
             memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
        %5 = memref.subview %arg2[%arg3, %arg4][%c2000, %c3000][%c1, %c1] :
             memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
        // CHECK: memref.alloc()
        // CHECK: linalg.copy
        // CHECK: linalg.matmul
        linalg.matmul ins(%3, %4: memref<?x?xf32, strided<[?, ?], offset: ?>>,
                                  memref<?x?xf32, strided<[?, ?], offset: ?>>)
                     outs(%5: memref<?x?xf32, strided<[?, ?], offset: ?>>)
      }
    }
  }
  return
}

// -----

// Promotion using stack allocation (alloca) instead of heap allocation.
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = "transform.structured.match"(%arg1) <{"ops" = ["linalg.matmul"]}> : (!transform.any_op) -> !transform.any_op
    %1 = "transform.structured.promote"(%0) <{operands_to_promote = [0 : i64], use_alloca}> : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @promote_first_subview_matmul(
func.func @promote_first_subview_matmul(%arg0: memref<?x?xf32, strided<[?, 1], offset: ?>>,
                             %arg1: memref<?x?xf32, strided<[?, 1], offset: ?>>,
                             %arg2: memref<?x?xf32, strided<[?, 1], offset: ?>>) {
  %c2000 = arith.constant 2000 : index
  %c3000 = arith.constant 3000 : index
  %c4000 = arith.constant 4000 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = "memref.dim"(%arg0, %c0) : (memref<?x?xf32, strided<[?, 1], offset: ?>>, index) -> index
  %1 = "memref.dim"(%arg0, %c1) : (memref<?x?xf32, strided<[?, 1], offset: ?>>, index) -> index
  %2 = "memref.dim"(%arg1, %c1) : (memref<?x?xf32, strided<[?, 1], offset: ?>>, index) -> index
  scf.for %arg3 = %c0 to %0 step %c2000 {
    scf.for %arg4 = %c0 to %2 step %c3000 {
      scf.for %arg5 = %c0 to %1 step %c4000 {
        %3 = memref.subview %arg0[%arg3, %arg5][%c2000, %c4000][%c1, %c1] :
             memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
        %4 = memref.subview %arg1[%arg5, %arg4][%c4000, %c3000][%c1, %c1] :
             memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
        %5 = memref.subview %arg2[%arg3, %arg4][%c2000, %c3000][%c1, %c1] :
             memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, ?], offset: ?>>
        // CHECK: memref.alloc()
        // CHECK: linalg.copy
        // CHECK: linalg.matmul
        linalg.matmul ins(%3, %4: memref<?x?xf32, strided<[?, ?], offset: ?>>,
                                  memref<?x?xf32, strided<[?, ?], offset: ?>>)
                     outs(%5: memref<?x?xf32, strided<[?, ?], offset: ?>>)
      }
    }
  }
  return
}
