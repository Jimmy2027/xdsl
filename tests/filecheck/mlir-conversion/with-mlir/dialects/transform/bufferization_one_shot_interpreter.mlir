// RUN: xdsl-opt %s --print-op-generic | mlir-opt --transform-interpreter -split-input-file | filecheck %s
// RUN: xdsl-opt %s --print-op-generic | filecheck %s --check-prefix=CHECK-XDSL

// Basic One-Shot Bufferize using default memcpy op.
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = "transform.structured.match"(%arg1) <{"ops" = ["func.func"]}> : (!transform.any_op) -> !transform.any_op
    %1 = "transform.bufferization.one_shot_bufferize"(%0) : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// Check that function_boundary_type_conversion attribute is printed generically.
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = "transform.structured.match"(%arg1) <{"ops" = ["func.func"]}> : (!transform.any_op) -> !transform.any_op
    %1 = "transform.bufferization.one_shot_bufferize"(%0) {function_boundary_type_conversion = 1 : i32} : (!transform.any_op) -> !transform.any_op
    // CHECK-XDSL: "transform.bufferization.one_shot_bufferize"(%{{.*}}) {function_boundary_type_conversion = 1 : i32}
    transform.yield
  }
}

// Also test another enumerant value.
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = "transform.structured.match"(%arg1) <{"ops" = ["func.func"]}> : (!transform.any_op) -> !transform.any_op
    %1 = "transform.bufferization.one_shot_bufferize"(%0) {function_boundary_type_conversion = 2 : i32} : (!transform.any_op) -> !transform.any_op
    // CHECK-XDSL: "transform.bufferization.one_shot_bufferize"(%{{.*}}) {function_boundary_type_conversion = 2 : i32}
    transform.yield
  }
}

// CHECK-LABEL: func @test_function(
//  CHECK-SAME:     %[[A:.*]]: tensor<?xf32>
func.func @test_function(%A : tensor<?xf32>, %v : vector<4xf32>) -> (tensor<?xf32>) {
  %c0 = arith.constant 0 : index
  // CHECK: bufferization.to_{{(buffer|memref)}} %[[A]]
  // CHECK: copy
  // CHECK: vector.transfer_write
  // CHECK: bufferization.to_tensor
  %0 = vector.transfer_write %v, %A[%c0] : vector<4xf32>, tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----

// Emit linalg.copy instead of memref.copy.
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = "transform.structured.match"(%arg1) <{"ops" = ["func.func"]}> : (!transform.any_op) -> !transform.any_op
    %1 = "transform.bufferization.one_shot_bufferize"(%0) <{memcpy_op = "linalg.copy"}> : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @test_function_memcpy(
//  CHECK-SAME:     %[[A:.*]]: tensor<?xf32>
func.func @test_function_memcpy(%A : tensor<?xf32>, %v : vector<4xf32>) -> (tensor<?xf32>) {
  %c0 = arith.constant 0 : index
  // CHECK: copy
  %0 = vector.transfer_write %v, %A[%c0] : vector<4xf32>, tensor<?xf32>
  return %0 : tensor<?xf32>
}
