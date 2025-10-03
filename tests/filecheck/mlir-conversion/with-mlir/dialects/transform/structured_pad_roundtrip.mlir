// RUN: xdsl-opt %s | mlir-opt | filecheck %s

// CHECK: module attributes {transform.with_named_sequence}
builtin.module attributes {"transform.with_named_sequence"} {
  // CHECK: transform.named_sequence @__transform_main
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    // CHECK: transform.structured.match ops{["linalg.matmul"]}
    %0 = "transform.structured.match"(%arg0) <{ops = ["linalg.matmul"]}> : (!transform.any_op) -> !transform.any_op

    // Basic pad
    // CHECK: transform.structured.pad %{{.*}} {padding_dimensions = [0, 1, 2], padding_values = [0.000000e+00 : f32, 0.000000e+00 : f32, 0.000000e+00 : f32]}
    %padded1, %pad1, %copy1 = "transform.structured.pad"(%0) <{
      operandSegmentSizes = array<i32: 1, 0>,
      padding_values = [0.0 : f32, 0.0 : f32, 0.0 : f32],
      padding_dimensions = [0 : i64, 1 : i64, 2 : i64]
    }> : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // Pad with nofold_flags
    // CHECK: transform.structured.pad %{{.*}} {nofold_flags = [1, 1, 0], padding_dimensions = [0, 1, 2], padding_values = [0.000000e+00 : f32, 0.000000e+00 : f32, 0.000000e+00 : f32]}
    %padded2, %pad2, %copy2 = "transform.structured.pad"(%0) <{
      operandSegmentSizes = array<i32: 1, 0>,
      padding_values = [0.0 : f32, 0.0 : f32, 0.0 : f32],
      padding_dimensions = [0 : i64, 1 : i64, 2 : i64],
      nofold_flags = [1 : i64, 1 : i64, 0 : i64]
    }> : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // Pad with static_pad_to_multiple_of
    // CHECK: transform.structured.pad %{{.*}} pad_to_multiple_of [4, 8, 1] {padding_dimensions = [0, 1, 2], padding_values = [0.000000e+00 : f32, 0.000000e+00 : f32, 0.000000e+00 : f32]}
    %padded3, %pad3, %copy3 = "transform.structured.pad"(%0) <{
      operandSegmentSizes = array<i32: 1, 0>,
      padding_values = [0.0 : f32, 0.0 : f32, 0.0 : f32],
      padding_dimensions = [0 : i64, 1 : i64, 2 : i64],
      static_pad_to_multiple_of = array<i64: 4, 8, 1>
    }> : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // Pad with copy_back_op
    // CHECK: transform.structured.pad %{{.*}} {copy_back_op = "linalg.copy", padding_dimensions = [0, 1, 2], padding_values = [0.000000e+00 : f32, 0.000000e+00 : f32, 0.000000e+00 : f32]}
    %padded4, %pad4, %copy4 = "transform.structured.pad"(%0) <{
      operandSegmentSizes = array<i32: 1, 0>,
      padding_values = [0.0 : f32, 0.0 : f32, 0.0 : f32],
      padding_dimensions = [0 : i64, 1 : i64, 2 : i64],
      copy_back_op = "linalg.copy"
    }> : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // Pad with transpose_paddings
    // CHECK: transform.structured.pad %{{.*}} {padding_dimensions = [0, 1], padding_values = [0.000000e+00 : f32, 0.000000e+00 : f32], transpose_paddings = {{\[}}[1, 0]]}
    %padded5, %pad5, %copy5 = "transform.structured.pad"(%0) <{
      operandSegmentSizes = array<i32: 1, 0>,
      padding_values = [0.0 : f32, 0.0 : f32],
      padding_dimensions = [0 : i64, 1 : i64],
      transpose_paddings = [[1 : i64, 0 : i64]]
    }> : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // CHECK: transform.yield
    transform.yield
  }
}
