// RUN: xdsl-opt %s | mlir-opt | filecheck %s

// CHECK: module attributes {transform.with_named_sequence}
builtin.module attributes {"transform.with_named_sequence"} {
  // CHECK: transform.named_sequence @__transform_main
  transform.named_sequence @__transform_main(%arg0: !transform.any_value {transform.readonly}) {

    // Test promote_tensor without memory_space
    // CHECK: transform.structured.promote_tensor %{{.*}} : !transform.any_value
    %0 = "transform.structured.promote_tensor"(%arg0) : (!transform.any_value) -> !transform.any_value

    transform.yield
  }

  // CHECK: transform.named_sequence @test_with_memory_space
  transform.named_sequence @test_with_memory_space(%arg0: !transform.any_value {transform.readonly}) {

    // Test promote_tensor with integer memory_space
    // CHECK: transform.structured.promote_tensor to 1 : i64 %{{.*}} : !transform.any_value
    %0 = "transform.structured.promote_tensor"(%arg0) {memory_space = 1 : i64} : (!transform.any_value) -> !transform.any_value

    // Test promote_tensor with different memory_space attribute
    // CHECK: transform.structured.promote_tensor to 2 : i32 %{{.*}} : !transform.any_value
    %1 = "transform.structured.promote_tensor"(%arg0) {memory_space = 2 : i32} : (!transform.any_value) -> !transform.any_value

    transform.yield
  }
}
