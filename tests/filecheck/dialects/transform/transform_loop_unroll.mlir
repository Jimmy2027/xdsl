// RUN: XDSL_ROUNDTRIP

// Test basic loop.unroll with factor
%loop = "test.op"() : () -> !transform.op<"scf.for">
// CHECK: transform.loop.unroll %loop {factor = 4 : i64} : !transform.op<"scf.for">
transform.loop.unroll %loop {factor = 4 : i64} : !transform.op<"scf.for">

// Test loop.unroll with different factor
%loop2 = "test.op"() : () -> !transform.any_op
// CHECK: transform.loop.unroll %loop2 {factor = 8 : i64} : !transform.any_op
transform.loop.unroll %loop2 {factor = 8 : i64} : !transform.any_op

// Test loop.unroll with affine.for op type
%affine_loop = "test.op"() : () -> !transform.op<"affine.for">
// CHECK: transform.loop.unroll %affine_loop {factor = 16 : i64} : !transform.op<"affine.for">
transform.loop.unroll %affine_loop {factor = 16 : i64} : !transform.op<"affine.for">
