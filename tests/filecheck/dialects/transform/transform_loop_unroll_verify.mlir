// RUN: xdsl-opt %s --verify-diagnostics --split-input-file | filecheck %s

%loop = "test.op"() : () -> !transform.op<"scf.for">

// CHECK: Operation does not verify
transform.loop.unroll %loop {factor = 0 : i64} : !transform.op<"scf.for">

// -----

%loop = "test.op"() : () -> !transform.op<"scf.for">

// CHECK: Operation does not verify
transform.loop.unroll %loop {factor = -4 : i64} : !transform.op<"scf.for">

// -----

%loop = "test.op"() : () -> !transform.op<"scf.for">

// CHECK-NOT: Operation does not verify
transform.loop.unroll %loop {factor = 4 : i64} : !transform.op<"scf.for">

// -----

%loop = "test.op"() : () -> !transform.op<"scf.for">

// CHECK: Expected attribute i64 but got i32
transform.loop.unroll %loop {factor = 4 : i32} : !transform.op<"scf.for">

// -----

%loop = "test.op"() : () -> !transform.op<"scf.for">

// CHECK: Expected attribute i64 but got i16
transform.loop.unroll %loop {factor = 4 : i16} : !transform.op<"scf.for">
