// RUN: XDSL_ROUNDTRIP

%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%c2 = arith.constant 2 : index
%c3 = arith.constant 3 : index

// CHECK: {my_attributes = 1 : i32}
scf.for %i0 = %c0 to %c3 step %c1 {
    "test.op"() {"not constant"} : () -> ()
} {my_attributes = 1:i32}
