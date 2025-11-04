// RUN: xdsl-opt %s | xdsl-opt | mlir-opt --allow-unregistered-dialect | filecheck %s

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = "test.op"() : () -> !transform.any_op
    "transform.loop.unroll"(%0) <{factor = 4 : i64}> : (!transform.any_op) -> ()
    transform.yield
  }
}

//CHECK: module attributes {transform.with_named_sequence} {
//CHECK-NEXT:   transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
//CHECK-NEXT:     %0 = "test.op"() : () -> !transform.any_op
//CHECK-NEXT:     transform.loop.unroll %0 {factor = 4 : i64} : !transform.any_op
//CHECK-NEXT:     transform.yield
//CHECK-NEXT:   }
//CHECK-NEXT: }
