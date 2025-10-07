// RUN: xdsl-opt %s | xdsl-opt | mlir-opt --allow-unregistered-dialect | filecheck %s

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = "test.op"() : () -> !transform.any_op
    %1 = "transform.get_operand"(%0) <{raw_position_list = array<i64: 0>}> : (!transform.any_op) -> !transform.any_value
    %2 = "transform.get_operand"(%0) <{raw_position_list = array<i64: 0, 1>}> : (!transform.any_op) -> !transform.any_value
    %3 = "transform.get_operand"(%0) <{is_inverted, raw_position_list = array<i64: 0>}> : (!transform.any_op) -> !transform.any_value
    %4 = "transform.get_operand"(%0) <{is_all, raw_position_list = array<i64>}> : (!transform.any_op) -> !transform.any_value
    transform.yield
  }
}

//CHECK: module attributes {transform.with_named_sequence} {
//CHECK-NEXT:   transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
//CHECK-NEXT:     %0 = "test.op"() : () -> !transform.any_op
//CHECK-NEXT:     %1 = transform.get_operand %0[0] : (!transform.any_op) -> !transform.any_value
//CHECK-NEXT:     %2 = transform.get_operand %0[0, 1] : (!transform.any_op) -> !transform.any_value
//CHECK-NEXT:     %3 = transform.get_operand %0[except(0)] : (!transform.any_op) -> !transform.any_value
//CHECK-NEXT:     %4 = transform.get_operand %0[all] : (!transform.any_op) -> !transform.any_value
//CHECK-NEXT:     transform.yield
//CHECK-NEXT:   }
//CHECK-NEXT: }
