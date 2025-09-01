// RUN: xdsl-opt %s --print-op-generic | mlir-opt --mlir-print-op-generic --allow-unregistered-dialect | filecheck %s

// Prepare handles of various transform types.
%any_op = "test.op"() : () -> !transform.any_op
%param_i64 = "test.op"() : () -> !transform.param<i64>
%any_param = "test.op"() : () -> !transform.any_param

// CHECK: "transform.annotate"(%{{.*}}) <{name = "unit_attr"}> : (!transform.any_op) -> ()
"transform.annotate"(%any_op) <{name = "unit_attr"}> : (!transform.any_op) -> ()

// CHECK: "transform.annotate"(%{{.*}}, %{{.*}}) <{name = "int_attr"}> : (!transform.any_op, !transform.param<i64>) -> ()
"transform.annotate"(%any_op, %param_i64) <{name = "int_attr"}> : (!transform.any_op, !transform.param<i64>) -> ()

// CHECK: "transform.annotate"(%{{.*}}, %{{.*}}) <{name = "any_attr"}> : (!transform.any_op, !transform.any_param) -> ()
"transform.annotate"(%any_op, %any_param) <{name = "any_attr"}> : (!transform.any_op, !transform.any_param) -> ()
