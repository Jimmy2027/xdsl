// RUN: xdsl-opt %s --print-op-generic | mlir-opt --mlir-print-op-generic --allow-unregistered-dialect | filecheck %s

// Test basic apply_patterns op with tensor.merge_consecutive_insert_extract_slice pattern
%0 = "test.op"() : () -> !transform.any_op

// CHECK: "transform.apply_patterns"(%0) <{max_iterations = -1 : i64, max_num_rewrites = -1 : i64}> ({
// CHECK:   "transform.apply_patterns.tensor.merge_consecutive_insert_extract_slice"() : () -> ()
// CHECK: }) : (!transform.any_op) -> ()
"transform.apply_patterns"(%0) ({
  "transform.apply_patterns.tensor.merge_consecutive_insert_extract_slice"() : () -> ()
}) {max_iterations = -1 : i64, max_num_rewrites = -1 : i64} : (!transform.any_op) -> ()

// Test apply_patterns with apply_cse attribute
// CHECK: "transform.apply_patterns"(%0) <{apply_cse, max_iterations = -1 : i64, max_num_rewrites = -1 : i64}> ({
// CHECK:   "transform.apply_patterns.tensor.merge_consecutive_insert_extract_slice"() : () -> ()
// CHECK: }) : (!transform.any_op) -> ()
"transform.apply_patterns"(%0) <{apply_cse}> ({
  "transform.apply_patterns.tensor.merge_consecutive_insert_extract_slice"() : () -> ()
}) {max_iterations = -1 : i64, max_num_rewrites = -1 : i64} : (!transform.any_op) -> ()

// Test apply_patterns with custom max_iterations
// CHECK: "transform.apply_patterns"(%0) <{max_iterations = 10 : i64, max_num_rewrites = -1 : i64}> ({
// CHECK:   "transform.apply_patterns.tensor.merge_consecutive_insert_extract_slice"() : () -> ()
// CHECK: }) : (!transform.any_op) -> ()
"transform.apply_patterns"(%0) ({
  "transform.apply_patterns.tensor.merge_consecutive_insert_extract_slice"() : () -> ()
}) {max_iterations = 10 : i64, max_num_rewrites = -1 : i64} : (!transform.any_op) -> ()

// Test apply_patterns with custom max_num_rewrites
// CHECK: "transform.apply_patterns"(%0) <{max_iterations = -1 : i64, max_num_rewrites = 100 : i64}> ({
// CHECK:   "transform.apply_patterns.tensor.merge_consecutive_insert_extract_slice"() : () -> ()
// CHECK: }) : (!transform.any_op) -> ()
"transform.apply_patterns"(%0) ({
  "transform.apply_patterns.tensor.merge_consecutive_insert_extract_slice"() : () -> ()
}) {max_iterations = -1 : i64, max_num_rewrites = 100 : i64} : (!transform.any_op) -> ()

// Test apply_patterns with all custom attributes
// CHECK: "transform.apply_patterns"(%0) <{apply_cse, max_iterations = 5 : i64, max_num_rewrites = 50 : i64}> ({
// CHECK:   "transform.apply_patterns.tensor.merge_consecutive_insert_extract_slice"() : () -> ()
// CHECK: }) : (!transform.any_op) -> ()
"transform.apply_patterns"(%0) <{apply_cse}> ({
  "transform.apply_patterns.tensor.merge_consecutive_insert_extract_slice"() : () -> ()
}) {max_iterations = 5 : i64, max_num_rewrites = 50 : i64} : (!transform.any_op) -> ()
