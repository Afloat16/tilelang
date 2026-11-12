import numpy as np

import tilelang
from tilelang import tvm as tvm
from tvm.ir import IRModule
import tilelang.testing
import tilelang.language as T


def merge_if_test():
    @T.prim_func
    def main():
        A = T.alloc_fragment((1,), T.float16)
        B = T.alloc_fragment((1,), T.float16)
        C = T.alloc_fragment((1,), T.float16)
        D = T.alloc_fragment((1,), T.float16)
        if A[0] == 0:
            A[0] = 0
        if B[0] == 0:
            B[0] = 0
        if C[0] == 0:
            C[0] = 0
        if D[0] == 0:
            D[0] = 0

    return main


def test_merge_if():
    func = merge_if_test()
    original_module = IRModule.from_expr(func)
    transformed = tilelang.transform.MergeIfStmt()(original_module)
    tvm.ir.assert_structural_equal(original_module["main"], transformed["main"], True)


def test_merge_if_preserves_re_evaluation_of_buffer_condition():
    @T.prim_func
    def main(A: T.Tensor((2,), T.int32)):
        if A[0] > 0:
            A[0] = 0
        if A[0] > 0:
            A[1] = 1

    original_module = IRModule.from_expr(main)
    transformed = tilelang.transform.MergeIfStmt()(original_module)
    executable = tvm.compile(transformed["main"], target="c").jit(options=["-std=c++17"])

    values = tvm.runtime.tensor(np.array([1, 7], dtype="int32"))
    executable(values)

    np.testing.assert_array_equal(values.numpy(), np.array([0, 7], dtype="int32"))


def test_if_stmt_binding_and_merge_if_preserve_single_evaluation():
    @T.prim_func
    def main(A: T.Tensor((2,), T.int32)):
        if A[0] > 0:
            A[0] = 0
            A[1] = 1

    transformed = IRModule.from_expr(main)
    transformed = tilelang.transform.IfStmtBinding()(transformed)
    transformed = tilelang.transform.MergeIfStmt()(transformed)
    executable = tvm.compile(transformed["main"], target="c").jit(options=["-std=c++17"])

    values = tvm.runtime.tensor(np.array([1, 7], dtype="int32"))
    executable(values)

    np.testing.assert_array_equal(values.numpy(), np.array([0, 1], dtype="int32"))


def test_merge_if_combines_newly_adjacent_nested_bodies():
    @T.prim_func
    def before(flag: T.int32, inner: T.int32, A: T.Tensor((4,), T.int32)):
        if flag > 0:
            if inner > 0:
                A[0] = 1
            if inner > 0:
                A[1] = 2
        if flag > 0:
            if inner > 0:
                A[2] = 3
            if inner > 0:
                A[3] = 4

    @T.prim_func
    def expected(flag: T.int32, inner: T.int32, A: T.Tensor((4,), T.int32)):
        if flag > 0:  # noqa: SIM102 - The expected IR retains both conditional nodes.
            if inner > 0:
                A[0] = 1
                A[1] = 2
                A[2] = 3
                A[3] = 4

    transformed = tilelang.transform.MergeIfStmt()(IRModule.from_expr(before))["before"]
    tvm.ir.assert_structural_equal(transformed.body, expected.body, True)
    executable = tvm.compile(transformed, target="c").jit(options=["-std=c++17"])
    for flag, inner in [(1, 1), (0, 1), (1, 0)]:
        values = tvm.runtime.tensor(np.zeros(4, dtype="int32"))
        executable(flag, inner, values)
        wanted = np.arange(1, 5, dtype="int32") if flag > 0 and inner > 0 else np.zeros(4, dtype="int32")
        np.testing.assert_array_equal(values.numpy(), wanted)


def test_merge_if_preserves_nested_else_and_statement_boundaries():
    @T.prim_func
    def before(flag: T.int32, inner: T.int32, A: T.Tensor((4,), T.int32)):
        if flag > 0:
            if inner > 0:
                A[0] = 1
            A[1] = 2
        if flag > 0:
            if inner > 0:
                A[2] = 3
            else:
                A[2] = -3
            if inner > 0:
                A[3] = 4

    @T.prim_func
    def expected(flag: T.int32, inner: T.int32, A: T.Tensor((4,), T.int32)):
        if flag > 0:
            if inner > 0:
                A[0] = 1
            A[1] = 2
            if inner > 0:
                A[2] = 3
            else:
                A[2] = -3
            if inner > 0:
                A[3] = 4

    transformed = tilelang.transform.MergeIfStmt()(IRModule.from_expr(before))["before"]
    tvm.ir.assert_structural_equal(transformed.body, expected.body, True)


if __name__ == "__main__":
    tilelang.testing.main()
