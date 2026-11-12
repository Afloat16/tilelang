"""Persistent coordinates stay in scope through CPU lowering and execution."""

import torch

import tilelang
import tilelang.cpu.language as T


def test_cpu_persistent_coordinate_bindings():
    @T.prim_func
    def main(A: T.Tensor((2, 3, 10), T.int32), B: T.Tensor((2, 3, 10), T.int32)):
        with T.Kernel(4) as worker:
            for i, j, k in T.Persistent([2, 3, 10], 4, worker, group_size=3):
                B[i, j, k] = A[i, j, k] + 1

    kernel = tilelang.compile(main, out_idx=[1], target="c", target_host="c", execution_backend="cython")
    values = torch.arange(60, dtype=torch.int32).reshape(2, 3, 10)
    torch.testing.assert_close(kernel(values), values + 1)
