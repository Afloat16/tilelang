"""Compile and execute CPU integer-vector bitwise operations."""

import re

import numpy as np
import pytest
import torch

import tilelang
import tilelang.cpu.language as T
from tilelang import tvm


def _bitwise_program(dtype, size, broadcast):
    @T.prim_func
    def program(
        A: T.Tensor((size,), dtype),
        B: T.Tensor((size,), dtype),
        Shifts: T.Tensor((size,), dtype),
        And: T.Tensor((size,), dtype),
        Or: T.Tensor((size,), dtype),
        Xor: T.Tensor((size,), dtype),
        Not: T.Tensor((size,), dtype),
        Left: T.Tensor((size,), dtype),
        Right: T.Tensor((size,), dtype),
    ):
        for i in T.Parallel(size):
            And[i] = T.bitwise_and(A[i], B[0] if broadcast else B[i])
        for i in T.Parallel(size):
            Or[i] = T.bitwise_or(A[i], B[0] if broadcast else B[i])
        for i in T.Parallel(size):
            Xor[i] = T.bitwise_xor(A[i], B[0] if broadcast else B[i])
        for i in T.Parallel(size):
            Not[i] = T.bitwise_not(A[i])
        for i in T.Parallel(size):
            Left[i] = T.shift_left(A[i], Shifts[0] if broadcast else Shifts[i])
        for i in T.Parallel(size):
            Right[i] = T.shift_right(A[i], Shifts[0] if broadcast else Shifts[i])

    return program


@pytest.mark.parametrize("dtype", ["int8", "int16", "int32", "int64", "uint8", "uint16", "uint64"])
@pytest.mark.parametrize("size", [256, 260, 259])
@pytest.mark.parametrize("broadcast", [False, True])
def test_cpu_integer_vector_bitwise(dtype, size, broadcast):
    program = _bitwise_program(dtype, size, broadcast)
    with tvm.target.Target("c"):
        source = tilelang.lower(program).kernel_source
    assert source is not None
    if size % 2 == 0:
        assert re.search(rf"\b{dtype}_t(?:2|4|8|16)\b", source), source

    kernel = tilelang.compile(program, target="c", target_host="c", execution_backend="cython")
    numpy_dtype = np.dtype(dtype)
    bits = numpy_dtype.itemsize * 8
    unsigned_dtype = np.dtype(f"uint{bits}")
    full_mask = (1 << bits) - 1
    alternating = full_mask // 3
    patterns = np.array([0, 1, full_mask, 1 << (bits - 1), (1 << (bits - 1)) - 1, alternating, full_mask ^ alternating],
                        dtype=unsigned_dtype)
    a = np.resize(patterns, size).view(numpy_dtype).copy()
    b = np.resize(np.roll(patterns, 2), size).view(numpy_dtype).copy()
    shifts = (np.arange(size) % bits).astype(numpy_dtype)
    shifts[0] = bits - 1
    mask = b[0] if broadcast else b
    counts = shifts[0] if broadcast else shifts
    expected = (
        np.bitwise_and(a, mask),
        np.bitwise_or(a, mask),
        np.bitwise_xor(a, mask),
        np.bitwise_not(a),
        np.left_shift(a, counts),
        np.right_shift(a, counts),
    )
    a_tensor, b_tensor, shifts_tensor = (torch.from_numpy(value) for value in (a, b, shifts))
    outputs = [torch.empty_like(a_tensor) for _ in expected]
    kernel(a_tensor, b_tensor, shifts_tensor, *outputs)
    for actual, reference in zip(outputs, expected):
        np.testing.assert_array_equal(actual.numpy(), reference)
