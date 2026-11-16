"""Public CPU kernel baseline probe; kept outside the upstream source tree."""

import json
import pathlib
import subprocess
import traceback

import numpy as np
import torch

import tilelang
import tilelang.cpu.language as T
from tilelang import tvm


def make_kernel(operation, dtype="int32", size=256, broadcast=False):
    @T.prim_func
    def kernel(A: T.Tensor((size,), dtype), B: T.Tensor((size,), dtype), C: T.Tensor((size,), dtype)):
        for i in T.Parallel(size):
            if operation == "and":
                C[i] = T.bitwise_and(A[i], B[0] if broadcast else B[i])
            elif operation == "or":
                C[i] = T.bitwise_or(A[i], B[0] if broadcast else B[i])
            elif operation == "xor":
                C[i] = T.bitwise_xor(A[i], B[0] if broadcast else B[i])
            elif operation == "not":
                C[i] = T.bitwise_not(A[i])
            elif operation == "left":
                C[i] = T.shift_left(A[i], B[0] if broadcast else B[i])
            elif operation == "right":
                C[i] = T.shift_right(A[i], B[0] if broadcast else B[i])
            elif operation == "add_control":
                C[i] = A[i] + B[i]

    return kernel


def main():
    output = pathlib.Path(__file__).parent / "vector-public-probe"
    output.mkdir(exist_ok=True)
    rows = []
    for operation in ("and", "or", "xor", "not", "left", "right", "add_control"):
        row = {"operation": operation, "dtype": "int32", "size": 256}
        try:
            program = make_kernel(operation)
            with tvm.target.Target("c"):
                source = tilelang.lower(program).kernel_source
            row["vectorized"] = "int32_t4" in source
            (output / f"{operation}.cc").write_text(source)
            kernel = tilelang.compile(program, target="c", target_host="c", execution_backend="cython")
            a = torch.arange(-128, 128, dtype=torch.int32)
            b = torch.arange(256, dtype=torch.int32) % 32
            c = torch.empty_like(a)
            kernel(a, b, c)
            reference = {
                "and": np.bitwise_and,
                "or": np.bitwise_or,
                "xor": np.bitwise_xor,
                "left": np.left_shift,
                "right": np.right_shift,
                "add_control": np.add,
            }
            expected = np.bitwise_not(a.numpy()) if operation == "not" else reference[operation](a.numpy(), b.numpy())
            np.testing.assert_array_equal(c.numpy(), expected)
            row["result"] = "passed"
        except Exception as exc:
            row["result"] = "failed"
            row["error_type"] = type(exc).__name__
            row["error"] = str(exc)
            (output / f"{operation}.log").write_text(traceback.format_exc())
        rows.append(row)
        print(json.dumps({key: value for key, value in row.items() if key != "error"}), flush=True)
    report = {
        "base_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "rows": rows,
    }
    (output / "baseline.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
