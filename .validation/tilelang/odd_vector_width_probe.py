"""Explore explicit CPU vector widths using unmodified public APIs."""
import json
import pathlib
import subprocess
import traceback
import numpy as np
import torch
import tilelang
import tilelang.cpu.language as T
from tilelang import tvm

def make_program(size, dtype):
    @T.prim_func
    def program(A: T.Tensor((size,), dtype), B: T.Tensor((size,), dtype), C: T.Tensor((size,), dtype)):
        for i in T.vectorized(size):
            C[i] = A[i] + B[i]
    return program

def main():
    output = pathlib.Path(__file__).parent / "reports" / "odd-vector-widths"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    c_types = {"float32": "float", "float64": "double", "int32": "int32_t", "int64": "int64_t"}
    for dtype in c_types:
        for size in (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16):
            row = {"dtype": dtype, "size": size}
            try:
                program = make_program(size, dtype)
                with tvm.target.Target("c"):
                    source = tilelang.lower(program).kernel_source
                (output / f"{dtype}-{size}.cc").write_text(source)
                row["requested_vector_alias_present"] = f"{c_types[dtype]}{size}" in source
                kernel = tilelang.compile(program, target="c", target_host="c", execution_backend="cython")
                a = torch.arange(size, dtype=getattr(torch, dtype), device="cpu")
                b = torch.full((size,), 3, dtype=a.dtype, device="cpu")
                c = torch.empty_like(a)
                kernel(a, b, c)
                np.testing.assert_array_equal(c.numpy(), a.numpy() + b.numpy())
                row["result"] = "passed"
            except Exception as exc:
                row.update(result="failed", error_type=type(exc).__name__, error=str(exc))
                (output / f"{dtype}-{size}.log").write_text(traceback.format_exc())
            rows.append(row)
            print(json.dumps({k:v for k,v in row.items() if k != "error"}), flush=True)
    (output / "baseline.json").write_text(json.dumps({"source_head": subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(), "rows":rows}, indent=2)+"\n")
if __name__ == "__main__":
    main()
