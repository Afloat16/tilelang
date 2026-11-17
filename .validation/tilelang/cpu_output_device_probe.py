"""Exercise real CPU kernels with an explicitly simulated accelerator fallback."""
import json
import pathlib
import subprocess
import traceback
import torch
import tilelang
import tilelang.cpu.language as T
from tilelang.jit.adapter.base import BaseKernelAdapter

@T.prim_func
def fill_constant(Output: T.Tensor((16,), "float32")):
    for i in T.Parallel(16):
        Output[i] = T.float32(3)

@T.prim_func
def fill_scalar(scale: T.float32, Output: T.Tensor((16,), "float32")):
    for i in T.Parallel(16):
        Output[i] = scale

def main():
    output = pathlib.Path(__file__).parent / "output-device-probe"
    output.mkdir(exist_ok=True)
    original = BaseKernelAdapter.__dict__["get_current_device_functor"]
    rows = []
    for scalar in (False, True):
        for simulated_accelerator in (False, True):
            row = {"scalar": scalar, "simulated_accelerator_provider": simulated_accelerator}
            try:
                BaseKernelAdapter.get_current_device_functor = staticmethod(
                    lambda: (lambda: torch.device("cuda:0" if simulated_accelerator else "cpu"))
                )
                kernel = tilelang.compile(
                    fill_scalar if scalar else fill_constant,
                    out_idx=1 if scalar else 0,
                    target="c", target_host="c", execution_backend="cython",
                )
                result = kernel(7.5) if scalar else kernel()
                row["output_device"] = str(result.device)
                assert result.device.type == "cpu"
                torch.testing.assert_close(
                    result, torch.full((16,), 7.5 if scalar else 3.0, dtype=torch.float32, device="cpu")
                )
                row["result"] = "passed"
            except Exception as exc:
                row.update(result="failed", error_type=type(exc).__name__, error=str(exc))
                (output / f"scalar-{scalar}-accelerator-{simulated_accelerator}.log").write_text(traceback.format_exc())
            finally:
                BaseKernelAdapter.get_current_device_functor = original
            rows.append(row)
            print(json.dumps(row), flush=True)
    report = {
        "base_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "torch": torch.__version__,
        "scope": "Real CPU compilation, allocation and execution; only current-device provider is simulated. No CUDA or NPU hardware was exercised.",
        "rows": rows,
    }
    (output / "baseline.json").write_text(json.dumps(report, indent=2) + "\n")
    assert all(r["result"] == "passed" for r in rows if not r["simulated_accelerator_provider"])

if __name__ == "__main__":
    main()
