import importlib.util
from pathlib import Path

import numpy as np
import pytest


_SOURCE = Path(__file__).resolve().parents[3] / "examples/bitnet-1.58b/kernel_benchmark/bitnet_weight_packing.py"
_SPEC = importlib.util.spec_from_file_location("bitnet_weight_packing", _SOURCE)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


@pytest.mark.parametrize("source_bits", [1, 2, 4, 8])
@pytest.mark.parametrize("dtype", [np.int8, np.float16])
@pytest.mark.parametrize("layout", ["contiguous", "strided", "empty_rows", "empty_columns"])
def test_bitnet_weight_packing_round_trip(source_bits, dtype, layout):
    generator = np.random.default_rng(0)
    codes = generator.integers(0, 1 << source_bits, size=(3, 130), dtype=np.uint8)
    weights = codes.astype(dtype)
    if layout == "strided":
        codes, weights = codes[:, ::2], weights[:, ::2]
    elif layout == "empty_rows":
        codes, weights = codes[:0], weights[:0]
    elif layout == "empty_columns":
        codes, weights = codes[:, :0], weights[:, :0]
    original = weights.copy()

    packed = _MODULE.general_compress(weights, source_bits)
    fields = 8 // source_bits
    columns = codes.shape[-1] // fields
    assert packed.dtype == np.int8
    assert packed.shape == (codes.shape[0], columns)
    bits = np.unpackbits(packed.view(np.uint8), axis=-1, bitorder="little")
    decoded = (bits.reshape(codes.shape[0], columns * fields, source_bits) * (1 << np.arange(source_bits))).sum(axis=-1)
    np.testing.assert_array_equal(decoded, codes[:, : columns * fields])
    np.testing.assert_array_equal(weights, original)


@pytest.mark.parametrize("storage_dtype", [np.int8, np.uint8, np.int32])
def test_bitnet_weight_packing_storage_view(storage_dtype):
    codes = np.tile(np.array([[0, 1, 2, 3, 3, 2, 1, 0]], dtype=np.int8), (2, 2))
    packed = _MODULE.general_compress(codes, 2, storage_dtype)
    assert packed.dtype == storage_dtype
    expected_bytes = np.tile(np.array([[0xE4, 0x1B, 0xE4, 0x1B]], dtype=np.uint8), (2, 1))
    np.testing.assert_array_equal(packed.view(np.uint8), expected_bytes)
