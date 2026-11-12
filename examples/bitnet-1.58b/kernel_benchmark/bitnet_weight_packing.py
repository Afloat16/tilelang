# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np


def general_compress(lowprecision_weight, source_bits=4, storage_dtype=np.int8):
    """Pack consecutive low-bit weights into bytes, least significant field first."""
    elems_per_byte = 8 // source_bits
    if lowprecision_weight.dtype == np.float16:
        lowprecision_weight = lowprecision_weight.astype(dtype=np.int8)
    packed = np.zeros(
        (*lowprecision_weight.shape[:-1], lowprecision_weight.shape[-1] // elems_per_byte),
        dtype=np.int8,
    )
    stop = packed.shape[-1] * elems_per_byte
    for k in range(elems_per_byte):
        packed |= lowprecision_weight[..., k:stop:elems_per_byte] << (source_bits * k)
    return packed.view(storage_dtype)
