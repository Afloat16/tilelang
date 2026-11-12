import pytest
import torch

from varlen_utils import varlen_attention_ref


@pytest.mark.parametrize(
    "q_lengths,k_lengths,heads,kv_heads,causal,window,dtype",
    [
        pytest.param([0, 0], [0, 0], 2, 2, False, None, torch.float32, id="all-empty"),
        pytest.param([0, 0], [3, 2], 2, 2, True, None, torch.float16, id="empty-query-causal"),
        pytest.param([0, 0], [3, 2], 4, 2, False, (1, 0), torch.float32, id="empty-query-gqa-window"),
        pytest.param([0, 2, 0, 1], [1, 3, 0, 2], 4, 2, True, None, torch.float32, id="mixed-empty-query"),
        pytest.param([2, 1], [0, 3], 2, 2, False, None, torch.float32, id="empty-key"),
        pytest.param([2, 3], [3, 2], 4, 2, True, (1, 0), torch.float32, id="nonempty-window"),
    ],
)
def test_varlen_attention_reference_empty_sequences(q_lengths, k_lengths, heads, kv_heads, causal, window, dtype):
    q_offsets = [0]
    k_offsets = [0]
    for q_length, k_length in zip(q_lengths, k_lengths):
        q_offsets.append(q_offsets[-1] + q_length)
        k_offsets.append(k_offsets[-1] + k_length)

    # Zero query/key scores make the reference an independently checkable mean
    # over permitted values. Value and query head widths intentionally differ.
    query = torch.zeros((q_offsets[-1], heads, 4), dtype=dtype, requires_grad=True)
    key = torch.zeros((k_offsets[-1], kv_heads, 4), dtype=dtype, requires_grad=True)
    value = torch.arange(k_offsets[-1] * kv_heads * 3, dtype=dtype).reshape(k_offsets[-1], kv_heads, 3).requires_grad_()
    output = varlen_attention_ref(
        query,
        key,
        value,
        torch.tensor(q_offsets, dtype=torch.int32),
        torch.tensor(k_offsets, dtype=torch.int32),
        causal=causal,
        window_size=window,
    )

    expected = torch.zeros((q_offsets[-1], heads, 3), dtype=dtype)
    for batch, (q_length, k_length) in enumerate(zip(q_lengths, k_lengths)):
        for row in range(q_length):
            aligned_row = row + k_length - q_length
            permitted = [
                col
                for col in range(k_length)
                if (not causal or col <= aligned_row)
                and (window is None or window[0] < 0 or col >= aligned_row - window[0])
                and (window is None or window[1] < 0 or col <= aligned_row + window[1])
            ]
            if permitted:
                indices = torch.tensor(permitted) + k_offsets[batch]
                for head in range(heads):
                    expected[q_offsets[batch] + row, head] = value.detach()[indices, head // (heads // kv_heads)].float().mean(0)

    assert output.dtype == dtype
    assert output.device == query.device
    torch.testing.assert_close(output, expected)
    output.sum().backward()
    for tensor in (query, key, value):
        assert tensor.grad is not None
        if q_offsets[-1] == 0:
            torch.testing.assert_close(tensor.grad, torch.zeros_like(tensor))
        else:
            assert torch.isfinite(tensor.grad).all()
