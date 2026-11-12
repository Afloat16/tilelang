import pytest
import torch

from bert_padding import index_first_axis, pad_input, unpad_input, unpad_input_for_concatenated_sequences


@pytest.mark.parametrize(
    "dtype,strided,mask_kind",
    [
        (torch.float32, False, "partial"),
        (torch.float64, True, "partial"),
        (torch.float16, False, "full"),
        (torch.bfloat16, True, "empty"),
    ],
)
def test_unpad_cpu_values_metadata_and_gradients(dtype, strided, mask_kind):
    values = torch.arange(120, dtype=dtype).reshape(3, 5, 4, 2)
    if strided:
        values = values.transpose(-1, -2)
    values.requires_grad_()
    mask = torch.tensor([[True, False, True, False, False], [False] * 5, [False, True, True, False, True]])
    if mask_kind != "partial":
        mask.fill_(mask_kind == "full")

    output, indices, cumulative, maximum = unpad_input(values, mask)
    torch.testing.assert_close(output, values[mask], rtol=0, atol=0)
    expected_indices = [i for i, keep in enumerate(mask.flatten().tolist()) if keep]
    torch.testing.assert_close(indices, torch.tensor(expected_indices, dtype=torch.int64))
    lengths = [sum(row) for row in mask.tolist()]
    expected_cumulative = [0]
    for length in lengths:
        expected_cumulative.append(expected_cumulative[-1] + length)
    torch.testing.assert_close(cumulative, torch.tensor(expected_cumulative, dtype=torch.int32))
    assert maximum == max(lengths)

    restored = pad_input(output, indices, 3, 5)
    expected_restored = torch.zeros_like(values)
    expected_restored[mask] = values.detach()[mask]
    torch.testing.assert_close(restored, expected_restored, rtol=0, atol=0)

    gradient = (torch.arange(output.numel(), dtype=dtype).reshape(output.shape) % 7) - 3
    output.backward(gradient)
    expected_gradient = torch.zeros_like(values)
    expected_gradient[mask] = gradient
    torch.testing.assert_close(values.grad, expected_gradient, rtol=0, atol=0)


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))]
)
def test_index_first_axis_preserves_index_order(device):
    values = torch.arange(30, dtype=torch.float64, device=device).reshape(5, 2, 3).transpose(1, 2).requires_grad_()
    indices = torch.tensor([4, 0, 2], dtype=torch.int64, device=device)
    output = index_first_axis(values, indices)
    torch.testing.assert_close(output, torch.stack([values[4], values[0], values[2]]), rtol=0, atol=0)
    gradient = torch.arange(18, dtype=values.dtype, device=device).reshape(3, 2, 3).transpose(1, 2)
    output.backward(gradient)
    expected_gradient = torch.zeros_like(values)
    for row, index in enumerate([4, 0, 2]):
        expected_gradient[index] = gradient[row]
    torch.testing.assert_close(values.grad, expected_gradient, rtol=0, atol=0)


def test_unpad_concatenated_cpu_sequences():
    values = torch.arange(48, dtype=torch.float32).reshape(2, 6, 4).requires_grad_()
    lengths = torch.tensor([[2, 3, 0, 0, 0, 0], [1, 2, 0, 0, 0, 0]], dtype=torch.int32)
    output, indices, cumulative, maximum = unpad_input_for_concatenated_sequences(values, lengths)
    torch.testing.assert_close(output, torch.cat([values[0, :5], values[1, :3]]), rtol=0, atol=0)
    torch.testing.assert_close(indices, torch.tensor([0, 1, 2, 3, 4, 6, 7, 8]))
    torch.testing.assert_close(cumulative, torch.tensor([0, 2, 5, 6, 8], dtype=torch.int32))
    assert maximum == 3
    output.sum().backward()
    expected_gradient = torch.zeros_like(values)
    expected_gradient[0, :5] = 1
    expected_gradient[1, :3] = 1
    torch.testing.assert_close(values.grad, expected_gradient, rtol=0, atol=0)
