import torch
import triton
import triton.language as tl

def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    output.copy_(torch.softmax(input, dim=-1))


@triton.jit
def softmax_kernel(output_ptr, input_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # 1. Find max for stability
    x_max = float('-inf')
    for k in range(0, N, BLOCK_SIZE):
        offsets = k + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        x = tl.load(input_ptr + offsets, mask=mask, other=float('-inf'))
        x_max = tl.maximum(tl.max(x), x_max)
    
    # 2. Compute sum of exp(x - max)
    denom = 0.0
    for k in range(0, N, BLOCK_SIZE):
        offsets = k + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        x = tl.load(input_ptr + offsets, mask=mask, other=float('-inf'))
        # Subtract max and compute exp
        denom += tl.sum(tl.exp(x - x_max))

    # 3. Compute and store result
    for k in range(0, N, BLOCK_SIZE):
        offsets = k + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        x = tl.load(input_ptr + offsets, mask=mask, other=float('-inf'))
        res = tl.exp(x - x_max) / denom
        tl.store(output_ptr + offsets, res, mask=mask)

def solve(input: torch.Tensor, output: torch.Tensor, N: int, BLOCK_SIZE: int = 1024):
    # Ensure input is 1D for this specific kernel
    assert input.is_contiguous()
    # Launch with 1 grid instance as per your request
    softmax_kernel[(1,)](output, input, N, BLOCK_SIZE=BLOCK_SIZE)
