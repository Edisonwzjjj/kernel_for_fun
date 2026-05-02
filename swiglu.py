import torch
from triton import jit
import triton.language as tl


"""SwiGLU
Implement the Swish-Gated Linear Unit (SWiGLU) activation function forward pass for 1D input vectors. 
Given an input tensor of shape [N] where N is the number of elements, 
compute the output using the elementwise formula. 
The input and output tensor must be of type float32.
"""


def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    first = input[:N//2]
    second = input[N//2:]
    silu = first * torch.sigmoid(first)
    output.copy_(silu * second)


def solve_triton(input: torch.Tensor, output: torch.Tensor, N: int):
    N_half = N // 2
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N_half, BLOCK_SIZE),)
    swiglu_kernel[grid](
        input, 
        output, 
        N_half, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    

@triton.jit
def swiglu_kernel(
    input_ptr, 
    output_ptr, 
    N_half,              # N 的一半，即实际计算的长度
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_half
    
    # 1. 加载 gate (前一半)
    gate = tl.load(input_ptr + offsets, mask=mask)
    
    # 2. 加载 data (后一半，地址需要加上 N_half)
    data = tl.load(input_ptr + offsets + N_half, mask=mask)
    
    # 3. Swish(gate) = gate * sigmoid(gate)
    swish_gate = gate * tl.sigmoid(gate)
    
    # 4. 门控输出
    out = swish_gate * data
    
    # 5. 存入 output
    tl.store(output_ptr + offsets, out, mask=mask)
