import torch
import triton
import triton.language as tl


def solve(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    M: int,
    N: int,
    K: int,
    alpha: float,
    beta: float,
):
    a = A.view(M, K)
    b = B.view(K, N)
    c = C.view(M, N)
    torch.addmm(c, a, b, beta=beta, alpha=alpha, out=C)


def solve_triton(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    M: int,
    N: int,
    K: int,
    alpha: float,
    beta: float,
):
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 64

    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))

    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        alpha, beta,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )


# ============================================================================
# Beginner-Friendly Triton GEMM
# ============================================================================
# GEMM: C = alpha * (A @ B) + beta * C
#   A: [M, K]   B: [K, N]   C: [M, N]
#
# Core idea (Tiled Matrix Multiplication):
#   We can't compute the entire C matrix at once — it's too big.
#   So we split C into small BLOCK_SIZE_M x BLOCK_SIZE_N tiles.
#   Each GPU program (thread block) computes ONE tile of C.
#
#   For each tile of C, we need the corresponding rows of A and
#   columns of B. But even those might be too big, so we split
#   the K dimension into chunks of BLOCK_SIZE_K and accumulate
#   partial results in a loop.
#
# Visual (2D grid of programs, each computes one tile of C):
#
#        N dimensions (split into BLOCK_SIZE_N columns)
#     ┌──────┬──────┬──────┐
#     │ prog │ prog │ prog │  ← each box = one program
#     │ (0,0)│ (1,0)│ (2,0)│
#     ├──────┼──────┼──────┤
#  M  │ prog │ prog │ prog │
#  dim│ (0,1)│ (1,1)│ (2,1)│
#     ├──────┼──────┼──────┤
#     │ prog │ prog │ prog │
#     │ (0,2)│ (1,2)│ (2,2)│
#     └──────┴──────┴──────┘
#
#   Program (px, py) computes C[py*BM : (py+1)*BM, px*BN : (px+1)*BN]
# ============================================================================

def solve_triton_beginner(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    M: int,
    N: int,
    K: int,
    alpha: float,
    beta: float,
):
    """Entry point for the beginner-friendly Triton GEMM.

    This computes: C = alpha * (A @ B) + beta * C
    where A is [M, K], B is [K, N], C is [M, N].
    """
    # Block sizes: each program computes a BM x BN tile of C,
    # and processes K in chunks of BK.
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 64

    # 2D grid: how many programs do we need in each dimension?
    #   - ceil(M / BLOCK_SIZE_M) programs along the M (row) dimension
    #   - ceil(N / BLOCK_SIZE_N) programs along the N (col) dimension
    grid = (triton.cdiv(N, BLOCK_SIZE_N), triton.cdiv(M, BLOCK_SIZE_M))

    matmul_kernel_beginner[grid](
        A, B, C,
        M, N, K,
        alpha, beta,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )


@triton.jit
def matmul_kernel_beginner(
    A_ptr, B_ptr, C_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    alpha: tl.constexpr, beta: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):

    pid_n = tl.program_id(0)   # which column tile
    pid_m = tl.program_id(1)   # which row tile

    # The starting row and column for this tile in the C matrix
    row_start = pid_m * BLOCK_SIZE_M   # first row this program handles
    col_start = pid_n * BLOCK_SIZE_N   # first column this program handles

    # ======================================================================
    # STEP 2: Pre-compute row and column indices for this tile
    # ======================================================================
    # We need to know which specific rows of A and columns of B to load.
    # These are used repeatedly in the K-loop, so compute them once.

    # Row indices for this tile: [row_start, row_start+1, ..., row_start+BM-1]
    row_indices = row_start + tl.arange(0, BLOCK_SIZE_M)   # shape: [BLOCK_SIZE_M]

    # Column indices for this tile: [col_start, col_start+1, ..., col_start+BN-1]
    col_indices = col_start + tl.arange(0, BLOCK_SIZE_N)   # shape: [BLOCK_SIZE_N]

    # K-dimension indices (will be shifted each iteration): [0, 1, ..., BK-1]
    k_indices = tl.arange(0, BLOCK_SIZE_K)                 # shape: [BLOCK_SIZE_K]

    # Masks for boundary checking (M or N might not be multiples of block size)
    valid_row = row_indices < M       # which rows actually exist in A/C
    valid_col = col_indices < N       # which columns actually exist in B/C

    # ======================================================================
    # STEP 3: Initialize the accumulator
    # ======================================================================
    # The accumulator holds the partial dot products as we iterate over K.
    # Shape: [BLOCK_SIZE_M, BLOCK_SIZE_N]
    #   - Each row corresponds to a row of A
    #   - Each column corresponds to a column of B
    #   - acc[i, j] will eventually hold the dot product of A_row_i and B_col_j
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # ======================================================================
    # STEP 4: Main K-loop — accumulate partial results
    # ======================================================================
    # The K dimension is too large to process at once, so we split it into
    # chunks of BLOCK_SIZE_K and accumulate partial dot products.
    #
    # Mathematically:  C[i,j] = sum_k A[i,k] * B[k,j]
    # We compute this as:
    #   for each k_block:
    #     acc += A_tile @ B_tile    (partial matrix multiply)
    #
    # Total iterations = ceil(K / BLOCK_SIZE_K)
    num_k_blocks = tl.cdiv(K, BLOCK_SIZE_K)

    for k_block_idx in tl.range(0, num_k_blocks):
        # ----- 4a: Compute actual K indices for this block -----
        # k_offset = [k_block_idx*BK, k_block_idx*BK+1, ..., k_block_idx*BK+BK-1]
        k_offset = k_block_idx * BLOCK_SIZE_K + k_indices   # shape: [BLOCK_SIZE_K]
        valid_k = k_offset < K   # which K indices are in bounds

        # ----- 4b: Load a tile of A -----
        # A has shape [M, K], stored in row-major order.
        # A[row, k] is at memory offset: row * K + k
        #
        # We want A[row_indices, k_offset], which forms a [BLOCK_SIZE_M, BLOCK_SIZE_K] tile.
        # Using broadcasting:
        #   row_indices[:, None]  shape: [BLOCK_SIZE_M, 1]
        #   k_offset[None, :]    shape: [1, BLOCK_SIZE_K]
        #   => A_offset          shape: [BLOCK_SIZE_M, BLOCK_SIZE_K]
        A_offset = row_indices[:, None] * K + k_offset[None, :]
        A_mask = valid_row[:, None] & valid_k[None, :]
        A_tile = tl.load(A_ptr + A_offset, mask=A_mask, other=0.0)
        # A_tile shape: [BLOCK_SIZE_M, BLOCK_SIZE_K]
        # Out-of-bounds values are loaded as 0.0, so they don't affect the dot product.

        # ----- 4c: Load a tile of B -----
        # B has shape [K, N], stored in row-major order.
        # B[k, col] is at memory offset: k * N + col
        #
        # We want B[k_offset, col_indices], which forms a [BLOCK_SIZE_K, BLOCK_SIZE_N] tile.
        B_offset = k_offset[:, None] * N + col_indices[None, :]
        B_mask = valid_k[:, None] & valid_col[None, :]
        B_tile = tl.load(B_ptr + B_offset, mask=B_mask, other=0.0)
        # B_tile shape: [BLOCK_SIZE_K, BLOCK_SIZE_N]

        # ----- 4d: Accumulate partial matrix multiply -----
        # tl.dot computes a matrix multiply and adds to the accumulator.
        # acc += A_tile @ B_tile
        # [BLOCK_SIZE_M, BLOCK_SIZE_K] @ [BLOCK_SIZE_K, BLOCK_SIZE_N]
        #   => [BLOCK_SIZE_M, BLOCK_SIZE_N]
        #
        # tl.dot is optimized for GPU tensor cores — much faster than manual loops.
        # It requires float16 or bfloat16 inputs for tensor core usage.
        acc = tl.dot(tl.cast(A_tile, tl.float16), tl.cast(B_tile, tl.float16), acc)
        # Note: acc stays in float32 for precision, but inputs to tl.dot
        # are cast to float16 so tensor cores can be used.

    # ======================================================================
    # STEP 5: Apply alpha and beta, then write the result
    # ======================================================================
    # GEMM formula: C = alpha * (A @ B) + beta * C
    # We've computed (A @ B) in acc. Now we need the original C values
    # for the beta term.

    # Compute memory offsets for the C tile
    # C has shape [M, N], stored in row-major order.
    # C[row, col] is at memory offset: row * N + col
    C_offset = row_indices[:, None] * N + col_indices[None, :]
    C_mask = valid_row[:, None] & valid_col[None, :]

    # Load the original C values (needed for the beta * C term)
    C_original = tl.load(C_ptr + C_offset, mask=C_mask, other=0.0)

    # Apply: output = alpha * acc + beta * C_original
    # Cast C_original to float32 for consistent arithmetic
    output = alpha * acc + beta * tl.cast(C_original, tl.float32)

    # Cast back to the original C dtype (assumed float16 here) and store
    tl.store(C_ptr + C_offset, tl.cast(output, tl.float16), mask=C_mask)

