import torch 
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


"""Implement a GPU program that computes the softmax attention operation for a given set of matrices. 
Given the query matrix Q of size M×d, key matrix K of size N×d, and value matrix V of size N×d"""

def solve(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, output: torch.Tensor, M: int, N: int, d: int):
    d_k = Q.size(-1)
    score = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
    score = F.softmax(score, dim=-1)
    output.copy_(torch.matmul(score, V))
    
    
def solve_triton(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, output: torch.Tensor, M: int, N: int, d: int):
    BLOCK_SIZE_D = max(32, triton.next_power_of_2(d))
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32

    grid = (triton.cdiv(M, BLOCK_SIZE_M), ) 

    attn[grid](
        Q, K, V, output,
        M, N, 
        d, BLOCK_SIZE_D, BLOCK_SIZE_M, BLOCK_SIZE_N
    )


def solve_triton_beginner(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                          output: torch.Tensor, M: int, N: int, d: int):
    """Entry point for the beginner-friendly Triton Flash Attention."""
    # Block sizes must be powers of 2 for Triton alignment requirements.
    # BLOCK_SIZE_D: how many columns of Q/K/V to process at once (along d dimension)
    # BLOCK_SIZE_M: how many rows of Q (query rows) each program handles
    # BLOCK_SIZE_N: how many rows of K/V each program processes per iteration
    BLOCK_SIZE_D = max(32, triton.next_power_of_2(d))
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32

    # Each program handles BLOCK_SIZE_M rows of Q.
    # Total number of programs = ceil(M / BLOCK_SIZE_M)
    grid = (triton.cdiv(M, BLOCK_SIZE_M),)

    attn_beginner[grid](
        Q, K, V, output,
        M, N,
        d, BLOCK_SIZE_D, BLOCK_SIZE_M, BLOCK_SIZE_N
    )


@triton.jit
def attn_beginner(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    M, N,
    d: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr
):
    pid_m = tl.program_id(0)
    row_indices = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)

    # ========================================================================
    # STEP 2: Load the Q block for this program
    # ========================================================================
    # Q has shape [M, d], stored in row-major order.
    # Element Q[row, col] is at offset: row * d + col
    # We compute all offsets at once using broadcasting:
    #   row_indices[:, None] has shape [BLOCK_SIZE_M, 1]
    #   col_indices[None, :] has shape [1, BLOCK_SIZE_D]
    #   => Q_offset has shape [BLOCK_SIZE_M, BLOCK_SIZE_D]
    col_indices = tl.arange(0, BLOCK_SIZE_D)
    Q_offset = row_indices[:, None] * d + col_indices[None, :]

    # We need masks because M and d might not be multiples of BLOCK_SIZE.
    # Without masks, we'd read past the array boundary.
    valid_col = col_indices < d                # which columns actually exist
    valid_row = row_indices < M                # which rows actually exist
    Q_mask = valid_row[:, None] & valid_col[None, :]

    # Load Q block. Out-of-bounds positions are filled with 0.0 (won't affect results
    # because we'll mask the final store).
    Q_block = tl.load(Q_ptr + Q_offset, mask=Q_mask, other=0.0)
    # Q_block shape: [BLOCK_SIZE_M, BLOCK_SIZE_D]

    # ========================================================================
    # STEP 3: Initialize online softmax running statistics
    # ========================================================================
    # These are per-row statistics (hence shape [BLOCK_SIZE_M, 1] for broadcasting).
    #
    # running_max: the maximum attention score seen so far for each row.
    #   Start at -inf so any real score will be larger.
    running_max = tl.full((BLOCK_SIZE_M, 1), float("-inf"), dtype=tl.float32)

    # running_sum: the sum of exp(score - running_max) for all scores seen so far.
    #   This is the "unnormalized" denominator of softmax.
    #   Start at 0 because we haven't seen any scores yet.
    running_sum = tl.zeros((BLOCK_SIZE_M, 1), dtype=tl.float32)

    # acc: the accumulated numerator of softmax * V.
    #   i.e., acc = Σ [exp(score - running_max) / running_sum] * V
    #   Shape: [BLOCK_SIZE_M, BLOCK_SIZE_D] because V has d columns.
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_D), dtype=tl.float32)

    # Scaling factor: 1/sqrt(d) as in standard scaled dot-product attention
    scale = d ** -0.5

    # ========================================================================
    # STEP 4: Main loop — iterate over K/V in blocks along the N dimension
    # ========================================================================
    # K and V have shape [N, d]. We process BLOCK_SIZE_N rows at a time.
    # Total iterations = ceil(N / BLOCK_SIZE_N)

    # Pre-compute the base offset for the first K/V block.
    # K[row, col] is at offset: row * d + col
    # row indices for the first block: [0, 1, ..., BLOCK_SIZE_N-1]
    # col indices: [0, 1, ..., BLOCK_SIZE_D-1]
    kv_row_indices = tl.arange(0, BLOCK_SIZE_N)
    KV_base_offset = kv_row_indices[:, None] * d + col_indices[None, :]

    num_blocks = tl.cdiv(N, BLOCK_SIZE_N)

    for block_idx in tl.range(0, num_blocks):
        # ---- 4a: Load current K and V blocks ----
        # Mask: which rows in this block are valid (not past the end of N)?
        rows_remaining = N - block_idx * BLOCK_SIZE_N
        valid_kv_row = kv_row_indices < rows_remaining
        KV_mask = valid_kv_row[:, None] & valid_col[None, :]

        # The actual memory offset for this block shifts by BLOCK_SIZE_N * d each iteration.
        # This is because each "row" of K/V takes up `d` contiguous memory cells.
        KV_offset = KV_base_offset + block_idx * BLOCK_SIZE_N * d

        K_block = tl.load(K_ptr + KV_offset, mask=KV_mask, other=0.0)
        V_block = tl.load(V_ptr + KV_offset, mask=KV_mask, other=0.0)
        # K_block shape: [BLOCK_SIZE_N, BLOCK_SIZE_D]
        # V_block shape: [BLOCK_SIZE_N, BLOCK_SIZE_D]

        # ---- 4b: Compute attention scores for this block ----
        # Q_block @ K_block^T gives shape [BLOCK_SIZE_M, BLOCK_SIZE_N]
        # Each element (i,j) = dot(Q_row_i, K_row_j) = how much query i attends to key j
        scores = tl.dot(Q_block, K_block.T) * scale

        # Mask out invalid positions. Setting them to -inf means exp(-inf) = 0,
        # so they contribute nothing to the softmax.
        scores = tl.where(valid_kv_row[None, :], scores, float("-inf"))

        # ---- 4c: Update running max ----
        old_max = running_max
        # New max = max of old max and the max score in the current block (per row)
        new_max = tl.maximum(old_max, scores.max(axis=-1, keep_dims=True))

        # ---- 4d: Update running sum ----
        # This is the key formula. Let's break it down:
        #
        # old_sum was computed relative to old_max:
        #   old_sum = Σ exp(old_scores - old_max)
        #
        # We need to rescale it to be relative to new_max:
        #   rescaled_old_sum = old_sum * exp(old_max - new_max)
        #
        # Then add the contribution from the new block:
        #   new_block_sum = Σ exp(new_scores - new_max)
        #
        # Total:
        #   new_sum = rescaled_old_sum + new_block_sum
        old_sum = running_sum
        rescaled_old_sum = old_sum * tl.exp(old_max - new_max)
        new_block_sum = tl.exp(scores - new_max).sum(axis=-1, keep_dims=True)
        new_sum = rescaled_old_sum + new_block_sum

        # ---- 4e: Update the accumulated result ----
        #
        # The current acc was computed with old normalization:
        #   acc = Σ [exp(old_scores - old_max) / old_sum] * V_old
        #
        # We need to:
        #   1. Rescale the old acc to account for the new max and new sum:
        #        acc *= rescaled_old_sum / new_sum
        #      (This shrinks old contributions because new_sum is larger,
        #       and rescales for the max change)
        #
        #   2. Add the new block's contribution:
        #        acc += [exp(scores - new_max) / new_sum] @ V_block
        #      (This is the weighted sum for the current K/V block)
        #
        # Combined update:
        acc = acc * (rescaled_old_sum / new_sum)
        acc = acc + tl.dot(tl.exp(scores - new_max) / new_sum, V_block)

        # ---- 4f: Commit the new running statistics ----
        running_max = new_max
        running_sum = new_sum

    # ========================================================================
    # STEP 5: Write the final result back to global memory
    # ========================================================================
    # After the loop, acc contains the fully normalized attention output.
    # acc[i, j] = Σ_k softmax(Q[i] @ K[k] / sqrt(d)) * V[k, j]
    tl.store(O_ptr + Q_offset, acc, mask=Q_mask)
    
