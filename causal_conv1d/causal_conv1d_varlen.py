import torch
from torch import Tensor

import triton
import triton.language as tl


@triton.jit
def _causal_conv1d_varlen_states(
    X,
    CU_SEQLENS,
    INITIAL_STATES,
    NEW_STATES,
    state_len,
    dim,
    stride_x_seqlen, stride_x_dim,
    stride_initial_states_batch, stride_initial_states_seqlen, stride_initial_states_dim,
    stride_new_states_batch, stride_new_states_seqlen, stride_new_states_dim,
    has_initial_states: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr
):
    """
    Triton kernel to update sliding window states for variable length sequences.

    For each sequence in the batch, this kernel computes the new state based on an initial
    state and new input data.

    If `has_initial_states` is True and the length of the new data (`current_seq_len`)
    is less than `state_len`, the new state is formed by taking the last
    `state_len - current_seq_len` elements of the `INITIAL_STATES` and appending
    the `current_seq_len` elements of new data from `X`. This is equivalent to a
    left-shift of the state window.

    Otherwise (if `has_initial_states` is False, or if `current_seq_len` is >= `state_len`),
    the state is created from scratch from `X`. The last `min(current_seq_len, state_len)`
    elements of the sequence from `X` are copied to the end of the `NEW_STATES` tensor.
    """
    batch_idx = tl.program_id(2)

    # Pointers for the current batch item
    new_states_batch_ptr = NEW_STATES + batch_idx * stride_new_states_batch

    # Sequence information
    end_idx = tl.load(CU_SEQLENS + batch_idx + 1)
    start_idx = tl.load(CU_SEQLENS + batch_idx)
    current_seq_len = end_idx - start_idx

    # Parallelize over state_len (BLOCK_M) and dim (BLOCK_N)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(0)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rows_in_state = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    # Common mask for bounds checking
    mask_common = (rows_in_state[:, None] < state_len) & (cols[None, :] < dim)

    # Case 1: Shift operation (requires initial states and short new sequence)
    if has_initial_states and current_seq_len < state_len:
        initial_states_batch_ptr = INITIAL_STATES + batch_idx * stride_initial_states_batch
        shift = current_seq_len
        break_point = state_len - shift  # The index where we switch from copying initial_state to new data.

        # --- Part 1: Load from initial_states ---
        # Corresponds to a left-shift: new_state[:break_point] = initial_state[shift:]
        rows_in_initial_state = rows_in_state[:, None] + shift
        initial_states_ptr = initial_states_batch_ptr + rows_in_initial_state * stride_initial_states_seqlen + cols[None, :] * stride_initial_states_dim

        mask_initial = mask_common & (rows_in_state[:, None] < break_point)
        vals_from_initial = tl.load(initial_states_ptr, mask=mask_initial, other=0.0)

        # --- Part 2: Load from X ---
        # Append new data: new_state[break_point:] = x[:]
        rows_in_x = start_idx + (rows_in_state[:, None] - break_point)
        x_ptr = X + rows_in_x * stride_x_seqlen + cols[None, :] * stride_x_dim

        mask_x = mask_common & (rows_in_state[:, None] >= break_point)
        vals_from_x = tl.load(x_ptr, mask=mask_x, other=0.0)

        # --- Combine and store ---
        # Use tl.where to select between the shifted initial state and the new data.
        final_vals = tl.where(rows_in_state[:, None] < break_point, vals_from_initial, vals_from_x)
        new_states_ptr = new_states_batch_ptr + rows_in_state[:, None] * stride_new_states_seqlen + cols[None, :] * stride_new_states_dim
        tl.store(new_states_ptr, final_vals, mask=mask_common)

    # Case 2: Create state from scratch
    else:
        # This handles:
        # - No initial_states provided.
        # - initial_states provided, but new sequence is long enough to fill the state.
        len_to_copy = tl.minimum(current_seq_len, state_len)

        # Start positions for reading from X and writing to NEW_STATES
        x_read_start = end_idx - len_to_copy
        state_write_start = state_len - len_to_copy

        # Map `rows_in_state` to the actual read/write indices
        rows_in_x = x_read_start + (rows_in_state[:, None] - state_write_start)

        # Mask for the relevant part of the state tensor
        mask_load_store = mask_common & (rows_in_state[:, None] >= state_write_start)

        x_ptr = X + rows_in_x * stride_x_seqlen + cols[None, :] * stride_x_dim
        new_states_ptr = new_states_batch_ptr + rows_in_state[:, None] * stride_new_states_seqlen + cols[None, :] * stride_new_states_dim

        x_vals = tl.load(x_ptr, mask=mask_load_store, other=0.0)

        # Store the loaded values. The Python wrapper ensures NEW_STATES is zero-initialized,
        # so we only write the new data from X.
        tl.store(new_states_ptr, x_vals, mask=mask_load_store)


def causal_conv1d_varlen_states(x: Tensor, cu_seqlens: Tensor, state_len: int, initial_states: Tensor = None) -> Tensor:
    """
    Updates sliding window states for variable length sequences.

    Forward pass only, does not support backward pass.
    Parameters:
        x: (total_tokens, dim). Input tensor containing concatenated sequences.
        cu_seqlens: (batch + 1). Cumulative sequence lengths.
        state_len: int. The length of the state window.
        initial_states: (batch, dim, state_len), optional. The previous states. If None,
                        the new states are created from scratch from x.
    Return:
        new_states: (batch, dim, state_len). The updated states.
    """
    _, dim = x.shape
    batch = cu_seqlens.shape[0] - 1

    has_initial_states = initial_states is not None

    if has_initial_states:
        assert initial_states.ndim == 3, f"initial_states must be 3D, got {initial_states.ndim}"
        assert initial_states.shape[0] == batch, f"Batch size mismatch: initial_states has {initial_states.shape[0]}, expected {batch}"
        assert initial_states.shape[1] == dim, f"Dimension mismatch: initial_states has {initial_states.shape[1]}, expected {dim}"
        assert initial_states.shape[2] == state_len, f"State length mismatch: initial_states has {initial_states.shape[2]}, expected {state_len}"
        initial_states = initial_states.contiguous()
        new_states = torch.empty_like(initial_states)
    else:
        # If no initial states, create a zero tensor for the output. This is the expected
        # behavior for the "from scratch" case.
        shape = (batch, dim, state_len)
        new_states = torch.zeros(shape, dtype=x.dtype, device=x.device)
        # The kernel expects a tensor for initial_states. We pass a dummy one (new_states itself),
        # but the `has_initial_states` flag ensures it's never actually read from.
        initial_states = new_states

    cu_seqlens = cu_seqlens.contiguous()

    BLOCK_M = min(triton.next_power_of_2(state_len), 16)
    BLOCK_N = min(triton.next_power_of_2(dim), 64)
    grid = (triton.cdiv(dim, BLOCK_N), triton.cdiv(state_len, BLOCK_M), batch)

    with torch.cuda.device(x.device.index):
        _causal_conv1d_varlen_states[grid](
            x,
            cu_seqlens,
            initial_states,
            new_states,
            state_len,
            dim,
            x.stride(0), x.stride(1),
            initial_states.stride(0), initial_states.stride(2), initial_states.stride(1),
            new_states.stride(0), new_states.stride(2), new_states.stride(1),
            has_initial_states=has_initial_states,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N
        )
    return new_states


def causal_conv1d_varlen_states_ref(x: Tensor, cu_seqlens: Tensor, state_len: int, initial_states: Tensor = None) -> Tensor:
    """
    Reference implementation for updating sliding window states.

    Forward pass only, does not support backward pass.
    Parameters:
        x: (total_tokens, dim). Input tensor containing concatenated sequences.
        cu_seqlens: (batch + 1). Cumulative sequence lengths.
        state_len: int. The length of the state window.
        initial_states: (batch, dim, state_len), optional. The previous states. If None,
                        the new states are created from scratch from x.
    Return:
        new_states: (batch, dim, state_len). The updated states.
    """
    _, dim = x.shape
    batch = cu_seqlens.shape[0] - 1

    if initial_states is not None:
        assert initial_states.shape == (batch, dim, state_len)
        new_states = torch.empty_like(initial_states)

        for i in range(batch):
            end_idx = cu_seqlens[i + 1]
            start_idx = cu_seqlens[i]
            current_seq_len = end_idx - start_idx

            if current_seq_len >= state_len:
                # New sequence is long, just take the last part.
                new_states[i] = x[end_idx - state_len : end_idx].T
            else:
                # New sequence is short, shift and append.
                shift = current_seq_len
                break_point = state_len - shift

                # Copy the latter part of the initial state.
                new_states[i, :, :break_point] = initial_states[i, :, shift:]

                # Append the new data.
                new_states[i, :, break_point:] = x[start_idx:end_idx].T
    else:
        # Original logic: create state from scratch, padding with zeros at the beginning.
        new_states = torch.zeros(batch, dim, state_len, dtype=x.dtype, device=x.device)
        for i in range(batch):
            end_idx = cu_seqlens[i + 1]
            start_idx = torch.maximum(cu_seqlens[i], end_idx - state_len)

            num_to_copy = end_idx - start_idx
            if num_to_copy > 0:
                new_states[i, :, -num_to_copy:] = x[start_idx:end_idx].T

    return new_states
