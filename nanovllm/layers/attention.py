import os
import torch
from torch import nn
import torch.nn.functional as F

# Try to import flashinfer first, then flash_attn, fallback to native PyTorch
DISABLE_FLASHINFER = os.getenv("NANOVLLM_DISABLE_FLASHINFER", "").lower() in {"1", "true", "yes", "on"}
DISABLE_FLASH_ATTN = os.getenv("NANOVLLM_DISABLE_FLASH_ATTN", "").lower() in {"1", "true", "yes", "on"}

if not DISABLE_FLASHINFER:
    try:
        from flashinfer import single_decode_with_kv_cache, single_prefill_with_kv_cache
        HAS_FLASHINFER = True
        print("Using flashinfer backend")
    except ImportError:
        HAS_FLASHINFER = False
        single_decode_with_kv_cache = None
        single_prefill_with_kv_cache = None
else:
    HAS_FLASHINFER = False
    single_decode_with_kv_cache = None
    single_prefill_with_kv_cache = None

if not DISABLE_FLASH_ATTN:
    try:
        from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
        HAS_FLASH_ATTN = True
        print("Using flash-attn backend")
    except ImportError:
        HAS_FLASH_ATTN = False
        flash_attn_varlen_func = None
        flash_attn_with_kvcache = None
else:
    HAS_FLASH_ATTN = False
    flash_attn_varlen_func = None
    flash_attn_with_kvcache = None

from nanovllm.utils.context import get_context


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache

        # Store KV to cache if cache exists
        if k_cache.numel() and v_cache.numel() and context.slot_mapping is not None:
            self._store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

        if HAS_FLASHINFER:
            return self._forward_flashinfer(q, k, v, context)
        elif HAS_FLASH_ATTN:
            return self._forward_flash_attn(q, k, v, context)
        else:
            return self._forward_torch(q, k, v, context)

    def _store_kvcache(self, k, v, k_cache, v_cache, slot_mapping):
        """Store key-value pairs to cache using slot mapping"""
        # Flatten cache if needed: (num_blocks, block_size, num_kv_heads, head_dim) -> (num_blocks * block_size, num_kv_heads, head_dim)
        if k_cache.dim() == 4:
            k_cache = k_cache.reshape(-1, k_cache.shape[-2], k_cache.shape[-1])
            v_cache = v_cache.reshape(-1, v_cache.shape[-2], v_cache.shape[-1])
            # Update self.k_cache/v_cache reference for next call
            self.k_cache_flat = k_cache
            self.v_cache_flat = v_cache
        elif hasattr(self, 'k_cache_flat'):
            k_cache = self.k_cache_flat
            v_cache = self.v_cache_flat

        # Keep KV writes graph-safe by avoiding any host-side branching on CUDA tensors.
        slots = slot_mapping.long()
        k_cache.index_copy_(0, slots, k)
        v_cache.index_copy_(0, slots, v)

    def _forward_flashinfer(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, context):
        """FlashInfer implementation (preferred for performance)"""
        wrapper = getattr(context, "flashinfer_wrapper", None)
        if wrapper is not None:
            return wrapper.run(q, (self.k_cache, self.v_cache))
        if (not context.is_prefill) and torch.cuda.is_current_stream_capturing():
            return self._forward_capture_decode(q, context)

        k_cache, v_cache = self.k_cache, self.v_cache

        if context.is_prefill:
            # Prefill phase
            if k_cache.numel() and context.block_tables is not None:
                # Use cache for prefix
                k = k_cache.reshape(-1, k_cache.shape[-2], k_cache.shape[-1])
                v = v_cache.reshape(-1, v_cache.shape[-2], v_cache.shape[-1])

            # Use single_prefill_with_kv_cache for simplicity
            # For varlen, we need to process each sequence separately
            cu_seqlens_q = context.cu_seqlens_q
            cu_seqlens_k = context.cu_seqlens_k
            batch_size = cu_seqlens_q.shape[0] - 1

            outputs = []
            for i in range(batch_size):
                q_start, q_end = cu_seqlens_q[i].item(), cu_seqlens_q[i + 1].item()
                k_start, k_end = cu_seqlens_k[i].item(), cu_seqlens_k[i + 1].item()

                q_i = q[q_start:q_end]  # (seq_len_q, num_heads, head_dim)

                if k_cache.numel() and context.block_tables is not None:
                    # Get KV from cache
                    block_table = context.block_tables[i]
                    k_i, v_i = self._get_kv_from_cache(block_table, k_cache, v_cache, k_end - k_start)
                else:
                    k_i = k[k_start:k_end]
                    v_i = v[k_start:k_end]

                # Reshape for flashinfer: (seq_len, num_heads, head_dim) -> (seq_len, num_heads, head_dim)
                o = single_prefill_with_kv_cache(
                    q_i, k_i, v_i,
                    causal=True,
                    sm_scale=self.scale
                )
                outputs.append(o)

            return torch.cat(outputs, dim=0)
        else:
            # Decode phase
            # Use decode_with_kv_cache
            k_cache_flat = k_cache.reshape(-1, k_cache.shape[-2], k_cache.shape[-1])
            v_cache_flat = v_cache.reshape(-1, v_cache.shape[-2], v_cache.shape[-1])

            context_lens = context.context_lens
            block_tables = context.block_tables

            batch_size = q.shape[0]
            outputs = []

            for i in range(batch_size):
                ctx_len = context_lens[i].item() if context_lens is not None else 1
                q_i = q[i]  # (num_heads, head_dim)

                if block_tables is not None:
                    block_table = block_tables[i]
                    k_i, v_i = self._get_kv_from_cache(block_table, k_cache, v_cache, ctx_len)
                else:
                    k_i = k[:ctx_len]
                    v_i = v[:ctx_len]

                # For decode, q has shape (num_heads, head_dim)
                # k, v have shape (seq_len, num_kv_heads, head_dim)
                o = single_decode_with_kv_cache(
                    q_i, k_i, v_i,
                    sm_scale=self.scale
                )
                outputs.append(o.unsqueeze(0))

            return torch.cat(outputs, dim=0)

    def _forward_capture_decode(self, q: torch.Tensor, context):
        """Graph-capture-safe decode path using only tensor ops."""
        k_cache = self.k_cache
        v_cache = self.v_cache
        if k_cache.dim() == 4:
            k_cache = k_cache.reshape(-1, k_cache.shape[-2], k_cache.shape[-1])
            v_cache = v_cache.reshape(-1, v_cache.shape[-2], v_cache.shape[-1])

        block_tables = context.block_tables.long()
        context_lens = context.context_lens.long()
        max_ctx_len = block_tables.size(1) * 256
        positions = torch.arange(max_ctx_len, device=q.device, dtype=torch.long)
        block_offsets = positions // 256
        token_offsets = positions % 256
        slots = block_tables[:, block_offsets].clamp_min_(0) * 256 + token_offsets

        k_seq = k_cache[slots]
        v_seq = v_cache[slots]
        if self.num_heads != self.num_kv_heads:
            repeat = self.num_heads // self.num_kv_heads
            k_seq = k_seq.repeat_interleave(repeat, dim=2)
            v_seq = v_seq.repeat_interleave(repeat, dim=2)

        q = q.unsqueeze(2)
        k_seq = k_seq.permute(0, 2, 1, 3)
        v_seq = v_seq.permute(0, 2, 1, 3)

        scores = torch.matmul(q, k_seq.transpose(-1, -2)) * self.scale
        valid = positions.unsqueeze(0) < context_lens.unsqueeze(1)
        scores = scores.masked_fill(~valid[:, None, None, :], float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        out = torch.matmul(probs, v_seq)
        return out.squeeze(2)

    def _get_kv_from_cache(self, block_table, k_cache, v_cache, seq_len):
        """Get KV cache for a sequence from block table"""
        k_cache_flat = k_cache.reshape(-1, k_cache.shape[-2], k_cache.shape[-1])
        v_cache_flat = v_cache.reshape(-1, v_cache.shape[-2], v_cache.shape[-1])

        block_size = 256
        valid_blocks = block_table[block_table >= 0].long()
        if valid_blocks.numel() == 0:
            return k_cache_flat[:seq_len], v_cache_flat[:seq_len]

        positions = torch.arange(seq_len, device=block_table.device, dtype=torch.long)
        block_offsets = positions // block_size
        token_offsets = positions % block_size
        slots = valid_blocks[block_offsets] * block_size + token_offsets
        return k_cache_flat[slots], v_cache_flat[slots]

    def _forward_flash_attn(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, context):
        """FlashAttention implementation"""
        k_cache, v_cache = self.k_cache, self.v_cache
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables,
                                        softmax_scale=self.scale, causal=True)
        return o

    def _forward_torch(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, context):
        """Native PyTorch attention implementation (fallback)"""
        num_heads = self.num_heads
        num_kv_heads = self.num_kv_heads
        head_dim = self.head_dim

        if context.is_prefill:
            # Prefill phase: process all tokens
            cu_seqlens_q = context.cu_seqlens_q
            cu_seqlens_k = context.cu_seqlens_k

            if cu_seqlens_q is None or cu_seqlens_k is None:
                # No varlen info, treat as single sequence
                return self._attention_single_seq(q, k, v)

            batch_size = cu_seqlens_q.shape[0] - 1

            # Process each sequence separately
            outputs = []
            for i in range(batch_size):
                q_start, q_end = cu_seqlens_q[i].item(), cu_seqlens_q[i + 1].item()
                k_start, k_end = cu_seqlens_k[i].item(), cu_seqlens_k[i + 1].item()

                q_i = q[q_start:q_end]  # (seq_len_q, num_heads, head_dim)
                k_i = k[k_start:k_end]  # (seq_len_k, num_kv_heads, head_dim)
                v_i = v[k_start:k_end]  # (seq_len_k, num_kv_heads, head_dim)

                outputs.append(self._attention_single_seq(q_i, k_i, v_i))

            o = torch.cat(outputs, dim=0)

        else:
            # Decode phase: one token per sequence
            batch_size = q.shape[0]
            context_lens = context.context_lens
            block_tables = context.block_tables

            # Get flattened cache
            k_cache = self.k_cache
            v_cache = self.v_cache
            if k_cache.dim() == 4:
                k_cache = k_cache.reshape(-1, k_cache.shape[-2], k_cache.shape[-1])
                v_cache = v_cache.reshape(-1, v_cache.shape[-2], v_cache.shape[-1])

            outputs = []
            for i in range(batch_size):
                ctx_len = context_lens[i].item() if context_lens is not None else 1

                # Get KV for this sequence
                if block_tables is not None:
                    block_table = block_tables[i]
                    k_seq = []
                    v_seq = []
                    for block_idx, block_id in enumerate(block_table):
                        if block_id < 0:
                            break
                        block_start = block_id * 256  # block_size
                        num_tokens = min(256, ctx_len - block_idx * 256)
                        if num_tokens <= 0:
                            break
                        for offset in range(num_tokens):
                            slot = block_start + offset
                            k_seq.append(k_cache[slot])
                            v_seq.append(v_cache[slot])

                    if k_seq:
                        k_i = torch.stack(k_seq, dim=0)[:ctx_len]
                        v_i = torch.stack(v_seq, dim=0)[:ctx_len]
                    else:
                        k_i = k[:1]
                        v_i = v[:1]
                else:
                    k_i = k[:1]
                    v_i = v[:1]

                outputs.append(self._attention_single_seq(q[i:i+1], k_i, v_i))

            o = torch.cat(outputs, dim=0)

        return o

    def _attention_single_seq(self, q, k, v):
        """Compute attention for a single sequence"""
        # q: (seq_len_q, num_heads, head_dim)
        # k: (seq_len_k, num_kv_heads, head_dim)
        # v: (seq_len_k, num_kv_heads, head_dim)

        seq_len_q = q.shape[0]
        seq_len_k = k.shape[0]
        num_heads = self.num_heads
        num_kv_heads = self.num_kv_heads
        head_dim = self.head_dim

        # Reshape for SDPA: (batch=1, num_heads, seq_len, head_dim)
        q = q.transpose(0, 1).unsqueeze(0)  # (1, num_heads, seq_q, head_dim)
        k = k.transpose(0, 1).unsqueeze(0)  # (1, num_kv_heads, seq_k, head_dim)
        v = v.transpose(0, 1).unsqueeze(0)  # (1, num_kv_heads, seq_k, head_dim)

        # Handle GQA: expand KV heads
        if num_heads != num_kv_heads:
            n_rep = num_heads // num_kv_heads
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)

        # Create causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len_q, seq_len_k, device=q.device, dtype=torch.bool),
            diagonal=seq_len_k - seq_len_q + 1
        )

        # Apply SDPA
        o = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=causal_mask,
            scale=self.scale
        )

        # Output: (1, num_heads, seq_q, head_dim) -> (seq_q, num_heads, head_dim)
        return o.squeeze(0).transpose(0, 1)
