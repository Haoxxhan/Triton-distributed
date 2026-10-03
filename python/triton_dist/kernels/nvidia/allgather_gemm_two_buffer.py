# SPDX-License-Identifier: MIT
"""Forward-only, intra-node ring AG GEMM with two reusable A shards.

Each rank owns [2, M_per_rank, K], rather than [world_size * M_per_rank, K].
The CUDA copy stream forwards a shard while a Triton GEMM reads the same shard.
A third stream releases the slot only after BOTH readers have finished.

This is an explicit, event-based reference implementation. It deliberately uses
one GEMM launch per shard with a manually selected tile configuration.
The GEMM is not persistent.
Initialize the repository's distributed/NVSHMEM runtime before creating a context.
All ranks must create, invoke and finalize contexts in the same order.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from triton_dist.kernels.nvidia.common_ops import _memcpy_async_cuda, _set_signal_cuda, _wait_eq_cuda
from triton_dist.utils import (
    get_triton_dist_local_world_size,
    get_triton_dist_world,
    nvshmem_barrier_all_on_stream,
    nvshmem_create_tensors,
    nvshmem_free_tensor_sync,
)


# Signals are int32 CUDA stream memory operations, not spinning compute CTAs.
READY = 0
CREDIT = 1

FIXED_GEMM_CONFIG = triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3)


@triton.jit(do_not_specialize=["output_row_start"])
def _slot_gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    shard_m: tl.constexpr, n: tl.constexpr, k: tl.constexpr,
    stride_bk: tl.constexpr, stride_bn: tl.constexpr,
    output_row_start,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    USE_TMA: tl.constexpr = False,
    USE_HOST_DESC: tl.constexpr = False, a_desc_arg=None, b_desc_arg=None,
):
    """Read rows starting at zero in a slot, write rows for the original rank."""
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    ks = tl.arange(0, BLOCK_K)
    # Use 64-bit offsets for matrix addressing even when row/column IDs fit i32.
    a_offsets = rows.to(tl.int64)[:, None] * k + ks[None, :]
    b_offsets = ks.to(tl.int64)[:, None] * stride_bk + cols.to(tl.int64)[None, :] * stride_bn
    if USE_TMA:
        if USE_HOST_DESC:
            a_desc = a_desc_arg
            b_desc = b_desc_arg
        else:
            a_desc = tl.make_tensor_descriptor(a_ptr, shape=[shard_m, k], strides=[k, 1],
                                               block_shape=[BLOCK_M, BLOCK_K])
            if stride_bk == 1:
                # B is backed by contiguous [N, K] storage, as in the old GEMM.
                b_desc = tl.make_tensor_descriptor(b_ptr, shape=[n, k], strides=[stride_bn, 1],
                                                   block_shape=[BLOCK_N, BLOCK_K])
            else:
                b_desc = tl.make_tensor_descriptor(b_ptr, shape=[k, n], strides=[stride_bk, 1],
                                                   block_shape=[BLOCK_K, BLOCK_N])
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for ki in range(tl.cdiv(k, BLOCK_K)):
        valid_k = ki * BLOCK_K + ks < k
        if USE_TMA:
            a = a_desc.load([tl.program_id(0) * BLOCK_M, ki * BLOCK_K])
            if stride_bk == 1:
                b = b_desc.load([tl.program_id(1) * BLOCK_N, ki * BLOCK_K]).T
            else:
                b = b_desc.load([ki * BLOCK_K, tl.program_id(1) * BLOCK_N])
        else:
            a = tl.load(a_ptr + a_offsets, (rows[:, None] < shard_m) & valid_k[None, :], other=0)
            b = tl.load(b_ptr + b_offsets, valid_k[:, None] & (cols[None, :] < n), other=0)
        acc = tl.dot(a, b, acc)
        a_offsets += BLOCK_K
        b_offsets += BLOCK_K * stride_bk

    global_rows = tl.cast(output_row_start, tl.int64) + rows.to(tl.int64)
    c_offsets = global_rows[:, None] * n + cols[None, :]
    tl.store(c_ptr + c_offsets, acc.to(c_ptr.dtype.element_ty),
             (rows[:, None] < shard_m) & (cols[None, :] < n))


@dataclass
class TwoBufferAGGEMMContext:
    rank: int
    world_size: int
    shard_m: int
    k: int
    n: int
    dtype: torch.dtype
    device: torch.device
    peer_buffers: List[torch.Tensor]
    peer_signals: List[torch.Tensor]
    send_stream: torch.cuda.Stream
    release_stream: torch.cuda.Stream
    compute_done: List[torch.cuda.Event]
    send_done: List[torch.cuda.Event]
    prepared: torch.cuda.Event
    finished: torch.cuda.Event
    epoch: int = 0
    has_pending_call: bool = False
    closed: bool = False
    gemm_config: Optional[triton.Config] = None
    use_tma: bool = True
    use_host_desc: bool = True
    a_descriptors: list = field(default_factory=list)
    a_descriptor_key: tuple = ()
    b_descriptor: Optional[TensorDescriptor] = None
    b_descriptor_key: tuple = ()

    @property
    def buffers(self):
        return self.peer_buffers[self.rank]

    @property
    def signals(self):
        return self.peer_signals[self.rank]

    @property
    def workspace_bytes(self):
        """A storage only; excludes signals, caller inputs, B and output C."""
        return self.buffers.numel() * self.buffers.element_size()

    def finalize(self):
        """Collective teardown; call before finalize_distributed()."""
        if self.closed:
            return
        with torch.cuda.device(self.device):
            stream = torch.cuda.current_stream()
            if self.has_pending_call:
                stream.wait_event(self.finished)
            # Other ranks must also finish their peer accesses before we free.
            nvshmem_barrier_all_on_stream(stream)
            stream.synchronize()
            # Descriptors hold tensor references: release them after readers
            # finish and before freeing symmetric storage.
            self.a_descriptors.clear()
            self.b_descriptor = None
            self.b_descriptor_key = ()
            local_buffers, local_signals = self.buffers, self.signals
            nvshmem_free_tensor_sync(local_signals)
            nvshmem_free_tensor_sync(local_buffers)
        self.peer_buffers.clear()
        self.peer_signals.clear()
        self.closed = True


def create_two_buffer_ag_gemm_context(shard_m: int, k: int, n: int,
                                      dtype: torch.dtype = torch.bfloat16, *,
                                      gemm_config: Optional[triton.Config] = None,
                                      use_tma: bool = True,
                                      use_host_desc: bool = True) -> TwoBufferAGGEMMContext:
    """Collective creation on the initialized WORLD, one rank per local GPU.

    ``n`` is the LOCAL output feature count, unlike the full N accepted by the
    original create_ag_gemm_context. All ranks must use the same sizes/dtype.
    Requires intra-node peer mappings and CUDA stream memory operations.
    Defaults to host-created TMA descriptors. Use use_host_desc=False for device
    descriptors, or use_tma=False for ordinary masked loads. TMA requires aligned
    row strides; it does not silently fall back for unsupported layouts.
    """
    if min(shard_m, k, n) <= 0:
        raise ValueError("shard_m, k and n must be positive")
    if dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("This reference kernel supports float16 and bfloat16")
    if gemm_config is None:
        gemm_config = FIXED_GEMM_CONFIG
    blocks = gemm_config.kwargs
    if set(blocks) != {"BLOCK_M", "BLOCK_N", "BLOCK_K"}:
        raise ValueError("gemm_config must contain BLOCK_M, BLOCK_N and BLOCK_K")
    if any(not isinstance(v, int) or v < 16 or v & (v - 1) for v in blocks.values()):
        raise ValueError("GEMM block sizes must be powers of two >= 16")
    if use_tma and (torch.cuda.get_device_capability()[0] < 9 or k % 8):
        raise ValueError("TMA requires SM90+ and a 16-byte-aligned A row stride (K multiple of 8)")
    if use_tma and not use_host_desc:
        # Device-created descriptors need launch scratch, even without the old
        # AG GEMM (which installs its own allocator as a side effect).
        def alloc_fn(size, alignment, stream):
            launch_stream = torch.cuda.ExternalStream(stream) if stream is not None else torch.cuda.current_stream()
            with torch.cuda.stream(launch_stream):
                storage = torch.empty(size + alignment, dtype=torch.uint8, device="cuda")
                offset = (-storage.data_ptr()) % alignment
                return storage[offset:offset + size]
        triton.set_allocator(alloc_fn)
    group = get_triton_dist_world()
    if group is None or not torch.distributed.is_initialized():
        raise RuntimeError("Call initialize_distributed() before creating the context")
    rank, world_size = group.rank(), group.size()
    if world_size != get_triton_dist_local_world_size():
        raise ValueError("Two-buffer v1 supports one node only")
    if world_size != torch.distributed.get_world_size() or rank != torch.distributed.get_rank():
        raise ValueError("Two-buffer v1 requires the full WORLD rank ordering")

    device = torch.device("cuda", torch.cuda.current_device())
    # This small metadata collective prevents mismatched symmetric allocations.
    metadata = (shard_m, k, n, str(dtype), device.index)
    all_metadata = [None] * world_size
    torch.distributed.all_gather_object(all_metadata, metadata, group=group)
    if any(item[:4] != metadata[:4] for item in all_metadata):
        raise ValueError("All ranks must use matching shard_m, k, n and dtype")
    if [item[4] for item in all_metadata] != list(range(world_size)):
        raise ValueError("Use the standard launcher: local CUDA device index must equal rank")
    if world_size > 1:
        previous, following = (rank - 1) % world_size, (rank + 1) % world_size
        accessible = all(torch.cuda.can_device_access_peer(device.index, peer) for peer in (previous, following))
        checks = [None] * world_size
        torch.distributed.all_gather_object(checks, accessible, group=group)
        if not all(checks):
            raise ValueError("Every adjacent ring pair needs CUDA P2P access")

    peer_buffers = nvshmem_create_tensors((2, shard_m, k), dtype, rank, world_size)
    # signals[READY, slot]: local data generation; signals[CREDIT, slot]:
    # permission received from NEXT rank to write that slot on NEXT rank.
    peer_signals = nvshmem_create_tensors((2, 2), torch.int32, rank, world_size)
    peer_signals[rank].zero_()
    nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
    torch.cuda.current_stream().synchronize()
    ctx = TwoBufferAGGEMMContext(
        rank, world_size, shard_m, k, n, dtype, device, peer_buffers, peer_signals,
        torch.cuda.Stream(), torch.cuda.Stream(),
        [torch.cuda.Event() for _ in range(world_size)],
        [torch.cuda.Event() for _ in range(world_size)],
        torch.cuda.Event(), torch.cuda.Event(),
        gemm_config=gemm_config, use_tma=use_tma, use_host_desc=use_host_desc,
    )
    if use_tma and use_host_desc:
        _prepare_a_descriptors(ctx)
    return ctx


def _prepare_a_descriptors(ctx):
    config = ctx.gemm_config.kwargs
    key = (config["BLOCK_M"], config["BLOCK_K"])
    if ctx.a_descriptor_key != key:
        ctx.a_descriptors = [TensorDescriptor(ctx.buffers[slot], [ctx.shard_m, ctx.k], [ctx.k, 1], list(key))
                             for slot in range(2)]
        ctx.a_descriptor_key = key


def prepare_two_buffer_host_descriptors(ctx, b):
    """Cache both A slots and only the most recently used B (bounded retention).

    Input validation belongs to ag_gemm_two_buffer. Reusing these Python objects
    still lets Triton's eager launcher encode the tensor maps on each launch.
    """
    _prepare_a_descriptors(ctx)
    config = ctx.gemm_config.kwargs
    key = (b.data_ptr(), tuple(b.shape), tuple(b.stride()), b.dtype,
           config["BLOCK_N"], config["BLOCK_K"])
    if ctx.b_descriptor_key != key:
        if b.stride(0) == 1:
            ctx.b_descriptor = TensorDescriptor(b, [ctx.n, ctx.k], [b.stride(1), 1],
                                                 [config["BLOCK_N"], config["BLOCK_K"]])
        else:
            ctx.b_descriptor = TensorDescriptor(b, [ctx.k, ctx.n], [b.stride(0), 1],
                                                 [config["BLOCK_K"], config["BLOCK_N"]])
        ctx.b_descriptor_key = key
    return ctx.a_descriptors, ctx.b_descriptor


def ag_gemm_two_buffer(
    a: torch.Tensor,
    b: torch.Tensor,
    ctx: TwoBufferAGGEMMContext,
    *,
    out: Optional[torch.Tensor] = None,
    compute_delay_cycles: int = 0,
    send_delay_cycles: int = 0,
) -> torch.Tensor:
    """C = concat(A from all ranks) @ local B, with two A shard slots.

    Calls on a context must be host-serialized and collective. Different caller
    streams are supported; the context joins its previous invocation explicitly.
    A and B may have non-contiguous strides. An optional output must be contiguous
    and must not alias A, B or the symmetric slots. This is forward-only; use
    torch.no_grad() for tensors requiring gradients. CUDA graph capture is not
    supported by this first version. Delay options are for manual race checks,
    in CUDA clock cycles, and should be zero for normal use.
    """
    if ctx.closed:
        raise RuntimeError("The context has been finalized")
    if a.shape != (ctx.shard_m, ctx.k) or b.shape != (ctx.k, ctx.n):
        raise ValueError("Expected A[shard_m, k] and B[k, local_n] matching the context")
    if a.device != ctx.device or b.device != ctx.device or torch.cuda.current_device() != ctx.device.index:
        raise ValueError("A, B and the current CUDA device must match the context")
    if a.dtype != ctx.dtype or b.dtype != ctx.dtype:
        raise ValueError("A and B must match the context dtype")
    if torch.is_grad_enabled() and (a.requires_grad or b.requires_grad):
        raise ValueError("Forward-only prototype: use torch.no_grad(); backward is not implemented")
    if compute_delay_cycles < 0 or send_delay_cycles < 0:
        raise ValueError("Delay cycles must be nonnegative")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("CUDA graph capture is not supported by two-buffer v1")
    if ctx.epoch + ctx.world_size + 4 >= 2**31:
        raise RuntimeError("Signal generation exhausted; collectively recreate the context")

    output_shape = (ctx.world_size * ctx.shard_m, ctx.n)
    if out is None:
        out = torch.empty(output_shape, dtype=ctx.dtype, device=ctx.device)
    elif out.shape != output_shape or out.dtype != ctx.dtype or out.device != ctx.device or not out.is_contiguous():
        raise ValueError("out must be a contiguous tensor with shape [world_size * shard_m, local_n]")
    if out.requires_grad:
        raise ValueError("out must not require gradients")
    # Storage comparison also catches overlapping views with different data_ptrs.
    out_storage = out.untyped_storage().data_ptr()
    if any(out_storage == t.untyped_storage().data_ptr() for t in (a, b, ctx.buffers)):
        raise ValueError("out must not share storage with A, B or the communication slots")
    if any(t.untyped_storage().data_ptr() == ctx.buffers.untyped_storage().data_ptr() for t in (a, b)):
        raise ValueError("A and B must not alias the reusable communication slots")

    if ctx.use_tma:
        bk, bn = b.stride()
        supported = (bk == 1 and bn % 8 == 0) or (bn == 1 and bk % 8 == 0)
        if not supported or b.data_ptr() % 16:
            raise ValueError("TMA B needs contiguous inner dimension and 16-byte-aligned pointer/outer stride")
    host_descriptors = ctx.use_tma and ctx.use_host_desc
    if host_descriptors:
        a_descriptors, b_descriptor = prepare_two_buffer_host_descriptors(ctx, b)
    gemm_config = ctx.gemm_config

    compute_stream = torch.cuda.current_stream()
    if ctx.has_pending_call:
        compute_stream.wait_event(ctx.finished)
    # Per-call rendezvous protects reinitialization from the previous call's
    # peer writes. There is NO global barrier between ring rounds.
    nvshmem_barrier_all_on_stream(compute_stream)
    base = ctx.epoch
    ctx.epoch += ctx.world_size + 4
    with torch.no_grad():
        ctx.buffers[0].copy_(a)
    _set_signal_cuda(ctx.signals[READY, 0], base + 1, compute_stream)
    # Slot 1 is initially empty on every rank. After the rendezvous this local
    # seed represents the next rank's initial credit for its slot 1.
    _set_signal_cuda(ctx.signals[CREDIT, 1], base + 2, compute_stream)
    nvshmem_barrier_all_on_stream(compute_stream)
    ctx.prepared.record(compute_stream)
    ctx.send_stream.wait_event(ctx.prepared)
    ctx.release_stream.wait_event(ctx.prepared)

    following = (ctx.rank + 1) % ctx.world_size
    previous = (ctx.rank - 1) % ctx.world_size
    shard_bytes = ctx.shard_m * ctx.k * a.element_size()
    grid = (triton.cdiv(ctx.shard_m, gemm_config.kwargs["BLOCK_M"]),
            triton.cdiv(ctx.n, gemm_config.kwargs["BLOCK_N"]))

    for step in range(ctx.world_size):
        slot = step % 2
        source_rank = (ctx.rank - step) % ctx.world_size
        generation = base + step + 1

        # Reader 1: all GEMM CTAs for this shard finish before compute_done.
        _wait_eq_cuda(ctx.signals[READY, slot], generation, compute_stream)
        if compute_delay_cycles:
            torch.cuda._sleep(compute_delay_cycles)
        _slot_gemm_kernel[grid](
            ctx.buffers[slot], b, out, ctx.shard_m, ctx.n, ctx.k,
            b.stride(0), b.stride(1), source_rank * ctx.shard_m,
            USE_TMA=ctx.use_tma, USE_HOST_DESC=host_descriptors,
            a_desc_arg=a_descriptors[slot] if host_descriptors else None,
            b_desc_arg=b_descriptor if host_descriptors else None,
            **gemm_config.all_kwargs(),
        )
        ctx.compute_done[step].record(compute_stream)

        # Reader 2: forward the same shard to NEXT rank's opposite slot.
        # CUDA copy completion (not merely host submission) protects the source.
        with torch.cuda.stream(ctx.send_stream):
            _wait_eq_cuda(ctx.signals[READY, slot], generation, ctx.send_stream)
            if step + 1 < ctx.world_size:
                next_slot = 1 - slot
                _wait_eq_cuda(ctx.signals[CREDIT, next_slot], generation + 1, ctx.send_stream)
                if send_delay_cycles:
                    torch.cuda._sleep(send_delay_cycles)
                _memcpy_async_cuda(ctx.peer_buffers[following][next_slot], ctx.buffers[slot],
                                   shard_bytes, ctx.send_stream)
                _set_signal_cuda(ctx.peer_signals[following][READY, next_slot],
                                 generation + 1, ctx.send_stream)
            ctx.send_done[step].record(ctx.send_stream)

        # Join BOTH readers before telling PREVIOUS rank it may overwrite us.
        # A separate stream is essential: neither reader is made to wait for
        # the other before it starts, and communication can overlap GEMM.
        ctx.release_stream.wait_event(ctx.compute_done[step])
        ctx.release_stream.wait_event(ctx.send_done[step])
        if ctx.world_size > 1:
            _set_signal_cuda(ctx.peer_signals[previous][CREDIT, slot],
                             generation + 2, ctx.release_stream)

    compute_stream.wait_stream(ctx.send_stream)
    compute_stream.wait_stream(ctx.release_stream)
    ctx.finished.record(compute_stream)
    ctx.has_pending_call = True
    # Preserve allocator lifetimes when inputs/output were allocated on another
    # stream. Callers still need to establish readiness of those input tensors.
    a.record_stream(compute_stream)
    b.record_stream(compute_stream)
    out.record_stream(compute_stream)
    return out
