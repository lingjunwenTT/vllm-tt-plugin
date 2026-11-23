# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""What ``TTModelRunner`` gives a worker-side KV connector."""

import torch
import ttnn
from vllm.distributed.kv_transfer.kv_connector.v1.base import CopyBlocksOp
from vllm.v1.kv_cache_interface import KVCacheConfig


def kv_cache_descriptions(kv_cache_config: KVCacheConfig) -> dict[str, torch.Tensor]:
    """A torch tensor per layer name that describes the layer's KV cache.

    Each has vLLM's logical KV shape ``(num_blocks, num_kv_heads, block_size,
    2 * head_size)``, K and V side by side in the last dimension, and the
    spec's dtype. No KV is behind it: it is one element expanded to that
    shape. The KV itself is in the model's ttnn tensors, which a connector
    written against torch cannot read.
    """
    descriptions = {}
    for group in kv_cache_config.kv_cache_groups:
        # ``UniformTypeKVCacheSpecs`` wraps a spec per layer.
        per_layer_specs = getattr(group.kv_cache_spec, "kv_cache_specs", None)
        for name in group.layer_names:
            spec = per_layer_specs[name] if per_layer_specs else group.kv_cache_spec
            shape = (
                kv_cache_config.num_blocks,
                spec.num_kv_heads,
                spec.block_size,
                2 * spec.head_size,
            )
            descriptions[name] = torch.empty(1, dtype=spec.dtype).expand(shape)
    return descriptions


def kv_tensors_by_layer_name(
    kv_caches, layer_index_by_name: dict[str, int]
) -> dict[str, tuple]:
    """The model's ``(k, v)`` ttnn tensors for each ``kv_cache_config`` layer name.

    ``kv_caches`` is what the model's allocator returned: ``[k, v]`` per layer,
    or that list inside a list with one entry per submesh. A KV connector runs
    without data parallelism, so there is one submesh.
    """
    per_layer = kv_caches
    if (
        isinstance(per_layer, (list, tuple))
        and len(per_layer) == 1
        and isinstance(per_layer[0], (list, tuple))
        and isinstance(per_layer[0][0], (list, tuple))
    ):
        (per_layer,) = per_layer
    if not isinstance(per_layer, (list, tuple)) or len(per_layer) != len(
        layer_index_by_name
    ):
        raise ValueError(
            "A KV connector needs one [k, v] KV cache entry per kv_cache_config "
            f"layer name; got layer names {list(layer_index_by_name)} and "
            f"kv_caches={kv_caches!r}."
        )
    kv_tensors = {}
    for name, index in layer_index_by_name.items():
        k, v = per_layer[index]
        kv_tensors[name] = (k, v)
    return kv_tensors


def _read_block(cache, block_id: int) -> torch.Tensor:
    """One block of a paged ttnn KV cache as ``(heads, block_size, head_dim)``.

    Each chip of the mesh holds its own share of the heads; they are joined in
    chip order.
    """
    block = cache[block_id : block_id + 1]
    per_chip = [ttnn.to_torch(chip) for chip in ttnn.get_device_tensors(block)]
    ttnn.deallocate(block)
    return torch.cat(per_chip, dim=1)[0]


def _write_block(cache, block_id: int, values: torch.Tensor) -> None:
    """Write ``(heads, block_size, head_dim)`` into one block of a KV cache.

    The heads are split evenly over the chips of the mesh, in chip order.
    """
    device = cache.device()
    block = ttnn.from_torch(
        values.unsqueeze(0),
        dtype=cache.dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        mesh_mapper=ttnn.ShardTensorToMesh(device, dim=1),
    )
    page_table = ttnn.from_torch(
        torch.tensor([[block_id]], dtype=torch.int32),
        dtype=ttnn.int32,
        device=device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )
    # The block is a one-block sequence whose page table names where it goes.
    ttnn.experimental.paged_fill_cache(cache, block, page_table, batch_idx=0)
    ttnn.deallocate(block)
    ttnn.deallocate(page_table)


def make_copy_blocks(kv_tensors: dict[str, tuple]) -> CopyBlocksOp:
    """The copy between the model's ttnn KV cache and a connector's CPU buffer.

    The buffer is the connector's own: per layer name, a CPU tensor shaped
    like ``kv_cache_descriptions``. The connector's "device" dict holds those
    descriptions and is ignored; ``kv_tensors`` is the real device side.
    """

    def copy_blocks(src, dst, src_block_ids, dst_block_ids, direction) -> None:
        for name, (k, v) in kv_tensors.items():
            head_dim = k.shape[-1]
            for src_id, dst_id in zip(src_block_ids, dst_block_ids):
                if direction == "d2h":
                    dst[name][dst_id] = torch.cat(
                        [_read_block(k, src_id), _read_block(v, src_id)], dim=-1
                    )
                else:
                    block = src[name][src_id].float()
                    _write_block(k, dst_id, block[..., :head_dim])
                    _write_block(v, dst_id, block[..., head_dim:])

    return copy_blocks
