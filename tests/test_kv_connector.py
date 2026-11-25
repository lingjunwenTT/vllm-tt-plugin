# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The KV connector lifecycle on the single-process TT path.

No bytes move here. A fake connector records what the plugin asks of it and
reports the transfers each test scripts.
"""

import threading
from collections import deque
from functools import partial
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.config.kv_transfer import KVTransferConfig
from vllm.distributed.kv_transfer import kv_transfer_state
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import RequestStatus

from tests.test_block_scheduler import BLOCK_SIZE, _request
from tests.test_block_scheduler import _runner_output as _block_runner_output
from tests.test_block_scheduler import _scheduler as _real_scheduler
from tests.test_sample_time_grammar import _model_input, _sync_forward
from vllm_tt_plugin import kv_transfer
from vllm_tt_plugin.async_decode import AsyncTTModelRunnerOutput
from vllm_tt_plugin.kv_transfer import kv_cache_descriptions
from vllm_tt_plugin.model_runner import TTModelRunner
from vllm_tt_plugin.platform import (
    TTPlatform,
    _install_tt_nixl_dram_registration_patch,
    _install_tt_nixl_single_rank_patch,
)


class FakeKVMetadata(KVConnectorMetadata):
    def __init__(self, loads: dict[str, list[int]]):
        # Request ID to the block IDs its external KV should be written into.
        self.loads = loads


class FakeKVConnector(KVConnectorBase_V1, SupportsHMA):
    """Both halves of a connector, with the transfers scripted by the test."""

    def __init__(self, vllm_config, role, kv_cache_config):
        super().__init__(vllm_config, role, kv_cache_config)
        self.calls: list[str] = []
        # Scheduler half: tokens of external KV to offer per request ID.
        self.external_tokens: dict[str, int] = {}
        self.pending_loads: dict[str, list[int]] = {}
        # Either half: finished requests it was told of. The scheduler half hears
        # when building metadata, the worker half when asked what has finished.
        self.finished_requests: set[str] = set()
        # Worker half: loads it was told to start, and what to report next.
        self.loads_started: dict[str, list[int]] = {}
        self.finished_recving: set[str] = set()
        self.finished_sending: set[str] = set()
        self.kv_caches = None
        # Scheduler half: keep a finished request's blocks until its KV is sent.
        self.delay_free = False

    # Worker half.

    def bind_connector_metadata(self, connector_metadata):
        self.calls.append("bind")
        super().bind_connector_metadata(connector_metadata)

    def register_kv_caches(self, kv_caches):
        self.kv_caches = kv_caches

    def set_host_xfer_buffer_ops(self, copy_operation):
        self.copy_blocks = copy_operation

    def start_load_kv(self, forward_context, **kwargs):
        self.calls.append("start_load_kv")
        self.loads_started.update(self._get_connector_metadata().loads)

    def wait_for_layer_load(self, layer_name):
        raise AssertionError("TT never calls the per-layer connector hooks")

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        raise AssertionError("TT never calls the per-layer connector hooks")

    def wait_for_save(self):
        self.calls.append("wait_for_save")

    def get_finished(self, finished_req_ids):
        self.calls.append("get_finished")
        self.finished_requests |= finished_req_ids
        finished_sending, self.finished_sending = self.finished_sending, set()
        finished_recving, self.finished_recving = self.finished_recving, set()
        return finished_sending, finished_recving

    def clear_connector_metadata(self):
        self.calls.append("clear")
        super().clear_connector_metadata()

    # Scheduler half.

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        num_tokens = self.external_tokens.get(request.request_id, 0)
        return num_tokens, num_tokens > 0

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        if num_external_tokens > 0:
            (block_ids,) = blocks.get_block_ids()
            self.pending_loads[request.request_id] = block_ids

    def build_connector_meta(self, scheduler_output):
        self.finished_requests |= scheduler_output.finished_req_ids
        meta = FakeKVMetadata(self.pending_loads)
        self.pending_loads = {}
        return meta

    def request_finished_all_groups(self, request, block_ids):
        return self.delay_free, None


@pytest.fixture
def vllm_config():
    return _real_scheduler(output_width=1).vllm_config


@pytest.fixture
def worker_connector(monkeypatch, vllm_config):
    """The worker-side connector, installed where ``TTWorker`` would put it."""
    config = SimpleNamespace(kv_transfer_config=object())
    connector = FakeKVConnector(config, KVConnectorRole.WORKER, None)
    monkeypatch.setattr(kv_transfer_state, "_KV_CONNECTOR_AGENT", connector)
    return connector


def _nixl_connector(vllm_config, kv_buffer_device: str, num_blocks: int = 4):
    """A real worker-side NixlConnector for one attention layer, and its
    ``kv_cache_config``."""
    pytest.importorskip("nixl")
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.connector import (
        NixlConnector,
    )

    vllm_config.kv_transfer_config = KVTransferConfig(
        kv_connector="NixlConnector",
        kv_role="kv_producer",
        kv_buffer_device=kv_buffer_device,
    )
    model_config = vllm_config.model_config
    spec = FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=model_config.get_total_num_kv_heads(),
        head_size=model_config.get_head_size(),
        dtype=model_config.dtype,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["model.layers.0.self_attn"], spec)],
    )
    # The platform config hook installs these in a server.
    _install_tt_nixl_single_rank_patch()
    _install_tt_nixl_dram_registration_patch()
    connector = NixlConnector(vllm_config, KVConnectorRole.WORKER, kv_cache_config)
    return connector, kv_cache_config


def test_the_nixl_connector_takes_tt_as_a_buffer_device(vllm_config):
    connector, _ = _nixl_connector(vllm_config, "tt")
    try:
        worker = connector.connector_worker
        # No CPU buffer: NIXL is to transfer card memory, which its
        # Tenstorrent build calls VRAM.
        assert not worker.use_host_buffer
        assert worker.nixl_memory_type == "VRAM"
        # The worker hands the platform its NIXL device ID before a handshake.
        TTPlatform.set_device(worker.device_id)
    finally:
        connector.shutdown()


def test_the_nixl_connector_stages_tt_kv_through_a_cpu_buffer(vllm_config):
    connector, kv_cache_config = _nixl_connector(vllm_config, "cpu")
    spec = kv_cache_config.kv_cache_groups[0].kv_cache_spec
    k = FakePackedKVTensor(heads=spec.num_kv_heads // CHIPS)
    kv_tensors = {"model.layers.0.self_attn": (k, FakePackedKVTensor(k.shape[1]))}

    try:
        connector.register_kv_caches(kv_cache_descriptions(kv_cache_config, kv_tensors))

        # NIXL transfers its own CPU buffer; a block of it holds exactly the
        # packed K and V of that block on every chip, and is the buffer's
        # memory in block order.
        (buffer,) = connector.connector_worker.host_xfer_buffers.values()
        assert buffer.device.type == "cpu"
        assert buffer.shape[0] == kv_cache_config.num_blocks
        assert buffer[0].nbytes == 2 * CHIPS * k.block_bytes
        assert kv_transfer._buffer_bytes(buffer).nbytes == buffer.nbytes
    finally:
        connector.shutdown()


def _scheduler_output() -> SchedulerOutput:
    output = SchedulerOutput.make_empty()
    output.kv_connector_metadata = FakeKVMetadata({"parked": [3]})
    output.num_scheduled_tokens = {"req-0": 1}
    output.total_num_scheduled_tokens = 1
    return output


def _runner(vllm_config, calls: list[str], *, async_decode: bool) -> SimpleNamespace:
    """A runner whose forward and sampling are doubles, as in the grammar tests."""
    model_input = _model_input()
    scheduled: list[str] = []

    def build_model_input(scheduler_output, _grammar):
        scheduled[:] = scheduler_output.num_scheduled_tokens
        return model_input if scheduled else None

    def _runner_output(*_args, **_kwargs):
        return ModelRunnerOutput(
            req_ids=list(scheduled),
            req_id_to_index={req_id: i for i, req_id in enumerate(scheduled)},
            sampled_token_ids=[[7] for _ in scheduled],
            prompt_logprobs_dict={},
            pooler_output=[],
        )

    def forward(received_input):
        calls.append("forward")
        return _sync_forward(received_input)

    def submit_async_decode(received_input, **_kwargs):
        calls.append("forward")
        controller = SimpleNamespace(
            complete_decode_step=lambda **_kwargs: SimpleNamespace(),
            build_runner_output_from_completed_step=_runner_output,
            enqueue_completed_decode_step=lambda _done: None,
        )
        return AsyncTTModelRunnerOutput(
            controller, object(), received_input, threading.Event(), object()
        )

    runner = SimpleNamespace(
        vllm_config=vllm_config,
        async_decode=SimpleNamespace(
            can_attempt_steady_decode_from_scheduler=lambda _output: False,
            must_drain_pending_async_steps=lambda *_args: False,
            can_use_steady_decode_fast_path=lambda _input: False,
            submit_async_decode=submit_async_decode,
        ),
        async_decode_scheduling=async_decode,
        scheduler_config=SimpleNamespace(async_scheduling=async_decode),
        _pending_samples=deque(),
        build_model_input=build_model_input,
        _forward_with_model_input=forward,
        _sample_sync_forward=lambda _fwd: ([torch.tensor([[42]])], [None]),
        apply_and_build_runner_output=_runner_output,
        _num_speculative_tokens=0,
    )
    for name in (
        "_reorder_grammar_bitmask",
        "_apply_grammar_to_input",
        "_finish_front_packed_sync",
        "_finish_async_decode",
    ):
        setattr(runner, name, partial(getattr(TTModelRunner, name), runner))
    return runner


@pytest.mark.parametrize("async_decode", [False, True], ids=["sync", "async_decode"])
def test_a_forward_step_runs_inside_the_connector_step(
    vllm_config, worker_connector, async_decode
):
    worker_connector.finished_recving = {"parked"}
    calls = worker_connector.calls
    runner = _runner(vllm_config, calls, async_decode=async_decode)

    assert TTModelRunner.execute_model(runner, _scheduler_output()) is None
    output = TTModelRunner.sample_tokens(runner, None)
    if async_decode:
        output = output.get_output()

    assert calls == [
        "bind",
        "forward",
        "start_load_kv",
        "wait_for_save",
        "get_finished",
        "clear",
    ]
    assert worker_connector.loads_started == {"parked": [3]}
    assert output.req_ids == ["req-0"]
    assert output.kv_connector_output.finished_recving == {"parked"}


def test_an_async_step_reports_its_connector_output_without_a_readback(
    vllm_config, worker_connector
):
    # vLLM reads the connector output this way when ``get_output`` raised, so
    # finished transfers are not lost with a failed readback.
    worker_connector.finished_recving = {"parked"}
    runner = _runner(vllm_config, worker_connector.calls, async_decode=True)

    assert TTModelRunner.execute_model(runner, _scheduler_output()) is None
    output = TTModelRunner.sample_tokens(runner, None)

    assert output.get_kv_connector_output().finished_recving == {"parked"}


def test_a_step_without_a_connector_reports_no_connector_output(vllm_config):
    runner = _runner(vllm_config, [], async_decode=False)

    assert TTModelRunner.execute_model(runner, _scheduler_output()) is None
    output = TTModelRunner.sample_tokens(runner, None)

    assert output.kv_connector_output is None


def _scheduler_with_running_decode():
    """A two-slot scheduler with a connector, whose request ``A`` is decoding."""
    scheduler = _real_scheduler(
        output_width=1,
        max_num_seqs=2,
        kv_transfer_config=KVTransferConfig(
            kv_connector="FakeKVConnector",
            kv_connector_module_path=__name__,
            kv_role="kv_both",
        ),
    )
    scheduler.add_request(_request(8, request_id="A"))
    prefill = scheduler.schedule()
    scheduler.update_from_output(prefill, _block_runner_output(prefill, [7]))
    return scheduler


def _initialize_kv_cache(
    layer_names: list[str], kv_caches: list, kv_buffer_device: str = "cpu"
) -> None:
    """``TTModelRunner.initialize_kv_cache`` over a model holding ``kv_caches``."""
    spec = FullAttentionSpec(
        block_size=BLOCK_SIZE, num_kv_heads=CHIPS, head_size=32, dtype=torch.float32
    )
    runner = SimpleNamespace(
        vllm_config=SimpleNamespace(
            kv_transfer_config=SimpleNamespace(
                kv_buffer_device=kv_buffer_device,
                get_from_extra_config=lambda _key, default: default,
            )
        ),
        tt_max_batch_size=2,
        model_config=SimpleNamespace(max_model_len=BLOCK_SIZE),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=BLOCK_SIZE),
        cache_config=SimpleNamespace(block_size=BLOCK_SIZE),
        vocab_size=8,
        _is_lane_mode=False,
        _host_logitsprocs=None,
        _is_block_output_model=False,
        _output_tokens_per_step=1,
        _validate_kv_cache_groups=TTModelRunner._validate_kv_cache_groups,
        _allocate_kv_caches=lambda _kv_cache_config: kv_caches,
    )
    TTModelRunner.initialize_kv_cache(
        runner,
        KVCacheConfig(
            num_blocks=PACKED_BLOCKS,
            kv_cache_tensors=[],
            kv_cache_groups=[KVCacheGroupSpec(layer_names, spec)],
        ),
    )


CHIPS = 2
PACKED_BLOCKS = 12


class FakePackedKVTensor:
    """One layer's bfloat8_b K or V: each chip's buffer as bytes."""

    dtype = kv_transfer.ttnn.bfloat8_b

    def __init__(self, heads: int = 1):
        self.shape = (PACKED_BLOCKS, heads, BLOCK_SIZE, 32)
        self.block_bytes = heads * (BLOCK_SIZE // 32) * 1088
        self.chips = [
            np.zeros(PACKED_BLOCKS * self.block_bytes, dtype=np.uint8)
            for _ in range(CHIPS)
        ]

    def device(self):
        return SimpleNamespace(get_num_devices=lambda: CHIPS)


@pytest.fixture
def fake_chunk_transfers(monkeypatch):
    """ttnn's chunk transfers, as documented, over ``FakePackedKVTensor``."""

    def to_host(cache, offset, size, count, dst, dst_offset, chunk_step, chip_step):
        for chip, data in enumerate(cache.chips):
            for i in range(count):
                at = dst_offset + i * chunk_step + chip * chip_step
                dst[at : at + size] = data[offset + i * size :][:size]

    def to_device(src, src_offset, chunk_step, chip_step, cache, offset, size, count):
        for chip, data in enumerate(cache.chips):
            for i in range(count):
                at = src_offset + i * chunk_step + chip * chip_step
                data[offset + i * size :][:size] = src[at : at + size]

    monkeypatch.setattr(kv_transfer.ttnn, "copy_device_tensor_chunks_to_host", to_host)
    monkeypatch.setattr(
        kv_transfer.ttnn, "copy_host_chunks_to_device_tensor", to_device
    )


@pytest.mark.parametrize("one_list_per_submesh", [False, True])
def test_kv_moves_between_two_servers_through_their_cpu_buffers(
    worker_connector, fake_chunk_transfers, one_list_per_submesh
):
    """KV crosses a sender's and a receiver's CPU buffer as the device's bytes."""

    def server():
        kv_caches = _model_kv_caches()
        _initialize_kv_cache(
            LAYER_NAMES, [kv_caches] if one_list_per_submesh else kv_caches
        )
        # NixlConnector's buffer: tokens before heads in memory, viewed logically.
        cpu_buffer = {
            name: torch.zeros(
                (d.shape[0], d.shape[2], d.shape[1], d.shape[3]), dtype=d.dtype
            ).permute(0, 2, 1, 3)
            for name, d in worker_connector.kv_caches.items()
        }
        return kv_caches, cpu_buffer, worker_connector.copy_blocks

    sent, sender_buffer, sender_copy = server()
    received, receiver_buffer, receiver_copy = server()
    rng = np.random.default_rng(0)
    for k, v in sent:
        for chip in (*k.chips, *v.chips):
            chip[:] = rng.integers(0, 256, chip.size, dtype=np.uint8)

    # A stretch that counts up, one that counts down, and a block on its own.
    sent_ids = [0, 1, 2, 9, 8, 7, 5]
    received_ids = [4, 5, 6, 11, 10, 9, 0]
    sender_copy(None, sender_buffer, sent_ids, sent_ids, "d2h")
    # The connector moves buffer blocks between servers; block IDs differ there.
    for name in LAYER_NAMES:
        receiver_buffer[name][received_ids] = sender_buffer[name][sent_ids]
    receiver_copy(receiver_buffer, None, received_ids, received_ids, "h2d")

    for sent_layer, received_layer in zip(sent, received):
        for sent_cache, received_cache in zip(sent_layer, received_layer):
            size = sent_cache.block_bytes
            for sent_chip, received_chip in zip(sent_cache.chips, received_cache.chips):
                sent_blocks = sent_chip.reshape(PACKED_BLOCKS, size)
                received_blocks = received_chip.reshape(PACKED_BLOCKS, size)
                assert (received_blocks[received_ids] == sent_blocks[sent_ids]).all()
                # Blocks nothing was sent to stay empty.
                assert not received_blocks[[1, 2, 3, 7, 8]].any()


def test_a_long_stretch_of_blocks_is_copied_in_pieces():
    ids = list(range(3, 3 + 2 * kv_transfer._MAX_RUN_BLOCKS + 1))
    runs = list(kv_transfer._runs_of_consecutive_blocks(ids, ids))
    assert sum(count for _, _, count in runs) == len(ids)
    assert max(count for _, _, count in runs) == kv_transfer._MAX_RUN_BLOCKS


BANKS = 8


def _deal_tiles_over_banks(tiles: np.ndarray) -> list[np.ndarray]:
    """A DRAM tensor's tiles as ttnn stores them: tile ``i`` is in bank
    ``i % BANKS``, ``i // BANKS`` tiles into that bank's share."""
    return [tiles[bank::BANKS].reshape(-1) for bank in range(BANKS)]


def test_a_kv_block_is_one_byte_range_in_each_dram_bank():
    # 16 tiles a block: two in each of the 8 banks.
    cache = SimpleNamespace(shape=(PACKED_BLOCKS, 2, 64, 128))
    tiles = np.random.default_rng(0).integers(
        0, 256, (PACKED_BLOCKS * 16, 1088), dtype=np.uint8
    )
    banks = _deal_tiles_over_banks(tiles)

    piece = kv_transfer._bfloat8_block_bytes_per_bank(cache, BANKS)

    assert piece == 2 * 1088
    block = 5
    block_tiles = tiles[block * 16 : (block + 1) * 16]
    for bank, data in enumerate(banks):
        in_bank = data[block * piece : (block + 1) * piece]
        assert (in_bank == block_tiles[bank::BANKS].reshape(-1)).all()


def test_a_kv_block_that_does_not_divide_over_the_banks_is_refused():
    # 4 tiles a block.
    cache = SimpleNamespace(shape=(PACKED_BLOCKS, 1, 32, 128))
    with pytest.raises(ValueError, match="4 tiles does not divide evenly over 8"):
        kv_transfer._bfloat8_block_bytes_per_bank(cache, BANKS)


BANK_BYTES = 0xFF000000


class FakeDramKVTensor:
    """One layer's bfloat8_b K or V on one chip: 4 tiles a block in each bank."""

    dtype = kv_transfer.ttnn.bfloat8_b
    shape = (PACKED_BLOCKS, 4, 64, 128)

    def __init__(self, address: int, device_ids=(0,)):
        self.address = address
        self.device_ids = list(device_ids)

    def tiles_in(self, card_dram: dict) -> np.ndarray:
        """This tensor's tiles in ``card_dram``, a byte array per (card, bank),
        found the way ttnn finds them: tile ``i`` is in bank ``i % BANKS``,
        ``i // BANKS`` tiles past the buffer address."""
        (card,) = self.device_ids
        num_tiles = PACKED_BLOCKS * self.shape[1] * 2 * 4
        per_bank = [
            card_dram[card, bank][self.address :][: num_tiles // BANKS * 1088]
            for bank in range(BANKS)
        ]
        return np.stack([b.reshape(-1, 1088) for b in per_bank], axis=1)

    def fill(self, card_dram: dict, rng) -> None:
        """Write random tiles where ``tiles_in`` reads them."""
        (card,) = self.device_ids
        size = self.tiles_in(card_dram).size // BANKS
        for bank in range(BANKS):
            card_dram[card, bank][self.address :][:size] = rng.integers(
                0, 256, size, dtype=np.uint8
            )

    def buffer_address(self):
        return self.address

    def buffer_aligned_page_size(self):
        return 1088

    def device(self):
        return SimpleNamespace(get_device_ids=lambda: self.device_ids)


def _fake_cluster_descriptor(monkeypatch, tmp_path, card_by_device: dict) -> None:
    path = tmp_path / "cluster_descriptor.yaml"
    path.write_text(
        "chips_with_mmio:\n"
        + "".join(f"  - {d}: {card}\n" for d, card in card_by_device.items())
    )
    monkeypatch.setattr(
        kv_transfer.ttnn.cluster, "serialize_cluster_descriptor", lambda: str(path)
    )


@pytest.fixture
def fake_dram_banks(monkeypatch, tmp_path):
    """Every device has all of a Blackhole chip's DRAM banks in use, and is
    the card of the same number."""
    _fake_cluster_descriptor(monkeypatch, tmp_path, {d: d for d in range(8)})
    banks = [
        {"bank_id": bank, "base_addr": 0, "bank_size": BANK_BYTES}
        for bank in range(BANKS)
    ]
    monkeypatch.setattr(
        kv_transfer.ttnn.cluster, "get_dram_bank_table", lambda _device_id: banks
    )
    return banks


def test_a_layers_kv_is_a_dram_region_per_cache_and_bank(fake_dram_banks):
    k, v = FakeDramKVTensor(0x100000), FakeDramKVTensor(0x900000)

    layout = kv_transfer.kv_dram_layout({"layers.0": (k, v)}, PACKED_BLOCKS)

    block_bytes = 4 * 1088
    assert layout.regions["layers.0"] == [
        kv_transfer.DramRegion(0, bank << 32 | cache.address, block_bytes, block_bytes)
        for cache in (k, v)
        for bank in range(BANKS)
    ]
    # Both caches are in the same banks, and a bank is registered once, whole.
    assert layout.banks == [(bank << 32, BANK_BYTES, 0) for bank in range(BANKS)]


def test_a_devices_card_is_the_one_the_cluster_descriptor_names(
    fake_dram_banks, monkeypatch, tmp_path
):
    # As seen with /dev/tenstorrent/21 and /29 visible: ttnn numbers them in
    # PCI address order, which is not the order of their card numbers.
    _fake_cluster_descriptor(monkeypatch, tmp_path, {0: 29, 1: 21})
    k = FakeDramKVTensor(0x100000, device_ids=[1])
    v = FakeDramKVTensor(0x900000, device_ids=[1])

    layout = kv_transfer.kv_dram_layout({"layers.0": (k, v)}, PACKED_BLOCKS)

    assert {region.card for region in layout.regions["layers.0"]} == {21}
    assert {card for _, _, card in layout.banks} == {21}


def test_a_mesh_has_dram_regions_on_each_chips_card(fake_dram_banks):
    k = FakeDramKVTensor(0x100000, device_ids=[2, 5])
    v = FakeDramKVTensor(0x900000, device_ids=[2, 5])

    layout = kv_transfer.kv_dram_layout({"layers.0": (k, v)}, PACKED_BLOCKS)

    assert layout.regions["layers.0"] == [
        kv_transfer.DramRegion(card, bank << 32 | cache.address, 4 * 1088, 4 * 1088)
        for cache in (k, v)
        for card in (2, 5)
        for bank in range(BANKS)
    ]
    assert layout.banks == [
        (bank << 32, BANK_BYTES, card) for card in (2, 5) for bank in range(BANKS)
    ]


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda k, banks: setattr(k, "dtype", "bfloat16"), "needs a bfloat8_b"),
        (lambda k, banks: banks.pop(), "has 7 DRAM banks in use"),
        (lambda k, banks: setattr(k, "device_ids", [9]), "Device 9 is not a PCIe"),
        (
            lambda k, banks: setattr(k, "address", BANK_BYTES - 1088),
            "do not fit DRAM bank 0",
        ),
        (
            lambda k, banks: setattr(k, "buffer_aligned_page_size", lambda: 2048),
            "pages of 2048 bytes",
        ),
    ],
)
def test_kv_that_nixl_cannot_address_in_dram_is_refused(
    fake_dram_banks, change, message
):
    k, v = FakeDramKVTensor(0x100000), FakeDramKVTensor(0x900000)
    change(k, fake_dram_banks)
    with pytest.raises(ValueError, match=message):
        kv_transfer.kv_dram_layout({"layers.0": (k, v)}, PACKED_BLOCKS)


def test_kv_moves_between_two_servers_card_dram(worker_connector, fake_dram_banks):
    """What NIXL is asked to move, block for block and region for region, is
    the block's KV in the sender's tensors and in the receiver's."""
    card_dram = {
        (card, bank): np.zeros(0x80000, dtype=np.uint8)
        for card in (0, 1)
        for bank in range(BANKS)
    }

    def server(card: int, first_address: int):
        kv_caches = [
            [
                FakeDramKVTensor(first_address + 0x20000 * (2 * layer + i), [card])
                for i in range(2)
            ]
            for layer in range(len(LAYER_NAMES))
        ]
        _initialize_kv_cache(LAYER_NAMES, kv_caches, kv_buffer_device="tt")
        return kv_caches, worker_connector.kv_caches

    sent, sender_layout = server(card=0, first_address=0x1000)
    received, receiver_layout = server(card=1, first_address=0x3000)
    rng = np.random.default_rng(0)
    for cache in (cache for layer in sent for cache in layer):
        cache.fill(card_dram, rng)

    sent_ids = [0, 1, 2, 9, 8, 7, 5]
    received_ids = [4, 5, 6, 11, 10, 9, 0]
    # NIXL: the bytes of block ``b`` of a region are at ``addr + b * block_stride``.
    for name in LAYER_NAMES:
        for src, dst in zip(sender_layout.regions[name], receiver_layout.regions[name]):
            assert src.block_bytes == dst.block_bytes
            src_bank = card_dram[src.card, src.addr >> 32]
            dst_bank = card_dram[dst.card, dst.addr >> 32]
            for sent_id, received_id in zip(sent_ids, received_ids):
                src_at = (src.addr & 0xFFFFFFFF) + sent_id * src.block_stride
                dst_at = (dst.addr & 0xFFFFFFFF) + received_id * dst.block_stride
                dst_bank[dst_at:][: dst.block_bytes] = src_bank[src_at:][
                    : src.block_bytes
                ]

    for sent_layer, received_layer in zip(sent, received):
        for sent_cache, received_cache in zip(sent_layer, received_layer):
            sent_blocks = sent_cache.tiles_in(card_dram).reshape(PACKED_BLOCKS, -1)
            received_blocks = received_cache.tiles_in(card_dram).reshape(
                PACKED_BLOCKS, -1
            )
            assert sent_blocks[sent_ids].any(axis=1).all()
            assert (received_blocks[received_ids] == sent_blocks[sent_ids]).all()
            assert not received_blocks[[1, 2, 3, 7, 8]].any()
    # No CPU buffer copy is installed.
    assert not hasattr(worker_connector, "copy_blocks")


def test_one_chip_sends_kv_to_two_chips(fake_dram_banks):
    """A one-chip server that lists its KV in two parts names, region for
    region, what each chip of a two-chip server holds."""
    tile_bytes = 1088
    one_chip = FakeDramKVTensor(4 * tile_bytes, device_ids=[0])
    one_chip.shape = (PACKED_BLOCKS, 8, 64, 128)
    two_chips = FakeDramKVTensor(12 * tile_bytes, device_ids=[1, 2])  # 4 heads each
    sender = kv_transfer.kv_dram_layout({"l": (one_chip,)}, PACKED_BLOCKS, parts=2)
    receiver = kv_transfer.kv_dram_layout({"l": (two_chips,)}, PACKED_BLOCKS)

    # A block of the one-chip cache is 64 tiles, heads 0 to 7 in order. ttnn
    # deals them over the banks; number each tile so it can be followed.
    # Card memory here is one number per tile.
    block = 5
    card_dram = {
        (card, bank): np.zeros(256, dtype=np.int64)
        for card in (0, 1, 2)
        for bank in range(BANKS)
    }
    for tile in range(64):
        card_dram[0, tile % BANKS][4 + block * 8 + tile // BANKS] = 1000 + tile

    def tile_at(region, at):
        bank, address = region.addr >> 32, region.addr & 0xFFFFFFFF
        return card_dram[region.card, bank], (address + at) // tile_bytes

    assert len(sender.regions["l"]) == len(receiver.regions["l"]) == 2 * BANKS
    for src, dst in zip(sender.regions["l"], receiver.regions["l"]):
        assert src.block_bytes == dst.block_bytes == 4 * tile_bytes
        for offset in range(0, src.block_bytes, tile_bytes):
            src_bank, src_tile = tile_at(src, block * src.block_stride + offset)
            dst_bank, dst_tile = tile_at(dst, block * dst.block_stride + offset)
            dst_bank[dst_tile] = src_bank[src_tile]

    # Each chip of the receiver now holds its 4 heads of the block: 32 tiles,
    # dealt over its banks the same way.
    for chip, card in enumerate((1, 2)):
        for tile in range(32):
            in_bank = 12 + block * 4 + tile // BANKS
            assert card_dram[card, tile % BANKS][in_bank] == 1000 + 32 * chip + tile


def test_kv_that_cannot_be_listed_in_the_parts_asked_for_is_refused(fake_dram_banks):
    k = FakeDramKVTensor(0x1000, device_ids=[1, 2])
    with pytest.raises(ValueError, match="2 chip.*cannot be listed in 3 equal parts"):
        kv_transfer.kv_dram_layout({"l": (k,)}, PACKED_BLOCKS, parts=3)


class RecordingNixl:
    """Stands in for a connector worker's NIXL agent where it would touch a
    card: registering memory and preparing descriptor lists."""

    def __init__(self, monkeypatch, worker):
        self.registered: list = []
        self.prepared: dict[str, np.ndarray] = {}
        for name in (
            "get_reg_descs",
            "register_memory",
            "deregister_memory",
            "prep_xfer_dlist",
            "release_dlist_handle",
        ):
            monkeypatch.setattr(worker.nixl_wrapper, name, getattr(self, name))

    def get_reg_descs(self, ranges, mem_type):
        return (ranges, mem_type)

    def register_memory(self, descs, backends):
        self.registered.append(descs)

    def deregister_memory(self, descs):
        self.registered.remove(descs)

    def prep_xfer_dlist(self, agent_name, descs, mem_type):
        assert mem_type == "VRAM"
        self.prepared[agent_name] = descs
        return len(self.prepared)

    def release_dlist_handle(self, handle):
        pass


def _dram_layout(card: int = 0, num_blocks: int = 4) -> kv_transfer.KVDramLayout:
    k = FakeDramKVTensor(0x100000, device_ids=[card])
    v = FakeDramKVTensor(0x900000, device_ids=[card])
    return kv_transfer.kv_dram_layout({"model.layers.0.self_attn": (k, v)}, num_blocks)


def test_the_nixl_connector_registers_kv_in_card_dram(
    vllm_config, monkeypatch, fake_dram_banks
):
    connector, kv_cache_config = _nixl_connector(vllm_config, "tt")
    nixl = RecordingNixl(monkeypatch, connector.connector_worker)
    layout = _dram_layout(card=3, num_blocks=kv_cache_config.num_blocks)
    try:
        connector.register_kv_caches(layout)

        # Each bank once, whole, as card memory.
        assert nixl.registered == [
            ([(bank << 32, BANK_BYTES, 3, "") for bank in range(BANKS)], "VRAM")
        ]
        # A strided descriptor per region: K's 8 banks, then V's.
        regions = layout.regions["model.layers.0.self_attn"]
        block_bytes = 4 * 1088
        assert nixl.prepared["NIXL_INIT_AGENT"].tolist() == [
            [r.addr, block_bytes, 3, block_bytes, kv_cache_config.num_blocks]
            for r in regions
        ]
        # What the other server learns in the handshake.
        metadata = _handshake_metadata(connector)
        assert metadata.block_lens == [block_bytes] * 2 * BANKS
        assert metadata.region_mem_types == ["VRAM"] * 2 * BANKS
        # ... including each region's card, which the worker's one device ID
        # cannot say. This server asks NIXL for the same regions on the same
        # cards when it is the remote one.
        one_rank = SimpleNamespace(source_ranks_per_group=[[0]], rank_offset_factor=0)
        remote = connector.connector_worker._build_fa_remote(one_rank, metadata, 1)
        assert remote.tolist() == nixl.prepared["NIXL_INIT_AGENT"].tolist()
    finally:
        connector.shutdown()


def test_the_nixl_connector_names_the_card_of_each_region_of_a_mesh(
    vllm_config, monkeypatch, fake_dram_banks
):
    connector, kv_cache_config = _nixl_connector(vllm_config, "tt")
    nixl = RecordingNixl(monkeypatch, connector.connector_worker)
    k = FakeDramKVTensor(0x100000, device_ids=[2, 5])
    v = FakeDramKVTensor(0x900000, device_ids=[2, 5])
    layout = kv_transfer.kv_dram_layout(
        {"model.layers.0.self_attn": (k, v)}, kv_cache_config.num_blocks
    )
    try:
        connector.register_kv_caches(layout)

        local = nixl.prepared["NIXL_INIT_AGENT"]
        regions = layout.regions["model.layers.0.self_attn"]
        assert local[:, 0].tolist() == [r.addr for r in regions]
        assert local[:, 2].tolist() == [r.card for r in regions]
        one_rank = SimpleNamespace(source_ranks_per_group=[[0]], rank_offset_factor=0)
        remote = connector.connector_worker._build_fa_remote(
            one_rank, _handshake_metadata(connector), 1
        )
        assert remote.tolist() == local.tolist()
    finally:
        connector.shutdown()


def _handshake_metadata(connector):
    import msgspec
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
        NixlAgentMetadata,
    )

    payload = connector.connector_worker.xfer_handshake_metadata
    return msgspec.msgpack.decode(payload.agent_metadata_bytes, type=NixlAgentMetadata)


def _changed_attributes(worker, register) -> set[str]:
    def snapshot():
        return {
            name: (id(value), len(value) if hasattr(value, "__len__") else None)
            for name, value in vars(worker).items()
        }

    before = snapshot()
    register()
    after = snapshot()
    return {name for name in after if before.get(name) != after[name]}


def test_registering_card_dram_records_what_upstream_records_for_tensors(
    vllm_config, monkeypatch, fake_dram_banks
):
    """The DRAM registration replaces an upstream method that other upstream
    code reads the records of. If a vLLM update makes it record more, this
    fails."""
    upstream, kv_cache_config = _nixl_connector(vllm_config, "cpu")
    spec = kv_cache_config.kv_cache_groups[0].kv_cache_spec
    k = FakePackedKVTensor(heads=spec.num_kv_heads // CHIPS)
    kv_tensors = {"model.layers.0.self_attn": (k, FakePackedKVTensor(k.shape[1]))}
    ours, _ = _nixl_connector(vllm_config, "tt")
    RecordingNixl(monkeypatch, ours.connector_worker)
    try:
        by_upstream = _changed_attributes(
            upstream.connector_worker,
            lambda: upstream.register_kv_caches(
                kv_cache_descriptions(kv_cache_config, kv_tensors)
            ),
        )
        by_ours = _changed_attributes(
            ours.connector_worker,
            lambda: ours.register_kv_caches(
                _dram_layout(num_blocks=kv_cache_config.num_blocks)
            ),
        )
        # The CPU buffer itself is the only record card DRAM does not need.
        assert by_upstream - by_ours == {"host_xfer_buffers"}
    finally:
        upstream.shutdown()
        ours.shutdown()


def _model_kv_caches(num_layers: int = 2) -> list:
    return [[FakePackedKVTensor(), FakePackedKVTensor()] for _ in range(num_layers)]


LAYER_NAMES = ["model.layers.1.self_attn", "model.layers.0.self_attn"]


def test_the_connector_is_handed_a_description_of_each_layers_kv(worker_connector):
    layer_names = LAYER_NAMES

    _initialize_kv_cache(layer_names, _model_kv_caches())

    assert list(worker_connector.kv_caches) == layer_names
    for description in worker_connector.kv_caches.values():
        assert description.shape[:3] == (PACKED_BLOCKS, CHIPS, BLOCK_SIZE)
        assert description.dtype == torch.uint8
        # A block is the packed K and V of that block on every chip.
        assert description[0].nbytes == 2 * CHIPS * FakePackedKVTensor().block_bytes
        # One element stands in for the whole shape.
        assert description.untyped_storage().nbytes() == 1


def test_a_connector_refuses_a_kv_cache_that_is_not_bfloat8_b(
    worker_connector, monkeypatch
):
    monkeypatch.setattr(FakePackedKVTensor, "dtype", "bfloat16")
    with pytest.raises(ValueError, match="needs a bfloat8_b KV cache.*bfloat16"):
        _initialize_kv_cache(LAYER_NAMES, _model_kv_caches())


def test_a_connector_refuses_kv_it_cannot_map_to_layer_names(worker_connector):
    with pytest.raises(ValueError, match="one \\[k, v\\] KV cache entry per"):
        _initialize_kv_cache(LAYER_NAMES, _model_kv_caches(num_layers=3))


def _run_step(scheduler, runner) -> SchedulerOutput:
    """One engine step: schedule, run the forward, sample, report back."""
    step = scheduler.schedule()
    output = TTModelRunner.execute_model(runner, step)
    if output is None:
        output = TTModelRunner.sample_tokens(runner, None)
    scheduler.update_from_output(step, output)
    return step


@pytest.mark.parametrize("another_request_is_decoding", [False, True])
def test_a_request_loads_remote_kv_then_holds_its_blocks_until_sent(
    worker_connector, another_request_is_decoding
):
    """One request through the whole connector lifecycle, scheduler to runner."""
    scheduler = _scheduler_with_running_decode()
    runner = _runner(scheduler.vllm_config, [], async_decode=False)
    # C is aborted before the step, so the step must tell both halves about it.
    scheduler.add_request(_request(8, request_id="C"))
    aborted = {"C"} if another_request_is_decoding else {"A", "C"}
    scheduler.finish_requests(aborted, RequestStatus.FINISHED_ABORTED)
    scheduler.connector.external_tokens["B"] = BLOCK_SIZE
    scheduler.connector.delay_free = True
    request = _request(1, request_id="B", prompt_length=BLOCK_SIZE + 40)
    scheduler.add_request(request)

    # B is parked on its load. The instruction reaches the worker half, either
    # with A's decode or on a step that schedules no tokens at all.
    step = _run_step(scheduler, runner)
    scheduled = {"A"} if another_request_is_decoding else set()
    assert set(step.num_scheduled_tokens) == scheduled
    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    (b_block_ids,) = scheduler.kv_cache_manager.get_block_ids("B")
    assert worker_connector.loads_started == {"B": b_block_ids}
    assert scheduler.connector.finished_requests == aborted
    assert worker_connector.finished_requests == aborted

    # The worker half reports the load; the scheduler hears it on that step's
    # output and prefills the rest of B's prompt on the next one.
    worker_connector.finished_recving = {"B"}
    _run_step(scheduler, runner)
    step = _run_step(scheduler, runner)
    assert step.num_scheduled_tokens["B"] == 40

    # B's one token finished it, but the connector asked to send its KV first.
    assert request.status == RequestStatus.FINISHED_LENGTH_CAPPED
    block_pool = scheduler.kv_cache_manager.block_pool
    free_blocks_while_held = block_pool.get_num_free_blocks()
    assert "B" in scheduler.requests
    worker_connector.finished_sending = {"B"}
    _run_step(scheduler, runner)
    assert "B" not in scheduler.requests
    assert block_pool.get_num_free_blocks() == free_blocks_while_held + 2
