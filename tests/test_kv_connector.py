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
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput
from vllm.v1.request import RequestStatus

from tests.test_block_scheduler import BLOCK_SIZE, _request
from tests.test_block_scheduler import _runner_output as _block_runner_output
from tests.test_block_scheduler import _scheduler as _real_scheduler
from tests.test_sample_time_grammar import _model_input, _sync_forward
from vllm_tt_plugin import kv_transfer
from vllm_tt_plugin.async_decode import AsyncTTModelRunnerOutput
from vllm_tt_plugin.kv_transfer import kv_cache_descriptions
from vllm_tt_plugin.model_runner import TTModelRunner
from vllm_tt_plugin.platform import _install_tt_nixl_single_rank_patch


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

    def handle_preemptions(self, kv_connector_metadata):
        self.calls.append("handle_preemptions")

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


def test_the_nixl_connector_stages_tt_kv_through_a_cpu_buffer(vllm_config):
    pytest.importorskip("nixl")
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.connector import (
        NixlConnector,
    )

    vllm_config.kv_transfer_config = KVTransferConfig(
        kv_connector="NixlConnector", kv_role="kv_producer", kv_buffer_device="cpu"
    )
    model_config = vllm_config.model_config
    spec = FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=model_config.get_total_num_kv_heads(),
        head_size=model_config.get_head_size(),
        dtype=model_config.dtype,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=4,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["model.layers.0.self_attn"], spec)],
    )
    # The platform config hook installs this in a server.
    _install_tt_nixl_single_rank_patch()

    k = FakePackedKVTensor(heads=spec.num_kv_heads // CHIPS)
    kv_tensors = {"model.layers.0.self_attn": (k, FakePackedKVTensor(k.shape[1]))}

    connector = NixlConnector(vllm_config, KVConnectorRole.WORKER, kv_cache_config)
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


def _scheduler_output(*, scheduled: bool) -> SchedulerOutput:
    output = SchedulerOutput.make_empty()
    output.kv_connector_metadata = FakeKVMetadata({"parked": [3]})
    if scheduled:
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

    assert (
        TTModelRunner.execute_model(runner, _scheduler_output(scheduled=True)) is None
    )
    output = TTModelRunner.sample_tokens(runner, None)
    if async_decode:
        output = output.get_output()

    assert calls == [
        "handle_preemptions",
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


def test_a_zero_token_step_still_polls_the_connector(vllm_config, worker_connector):
    # A request parked on a remote load schedules no tokens. With nothing else
    # running, this step is the only one that can report its load as finished.
    worker_connector.finished_recving = {"parked"}
    runner = _runner(vllm_config, worker_connector.calls, async_decode=False)

    output = TTModelRunner.execute_model(runner, _scheduler_output(scheduled=False))

    assert worker_connector.calls == [
        "handle_preemptions",
        "bind",
        "start_load_kv",
        "wait_for_save",
        "get_finished",
        "clear",
    ]
    assert worker_connector.loads_started == {"parked": [3]}
    assert output.kv_connector_output.finished_recving == {"parked"}
    assert not runner._pending_samples


def test_a_step_without_a_connector_reports_no_connector_output(vllm_config):
    runner = _runner(vllm_config, [], async_decode=False)

    assert (
        TTModelRunner.execute_model(runner, _scheduler_output(scheduled=True)) is None
    )
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
    scheduler.update_from_output(prefill, _decode_output(prefill))
    return scheduler


def _decode_output(scheduler_output, **connector_results) -> ModelRunnerOutput:
    output = _block_runner_output(scheduler_output, [7])
    if connector_results:
        output.kv_connector_output = KVConnectorOutput(**connector_results)
    return output


def test_a_remote_load_is_not_lost_when_the_step_falls_back_to_decode():
    scheduler = _scheduler_with_running_decode()
    # The connector holds B's first block; B computes the rest itself.
    scheduler.connector.external_tokens["B"] = BLOCK_SIZE
    request = _request(8, request_id="B", prompt_length=BLOCK_SIZE + 40)
    scheduler.add_request(request)

    # The prefill pass parks B and schedules nothing, so A's decode runs.
    step = scheduler.schedule()

    assert set(step.num_scheduled_tokens) == {"A"}
    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    (b_block_ids,) = scheduler.kv_cache_manager.get_block_ids("B")
    assert step.kv_connector_metadata.loads == {"B": b_block_ids}

    scheduler.update_from_output(step, _decode_output(step, finished_recving={"B"}))
    step = scheduler.schedule()

    assert step.num_scheduled_tokens == {"B": 40}


def _initialize_kv_cache(layer_names: list[str], kv_caches: list) -> None:
    """``TTModelRunner.initialize_kv_cache`` over a model holding ``kv_caches``."""
    spec = FullAttentionSpec(
        block_size=BLOCK_SIZE, num_kv_heads=1, head_size=1, dtype=torch.float32
    )
    runner = SimpleNamespace(
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
            num_blocks=4,
            kv_cache_tensors=[],
            kv_cache_groups=[KVCacheGroupSpec(layer_names, spec)],
        ),
    )


class FakeTTKVTensor:
    """One layer's K or V on device, as (blocks, heads, tokens, dim)."""

    dtype = "bfloat16"

    def __init__(self):
        self.blocks = torch.zeros(4, 1, BLOCK_SIZE, 1)
        self.shape = self.blocks.shape


@pytest.fixture
def fake_tt_kv(monkeypatch):
    """Block reads and writes land on ``FakeTTKVTensor`` instead of a device."""
    monkeypatch.setattr(
        kv_transfer, "_read_block", lambda cache, block_id: cache.blocks[block_id]
    )

    def write_block(cache, block_id, values):
        cache.blocks[block_id] = values

    monkeypatch.setattr(kv_transfer, "_write_block", write_block)


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


def test_packed_kv_moves_between_two_servers_as_bytes(fake_chunk_transfers):
    """bfloat8_b KV crosses the CPU buffers as the device's own bytes."""
    spec = SimpleNamespace(
        num_kv_heads=CHIPS, block_size=BLOCK_SIZE, head_size=32, dtype=torch.bfloat16
    )
    kv_cache_config = SimpleNamespace(
        num_blocks=PACKED_BLOCKS,
        kv_cache_groups=[SimpleNamespace(kv_cache_spec=spec, layer_names=LAYER_NAMES)],
    )

    def server():
        kv_tensors = {
            name: (FakePackedKVTensor(), FakePackedKVTensor()) for name in LAYER_NAMES
        }
        descriptions = kv_cache_descriptions(kv_cache_config, kv_tensors)
        # NixlConnector's buffer: tokens before heads in memory, viewed logically.
        cpu_buffer = {
            name: torch.zeros(
                (d.shape[0], d.shape[2], d.shape[1], d.shape[3]), dtype=d.dtype
            ).permute(0, 2, 1, 3)
            for name, d in descriptions.items()
        }
        return kv_tensors, cpu_buffer, kv_transfer.make_copy_blocks(kv_tensors)

    sent, sender_buffer, sender_copy = server()
    received, receiver_buffer, receiver_copy = server()
    rng = np.random.default_rng(0)
    for k, v in sent.values():
        for chip in (*k.chips, *v.chips):
            chip[:] = rng.integers(0, 256, chip.size, dtype=np.uint8)

    # A stretch that counts up, one that counts down, and a block on its own.
    sent_ids = [0, 1, 2, 9, 8, 7, 5]
    received_ids = [4, 5, 6, 11, 10, 9, 0]
    sender_copy(None, sender_buffer, sent_ids, sent_ids, "d2h")
    for name in LAYER_NAMES:
        receiver_buffer[name][received_ids] = sender_buffer[name][sent_ids]
    receiver_copy(receiver_buffer, None, received_ids, received_ids, "h2d")

    for name in LAYER_NAMES:
        for sent_cache, received_cache in zip(sent[name], received[name]):
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


def _model_kv_caches(num_layers: int = 2) -> list:
    return [[FakeTTKVTensor(), FakeTTKVTensor()] for _ in range(num_layers)]


LAYER_NAMES = ["model.layers.1.self_attn", "model.layers.0.self_attn"]


def test_the_connector_is_handed_a_description_of_each_layers_kv(worker_connector):
    layer_names = LAYER_NAMES

    _initialize_kv_cache(layer_names, _model_kv_caches())

    assert list(worker_connector.kv_caches) == layer_names
    for description in worker_connector.kv_caches.values():
        # (num_blocks, num_kv_heads, block_size, K and V of head_size 1)
        assert description.shape == (4, 1, BLOCK_SIZE, 2)
        assert description.dtype == torch.float32
        # One element stands in for the whole shape.
        assert description.untyped_storage().nbytes() == 4


@pytest.mark.parametrize("one_list_per_submesh", [False, True])
def test_kv_moves_between_two_servers_through_their_cpu_buffers(
    worker_connector, fake_tt_kv, one_list_per_submesh
):
    """The copy a host-staging connector is given, on a sender and a receiver."""

    def server():
        kv_caches = _model_kv_caches()
        _initialize_kv_cache(
            LAYER_NAMES, [kv_caches] if one_list_per_submesh else kv_caches
        )
        descriptions = worker_connector.kv_caches
        cpu_buffer = {
            name: torch.zeros(description.shape, dtype=description.dtype)
            for name, description in descriptions.items()
        }
        return kv_caches, descriptions, cpu_buffer, worker_connector.copy_blocks

    sent, sender_descriptions, sender_buffer, sender_copy = server()
    received, receiver_descriptions, receiver_buffer, receiver_copy = server()
    for k, v in sent:
        k.blocks.normal_()
        v.blocks.normal_()

    sent_ids, received_ids = [0, 1, 3], [1, 2, 0]
    sender_copy(sender_descriptions, sender_buffer, sent_ids, sent_ids, "d2h")
    # The connector moves buffer blocks between servers; block IDs differ there.
    for name in LAYER_NAMES:
        receiver_buffer[name][received_ids] = sender_buffer[name][sent_ids]
    receiver_copy(
        receiver_buffer, receiver_descriptions, received_ids, received_ids, "h2d"
    )

    # The buffer holds K and V side by side, per layer name.
    k1, v1 = sent[1]
    assert torch.equal(sender_buffer[LAYER_NAMES[0]][1, ..., :1], k1.blocks[1])
    assert torch.equal(sender_buffer[LAYER_NAMES[0]][1, ..., 1:], v1.blocks[1])
    for (sent_k, sent_v), (received_k, received_v) in zip(sent, received):
        assert torch.equal(received_k.blocks[received_ids], sent_k.blocks[sent_ids])
        assert torch.equal(received_v.blocks[received_ids], sent_v.blocks[sent_ids])
        assert not received_k.blocks[3].any()


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
    if not another_request_is_decoding:
        scheduler.finish_requests("A", RequestStatus.FINISHED_ABORTED)
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
