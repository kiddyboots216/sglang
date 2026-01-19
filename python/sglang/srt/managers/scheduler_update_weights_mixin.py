from __future__ import annotations

import logging
import traceback
from typing import TYPE_CHECKING, Tuple

import torch

from sglang.srt.constants import (
    GPU_MEMORY_ALL_TYPES,
    GPU_MEMORY_TYPE_CUDA_GRAPH,
    GPU_MEMORY_TYPE_KV_CACHE,
    GPU_MEMORY_TYPE_WEIGHTS,
)
from sglang.srt.managers.io_struct import (
    CheckWeightsReqInput,
    CheckWeightsReqOutput,
    CompleteWeightsUpdateReqInput,
    CompleteWeightsUpdateReqOutput,
    DestroyWeightsUpdateGroupReqInput,
    DestroyWeightsUpdateGroupReqOutput,
    GetWeightsByNameReqInput,
    GetWeightsByNameReqOutput,
    InitWeightsUpdateGroupReqInput,
    InitWeightsUpdateGroupReqOutput,
    PrepareWeightsUpdateReqInput,
    PrepareWeightsUpdateReqOutput,
    ReceiveWeightsReqInput,
    ReceiveWeightsReqOutput,
    ReleaseMemoryOccupationReqInput,
    ReleaseMemoryOccupationReqOutput,
    ResumeMemoryOccupationReqInput,
    ResumeMemoryOccupationReqOutput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightFromDiskReqOutput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromDistributedReqOutput,
    UpdateWeightsFromIPCReqInput,
    UpdateWeightsFromIPCReqOutput,
    UpdateWeightsFromTensorReqInput,
    UpdateWeightsFromTensorReqOutput,
    WeightUpdatePauseReq,
    WeightUpdateResumeReq,
)

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)


class SchedulerUpdateWeightsMixin:

    def update_weights_from_disk(
        self: Scheduler, recv_req: UpdateWeightFromDiskReqInput
    ):
        """In-place update of the weights from disk."""
        success, message = self.tp_worker.update_weights_from_disk(recv_req)
        if success:
            if recv_req.flush_cache:
                flush_cache_success = self.flush_cache()
                assert flush_cache_success, "Cache flush failed after updating weights"
        else:
            logger.error(message)
        return UpdateWeightFromDiskReqOutput(success, message, 0)

    def init_weights_update_group(
        self: Scheduler, recv_req: InitWeightsUpdateGroupReqInput
    ):
        """Initialize the online model parameter update group."""
        # Notify detokenizer before blocking NCCL operation
        self.send_to_detokenizer.send_output(WeightUpdatePauseReq(), recv_req)

        try:
            success, message = self.tp_worker.init_weights_update_group(recv_req)
        finally:
            # Notify detokenizer after NCCL operation completes
            self.send_to_detokenizer.send_output(WeightUpdateResumeReq(), recv_req)

        return InitWeightsUpdateGroupReqOutput(success, message)

    def destroy_weights_update_group(
        self: Scheduler, recv_req: DestroyWeightsUpdateGroupReqInput
    ):
        """Destroy the online model parameter update group."""
        success, message = self.tp_worker.destroy_weights_update_group(recv_req)
        return DestroyWeightsUpdateGroupReqOutput(success, message)

    def update_weights_from_distributed(
        self,
        recv_req: UpdateWeightsFromDistributedReqInput,
    ) -> Tuple[bool, str]:
        """Update the online model parameter."""
        kv_cache_freed = False
        try:
            # Free KV cache memory before receiving weights if requested
            # This helps avoid OOM when the KV cache takes most of GPU memory
            if recv_req.free_kv_cache_before_recv:
                if not self._is_no_request():
                    return UpdateWeightsFromDistributedReqOutput(
                        success=False,
                        message="Cannot free KV cache: there are pending requests. "
                        "Call pause_generation first.",
                    )
                logger.info("Freeing KV cache memory before receiving weights...")
                # Clear allocator state
                self.flush_cache()
                # Free the actual KV cache buffer memory
                kv_cache = self.token_to_kv_pool_allocator.get_kvcache()
                kv_cache._clear_buffers()
                torch.cuda.empty_cache()
                kv_cache_freed = True
                logger.info("KV cache memory freed successfully.")

            success, message = self.tp_worker.update_weights_from_distributed(recv_req)

            # Recreate KV cache buffers if we freed them
            if kv_cache_freed:
                logger.info("Recreating KV cache buffers after weight update...")
                kv_cache._create_buffers()
                logger.info("KV cache buffers recreated successfully.")

            if success:
                if recv_req.flush_cache:
                    flush_cache_success = self.flush_cache()
                    assert (
                        flush_cache_success
                    ), "Cache flush failed after updating weights"
            else:
                logger.error(message)
        except Exception as e:
            logger.error(f"update_weights_from_distributed failed: {e}")
            # Try to recreate KV cache buffers if we freed them and failed
            if kv_cache_freed:
                try:
                    logger.info("Attempting to recreate KV cache buffers after failure...")
                    kv_cache._create_buffers()
                except Exception as e2:
                    logger.error(f"Failed to recreate KV cache buffers: {e2}")
            return UpdateWeightsFromDistributedReqOutput(success=False, message=str(e))

        return UpdateWeightsFromDistributedReqOutput(success, message)

    def prepare_weights_update(
        self: Scheduler,
        recv_req: PrepareWeightsUpdateReqInput,
    ) -> PrepareWeightsUpdateReqOutput:
        """Phase 1 of two-phase weight update protocol.

        This starts background recv threads that are ready to receive NCCL broadcasts.
        Returns immediately once the recv threads are started and ready.
        """
        # Notify detokenizer before starting the weight update operation
        self.send_to_detokenizer.send_output(WeightUpdatePauseReq(), recv_req)

        try:
            success, message = self.tp_worker.prepare_weights_update(recv_req)
        except Exception as e:
            # If preparation fails, resume detokenizer
            self.send_to_detokenizer.send_output(WeightUpdateResumeReq(), recv_req)
            logger.error(f"Failed to prepare weights update: {e}")
            return PrepareWeightsUpdateReqOutput(success=False, message=str(e))

        if not success:
            # If preparation fails, resume detokenizer
            self.send_to_detokenizer.send_output(WeightUpdateResumeReq(), recv_req)
            logger.error(message)

        return PrepareWeightsUpdateReqOutput(success=success, message=message)

    def complete_weights_update(
        self: Scheduler,
        recv_req: CompleteWeightsUpdateReqInput,
    ) -> CompleteWeightsUpdateReqOutput:
        """Phase 2 of two-phase weight update protocol.

        This waits for the background recv threads to complete and applies the weights.
        """
        try:
            success, message = self.tp_worker.complete_weights_update(recv_req)
            if success:
                if recv_req.flush_cache:
                    flush_cache_success = self.flush_cache()
                    assert (
                        flush_cache_success
                    ), "Cache flush failed after updating weights"
            else:
                logger.error(message)
        finally:
            # Notify detokenizer after weight update operation completes
            self.send_to_detokenizer.send_output(WeightUpdateResumeReq(), recv_req)

        return CompleteWeightsUpdateReqOutput(success=success, message=message)

    def receive_weights(
        self: Scheduler,
        recv_req: ReceiveWeightsReqInput,
    ) -> ReceiveWeightsReqOutput:
        """Receive weights via NCCL broadcast from an existing process group.

        This is used as part of the pause/broadcast/resume weight sync protocol.
        Assumes inference is already paused and NCCL group is initialized.
        """
        kv_cache_freed = False
        try:
            # Free KV cache memory before receiving weights if requested
            # This helps avoid OOM when the KV cache takes most of GPU memory
            if recv_req.free_kv_cache_before_recv:
                if not self._is_no_request():
                    return ReceiveWeightsReqOutput(
                        success=False,
                        message="Cannot free KV cache: there are pending requests. "
                        "Call pause_generation first.",
                    )
                logger.info("Freeing KV cache memory before receiving weights...")
                # Clear allocator state
                self.flush_cache()
                # Free the actual KV cache buffer memory
                kv_cache = self.token_to_kv_pool_allocator.get_kvcache()
                kv_cache._clear_buffers()
                torch.cuda.empty_cache()
                kv_cache_freed = True
                logger.info("KV cache memory freed successfully.")

            success, message = self.tp_worker.receive_weights(recv_req)

            # Recreate KV cache buffers if we freed them
            if kv_cache_freed:
                logger.info("Recreating KV cache buffers after weight update...")
                kv_cache._create_buffers()
                logger.info("KV cache buffers recreated successfully.")

            if success and recv_req.flush_cache:
                flush_cache_success = self.flush_cache()
                assert flush_cache_success, "Cache flush failed after receiving weights"
        except Exception as e:
            logger.error(f"receive_weights failed: {e}")
            # Try to recreate KV cache buffers if we freed them and failed
            if kv_cache_freed:
                try:
                    logger.info("Attempting to recreate KV cache buffers after failure...")
                    kv_cache._create_buffers()
                except Exception as e2:
                    logger.error(f"Failed to recreate KV cache buffers: {e2}")
            return ReceiveWeightsReqOutput(success=False, message=str(e))

        return ReceiveWeightsReqOutput(success=success, message=message)

    def update_weights_from_tensor(
        self: Scheduler, recv_req: UpdateWeightsFromTensorReqInput
    ):
        """Update the online model parameter from tensors."""
        worker = self.draft_worker or self.tp_worker
        success, message = worker.update_weights_from_tensor(recv_req)
        # TODO extract common code b/t update_weights_from_distributed and update_weights_from_tensor later
        if success:
            if recv_req.flush_cache:
                flush_cache_success = self.flush_cache()
                assert flush_cache_success, "Cache flush failed after updating weights"
        else:
            logger.error(message)
        torch.distributed.barrier(group=self.tp_cpu_group)
        return UpdateWeightsFromTensorReqOutput(success, message)

    def update_weights_from_ipc(
        self: Scheduler, recv_req: UpdateWeightsFromIPCReqInput
    ):
        """Update the online model parameter from IPC for checkpoint-engine integration."""
        success, message = self.tp_worker.update_weights_from_ipc(recv_req)
        if success:
            if recv_req.flush_cache:
                flush_cache_success = self.flush_cache()
                assert flush_cache_success, "Cache flush failed after updating weights"
        else:
            logger.error(message)
        torch.distributed.barrier(group=self.tp_cpu_group)
        return UpdateWeightsFromIPCReqOutput(success, message)

    def get_weights_by_name(self: Scheduler, recv_req: GetWeightsByNameReqInput):
        parameter = self.tp_worker.get_weights_by_name(recv_req)
        return GetWeightsByNameReqOutput(parameter)

    def release_memory_occupation(
        self: Scheduler, recv_req: ReleaseMemoryOccupationReqInput
    ):
        assert (
            self._is_no_request()
        ), "release_memory_occupation should be called only when no ongoing request."

        tags = recv_req.tags

        if tags is None or len(tags) == 0:
            tags = GPU_MEMORY_ALL_TYPES

        for tag in tags:
            self.offload_tags.add(tag)

        if GPU_MEMORY_TYPE_KV_CACHE in tags:
            self.memory_saver_adapter.pause(GPU_MEMORY_TYPE_KV_CACHE)
            self.flush_cache()

        if GPU_MEMORY_TYPE_WEIGHTS in tags:
            self.stashed_model_static_state = _export_static_state(
                self.tp_worker.model_runner.model
            )
            torch.distributed.barrier(self.tp_cpu_group)
            self.memory_saver_adapter.pause(GPU_MEMORY_TYPE_WEIGHTS)

        if GPU_MEMORY_TYPE_CUDA_GRAPH in tags:
            self.memory_saver_adapter.pause(GPU_MEMORY_TYPE_CUDA_GRAPH)

        torch.get_device_module().synchronize()

        return ReleaseMemoryOccupationReqOutput()

    def resume_memory_occupation(
        self: Scheduler, recv_req: ResumeMemoryOccupationReqInput
    ):
        tags = recv_req.tags

        if tags is None or len(tags) == 0:
            tags = GPU_MEMORY_ALL_TYPES

        for tag in tags:
            self.offload_tags.remove(tag)

        if GPU_MEMORY_TYPE_CUDA_GRAPH in tags:
            self.memory_saver_adapter.resume(GPU_MEMORY_TYPE_CUDA_GRAPH)

        if GPU_MEMORY_TYPE_WEIGHTS in tags:
            self.memory_saver_adapter.resume(GPU_MEMORY_TYPE_WEIGHTS)
            torch.distributed.barrier(self.tp_cpu_group)
            _import_static_state(
                self.tp_worker.model_runner.model,
                self.stashed_model_static_state,
            )
            del self.stashed_model_static_state

        if GPU_MEMORY_TYPE_KV_CACHE in tags:
            self.memory_saver_adapter.resume(GPU_MEMORY_TYPE_KV_CACHE)

        return ResumeMemoryOccupationReqOutput()

    def check_weights(self: Scheduler, recv_req: CheckWeightsReqInput):
        try:
            self.tp_worker.model_runner.check_weights(action=recv_req.action)
            return CheckWeightsReqOutput(success=True, message="Success.")
        except Exception as e:
            logger.warning(f"check_weights see error: {e}")
            traceback.print_exc()
            return CheckWeightsReqOutput(success=False, message=f"{e}")

    def save_remote_model(self: Scheduler, params):
        url = params["url"]

        self.tp_worker.model_runner.save_remote_model(url)

        if self.draft_worker is not None:
            draft_url = params.get("draft_url", None)
            assert (
                draft_url is not None
            ), "draft_url must be provided when draft model is enabled"
            self.draft_worker.model_runner.save_remote_model(draft_url)

    def save_sharded_model(self: Scheduler, params):
        self.tp_worker.model_runner.save_sharded_model(
            path=params["path"],
            pattern=params["pattern"],
            max_size=params["max_size"],
        )


def _export_static_state(model):
    return dict(
        buffers=[
            (name, buffer.detach().clone()) for name, buffer in model.named_buffers()
        ]
    )


def _import_static_state(model, static_params):
    self_named_buffers = dict(model.named_buffers())
    for name, tensor in static_params["buffers"]:
        self_named_buffers[name][...] = tensor
