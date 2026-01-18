# Two-Phase Weight Update Protocol - Implementation Summary

## Problem Statement

When using NCCL to transfer model weights from a training server (Tomni) to an inference server (SGLang), there's a **race condition**: the training side starts NCCL broadcasts before SGLang's recv is ready. NCCL requires all ranks to call the collective operation synchronously, causing hangs.

### Original Flow (Broken)
```
Tomni                              SGLang
======                              ======
POST /update_weights_from_distributed ──►  Start NCCL recv (blocking)
dist.broadcast() ────────────────────►     (race - might not be ready!)
                                           HANG if recv not ready
```

## Solution: Two-Phase Protocol

Split the weight update into two phases to ensure SGLang is ready before Tomni broadcasts.

### New Flow
```
Tomni                              SGLang
======                              ======
POST /prepare_weights_update ───────►  Start background recv threads
                                       All TP ranks barrier sync
                                       Signal "ready"
◄─── 200 OK {"status": "ready"} ────

dist.broadcast() (bucket 1) ────────►  Background threads recv
dist.broadcast() (bucket 2) ────────►
...
dist.broadcast() (bucket N) ────────►

POST /complete_weights_update ──────►  Wait for recv threads
                                       Apply weights to model
◄─── 200 OK ────────────────────────
```

## Architecture

### NCCL Process Group
- **World size**: 5 (1 Tomni rank + 4 SGLang TP ranks)
- **Rank 0**: Tomni (training server, cuda:0)
- **Ranks 1-4**: SGLang TP workers (cuda:4-7 typically)
- **Group name**: `weight_sync_group`

### Weight Transfer
- 579 parameters, ~61 GB total
- Split into 73 buckets (~1GB each)
- Each bucket: multiple tensors broadcast individually with `async_op=True`

## Files Modified in SGLang

### 1. `python/sglang/srt/managers/io_struct.py`
Added request/response dataclasses:
- `WeightBucket` - metadata for a single bucket (names, dtypes, shapes)
- `PrepareWeightsUpdateReqInput` - phase 1 request (supports batched bucket format)
- `PrepareWeightsUpdateReqOutput` - phase 1 response
- `CompleteWeightsUpdateReqInput` - phase 2 request
- `CompleteWeightsUpdateReqOutput` - phase 2 response

### 2. `python/sglang/srt/entrypoints/http_server.py`
Added HTTP endpoints:
- `POST /prepare_weights_update` - Returns `{"status": "ready"}` when all TP ranks are waiting
- `POST /complete_weights_update` - Waits for recv, applies weights

### 3. `python/sglang/srt/model_executor/model_runner.py`
Core implementation:
- `prepare_weights_update()` - Starts background recv thread, waits for ready signal
- `_recv_weights_background()` - Background thread that receives all buckets
- `_recv_single_bucket_individual()` - Receives one bucket (multiple broadcasts)
- `_recv_single_bucket_flattened()` - Alternative flattened bucket format
- `complete_weights_update()` - Waits for background thread, applies weights

Key instance variables:
```python
self._pending_weight_update_thread = None
self._pending_weight_update_result = None
self._pending_weight_update_weights = None
self._pending_weight_update_lock = threading.Lock()
self._pending_weight_update_ready_event = None  # Signals when recv is ready
```

### 4. `python/sglang/srt/managers/tokenizer_manager.py`
Added methods to route requests to scheduler.

### 5. `python/sglang/srt/managers/scheduler.py`
Added dispatcher entries for the new request types.

### 6. `python/sglang/srt/managers/scheduler_update_weights_mixin.py`
Added handler methods that fan out to TP workers.

### 7. `python/sglang/srt/managers/tp_worker.py`
Added delegation methods to ModelRunner.

### 8. `python/sglang/srt/utils/common.py`
- `init_custom_process_group()` - Uses standard `rendezvous()` for NCCL process group creation
- Fixed PyTorch version comparison for `backend_options` parameter

## Current Issue: NCCL Process Group Creation Hangs

### Symptoms
1. HTTP `/init_weights_update_group` is called
2. Both Tomni and SGLang complete TCPStore rendezvous
3. But `_new_process_group_helper()` hangs during NCCL initialization

### What We've Tried

1. **Barrier synchronization** - Added `torch.distributed.barrier(group=tp_group.cpu_group)` before signaling ready. Works correctly.

2. **Matching NCCL settings** - Set `NCCL_CUMEM_ENABLE=0` on both sides.

3. **Explicit device_id** - Pass `device_id=torch.device(f"cuda:{gpu_id}")` to process group creation.

4. **torch.cuda.set_device()** - Call before process group creation to ensure correct CUDA context.

5. **Standard rendezvous** - Changed from direct TCPStore creation to standard `rendezvous()` function for cross-node compatibility.

### Current State

Both sides now use standard PyTorch `rendezvous()` for process group creation:
- Tomni (rank 0) and SGLang (ranks 1-4) all call `rendezvous(tcp://host:port, rank, world_size)`
- Rendezvous completes successfully (TCPStore coordination works)
- Hang occurs in `_new_process_group_helper()` during NCCL communicator initialization

### Suspected Root Causes

1. **CUDA device context** - NCCL may not know which GPU each rank should use
2. **Cross-GPU topology** - Tomni on GPU 0, SGLang on GPUs 4-7 may have NCCL communication issues
3. **CUDA_VISIBLE_DEVICES mismatch** - Different processes may see different GPU indices

## Debug Logging Added

In `model_runner.py`:
```python
# During init_weights_update_group:
logger.info(f"Device info: self.device={self.device}, self.gpu_id={self.gpu_id}, "
            f"CUDA_VISIBLE_DEVICES={cuda_visible}, current_cuda_device={current_device}")
logger.info(f"Set CUDA device to {self.gpu_id} before process group creation")

# During broadcast:
logger.info(f"Starting NCCL broadcasts: self.device={self.device}, self.gpu_id={self.gpu_id}, "
            f"CUDA_VISIBLE_DEVICES={cuda_visible}, current_cuda_device={current_device}, "
            f"num_weights={len(weights)}, first_weight_device={weights[0][1].device}")
```

## Next Steps to Debug

1. **Check CUDA_VISIBLE_DEVICES** - Verify GPU visibility in both processes
2. **Check NCCL_DEBUG=INFO** - Get detailed NCCL initialization logs
3. **Verify GPU topology** - Ensure NCCL can communicate between GPU 0 and GPUs 4-7
4. **Test with single GPU** - Simplify to world_size=2 (1 Tomni + 1 SGLang) to isolate issue

## API Reference

### POST /prepare_weights_update
```json
// Request
{
  "num_buckets": 73,
  "buckets": [
    {"names": ["model.embed_tokens.weight", ...], "dtypes": ["bfloat16", ...], "shapes": [[151936, 2048], ...]},
    ...
  ],
  "group_name": "weight_sync_group"
}

// Response (success)
{"status": "ready", "success": true, "message": "..."}
```

### POST /complete_weights_update
```json
// Request
{"group_name": "weight_sync_group", "flush_cache": false}

// Response (success)
{"success": true, "message": "..."}
```

## Related Files

- `/scratch/apanda/full-weights-rl/src/tomni/server/weight_sync/nccl_weight_sync.py` - Tomni's weight sync implementation
- `/scratch/apanda/sglang/SGL_HANG.md` - Detailed hang analysis document
