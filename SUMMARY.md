# Fix: Detokenizer Unresponsive During Weight Update Group Initialization

## Problem Summary

When `/init_weights_update_group` is called on an SGLang server to enable online weight updates (e.g., from a training process like SLIME), the server could become unresponsive and fail health checks:

1. The scheduler calls `init_custom_process_group()` which performs **blocking** NCCL group initialization
2. During this blocking call (can take 20+ seconds), the scheduler's event loop stalls
3. The scheduler cannot send batch outputs to the detokenizer
4. The detokenizer blocks on `recv_from_scheduler.recv_pyobj()` waiting for data
5. The detokenizer's soft watchdog times out, or external health checks fail because the server appears unresponsive

## Solution Overview

This fix implements two complementary approaches:

1. **Health check awareness** - The HTTP health check endpoint returns 200 OK immediately during weight update operations, preventing external health monitors from marking the server as unhealthy.

2. **Pause/resume signaling** - The scheduler notifies the detokenizer before and after blocking NCCL operations, allowing the detokenizer to pause its watchdog and avoid false timeout detection.

## Files Modified

### 1. `python/sglang/srt/managers/tokenizer_manager.py`
- Added `_weight_update_in_progress: bool = False` flag in `init_weight_update()` method
- This flag tracks when any weight update operation is in progress

### 2. `python/sglang/srt/managers/tokenizer_communicator_mixin.py`
- Wrapped the following methods with the `_weight_update_in_progress` flag:
  - `init_weights_update_group()` - NCCL process group initialization
  - `update_weights_from_distributed()` - Weight transfer via NCCL
  - `update_weights_from_tensor()` - Weight update from tensor
  - `get_weights_by_name()` - Weight retrieval

### 3. `python/sglang/srt/entrypoints/http_server.py`
- Modified `health_generate()` to return HTTP 200 immediately when `_weight_update_in_progress` is True
- Uses `getattr()` with default to safely handle different TokenizerManager types

### 4. `python/sglang/srt/managers/io_struct.py`
- Added `WeightUpdatePauseReq` dataclass - signals detokenizer that a blocking operation is starting
- Added `WeightUpdateResumeReq` dataclass - signals detokenizer that the blocking operation completed

### 5. `python/sglang/srt/managers/scheduler_update_weights_mixin.py`
- Added imports for `WeightUpdatePauseReq` and `WeightUpdateResumeReq`
- Modified `init_weights_update_group()` to:
  - Send `WeightUpdatePauseReq` to detokenizer before NCCL init
  - Send `WeightUpdateResumeReq` to detokenizer after NCCL init (in finally block)

### 6. `python/sglang/srt/managers/detokenizer_manager.py`
- Added imports for `WeightUpdatePauseReq` and `WeightUpdateResumeReq`
- Added `_weight_update_paused` state tracking
- Added handlers to the request dispatcher:
  - `_handle_weight_update_pause()` - pauses the watchdog
  - `_handle_weight_update_resume()` - resumes the watchdog

### 7. `python/sglang/srt/utils/watchdog.py`
- Added `pause()` and `resume()` methods to base `Watchdog` class (no-op)
- Added `_paused` flag to `_WatchdogReal` class
- Modified `is_active` lambda to check both `_active` and `_paused` states
- Implemented `pause()` and `resume()` in `_WatchdogReal`:
  - `pause()` sets `_paused = True`
  - `resume()` sets `_paused = False` and increments counter to prevent immediate timeout

## What This Supports

### Online Weight Updates from Training
This fix enables the SGLang inference server to receive weight updates from a distributed training process (e.g., SLIME, DeepSpeed, Megatron) without crashing or appearing unhealthy. The typical workflow is:

```
Training Process                    SGLang Server
      |                                   |
      |  1. POST /init_weights_update_group
      |-------------------------------------->
      |                                   |
      |  2. NCCL group initialization (blocking)
      |<=====================================>
      |                                   |
      |  3. POST /update_weights_from_distributed
      |-------------------------------------->
      |                                   |
      |  4. NCCL broadcast weights
      |<=====================================>
      |                                   |
      |  5. POST /destroy_weights_update_group
      |-------------------------------------->
```

### Reinforcement Learning from Human Feedback (RLHF)
This is particularly important for RLHF workflows where:
- A training process updates model weights based on reward signals
- The inference server needs to be updated with new weights without restarting
- The system needs to remain healthy during the weight synchronization

### Continuous Training / Online Learning
Supports scenarios where models are continuously updated:
- A/B testing with live weight updates
- Incremental model improvements
- Hot-swapping model weights without downtime

## Alternative: Environment Variable Workaround

If a minimal change is preferred, the health check timeout can be increased:

```bash
export SGLANG_HEALTH_CHECK_TIMEOUT=120
```

However, this doesn't fix the underlying watchdog timeout issue and may mask real failures.

## Testing

To verify the fix:

1. Start an SGLang server with TP > 1
2. Call `/init_weights_update_group` with appropriate NCCL configuration
3. Verify health checks return 200 during the operation
4. Verify the server is responsive after the operation completes
5. Verify no watchdog timeout errors in the logs
