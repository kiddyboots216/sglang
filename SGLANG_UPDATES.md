# SGLang Updates Required for Two-Phase Weight Sync

## Problem

The current weight synchronization between the training server (Tomni) and inference server (SGLang) has a race condition that causes NCCL broadcasts to hang.

### Current Flow (Broken)
```
Training                                    SGLang
========                                    ======
POST /update_weights_from_distributed -->   [HTTP handler starts]
sleep(5s)  # bandaid fix                    [Starts NCCL recv inside HTTP handler]
NCCL broadcast                              [NCCL recv]
                                            [HTTP handler returns]
```

**Problem:** The training server starts NCCL broadcast before SGLang's NCCL recv is ready, causing hangs on bucket 1/73.

## Solution: Two-Phase Protocol

Split the single HTTP call into two phases to ensure NCCL recv is ready before broadcast starts.

### New Flow
```
Training                                    SGLang
========                                    ======
     |                                           |
     |--- POST /prepare_weights_update ------->  |
     |    (metadata for ALL buckets)             |
     |                                           | [Start NCCL recv loop in background thread]
     |<-- {"status": "ready"} ------------------|
     |                                           |
     | [NCCL broadcast bucket 1]                 | [recv bucket 1]
     | [NCCL broadcast bucket 2]                 | [recv bucket 2]
     | ...                                       | ...
     | [NCCL broadcast bucket N]                 | [recv bucket N]
     |                                           |
     |--- POST /complete_weights_update ------> |
     |    (flush_cache=false)                    | [Wait for recv loop, apply weights]
     |<-- {"success": true, num_buckets: N} ----|
     |                                           |
```

**Benefits:**
- Only 2 HTTP calls total (not 3*N calls per bucket)
- No race condition - recv loop is guaranteed running before broadcast starts
- Training can broadcast at full speed without waiting

## Required SGLang Endpoints

### 1. `POST /prepare_weights_update`

Called once before any NCCL transfers. SGLang should start a background thread/task that receives all buckets.

**Request:**
```json
{
  "num_buckets": 73,
  "buckets": [
    {
      "names": ["model.layers.0.self_attn.q_proj.weight", ...],
      "dtypes": ["bfloat16", ...],
      "shapes": [[4096, 4096], ...]
    },
    ...
  ],
  "group_name": "weight_sync_group"
}
```

**Response (return immediately after starting recv loop):**
```json
{
  "status": "ready",
  "message": ""
}
```

**Implementation Notes:**
- Start a background thread that loops through all buckets
- For each bucket, allocate tensors based on metadata and call NCCL recv
- Return "ready" immediately after the background thread starts (don't wait for recv to complete)
- Store received tensors in a temporary buffer

### 2. `POST /complete_weights_update`

Called once after all NCCL broadcasts are done. SGLang should wait for the recv loop to finish and apply weights to the model.

**Request:**
```json
{
  "group_name": "weight_sync_group",
  "flush_cache": false
}
```

**Response (after weights are applied):**
```json
{
  "success": true,
  "num_buckets_received": 73,
  "message": ""
}
```

**Implementation Notes:**
- Wait for the background recv thread to complete
- Apply all received weights to the model
- If `flush_cache=true`, invalidate KV cache (usually not needed for full-weights sync)
- Return success with count of buckets received

## Pseudocode for SGLang Implementation

```python
# Global state for weight update
_weight_update_state = {
    "recv_thread": None,
    "received_weights": {},  # bucket_idx -> list of tensors
    "error": None,
    "num_buckets_expected": 0,
}

@app.post("/prepare_weights_update")
async def prepare_weights_update(request: PrepareWeightsUpdateRequest):
    global _weight_update_state

    # Get the NCCL process group (already initialized via /init_weights_update_group)
    pg = get_weight_update_group(request.group_name)
    if pg is None:
        return {"status": "error", "message": "Process group not initialized"}

    _weight_update_state["num_buckets_expected"] = request.num_buckets
    _weight_update_state["received_weights"] = {}
    _weight_update_state["error"] = None

    def recv_loop():
        try:
            for bucket_idx, bucket_meta in enumerate(request.buckets):
                tensors = []
                for name, dtype, shape in zip(bucket_meta.names, bucket_meta.dtypes, bucket_meta.shapes):
                    # Allocate tensor on GPU
                    tensor = torch.empty(shape, dtype=getattr(torch, dtype), device="cuda")
                    # NCCL recv (blocking)
                    dist.broadcast(tensor, src=0, group=pg)
                    tensors.append((name, tensor))
                _weight_update_state["received_weights"][bucket_idx] = tensors
        except Exception as e:
            _weight_update_state["error"] = str(e)

    # Start recv loop in background thread
    _weight_update_state["recv_thread"] = threading.Thread(target=recv_loop)
    _weight_update_state["recv_thread"].start()

    # Return immediately - recv is now waiting for data
    return {"status": "ready", "message": f"Started recv loop for {request.num_buckets} buckets"}


@app.post("/complete_weights_update")
async def complete_weights_update(request: CompleteWeightsUpdateRequest):
    global _weight_update_state

    # Wait for recv thread to finish
    if _weight_update_state["recv_thread"] is not None:
        _weight_update_state["recv_thread"].join(timeout=300)
        if _weight_update_state["recv_thread"].is_alive():
            return {"success": False, "message": "Recv thread timed out"}

    # Check for errors
    if _weight_update_state["error"]:
        return {"success": False, "message": _weight_update_state["error"]}

    # Apply weights to model
    num_received = len(_weight_update_state["received_weights"])
    for bucket_idx in sorted(_weight_update_state["received_weights"].keys()):
        for name, tensor in _weight_update_state["received_weights"][bucket_idx]:
            # Apply weight to model (implementation depends on SGLang internals)
            apply_weight_to_model(name, tensor)

    # Optionally flush KV cache
    if request.flush_cache:
        flush_kv_cache()

    # Cleanup
    _weight_update_state["received_weights"] = {}
    _weight_update_state["recv_thread"] = None

    return {
        "success": True,
        "num_buckets_received": num_received,
        "message": f"Applied weights from {num_received} buckets"
    }
```

## Type Definitions

For reference, here are the Pydantic models used by Tomni:

```python
class BucketMetadata(BaseModel):
    names: List[str]      # Parameter names
    dtypes: List[str]     # e.g., ["bfloat16", "float32"]
    shapes: List[List[int]]  # e.g., [[4096, 4096], [4096]]

class PrepareWeightsUpdateRequest(BaseModel):
    num_buckets: int
    buckets: List[BucketMetadata]
    group_name: str

class PrepareWeightsUpdateResponse(BaseModel):
    status: str  # "ready" or "error"
    message: str = ""

class CompleteWeightsUpdateRequest(BaseModel):
    group_name: str
    flush_cache: bool = False

class CompleteWeightsUpdateResponse(BaseModel):
    success: bool
    num_buckets_received: int = 0
    message: str = ""
```

## Testing

Once implemented, test with:

```bash
# 1. Start both servers
./scripts/start_servers.sh

# 2. Register endpoint
curl -X POST http://localhost:8990/add_inference_endpoint \
    -H "Content-Type: application/json" \
    -d '{"host": "localhost", "port": 30000, "world_size": 4}'

# 3. Trigger weight sync
curl -X POST http://localhost:8990/sync_inference_weights \
    -H "Content-Type: application/json" \
    -d '{"master_port": 29600, "buffer_size_mb": 1024}'

# Expected: All 73 buckets transfer successfully
```

## Existing Endpoints (Already Implemented in SGLang)

These endpoints already exist and work correctly:

- `POST /init_weights_update_group` - Initialize NCCL process group
- `POST /destroy_weights_update_group` - Cleanup process group
- `POST /update_weights_from_distributed` - Old single-phase approach (has race condition)
