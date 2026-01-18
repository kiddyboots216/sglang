# Weight Sync NCCL Hang Analysis

## Current Status

The weight sync hangs on bucket 1/73 during NCCL broadcast. Both sides appear synchronized (SGLang returns "ready" after all TP ranks are waiting), but the actual NCCL transfer never completes.

## Full Weight Sync Flow

### Timeline

```
TIME    TOMNI (Training, GPUs 0-3)                  SGLANG (Inference, GPUs 4-7)
════    ══════════════════════════                  ════════════════════════════

        ┌─────────────────────────────────────────────────────────────────────┐
        │ STEP 1: Initialize NCCL Process Group (world_size=5)                │
        └─────────────────────────────────────────────────────────────────────┘

T+0     Start background thread for inference init
        │
T+0     POST /init_weights_update_group ─────────►  Receive request
        │   {                                       │
        │     master_address: "...",                │
        │     master_port: 29600,                   │
        │     rank_offset: 1,          ◄── SGLang TP0=rank1, TP1=rank2, etc.
        │     world_size: 5,                        │
        │     group_name: "weight_sync_group"       │
        │   }                                       │
        │                                           │
T+2s    sleep(2s)                                   TP0: dist.init_process_group(rank=1)
        │                                           TP1: dist.init_process_group(rank=2)
        │                                           TP2: dist.init_process_group(rank=3)
        │                                           TP3: dist.init_process_group(rank=4)
        │                                           │
        │                                           [All 4 block waiting for rank 0]
        │
T+2s    dist.init_process_group(rank=0) ◄─────────► [NCCL rendezvous completes]
        │                                           │
        ◄──────────────────────────────────────────  200 OK {"success": true}

        ┌─────────────────────────────────────────────────────────────────────┐
        │ STEP 2: Build Buckets (local, no HTTP)                              │
        └─────────────────────────────────────────────────────────────────────┘

        579 params → 73 buckets (~1GB each)
        Bucket 0: [embed_tokens, layer0.q_proj, ...] (9 params, 1063 MB)
        Bucket 1: [layer0.experts.up_proj, ...]      (11 params, 1050 MB)
        ...

        ┌─────────────────────────────────────────────────────────────────────┐
        │ STEP 3: Prepare Transfer (Two-Phase Protocol)                       │
        └─────────────────────────────────────────────────────────────────────┘

        POST /prepare_weights_update ─────────────►  Receive request
          {                                          │
            num_buckets: 73,                         │
            buckets: [                               │
              {names: [...], dtypes: [...],          │
               shapes: [...]},                       │
              ...                                    │
            ],                                       │
            group_name: "weight_sync_group"          │
          }                                          │
                                                     │
                                                     Start 4 background threads (one per TP):
                                                     │
                                                     TP0 thread: allocate tensors for bucket 0
                                                     TP1 thread: allocate tensors for bucket 0
                                                     TP2 thread: allocate tensors for bucket 0
                                                     TP3 thread: allocate tensors for bucket 0
                                                     │
                                                     barrier() ◄── all 4 TP ranks sync here
                                                     │
                                                     TP0: ready_event.set()
                                                     │
        ◄──────────────────────────────────────────  200 OK {"status": "ready"}
                                                     │
                                                     [Now all 4 threads call dist.broadcast()
                                                      to RECEIVE - they block waiting for
                                                      rank 0 to send]

        ┌─────────────────────────────────────────────────────────────────────┐
        │ STEP 4: NCCL Broadcast (THE HANG HAPPENS HERE)                      │
        └─────────────────────────────────────────────────────────────────────┘

        FOR bucket_idx in 0..72:
          FOR param in bucket[bucket_idx]:
            │
            │ Training (rank 0, cuda:0):
            │   tensor = param.to(cuda:0)
            │   dist.broadcast(tensor, src=0, group=pg)  ◄── SENDS
            │                    │
            │                    │     SGLang (ranks 1-4, cuda:4-7):
            │                    │       tensor = empty(..., device=cuda:X)
            │                    └────►  dist.broadcast(tensor, src=0, group=pg)  ◄── RECEIVES
            │
          END FOR
        END FOR

        ┌─────────────────────────────────────────────────────────────────────┐
        │ STEP 5: Complete Transfer                                           │
        └─────────────────────────────────────────────────────────────────────┘

        POST /complete_weights_update ────────────►  Wait for recv threads to finish
          {group_name: "...", flush_cache: false}    Apply weights to model
                                                     │
        ◄──────────────────────────────────────────  200 OK {"success": true}

        ┌─────────────────────────────────────────────────────────────────────┐
        │ STEP 6: Cleanup                                                     │
        └─────────────────────────────────────────────────────────────────────┘

        POST /destroy_weights_update_group ───────►  dist.destroy_process_group()
        dist.destroy_process_group()
```

## Key Details

### NCCL Process Group Setup

| Role | Rank | Device | Process |
|------|------|--------|---------|
| Training | 0 | cuda:0 | Tomni rank 0 worker |
| SGLang TP0 | 1 | cuda:4 | SGLang TP rank 0 |
| SGLang TP1 | 2 | cuda:5 | SGLang TP rank 1 |
| SGLang TP2 | 3 | cuda:6 | SGLang TP rank 2 |
| SGLang TP3 | 4 | cuda:7 | SGLang TP rank 3 |

- World size = 5 (1 training rank + 4 SGLang TP ranks)
- All use same rendezvous: `tcp://<master_address>:29600`
- Group name: `weight_sync_group`

### Bucket 0 Transfer (where it hangs)

Training broadcasts 9 tensors sequentially:
```python
dist.broadcast(embed_tokens.weight, src=0)      # shape [151936, 2048]
dist.broadcast(layer0.q_proj.weight, src=0)     # shape [4096, 2048]
dist.broadcast(layer0.k_proj.weight, src=0)     # shape [512, 2048]
dist.broadcast(layer0.v_proj.weight, src=0)     # shape [512, 2048]
dist.broadcast(layer0.o_proj.weight, src=0)     # shape [2048, 4096]
dist.broadcast(layer0.q_norm.weight, src=0)     # shape [128]
dist.broadcast(layer0.k_norm.weight, src=0)     # shape [128]
dist.broadcast(layer0.gate.weight, src=0)       # shape [128, 2048]
dist.broadcast(layer0.experts.gate_proj, src=0) # shape [128, 768, 2048]
```

SGLang must receive in SAME ORDER:
```python
# All 4 TP ranks call broadcast for EACH tensor:
TP0: dist.broadcast(tensor0, src=0)  # receives embed_tokens
TP1: dist.broadcast(tensor0, src=0)  # receives embed_tokens
TP2: dist.broadcast(tensor0, src=0)  # receives embed_tokens
TP3: dist.broadcast(tensor0, src=0)  # receives embed_tokens

TP0: dist.broadcast(tensor1, src=0)  # receives layer0.q_proj
TP1: dist.broadcast(tensor1, src=0)  # receives layer0.q_proj
... etc for all 9 tensors
```

## Observed Behavior

### Training Server Logs
```
[23:03:50] Step 3: Preparing inference endpoints for batch transfer...
[23:03:51] Step 4: Transferring weights via NCCL...
[23:03:51] [Training] Broadcasting bucket 1/73 (9 params, 1063.3 MB)
[23:03:51] [Training] First param: name=model.embed_tokens.weight, shape=torch.Size([151936, 2048]), dtype=torch.bfloat16, device=cuda:0
# HANGS HERE - no more logs
```

### SGLang Server Logs
```
[23:03:50] Starting weight update with 73 bucket(s) on group weight_sync_group  # All 4 TP ranks
[23:03:51] Weight update recv thread is now waiting for NCCL broadcasts (73 bucket(s))  # All 4 TP ranks
[23:03:51] POST /prepare_weights_update 200 OK
# No more logs - recv threads are blocked in dist.broadcast()
```

## Potential Causes

### 1. Process Group Mismatch
The NCCL process group might not actually be connected between Training (GPU 0) and SGLang (GPUs 4-7). Even though `init_process_group` succeeds on both sides, they might be in separate groups.

**Check:** Are both sides using the exact same:
- `master_address`
- `master_port`
- `world_size`
- `group_name`

### 2. Barrier Using Wrong Process Group
The SGLang barrier warning suggests it might be using a different process group:
```
UserWarning: barrier(): using the device under current context
```

**Check:** Is the barrier in SGLang using the weight sync process group or the TP process group?

### 3. Broadcast Order Mismatch
NCCL broadcast is a collective - all ranks must call it in the same order for the same tensors.

**Check:** Is SGLang iterating through `buckets[0].names` in the exact same order as Training?

### 4. Device Assignment
Training broadcasts from `cuda:0`. SGLang receives on `cuda:4-7`.

**Check:** Does the NCCL process group support cross-GPU communication? Are all GPUs visible to both processes?

### 5. Tensor Shape/Type Mismatch
If SGLang allocates a tensor with different shape or dtype than Training sends, NCCL might hang.

**Check:** Does SGLang allocate tensors with exact shapes from `buckets[i].shapes` and dtypes from `buckets[i].dtypes`?

## HTTP Endpoints Called

| Endpoint | When | Purpose |
|----------|------|---------|
| `POST /init_weights_update_group` | Step 1 | Create NCCL process group on SGLang side |
| `POST /prepare_weights_update` | Step 3 | Start recv threads, wait for them to be ready |
| `POST /complete_weights_update` | Step 5 | Wait for recv to finish, apply weights |
| `POST /destroy_weights_update_group` | Step 6 | Cleanup NCCL process group |

## Request/Response Schemas

### /init_weights_update_group
```json
// Request
{
  "master_address": "research-secure-b200-02.cloud.together.ai",
  "master_port": 29600,
  "rank_offset": 1,
  "world_size": 5,
  "group_name": "weight_sync_group",
  "backend": "nccl"
}

// Response
{"success": true, "message": "..."}
```

### /prepare_weights_update
```json
// Request
{
  "num_buckets": 73,
  "buckets": [
    {
      "names": ["model.embed_tokens.weight", "model.layers.0.self_attn.q_proj.weight", ...],
      "dtypes": ["bfloat16", "bfloat16", ...],
      "shapes": [[151936, 2048], [4096, 2048], ...]
    },
    ...
  ],
  "group_name": "weight_sync_group"
}

// Response
{"status": "ready", "success": true, "message": "..."}
```

### /complete_weights_update
```json
// Request
{
  "group_name": "weight_sync_group",
  "flush_cache": false
}

// Response
{"success": true, "num_buckets_received": 73, "message": "..."}
```

## Tomni Code References

- `nccl_weight_sync.py:sync_weights()` - Main orchestration (line 604)
- `nccl_weight_sync.py:_init_training_process_group()` - Training side NCCL init (line 160)
- `nccl_weight_sync.py:_init_inference_endpoints()` - Calls SGLang /init_weights_update_group (line 217)
- `nccl_weight_sync.py:_prepare_transfer()` - Calls SGLang /prepare_weights_update (line 431)
- `nccl_weight_sync.py:_broadcast_bucket()` - NCCL broadcast loop (line 558)
- `nccl_weight_sync.py:_complete_transfer()` - Calls SGLang /complete_weights_update (line 495)
