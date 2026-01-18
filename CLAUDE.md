# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SGLang is a high-performance serving framework for large language models (LLMs) and multimodal models. It provides fast inference with RadixAttention for prefix caching, continuous batching, paged attention, speculative decoding, and multi-hardware support (NVIDIA/AMD GPUs, Intel CPUs, TPUs, Ascend NPUs).

## Repository Structure

```
sglang/
├── python/sglang/          # Main Python package
│   ├── lang/               # Frontend language API (DSL for LLM programs)
│   ├── srt/                # SGLang Runtime (backend inference engine)
│   │   ├── models/         # 140+ model implementations
│   │   ├── layers/         # Custom CUDA kernels (attention, MoE, quantization)
│   │   ├── managers/       # Request scheduling, batch management
│   │   ├── mem_cache/      # RadixAttention prefix caching
│   │   ├── entrypoints/    # HTTP/gRPC servers, OpenAI-compatible API
│   │   └── distributed/    # Tensor/pipeline/expert parallelism
│   ├── jit_kernel/         # JIT CUDA kernel compilation
│   └── multimodal_gen/     # Diffusion model support
├── sgl-kernel/             # Optimized CUDA kernels (separate C++/CUDA package)
├── sgl-model-gateway/      # Rust-based tokenizer/loader service
├── test/                   # Test suites
├── benchmark/              # Performance benchmarking
└── docs/                   # Sphinx documentation
```

## Build and Development

### Installation

```bash
# Main package (development mode)
cd python && pip install -e .

# With test dependencies
pip install -e ".[dev]"

# sgl-kernel (CUDA kernels)
cd sgl-kernel && make build

# Control build parallelism
make build MAX_JOBS=4
```

### Running Tests

Tests are in `test/registered/` and use a suite-based CI system:

```bash
cd test

# Run a specific test suite
python run_suite.py --hw cuda --suite stage-a-test-1

# Run nightly tests
python run_suite.py --hw cuda --suite nightly-1-gpu --nightly

# Run a single test file directly
python -m pytest test/registered/path/to/test_file.py

# Run specific test
python -m pytest test/registered/path/to/test_file.py::TestClass::test_method
```

### Code Formatting

```bash
# Run all pre-commit hooks
pre-commit run --all-files

# Individual tools
isort .
black .
ruff check --fix .
```

Pre-commit hooks: isort, black, ruff (F401/F821), clang-format (C++/CUDA), codespell.

### Documentation

```bash
cd docs
make serve          # Auto-build server at :8003
make compile        # Execute Jupyter notebooks
```

## Key Architecture Concepts

**Frontend (`lang/`)**: DSL for composing LLM programs with `@sgl.function` decorator. Backend-agnostic - works with OpenAI, Anthropic, or local SGLang runtime.

**SGLang Runtime (`srt/`)**: Core inference engine with:
- Zero-overhead CPU scheduler for batch management
- RadixAttention with radix tree for prefix caching
- Paged attention and continuous batching
- Support for TP/PP/EP/DP parallelism

**Model implementations** in `srt/models/` follow a consistent pattern - check `llama.py` as reference.

## Common Entry Points

```bash
# Launch server
python -m sglang.launch_server --model-path <model> --host 0.0.0.0 --port 30000

# CLI
sglang serve --model-path <model>
sglang generate --model-path <model>
```

**Engine API**:
```python
from sglang import Engine
engine = Engine(model_path="...")
```

## PR Process

1. Format code with pre-commit
2. Add tests for new functionality
3. Tag PR with `run-ci` to trigger CI
4. Get CODEOWNER approvals
5. Merge Oncall handles merge (can bypass flaky CI when needed)

CI is organized by hardware platform and test suites (per-commit vs nightly).
