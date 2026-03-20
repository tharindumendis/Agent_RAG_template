# Agent_rag Installation Guide

This guide explains how to install `agent-rag-mcp` with optimal dependencies for your use case.

## Quick Start: Which Installation Path?

| **Use Case** | **Command** | **Size** | **Needs** |
|---|---|---|---|
| **Ollama users** (recommended for light deployments) | `uv install agent-rag-mcp[ollama]` | ~50 MB | Ollama running locally |
| **Local embeddings** (offline, no external services) | `uv install agent-rag-mcp[local]` | ~300 MB | Stable CPU environment |
| **Everything** (all providers + GPU support) | `uv install agent-rag-mcp[all]` | **~2 GB** | GPU (NVIDIA only) |
| **Minimal** (bare install, manual config) | `uv install agent-rag-mcp` | ~50 MB | Not recommended |

---

## Installation Methods

### 1. **For Ollama Users** (Recommended for lightweight deployments)

```bash
# Install minimal dependencies — no torch/cuda
uv install agent-rag-mcp[ollama]

# Then start the RAG server (uses Ollama for embeddings)
python Agent_rag/server.py
```

**Config** (`Agent_rag/config.yaml`):
```yaml
embeddings:
  provider: "ollama"
  ollama_base_url: "http://localhost:11434"
  ollama_model: "nomic-embed-text"  # lightweight, fast
```

**Benefits:**
- ✅ Minimal installation (~50 MB)
- ✅ No CUDA/PyTorch overhead
- ✅ Works on CPU-only machines
- ✅ Models run in Ollama (separate process, isolated)
- ⚠️ Requires Ollama to be running

**Start Ollama:**
```bash
# On Windows/Mac/Linux
ollama serve

# In another terminal, pull embedding model
ollama pull nomic-embed-text
```

---

### 2. **For Local Embeddings without GPU**

```bash
# Install ONNX runtime for local embeddings
uv install agent-rag-mcp[local]

# Start the RAG server (uses local embeddings)
python Agent_rag/server.py
```

**Config** (`Agent_rag/config.yaml`):
```yaml
embeddings:
  provider: "onnx"
  model: "all-MiniLM-L6-v2"  # auto-downloads on first run
  device: "cpu"
```

**Benefits:**
- ✅ No external service dependency (Ollama not required)
- ✅ Self-contained embedding engine
- ✅ Fast on modern CPUs
- ✅ ~300 MB footprint
- ⚠️ Slower than GPU embeddings

---

### 3. **For GPU-Accelerated Embeddings (NVIDIA)**

```bash
# Install PyTorch + sentence-transformers for GPU
uv install agent-rag-mcp[gpu]

# Start the RAG server (uses GPU)
python Agent_rag/server.py
```

**Config** (`Agent_rag/config.yaml`):
```yaml
embeddings:
  provider: "onnx"
  model: "all-MiniLM-L6-v2"
  device: "cuda"  # GPU mode
```

**Benefits:**
- ✅ Fastest embeddings (~10x faster than CPU)
- ✅ GPU-optimized sentence-transformers
- ⚠️ Large download (~1.2 GB with CUDA packages)
- ⚠️ Requires NVIDIA GPU + CUDA drivers

---

### 4. **For Development / All Features**

```bash
# Install everything for full flexibility
uv install agent-rag-mcp[all]
```

Includes:
- ONNX runtime (local CPU embeddings)
- PyTorch + sentence-transformers (GPU embeddings)
- All providers available

---

## Comparison Table

| Feature | Ollama | Local (ONNX) | GPU (PyTorch) |
|---------|--------|--------------|---------------|
| **Installation size** | 50 MB | 300 MB | 1.2 GB |
| **Embed speed** | Medium (depends on Ollama) | Fast | Very fast |
| **External service** | ✅ Ollama server | ❌ None | ❌ None |
| **GPU required** | ❌ No | ❌ No | ✅ Yes |
| **CPU-only friendly** | ✅ Yes | ✅ Yes | ⚠️ Works but slow |
| **Production ready** | ✅ Yes | ✅ Yes | ✅ Yes |

---

## Troubleshooting

### Problem: "ImportError: No module named torch"
**Solution:** You're using ONNX provider but didn't install it.
```bash
uv install agent-rag-mcp[local]
```

### Problem: "ConnectionRefusedError: Ollama not running"
**Solution:** Start Ollama first:
```bash
ollama serve
```

### Problem: Large CUDA packages downloading
**Solution:** Switch to Ollama or local ONNX:
```bash
uv pip uninstall torch nvidia-cuda* -y
uv install agent-rag-mcp[ollama]
```

### Problem: Embeddings slow on startup
**Solution:** Pre-warm the embedder (the code does this automatically, but you can test):
```bash
python -c "from agent_rag.rag.embedder import OnnxEmbedder; OnnxEmbedder().prewarm()"
```

---

## Migration: Moving from GPU to Ollama

If you installed with GPU and want to switch to lightweight Ollama:

```bash
# Remove torch/CUDA
uv pip uninstall torch nvidia-cuda* sentence-transformers -y

# Install Ollama-only
uv install agent-rag-mcp[ollama]

# Update config.yaml
# Change: provider: "onnx" → provider: "ollama"

# Verify
python Agent_rag/server.py
```

This will free up ~1.2 GB of disk space and ~1 GB of RAM per instance.

---

## Advanced: Environment-Specific Installations

### Docker Ollama Only
```dockerfile
# Small footprint, no GPU needed
FROM python:3.11-slim
RUN pip install uv
RUN uv install agent-rag-mcp[ollama]
```

### Docker with GPU Support
```dockerfile
# Larger image, CUDA support
FROM nvidia/cuda:12.1-runtime-ubuntu22.04
RUN apt-get install -y python3.11 python3-pip
RUN pip install uv
RUN uv install agent-rag-mcp[gpu]
```

---

## Questions?

- **Ollama docs:** https://ollama.ai
- **ChromaDB docs:** https://docs.trychroma.com
- **OpenAI embeddings:** See `Agent_head` for integration examples
