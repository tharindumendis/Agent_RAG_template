# Agent_rag

Agent_rag is a RAG (Retrieval-Augmented Generation) MCP (Model Context Protocol) Server. It uses ChromaDB for vector storage and provides flexible embedding options—from lightweight Ollama integration to local ONNX models—with zero external dependencies.

## Available Tools

This server exposes several MCP tools for the orchestrator (`Agent_head`) or any other MCP client:

- **`rag_ingest`** — Ingest documents, directories, or raw text into a collection
- **`rag_search`** — Perform semantic search against your knowledge base
- **`rag_list_collections`** — List all active collections
- **`rag_delete_collection`** — Delete a specific collection

## Features

- **Flexible Embedding Providers** — Choose your embedding backend:
  - **Ollama** (~50 MB): Lightweight, runs with Ollama service—94% smaller than torch-based setup
  - **Local ONNX** (~300 MB): Fully offline with CPU/GPU support—no external services needed
  - **GPU-Accelerated** (~1.2 GB): PyTorch + sentence-transformers for maximum performance
- **ChromaDB Integration**: Persistent vector database for efficient semantic search
- **FastMCP Built-in**: Asynchronous, thread-safe tool execution
- **Easy Configuration**: Flexible `config.yaml` for chunk size, collections, embedding models, and provider selection
- **Zero External Dependencies**: No API keys required—everything runs locally

## Quick Start

For a complete installation guide with size comparisons, troubleshooting, and provider-specific setup:

👉 **See [INSTALLATION.md](INSTALLATION.md)**

## Installation & Usage

### Interactive Setup (Recommended)

```bash
cd Agent_rag
python setup_agent_rag.py
```

This interactive script guides you through provider selection and runs the appropriate installation command.

### Direct Installation

**For Ollama** (lightweight, ~50 MB):
```bash
uv install .[ollama]
python server.py
```

**For Local Offline** (ONNX, ~300 MB):
```bash
uv install .[local]
python server.py
```

**For GPU** (PyTorch acceleration, ~1.2 GB):
```bash
uv install .[gpu]
python server.py
```

**For All Providers** (complete setup):
```bash
uv install .[all]
python server.py
```

### Running with `uvx`

You can run the published MCP server directly. `uvx` will automatically download and run the latest version:

```bash
uvx agent-rag-mcp
```

**Transport Modes**

By default, the server runs in `stdio` transport mode (designed to be spawned as a subprocess by MCP clients like `Agent_head`).

To run it over HTTP using Server-Sent Events (SSE):

```bash
uvx agent-rag-mcp --transport sse --port 8002 --host 0.0.0.0
```

### Specifying a Test Registry (If using TestPyPI)

If you published the package to TestPyPI instead of the main PyPI, run it via:

```bash
uvx --extra-index-url https://test.pypi.org/simple/ --index-strategy unsafe-best-match agent-rag-mcp@latest
```

## Configuration

The embedding provider and collection behavior are configured in `config.yaml`:

```yaml
embeddings:
  provider: "ollama"              # Options: "ollama", "onnx"
  ollama_base_url: "http://localhost:11434"  # Ollama service URL
  ollama_model: "nomic-embed-text"  # Ollama embedding model

  # For ONNX provider:
  # model: "all-MiniLM-L6-v2"
  # device: "cpu"                 # Options: "cpu", "cuda"

database:
  persist_directory: "./chroma_db"
  
document_processing:
  chunk_size: 1000
  chunk_overlap: 200
```

For detailed configuration options and provider-specific settings, see [INSTALLATION.md](INSTALLATION.md#configuration).

## Integrating with `Agent_head`

To connect this RAG server to your `Agent_head` orchestrator, add the following configuration to `Agent_head/config.yaml`:

```yaml
memory:
  enabled: true
  backend: "rag"

  # Configure this if backend is set to "rag"
  rag_server:
    command: "uvx"
    args: ["agent-rag-mcp"] # Or ["--from", "/path/to/local/Agent_rag", "agent-rag-mcp"] for local development
    collection: "agent_memory"
```

## Local Development

If you are developing this package locally:

1. **Install dependencies**:
   ```bash
   uv sync
   ```
2. **Run locally**:
   ```bash
   uv run agent-rag-mcp
   ```
3. **Test the server**:
   ```bash
   python test_mcp_client.py
   ```
4. **Build the package**:
   ```bash
   uv build
   ```

## Architecture

Agent_rag uses a modular provider system:

- **Embeddings Layer** — Pluggable providers (Ollama, ONNX, future extensibility)
- **ChromaDB** — Local, persistent vector database with SQLite backend
- **MCP Server** — FastMCP-based async tool execution and stdio/SSE transport
- **Document Pipeline** — Configurable chunking, ingestion, and collection management

For more details, see [INSTALLATION.md](INSTALLATION.md).
