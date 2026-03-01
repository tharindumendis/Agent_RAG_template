# Agent_rag

Agent_rag is a RAG (Retrieval-Augmented Generation) MCP (Model Context Protocol) Server. It uses ChromaDB and `sentence-transformers` locally to provide a vector store for your intelligent agents.

This server exposes several tools for the orchestrator (`Agent_head`) or any other MCP client to call:

- `rag_ingest`: Ingest documents, directories, or raw text into a collection.
- `rag_search`: Semantic search against the ingested knowledge base.
- `rag_list_collections`: List all active collections.
- `rag_delete_collection`: Delete a specific collection.

## Features

- **Local Embeddings**: Uses `sentence-transformers` (default: `all-MiniLM-L6-v2`) locally, meaning no API keys or external services are required for embedding generation.
- **FastMCP Built-in**: Asynchronous and thread-safe tool execution using `FastMCP`.
- **Easy Configuration**: Configurable via `config.yaml` to set your desired chunk size, collection names, and embedding models.

## Installation & Usage

`Agent_rag` is packaged and distributed via standard Python mechanisms. You can run it effortlessly using [`uv`](https://github.com/astral-sh/uv) without needing to clone the repository or manually manage virtual environments.

### Running with `uvx`

You can run the MCP server directly. `uvx` will automatically download and run the latest version of the `agent-rag` CLI:

```bash
uvx agent-rag
```

**Transport Modes**
By default, the server runs in `stdio` transport mode (designed to be spawned as a subprocess by MCP clients like `Agent_head`).

To run it over HTTP using Server-Sent Events (SSE):

```bash
uvx agent-rag --transport sse --port 8002 --host 0.0.0.0
```

### Specifying a Test Registry (If using TestPyPI)

If you published the package to TestPyPI instead of the main PyPI, run it via:

```bash
uvx --extra-index-url https://test.pypi.org/simple/ --index-strategy unsafe-best-match agent-rag@latest
```

## Integrating with `Agent_head`

To connect this RAG server to your `Agent_head` orchestrator, add the following configuration to your `Agent_head/config.yaml`:

```yaml
memory:
  enabled: true
  backend: "rag"

  # Configure this if backend is set to "rag"
  rag_server:
    command: "uvx"
    args: ["agent-rag"] # Or ["--from", "/path/to/local/Agent_rag", "agent-rag"] for local development
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
   uv run agent-rag
   ```
3. **Build the package**:
   ```bash
   uv build
   ```
