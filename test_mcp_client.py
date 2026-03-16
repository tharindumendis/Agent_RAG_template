r"""
test_mcp_client.py
------------------
Direct MCP stdio client test — bypasses Agent_head entirely.
This tells us definitively whether the RAG server + rag_ingest tool
works over the MCP protocol as a subprocess.

Run from Agent_head directory:
    .venv\Scripts\python.exe ..\Agent_rag\test_mcp_client.py

Or from Agent_rag directory:
    ..\Agent_head\.venv\Scripts\python.exe test_mcp_client.py
"""

import asyncio
import sys
import time
from pathlib import Path

# Use uvx to run the published package
RAG_COMMAND = "uvx"
RAG_ARGS = ["agent-rag-mcp"]
TEST_FILE  = str(Path(__file__).parent / "README.md")  # Use local README for testing
TEST_COL   = "test_collection"


async def run():
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(
        command=RAG_COMMAND,
        args=RAG_ARGS,
        env=None,
    )

    print(f"\n[1] Spawning RAG server subprocess ...")
    print(f"    command: {RAG_COMMAND}")
    print(f"    args: {RAG_ARGS}")

    t0 = time.time()

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            print(f"[1.5] Waiting for server to start...")
            await asyncio.sleep(5)  # Give server time to initialize
            await asyncio.wait_for(session.initialize(), timeout=30.0)  # Add timeout
            print(f"[2] MCP handshake complete in {time.time()-t0:.2f}s")

            tools = await session.list_tools()
            tool_names = [t.name for t in tools.tools]
            print(f"[3] Tools discovered: {tool_names}")

            assert "rag_ingest" in tool_names, "rag_ingest not found!"
            assert "rag_search" in tool_names, "rag_search not found!"
            assert "rag_list_collections" in tool_names, "rag_list_collections not found!"
            assert "rag_delete_collection" in tool_names, "rag_delete_collection not found!"
            assert "rag_delete_items" in tool_names, "rag_delete_items not found!"

            # Test rag_list_collections
            print(f"\n[3.1] Calling rag_list_collections ...")
            t_call = time.time()
            result = await session.call_tool("rag_list_collections", {})
            elapsed = time.time() - t_call
            content = result.content[0].text if result.content else "(empty)"
            print(f"    Result ({elapsed:.1f}s): {content[:200]}")

            # Test rag_ingest
            print(f"\n[4] Calling rag_ingest on '{Path(TEST_FILE).name}' ...")
            t_call = time.time()
            result = await session.call_tool(
                "rag_ingest",
                {"source": TEST_FILE, "collection": TEST_COL},
            )
            elapsed = time.time() - t_call
            content = result.content[0].text if result.content else "(empty)"
            print(f"    Result ({elapsed:.1f}s): {content[:200]}")

            if "ERROR" in content.upper():
                print("[FAIL] rag_ingest returned an error.")
                return False

            # Test rag_search
            print(f"\n[5] Calling rag_search ...")
            t_call = time.time()
            result = await session.call_tool(
                "rag_search",
                {"query": "run localy", "collection": TEST_COL},
            )
            elapsed = time.time() - t_call
            content = result.content[0].text if result.content else "(empty)"
            print(f"    Result ({elapsed:.1f}s): {content[:200]}")

            # Test string ingestion with metadata
            print(f"\n[5.1] Calling rag_ingest with raw string and metadata ...")
            string_source = "This is a secret string source containing classified information about Project Blue."
            t_call = time.time()
            result = await session.call_tool(
                "rag_ingest",
                {
                    "source": string_source, 
                    "collection": TEST_COL,
                    "metadata": {"test_type": "string_ingest", "project": "Blue"}
                },
            )
            elapsed = time.time() - t_call
            content = result.content[0].text if result.content else "(empty)"
            print(f"    Result ({elapsed:.1f}s): {content[:200]}")

            # Test search with metadata filter
            print(f"\n[5.2] Calling rag_search with metadata filter ...")
            t_call = time.time()
            result = await session.call_tool(
                "rag_search",
                {
                    "query": "classified information project", 
                    "collection": TEST_COL,
                    "metadata_filter": {"project": "Blue"}
                },
            )
            elapsed = time.time() - t_call
            content = result.content[0].text if result.content else "(empty)"
            print(f"    Result ({elapsed:.1f}s): {content[:200]}")

            # Test delete items with metadata
            print(f"\n[5.3] Calling rag_delete_items with metadata filter ...")
            t_call = time.time()
            result = await session.call_tool(
                "rag_delete_items",
                {
                    "collection": TEST_COL,
                    "metadata_filter": {"test_type": "string_ingest"}
                },
            )
            elapsed = time.time() - t_call
            content = result.content[0].text if result.content else "(empty)"
            print(f"    Result ({elapsed:.1f}s): {content[:200]}")

            # Cleanup
            print(f"\n[6] Deleting test collection '{TEST_COL}' ...")
            await session.call_tool("rag_delete_collection", {"collection": TEST_COL})
            print("    Done.")

    elapsed_total = time.time() - t0
    print(f"\n[OK] Full MCP round-trip complete in {elapsed_total:.1f}s")
    return True


if __name__ == "__main__":
    ok = asyncio.run(run())
    sys.exit(0 if ok else 1)
