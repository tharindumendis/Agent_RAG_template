"""
simple_mcp_test.py
------------------
Simple MCP client to test all tools with uvx agent-rag-mcp.
"""

import asyncio
import sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def test_all_tools():
    # Use uvx to run the published package
    params = StdioServerParameters(
        command="uvx",
        args=["agent-rag-mcp"],
        env=None,
    )

    print("Starting MCP client test with uvx agent-rag-mcp...")

    try:
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                print("Waiting for server...")
                await asyncio.sleep(10)  # Give time for download/init
                await asyncio.wait_for(session.initialize(), timeout=60.0)
                print("MCP initialized.")

                # List tools
                tools = await session.list_tools()
                tool_names = [t.name for t in tools.tools]
                print(f"Available tools: {tool_names}")

                # Test rag_list_collections
                print("\nTesting rag_list_collections...")
                result = await session.call_tool("rag_list_collections", {})
                print(f"Result: {result.content[0].text}")

                # Test rag_ingest
                print("\nTesting rag_ingest...")
                result = await session.call_tool(
                    "rag_ingest",
                    {"source": "This is a test document for RAG.", "collection": "test"}
                )
                print(f"Result: {result.content[0].text}")

                # Test rag_search
                print("\nTesting rag_search...")
                result = await session.call_tool(
                    "rag_search",
                    {"query": "test document", "collection": "test"}
                )
                print(f"Result: {result.content[0].text}")

                # Test rag_delete_items
                print("\nTesting rag_delete_items...")
                result = await session.call_tool(
                    "rag_delete_items",
                    {"collection": "test", "metadata_filter": {}}
                )
                print(f"Result: {result.content[0].text}")

                # Test rag_delete_collection
                print("\nTesting rag_delete_collection...")
                result = await session.call_tool("rag_delete_collection", {"collection": "test"})
                print(f"Result: {result.content[0].text}")

                print("\nAll tools tested successfully!")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(test_all_tools())