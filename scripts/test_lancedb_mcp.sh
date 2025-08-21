#!/bin/bash
# Test the LanceDB MCP server functionality
./target/debug/groma-lancedb mcp --debug 2>&1 | head -20
