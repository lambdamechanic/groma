#!/bin/bash

# Clean up old log
rm -f /tmp/groma.log

echo "Starting Groma LanceDB MCP server with debug logging..."
echo "Log file: /tmp/groma.log"
echo ""

# Start the MCP server in the background
./target/debug/groma-lancedb mcp --debug &
MCP_PID=$!

echo "MCP server started with PID: $MCP_PID"
echo "Waiting 2 seconds for initialization..."
sleep 2

echo ""
echo "=== Log contents: ==="
cat /tmp/groma.log

echo ""
echo "Killing MCP server..."
kill $MCP_PID 2>/dev/null

echo ""
echo "=== Final log contents: ==="
cat /tmp/groma.log

echo ""
echo "Test complete. Log file is at /tmp/groma.log"
