#!/usr/bin/env python3
"""
Simple test script for the groma-lancedb MCP server.
This script sends a list_tools request to verify the server is working.
"""

import json
import subprocess
import sys

def send_request(request):
    """Send a JSON-RPC request to the MCP server via stdin."""
    proc = subprocess.Popen(
        ['/Users/micn/Documents/code/groma/target/release/groma-lancedb', '--mcp-server'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Send request
    request_str = json.dumps(request)
    stdout, stderr = proc.communicate(input=request_str + '\n')
    
    if stderr:
        print(f"Stderr: {stderr}", file=sys.stderr)
    
    # Parse response
    try:
        # MCP uses JSON-RPC, so responses come line by line
        for line in stdout.strip().split('\n'):
            if line:
                response = json.loads(line)
                print(json.dumps(response, indent=2))
    except json.JSONDecodeError as e:
        print(f"Failed to parse response: {e}")
        print(f"Raw output: {stdout}")

# MCP requires an initialization handshake first
print("Testing MCP server with initialization:")

# First send initialization request
init_request = {
    "jsonrpc": "2.0",
    "method": "initialize",
    "params": {
        "protocolVersion": "2024-11-05",
        "capabilities": {
            "roots": {"listChanged": True},
            "sampling": {}
        },
        "clientInfo": {
            "name": "test-client",
            "version": "1.0.0"
        }
    },
    "id": 1
}

print("\nSending initialize request:")
send_request(init_request)

# Then test list_tools request
print("\nTesting list_tools request:")
list_tools_request = {
    "jsonrpc": "2.0",
    "method": "tools/list",
    "id": 2
}
send_request(list_tools_request)