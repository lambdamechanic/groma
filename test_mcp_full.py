#!/usr/bin/env python3
"""
Complete test script for groma-lancedb MCP server.
This handles the full MCP protocol including initialization.
"""

import json
import subprocess
import sys

def test_mcp_server():
    """Test the MCP server with proper initialization flow."""
    proc = subprocess.Popen(
        ['/Users/micn/Documents/code/groma/target/release/groma-lancedb', '--mcp-server'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    try:
        # 1. Send initialize request
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
        
        print("Sending initialize request...")
        proc.stdin.write(json.dumps(init_request) + '\n')
        proc.stdin.flush()
        
        # Read response (looking for JSON-RPC response)
        while True:
            line = proc.stdout.readline()
            if not line:
                break
            if line.startswith('{'):
                response = json.loads(line)
                print("Initialize response:", json.dumps(response, indent=2))
                break
        
        # 2. Send initialized notification (required by MCP protocol)
        initialized_notification = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized"
        }
        
        print("\nSending initialized notification...")
        proc.stdin.write(json.dumps(initialized_notification) + '\n')
        proc.stdin.flush()
        
        # 3. Now we can send list_tools request
        list_tools_request = {
            "jsonrpc": "2.0",
            "method": "tools/list",
            "id": 2
        }
        
        print("\nSending list_tools request...")
        proc.stdin.write(json.dumps(list_tools_request) + '\n')
        proc.stdin.flush()
        
        # Read response
        while True:
            line = proc.stdout.readline()
            if not line:
                break
            if line.startswith('{'):
                response = json.loads(line)
                print("List tools response:", json.dumps(response, indent=2))
                break
        
        # 4. Test calling the find_code tool
        call_tool_request = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "find_code",
                "arguments": {
                    "query": "MCP server",
                    "folder": "/Users/micn/Documents/code/groma",
                    "cutoff": 0.5
                }
            },
            "id": 3
        }
        
        print("\nSending call_tool request (find_code)...")
        proc.stdin.write(json.dumps(call_tool_request) + '\n')
        proc.stdin.flush()
        
        # Read response
        while True:
            line = proc.stdout.readline()
            if not line:
                break
            if line.startswith('{'):
                response = json.loads(line)
                print("Call tool response:", json.dumps(response, indent=2))
                break
        
        # Terminate the server
        proc.stdin.close()
        proc.wait(timeout=5)
        
    except Exception as e:
        print(f"Error: {e}")
        proc.terminate()
        proc.wait()

if __name__ == "__main__":
    test_mcp_server()
