#!/bin/bash

echo "Testing fastembed cache location..."
echo ""

# Clean up any existing cache
rm -rf ~/.local/share/groma/fastembed_cache
rm -rf .fastembed_cache

echo "Before running groma-lancedb:"
echo "~/.local/share/groma/fastembed_cache exists: $([ -d ~/.local/share/groma/fastembed_cache ] && echo 'YES' || echo 'NO')"
echo ".fastembed_cache exists: $([ -d .fastembed_cache ] && echo 'YES' || echo 'NO')"
echo ""

# Run a simple query to trigger model download
echo "Running groma-lancedb to trigger model download..."
echo "test query" | ./target/debug/groma-lancedb . --suppress-updates 2>/dev/null || true

echo ""
echo "After running groma-lancedb:"
echo "~/.local/share/groma/fastembed_cache exists: $([ -d ~/.local/share/groma/fastembed_cache ] && echo 'YES' || echo 'NO')"
echo ".fastembed_cache exists: $([ -d .fastembed_cache ] && echo 'YES' || echo 'NO')"

if [ -d ~/.local/share/groma/fastembed_cache ]; then
    echo ""
    echo "Contents of ~/.local/share/groma/fastembed_cache:"
    ls -la ~/.local/share/groma/fastembed_cache/
fi

echo ""
echo "✅ Test complete"
