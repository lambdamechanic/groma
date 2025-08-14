# Groma with LanceDB Support

This document explains how to use Groma with either Qdrant or LanceDB as the vector store backend.

## Building with Different Backends

### With Qdrant (default)
```bash
cargo build --release
# or explicitly:
cargo build --release --features qdrant
```

### With LanceDB
```bash
cargo build --release --no-default-features --features lancedb
```

### With Both Backends
```bash
cargo build --release --features "qdrant,lancedb"
```

## Running with Different Backends

### Using Qdrant (default)
```bash
# Ensure Qdrant is running:
docker run -v $HOME/.qdrant_data:/qdrant/storage -p 6333:6333 -p 6334:6334 qdrant/qdrant

# Run groma
echo "search query" | groma /path/to/repo --cutoff 0.3
# or explicitly:
echo "search query" | groma /path/to/repo --cutoff 0.3 --vector-store qdrant
```

### Using LanceDB
```bash
# No server needed - LanceDB is embedded
echo "search query" | groma /path/to/repo --cutoff 0.3 --vector-store lancedb

# Optionally specify a custom data directory:
echo "search query" | groma /path/to/repo --cutoff 0.3 --vector-store lancedb --lancedb-path /custom/path/to/lancedb
```

## Environment Variables

### For Qdrant
- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `QDRANT_URL`: Qdrant server URL (default: http://localhost:6334)
- `VECTOR_STORE`: Set to "qdrant" (default)

### For LanceDB
- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `LANCEDB_PATH`: Path to LanceDB data directory (default: ~/.groma/lancedb)
- `VECTOR_STORE`: Set to "lancedb"

## Advantages of Each Backend

### Qdrant
- **Client-server architecture**: Can be shared between multiple users/machines
- **Production-ready**: Mature, battle-tested vector database
- **Advanced features**: Rich query capabilities and filtering options
- **Scalability**: Can handle very large datasets

### LanceDB
- **Embedded/serverless**: No separate server process needed
- **Rust-native**: Written in Rust, potentially better integration
- **Simple deployment**: Just a data directory, no Docker required
- **Lower latency**: Direct file access without network overhead
- **Portable**: Easy to move data between machines

## Migration Between Backends

Currently, direct migration between backends is not supported. If you need to switch:

1. Re-index your codebase with the new backend
2. The collection names and data structure remain compatible

## Troubleshooting

### Qdrant Issues
- Ensure the Qdrant server is running
- Check that the gRPC port (6334) is accessible
- Verify the QDRANT_URL is correct

### LanceDB Issues
- Ensure the data directory has write permissions
- Check available disk space
- If corrupted, delete the LanceDB directory and re-index

## Performance Considerations

- **Indexing Speed**: LanceDB may be faster for initial indexing due to no network overhead
- **Query Speed**: Both should provide similar query performance for typical use cases
- **Memory Usage**: LanceDB uses memory-mapped files, which can be more efficient
- **Concurrent Access**: Qdrant handles concurrent access better in multi-user scenarios
