# Groma with LanceDB - LOCAL Embeddings Version

This is the LanceDB version of Groma that runs **completely offline** using **local embeddings** (fastembed), **NOT OpenAI**.

## Key Differences from Original Groma

| Feature | Original Groma (Qdrant) | Groma-LanceDB |
|---------|-------------------------|---------------|
| **Embeddings** | OpenAI API (costs money) | LOCAL fastembed (FREE) |
| **Model** | text-embedding-3-small | AllMiniLML6V2 |
| **API Key Required** | Yes (OpenAI) | No |
| **Internet Required** | Yes | No |
| **Database** | Qdrant (server) | LanceDB (embedded) |
| **Docker Required** | Yes | No |
| **Privacy** | Sends code to OpenAI | 100% local |

## Installation

```bash
cargo install --path . --bin groma-lancedb --features lancedb
```

## Usage

**EXACTLY the same interface as original groma:**

```bash
echo "your search query" | groma-lancedb /path/to/repo --cutoff 0.3
```

### Options

- `<FOLDER_PATH>`: Path to the folder within a Git repository to scan
- `--cutoff <FLOAT>` or `-c <FLOAT>`: Relevance score cutoff (0.0 to 1.0)
- `--lancedb-path <PATH>`: Database directory (default: `.groma_lancedb`)
- `--suppress-updates`: Skip indexing, only search existing data
- `--debug`: Enable debug logging

## How It Works

1. **First Run**: Automatically indexes your repository using LOCAL embeddings
   - Downloads AllMiniLML6V2 model (~45MB) on first use
   - Creates embeddings locally (no API calls)
   - Stores in LanceDB database

2. **Subsequent Runs**: Only re-indexes changed files
   - Checks file hashes
   - Updates only modified files
   - Incremental and efficient

3. **Search**: Finds semantically similar code
   - Embeds your query locally
   - Performs vector similarity search
   - Returns results in same JSON format as original groma

## Example

```bash
# Index and search a repository
echo "database connection pooling" | groma-lancedb ./src --cutoff 0.3

# Output (same format as original groma):
{
  "files_by_relevance": [
    [ 0.85, "src/db/connection.rs" ],
    [ 0.78, "src/db/pool.rs" ],
    [ 0.71, "src/config/database.yml" ]
  ]
}
```

## Performance

- **Indexing**: Comparable speed, no network overhead
- **Search**: Fast local vector search
- **Quality**: Good results, though OpenAI embeddings may be slightly better
- **Cost**: FREE (no API costs)

## Privacy & Security

- **100% Local**: Your code never leaves your machine
- **No API Keys**: No credentials to manage or leak
- **Offline**: Works without internet connection
- **Private**: No telemetry, no tracking

## Compatibility

- Works with same helper scripts (`gromaload.jq`, `gromaprompt.jq`)
- Same JSON output format
- Same command-line interface
- Can be used as drop-in replacement for original groma
