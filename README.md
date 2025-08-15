# Groma

A semantic code search tool for Git repositories that uses vector embeddings to find relevant files based on natural language queries.

## Two Versions Available

### 1. `groma` - Cloud-based with Qdrant
- Uses **Qdrant** vector database (requires Docker)
- Uses **OpenAI API** for embeddings (requires API key, costs money)
- Higher quality embeddings
- Needs internet connection

### 2. `groma-lancedb` - Fully Local & Free
- Uses **LanceDB** (embedded, no server needed)
- Uses **local fastembed model** (AllMiniLML6V2)
- 100% offline, no API calls
- Completely free
- Your code never leaves your machine

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/groma.git
cd groma

# Build both versions
cargo build --release --features qdrant --bin groma
cargo build --release --features lancedb --bin groma-lancedb

# Install to your PATH
cp target/release/groma ~/.local/bin/
cp target/release/groma-lancedb ~/.local/bin/
```

## Usage

Both versions use the same command-line interface:

```bash
# Basic usage - pipe your query through stdin
echo "authentication logic" | groma /path/to/repo --cutoff 0.3

# Or use the LanceDB version (no setup needed!)
echo "authentication logic" | groma-lancedb /path/to/repo --cutoff 0.3
```

### Options
- `--cutoff` - Similarity threshold (0.0-1.0, default: 0.7)
- `--suppress-updates` - Skip indexing, query existing data only
- `--debug` - Enable debug logging

## Setup Requirements

### For `groma` (Qdrant version)

1. **Start Qdrant Docker container:**
```bash
docker run -p 6334:6334 -v ~/.qdrant_data:/qdrant/storage qdrant/qdrant
```

2. **Set OpenAI API key:**
```bash
export OPENAI_API_KEY='your-api-key-here'
```

3. **Optional - Set custom Qdrant URL:**
```bash
export QDRANT_URL='http://your-qdrant-host:6334'
```

### For `groma-lancedb` (Local version)

**No setup required!** Just run it. The first run will download the embedding model (~80MB) automatically.

## How It Works

1. **Indexing**: On first run, Groma scans your Git repository and creates embeddings for all tracked files
2. **Incremental Updates**: Subsequent runs only process changed files
3. **Semantic Search**: Your query is embedded and compared against the indexed files
4. **Results**: Returns relevant file paths and content snippets in JSON format

## File Filtering

Both versions respect:
- `.gitignore` - Files ignored by Git are not indexed
- `.gromaignore` - Additional patterns to exclude from indexing
- Only Git-tracked files are processed
- Binary files are automatically skipped

## Output Format

Results are returned as JSON for easy integration with other tools:

```json
{
  "path": "src/auth.rs",
  "score": 0.82,
  "content": "impl Authentication {\n    pub fn verify_token..."
}
```

## Integration with Aider

Groma works great with [aider](https://aider.chat) for AI-assisted coding:

```bash
# Use with aider's --read flag
aider --read $(echo "authentication" | groma . --cutoff 0.3 | jq -r '.path')

# Or use the helper script
aider --read $(groma-files "authentication logic" .)
```

## Performance Comparison

| Feature | `groma` (Qdrant) | `groma-lancedb` (Local) |
|---------|------------------|-------------------------|
| Setup Required | Docker + API Key | None |
| Internet Required | Yes | No |
| Cost | OpenAI API fees | Free |
| Privacy | API calls | 100% local |
| Embedding Quality | Higher | Good |
| Speed | Fast after indexing | Fast after indexing |
| Storage | External (Qdrant) | Local (.groma_lancedb) |

## Why Groma?

The name comes from the [groma](https://en.wikipedia.org/wiki/Groma_(surveying)), a surveying instrument used in the Roman Empire. Just as the ancient groma helped surveyors find structure in the physical landscape, this tool helps you find relevant files within your codebase.

## License

MIT
