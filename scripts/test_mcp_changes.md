# Testing MCP Server Changes

## Summary of Changes

The MCP server variant of `lancedb_main.rs` has been updated to:

1. **Store databases in `~/.local/share/groma/db_{hash}/`** subdirectories instead of the current directory
2. **Create folder-specific database directories** using a hash of the canonical folder path
3. **Automatically index new folders** when they haven't been seen before
4. **Handle multiple repositories** by maintaining separate database directories for each

## Key Implementation Details

### Database Location
- Each repository gets its own database directory: `~/.local/share/groma/db_{hash}/`
- The hash is the first 16 characters of SHA256(canonical_folder_path)
- The base directory `~/.local/share/groma` is created automatically if it doesn't exist
- The functions `get_groma_db_base_path()` and `get_folder_db_path()` handle this

### Folder-Specific Databases
- Each folder gets its own complete LanceDB database in a separate directory
- The database directory is named `db_{hash}` where hash is based on the canonical folder path
- Within each database, the standard `code_chunks` table is used
- This allows multiple repositories to be indexed without conflicts

### Automatic Indexing
- When a query is made for a folder that hasn't been indexed:
  1. The MCP server checks if a database directory exists for that folder
  2. If not, it starts background indexing via the CLI
  3. Returns a status message that indexing has started
  4. The CLI runs with `--lancedb-path` pointing to the specific database directory

### CLI Mode Updates
- The CLI mode also supports the new database location
- If `--lancedb-path` is not specified or is the default `.groma_lancedb`, it uses `~/.local/share/groma/db_{hash}/`
- Each folder gets its own database directory using the same hashing mechanism
- The CLI can now handle empty stdin (for indexing-only operations)

## Testing Instructions

### Test 1: MCP Server Mode
```bash
# Build with lancedb feature
cargo build --bin groma-lancedb --features lancedb

# Run as MCP server
./target/debug/groma-lancedb mcp

# Send a query for a new folder - it should start indexing
# Send a query for an indexed folder - it should return results
```

### Test 2: CLI Mode with New Database Location
```bash
# Index a repository
echo "" | ./target/debug/groma-lancedb /path/to/repo

# Query the repository
echo "search terms" | ./target/debug/groma-lancedb /path/to/repo

# Check that database is created in ~/.local/share/groma/db_{hash}/
ls -la ~/.local/share/groma/db_{hash}/
```

### Test 3: Multiple Repositories
```bash
# Index multiple repositories via MCP or CLI
# Each should get its own database directory
# Verify database directories are created with unique names based on folder hash
```

## What Changed from Original Behavior

1. **Database Location**: Changed from `.groma_lancedb` in repository directory to `~/.local/share/groma/db_{hash}/`
2. **Database Structure**: Each repository gets its own complete database directory (not just a different table)
3. **MCP Behavior**: Now automatically indexes new folders instead of just checking for a single `code_chunks` table
4. **CLI Behavior**: Can handle empty stdin for indexing-only operations

## Benefits

- **Centralized Root Location**: All indexed repositories stored under `~/.local/share/groma/`
- **Complete Isolation**: Each repository has its own database directory (just like before with `.groma_lancedb`)
- **No Conflicts**: Complete separation between repositories
- **Automatic Indexing**: MCP server handles new repositories transparently
- **Background Processing**: Indexing happens in background, doesn't block queries

## Directory Structure

```
~/.local/share/groma/
├── db_abc123def456789/     # Database for repository 1
│   ├── _versions/
│   ├── _indices/
│   └── code_chunks/
├── db_fedcba987654321/     # Database for repository 2
│   ├── _versions/
│   ├── _indices/
│   └── code_chunks/
└── db_123456789abcdef/     # Database for repository 3
    ├── _versions/
    ├── _indices/
    └── code_chunks/
```