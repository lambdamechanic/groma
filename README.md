# Groma

Groma is a command-line tool that scans a folder within a Git repository, embeds the content of tracked files using OpenAI, stores these embeddings in a Qdrant vector database, and allows you to query for relevant files based on semantic similarity.

## Why "Groma"?

The name comes from the [groma](https://en.wikipedia.org/wiki/Groma_(surveying)), a surveying instrument used in the Roman Empire. 

Just as the ancient groma helped surveyors find straight lines and structure in the physical landscape, this tool helps you find relevant files (the "straight lines") within the structured landscape of your codebase.

## Prerequisites

Groma requires a running Qdrant vector database instance. You can easily start one using Docker:

```bash
mkdir ~/.qdrant_data
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant   -v $HOME/.qdrant_data:/qdrant/storage
```

This command exposes Qdrant's HTTP API on port 6333 and its gRPC API (which Groma uses) on port 6334, and makes sure the data is persisted locally.

You also need an OpenAI API key.

## Usage

1.  **Set Environment Variables:**
    *   Export your OpenAI API key:
        ```bash
        export OPENAI_API_KEY='your-api-key-here'
        ```
    *   Optionally, set the Qdrant URL if it's not running on the default `http://localhost:6334`:
        ```bash
        export QDRANT_URL='http://your-qdrant-host:6334'
        ```

2.  **Install the Tool:**
    Navigate to the cloned repository directory and run:
    ```bash
    cargo install --path .
    ```
    This will compile the `groma` binary and place it in your Cargo bin directory (usually `~/.cargo/bin/`), making it available in your PATH.

3.  **Run Groma:**

    Groma reads your query from standard input (stdin). Pipe your query into the command or type it and press Ctrl+D.

    **Example:** Find files related to "database connection pooling".

    ```bash
    echo "database connection pooling" | groma /path/to/your/repo/subdir --cutoff 0.3
    ```

    **Arguments:**

    *   `<FOLDER_PATH>`: (Required) The path to the folder within a Git repository to scan. This is a positional argument.
    *   `--cutoff <FLOAT>` or `-c <FLOAT>`: (Required) The relevance score cutoff (between 0.0 and 1.0). Only results with a score above this threshold will be shown. A higher value means stricter relevance.
    *   `--openai-key <KEY>`: (Optional) Your OpenAI API key. Defaults to the `OPENAI_API_KEY` environment variable.
    *   `--openai-model <MODEL_NAME>`: (Optional) The OpenAI embedding model to use. Defaults to `text-embedding-3-small`.
    *   `--qdrant-url <URL>`: (Optional) The URL for the Qdrant gRPC endpoint. Defaults to `http://localhost:6334` or the `QDRANT_URL` environment variable.
    *   `--suppress-updates`: (Optional Flag) If present, skips the initial scan, embedding, and upserting steps. Useful if you only want to query existing data.
    *   `--debug`: (Optional Flag) Enables detailed debug logging. By default, Groma produces no log output unless this flag is present or the `RUST_LOG` environment variable is set (e.g., `RUST_LOG=info`).

    **Output:**

    Groma outputs a JSON object containing a list of files sorted by relevance score (highest first):

    ```json
    {
      "files_by_relevance": [
        [ 0.85, "src/db/connection.rs" ],
        [ 0.78, "config/database.yml" ],
        [ 0.71, "docs/architecture.md" ]
      ]
    }
    ```
