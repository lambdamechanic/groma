use anyhow::{anyhow, Context, Result};
use arrow_array::{ArrayRef, FixedSizeListArray, Float32Array, RecordBatch, RecordBatchIterator, StringArray, Int32Array, types::Float32Type};
use arrow_schema::{DataType, Field, Schema};
use clap::{Parser, Subcommand};
use fastembed::{TextEmbedding, EmbeddingModel, InitOptions};
use git2::{Repository, DiffOptions, Oid};
use lancedb::query::{ExecutableQuery, QueryBase};
use lancedb::table::Table;
use serde::{Serialize, Deserialize};
use serde_json;
use sha2::{Digest, Sha256};
use std::{
    collections::{HashMap, HashSet},
    fs,
    io::{self, Read, BufWriter},
    path::{Path, PathBuf},
    sync::Arc,
};
use tracing::{debug, info, warn, error};
use tracing_subscriber::{EnvFilter, FmtSubscriber};
use ignore::gitignore::GitignoreBuilder;
use uuid::Uuid;
use futures::TryStreamExt;

// MCP server imports
use mcp_core::{
    content::Content,
    handler::{PromptError, ResourceError, ToolError},
    protocol::ServerCapabilities,
    resource::Resource,
    tool::{Tool, ToolAnnotations},
};
use mcp_server::{
    router::{CapabilitiesBuilder, RouterService},
    ByteTransport, Router, Server,
};
use std::{
    future::Future,
    pin::Pin,
};
use tokio::io::{stdin as async_stdin, stdout as async_stdout};

pub const EMBEDDING_DIMENSION: usize = 384; // AllMiniLML6V2 dimension

/// Command-line arguments - matches the original groma interface
#[derive(Parser, Debug)]
#[command(author, version, about = "Groma with LanceDB - uses LOCAL embeddings (fastembed), NOT OpenAI", long_about = None)]
struct Args {
    /// Path to the folder within a Git repository to scan.
    folder: Option<PathBuf>,

    /// Relevance cutoff for results (e.g., 0.7). Only results with a score
    /// above this threshold will be shown.
    #[arg(short, long, default_value_t = 0.7)]
    cutoff: f32,

    /// Path to LanceDB database directory
    #[arg(long, default_value = ".groma_lancedb")]
    lancedb_path: String,

    /// Suppress checking for file updates and upserting to vector store.
    /// Useful for querying existing data without modifying the index.
    #[arg(long)]
    suppress_updates: bool,

    /// Enable debug logging.
    #[arg(long)]
    debug: bool,
    
    /// Subcommands
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    #[command(about = "Run as an MCP server using stdio for communication")]
    Mcp {
        /// Enable debug logging
        #[arg(long)]
        debug: bool,
    },
}

/// Stores the state between runs, specifically the last processed Git commit OID.
#[derive(Serialize, Deserialize, Debug, Clone)]
struct GromaState {
    last_processed_oid: String,
}

// State Management Functions

/// Returns the expected path for the .gromastate file within the repository's workdir.
fn get_state_file_path(repo: &Repository) -> Result<PathBuf> {
    let workdir = repo
        .workdir()
        .ok_or_else(|| anyhow!("Cannot get state file path: repository is bare"))?;
    Ok(workdir.join(".gromastate"))
}

/// Loads the GromaState from the .gromastate file.
/// Returns Ok(None) if the file doesn't exist (first run).
fn load_state(repo: &Repository) -> Result<Option<GromaState>> {
    let state_file_path = get_state_file_path(repo)?;
    if !state_file_path.exists() {
        info!(
            "No previous state file found at '{}'. Assuming first run.",
            state_file_path.display()
        );
        return Ok(None);
    }

    let file = fs::File::open(&state_file_path)
        .with_context(|| format!("Failed to open state file: {}", state_file_path.display()))?;
    let reader = std::io::BufReader::new(file);
    let state: GromaState = serde_json::from_reader(reader).with_context(|| {
        format!(
            "Failed to deserialize state from: {}",
            state_file_path.display()
        )
    })?;
    info!("Loaded previous state (OID: {})", state.last_processed_oid);
    Ok(Some(state))
}

/// Saves the given GromaState to the .gromastate file.
fn save_state(repo: &Repository, state: &GromaState) -> Result<()> {
    let state_file_path = get_state_file_path(repo)?;
    debug!(
        "Saving current state (OID: {}) to '{}'",
        state.last_processed_oid,
        state_file_path.display()
    );
    let file = fs::File::create(&state_file_path)
        .with_context(|| format!("Failed to create state file: {}", state_file_path.display()))?;
    serde_json::to_writer_pretty(BufWriter::new(file), state).with_context(|| {
        format!(
            "Failed to serialize state to: {}",
            state_file_path.display()
        )
    })?;
    info!("Saved current state (OID: {})", state.last_processed_oid);
    Ok(())
}

// LanceDB-specific vector store implementation
struct LanceDBStore {
    table: Table,
    embedding_dimension: usize,
}

impl LanceDBStore {
    async fn new(db_path: &str, embedding_dimension: usize) -> Result<Self> {
        // Create LanceDB connection
        let db = lancedb::connect(db_path).execute().await?;
        
        // Define schema
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("path", DataType::Utf8, false),
            Field::new("hash", DataType::Utf8, false),
            Field::new("chunk_index", DataType::Int32, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    embedding_dimension as i32,
                ),
                false,
            ),
        ]));

        // Try to open existing table or create new one
        let table = match db.open_table("code_chunks").execute().await {
            Ok(table) => table,
            Err(_) => {
                info!("Creating new LanceDB table 'code_chunks'");
                // Create empty table with schema
                let empty_batch = RecordBatch::new_empty(schema.clone());
                let batch_iter = RecordBatchIterator::new(
                    vec![Ok(empty_batch)], 
                    schema.clone()
                );
                db.create_table("code_chunks", Box::new(batch_iter))
                    .execute()
                    .await?
            }
        };

        Ok(Self {
            table,
            embedding_dimension,
        })
    }
    
    async fn upsert_chunks(&self, chunks: Vec<ChunkData>) -> Result<()> {
        if chunks.is_empty() {
            return Ok(());
        }

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("path", DataType::Utf8, false),
            Field::new("hash", DataType::Utf8, false),
            Field::new("chunk_index", DataType::Int32, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    self.embedding_dimension as i32,
                ),
                false,
            ),
        ]));

        let mut ids = Vec::new();
        let mut paths = Vec::new();
        let mut hashes = Vec::new();
        let mut chunk_indices = Vec::new();
        let mut flat_vectors = Vec::new();

        for chunk in chunks {
            ids.push(chunk.id);
            paths.push(chunk.path);
            hashes.push(chunk.hash);
            chunk_indices.push(chunk.chunk_index);
            for val in chunk.vector {
                flat_vectors.push(Some(val));
            }
        }

        let vector_array = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
            flat_vectors.chunks(self.embedding_dimension).map(|chunk| Some(chunk.to_vec())),
            self.embedding_dimension as i32,
        );

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(ids)),
                Arc::new(StringArray::from(paths)),
                Arc::new(StringArray::from(hashes)),
                Arc::new(Int32Array::from(chunk_indices)),
                Arc::new(vector_array) as ArrayRef,
            ],
        )?;

        let batch_iter = RecordBatchIterator::new(
            vec![Ok(batch)],
            schema.clone()
        );
        self.table.add(Box::new(batch_iter)).execute().await?;
        Ok(())
    }

    async fn search(&self, query_vector: Vec<f32>, limit: usize) -> Result<Vec<SearchResult>> {
        let normalized = normalize_vector(query_vector);
        
        let results = self.table
            .vector_search(normalized)?
            .limit(limit)
            .execute()
            .await?;

        let batch = results.try_collect::<Vec<_>>().await?;
        
        let mut search_results = Vec::new();
        for batch in batch {
            let ids = batch.column_by_name("id")
                .ok_or_else(|| anyhow!("Missing id column"))?
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| anyhow!("Invalid id column type"))?;
            
            let paths = batch.column_by_name("path")
                .ok_or_else(|| anyhow!("Missing path column"))?
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| anyhow!("Invalid path column type"))?;
            
            let distances = batch.column_by_name("_distance")
                .ok_or_else(|| anyhow!("Missing distance column"))?
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| anyhow!("Invalid distance column type"))?;

            for i in 0..batch.num_rows() {
                search_results.push(SearchResult {
                    id: ids.value(i).to_string(),
                    path: paths.value(i).to_string(),
                    distance: distances.value(i),
                });
            }
        }

        Ok(search_results)
    }

    async fn delete_file_chunks(&self, file_path: &str) -> Result<()> {
        // LanceDB doesn't have a direct delete by filter, so we'll need to rebuild
        // For now, we'll just log a warning
        warn!("Deletion not implemented for LanceDB - file {} will be re-indexed", file_path);
        Ok(())
    }
}

#[derive(Debug)]
struct ChunkData {
    id: String,
    path: String,
    hash: String,
    chunk_index: i32,
    vector: Vec<f32>,
}

#[derive(Debug, Serialize)]
struct SearchResult {
    id: String,
    path: String,
    distance: f32,
}

pub fn normalize_vector(vector: Vec<f32>) -> Vec<f32> {
    let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        vector.iter().map(|x| x / norm).collect()
    } else {
        vector
    }
}

fn chunk_text(text: &str, chunk_size: usize, overlap: usize) -> Vec<String> {
    let lines: Vec<&str> = text.lines().collect();
    let mut chunks = Vec::new();
    let mut i = 0;
    
    while i < lines.len() {
        let end = (i + chunk_size).min(lines.len());
        let chunk = lines[i..end].join("\n");
        if !chunk.trim().is_empty() {
            chunks.push(chunk);
        }
        
        if end >= lines.len() {
            break;
        }
        
        i += chunk_size - overlap;
    }
    
    chunks
}

fn calculate_file_hash(content: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    format!("{:x}", hasher.finalize())
}

fn initialize_logging(debug_mode: bool) {
    use std::fs::OpenOptions;
    
    // Open log file in /tmp
    let log_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open("/tmp/groma.log")
        .expect("Failed to open log file");
    
    let filter = if debug_mode {
        EnvFilter::new("debug")
    } else {
        EnvFilter::from_default_env()
            .add_directive("groma=info".parse().unwrap())
    };

    let subscriber = FmtSubscriber::builder()
        .with_env_filter(filter)
        .with_target(true)
        .with_thread_ids(true)
        .with_thread_names(true)
        .with_file(true)
        .with_line_number(true)
        .with_writer(log_file)
        .finish();

    tracing::subscriber::set_global_default(subscriber)
        .expect("Failed to set tracing subscriber");
    
    // Log startup
    info!("=== Groma LanceDB starting at {} ===", chrono::Local::now().format("%Y-%m-%d %H:%M:%S"));
    info!("Debug mode: {}", debug_mode);
}

async fn perform_file_updates(
    model: &mut TextEmbedding,
    store: &LanceDBStore,
    canonical_folder_path: &Path,
    repo_path: &Path,
) -> Result<()> {
    // Collect all the file operations first without holding git2 types
    let (files_to_index, paths_to_delete, head_commit_oid) = {
        let repo = Repository::discover(repo_path)?;
        let workdir = repo.workdir().ok_or_else(|| anyhow!("Failed to get working directory"))?;
        
        // Initialize gitignore builder
        let mut gitignore_builder = GitignoreBuilder::new(canonical_folder_path);
        gitignore_builder.add_line(None, ".git")?;
        gitignore_builder.add_line(None, ".env")?;
        gitignore_builder.add_line(None, "*.log")?;
        gitignore_builder.add_line(None, "*.tmp")?;
        gitignore_builder.add_line(None, "target")?;
        gitignore_builder.add_line(None, "node_modules")?;
        gitignore_builder.add_line(None, ".gromadb")?;
        
        // Add .gitignore if it exists
        let gitignore_path = canonical_folder_path.join(".gitignore");
        if gitignore_path.exists() {
            gitignore_builder.add(&gitignore_path);
            info!("Using .gitignore file");
        }
        
        // Add .gromaignore if it exists
        let gromaignore_path = canonical_folder_path.join(".gromaignore");
        if gromaignore_path.exists() {
            gitignore_builder.add(&gromaignore_path);
            info!("Using .gromaignore file");
        }
        
        let gitignore = gitignore_builder.build()?;
        
        // Load previous state
        info!("Loading previous state...");
        let previous_state = load_state(&repo)?;
        let head_commit_oid = repo.head()?.peel_to_commit()?.id();
        info!("Current HEAD OID: {}", head_commit_oid);
        
        let previous_tree = match previous_state {
            Some(ref state) => {
                let oid = Oid::from_str(&state.last_processed_oid)
                    .with_context(|| format!("Failed to parse OID: {}", state.last_processed_oid))?;
                match repo.find_commit(oid) {
                    Ok(commit) => match commit.tree() {
                        Ok(tree) => Some(tree),
                        Err(e) => {
                            warn!(
                                "Failed to find tree for previous OID {}: {}. Processing all files.",
                                state.last_processed_oid, e
                            );
                            None
                        }
                    },
                    Err(e) => {
                        warn!(
                            "Failed to find commit for previous OID {}: {}. Processing all files.",
                            state.last_processed_oid, e
                        );
                        None
                    }
                }
            }
            None => None, // First run, compare against empty tree
        };
        
        // Prepare to collect files to process
        let mut files_to_index = Vec::new();
        let mut paths_to_delete = Vec::new();
        let mut processed_paths: HashSet<PathBuf> = HashSet::new();
        
        // Configure diff options
        let mut diff_opts = DiffOptions::new();
        diff_opts.include_ignored(false);
        diff_opts.include_untracked(false);
        diff_opts.include_typechange(true);
        
        // Limit diff to the target folder
        let pathspec = canonical_folder_path.strip_prefix(workdir).unwrap_or(canonical_folder_path);
        diff_opts.pathspec(pathspec);
        info!("Using pathspec for diff: {}", pathspec.display());
        
        // Get diff between previous tree and working directory
        let diff = repo.diff_tree_to_workdir(previous_tree.as_ref(), Some(&mut diff_opts))?;
        
        info!(
            "Git diff found {} changed items within the target folder.",
            diff.deltas().len()
        );
        
        // Helper function to check if file should be processed
        let should_process_file = |path: &Path| -> bool {
            let extension = path.extension().and_then(|e| e.to_str()).unwrap_or("");
            let allowed_extensions = vec![
                "rs", "py", "js", "ts", "jsx", "tsx", "java", "c", "cpp", "cc", "cxx", 
                "h", "hpp", "go", "rb", "php", "swift", "kt", "scala", "r", "m", "mm",
                "cs", "vb", "fs", "ml", "clj", "ex", "exs", "erl", "hrl", "lua", "pl",
                "sh", "bash", "zsh", "fish", "ps1", "psm1", "psd1", "bat", "cmd",
                "md", "txt", "rst", "adoc", "tex", "org", "wiki",
                "yml", "yaml", "toml", "json", "xml", "html", "css", "scss", "sass", "less",
                "sql", "dockerfile", "makefile", "cmake", "gradle", "maven", "sbt",
                "vim", "el", "lisp", "scm", "rkt", "hs", "lhs", "purs", "elm", "nim"
            ];
            
            allowed_extensions.contains(&extension) || 
                path.file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| n == "Makefile" || n == "Dockerfile" || n == "Jenkinsfile" || n == "Vagrantfile")
                    .unwrap_or(false)
        };
        
        // Process first run - all files in HEAD
        if previous_tree.is_none() {
            info!("First run detected. Processing all tracked files in target folder...");
            let current_tree = repo.head()?.peel_to_tree()?;
            current_tree.walk(git2::TreeWalkMode::PreOrder, |root, entry| {
                if let Some(entry_name) = entry.name() {
                    let full_path = workdir.join(root).join(entry_name);
                    
                    // Check if within target folder
                    if !full_path.starts_with(canonical_folder_path) {
                        return git2::TreeWalkResult::Ok;
                    }
                    
                    // Check gitignore
                    let rel_path = full_path.strip_prefix(workdir).unwrap_or(&full_path);
                    if gitignore.matched(rel_path, false).is_ignore() {
                        debug!("Skipping ignored file: {}", rel_path.display());
                        return git2::TreeWalkResult::Ok;
                    }
                    
                    // Only process files, not directories
                    if entry.kind() == Some(git2::ObjectType::Blob) {
                        if should_process_file(&full_path) {
                            // Check file size (skip files > 1MB)
                            if let Ok(metadata) = fs::metadata(&full_path) {
                                if metadata.len() > 1_000_000 {
                                    debug!("Skipping large file (>1MB): {}", full_path.display());
                                    return git2::TreeWalkResult::Ok;
                                }
                            }
                            
                            // Read file content
                            if let Ok(content) = fs::read_to_string(&full_path) {
                                let path_str = full_path.strip_prefix(canonical_folder_path)
                                    .unwrap_or(&full_path)
                                    .to_string_lossy()
                                    .to_string();
                                let hash = calculate_file_hash(&content);
                                files_to_index.push((path_str, content, hash));
                                processed_paths.insert(full_path);
                            }
                        }
                    }
                }
                git2::TreeWalkResult::Ok
            })?;
        } else {
            // Process diff for incremental updates
            for delta in diff.deltas() {
                let new_file = delta.new_file();
                let old_file = delta.old_file();
                
                match delta.status() {
                    git2::Delta::Added | git2::Delta::Modified | git2::Delta::Typechange => {
                        if let Some(path) = new_file.path() {
                            let full_path = workdir.join(path);
                            
                            // Check if within target folder
                            if !full_path.starts_with(canonical_folder_path) {
                                continue;
                            }
                            
                            // Check gitignore
                            if gitignore.matched(path, false).is_ignore() {
                                debug!("Skipping ignored file: {}", path.display());
                                continue;
                            }
                            
                            if should_process_file(&full_path) && !processed_paths.contains(&full_path) {
                                // Check file size (skip files > 1MB)
                                if let Ok(metadata) = fs::metadata(&full_path) {
                                    if metadata.len() > 1_000_000 {
                                        debug!("Skipping large file (>1MB): {}", full_path.display());
                                        continue;
                                    }
                                }
                                
                                // Read file content
                                if let Ok(content) = fs::read_to_string(&full_path) {
                                    let path_str = full_path.strip_prefix(canonical_folder_path)
                                        .unwrap_or(&full_path)
                                        .to_string_lossy()
                                        .to_string();
                                    let hash = calculate_file_hash(&content);
                                    files_to_index.push((path_str, content, hash));
                                    processed_paths.insert(full_path);
                                }
                            }
                        }
                    }
                    git2::Delta::Deleted => {
                        if let Some(path) = old_file.path() {
                            let full_path = workdir.join(path);
                            if full_path.starts_with(canonical_folder_path) {
                                let path_str = full_path.strip_prefix(canonical_folder_path)
                                    .unwrap_or(&full_path)
                                    .to_string_lossy()
                                    .to_string();
                                paths_to_delete.push(path_str);
                            }
                        }
                    }
                    git2::Delta::Renamed => {
                        // Handle rename as delete + add
                        if let Some(old_path) = old_file.path() {
                            let full_path = workdir.join(old_path);
                            if full_path.starts_with(canonical_folder_path) {
                                let path_str = full_path.strip_prefix(canonical_folder_path)
                                    .unwrap_or(&full_path)
                                    .to_string_lossy()
                                    .to_string();
                                paths_to_delete.push(path_str);
                            }
                        }
                        if let Some(new_path) = new_file.path() {
                            let full_path = workdir.join(new_path);
                            if full_path.starts_with(canonical_folder_path) && 
                               should_process_file(&full_path) && 
                               !processed_paths.contains(&full_path) {
                                // Check file size (skip files > 1MB)
                                if let Ok(metadata) = fs::metadata(&full_path) {
                                    if metadata.len() > 1_000_000 {
                                        debug!("Skipping large file (>1MB): {}", full_path.display());
                                        continue;
                                    }
                                }
                                
                                if let Ok(content) = fs::read_to_string(&full_path) {
                                    let path_str = full_path.strip_prefix(canonical_folder_path)
                                        .unwrap_or(&full_path)
                                        .to_string_lossy()
                                        .to_string();
                                    let hash = calculate_file_hash(&content);
                                    files_to_index.push((path_str, content, hash));
                                    processed_paths.insert(full_path);
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        
        (files_to_index, paths_to_delete, head_commit_oid.to_string())
    };

    // Handle deletions
    for path in &paths_to_delete {
        info!("Deleting chunks for removed file: {}", path);
        store.delete_file_chunks(path).await?;
    }
    
    if files_to_index.is_empty() && paths_to_delete.is_empty() {
        info!("No files need updating. Index is up to date.");
        // Still save state to update the OID - rediscover repo
        {
            let repo = Repository::discover(repo_path)?;
            let new_state = GromaState {
                last_processed_oid: head_commit_oid.clone(),
            };
            save_state(&repo, &new_state)?;
        }
        return Ok(());
    }
    
    info!("Processing {} files to index and {} files to delete", 
         files_to_index.len(), paths_to_delete.len());
    
    // Process files and generate embeddings
    for (path, content, hash) in files_to_index {
        debug!("Processing file: {}", path);
        
        // Delete existing chunks for this file
        store.delete_file_chunks(&path).await?;
        
        // Chunk the content
        let chunks = chunk_text(&content, 50, 10); // Chunk by lines for simplicity
        if chunks.is_empty() {
            continue;
        }
        
        // Generate embeddings
        let embeddings = model.embed(chunks.clone(), None)?;
        
        // Prepare chunk data
        let mut chunk_data = Vec::new();
        for (i, (_chunk_text, embedding)) in chunks.iter().zip(embeddings.iter()).enumerate() {
            let id = format!("{}#{}", 
                Uuid::new_v5(&Uuid::NAMESPACE_OID, path.as_bytes()),
                i
            );
            
            chunk_data.push(ChunkData {
                id,
                path: path.clone(),
                hash: hash.clone(),
                chunk_index: i as i32,
                vector: normalize_vector(embedding.to_vec()),
            });
        }
        
        // Upsert to LanceDB
        store.upsert_chunks(chunk_data).await?;
    }
    
    info!("Indexing complete.");
    
    // Save the current state - rediscover repo to save state
    {
        let repo = Repository::discover(repo_path)?;
        let new_state = GromaState {
            last_processed_oid: head_commit_oid,
        };
        save_state(&repo, &new_state)?;
    }
    
    Ok(())
}

async fn process_query(
    query: &str,
    model: &mut TextEmbedding,
    store: &LanceDBStore,
    cutoff: f32,
    is_mcp_mode: bool,
) -> Result<String> {
    // Generate embedding for query
    let query_embedding = model.embed(vec![query.to_string()], None)?;
    let query_vector = query_embedding[0].to_vec();
    
    // Search in LanceDB
    let results = store.search(query_vector, 100).await?;
    
    // Convert distance to similarity score (assuming cosine distance)
    // LanceDB returns L2 distance, so we need to convert
    let mut files_by_relevance: HashMap<String, f32> = HashMap::new();
    
    for result in results {
        // Convert L2 distance to similarity score
        // Similarity = 1 / (1 + distance)
        let similarity = 1.0 / (1.0 + result.distance);
        
        if similarity >= cutoff {
            files_by_relevance
                .entry(result.path.clone())
                .and_modify(|s| *s = s.max(similarity))
                .or_insert(similarity);
        }
    }
    
    // Sort by relevance
    let mut sorted_files: Vec<_> = files_by_relevance.into_iter().collect();
    sorted_files.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    
    // Format as JSON output (matching original groma format)
    let output = serde_json::json!({
        "files_by_relevance": sorted_files.into_iter()
            .map(|(path, score)| vec![
                serde_json::Value::from(score),
                serde_json::Value::from(path)
            ])
            .collect::<Vec<_>>()
    });
    
    let json_output = serde_json::to_string_pretty(&output)?;
    
    // CRITICAL: Only print to stdout when NOT in MCP mode
    if !is_mcp_mode {
        println!("{}", json_output);
    } else {
        // In MCP mode, log the output instead
        debug!("Query results (MCP mode, not printing to stdout): {}", json_output);
    }
    
    Ok(json_output)
}

#[tokio::main]
async fn main() -> Result<()> {
    // Set up panic handler to log panics to file
    std::panic::set_hook(Box::new(|panic_info| {
        use std::fs::OpenOptions;
        use std::io::Write;
        
        if let Ok(mut file) = OpenOptions::new()
            .create(true)
            .append(true)
            .open("/tmp/groma.log")
        {
            let _ = writeln!(file, "PANIC: {}", panic_info);
            let _ = writeln!(file, "Backtrace: {:?}", std::backtrace::Backtrace::capture());
        }
    }));
    
    let args = Args::parse();
    
    // Check if we're running in MCP mode
    match &args.command {
        Some(Commands::Mcp { debug }) => {
            let debug_mode = *debug;
            initialize_logging(debug_mode);
            info!("Starting Groma LanceDB in MCP server mode (debug={})", debug_mode);
            debug!("About to call run_lancedb_mcp_server()");
            
            let result = run_lancedb_mcp_server().await;
            match &result {
                Ok(_) => info!("MCP server exited successfully"),
                Err(e) => error!("MCP server failed: {:?}", e),
            }
            return result;
        }
        None => {
            // Standard CLI mode
            initialize_logging(args.debug);
            
            let folder = args.folder.as_ref().ok_or_else(|| 
                anyhow!("Folder argument is required when not using MCP mode"))?;
            
            info!("Starting Groma with LanceDB (using LOCAL fastembed, NOT OpenAI)...");
            
            // Initialize the LOCAL embedding model
            info!("Initializing LOCAL embedding model (AllMiniLML6V2)...");
            let mut model = TextEmbedding::try_new(
                InitOptions::new(EmbeddingModel::AllMiniLML6V2)
                    .with_show_download_progress(false)
            )?;
            info!("LOCAL embedding model initialized (no API calls required).");
            
            // Canonicalize folder path
            let canonical_folder_path = fs::canonicalize(&folder)?;
            info!("Using folder: {}", canonical_folder_path.display());
            
            // Find Git repository
            let _repo = Repository::discover(&canonical_folder_path)
                .with_context(|| format!("Failed to find Git repository for {}", canonical_folder_path.display()))?;
            
            // Determine LanceDB path - use provided path or default to folder-specific path in ~/.local/share/groma
            let lancedb_path = if args.lancedb_path != ".groma_lancedb" {
                // User provided a custom path
                args.lancedb_path.clone()
            } else {
                // Use folder-specific database directory in ~/.local/share/groma
                get_folder_db_path(&canonical_folder_path)?
            };
            
            // Initialize LanceDB with standard table name (since each folder has its own DB)
            info!("Initializing LanceDB at: {}", lancedb_path);
            let store = LanceDBStore::new(&lancedb_path, EMBEDDING_DIMENSION).await?;
            
            // Perform file updates unless suppressed
            if !args.suppress_updates {
                perform_file_updates(
                    &mut model,
                    &store,
                    &canonical_folder_path,
                    &canonical_folder_path,
                ).await?;
            } else {
                info!("Skipping file updates (--suppress-updates specified).");
            }
            
            // Read query from stdin (optional - if no input, just index)
            info!("Reading query from stdin...");
            let mut query = String::new();
            let stdin_result = io::stdin().read_to_string(&mut query);
            
            match stdin_result {
                Ok(_) => {
                    let query = query.trim();
                    if !query.is_empty() {
                        info!("Processing query: {}", query);
                        // Not in MCP mode, so stdout is allowed
                        process_query(query, &mut model, &store, args.cutoff, false).await?;
                    } else {
                        info!("No query provided - indexing complete.");
                    }
                }
                Err(e) => {
                    // If there's an error reading stdin (e.g., it's closed), just finish indexing
                    info!("No stdin available ({}), indexing complete.", e);
                }
            }
            
            Ok(())
        }
    }
}


// ====== MCP Server Implementation ======

/// Get the path to the groma database directory in ~/.local/share/groma
fn get_groma_db_base_path() -> Result<PathBuf> {
    let home_dir = std::env::var("HOME")
        .map_err(|_| anyhow!("Could not determine home directory"))?;
    let db_dir = PathBuf::from(home_dir)
        .join(".local")
        .join("share")
        .join("groma");
    
    // Create the base directory if it doesn't exist
    fs::create_dir_all(&db_dir)
        .with_context(|| format!("Failed to create database directory: {}", db_dir.display()))?;
    
    Ok(db_dir)
}

/// Get a folder-specific database path based on the canonical folder path
fn get_folder_db_path(canonical_folder_path: &Path) -> Result<String> {
    let base_path = get_groma_db_base_path()?;
    
    // Create a unique subdirectory name based on the folder path
    // Use a hash of the canonical path to ensure uniqueness and avoid path issues
    let mut hasher = Sha256::new();
    hasher.update(canonical_folder_path.to_string_lossy().as_bytes());
    let folder_hash = format!("{:x}", hasher.finalize());
    let db_name = format!("db_{}", &folder_hash[0..16]); // Use first 16 chars of hash
    
    let db_path = base_path.join(db_name);
    
    // Create the folder-specific directory if it doesn't exist
    fs::create_dir_all(&db_path)
        .with_context(|| format!("Failed to create folder database directory: {}", db_path.display()))?;
    
    Ok(db_path.to_string_lossy().to_string())
}

/// A router that wraps our LanceDB-based Groma functionality to expose it via MCP
#[derive(Clone)]
pub struct GromaLanceDBRouter {}

impl GromaLanceDBRouter {
    pub fn new() -> Self {
        Self {}
    }

    /// Process a query and return the results using LanceDB backend
    async fn process_mcp_query(&self, query: String, folder: String, cutoff: f32) -> Result<String, ToolError> {
        info!("process_mcp_query called with query: '{}', folder: '{}', cutoff: {}", query, folder, cutoff);
        
        // Initialize the embedding model
        debug!("Initializing embedding model...");
        let mut model = TextEmbedding::try_new(
            InitOptions::new(EmbeddingModel::AllMiniLML6V2).with_show_download_progress(false),
        ).map_err(|e| {
            error!("Failed to initialize embedding model: {}", e);
            ToolError::ExecutionError(format!("Failed to initialize embedding model: {}", e))
        })?;
        info!("Embedding model initialized successfully");
        
        // Prepare paths
        let folder_path = PathBuf::from(&folder);
        let canonical_folder_path = folder_path.canonicalize()
            .map_err(|_| {
                ToolError::ExecutionError(format!("Folder not found: {}", folder))
            })?;
        
        // Get the folder-specific LanceDB path in ~/.local/share/groma/db_{hash}
        let lancedb_path = get_folder_db_path(&canonical_folder_path)
            .map_err(|e| ToolError::ExecutionError(format!("Failed to get database path: {}", e)))?;
        
        // Initialize LanceDB Store for this folder's database
        let store = LanceDBStore::new(&lancedb_path, EMBEDDING_DIMENSION).await
            .map_err(|e| ToolError::ExecutionError(format!("Failed to initialize LanceDB store: {}", e)))?;
        
        // Perform file updates in a block scope to avoid holding git2 types across await
        {
            // Find Git repository
            let _repo = Repository::discover(&canonical_folder_path)
                .map_err(|e| ToolError::ExecutionError(format!("Failed to find Git repository: {}", e)))?;
            
            // Always perform file updates to ensure index is current
            // Just like the CLI - index if needed
            let _args = Args {
                folder: Some(canonical_folder_path.clone()),
                cutoff,
                lancedb_path: lancedb_path.clone(),
                suppress_updates: false,
                debug: false,
                command: None,
            };
            
            perform_file_updates(&mut model, &store, &canonical_folder_path, &canonical_folder_path).await
                .map_err(|e| ToolError::ExecutionError(format!("Failed to update index: {}", e)))?;
        }
        
        // Generate embedding for the query
        let query_embedding = model.embed(vec![query.clone()], None)
            .map_err(|e| ToolError::ExecutionError(format!("Failed to embed query: {}", e)))?;
        
        if query_embedding.is_empty() || query_embedding[0].is_empty() {
            return Err(ToolError::ExecutionError("Failed to generate query embedding".to_string()));
        }
        
        // Search using our existing logic
        let query_vector = query_embedding[0].to_vec();
        let results = store.search(query_vector, 100).await
            .map_err(|e| ToolError::ExecutionError(format!("Failed to execute search: {}", e)))?;
        
        // Process results (reuse existing logic)
        let mut files_by_relevance: HashMap<String, f32> = HashMap::new();
        
        for result in results {
            // Convert L2 distance to similarity score
            let similarity = 1.0 / (1.0 + result.distance);
            
            if similarity >= cutoff {
                // Make path relative to the query folder
                let relative_path = if result.path.starts_with(canonical_folder_path.to_str().unwrap_or("")) {
                    result.path.strip_prefix(canonical_folder_path.to_str().unwrap_or(""))
                        .unwrap_or(&result.path)
                        .trim_start_matches('/')
                } else {
                    &result.path
                };
                
                files_by_relevance
                    .entry(relative_path.to_string())
                    .and_modify(|s| *s = s.max(similarity))
                    .or_insert(similarity);
            }
        }
        
        // Sort by relevance
        let mut sorted_files: Vec<_> = files_by_relevance.into_iter().collect();
        sorted_files.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        // Create the final output
        let json_output = serde_json::json!({
            "query": query,
            "cutoff": cutoff,
            "files_by_relevance": sorted_files.into_iter()
                .map(|(file, relevance)| serde_json::json!({
                    "file": file,
                    "relevance": relevance
                }))
                .collect::<Vec<_>>()
        });
        
        serde_json::to_string_pretty(&json_output)
            .map_err(|e| ToolError::ExecutionError(format!("Failed to serialize results: {}", e)))
    }
}

impl Router for GromaLanceDBRouter {
    fn name(&self) -> String {
        let name = "groma-lancedb".to_string();
        debug!("Router.name() called, returning: {}", name);
        name
    }

    fn instructions(&self) -> String {
        debug!("Router.instructions() called");
        "Use this for finding semantically similar files in a given repository using LanceDB backend. \
        The results will be returned as a JSON object with relevant files listed. \
        This version uses local embeddings (fastembed) instead of OpenAI. \
        It is highly recommended to use this along with rg for searching".to_string()
    }

    fn capabilities(&self) -> ServerCapabilities {
        debug!("Router.capabilities() called");
        let caps = CapabilitiesBuilder::new()
            .with_tools(false)  // We don't need tool change notifications
            .with_resources(false, false)  // We don't need resource capabilities
            .with_prompts(false)  // We don't need prompt capabilities
            .build();
        debug!("Capabilities built: {:?}", caps);
        caps
    }

    fn list_tools(&self) -> Vec<Tool> {
        debug!("Router.list_tools() called");
        let tools = vec![
            Tool::new(
                "query".to_string(),
                "pass in search terms to find related files that are similar in concept".to_string(),
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "what to search for, terms, concepts, snippets etc"
                        },
                        "folder": {
                            "type": "string",
                            "description": "The path to the repository to search"
                        },
                        "cutoff": {
                            "type": "number",
                            "description": "Relevance cutoff (0.0-1.0)",
                            "default": 0.3
                        }
                    },
                    "required": ["query", "folder"]
                }),
                Some(ToolAnnotations {
                    title: Some("Search Repository".to_string()),
                    read_only_hint: true,
                    destructive_hint: false,
                    idempotent_hint: true,
                    open_world_hint: false,
                }),
            ),
        ];
        debug!("Returning {} tools", tools.len());
        tools
    }

    fn call_tool(
        &self,
        tool_name: &str,
        arguments: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<Content>, ToolError>> + Send + 'static>> {
        info!("Router.call_tool() called with tool_name: '{}', arguments: {}", tool_name, arguments);
        
        let this = self.clone();
        let tool_name = tool_name.to_string();
        let arguments = arguments.clone();

        Box::pin(async move {
            info!("Executing tool: {}", tool_name);
            match tool_name.as_str() {
                "query" => {
                    debug!("Processing 'query' tool");
                    // Extract arguments
                    let query = arguments
                        .get("query")
                        .and_then(|v| v.as_str())
                        .ok_or_else(|| ToolError::InvalidParameters("Missing 'query' argument".to_string()))?
                        .to_string();
                    
                    let folder = arguments
                        .get("folder")
                        .and_then(|v| v.as_str())
                        .ok_or_else(|| ToolError::InvalidParameters("Missing 'folder' argument".to_string()))?
                        .to_string();
                    
                    let cutoff = arguments
                        .get("cutoff")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.3) as f32;
                    
                    // Process the query and return results directly
                    info!("Calling process_mcp_query with query='{}', folder='{}', cutoff={}", query, folder, cutoff);
                    let result = this.process_mcp_query(query, folder, cutoff).await?;
                    
                    info!("Query processed successfully, result length: {} bytes", result.len());
                    // Return the result as text content
                    Ok(vec![Content::text(result)])
                },
                _ => {
                    error!("Tool not found: {}", tool_name);
                    Err(ToolError::NotFound(format!("Tool {} not found", tool_name)))
                },
            }
        })
    }

    // Implement the required resource methods with empty implementations
    fn list_resources(&self) -> Vec<Resource> {
        vec![]
    }

    fn read_resource(
        &self,
        _uri: &str,
    ) -> Pin<Box<dyn Future<Output = Result<String, ResourceError>> + Send + 'static>> {
        Box::pin(async {
            Err(ResourceError::NotFound("Resources not supported".to_string()))
        })
    }

    // Implement required prompt methods with empty implementations
    fn list_prompts(&self) -> Vec<mcp_core::prompt::Prompt> {
        vec![]
    }

    fn get_prompt(
        &self,
        _prompt_name: &str,
    ) -> Pin<Box<dyn Future<Output = Result<String, PromptError>> + Send + 'static>> {
        Box::pin(async {
            Err(PromptError::NotFound("Prompts not supported".to_string()))
        })
    }
}

/// Run the LanceDB MCP server
async fn run_lancedb_mcp_server() -> Result<()> {
    info!("Initializing LanceDB MCP server...");
    debug!("Creating router instance");

    // Create an instance of our router
    let router = RouterService(GromaLanceDBRouter::new());
    info!("Router created successfully");

    // Create and run the server
    debug!("Creating MCP server");
    let server = Server::new(router);
    
    debug!("Setting up ByteTransport for stdin/stdout");
    let transport = ByteTransport::new(async_stdin(), async_stdout());

    info!("LanceDB MCP server initialized and ready to handle requests");
    info!("Starting server.run() loop...");
    
    let result = server.run(transport).await;
    
    match &result {
        Ok(_) => info!("MCP server exited normally"),
        Err(e) => error!("MCP server error: {:?}", e),
    }
    
    result?;
    Ok(())
}