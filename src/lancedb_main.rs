// LanceDB implementation of Groma - uses LOCAL embeddings (fastembed), NOT OpenAI
// This version runs completely offline with no API calls

use anyhow::{anyhow, Context, Result};
use arrow_array::{ArrayRef, FixedSizeListArray, Float32Array, RecordBatch, RecordBatchIterator, StringArray, Int32Array, types::Float32Type};
use arrow_schema::{DataType, Field, Schema};
use clap::Parser;
use fastembed::{TextEmbedding, EmbeddingModel, InitOptions};
use git2::Repository;
use lancedb::query::{ExecutableQuery, QueryBase};
use lancedb::table::Table;
use serde::Serialize;
use serde_json;
use sha2::{Digest, Sha256};
use std::{
    collections::HashMap,
    fs,
    io::{self, Read},
    path::{Path, PathBuf},
    sync::Arc,
};
use tracing::{debug, info, warn};
use tracing_subscriber::{EnvFilter, FmtSubscriber};
use ignore::gitignore::GitignoreBuilder;
use uuid::Uuid;
use futures::TryStreamExt;
use ignore::WalkBuilder;

const EMBEDDING_DIMENSION: usize = 384; // AllMiniLML6V2 dimension

/// Command-line arguments - matches the original groma interface
#[derive(Parser, Debug)]
#[command(author, version, about = "Groma with LanceDB - uses LOCAL embeddings (fastembed), NOT OpenAI", long_about = None)]
struct Args {
    /// Path to the folder within a Git repository to scan.
    folder: PathBuf,

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

    async fn get_existing_files(&self) -> Result<HashMap<String, String>> {
        let mut files = HashMap::new();
        
        debug!("Querying existing files from LanceDB...");
        
        // Try to query the table - it might be empty on first run
        let results = match self.table.query().execute().await {
            Ok(r) => r,
            Err(e) => {
                debug!("Error querying table (probably empty): {}", e);
                return Ok(files); // Return empty HashMap if table is empty
            }
        };
        
        let batches = match results.try_collect::<Vec<_>>().await {
            Ok(b) => b,
            Err(e) => {
                debug!("Error collecting batches (probably empty table): {}", e);
                return Ok(files); // Return empty HashMap if no data
            }
        };
        
        for batch in batches {
            let paths = batch.column_by_name("path")
                .ok_or_else(|| anyhow!("Missing path column"))?
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| anyhow!("Invalid path column type"))?;
            
            let hashes = batch.column_by_name("hash")
                .ok_or_else(|| anyhow!("Missing hash column"))?
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| anyhow!("Invalid hash column type"))?;
            
            for i in 0..batch.num_rows() {
                let path = paths.value(i).to_string();
                let hash = hashes.value(i).to_string();
                files.entry(path).or_insert(hash);
            }
        }
        
        Ok(files)
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

fn normalize_vector(vector: Vec<f32>) -> Vec<f32> {
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

fn initialize_logging(debug: bool) {
    let filter = if debug {
        EnvFilter::new("debug")
    } else {
        EnvFilter::from_default_env()
            .add_directive("groma=info".parse().unwrap())
    };

    let subscriber = FmtSubscriber::builder()
        .with_env_filter(filter)
        .with_target(false)
        .with_thread_ids(false)
        .with_thread_names(false)
        .finish();

    tracing::subscriber::set_global_default(subscriber)
        .expect("Failed to set global tracing subscriber");
}

async fn perform_file_updates(
    _args: &Args,
    repo: &Repository,
    model: &mut TextEmbedding,
    store: &LanceDBStore,
    canonical_folder_path: &Path,
) -> Result<()> {
    info!("Checking for file updates...");
    
    // Get existing files from the database
    info!("Getting existing files from database...");
    let existing_files = store.get_existing_files().await?;
    info!("Found {} existing files in database", existing_files.len());
    
    // Build gitignore matcher for both .gitignore and .gromaignore
    let mut gitignore_builder = GitignoreBuilder::new(canonical_folder_path);
    
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
    
    // Walk through all files in the repository
    let mut files_to_index = Vec::new();
    let mut total_files = 0;
    
    info!("Walking directory: {}", canonical_folder_path.display());
    
    // Use WalkBuilder which respects .gitignore automatically
    let walker = WalkBuilder::new(canonical_folder_path)
        .hidden(false)  // Don't process hidden files
        .git_ignore(true)  // Respect .gitignore
        .git_exclude(true)  // Also respect .git/info/exclude
        .follow_links(false)
        .build();
    
    for entry in walker {
        let entry = match entry {
            Ok(e) => e,
            Err(e) => {
                warn!("Error walking directory: {}", e);
                continue;
            }
        };
        
        // Skip directories
        if entry.file_type().map_or(false, |ft| ft.is_dir()) {
            continue;
        }
        
        let path = entry.path();
        debug!("Found file: {}", path.display());

        // WalkBuilder already skips hidden files and respects .gitignore
        // But we can double-check with our gromaignore if it exists
        if gromaignore_path.exists() {
            // Need to get relative path for gitignore matcher
            if let Ok(rel_path) = path.strip_prefix(canonical_folder_path) {
                if gitignore.matched(rel_path, false).is_ignore() {
                    debug!("Skipping file due to .gromaignore: {}", path.display());
                    continue;
                }
            }
        }
        
        // Check file extension - only process known text/code files
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
        
        if !allowed_extensions.contains(&extension) && !path.file_name()
            .and_then(|n| n.to_str())
            .map(|n| n == "Makefile" || n == "Dockerfile" || n == "Jenkinsfile" || n == "Vagrantfile")
            .unwrap_or(false)
        {
            continue; // Skip files with unknown extensions
        }
        
        // Check if file is tracked by git
        if let Ok(relative_path) = path.strip_prefix(repo.workdir().unwrap()) {
            if repo.status_file(&relative_path).is_err() {
                continue; // Skip untracked files
            }
        }
        
        // Read file content
        let content = match fs::read_to_string(path) {
            Ok(c) => c,
            Err(_) => continue, // Skip binary files or files that can't be read as UTF-8
        };
        
        let hash = calculate_file_hash(&content);
        let path_str = path.strip_prefix(canonical_folder_path)
            .unwrap_or(path)
            .to_string_lossy()
            .to_string();
        
        // Check if file needs updating
        if existing_files.get(&path_str).map_or(true, |h| h != &hash) {
            files_to_index.push((path_str, content, hash));
        }
        
        total_files += 1;
    }
    
    if files_to_index.is_empty() {
        info!("No files need updating. Index is up to date.");
        return Ok(());
    }
    
    info!("Indexing {} files (out of {} total)...", files_to_index.len(), total_files);
    
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
    Ok(())
}

async fn process_query(
    query: &str,
    model: &mut TextEmbedding,
    store: &LanceDBStore,
    cutoff: f32,
) -> Result<()> {
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
    
    println!("{}", serde_json::to_string_pretty(&output)?);
    
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    initialize_logging(args.debug);
    
    info!("Starting Groma with LanceDB (using LOCAL fastembed, NOT OpenAI)...");
    
    // Initialize the LOCAL embedding model
    info!("Initializing LOCAL embedding model (AllMiniLML6V2)...");
    let mut model = TextEmbedding::try_new(
        InitOptions::new(EmbeddingModel::AllMiniLML6V2)
            .with_show_download_progress(false)
    )?;
    info!("LOCAL embedding model initialized (no API calls required).");
    
    // Canonicalize folder path
    let canonical_folder_path = fs::canonicalize(&args.folder)?;
    info!("Using folder: {}", canonical_folder_path.display());
    
    // Find Git repository
    let repo = Repository::discover(&canonical_folder_path)
        .context("Failed to find Git repository")?;
    
    // Initialize LanceDB
    info!("Initializing LanceDB at: {}", args.lancedb_path);
    let store = LanceDBStore::new(&args.lancedb_path, EMBEDDING_DIMENSION).await?;
    
    // Perform file updates unless suppressed
    if !args.suppress_updates {
        perform_file_updates(
            &args,
            &repo,
            &mut model,
            &store,
            &canonical_folder_path,
        ).await?;
    } else {
        info!("Skipping file updates (--suppress-updates specified).");
    }
    
    // Read query from stdin
    info!("Reading query from stdin...");
    let mut query = String::new();
    io::stdin().read_to_string(&mut query)?;
    let query = query.trim();
    
    if query.is_empty() {
        return Err(anyhow!("No query provided via stdin"));
    }
    
    info!("Processing query: {}", query);
    process_query(query, &mut model, &store, args.cutoff).await?;
    
    Ok(())
}