// Standard library imports
use std::{
    collections::HashMap, // Moved here
    fs,
    io::{self, Read},
    path::{Path, PathBuf},
    sync::Arc,
};

// External crate imports
use anyhow::{anyhow, Context, Result}; // Removed bail
use clap::Parser;
use git2::{DiffOptions, Oid, Repository, Status}; // Added DiffOptions, Oid, Status
use hex; // Added for encoding hashes
use qdrant_client::{
    qdrant::{
        point_id::PointIdOptions, r#match::MatchValue, Condition, CreateCollectionBuilder, // Added MatchValue, Condition
        Distance, Filter, PayloadIncludeSelector, PointId, PointStruct, SearchPointsBuilder, // Added Filter
        UpsertPointsBuilder, VectorParams, VectorsConfig,
    },
    Payload, Qdrant,
};
use rig::{
    embeddings::{
        embed::{Embed, EmbedError, TextEmbedder},
        embedding::EmbeddingModel,
        EmbeddingsBuilder,
    },
    providers::openai,
};
use serde::{Deserialize, Serialize}; // Already present, used for FileMetadata and now GromaState
use serde_json; // Added for state serialization
use sha2::{Digest, Sha256};
use text_splitter::TextSplitter;
use tiktoken_rs::{cl100k_base, CoreBPE};
use tracing::{debug, error, info, warn}; // Removed Level
use tracing_subscriber::{EnvFilter, FmtSubscriber};
use url::Url;
use uuid::Uuid;

// --- Constants ---

/// The embedding dimension used by the default OpenAI model (`text-embedding-3-small`).
/// This MUST match the dimension of the chosen embedding model.
const EMBEDDING_DIMENSION: u64 = 1536;
/// The number of points to upsert to Qdrant in a single batch.
const QDRANT_UPSERT_BATCH_SIZE: usize = 100;
/// The target size for text chunks in tokens before embedding.
/// Aiming for the maximum supported by models like text-embedding-3-small (8192).
const TARGET_CHUNK_SIZE_TOKENS: usize = 8192;

// --- Command Line Arguments ---

/// Defines the command-line arguments accepted by the application.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the folder within a Git repository to scan.
    folder: PathBuf,

    /// Relevance cutoff for results (e.g., 0.7). Only results with a score
    /// above this threshold will be shown.
    #[arg(short, long)]
    cutoff: f32,

    /// OpenAI API Key. Can also be set via the OPENAI_API_KEY environment variable.
    #[arg(long, env = "OPENAI_API_KEY")]
    openai_key: String,

    /// OpenAI Embedding Model name (e.g., "text-embedding-3-small").
    #[arg(long, default_value = "text-embedding-3-small")]
    openai_model: String,

    /// Qdrant server URL (points to the gRPC port, e.g., http://localhost:6334).
    /// Can also be set via the QDRANT_URL environment variable.
    #[arg(long, env = "QDRANT_URL", default_value = "http://localhost:6334")]
    qdrant_url: Url,

    /// Suppress checking for file updates and upserting to Qdrant.
    /// Useful for querying existing data without modifying the index.
    #[arg(long)]
    suppress_updates: bool,

    /// Enable debug logging. By default, only errors are logged unless RUST_LOG is set.
    #[arg(long)]
    debug: bool,
}

// --- Data Structures ---

/// Metadata associated with each chunk stored in Qdrant.
#[derive(Serialize, Deserialize, Debug, Clone)]
struct FileMetadata {
    /// The canonical path of the original file.
    path: String,
    /// The SHA256 hash of the entire original file content.
    hash: String,
    /// The 0-based index of this chunk within the original file.
    chunk_index: usize,
}

/// Stores the state between runs, specifically the last processed Git commit OID.
#[derive(Serialize, Deserialize, Debug, Clone)]
struct GromaState {
    last_processed_oid: String,
}

/// Represents a document (file) to be processed and embedded.
/// Implements the `Embed` trait for use with `rig::EmbeddingsBuilder`.
#[derive(Debug)]
struct LongDocument<'a> {
    /// The canonical path of the file.
    path_str: String,
    /// The current SHA256 hash of the file content.
    current_hash: String,
    /// The full content of the file.
    content: String,
    /// A reference to the text splitter for chunking.
    text_splitter: &'a TextSplitter<CoreBPE>,
}

impl<'a> Embed for LongDocument<'a> {
    /// Chunks the document's content using the text splitter and provides
    /// each non-empty chunk to the `TextEmbedder`.
    fn embed(&self, embedder: &mut TextEmbedder) -> Result<(), EmbedError> {
        // Use the text_splitter reference to chunk the content
        let chunks = self
            .text_splitter
            .chunks(&self.content, TARGET_CHUNK_SIZE_TOKENS);

        // Removed unused chunk_count variable
        // Add each chunk's text to the embedder
        for chunk in chunks {
            if !chunk.trim().is_empty() {
                embedder.embed(chunk.to_string()); // Pass ownership of the string chunk
                // Removed chunk_count increment
            }
        }
        // It's okay if a document yields no chunks (e.g., only whitespace).
        // The calling code handles skipping empty files before creating LongDocument.
        Ok(())
    }
}

// --- Helper Functions ---

/// Generates a stable, deterministic UUID (version 5) for a specific file chunk
/// based on the file's canonical path and the chunk's index.
fn generate_uuid_for_chunk(path_str: &str, chunk_index: usize) -> Uuid {
    let identifier = format!("{}:{}", path_str, chunk_index);
    Uuid::new_v5(&Uuid::NAMESPACE_DNS, identifier.as_bytes())
}

/// Converts a `Uuid` into a Qdrant `PointId`.
fn uuid_to_point_id(uuid: Uuid) -> PointId {
    PointId {
        point_id_options: Some(PointIdOptions::Uuid(uuid.to_string())),
    }
}

/// Initializes the tracing subscriber for logging based on the `--debug` flag
/// and the `RUST_LOG` environment variable.
fn initialize_logging(debug_enabled: bool) {
    // Default level is OFF, unless --debug is passed (then DEBUG).
    // RUST_LOG environment variable overrides the default.
    let default_level = if debug_enabled { "debug" } else { "off" };
    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(default_level));

    let subscriber = FmtSubscriber::builder()
        .with_env_filter(env_filter)
        .with_writer(std::io::stderr)
        .finish();

    tracing::subscriber::set_global_default(subscriber)
        .expect("setting default subscriber failed");
}

/// Generates a Qdrant collection name based on a hash of the canonical folder path.
/// Ensures the name is valid for Qdrant.
fn generate_collection_name(folder_path: &Path) -> Result<String> {
    // Canonicalize path for consistency across different invocations
    let canonical_path = fs::canonicalize(folder_path)
        .with_context(|| format!("Failed to canonicalize folder path: {}", folder_path.display()))?;
    let path_str = canonical_path.to_string_lossy();

    // Use SHA256 hash of the canonical path for a stable identifier
    let mut hasher = Sha256::new();
    hasher.update(path_str.as_bytes());
    let hash_bytes = hasher.finalize();
    let hash_hex = hex::encode(hash_bytes);

    // Use the first 16 chars of the hex hash for brevity, prefixed with "groma-"
    // Qdrant names must match ^[a-zA-Z0-9_-]{1,255}$
    let collection_name = format!("groma-{}", &hash_hex[..16]);
    debug!("Generated collection name '{}' from path '{}'", collection_name, path_str);

    Ok(collection_name)
}

/// Creates a `TextSplitter` configured with the `cl100k_base` tokenizer (used by OpenAI).
fn create_text_splitter() -> Result<TextSplitter<CoreBPE>> {
    let tokenizer = cl100k_base().context("Failed to load cl100k_base tokenizer")?;
    // Configure the splitter. Chunk size is passed to .chunks() later.
    Ok(TextSplitter::new(tokenizer).with_trim_chunks(true))
}

/// Ensures that the specified Qdrant collection exists, creating it if necessary.
async fn ensure_qdrant_collection(client: Arc<Qdrant>, collection_name: &str) -> Result<()> {
    let collections_list = client.list_collections().await?;
    if !collections_list
        .collections
        .iter()
        .any(|c| c.name == collection_name)
    {
        info!("Collection '{}' not found. Creating...", collection_name);
        client
            .create_collection(
                CreateCollectionBuilder::new(collection_name)
                    .vectors_config(VectorsConfig::from(VectorParams {
                        size: EMBEDDING_DIMENSION,
                        distance: Distance::Cosine.into(),
                        ..Default::default() // Use default for other vector params
                    }))
                    // Consider adding payload indexing for 'path' and 'hash' here
                    // for potentially faster filtering/lookups if performance becomes an issue.
                    // Example: .add_payload_index(...)
            )
            .await?;
        info!("Collection '{}' created.", collection_name);
    } else {
        info!("Collection '{}' already exists.", collection_name);
    }
    Ok(())
}

// --- State Management ---

/// Returns the expected path for the .gromastate file within the repository's .git directory.
fn get_state_file_path(repo: &Repository) -> Result<PathBuf> {
    Ok(repo.path().join(".gromastate")) // Store state inside .git folder
}

/// Loads the GromaState from the .gromastate file.
/// Returns Ok(None) if the file doesn't exist (first run).
fn load_state(repo: &Repository) -> Result<Option<GromaState>> {
    let state_file_path = get_state_file_path(repo)?;
    if !state_file_path.exists() {
        info!("No previous state file found at '{}'. Assuming first run.", state_file_path.display());
        return Ok(None);
    }

    debug!("Loading state from '{}'", state_file_path.display());
    let file = fs::File::open(&state_file_path)
        .with_context(|| format!("Failed to open state file: {}", state_file_path.display()))?;
    let state: GromaState = serde_json::from_reader(io::BufReader::new(file))
        .with_context(|| format!("Failed to deserialize state from: {}", state_file_path.display()))?;
    info!("Loaded previous state (OID: {})", state.last_processed_oid);
    Ok(Some(state))
}

/// Saves the given GromaState to the .gromastate file.
fn save_state(repo: &Repository, state: &GromaState) -> Result<()> {
    let state_file_path = get_state_file_path(repo)?;
    debug!("Saving current state (OID: {}) to '{}'", state.last_processed_oid, state_file_path.display());
    let file = fs::File::create(&state_file_path)
        .with_context(|| format!("Failed to create state file: {}", state_file_path.display()))?;
    serde_json::to_writer_pretty(io::BufWriter::new(file), state)
        .with_context(|| format!("Failed to serialize state to: {}", state_file_path.display()))?;
    info!("Saved current state (OID: {})", state.last_processed_oid);
    Ok(())
}


// --- Hashing & Qdrant Interaction (mostly unchanged) ---

/// Calculates the SHA256 hash of a file's content.
fn calculate_hash(file_path: &Path) -> Result<String> {
    let mut file = fs::File::open(file_path)
        .with_context(|| format!("Failed to open file for hashing: {}", file_path.display()))?;
    let mut hasher = Sha256::new();
    io::copy(&mut file, &mut hasher)
        .with_context(|| format!("Failed to read file for hashing: {}", file_path.display()))?;
    let hash_bytes = hasher.finalize();
    Ok(hex::encode(hash_bytes))
}

/// Fetches the stored hash for *one* chunk associated with the given file path
/// from the Qdrant collection. Returns `Ok(None)` if no chunk is found for the path.
/// Note: This is less critical now as we rely on Git diff, but can be useful for double-checking
/// if a file marked as 'Modified' actually existed before.
async fn get_existing_file_hash(client: Arc<Qdrant>, collection_name: &str, path_str: &str) -> Result<Option<String>> {
    debug!("Checking Qdrant for existing hash for path: {}", path_str);
    // Filter points where the 'path' payload field matches the given path_str
    let filter = Filter::must([Condition::matches(
        "path",
        MatchValue::Keyword(path_str.to_string()),
    )]);

    // Search for just one point matching the filter, requesting only the 'hash' payload field.
    // We don't need the vector for this check.
    let search_req = SearchPointsBuilder::new(collection_name, vec![0.0; EMBEDDING_DIMENSION as usize], 1)
        .filter(filter)
        .with_payload(PayloadIncludeSelector { fields: vec!["hash".to_string()] })
        .with_vectors(false);

    let search_result = client.search_points(search_req).await?;

    // If a point is found, extract the hash from its payload.
    if let Some(point) = search_result.result.into_iter().next() {
        if let Some(hash_value) = point.payload.get("hash") {
            // Convert the Qdrant Value to a String
            return Ok(hash_value.as_str().map(String::from));
        } else {
            warn!("Found point for path '{}' but it's missing the 'hash' payload.", path_str);
        }
    }
    // No point found for this path
    Ok(None)
}

/// Deletes all points associated with a specific file path from the Qdrant collection.
async fn delete_points_by_path(client: Arc<Qdrant>, collection_name: &str, path_str: &str) -> Result<()> {
    info!("Deleting existing chunks for file '{}' from collection '{}'", path_str, collection_name);
    // Filter points where the 'path' payload field matches the given path_str
    let filter = Filter::must([Condition::matches(
        "path",
        MatchValue::Keyword(path_str.to_string()),
    )]);

    // Use the filter to specify which points to delete.
    // Note: Constructing DeletePoints directly as the builder might have issues depending on the client version.
    let delete_request = qdrant_client::qdrant::DeletePoints {
        collection_name: collection_name.to_string(),
        wait: None, // Set to Some(true) to wait for operation completion if needed
        ordering: None,
        points: Some(qdrant_client::qdrant::PointsSelector {
            points_selector_one_of: Some(
                qdrant_client::qdrant::points_selector::PointsSelectorOneOf::Filter(filter),
            ),
        }),
        shard_key_selector: None, // Assuming default sharding
    };

    client.delete_points(delete_request).await?;
    Ok(())
}

/// Upserts a batch of points into the specified Qdrant collection.
async fn upsert_batch(client: Arc<Qdrant>, collection_name: &str, points: Vec<PointStruct>) -> Result<()> { // Takes ownership now
    if points.is_empty() {
        return Ok(());
    }
    client
        .upsert_points(
            UpsertPointsBuilder::new(collection_name, points) // Use owned points directly
            // .wait(true) // Optionally wait for operation to complete
        )
        .await
        .context("Failed to upsert batch to Qdrant")?;
    Ok(())
}

// --- Main Application Logic ---

#[tokio::main]
async fn main() -> Result<()> {
    // 1. Parse Arguments & Initialize Logging
    let args = Args::parse();
    initialize_logging(args.debug);

    // 2. Canonicalize Folder Path (used for relative path calculation later)
    let canonical_folder_path = fs::canonicalize(&args.folder)
        .with_context(|| format!("Failed to canonicalize input folder path: {}", args.folder.display()))?;
    info!("Using canonical folder path: {}", canonical_folder_path.display());

    // 3. Initialize Clients (OpenAI, Qdrant)
    info!("Initializing clients...");
    let openai_client = rig::providers::openai::Client::new(&args.openai_key);
    let embedding_model: Arc<openai::EmbeddingModel> = Arc::new(
        openai_client.embedding_model(&args.openai_model)
    );
    info!(
        "Using OpenAI Embedding model: {} (Dimension: {})",
        args.openai_model, EMBEDDING_DIMENSION
    );
    let qdrant_client = Arc::new(Qdrant::from_url(&args.qdrant_url.to_string()).build()?);
    info!("Connected to Qdrant at {}", args.qdrant_url);

    // 4. Determine Collection Name & Ensure It Exists
    let collection_name = generate_collection_name(&args.folder)?;
    info!("Using Qdrant collection: {}", collection_name);
    ensure_qdrant_collection(qdrant_client.clone(), &collection_name).await?;

    // 5. Discover Repository and Perform File Updates if not suppressed
    let repo = Repository::discover(&args.folder)
        .with_context(|| format!("Failed to find Git repository containing path: {}", args.folder.display()))?;
    let workdir = repo.workdir().ok_or_else(|| anyhow!("Git repository is bare, cannot process files"))?;
    info!("Found Git repository at: {}", workdir.display());

    if !args.suppress_updates {
        perform_file_updates(
            &args,
            &repo, // Pass repository reference
            qdrant_client.clone(),
            embedding_model.clone(),
            &collection_name,
            &canonical_folder_path, // Pass canonical folder path
        ).await?;
    } else {
        info!("Skipping file updates (--suppress-updates specified).");
    }

    // 6. Process User Query (Read, Embed, Search, Format Output)
    process_query(
        &args,
        qdrant_client.clone(),
        embedding_model.clone(),
        &collection_name,
        &canonical_folder_path,
    ).await?;

    Ok(())
} // Close main function block here

/// Handles the core logic for detecting file changes using Git, embedding changes, and upserting to Qdrant.
async fn perform_file_updates(
    args: &Args,
    repo: &Repository, // Use Git repository
    qdrant_client: Arc<Qdrant>,
    embedding_model: Arc<openai::EmbeddingModel>,
    collection_name: &str,
    canonical_folder_path: &Path, // Base path for filtering and relative paths
) -> Result<()> {
    info!("Checking for file updates using Git and processing changes...");

    let workdir = repo.workdir().ok_or_else(|| anyhow!("Git repository is bare, cannot process files"))?;

    // --- Phase 1: Load State & Determine Git Diff ---
    let previous_state = load_state(repo)?;
    let head_commit_oid = repo.head()?.peel_to_commit()?.id();
    info!("Current HEAD OID: {}", head_commit_oid);

    let previous_tree = match previous_state {
        Some(ref state) => {
            let oid = Oid::from_str(&state.last_processed_oid)?;
            match repo.find_commit(oid)?.tree() {
                Ok(tree) => Some(tree),
                Err(e) => {
                    warn!("Failed to find tree for previous OID {}: {}. Processing all files.", state.last_processed_oid, e);
                    None // Fallback to processing all if previous commit/tree is missing
                }
            }
        }
        None => None, // First run, compare against empty tree
    };

    // Diff HEAD against the previous tree (or empty tree if first run)
    // We compare against the committed state (HEAD) to ensure reproducibility.
    // Changes in the working directory or index that are not committed yet won't be processed.
    let current_tree = repo.head()?.peel_to_tree()?;
    let mut diff_opts = DiffOptions::new();
    // diff_opts.include_renames(true); // Rename detection is often default or handled differently
    diff_opts.include_typechange(true); // Correct method name
    // Convert the target folder path to be relative to the workdir for pathspec
    let pathspec = args.folder.strip_prefix(workdir).unwrap_or(&args.folder);
    diff_opts.pathspec(pathspec); // Limit diff to the target folder relative to repo root
    info!("Using pathspec for diff: {}", pathspec.display());


    let diff = repo.diff_tree_to_tree(previous_tree.as_ref(), Some(&current_tree), Some(&mut diff_opts))?;

    info!("Git diff found {} changed items potentially within the target folder.", diff.deltas().len());

    let text_splitter = create_text_splitter().context("Failed to create text splitter")?;
    let mut documents_to_embed: Vec<LongDocument> = Vec::new();
    let mut paths_to_delete: Vec<String> = Vec::new();
    let mut processed_new_paths: std::collections::HashSet<PathBuf> = std::collections::HashSet::new(); // Track processed adds/renames
    let mut processed_count = 0;

    // --- Phase 2: Process Diff Deltas ---
    for delta in diff.deltas() {
        let status = delta.status();
        let old_repo_path = delta.old_file().path(); // Path relative to repo root
        let new_repo_path = delta.new_file().path(); // Path relative to repo root

        // Construct absolute paths based on workdir
        let old_absolute_path = old_repo_path.map(|p| workdir.join(p));
        let new_absolute_path = new_repo_path.map(|p| workdir.join(p));

        // --- Crucial Filter: Ensure the *new* path (for Add/Modify/Rename) or *old* path (for Delete)
        // --- is actually within the *canonical target folder*. Git's pathspec is prefix-based
        // --- and might include files outside the exact target folder if names overlap.
        // Use constants for status checks (INDEX_* for tree-to-tree diff)
        let relevant_path_for_filter = if status == Status::INDEX_DELETED {
             old_absolute_path.as_ref()
        } else {
             new_absolute_path.as_ref() // INDEX_NEW, INDEX_MODIFIED, INDEX_RENAMED, INDEX_TYPECHANGE etc.
        };


        if relevant_path_for_filter.map_or(true, |p| !p.starts_with(canonical_folder_path)) {
             debug!(
                 "Skipping delta outside target folder (canonical check): old={:?}, new={:?}, status={:?}",
                 old_repo_path, new_repo_path, status
             );
             continue;
        }
        // --- End Filter ---

        processed_count += 1; // Count files actually processed after filtering

        // Use if/else if with status constants (bitflags)
        if status == Status::INDEX_NEW || status == Status::INDEX_MODIFIED || status == Status::INDEX_TYPECHANGE {
            if let Some(new_path_abs) = new_absolute_path {
                // Canonicalize AFTER confirming it starts with the canonical_folder_path prefix
                if let Ok(canonical_new) = fs::canonicalize(&new_path_abs) {
                    let path_str = canonical_new.to_string_lossy().to_string();
                    info!("Detected NEW/MODIFIED/TYPECHANGE: {}", path_str);
                    processed_new_paths.insert(canonical_new.clone()); // Track this path

                    // If modified/typechange, ensure old points are deleted (using canonical path)
                    // This handles cases where filename case might change but path object differs
                    if status == Status::INDEX_MODIFIED || status == Status::INDEX_TYPECHANGE {
                         if let Some(old_path_abs) = old_absolute_path.as_ref() {
                            // Attempt to canonicalize the old path for deletion consistency
                            if let Ok(canonical_old) = fs::canonicalize(old_path_abs) {
                                    paths_to_delete.push(canonical_old.to_string_lossy().to_string());
                                } else {
                                     // If canonicalization fails (e.g., file already gone), use the absolute path string
                                     warn!("Could not canonicalize old path for MODIFIED/TYPECHANGE file: {}. Using non-canonical path for deletion.", old_path_abs.display());
                                     paths_to_delete.push(old_path_abs.to_string_lossy().to_string());
                                }
                             }
                        }

                        // Process the new/modified file content for embedding
                        match process_file_for_embedding(&canonical_new, &text_splitter) {
                            Ok(Some(doc)) => documents_to_embed.push(doc),
                            Ok(None) => info!("Skipping empty or unreadable file: {}", path_str), // Empty file case
                            Err(e) => warn!("Failed processing NEW/MODIFIED/TYPECHANGE file {}: {}", path_str, e),
                        }
                    } else {
                        warn!("Could not canonicalize new path for NEW/MODIFIED/TYPECHANGE file: {}", new_path_abs.display());
                    }
                }
            }
        } else if status == Status::INDEX_DELETED {
            if let Some(old_path_abs) = old_absolute_path {
                 // Attempt to canonicalize the path for deletion consistency
                 if let Ok(canonical_old) = fs::canonicalize(&old_path_abs) {
                        let path_str = canonical_old.to_string_lossy().to_string();
                        info!("Detected DELETED: {}", path_str);
                        paths_to_delete.push(path_str);
                     } else {
                         // If canonicalization fails (file is already gone), use the absolute path string
                         warn!("Could not canonicalize path for DELETED file: {}. Using non-canonical path for deletion.", old_path_abs.display());
                         paths_to_delete.push(old_path_abs.to_string_lossy().to_string());
                 }
            }
        } else if status == Status::INDEX_RENAMED {
            if let (Some(old_path_abs), Some(new_path_abs)) = (old_absolute_path, new_absolute_path) {
                // Canonicalize AFTER confirming the new path starts with the canonical_folder_path prefix
                if let Ok(canonical_new) = fs::canonicalize(&new_path_abs) {
                        let new_path_str = canonical_new.to_string_lossy().to_string();
                        processed_new_paths.insert(canonical_new.clone()); // Track this path

                        // Attempt to canonicalize the old path for deletion consistency
                        let old_path_str_for_delete = match fs::canonicalize(&old_path_abs) {
                             Ok(canonical_old) => canonical_old.to_string_lossy().to_string(),
                             Err(_) => {
                                 warn!("Could not canonicalize old path for RENAMED file: {}. Using non-canonical path for deletion.", old_path_abs.display());
                                 old_path_abs.to_string_lossy().to_string()
                             }
                        };

                        info!("Detected RENAMED: {} -> {}", old_path_str_for_delete, new_path_str);

                        // Simple approach: Delete old, process new as ADDED
                        paths_to_delete.push(old_path_str_for_delete);
                        match process_file_for_embedding(&canonical_new, &text_splitter) {
                            Ok(Some(doc)) => documents_to_embed.push(doc),
                            Ok(None) => info!("Skipping empty or unreadable renamed file: {}", new_path_str),
                            Err(e) => warn!("Failed processing RENAMED file {}: {}", new_path_str, e),
                        }
                    } else {
                         warn!("Could not canonicalize new path for RENAMED file: {} -> {}", old_path_abs.display(), new_path_abs.display());
                    }
                }
            }
            // Note: The if/else if chain handles NEW, MODIFIED, TYPECHANGE, DELETED, RENAMED.
            // We implicitly ignore other statuses by not having an `else` block here.
            // The debug log for ignored statuses was removed as it was part of the old `_` match arm.
        }
        // Note: We are ignoring other statuses like CONFLICTED, IGNORED, UNTRACKED, etc.
        // as they shouldn't appear in a tree-to-tree diff reflecting committed changes.
    } // Closes the `for delta in diff.deltas()` loop
    info!("Finished processing {} Git diff deltas relevant to the target folder.", processed_count);


    // --- Phase 3: Delete Obsolete Points ---
    // Deduplicate paths to delete before executing deletions
    paths_to_delete.sort();
    paths_to_delete.dedup();
    if !paths_to_delete.is_empty() {
        info!("Deleting points for {} removed/modified/renamed paths...", paths_to_delete.len());
        for path_str in paths_to_delete {
            // Ensure we don't delete paths that were immediately re-added (e.g., case-only rename on some systems)
            // This check might be overly cautious depending on exact diff behavior but safer.
            let path_buf = PathBuf::from(&path_str);
            if processed_new_paths.contains(&path_buf) {
                 debug!("Skipping deletion for path that was also added/renamed in this run: {}", path_str);
                 continue;
            }
            if let Err(e) = delete_points_by_path(qdrant_client.clone(), collection_name, &path_str).await {
                warn!("Failed to delete points for path {}: {}", path_str, e);
            }
        }
    } else {
        info!("No points marked for deletion.");
    }


    // --- Phase 4 & 5: Batch Embed and Create Points ---
    let mut all_points_to_upsert: Vec<PointStruct> = Vec::new();
    if !documents_to_embed.is_empty() {
        info!("Starting batch embedding for {} added/modified/renamed documents...", documents_to_embed.len());
        // Need to pass the owned documents to the builder
        let docs_for_builder: Vec<LongDocument> = documents_to_embed; // Move ownership
        let embedding_results = EmbeddingsBuilder::new((*embedding_model).clone())
            .documents(docs_for_builder)? // Pass owned Vec
            .build()
            .await
            .context("Failed to generate embeddings using EmbeddingsBuilder")?;
        info!("Successfully generated embeddings.");

        info!("Constructing Qdrant points from embeddings...");
        for (long_document, embedding_result) in embedding_results.into_iter() {
            debug!("Processing {} embedding(s) for document '{}'", embedding_result.len(), long_document.path_str);
            for (chunk_index, embedding) in embedding_result.into_iter().enumerate() {
                let chunk_uuid = generate_uuid_for_chunk(&long_document.path_str, chunk_index);
                let point_id = uuid_to_point_id(chunk_uuid);

                let metadata = FileMetadata {
                    path: long_document.path_str.clone(),
                    hash: long_document.current_hash.clone(), // Store the full file hash
                    chunk_index,
                };
                let payload: Payload = match serde_json::to_value(metadata)
                    .context("Failed to serialize metadata")?
                    .try_into() {
                        Ok(p) => p,
                        Err(e) => {
                            error!("Failed to convert metadata JSON to Qdrant Payload for chunk {} of {}: {}", chunk_index, long_document.path_str, e);
                            continue; // Skip this point
                        }
                    };

                // Convert f64 embedding from rig to f32 for Qdrant
                let vector_f32: Vec<f32> = embedding.vec.into_iter().map(|v| v as f32).collect();
                let vectors: qdrant_client::qdrant::Vectors = vector_f32.into();
                let point = PointStruct::new(point_id, vectors, payload);
                all_points_to_upsert.push(point);
            }
        }
        info!("Finished constructing {} points.", all_points_to_upsert.len());
    } else {
        info!("No documents require embedding.");
    }

    // --- Phase 6: Batch Upsert All Collected Points ---
    if !all_points_to_upsert.is_empty() {
        info!(
            "Starting batch upsert of {} total points to collection '{}'...",
            all_points_to_upsert.len(),
            collection_name
        );
        // Upsert in batches
        for chunk in all_points_to_upsert.chunks(QDRANT_UPSERT_BATCH_SIZE).map(|c| c.to_vec()) {
             debug!("Upserting batch of {} points...", chunk.len());
             // Pass ownership of the chunk Vec to upsert_batch
             upsert_batch(qdrant_client.clone(), collection_name, chunk).await?;
        }
        info!("Finished upserting points.");
    } else {
        info!("No new points to upsert.");
    }

    // --- Phase 7: Save State ---
    // Only save state if the diff processing and upserting seemed successful
    // (Error handling within the loops might allow partial success, consider if this is desired)
    let current_state = GromaState {
        last_processed_oid: head_commit_oid.to_string(),
    };
    save_state(repo, &current_state)?;

    Ok(())
}

/// Helper function to read, hash, and prepare a single file for embedding.
/// Returns Ok(None) if the file is empty or cannot be read.
fn process_file_for_embedding<'a>(
    file_path: &Path,
    text_splitter: &'a TextSplitter<CoreBPE>, // Borrow splitter
) -> Result<Option<LongDocument<'a>>> {
    let path_str = file_path.to_string_lossy().to_string();
    let path_for_error = path_str.clone(); // Clone for error context

    // Check if it's actually a file and not a directory/symlink etc. before reading
    if !file_path.is_file() {
        warn!("Path is not a file: {}. Skipping.", path_str);
        return Ok(None);
    }

    match fs::read_to_string(file_path) {
        Ok(content) => {
            if content.trim().is_empty() {
                warn!("Skipping empty file: {}", path_str);
                Ok(None) // Skip empty files
            } else {
                // Calculate hash only if content is read successfully
                let current_hash = calculate_hash(file_path)
                    .with_context(|| format!("Failed to calculate hash for {}", path_for_error))?;

                Ok(Some(LongDocument {
                    path_str,
                    current_hash,
                    content,
                    text_splitter, // Pass borrowed splitter
                }))
            }
        }
        Err(e) => {
            // Log error but return Ok(None) to allow skipping problematic files
            warn!("Failed to read file content for {}: {}. Skipping file.", path_for_error, e);
            Ok(None)
        }
    }
}


/// Handles reading the user query, embedding it, searching Qdrant, and printing results.
async fn process_query(
    args: &Args,
    qdrant_client: Arc<Qdrant>,
    embedding_model: Arc<openai::EmbeddingModel>,
    collection_name: &str,
    canonical_folder_path: &Path, // Needed for making paths relative
) -> Result<()> {
    // --- Phase 5: Read Query from Stdin ---
    info!("Please enter your query and press Enter (Ctrl+D to finish):");
    let mut query = String::new();
    io::stdin().read_to_string(&mut query).context("Failed to read query from stdin")?;
    let query = query.trim();

    if query.is_empty() {
        info!("No query provided. Exiting.");
        return Ok(());
    }
    info!("Processing query: '{}'", query);

    // --- Phase 6: Embed Query ---
    let query_embedding = embedding_model
        .embed_text(query)
        .await
        .context("Failed to embed query")?;

    // --- Phase 7: Search Qdrant ---
    info!("Searching for relevant files...");
    let query_vector_f32: Vec<f32> = query_embedding.vec.into_iter().map(|v| v as f32).collect();

    let search_result = qdrant_client
        .search_points(
            SearchPointsBuilder::new(collection_name, query_vector_f32, 100) // Limit results
                .with_payload(true) // Request payload to get file path
                .score_threshold(args.cutoff), // Apply relevance cutoff
        )
        .await
        .context("Failed to search Qdrant")?;
    info!("Found {} potential matching chunks.", search_result.result.len());

    // --- Phase 8: Aggregate Results by File Path ---
    // Group results by file path, keeping the highest score for each file.
    let mut file_scores: HashMap<String, f32> = HashMap::new();
    for hit in search_result.result {
        if let Some(path_val) = hit.payload.get("path") {
            if let Some(absolute_path_str) = path_val.as_str() {
                let score = hit.score;
                file_scores
                    .entry(absolute_path_str.to_string())
                    .and_modify(|e| *e = e.max(score)) // Update score if higher
                    .or_insert(score); // Insert if new
            } else {
                 warn!("Found hit with non-string 'path' payload: {:?}", hit.payload);
            }
        } else {
            warn!("Found hit without 'path' payload: {:?}", hit.payload);
        }
    }

    // --- Phase 9: Format and Print Aggregated Results as JSON ---
    // Convert absolute paths to relative paths and sort by score.
    let mut aggregated_results: Vec<(f32, String)> = file_scores
        .into_iter()
        .filter_map(|(absolute_path_str, score)| {
            let absolute_path = PathBuf::from(&absolute_path_str);
            match absolute_path.strip_prefix(canonical_folder_path) {
                Ok(relative_path) => Some((score, relative_path.to_string_lossy().to_string())),
                Err(e) => {
                    warn!(
                        "Failed to make path relative: Cannot strip prefix '{}' from path '{}': {}. Skipping result.",
                        canonical_folder_path.display(), absolute_path_str, e
                    );
                    None
                }
            }
        })
        .collect();

    // Sort by score descending
    aggregated_results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    // Prepare JSON output structure: {"files_by_relevance": [[score, path], ...]}
    let json_output_map = HashMap::from([("files_by_relevance", aggregated_results)]);

    if json_output_map["files_by_relevance"].is_empty() {
        info!("No files found matching the query with cutoff {}", args.cutoff);
        println!("{}", serde_json::json!({ "files_by_relevance": [] }));
    } else {
        info!("Printing results as JSON:");
        match serde_json::to_string_pretty(&json_output_map) {
            Ok(json_str) => println!("{}", json_str),
            Err(e) => error!("Failed to serialize results to JSON: {}", e),
        }
    }

    Ok(())
}
