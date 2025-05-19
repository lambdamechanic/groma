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
use clap::{Parser, Subcommand};
use git2::{Delta, DiffOptions, Oid, Repository}; // Added Delta, DiffOptions, Oid, Removed Status
use hex; // Added for encoding hashes
use qdrant_client::{
    qdrant::{
        point_id::PointIdOptions,
        r#match::MatchValue,
        Condition,
        CreateCollectionBuilder, // Added MatchValue, Condition
        Distance,
        Filter,
        PointId,
        PointStruct,
        SearchPointsBuilder, // Added Filter
        UpsertPointsBuilder,
        VectorParams,
        VectorsConfig,
    },
    Payload, Qdrant,
};
use rig::{
    embeddings::{
        embed::{Embed, EmbedError, TextEmbedder},
        embedding::EmbeddingModel, // Removed OneOrMany from here
        EmbeddingsBuilder,
    },
    providers::openai,
    OneOrMany, // Import OneOrMany directly from rig
};
use serde::{Deserialize, Serialize}; // Already present, used for FileMetadata and now GromaState
use serde_json; // Added for state serialization
use sha2::{Digest, Sha256};
// Removed unused TextSplitter import
use tiktoken_rs::{cl100k_base, CoreBPE};
use tracing::{debug, error, info, warn}; // Removed Level
use tracing_subscriber::{EnvFilter, FmtSubscriber};
use url::Url;
use uuid::Uuid;

// Import our MCP server module
#[cfg(feature = "mcp")]
mod mcp_server;

// --- Constants ---

/// The embedding dimension used by the default OpenAI model (`text-embedding-3-small`).
/// This MUST match the dimension of the chosen embedding model.
const EMBEDDING_DIMENSION: u64 = 1536;
/// The number of points to upsert to Qdrant in a single batch.
const QDRANT_UPSERT_BATCH_SIZE: usize = 100;
/// The target size for text chunks in tokens before embedding.
/// Aiming for the maximum supported by models like text-embedding-3-small (8192).
const TARGET_CHUNK_SIZE_TOKENS: usize = 8192;
/// Maximum number of tokens to include in a single request to the embedding API.
/// OpenAI has a limit of 300,000 tokens per request for embeddings.
/// Setting this to 200,000 to provide a larger safety margin.
const MAX_TOKENS_PER_EMBEDDING_REQUEST: usize = 200_000;

// --- Command Line Arguments ---

/// Defines the command-line arguments accepted by the application.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the folder within a Git repository to scan.
    folder: Option<PathBuf>,

    /// Relevance cutoff for results (e.g., 0.7). Only results with a score
    /// above this threshold will be shown.
    #[arg(short, long, default_value_t = 0.7)]
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

    /// Subcommands
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Run as an MCP server using stdio for communication
    Mcp {
        /// Enable debug logging. By default, only errors are logged unless RUST_LOG is set.
        #[arg(long)]
        debug: bool,
    },
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
    /// A reference to the tokenizer for chunking.
    tokenizer: &'a CoreBPE,
}

impl<'a> Embed for LongDocument<'a> {
    /// Tokenizes the document's content, splits tokens into chunks, decodes chunks back to strings,
    /// and provides each non-empty chunk to the `TextEmbedder`.
    fn embed(&self, embedder: &mut TextEmbedder) -> Result<(), EmbedError> {
        // Tokenize the entire content using the referenced tokenizer
        let tokens = self.tokenizer.encode_ordinary(&self.content);

        if tokens.is_empty() {
            debug!("No tokens generated for file: {}", self.path_str);
            return Ok(()); // Nothing to embed
        }

        // Split the tokens into chunks of the target size
        for token_chunk in tokens.chunks(TARGET_CHUNK_SIZE_TOKENS) {
            // Decode the token chunk back to a string.
            // Using decode_bytes for potentially better handling of arbitrary bytes if needed,
            // though decode should work fine with encode_ordinary.
            match self.tokenizer.decode(token_chunk.to_vec()) {
                Ok(text_chunk) => {
                    let trimmed_chunk = text_chunk.trim();
                    if !trimmed_chunk.is_empty() {
                        // Pass ownership of the original decoded string (or trimmed if needed, but embedder likely handles)
                        embedder.embed(text_chunk);
                    } else {
                        debug!(
                            "Skipping empty decoded chunk for file: {}",
                            self.path_str
                        );
                    }
                }
                Err(e) => {
                    // Log the error and potentially skip this chunk or return an error.
                    // Returning an error might halt the whole embedding process for all files.
                    // Logging and skipping seems more robust for handling potentially rare decoding issues.
                    warn!(
                        "Failed to decode token chunk for file {}: {}. Skipping chunk.",
                        self.path_str, e
                    );
                    // Optionally, convert to EmbedError:
                    // return Err(EmbedError::Other(anyhow!("Failed to decode token chunk for {}: {}", self.path_str, e)));
                }
            }
        }
        Ok(())
    }
}

// Implement additional methods for LongDocument outside the Embed trait
impl<'a> LongDocument<'a> {
    // Add a method to estimate the number of embedding API calls this document will require
    fn estimated_embedding_calls(&self) -> usize {
        let tokens = self.tokenizer.encode_ordinary(&self.content);
        if tokens.is_empty() {
            return 0;
        }
        // Calculate how many chunks this document will be split into
        (tokens.len() + TARGET_CHUNK_SIZE_TOKENS - 1) / TARGET_CHUNK_SIZE_TOKENS
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
    let env_filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(default_level));

    let subscriber = FmtSubscriber::builder()
        .with_env_filter(env_filter)
        .with_writer(std::io::stderr)
        .finish();

    tracing::subscriber::set_global_default(subscriber).expect("setting default subscriber failed");
}

/// Generates a Qdrant collection name based on a hash of the canonical folder path.
/// Ensures the name is valid for Qdrant.
fn generate_collection_name(folder_path: &Path) -> Result<String> {
    // Canonicalize path for consistency across different invocations
    let canonical_path = fs::canonicalize(folder_path).with_context(|| {
        format!(
            "Failed to canonicalize folder path: {}",
            folder_path.display()
        )
    })?;
    let path_str = canonical_path.to_string_lossy();

    // Use SHA256 hash of the canonical path for a stable identifier
    let mut hasher = Sha256::new();
    hasher.update(path_str.as_bytes());
    let hash_bytes = hasher.finalize();
    let hash_hex = hex::encode(hash_bytes);

    // Use the first 16 chars of the hex hash for brevity, prefixed with "groma-"
    // Qdrant names must match ^[a-zA-Z0-9_-]{1,255}$
    let collection_name = format!("groma-{}", &hash_hex[..16]);
    debug!(
        "Generated collection name '{}' from path '{}'",
        collection_name, path_str
    );

    Ok(collection_name)
}

// Removed create_text_splitter function

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
                CreateCollectionBuilder::new(collection_name).vectors_config(VectorsConfig::from(
                    VectorParams {
                        size: EMBEDDING_DIMENSION,
                        distance: Distance::Cosine.into(),
                        ..Default::default() // Use default for other vector params
                    },
                )), // Consider adding payload indexing for 'path' and 'hash' here
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

/// Returns the expected path for the .gromastate file within the repository's workdir.
fn get_state_file_path(repo: &Repository) -> Result<PathBuf> {
    let workdir = repo
        .workdir()
        .ok_or_else(|| anyhow!("Cannot get state file path: repository is bare"))?;
    Ok(workdir.join(".gromastate")) // Store state file in the workdir root
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

    debug!("Loading state from '{}'", state_file_path.display());
    let file = fs::File::open(&state_file_path)
        .with_context(|| format!("Failed to open state file: {}", state_file_path.display()))?;
    let state: GromaState =
        serde_json::from_reader(io::BufReader::new(file)).with_context(|| {
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
    serde_json::to_writer_pretty(io::BufWriter::new(file), state).with_context(|| {
        format!(
            "Failed to serialize state to: {}",
            state_file_path.display()
        )
    })?;
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

/// Deletes all points associated with a specific file path from the Qdrant collection.
async fn delete_points_by_path(
    client: Arc<Qdrant>,
    collection_name: &str,
    path_str: &str,
) -> Result<()> {
    info!(
        "Deleting existing chunks for file '{}' from collection '{}'",
        path_str, collection_name
    );
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
async fn upsert_batch(
    client: Arc<Qdrant>,
    collection_name: &str,
    points: Vec<PointStruct>,
) -> Result<()> {
    // Takes ownership now
    if points.is_empty() {
        return Ok(());
    }
    client
        .upsert_points(
            UpsertPointsBuilder::new(collection_name, points), // Use owned points directly
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
    
    // Handle subcommands
    if let Some(command) = &args.command {
        match command {
            #[cfg(feature = "mcp")]
            Commands::Mcp { debug } => {
                // Use the debug flag from the subcommand
                let debug_enabled = args.debug || *debug;
                initialize_logging(debug_enabled);
                info!("Running in MCP server mode");
                return mcp_server::run_mcp_server().await;
            },
            #[cfg(not(feature = "mcp"))]
            _ => return Err(anyhow!("MCP feature is not enabled; rebuild with --features \"mcp\"")),
        }
    } else if let Some(folder) = &args.folder {
        // Initialize logging for CLI mode
        initialize_logging(args.debug);
        info!("Starting Groma process...");
        // If we get here, we're running in normal CLI mode
        // Initialize Tokenizer early
        info!("Stage: Initializing tokenizer...");
        let tokenizer = cl100k_base().context("Failed to load cl100k_base tokenizer")?;
        info!("Stage: Tokenizer initialized.");

        // 2. Canonicalize Folder Path (used for relative path calculation later)
        let canonical_folder_path = fs::canonicalize(folder).with_context(|| {
            format!(
                "Failed to canonicalize input folder path: {}",
                folder.display()
            )
        })?;
        info!(
            "Using canonical folder path: {}",
            canonical_folder_path.display()
        );
        info!("Stage: Canonicalized folder path.");

        // 3. Initialize Clients (OpenAI, Qdrant)
        info!("Stage: Initializing clients...");
        let config = GromaConfig {
            openai_key: args.openai_key.clone(),
            openai_model: args.openai_model.clone(),
            qdrant_url: args.qdrant_url.to_string(),
        };
        
        let (embedding_model, qdrant_client) = config.initialize_clients()?;
        info!(
            "Using OpenAI Embedding model: {} (Dimension: {})",
            config.openai_model, EMBEDDING_DIMENSION
        );
        info!("Connected to Qdrant at {}", config.qdrant_url);
        info!("Stage: Clients initialized.");

        // 4. Determine Collection Name & Ensure It Exists
        info!("Stage: Determining collection name...");
        let collection_name = generate_collection_name(folder)?;
        info!("Using Qdrant collection: {}", collection_name);
        ensure_qdrant_collection(qdrant_client.clone(), &collection_name).await?;
        info!("Stage: Qdrant collection ensured.");

        // 5. Discover Repository and Perform File Updates if not suppressed
        info!("Stage: Discovering Git repository...");
        let repo = Repository::discover(folder).with_context(|| {
            format!(
                "Failed to find Git repository containing path: {}",
                folder.display()
            )
        })?;
        let workdir = repo
            .workdir()
            .ok_or_else(|| anyhow!("Git repository is bare, cannot process files"))?;
        info!("Found Git repository at: {}", workdir.display());
        info!("Stage: Git repository discovered.");

        if !args.suppress_updates {
            info!("Stage: Starting file updates...");
            perform_file_updates(
                &args,
                &repo,               // Pass repository reference
                qdrant_client.clone(),
                embedding_model.clone(),
                &tokenizer, // Pass tokenizer reference
                &collection_name,
                &canonical_folder_path, // Pass canonical folder path
            )
            .await?;
        } else {
            info!("Skipping file updates (--suppress-updates specified).");
        }
        info!("Stage: File updates complete (or skipped).");

        // 6. Process User Query (Read, Embed, Search, Format Output)
        info!("Stage: Starting query processing...");
        process_query(
            &args,
            qdrant_client.clone(),
            embedding_model.clone(),
            &collection_name,
            &canonical_folder_path,
        )
        .await?;
        info!("Stage: Query processing complete.");

        info!("Groma process finished successfully.");
    } else {
        // Neither folder nor subcommand was provided
        return Err(anyhow!("Either a folder path or a subcommand must be provided. Use --help for more information."));
    }
    
    Ok(())
} // Close main function block here


/// Processes embedding results and adds corresponding PointStructs to the output vector.
fn process_embedding_results(
    embedding_results: &Vec<(LongDocument, OneOrMany<rig::embeddings::embedding::Embedding>)>, // Use imported OneOrMany
    all_points_to_upsert: &mut Vec<PointStruct>, // Mutably borrow the points vector
) -> Result<()> {
    debug!("Constructing Qdrant points from embedding results...");
    for (long_document, one_or_many_embeddings) in embedding_results.iter() { // Iterate over borrowed results
        // Convert OneOrMany<Embedding> into Vec<Embedding> for consistent processing
        // Use into_iter() and collect() as into_vec() doesn't exist
        let embedding_vec: Vec<_> = one_or_many_embeddings.clone().into_iter().collect();
        debug!(
            "Processing {} embedding(s) for document '{}'",
            embedding_vec.len(),
            long_document.path_str,
        );
        for (chunk_index, embedding) in embedding_vec.iter().enumerate() { // Iterate over the converted Vec
            let chunk_uuid = generate_uuid_for_chunk(&long_document.path_str, chunk_index);
            let point_id = uuid_to_point_id(chunk_uuid);

            let metadata = FileMetadata {
                path: long_document.path_str.clone(),
                hash: long_document.current_hash.clone(), // Store the full file hash
                chunk_index,
            };
            let payload: Payload = match serde_json::to_value(metadata)
                .context("Failed to serialize metadata")?
                .try_into()
            {
                Ok(p) => p,
                Err(e) => {
                    error!("Failed to convert metadata JSON to Qdrant Payload for chunk {} of {}: {}", chunk_index, long_document.path_str, e);
                    continue; // Skip this point
                }
            };

            // Convert f64 embedding from rig to f32 for Qdrant
            let vector_f32: Vec<f32> = embedding.vec.iter().map(|&v| v as f32).collect(); // Borrow v
            let vectors: qdrant_client::qdrant::Vectors = vector_f32.into();
            let point = PointStruct::new(point_id, vectors, payload);
            all_points_to_upsert.push(point);
        }
    }
    debug!("Finished constructing points for this batch.");
    Ok(())
}


/// Handles the core logic for detecting file changes using Git, embedding changes, and upserting to Qdrant.
async fn perform_file_updates(
    args: &Args,
    repo: &Repository, // Use Git repository
    qdrant_client: Arc<Qdrant>,
    embedding_model: Arc<openai::EmbeddingModel>,
    tokenizer: &CoreBPE, // Accept tokenizer reference
    collection_name: &str,
    canonical_folder_path: &Path, // Base path for filtering and relative paths
) -> Result<()> {
    info!("Entering perform_file_updates function...");
    info!("Checking for file updates using Git and processing changes...");

    let workdir = repo
        .workdir()
        .ok_or_else(|| anyhow!("Git repository is bare, cannot process files"))?;
    info!("Stage [Update]: Determined workdir.");

    // --- Phase 1: Load State & Determine Git Diff ---
    info!("Stage [Update]: Loading previous state...");
    let previous_state = load_state(repo)?;
    let head_commit_oid = repo.head()?.peel_to_commit()?.id();
    info!("Current HEAD OID: {}", head_commit_oid);
    info!("Stage [Update]: State loaded.");

    let previous_tree = match previous_state {
        Some(ref state) => {
            let oid = Oid::from_str(&state.last_processed_oid)?;
            match repo.find_commit(oid)?.tree() {
                Ok(tree) => Some(tree),
                Err(e) => {
                    warn!(
                        "Failed to find tree for previous OID {}: {}. Processing all files.",
                        state.last_processed_oid, e
                    );
                    None // Fallback to processing all if previous commit/tree is missing
                }
            }
        }
        None => None, // First run, compare against empty tree
    };
    info!("Stage [Update]: Determined previous Git tree.");

    // Diff HEAD against the previous tree (or empty tree if first run)
    // We compare against the committed state (HEAD) to ensure reproducibility.
    // Changes in the working directory or index that are not committed yet won't be processed.
    info!("Stage [Update]: Calculating Git diff against working directory...");
    // Compare the previous tree state (or empty if first run) against the current working directory
    let mut diff_opts = DiffOptions::new();
    diff_opts.include_ignored(false); // Don't include ignored files
    diff_opts.include_untracked(false); // Don't include untracked files
    diff_opts.include_typechange(true); // Detect type changes (file to dir etc.)
                                        // Convert the target folder path to be relative to the workdir for pathspec
    let folder = args.folder.as_ref().expect("Folder is required for file updates");
    let pathspec = folder.strip_prefix(workdir).unwrap_or(folder);
    diff_opts.pathspec(pathspec); // Limit diff to the target folder relative to repo root
    info!("Using pathspec for diff: {}", pathspec.display());

    // Use diff_tree_to_workdir to compare the last indexed tree state with the current working directory files
    let diff = repo.diff_tree_to_workdir(previous_tree.as_ref(), Some(&mut diff_opts))?;

    info!(
        "Git diff (vs workdir) found {} changed items potentially within the target folder.",
        diff.deltas().len()
    );
    info!("Stage [Update]: Git diff calculated (or skipped for first run).");

    // Removed text_splitter creation
    let mut documents_to_embed: Vec<LongDocument> = Vec::new();
    let mut paths_to_delete: Vec<String> = Vec::new();
    let mut processed_new_paths: std::collections::HashSet<PathBuf> =
        std::collections::HashSet::new(); // Track processed adds/renames
    let mut processed_count = 0;
    info!("Stage [Update]: Initialized variables for processing.");

    // --- Phase 2: Process Changes ---
    if previous_tree.is_none() {
        // --- First Run: Process all tracked files in HEAD within the target folder ---
        info!("Stage [Update]: First run detected. Processing all tracked files in target folder...");
        let current_tree = repo.head()?.peel_to_tree()?;
        current_tree.walk(git2::TreeWalkMode::PreOrder, |root, entry| {
            if let Some(entry_name) = entry.name() { // Use name() instead of path()
                let full_path = workdir.join(root).join(entry_name); // Construct full path
                // Filter: Ensure the path is within the canonical target folder
                if full_path.starts_with(canonical_folder_path) && entry.kind() == Some(git2::ObjectType::Blob) {
                    if let Ok(canonical_path) = fs::canonicalize(&full_path) {
                        let path_str = canonical_path.to_string_lossy().to_string();
                        info!("Processing (first run): {}", path_str);
                        processed_count += 1;
                        processed_new_paths.insert(canonical_path.clone()); // Track this path

                        match process_file_for_embedding(&canonical_path, tokenizer) { // Pass tokenizer
                            Ok(Some(doc)) => documents_to_embed.push(doc),
                            Ok(None) => info!("Skipping empty or unreadable file: {}", path_str),
                            Err(e) => warn!("Failed processing file {}: {}", path_str, e),
                        }
                    } else {
                         warn!("Could not canonicalize path during first run walk: {}", full_path.display());
                    }
                }
            }
            git2::TreeWalkResult::Ok // Continue walking
        })?;
        info!("Finished processing {} files during first run.", processed_count);

    } else {
        // --- Subsequent Run: Process Diff Deltas ---
        info!("Stage [Update]: Starting processing of Git diff deltas...");
        for diff_delta in diff.deltas() {
            let delta = diff_delta.status();
            let old_repo_path = diff_delta.old_file().path(); // Path relative to repo root
            let new_repo_path = diff_delta.new_file().path(); // Path relative to repo root

        // Construct absolute paths based on workdir
        let old_absolute_path = old_repo_path.map(|p| workdir.join(p));
        let new_absolute_path = new_repo_path.map(|p| workdir.join(p));

        // --- Crucial Filter: Ensure the *new* path (for Add/Modify/Rename) or *old* path (for Delete)
        // --- is actually within the *canonical target folder*. Git's pathspec is prefix-based
        // --- and might include files outside the exact target folder if names overlap.
        // Use Delta enum variants for status checks
        let relevant_path_for_filter = if delta == Delta::Deleted {
            old_absolute_path.as_ref()
        } else {
            new_absolute_path.as_ref() // Added, Modified, Renamed, Typechange etc.
        };

        if relevant_path_for_filter.map_or(true, |p| !p.starts_with(canonical_folder_path)) {
            debug!(
                 "Skipping delta outside target folder (canonical check): old={:?}, new={:?}, delta={:?}",
                 old_repo_path, new_repo_path, delta
             );
            continue;
        }
        // --- End Filter ---

        processed_count += 1; // Count files actually processed after filtering

        // Use if/else if with Delta enum variants
        if delta == Delta::Added || delta == Delta::Modified || delta == Delta::Typechange {
            if let Some(new_path_abs) = new_absolute_path {
                // Canonicalize AFTER confirming it starts with the canonical_folder_path prefix
                if let Ok(canonical_new) = fs::canonicalize(&new_path_abs) {
                    let path_str = canonical_new.to_string_lossy().to_string();
                    info!("Detected NEW/MODIFIED/TYPECHANGE: {}", path_str);
                    processed_new_paths.insert(canonical_new.clone()); // Track this path

                    // If modified/typechange, ensure old points are deleted (using canonical path)
                    // This handles cases where filename case might change but path object differs
                    if delta == Delta::Modified || delta == Delta::Typechange {
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
                    match process_file_for_embedding(&canonical_new, tokenizer) { // Pass tokenizer
                        Ok(Some(doc)) => documents_to_embed.push(doc),
                        Ok(None) => info!("Skipping empty or unreadable file: {}", path_str), // Empty file case
                        Err(e) => warn!(
                            "Failed processing NEW/MODIFIED/TYPECHANGE file {}: {}",
                            path_str, e
                        ),
                    }
                } else {
                    warn!(
                        "Could not canonicalize new path for NEW/MODIFIED/TYPECHANGE file: {}",
                        new_path_abs.display()
                    );
                }
            }
        } else if delta == Delta::Deleted {
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
        } else if delta == Delta::Renamed {
            if let (Some(old_path_abs), Some(new_path_abs)) = (old_absolute_path, new_absolute_path)
            {
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

                    info!(
                        "Detected RENAMED: {} -> {}",
                        old_path_str_for_delete, new_path_str
                    );

                    // Simple approach: Delete old, process new as ADDED
                    paths_to_delete.push(old_path_str_for_delete);
                    match process_file_for_embedding(&canonical_new, tokenizer) { // Pass tokenizer
                        Ok(Some(doc)) => documents_to_embed.push(doc),
                        Ok(None) => info!(
                            "Skipping empty or unreadable renamed file: {}",
                            new_path_str
                        ),
                        Err(e) => warn!("Failed processing RENAMED file {}: {}", new_path_str, e),
                    }
                } else {
                    warn!(
                        "Could not canonicalize new path for RENAMED file: {} -> {}",
                        old_path_abs.display(),
                        new_path_abs.display()
                    );
                }
            }
        } // Closes the `else if status == Status::INDEX_RENAMED` block.
          // Note: The if/else if chain handles NEW, MODIFIED, TYPECHANGE, DELETED, RENAMED.
          // We implicitly ignore other statuses by not having an `else` block here.
          // The debug log for ignored statuses was removed as it was part of the old `_` match arm.
            // Note: We are ignoring other statuses like CONFLICTED, IGNORED, UNTRACKED, etc.
            // as they shouldn't appear in a tree-to-tree diff reflecting committed changes.
        } // Closes the `for delta in diff.deltas()` loop
        info!(
            "Finished processing {} Git diff deltas relevant to the target folder.",
            processed_count
        );
        info!("Stage [Update]: Finished processing Git diff deltas.");
    } // Closes the else block for subsequent runs

    // --- Phase 3: Delete Obsolete Points ---
    info!("Stage [Update]: Starting deletion of obsolete Qdrant points...");
    // Deduplicate paths to delete before executing deletions
    paths_to_delete.sort();
    paths_to_delete.dedup();
    if !paths_to_delete.is_empty() {
        info!(
            "Deleting points for {} removed/modified/renamed paths...",
            paths_to_delete.len()
        );
        for path_str in paths_to_delete {
            // Ensure we don't delete paths that were immediately re-added (e.g., case-only rename on some systems)
            // This check might be overly cautious depending on exact diff behavior but safer.
            let path_buf = PathBuf::from(&path_str);
            if processed_new_paths.contains(&path_buf) {
                debug!(
                    "Skipping deletion for path that was also added/renamed in this run: {}",
                    path_str
                );
                continue;
            }
            if let Err(e) =
                delete_points_by_path(qdrant_client.clone(), collection_name, &path_str).await
            {
                warn!("Failed to delete points for path {}: {}", path_str, e);
            }
        }
    } else {
        info!("No points marked for deletion.");
    }
    info!("Stage [Update]: Finished deletion of obsolete Qdrant points.");

    // --- Phase 4 & 5: Batch Embed (Token-Based) and Create Points ---
    info!("Stage [Update]: Starting token-based batch embedding and point creation...");
    let mut all_points_to_upsert: Vec<PointStruct> = Vec::new();

    if !documents_to_embed.is_empty() {
        info!(
            "Processing {} documents for embedding, batching by token limit ({} tokens/request)...",
            documents_to_embed.len(),
            MAX_TOKENS_PER_EMBEDDING_REQUEST
        );

        // Temporary structure to hold data needed for batching without tokenizer reference
        struct DocForBatch {
            path_str: String,
            current_hash: String,
            content: String,
        }

        let mut current_batch_docs: Vec<DocForBatch> = Vec::new();
        let mut current_batch_tokens: usize = 0;
        let mut current_batch_chunks: usize = 0;
        
        // Calculate the maximum number of chunks we can safely include in a batch
        // Each chunk is TARGET_CHUNK_SIZE_TOKENS tokens, and we need to stay under MAX_TOKENS_PER_EMBEDDING_REQUEST
        let max_chunks_per_batch = MAX_TOKENS_PER_EMBEDDING_REQUEST / TARGET_CHUNK_SIZE_TOKENS;
        info!("Maximum chunks per embedding batch: {}", max_chunks_per_batch);

        for doc in documents_to_embed { // Consumes the vector
            let tokens = tokenizer.encode_ordinary(&doc.content);
            let doc_token_count = tokens.len();
            
            // Estimate how many chunks this document will generate
            let doc_chunks = doc.estimated_embedding_calls();
            
            // Skip documents that are themselves too large for a single request
            if doc_token_count > MAX_TOKENS_PER_EMBEDDING_REQUEST {
                warn!("Skipping document {} because its token count ({}) exceeds the per-request limit ({})",
                      doc.path_str, doc_token_count, MAX_TOKENS_PER_EMBEDDING_REQUEST);
                continue;
            }

            // If adding the current document would exceed the chunk limit, process the existing batch first
            if !current_batch_docs.is_empty() && 
               (current_batch_chunks + doc_chunks > max_chunks_per_batch || 
                current_batch_tokens + doc_token_count > MAX_TOKENS_PER_EMBEDDING_REQUEST) {
                info!(
                    "Processing embedding batch. Token count: {}, Chunk count: {}, Document count: {}",
                    current_batch_tokens, current_batch_chunks, current_batch_docs.len()
                );

                // Convert DocForBatch back to LongDocument for EmbeddingsBuilder
                let batch_long_docs: Vec<LongDocument> = current_batch_docs.iter().map(|d| LongDocument {
                    path_str: d.path_str.clone(),
                    current_hash: d.current_hash.clone(),
                    content: d.content.clone(),
                    tokenizer, // Pass the reference
                }).collect();

                // Call embedding API
                match EmbeddingsBuilder::new((*embedding_model).clone())
                    .documents(batch_long_docs)? // Pass owned Vec for the batch
                    .build()
                    .await {
                        Ok(embedding_results) => {
                             info!("Successfully generated embeddings for batch.");
                             // Process results and add points to all_points_to_upsert
                             process_embedding_results(&embedding_results, &mut all_points_to_upsert)?;
                             info!("Finished constructing points for batch. Total points so far: {}", all_points_to_upsert.len());
                        },
                        Err(e) => {
                            // Log the error and continue to the next batch, or propagate?
                            // Propagating seems safer to indicate failure.
                             error!("Failed to generate embeddings for batch: {}", e);
                             return Err(anyhow!("Failed to generate embeddings for batch: {}", e)
                                        .context("Error during token-based batch embedding processing"));
                        }
                    }

                // Clear the batch for the next set of documents
                current_batch_docs.clear();
                current_batch_tokens = 0;
                current_batch_chunks = 0;
            }

            // Now add the current document to the new or existing batch
            current_batch_tokens += doc_token_count;
            current_batch_chunks += doc_chunks;
            current_batch_docs.push(DocForBatch {
                path_str: doc.path_str, // Move strings
                current_hash: doc.current_hash, // Move strings
                content: doc.content, // Move strings
            });
        }

        // --- Process the final batch if any documents remain ---
        if !current_batch_docs.is_empty() {
            info!(
                "Processing final embedding batch. Token count: {}, Chunk count: {}, Document count: {}",
                current_batch_tokens, current_batch_chunks, current_batch_docs.len()
            );
            
            let final_batch_long_docs: Vec<LongDocument> = current_batch_docs.iter().map(|d| LongDocument {
                path_str: d.path_str.clone(),
                current_hash: d.current_hash.clone(),
                content: d.content.clone(),
                tokenizer,
            }).collect();

            match EmbeddingsBuilder::new((*embedding_model).clone())
                .documents(final_batch_long_docs)?
                .build()
                .await {
                    Ok(embedding_results) => {
                         info!("Successfully generated embeddings for final batch.");
                         process_embedding_results(&embedding_results, &mut all_points_to_upsert)?;
                         info!("Finished constructing points for final batch. Total points so far: {}", all_points_to_upsert.len());
                    },
                    Err(e) => {
                         error!("Failed to generate embeddings for final batch: {}", e);
                         return Err(anyhow!("Failed to generate embeddings for final batch: {}", e)
                                    .context("Error during final token-based batch embedding processing"));
                    }
                }
        }

        info!(
            "Finished embedding and constructing {} total points from all batches.",
            all_points_to_upsert.len()
        );

    } else {
        info!("No documents require embedding.");
    }
    info!("Stage [Update]: Finished embedding and point creation.");

    // --- Phase 6: Batch Upsert All Collected Points ---
    info!("Stage [Update]: Starting batch upsert to Qdrant...");
    if !all_points_to_upsert.is_empty() {
        info!(
            "Starting batch upsert of {} total points to collection '{}'...",
            all_points_to_upsert.len(),
            collection_name
        );
        // Upsert in batches
        for chunk in all_points_to_upsert
            .chunks(QDRANT_UPSERT_BATCH_SIZE)
            .map(|c| c.to_vec())
        {
            debug!("Upserting batch of {} points...", chunk.len());
            // Pass ownership of the chunk Vec to upsert_batch
            upsert_batch(qdrant_client.clone(), collection_name, chunk).await?;
        }
        info!("Finished upserting points.");
    } else {
        info!("No new points to upsert.");
    }
    info!("Stage [Update]: Finished batch upsert to Qdrant.");

    // --- Phase 7: Save State ---
    info!("Stage [Update]: Saving state...");
    // Only save state if the diff processing and upserting seemed successful
    // (Error handling within the loops might allow partial success, consider if this is desired)
    let current_state = GromaState {
        last_processed_oid: head_commit_oid.to_string(),
    };
    save_state(repo, &current_state)?;
    info!("Stage [Update]: State saved.");

    info!("Exiting perform_file_updates function.");
    Ok(())
}

/// Helper function to read, hash, and prepare a single file for embedding.
/// Returns Ok(None) if the file is empty or cannot be read.
fn process_file_for_embedding<'a>(
    file_path: &Path,
    tokenizer: &'a CoreBPE, // Accept tokenizer reference
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
                    tokenizer, // Pass borrowed tokenizer
                }))
            }
        }
        Err(e) => {
            // Log error but return Ok(None) to allow skipping problematic files
            warn!(
                "Failed to read file content for {}: {}. Skipping file.",
                path_for_error, e
            );
            Ok(None)
        }
    }
}

/// Configuration structure that can be used by both CLI and MCP server
#[derive(Clone)]
pub struct GromaConfig {
    pub openai_key: String,
    pub openai_model: String,
    pub qdrant_url: String,
}

impl GromaConfig {
    /// Create a new configuration from environment variables or defaults
    pub fn from_env() -> Result<Self> {
        let openai_key = std::env::var("OPENAI_API_KEY")
            .context("OPENAI_API_KEY environment variable not set")?;
        
        let openai_model = std::env::var("OPENAI_MODEL")
            .unwrap_or_else(|_| "text-embedding-3-small".to_string());
        
        let qdrant_url = std::env::var("QDRANT_URL")
            .unwrap_or_else(|_| "http://localhost:6334".to_string());
        
        Ok(Self {
            openai_key,
            openai_model,
            qdrant_url,
        })
    }
    
    /// Initialize clients using this configuration
    pub fn initialize_clients(&self) -> Result<(Arc<openai::EmbeddingModel>, Arc<Qdrant>)> {
        // Create OpenAI client
        let openai_client = rig::providers::openai::Client::new(&self.openai_key);
        let embedding_model = Arc::new(openai_client.embedding_model(&self.openai_model));
        
        // Create Qdrant client
        let qdrant_client = Arc::new(Qdrant::from_url(&self.qdrant_url).build()
            .context(format!("Failed to connect to Qdrant at {}", self.qdrant_url))?);
        
        Ok((embedding_model, qdrant_client))
    }
}

/// Helper function to prepare everything needed for a query
pub async fn prepare_for_query(
    folder_path: &Path,
    _cutoff: f32, // Unused for now, but kept for future use
    config: &GromaConfig,
) -> Result<(Arc<openai::EmbeddingModel>, Arc<Qdrant>, String, PathBuf)> {
    // Get canonical folder path
    let canonical_folder_path = fs::canonicalize(folder_path)
        .with_context(|| format!("Failed to canonicalize folder path: {}", folder_path.display()))?;
    
    // Initialize clients
    let (embedding_model, qdrant_client) = config.initialize_clients()?;
    
    // Get collection name
    let collection_name = generate_collection_name(folder_path)?;
    
    Ok((embedding_model, qdrant_client, collection_name, canonical_folder_path))
}

/// Core query processing function that can be reused by both CLI and MCP server
/// Returns a JSON-serializable structure with search results
pub async fn process_query_core(
    query: &str,
    qdrant_client: Arc<Qdrant>,
    embedding_model: Arc<openai::EmbeddingModel>,
    collection_name: &str,
    canonical_folder_path: &Path,
    cutoff: f32,
) -> Result<serde_json::Value> {
    // --- Phase 1: Embed Query ---
    info!("Processing query: '{}'", query);
    info!("Stage [Query Core]: Embedding query...");
    let query_embedding = embedding_model
        .embed_text(query)
        .await
        .context("Failed to embed query")?;
    info!("Stage [Query Core]: Query embedded.");

    // --- Phase 2: Search Qdrant ---
    info!("Stage [Query Core]: Searching Qdrant...");
    info!("Searching for relevant files...");
    let query_vector_f32: Vec<f32> = query_embedding.vec.into_iter().map(|v| v as f32).collect();

    let search_result = qdrant_client
        .search_points(
            SearchPointsBuilder::new(collection_name, query_vector_f32, 100) // Limit results
                .with_payload(true) // Request payload to get file path
                .score_threshold(cutoff), // Apply relevance cutoff
        )
        .await
        .context("Failed to search Qdrant")?;
    info!(
        "Found {} potential matching chunks.",
        search_result.result.len()
    );
    info!("Stage [Query Core]: Qdrant search complete.");

    // --- Phase 3: Aggregate Results by File Path ---
    info!("Stage [Query Core]: Aggregating results by file path...");
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
                warn!(
                    "Found hit with non-string 'path' payload: {:?}",
                    hit.payload
                );
            }
        } else {
            warn!("Found hit without 'path' payload: {:?}", hit.payload);
        }
    }
    info!("Stage [Query Core]: Results aggregated.");

    // --- Phase 4: Format Aggregated Results ---
    info!("Stage [Query Core]: Formatting results...");
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
    let json_output = serde_json::json!({
        "files_by_relevance": aggregated_results
    });
    
    info!("Stage [Query Core]: Results formatted.");
    Ok(json_output)
}

/// Handles reading the user query, embedding it, searching Qdrant, and printing results.
async fn process_query(
    args: &Args,
    qdrant_client: Arc<Qdrant>,
    embedding_model: Arc<openai::EmbeddingModel>,
    collection_name: &str,
    canonical_folder_path: &Path, // Needed for making paths relative
) -> Result<()> {
    info!("Entering process_query function...");
    // --- Phase 5: Read Query from Stdin ---
    info!("Stage [Query]: Reading query from stdin...");
    info!("Please enter your query and press Enter (Ctrl+D to finish):");
    let mut query = String::new();
    io::stdin()
        .read_to_string(&mut query)
        .context("Failed to read query from stdin")?;
    let query = query.trim();

    if query.is_empty() {
        info!("No query provided. Exiting.");
        return Ok(());
    }
    info!("Processing query: '{}'", query);
    info!("Stage [Query]: Query read.");

    // Call the core query processing function
    let json_output = process_query_core(
        query, 
        qdrant_client, 
        embedding_model, 
        collection_name, 
        canonical_folder_path,
        args.cutoff
    ).await?;

    // Print the results
    info!("Stage [Query]: Printing results...");
    if json_output["files_by_relevance"].as_array().map_or(true, |arr| arr.is_empty()) {
        info!(
            "No files found matching the query with cutoff {}",
            args.cutoff
        );
        println!("{}", serde_json::json!({ "files_by_relevance": [] }));
    } else {
        info!("Printing results as JSON:");
        match serde_json::to_string_pretty(&json_output) {
            Ok(json_str) => println!("{}", json_str),
            Err(e) => error!("Failed to serialize results to JSON: {}", e),
        }
    }
    info!("Stage [Query]: Results printed.");

    info!("Exiting process_query function.");
    Ok(())
}
