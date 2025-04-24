use anyhow::{anyhow, Context, Result};
use clap::Parser;
// Removed unused futures imports
// Use the new Qdrant client struct and builder patterns
use qdrant_client::{
    Payload, // Import Payload struct directly from crate root
    qdrant::{
        point_id::PointIdOptions, CreateCollectionBuilder, Distance, PointStruct,
        SearchPointsBuilder, VectorParams, VectorsConfig, PointId, // Removed GetPointsBuilder
        UpsertPointsBuilder, // Import UpsertPointsBuilder
        PayloadIncludeSelector, // Import PayloadIncludeSelector
    },
    Qdrant, // Use the new Qdrant struct
};
// Use the main `rig` crate
use rig::{
    embeddings::embedding::EmbeddingModel, // Import the trait
    // Removed unused Embeddings, EmbeddingsBuilder
    // Removed unused vector_store imports (Point, PointData, VectorStoreIndex)
    providers::openai, // Import the openai provider module
};
// Removed unused import: use rig_qdrant::QdrantVectorStore;
use serde::{Deserialize, Serialize};
use git2::Repository; // Import Repository from git2
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::{self, Read},
    path::{Path, PathBuf},
    sync::Arc,
};
// Removed tabled imports
use tracing::{debug, error, info, warn, Level};
use tracing_subscriber::FmtSubscriber;
use url::Url;
// Import TextSplitter
use text_splitter::TextSplitter; // Removed note about ChunkConfig
// Import the specific tokenizer type needed for TextSplitter signature
use tiktoken_rs::{cl100k_base, CoreBPE};
use uuid::Uuid;

// Removed const QDRANT_COLLECTION_NAME
// This needs to match the output dimension of the chosen OpenRouter embedding model
// Example: text-embedding-3-small is 1536, text-embedding-ada-002 is 1536
// Let's make it configurable or fetch dynamically if possible, but start with a common default.
const EMBEDDING_DIMENSION: u64 = 1536;
const BATCH_SIZE: usize = 32; // For batch embedding generation
const QDRANT_UPSERT_BATCH_SIZE: usize = 100; // For batch upserting to Qdrant

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the folder to scan
    #[arg(short, long)]
    folder: PathBuf,

    /// Relevance cutoff for results (e.g., 0.7)
    #[arg(short, long)]
    cutoff: f32,

    /// OpenAI API Key
    #[arg(long, env = "OPENAI_API_KEY")]
    openai_key: String,

    /// OpenAI Embedding Model name (e.g., "text-embedding-3-small")
    #[arg(long, default_value = "text-embedding-3-small")]
    openai_model: String,

    /// Qdrant server URL (points to gRPC port)
    #[arg(long, env = "QDRANT_URL", default_value = "http://localhost:6334")]
    qdrant_url: Url,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct FileMetadata {
    path: String, // Original file path
    hash: String, // Hash of the entire original file
    chunk_index: usize, // Index of this chunk within the file
}

// Removed ResultRow struct as it's no longer needed for tabled output

// Helper to generate a stable UUID for a specific chunk
fn generate_uuid_for_chunk(path_str: &str, chunk_index: usize) -> Uuid {
    let identifier = format!("{}:{}", path_str, chunk_index);
    Uuid::new_v5(&Uuid::NAMESPACE_DNS, identifier.as_bytes())
}

// Helper to convert Uuid to Qdrant PointId
fn uuid_to_point_id(uuid: Uuid) -> PointId {
    PointId {
        // Use the correct enum path PointIdOptions
        point_id_options: Some(PointIdOptions::Uuid(uuid.to_string())),
    }
}


#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing subscriber for logging to stderr
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO) // Adjust level as needed (e.g., DEBUG for more details)
        .with_writer(std::io::stderr) // Configure the writer to use stderr
        .finish();
    tracing::subscriber::set_global_default(subscriber)
        .expect("setting default subscriber failed");

    // Parse CLI arguments
    let args = Args::parse();

    // --- Initialize Clients ---
    info!("Initializing clients...");

    // --- OpenAI Client & Embedding Model ---
    // Use the standard OpenAI client from the rig crate
    let openai_client = rig::providers::openai::Client::new(
        &args.openai_key, // Use the provided OpenAI key
    );

    let embedding_model: Arc<openai::EmbeddingModel> = Arc::new(
        openai_client
            .embedding_model(&args.openai_model) // Use the specified OpenAI model name
    );
    info!(
        "Using OpenAI Embedding model: {} (Dimension: {})", // Note: EMBEDDING_DIMENSION might need adjustment based on the actual model
        args.openai_model, EMBEDDING_DIMENSION
    );

    // Qdrant Client (use new Qdrant struct and builder)
    let qdrant_client = Arc::new(Qdrant::from_url(&args.qdrant_url.to_string()).build()?);
    info!("Connected to Qdrant at {}", args.qdrant_url);

    // Generate collection name based on folder path
    let collection_name = generate_collection_name(&args.folder)?;
    info!("Using Qdrant collection: {}", collection_name);


    // Ensure Qdrant collection exists
    ensure_qdrant_collection(qdrant_client.clone(), &collection_name).await?;

    // Create the text splitter once
    let text_splitter = create_text_splitter()?;

    // --- Process Files ---
    info!("Scanning folder: {}", args.folder.display());
    let files_to_scan = scan_folder(&args.folder)?;
    info!("Found {} tracked files to potentially process.", files_to_scan.len());

    let mut all_points_to_upsert: Vec<PointStruct> = Vec::new();
    let mut total_processed_files = 0;
    let mut total_skipped_files = 0;
    let mut total_failed_files = 0;

    for file_path in files_to_scan {
        let path_str = file_path.to_string_lossy().to_string();

        // Pass the text_splitter reference to process_file
        match process_file(
            qdrant_client.clone(),
            embedding_model.clone(),
            &text_splitter, // Pass splitter reference
            &text_splitter, // Pass splitter reference
            &file_path,
            &path_str,
            &collection_name, // Pass collection name
        )
        .await
        {
            Ok(new_points) => {
                if new_points.is_empty() {
                    // File was skipped (up-to-date or empty)
                    total_skipped_files += 1;
                } else {
                    // File was processed, add its points to the main list
                    total_processed_files += 1;
                    all_points_to_upsert.extend(new_points);

                    // Check if the accumulated batch is large enough to upsert
                    if all_points_to_upsert.len() >= QDRANT_UPSERT_BATCH_SIZE {
                        info!(
                            "Upserting batch of {} points to collection '{}'...",
                            all_points_to_upsert.len(),
                            collection_name
                        );
                        upsert_batch(qdrant_client.clone(), &collection_name, &all_points_to_upsert).await?;
                        all_points_to_upsert.clear();
                    }
                }
            }
            Err(e) => {
                warn!("Failed to process file {}: {}", path_str, e);
                total_failed_files += 1;
            }
        }
    }

    // Upsert any remaining points
    if !all_points_to_upsert.is_empty() {
        info!(
            "Upserting final batch of {} points to collection '{}'...",
            all_points_to_upsert.len(),
            collection_name
        );
        upsert_batch(qdrant_client.clone(), &collection_name, &all_points_to_upsert).await?;
        all_points_to_upsert.clear();
    }

    info!(
        "File processing complete. Processed: {}, Skipped: {}, Failed: {}",
        total_processed_files, total_skipped_files, total_failed_files
    );

    // --- Read Query from Stdin ---
    info!("Please enter your query and press Enter (or Ctrl+D to finish):");
    let mut query = String::new();
    io::stdin().read_to_string(&mut query)?;
    let query = query.trim();

    if query.is_empty() {
        info!("No query provided. Exiting.");
        return Ok(());
    }
    info!("Processing query...");

    // --- Embed Query ---
    // Use embed_text and pass the query as &str
    let query_embedding = embedding_model
        .embed_text(query) // Use embed_text and pass &str directly
        .await
        .context("Failed to embed query")?;

    // --- Search Qdrant ---
    info!("Searching for relevant files...");
    // Convert query embedding Vec<f64> to Vec<f32> for Qdrant
    let query_vector_f32: Vec<f32> = query_embedding.vec.into_iter().map(|v| v as f32).collect();

    // Use the builder pattern for search_points
    let search_result = qdrant_client
        .search_points(
            SearchPointsBuilder::new(&collection_name, query_vector_f32, 100) // Use dynamic collection name
                .with_payload(true) // Request payload
                .score_threshold(args.cutoff), // Apply cutoff
        )
        .await
        .context("Failed to search Qdrant")?;

    info!(
        "Found {} potential matching chunks.",
        search_result.result.len()
    );

    // --- Aggregate Results by File Path ---
    use std::collections::HashMap;

    let mut file_scores: HashMap<String, f32> = HashMap::new();

    for hit in search_result.result {
        if let Some(path_val) = hit.payload.get("path") {
            if let Some(path_str) = path_val.as_str() {
                let score = hit.score;
                // Insert or update the score, keeping the highest one
                file_scores
                    .entry(path_str.to_string())
                    .and_modify(|e| *e = e.max(score))
                    .or_insert(score);
            }
        }
    }

    // --- Format and Print Aggregated Results as JSON ---
    // Convert the HashMap into a Vec of (score, path) tuples
    let mut aggregated_results: Vec<(f32, String)> = file_scores
        .into_iter()
        .map(|(path, score)| (score, path)) // Create tuples (score, path)
        .collect();

    // Sort by score descending
    aggregated_results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    // Create the final JSON structure
    let mut json_output = HashMap::new();
    json_output.insert("files_by_relevance", aggregated_results); // Insert the sorted list

    if json_output["files_by_relevance"].is_empty() {
        info!("No files found matching the query with cutoff {}", args.cutoff);
        // Optionally print an empty JSON object or a specific message
        println!("{}", serde_json::json!({ "files_by_relevance": [] }));
    } else {
        info!("Printing results as JSON:");
        // Serialize to JSON and print
        match serde_json::to_string_pretty(&json_output) { // Use to_string_pretty for readability
            Ok(json_str) => println!("{}", json_str),
            Err(e) => error!("Failed to serialize results to JSON: {}", e),
        }
    }

    Ok(())
}

// Helper function to generate a Qdrant collection name from a folder path
fn generate_collection_name(folder_path: &Path) -> Result<String> {
    let canonical_path = fs::canonicalize(folder_path)
        .with_context(|| format!("Failed to canonicalize folder path: {}", folder_path.display()))?;
    let path_str = canonical_path.to_string_lossy();

    // Use SHA256 for hashing the canonical path
    let mut hasher = Sha256::new();
    hasher.update(path_str.as_bytes());
    let hash_bytes = hasher.finalize();
    let hash_hex = hex::encode(hash_bytes);

    // Use the first 16 chars of the hex hash for brevity, prefixed
    // Qdrant names must match ^[a-zA-Z0-9_-]{1,255}$
    let collection_name = format!("groma-{}", &hash_hex[..16]);

    Ok(collection_name)
}


// Helper function to create a text splitter configured for token-based chunking
// Update the return type to use the concrete CoreBPE tokenizer type
fn create_text_splitter() -> Result<TextSplitter<CoreBPE>> {
    let tokenizer = cl100k_base().context("Failed to load cl100k_base tokenizer")?;
    // Configure the splitter directly. Chunk size is passed to .chunks() later.
    // Overlap is handled implicitly by the splitter algorithm aiming for the target size.
    Ok(TextSplitter::new(tokenizer)
        .with_trim_chunks(true)) // Use with_trim_chunks
}
// Note: Chunk size (e.g., 512) and overlap (e.g., 50) are now handled
// differently. Size is passed to .chunks(), overlap is less explicit.


// Update function signature to accept collection_name
async fn ensure_qdrant_collection(client: Arc<Qdrant>, collection_name: &str) -> Result<()> {
    // Use new list_collections method
    let collections_list = client.list_collections().await?;
    if !collections_list
        .collections
        .iter()
        .any(|c| c.name == collection_name) // Use dynamic name
    {
        info!("Collection '{}' not found. Creating...", collection_name); // Use dynamic name
        // Use new create_collection method with builder
        client
            .create_collection(
                CreateCollectionBuilder::new(collection_name) // Use dynamic name
                    .vectors_config(VectorsConfig::from(VectorParams {
                        size: EMBEDDING_DIMENSION,
                        distance: Distance::Cosine.into(),
                        ..Default::default()
                    }))
                    // Add payload indexing for faster filtering/lookup if needed, especially on 'path' or 'hash'
                    // .payload_schema(...) // Consider adding schema for path/hash indexing
            )
            .await?;
        info!("Collection '{}' created.", collection_name); // Use dynamic name
        // It might be wise to explicitly create payload indices here too
        // client.create_payload_index(collection_name, "path", FieldType::Keyword, ...).await?;
        // client.create_payload_index(collection_name, "hash", FieldType::Keyword, ...).await?;

    } else {
        info!("Collection '{}' already exists.", collection_name); // Use dynamic name
    }
    Ok(())
}

// Updated scan_folder to use git ls-files logic
fn scan_folder(folder_path: &Path) -> Result<Vec<PathBuf>> {
    info!("Scanning for Git tracked files in: {}", folder_path.display());

    // Discover the git repository containing the folder_path
    let repo = Repository::discover(folder_path)
        .with_context(|| format!("Failed to find Git repository containing path: {}", folder_path.display()))?;

    let workdir = repo.workdir().ok_or_else(|| anyhow!("Git repository is bare, cannot list files in workdir"))?;
    info!("Found Git repository at: {}", workdir.display());

    // Ensure the input folder_path is absolute for correct prefix matching
    let absolute_folder_path = fs::canonicalize(folder_path)?;

    let mut tracked_files = Vec::new();
    let index = repo.index().context("Failed to get repository index")?;

    for entry in index.iter() {
        // entry.path is relative to the repository root
        let repo_relative_path = PathBuf::from(String::from_utf8_lossy(&entry.path).as_ref());
        let absolute_path = workdir.join(&repo_relative_path);

        // Check if the file exists and is within the target folder
        if absolute_path.is_file() && absolute_path.starts_with(&absolute_folder_path) {
            // Use canonicalize to ensure consistent path format
            if let Ok(canonical_path) = fs::canonicalize(&absolute_path) {
                 tracked_files.push(canonical_path);
            } else {
                warn!("Could not canonicalize path for tracked file: {}", absolute_path.display());
            }
        }
    }

    if tracked_files.is_empty() {
        warn!("No tracked files found within the specified folder: {}", folder_path.display());
    } else {
        info!("Found {} tracked files in the specified folder.", tracked_files.len());
    }

    Ok(tracked_files)
}


fn calculate_hash(file_path: &Path) -> Result<String> {
    let mut file = fs::File::open(file_path)
        .with_context(|| format!("Failed to open file for hashing: {}", file_path.display()))?;
    let mut hasher = Sha256::new();
    io::copy(&mut file, &mut hasher)
        .with_context(|| format!("Failed to read file for hashing: {}", file_path.display()))?;
    let hash_bytes = hasher.finalize();
    Ok(hex::encode(hash_bytes))
} // <-- Added missing closing brace

// Use the correct import path for MatchValue and remove unused Match
use qdrant_client::qdrant::{
    r#match::MatchValue, // Correct import path for MatchValue
    Condition, Filter, // Removed unused PointsSelector, Match, and DeletePointsBuilder
}; // Imports for filtering/deleting

// Fetches the hash of *one* existing chunk for a given file path.
// Update signature to accept collection_name
async fn get_existing_file_hash(client: Arc<Qdrant>, collection_name: &str, path_str: &str) -> Result<Option<String>> {
    // Use MatchValue::Keyword directly in Condition::matches (Corrected case)
    let filter = Filter::must([Condition::matches(
        "path", // Field name in payload
        MatchValue::Keyword(path_str.to_string()), // Use MatchValue::Keyword (capital K)
    )]);

    // Use search instead of get, limit to 1, only fetch hash payload
    let search_req = SearchPointsBuilder::new(collection_name, vec![0.0; EMBEDDING_DIMENSION as usize], 1) // Use dynamic collection name
        .filter(filter)
        .with_payload(PayloadIncludeSelector { fields: vec!["hash".to_string()] })
        .with_vectors(false);

    let search_result = client.search_points(search_req).await?;

    if let Some(point) = search_result.result.into_iter().next() {
        if let Some(hash_value) = point.payload.get("hash") {
            return Ok(hash_value.as_str().map(String::from));
        }
    }
    Ok(None)
}

// Deletes all points associated with a specific file path using a filter.
// Update signature to accept collection_name
async fn delete_points_by_path(client: Arc<Qdrant>, collection_name: &str, path_str: &str) -> Result<()> {
    info!("Deleting existing chunks for file '{}' from collection '{}'", path_str, collection_name);
    // Use MatchValue::Keyword directly in Condition::matches (Corrected case)
    let filter = Filter::must([Condition::matches(
        "path",
        MatchValue::Keyword(path_str.to_string()), // Use MatchValue::Keyword (capital K)
    )]);

    // Manually construct DeletePoints to bypass potential builder method issue
    let delete_request = qdrant_client::qdrant::DeletePoints {
        collection_name: collection_name.to_string(), // Use dynamic collection name
        wait: None, // Or Some(true) if needed
        ordering: None,
        // Use the 'points' field instead of 'points_selector'
        points: Some(qdrant_client::qdrant::PointsSelector {
            points_selector_one_of: Some(
                qdrant_client::qdrant::points_selector::PointsSelectorOneOf::Filter(filter),
            ),
        }),
        shard_key_selector: None, // Assuming None is appropriate for this version/setup
    };

    client.delete_points(delete_request).await?; // Pass the constructed struct
    Ok(())
}


// Processes a single file: checks hash, deletes old chunks if needed,
// chunks content, embeds chunks, and returns points to be upserted.
// Update function signature to use the concrete TextSplitter<CoreBPE> type
async fn process_file<E>(
    qdrant_client: Arc<Qdrant>,
    embedding_model: Arc<E>,
    text_splitter: &TextSplitter<CoreBPE>, // Use concrete CoreBPE type
    file_path: &Path,
    path_str: &str,
    collection_name: &str, // Add collection_name parameter
) -> Result<Vec<PointStruct>> // Return Vec instead of Option
where
    E: EmbeddingModel + Send + Sync + 'static,
{
    debug!("Processing file: {} for collection {}", path_str, collection_name);
    let current_hash = calculate_hash(file_path)?;
    // Pass collection_name to get_existing_file_hash
    let existing_hash = get_existing_file_hash(qdrant_client.clone(), collection_name, path_str).await?;

    if Some(&current_hash) == existing_hash.as_ref() {
        debug!("File hash matches, skipping: {} in collection {}", path_str, collection_name);
        return Ok(Vec::new()); // Return empty vec, no points to upsert
    }

    info!(
        "File is new or hash changed. Processing and embedding chunks for: {}",
        path_str
    );

    // If hash differs or file is new, delete existing points for this path
    if existing_hash.is_some() {
        // Pass collection_name to delete_points_by_path
        delete_points_by_path(qdrant_client.clone(), collection_name, path_str).await?;
    }

    let content = fs::read_to_string(file_path)
        .with_context(|| format!("Failed to read file content: {}", file_path.display()))?;

    if content.trim().is_empty() {
        warn!("Skipping empty file: {}", path_str);
        return Ok(Vec::new());
    }

    // Chunk the content using the text splitter's token-based chunks method
    // Pass the desired chunk size (in tokens) here.
    const TARGET_CHUNK_SIZE_TOKENS: usize = 512;
    let chunks: Vec<&str> = text_splitter.chunks(&content, TARGET_CHUNK_SIZE_TOKENS).collect(); // Use .chunks() and pass size
    info!("Split '{}' into {} chunks (target size: {} tokens).", path_str, chunks.len(), TARGET_CHUNK_SIZE_TOKENS);

    let mut points_to_upsert = Vec::with_capacity(chunks.len());

    // Embed chunks in batches for efficiency
    let chunk_batches = chunks.chunks(BATCH_SIZE); // Use BATCH_SIZE defined earlier

    for (batch_index, text_batch) in chunk_batches.enumerate() {
        debug!("Embedding batch {} for file {}", batch_index, path_str);
        // Convert Vec<&str> to Vec<String> for embed_texts
        let text_batch_strings: Vec<String> = text_batch.iter().map(|&s| s.to_string()).collect();
        // Use embed_texts for batching with the converted strings
        let embeddings = match embedding_model.embed_texts(text_batch_strings).await {
             Ok(embs) => embs,
             Err(e) => {
                 error!("Failed to embed batch {} for file {}: {}", batch_index, path_str, e);
                 // Skip this batch, or potentially the whole file? For now, skip batch.
                 continue; // Or return Err(...) if one batch failure should stop the file processing
             }
         };

        // Create points for the successful batch
        for (i, embedding) in embeddings.into_iter().enumerate() {
            let chunk_index = batch_index * BATCH_SIZE + i; // Calculate overall chunk index
            let chunk_uuid = generate_uuid_for_chunk(path_str, chunk_index);
            let point_id = uuid_to_point_id(chunk_uuid);

            let metadata = FileMetadata {
                path: path_str.to_string(),
                hash: current_hash.clone(), // Use the hash of the whole file
                chunk_index,
            };
            let payload: Payload = match serde_json::to_value(metadata) {
                Ok(val) => match val.try_into() {
                     Ok(p) => p,
                     Err(e) => {
                         error!("Failed to convert metadata to Qdrant Payload for chunk {} of {}: {}", chunk_index, path_str, e);
                         continue; // Skip this chunk
                     }
                 },
                 Err(e) => {
                     error!("Failed to serialize metadata for chunk {} of {}: {}", chunk_index, path_str, e);
                     continue; // Skip this chunk
                 }
             };


            let vector_f32: Vec<f32> = embedding.vec.into_iter().map(|v| v as f32).collect();
            let vectors: qdrant_client::qdrant::Vectors = vector_f32.into();
            let point = PointStruct::new(point_id, vectors, payload);
            points_to_upsert.push(point);
        }
    }


    Ok(points_to_upsert)
}

// Update function signature to accept collection_name
async fn upsert_batch(client: Arc<Qdrant>, collection_name: &str, points: &[PointStruct]) -> Result<()> {
    if points.is_empty() {
        return Ok(());
    }
    // Use new upsert_points method with builder
    client
        .upsert_points(
            UpsertPointsBuilder::new(collection_name, points.to_vec()) // Use dynamic collection name
            // .wait(true) // Optionally wait for operation to complete
        )
        .await
        .context("Failed to upsert batch to Qdrant")?;
    Ok(())
}
