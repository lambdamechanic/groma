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
use tabled::{Table, Tabled};
use tracing::{debug, error, info, warn, Level};
use tracing_subscriber::FmtSubscriber;
use url::Url;
// Import TextSplitter
use text_splitter::TextSplitter; // Removed note about ChunkConfig
// Import the specific tokenizer type needed for TextSplitter signature
use tiktoken_rs::{cl100k_base, CoreBPE};
use uuid::Uuid;

const QDRANT_COLLECTION_NAME: &str = "groma-files";
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

#[derive(Tabled)]
struct ResultRow {
    #[tabled(rename = "Relevance")]
    score: f32,
    #[tabled(rename = "File Path")]
    path: String,
}

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
    // Initialize tracing subscriber for logging
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO) // Adjust level as needed (e.g., DEBUG for more details)
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

    // Ensure Qdrant collection exists
    ensure_qdrant_collection(qdrant_client.clone()).await?;

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
            &file_path,
            &path_str,
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
                            "Upserting batch of {} points...",
                            all_points_to_upsert.len()
                        );
                        upsert_batch(qdrant_client.clone(), &all_points_to_upsert).await?;
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
            "Upserting final batch of {} points...",
            all_points_to_upsert.len()
        );
        upsert_batch(qdrant_client.clone(), &all_points_to_upsert).await?;
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
            SearchPointsBuilder::new(QDRANT_COLLECTION_NAME, query_vector_f32, 100) // collection, vector (now f32), limit
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

    // --- Format and Print Aggregated Results ---
    let mut aggregated_results: Vec<ResultRow> = file_scores
        .into_iter()
        .map(|(path, score)| ResultRow { score, path })
        .collect();

    // Sort by score descending
    aggregated_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));


    if aggregated_results.is_empty() {
        info!("No files found matching the query with cutoff {}", args.cutoff);
    } else {
        info!("Aggregated results by file (showing highest score per file):");
        println!("{}", Table::new(aggregated_results));
    }

    Ok(())
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


// Update function signature to use new Qdrant client type
async fn ensure_qdrant_collection(client: Arc<Qdrant>) -> Result<()> {
    // Use new list_collections method
    let collections_list = client.list_collections().await?;
    if !collections_list
        .collections
        .iter()
        .any(|c| c.name == QDRANT_COLLECTION_NAME)
    {
        info!("Collection '{}' not found. Creating...", QDRANT_COLLECTION_NAME);
        // Use new create_collection method with builder
        client
            .create_collection(
                CreateCollectionBuilder::new(QDRANT_COLLECTION_NAME)
                    .vectors_config(VectorsConfig::from(VectorParams {
                        size: EMBEDDING_DIMENSION,
                        distance: Distance::Cosine.into(),
                        ..Default::default()
                    }))
                    // Add payload indexing for faster filtering/lookup if needed, especially on 'path' or 'hash'
                    // .payload_schema(...) // Consider adding schema for path/hash indexing
            )
            .await?;
        info!("Collection '{}' created.", QDRANT_COLLECTION_NAME);
        // It might be wise to explicitly create payload indices here too
        // client.create_payload_index(QDRANT_COLLECTION_NAME, "path", FieldType::Keyword, ...).await?;
        // client.create_payload_index(QDRANT_COLLECTION_NAME, "hash", FieldType::Keyword, ...).await?;

    } else {
        info!("Collection '{}' already exists.", QDRANT_COLLECTION_NAME);
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
    Condition, DeletePointsBuilder, Filter, // Removed unused PointsSelector and Match
}; // Imports for filtering/deleting

// Fetches the hash of *one* existing chunk for a given file path.
async fn get_existing_file_hash(client: Arc<Qdrant>, path_str: &str) -> Result<Option<String>> {
    // Use MatchValue::Keyword directly in Condition::matches (Corrected case)
    let filter = Filter::must([Condition::matches(
        "path", // Field name in payload
        MatchValue::Keyword(path_str.to_string()), // Use MatchValue::Keyword (capital K)
    )]);

    // Use search instead of get, limit to 1, only fetch hash payload
    let search_req = SearchPointsBuilder::new(QDRANT_COLLECTION_NAME, vec![0.0; EMBEDDING_DIMENSION as usize], 1) // Dummy vector, limit 1
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
async fn delete_points_by_path(client: Arc<Qdrant>, path_str: &str) -> Result<()> {
    info!("Deleting existing chunks for file: {}", path_str);
    // Use MatchValue::Keyword directly in Condition::matches (Corrected case)
    let filter = Filter::must([Condition::matches(
        "path",
        MatchValue::Keyword(path_str.to_string()), // Use MatchValue::Keyword (capital K)
    )]);

    // Manually construct DeletePoints to bypass potential builder method issue
    let delete_request = qdrant_client::qdrant::DeletePoints {
        collection_name: QDRANT_COLLECTION_NAME.to_string(),
        wait: None, // Or Some(true) if needed
        ordering: None,
        points_selector: Some(qdrant_client::qdrant::PointsSelector {
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
) -> Result<Vec<PointStruct>> // Return Vec instead of Option
where
    E: EmbeddingModel + Send + Sync + 'static,
{
    debug!("Processing file: {}", path_str);
    let current_hash = calculate_hash(file_path)?;
    let existing_hash = get_existing_file_hash(qdrant_client.clone(), path_str).await?;

    if Some(&current_hash) == existing_hash.as_ref() {
        debug!("File hash matches, skipping: {}", path_str);
        return Ok(Vec::new()); // Return empty vec, no points to upsert
    }

    info!(
        "File is new or hash changed. Processing and embedding chunks for: {}",
        path_str
    );

    // If hash differs or file is new, delete existing points for this path
    if existing_hash.is_some() {
        delete_points_by_path(qdrant_client.clone(), path_str).await?;
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

// Update function signature to use new Qdrant client type
async fn upsert_batch(client: Arc<Qdrant>, points: &[PointStruct]) -> Result<()> {
    if points.is_empty() {
        return Ok(());
    }
    // Use new upsert_points method with builder
    client
        .upsert_points(
            UpsertPointsBuilder::new(QDRANT_COLLECTION_NAME, points.to_vec())
            // .wait(true) // Optionally wait for operation to complete
        )
        .await
        .context("Failed to upsert batch to Qdrant")?;
    Ok(())
}
