use anyhow::{anyhow, Context, Result};
use clap::Parser;
use futures::stream::{StreamExt, TryStreamExt};
// Use the new Qdrant client struct and builder patterns
use qdrant_client::{
    payload::Payload, // Import Payload struct directly
    qdrant::{
        point_id::PointIdOptions, vectors_config::Config, CreateCollectionBuilder, Distance,
        GetPointsBuilder, PointStruct, QueryPointsBuilder, SearchPointsBuilder, VectorParams,
        VectorsConfig, PointId, WithPayloadSelector, with_payload_selector, // Keep for SelectorOptions::Include
        UpsertPointsBuilder, // Import UpsertPointsBuilder
        PayloadSelector, // Import PayloadSelector directly
    },
    Qdrant, // Use the new Qdrant struct
};
use rig_core::{
    embeddings::{embedding::EmbeddingModel, Embeddings, EmbeddingsBuilder}, // Add EmbeddingsBuilder
    vector_store::{VectorStoreIndex, Point, PointData},
    providers::openrouter::OpenRouterProvider,
};
use rig_qdrant::QdrantVectorStore; // Re-add QdrantVectorStore import
use serde::{Deserialize, Serialize};
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

    /// OpenRouter API Key
    #[arg(long, env = "OPENROUTER_API_KEY")]
    openrouter_key: String,

    /// OpenRouter Embedding Model name (e.g., "openai/text-embedding-ada-002")
    #[arg(long, default_value = "openai/text-embedding-ada-002")]
    openrouter_model: String,

    /// Qdrant server URL
    #[arg(long, env = "QDRANT_URL", default_value = "http://localhost:6333")]
    qdrant_url: Url,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct FileMetadata {
    path: String,
    hash: String,
}

#[derive(Tabled)]
struct ResultRow {
    #[tabled(rename = "Relevance")]
    score: f32,
    #[tabled(rename = "File Path")]
    path: String,
}

// Helper to generate a stable UUID based on the file path string
fn generate_uuid_from_path(path_str: &str) -> Uuid {
    Uuid::new_v5(&Uuid::NAMESPACE_DNS, path_str.as_bytes())
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

    // OpenRouter Client & Embedding Model
    // Use the correct path from rig_core::providers::openrouter
    let openrouter_provider = OpenRouterProvider::new(args.openrouter_key.clone());
    let embedding_model = Arc::new(
        openrouter_provider
            .embedding_model(&args.openrouter_model)
            .await?,
    );
    info!(
        "Using OpenRouter model: {} (Dimension: {})",
        args.openrouter_model, EMBEDDING_DIMENSION
    );

    // Qdrant Client (use new Qdrant struct and builder)
    let qdrant_client = Arc::new(Qdrant::from_url(&args.qdrant_url.to_string()).build()?);
    info!("Connected to Qdrant at {}", args.qdrant_url);

    // Ensure Qdrant collection exists
    ensure_qdrant_collection(qdrant_client.clone()).await?;

    // --- Process Files ---
    info!("Scanning folder: {}", args.folder.display());
    let files_to_process = scan_folder(&args.folder)?;
    info!("Found {} files to potentially process.", files_to_process.len());

    let mut points_to_upsert: Vec<PointStruct> = Vec::new();
    let mut processed_count = 0;

    for file_path in files_to_process {
        let path_str = file_path.to_string_lossy().to_string();
        let point_uuid = generate_uuid_from_path(&path_str);
        let point_id = uuid_to_point_id(point_uuid);

        match process_file(
            qdrant_client.clone(),
            embedding_model.clone(),
            &file_path,
            &path_str,
            point_id.clone(),
        )
        .await
        {
            Ok(Some(point)) => {
                points_to_upsert.push(point);
                if points_to_upsert.len() >= QDRANT_UPSERT_BATCH_SIZE {
                    upsert_batch(qdrant_client.clone(), &points_to_upsert).await?;
                    processed_count += points_to_upsert.len();
                    info!("Upserted {} points...", processed_count);
                    points_to_upsert.clear();
                }
            }
            Ok(None) => {
                // File is up-to-date, skip
                debug!("Skipping up-to-date file: {}", path_str);
            }
            Err(e) => {
                warn!("Failed to process file {}: {}", path_str, e);
            }
        }
    }

    // Upsert any remaining points
    if !points_to_upsert.is_empty() {
        upsert_batch(qdrant_client.clone(), &points_to_upsert).await?;
        processed_count += points_to_upsert.len();
        info!("Upserted final {} points. Total processed: {}", points_to_upsert.len(), processed_count);
        points_to_upsert.clear();
    }

    info!("File processing complete.");

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
    let query_embedding = embedding_model
        .embed_string(query.to_string())
        .await
        .context("Failed to embed query")?;

    // --- Search Qdrant ---
    info!("Searching for relevant files...");
    // Use the builder pattern for search_points
    let search_result = qdrant_client
        .search_points(
            SearchPointsBuilder::new(QDRANT_COLLECTION_NAME, query_embedding.vector.into(), 100) // collection, vector, limit
                .with_payload(true) // Request payload
                .score_threshold(args.cutoff), // Apply cutoff
        )
        .await
        .context("Failed to search Qdrant")?;

    info!("Found {} potential matches.", search_result.result.len());

    // --- Format and Print Results ---
    let mut results: Vec<ResultRow> = search_result
        .result
        .into_iter()
        .filter_map(|hit| {
            let score = hit.score;
            let payload = hit.payload;
            // Extract path from payload
            payload.get("path").and_then(|v| v.as_str()).map(|path_str| ResultRow {
                score,
                path: path_str.to_string(),
            })
        })
        .collect();

    // Qdrant search results are already sorted by score descending
    // results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

    if results.is_empty() {
        info!("No files found matching the query with cutoff {}", args.cutoff);
    } else {
        println!("{}", Table::new(results));
    }

    Ok(())
}

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

fn scan_folder(folder_path: &Path) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for entry in walkdir::WalkDir::new(folder_path)
        .into_iter()
        .filter_map(Result::ok) // Ignore errors during walk
        .filter(|e| e.file_type().is_file())
    {
        files.push(entry.path().to_path_buf());
    }
    Ok(files)
}

fn calculate_hash(file_path: &Path) -> Result<String> {
    let mut file = fs::File::open(file_path)
        .with_context(|| format!("Failed to open file for hashing: {}", file_path.display()))?;
    let mut hasher = Sha256::new();
    io::copy(&mut file, &mut hasher)
        .with_context(|| format!("Failed to read file for hashing: {}", file_path.display()))?;
    let hash_bytes = hasher.finalize();
    Ok(hex::encode(hash_bytes))
}

// Update function signature to use new Qdrant client type
async fn get_existing_hash(client: Arc<Qdrant>, point_id: PointId) -> Result<Option<String>> {
    // Use new get_points method with builder
    let get_points_req = GetPointsBuilder::new(QDRANT_COLLECTION_NAME, vec![point_id])
        .with_payload(WithPayloadSelector { // Request only the 'hash' field
            selector_options: Some(with_payload_selector::SelectorOptions::Include(
                PayloadSelector {
                    include_points: vec!["hash".to_string()],
                },
            )),
        })
        .with_vectors(false); // Don't need vectors for this check

    let points_response = client.get_points(get_points_req).await?;

    // Process the response which is now Vec<RetrievedPoint>
    if let Some(point) = points_response.result.into_iter().next() {
        // Payload is now Option<Payload>
        if let Some(payload) = point.payload {
            if let Some(hash_value) = payload.inner().get("hash") {
                // Access the inner map
                return Ok(hash_value.as_str().map(String::from));
            }
        }
    }
    Ok(None)
}

// Update function signature to use new Qdrant client type
async fn process_file(
    qdrant_client: Arc<Qdrant>, // Use new Qdrant type
    embedding_model: Arc<dyn EmbeddingModel>,
    file_path: &Path,
    path_str: &str,
    point_id: PointId, // Add missing point_id parameter
) -> Result<Option<PointStruct>> { // Add missing return type
    debug!("Processing: {}", path_str);
    let current_hash = calculate_hash(file_path)?;
    let existing_hash = get_existing_hash(qdrant_client.clone(), point_id.clone()).await?;

    if Some(&current_hash) == existing_hash.as_ref() {
        // Hashes match, file is up-to-date
        return Ok(None);
    }

    info!("Hashing difference detected or file is new. Embedding: {}", path_str);

    // Read file content
    let content = fs::read_to_string(file_path)
        .with_context(|| format!("Failed to read file content: {}", file_path.display()))?;

    if content.trim().is_empty() {
        warn!("Skipping empty file: {}", path_str);
        // Optionally delete from Qdrant if it exists?
        return Ok(None);
    }

    // Generate embedding
    // Handle potential errors during embedding
    let embedding = match embedding_model.embed_string(content).await {
        Ok(emb) => emb,
        Err(e) => {
            error!("Failed to embed file {}: {}", path_str, e);
            // Decide how to handle embedding errors: skip file, retry, etc.
            // For now, we skip the file by returning Ok(None) or an error
            return Err(anyhow!("Embedding failed for {}: {}", path_str, e));
        }
    };


    // Prepare metadata payload
    let metadata = FileMetadata {
        path: path_str.to_string(),
        hash: current_hash,
    };
    // Convert metadata directly to Qdrant Payload type
    let payload: Payload = serde_json::to_value(metadata)?
        .try_into()
        .map_err(|e| anyhow!("Failed to convert metadata to Qdrant Payload: {}", e))?;

    // Create Qdrant point
    let point = PointStruct::new(point_id, embedding.vector.into(), payload);

    Ok(Some(point))
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
