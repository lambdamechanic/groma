// Common functionality shared between Qdrant and LanceDB implementations

use anyhow::{Result, Context, anyhow};
use git2::{Repository, Oid, Delta, DiffOptions};
use ignore::gitignore::{GitignoreBuilder, Gitignore};
use rig::{
    embeddings::{
        embed::{Embed, EmbedError, TextEmbedder},
        embedding::EmbeddingModel,
        EmbeddingsBuilder,
    },
    providers::openai,
    OneOrMany,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::HashMap,
    fs,
    io::{self, BufReader},
    path::{Path, PathBuf},
    sync::Arc,
};
use tiktoken_rs::{cl100k_base, CoreBPE};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

// --- Constants ---

/// The embedding dimension used by the default OpenAI model (`text-embedding-3-small`).
pub const EMBEDDING_DIMENSION: u64 = 1536;

/// The target size for text chunks in tokens before embedding.
pub const TARGET_CHUNK_SIZE_TOKENS: usize = 8192;

/// Maximum number of tokens to include in a single request to the embedding API.
pub const MAX_TOKENS_PER_EMBEDDING_REQUEST: usize = 200_000;

// --- Data Structures ---

/// Metadata associated with each chunk stored in the vector database.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileMetadata {
    pub path: String,
    pub hash: String,
    pub chunk_index: usize,
}

/// State information for tracking processed files.
#[derive(Debug, Serialize, Deserialize)]
pub struct GromaState {
    pub last_processed_oid: String,
}

/// Represents a document ready for embedding.
pub struct LongDocument<'a> {
    pub path: String,
    pub content: String,
    pub tokenizer: &'a CoreBPE,
}

/// Common configuration for Groma
#[derive(Clone)]
pub struct CommonConfig {
    pub openai_key: String,
    pub openai_model: String,
}

impl CommonConfig {
    /// Create a new configuration from environment variables or defaults
    pub fn from_env() -> Result<Self> {
        let openai_key = std::env::var("OPENAI_API_KEY")
            .context("OPENAI_API_KEY environment variable not set")?;
        
        let openai_model = std::env::var("OPENAI_MODEL")
            .unwrap_or_else(|_| "text-embedding-3-small".to_string());
        
        Ok(Self {
            openai_key,
            openai_model,
        })
    }
    
    /// Create an OpenAI embedding model
    pub fn create_embedding_model(&self) -> Arc<openai::EmbeddingModel> {
        let openai_client = openai::Client::new(&self.openai_key);
        Arc::new(openai_client.embedding_model(&self.openai_model))
    }
}

// --- File Processing ---

/// Process a file and prepare it for embedding.
pub fn process_file_for_embedding<'a>(
    file_path: &Path,
    tokenizer: &'a CoreBPE,
) -> Result<Option<LongDocument<'a>>> {
    let content = fs::read_to_string(file_path)
        .with_context(|| format!("Failed to read file: {}", file_path.display()))?;
    
    if content.is_empty() {
        return Ok(None);
    }
    
    Ok(Some(LongDocument {
        path: file_path.to_string_lossy().to_string(),
        content,
        tokenizer,
    }))
}

/// Calculate SHA256 hash of a string.
pub fn calculate_hash(content: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    hex::encode(hasher.finalize())
}

/// Generate a collection/table name based on folder path.
pub fn generate_collection_name(folder_path: &Path) -> Result<String> {
    let canonical_path = fs::canonicalize(folder_path)
        .with_context(|| format!("Failed to canonicalize path: {}", folder_path.display()))?;
    
    let path_str = canonical_path.to_string_lossy().to_string();
    let namespace = Uuid::NAMESPACE_URL;
    let uuid = Uuid::new_v5(&namespace, path_str.as_bytes());
    let raw_name = format!("groma_{}", uuid.simple());
    
    // Ensure name is valid (alphanumeric, underscore, hyphen)
    let sanitized_name: String = raw_name
        .chars()
        .map(|c| if c.is_alphanumeric() || c == '_' || c == '-' { c } else { '_' })
        .collect();
    
    Ok(sanitized_name)
}

// --- State Management ---

/// Get the path for the .gromastate file.
pub fn get_state_file_path(repo: &Repository) -> Result<PathBuf> {
    let workdir = repo
        .workdir()
        .ok_or_else(|| anyhow!("Cannot get state file path: repository is bare"))?;
    Ok(workdir.join(".gromastate"))
}

/// Load the GromaState from the .gromastate file.
pub fn load_state(repo: &Repository) -> Result<Option<GromaState>> {
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
        serde_json::from_reader(BufReader::new(file)).with_context(|| {
            format!(
                "Failed to deserialize state from: {}",
                state_file_path.display()
            )
        })?;
    info!("Loaded previous state (OID: {})", state.last_processed_oid);
    Ok(Some(state))
}

/// Save the GromaState to the .gromastate file.
pub fn save_state(repo: &Repository, state: &GromaState) -> Result<()> {
    let state_file_path = get_state_file_path(repo)?;
    debug!(
        "Saving current state (OID: {}) to '{}'",
        state.last_processed_oid,
        state_file_path.display()
    );
    
    let file = fs::File::create(&state_file_path)
        .with_context(|| format!("Failed to create state file: {}", state_file_path.display()))?;
    serde_json::to_writer_pretty(file, state).with_context(|| {
        format!(
            "Failed to serialize state to: {}",
            state_file_path.display()
        )
    })?;
    debug!("State saved successfully.");
    Ok(())
}

// --- Git Utilities ---

/// Build an ignore matcher for .gitignore and .gromaignore files.
pub fn build_ignore_matcher(workdir: &Path, target_folder: &Path) -> Result<Gitignore> {
    let mut ignore_builder = GitignoreBuilder::new(workdir);
    
    let gitignore_path = workdir.join(".gitignore");
    if gitignore_path.exists() {
        ignore_builder.add(gitignore_path);
    }
    
    let gromaignore_path = target_folder.join(".gromaignore");
    if gromaignore_path.exists() {
        ignore_builder.add(gromaignore_path);
    }
    
    ignore_builder
        .build()
        .context("Failed to build ignore matcher")
}

// --- Chunking ---

/// Split a document into chunks for embedding.
pub fn chunk_document(doc: &LongDocument, chunk_size: usize) -> Vec<(String, usize)> {
    let tokens = doc.tokenizer.encode_ordinary(&doc.content);
    let mut chunks = Vec::new();
    
    if tokens.is_empty() {
        return chunks;
    }
    
    for (chunk_index, token_chunk) in tokens.chunks(chunk_size).enumerate() {
        if let Ok(text) = doc.tokenizer.decode(token_chunk.to_vec()) {
            if !text.trim().is_empty() {
                chunks.push((text, chunk_index));
            }
        }
    }
    
    chunks
}

/// Batch documents for embedding based on token limits.
pub fn batch_documents_for_embedding(
    documents: Vec<LongDocument>,
    tokenizer: &CoreBPE,
    max_tokens: usize,
    chunk_size: usize,
) -> Vec<Vec<(String, FileMetadata)>> {
    let mut batches = Vec::new();
    let mut current_batch = Vec::new();
    let mut current_token_count = 0;
    
    for doc in documents {
        let chunks = chunk_document(&doc, chunk_size);
        
        for (chunk_text, chunk_index) in chunks {
            let chunk_tokens = tokenizer.encode_ordinary(&chunk_text).len();
            
            // Start new batch if adding this chunk would exceed limit
            if current_token_count + chunk_tokens > max_tokens && !current_batch.is_empty() {
                batches.push(current_batch);
                current_batch = Vec::new();
                current_token_count = 0;
            }
            
            let metadata = FileMetadata {
                path: doc.path.clone(),
                hash: calculate_hash(&chunk_text),
                chunk_index,
            };
            
            current_batch.push((chunk_text, metadata));
            current_token_count += chunk_tokens;
        }
    }
    
    if !current_batch.is_empty() {
        batches.push(current_batch);
    }
    
    batches
}

/// Create a UUID from path and chunk information.
pub fn create_chunk_uuid(path: &str, chunk_index: usize) -> Uuid {
    let combined = format!("{}_{}", path, chunk_index);
    Uuid::new_v5(&Uuid::NAMESPACE_URL, combined.as_bytes())
}
