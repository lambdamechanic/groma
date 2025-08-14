// Vector store abstraction layer for supporting multiple backends

use anyhow::Result;
use async_trait::async_trait;
use serde_json::Value;
use std::sync::Arc;
use uuid::Uuid;

/// Point structure for vector storage
#[derive(Debug, Clone)]
pub struct VectorPoint {
    pub id: Uuid,
    pub vector: Vec<f32>,
    pub payload: Value,
}

/// Search result from vector store
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub id: Uuid,
    pub score: f32,
    pub payload: Value,
}

/// Configuration for vector store initialization
#[derive(Debug, Clone)]
pub enum VectorStoreConfig {
    #[cfg(feature = "qdrant")]
    Qdrant {
        url: String,
        /// Dimension of vectors
        dimension: u64,
    },
    #[cfg(feature = "lancedb")]
    LanceDB {
        /// Path to the database directory
        path: String,
        /// Dimension of vectors
        dimension: usize,
    },
}

/// Abstract interface for vector stores
#[async_trait]
pub trait VectorStore: Send + Sync {
    /// Ensure collection/table exists, create if necessary
    async fn ensure_collection(&self, collection_name: &str) -> Result<()>;
    
    /// Check if collection exists
    async fn collection_exists(&self, collection_name: &str) -> Result<bool>;
    
    /// Insert or update points
    async fn upsert_points(&self, collection_name: &str, points: Vec<VectorPoint>) -> Result<()>;
    
    /// Delete points by filter (e.g., by path field)
    async fn delete_by_filter(&self, collection_name: &str, field_name: &str, field_value: &str) -> Result<()>;
    
    /// Search for nearest vectors
    async fn search(
        &self,
        collection_name: &str,
        query_vector: Vec<f32>,
        limit: usize,
        score_threshold: f32,
    ) -> Result<Vec<SearchResult>>;
}

/// Factory function to create vector store based on config
pub async fn create_vector_store(config: VectorStoreConfig) -> Result<Arc<dyn VectorStore>> {
    match config {
        #[cfg(feature = "qdrant")]
        VectorStoreConfig::Qdrant { url, dimension } => {
            let store = crate::vector_store_qdrant::QdrantStore::new(url, dimension).await?;
            Ok(Arc::new(store))
        },
        #[cfg(feature = "lancedb")]
        VectorStoreConfig::LanceDB { path, dimension } => {
            let store = crate::vector_store_lancedb::LanceDBStore::new(path, dimension).await?;
            Ok(Arc::new(store))
        },
        #[allow(unreachable_patterns)]
        _ => {
            use anyhow::anyhow;
            Err(anyhow!("No vector store backend enabled. Build with either --features qdrant or --features lancedb"))
        }
    }
}