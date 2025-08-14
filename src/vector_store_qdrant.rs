// Qdrant implementation of VectorStore trait

use crate::vector_store::{VectorStore, VectorPoint, SearchResult};
use anyhow::{Result, Context};
use async_trait::async_trait;
use qdrant_client::{
    qdrant::{
        point_id::PointIdOptions,
        Condition,
        CreateCollectionBuilder,
        DeletePointsBuilder,
        Distance,
        Filter,
        PointId,
        PointStruct,
        SearchPointsBuilder,
        UpsertPointsBuilder,
        VectorParams,
    },
    Payload, Qdrant,
};
use serde_json::Value;
use std::sync::Arc;
use url::Url;
use uuid::Uuid;

pub struct QdrantStore {
    client: Arc<Qdrant>,
    dimension: u64,
}

impl QdrantStore {
    pub async fn new(url: String, dimension: u64) -> Result<Self> {
        let parsed_url = Url::parse(&url)
            .with_context(|| format!("Failed to parse Qdrant URL: {}", url))?;
        
        let client = Qdrant::from_url(&parsed_url.to_string())
            .build()
            .with_context(|| format!("Failed to connect to Qdrant at {}", url))?;
        
        Ok(Self {
            client: Arc::new(client),
            dimension,
        })
    }
    
    /// Convert UUID to Qdrant PointId
    fn uuid_to_point_id(uuid: &Uuid) -> PointId {
        PointId {
            point_id_options: Some(PointIdOptions::Uuid(uuid.to_string())),
        }
    }
}

#[async_trait]
impl VectorStore for QdrantStore {
    async fn ensure_collection(&self, collection_name: &str) -> Result<()> {
        let collections = self.client.list_collections().await?;
        let exists = collections
            .collections
            .iter()
            .any(|c| c.name == collection_name);
        
        if !exists {
            self.client
                .create_collection(
                    CreateCollectionBuilder::new(collection_name)
                        .vectors_config(
                            VectorParams {
                                size: self.dimension,
                                distance: Distance::Cosine.into(),
                                ..Default::default()
                            }
                        ),
                )
                .await?;
        }
        Ok(())
    }
    
    async fn collection_exists(&self, collection_name: &str) -> Result<bool> {
        let collections = self.client.list_collections().await?;
        Ok(collections
            .collections
            .iter()
            .any(|c| c.name == collection_name))
    }
    
    async fn upsert_points(&self, collection_name: &str, points: Vec<VectorPoint>) -> Result<()> {
        let qdrant_points: Vec<PointStruct> = points
            .into_iter()
            .map(|point| {
                let payload: Payload = match serde_json::from_value(point.payload) {
                    Ok(p) => p,
                    Err(_) => Payload::new(),
                };
                
                PointStruct {
                    id: Some(Self::uuid_to_point_id(&point.id)),
                    vectors: Some(point.vector.into()),
                    payload: payload.into(),
                }
            })
            .collect();
        
        self.client
            .upsert_points(UpsertPointsBuilder::new(collection_name, qdrant_points))
            .await
            .context("Failed to upsert points to Qdrant")?;
        
        Ok(())
    }
    
    async fn delete_by_filter(&self, collection_name: &str, field_name: &str, field_value: &str) -> Result<()> {
        let filter = Filter {
            must: vec![Condition::matches(field_name, field_value.to_string())],
            ..Default::default()
        };
        
        let delete_result = self.client
            .delete_points(
                DeletePointsBuilder::new(collection_name)
                    .points(filter)
            )
            .await;
        
        match delete_result {
            Ok(_) => Ok(()),
            Err(e) => {
                // Log the error but don't fail - the file might not have been indexed yet
                tracing::warn!("Failed to delete points for {}: {}", field_value, e);
                Ok(())
            }
        }
    }
    
    async fn search(
        &self,
        collection_name: &str,
        query_vector: Vec<f32>,
        limit: usize,
        score_threshold: f32,
    ) -> Result<Vec<SearchResult>> {
        let search_result = self.client
            .search_points(
                SearchPointsBuilder::new(collection_name, query_vector, limit as u64)
                    .with_payload(true)
                    .score_threshold(score_threshold),
            )
            .await
            .context("Failed to search Qdrant")?;
        
        let results = search_result
            .result
            .into_iter()
            .map(|hit| {
                let payload = serde_json::to_value(hit.payload).unwrap_or(Value::Null);
                let id = if let Some(PointIdOptions::Uuid(uuid_str)) = hit.id.and_then(|pid| pid.point_id_options) {
                    Uuid::parse_str(&uuid_str).unwrap_or_default()
                } else {
                    Uuid::default()
                };
                
                SearchResult {
                    id,
                    score: hit.score,
                    payload,
                }
            })
            .collect();
        
        Ok(results)
    }
}
