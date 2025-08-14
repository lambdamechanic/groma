// LanceDB implementation of VectorStore trait

use crate::vector_store::{VectorStore, VectorPoint, SearchResult};
use anyhow::{Result, Context, anyhow};
use async_trait::async_trait;
use arrow_array::{
    FixedSizeListArray, RecordBatch, StringArray,
    types::Float32Type,
};
use arrow_schema::{DataType, Field, Schema};
use lancedb::{
    connect, 
    Connection,
    query::{ExecutableQuery, QueryBase},
    Table,
};
use futures::TryStreamExt;
use serde_json::Value;
use std::sync::Arc;
use uuid::Uuid;

pub struct LanceDBStore {
    connection: Arc<Connection>,
    dimension: usize,
}

impl LanceDBStore {
    pub async fn new(path: String, dimension: usize) -> Result<Self> {
        // Create directory if it doesn't exist
        std::fs::create_dir_all(&path)
            .with_context(|| format!("Failed to create LanceDB directory: {}", path))?;
        
        let connection = connect(&path)
            .execute()
            .await
            .with_context(|| format!("Failed to connect to LanceDB at {}", path))?;
        
        Ok(Self {
            connection: Arc::new(connection),
            dimension,
        })
    }
    
    /// Get schema for the table
    fn get_schema(&self) -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("path", DataType::Utf8, false),
            Field::new("hash", DataType::Utf8, true),
            Field::new("chunk_index", DataType::Utf8, true),
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    self.dimension as i32,
                ),
                false,
            ),
        ]))
    }
    
    /// Convert VectorPoint to RecordBatch
    fn points_to_batch(&self, points: Vec<VectorPoint>) -> Result<RecordBatch> {
        let schema = self.get_schema();
        
        let mut ids = Vec::new();
        let mut paths = Vec::new();
        let mut hashes = Vec::new();
        let mut chunk_indices = Vec::new();
        let mut vectors = Vec::new();
        
        for point in points {
            ids.push(point.id.to_string());
            
            // Extract fields from payload
            let path = point.payload.get("path")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            paths.push(path);
            
            let hash = point.payload.get("hash")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            hashes.push(Some(hash));
            
            let chunk_index = point.payload.get("chunk_index")
                .map(|v| v.to_string())
                .unwrap_or_else(|| "0".to_string());
            chunk_indices.push(Some(chunk_index));
            
            vectors.push(Some(point.vector));
        }
        
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(ids)),
                Arc::new(StringArray::from(paths)),
                Arc::new(StringArray::from(hashes)),
                Arc::new(StringArray::from(chunk_indices)),
                Arc::new(FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
                    vectors.into_iter(),
                    self.dimension as i32,
                )),
            ],
        )?;
        
        Ok(batch)
    }
}

#[async_trait]
impl VectorStore for LanceDBStore {
    async fn ensure_collection(&self, collection_name: &str) -> Result<()> {
        let table_names = self.connection
            .table_names()
            .execute()
            .await?;
        
        if !table_names.contains(&collection_name.to_string()) {
            // Create empty table with schema
            let schema = self.get_schema();
            let empty_batch = RecordBatch::new_empty(schema);
            
            self.connection
                .create_table(collection_name, Box::new(std::iter::once(Ok(empty_batch))))
                .execute()
                .await?;
        }
        
        Ok(())
    }
    
    async fn collection_exists(&self, collection_name: &str) -> Result<bool> {
        let table_names = self.connection
            .table_names()
            .execute()
            .await?;
        
        Ok(table_names.contains(&collection_name.to_string()))
    }
    
    async fn upsert_points(&self, collection_name: &str, points: Vec<VectorPoint>) -> Result<()> {
        if points.is_empty() {
            return Ok(());
        }
        
        let table = self.connection
            .open_table(collection_name)
            .execute()
            .await?;
        
        // Convert points to RecordBatch
        let batch = self.points_to_batch(points)?;
        
        // In LanceDB, we need to handle upsert manually
        // For now, we'll just append (could implement deduplication logic later)
        table.add(Box::new(std::iter::once(Ok(batch))))
            .execute()
            .await?;
        
        Ok(())
    }
    
    async fn delete_by_filter(&self, collection_name: &str, field_name: &str, field_value: &str) -> Result<()> {
        let table = match self.connection
            .open_table(collection_name)
            .execute()
            .await
        {
            Ok(t) => t,
            Err(_) => {
                // Table doesn't exist, nothing to delete
                return Ok(());
            }
        };
        
        // Build filter expression
        let filter = format!("{} = '{}'", field_name, field_value);
        
        match table.delete(&filter).await {
            Ok(_) => Ok(()),
            Err(e) => {
                // Log the error but don't fail
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
        let table = self.connection
            .open_table(collection_name)
            .execute()
            .await?;
        
        // Perform vector search
        let results = table
            .query()
            .limit(limit)
            .nearest_to(query_vector)?
            .execute()
            .await?
            .try_collect::<Vec<_>>()
            .await?;
        
        let mut search_results = Vec::new();
        
        for batch in results {
            // Extract id column
            let id_array = batch.column_by_name("id")
                .ok_or_else(|| anyhow!("Missing id column"))?
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| anyhow!("Invalid id column type"))?;
            
            // Extract path column
            let path_array = batch.column_by_name("path")
                .ok_or_else(|| anyhow!("Missing path column"))?
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| anyhow!("Invalid path column type"))?;
            
            // Extract other columns for payload
            let hash_array = batch.column_by_name("hash")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>());
            
            let chunk_index_array = batch.column_by_name("chunk_index")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>());
            
            // LanceDB doesn't directly provide scores in the same way as Qdrant
            // We'll need to calculate cosine similarity manually or use a placeholder
            // For now, using a decreasing score based on result order
            let base_score = 1.0 - (search_results.len() as f32 * 0.01);
            
            for i in 0..batch.num_rows() {
                let score = base_score - (i as f32 * 0.001);
                
                // Skip results below threshold
                if score < score_threshold {
                    continue;
                }
                
                let id_str = id_array.value(i);
                let id = Uuid::parse_str(id_str).unwrap_or_default();
                
                let mut payload = serde_json::json!({
                    "path": path_array.value(i),
                });
                
                if let Some(hash_arr) = hash_array {
                    if !hash_arr.is_null(i) {
                        payload["hash"] = Value::String(hash_arr.value(i).to_string());
                    }
                }
                
                if let Some(chunk_arr) = chunk_index_array {
                    if !chunk_arr.is_null(i) {
                        payload["chunk_index"] = Value::String(chunk_arr.value(i).to_string());
                    }
                }
                
                search_results.push(SearchResult {
                    id,
                    score,
                    payload,
                });
            }
        }
        
        Ok(search_results)
    }
}
