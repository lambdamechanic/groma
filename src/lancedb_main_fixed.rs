// Simplified LanceDB implementation for Groma
use anyhow::{Context, Result};
use arrow_array::{
    types::Float32Type, ArrayRef, FixedSizeListArray, RecordBatch, StringArray,
};
use arrow_schema::{DataType, Field, Schema};
use lancedb::{
    connection::Connection,
    query::{Query, VectorQuery},
};
use rig::{
    embeddings::EmbeddingModel,
    providers::openai::{self, TEXT_EMBEDDING_3_SMALL},
};
use serde::{Deserialize, Serialize};
use std::{collections::HashSet, path::PathBuf, sync::Arc};
use tracing::{debug, info};

// Constants
const EMBEDDING_DIMENSION: usize = 1536;
const TABLE_NAME: &str = "code_chunks";

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FileMetadata {
    path: String,
    hash: String,
    chunk_index: usize,
}

fn get_table_schema(dimension: usize) -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("path", DataType::Utf8, false),
        Field::new("hash", DataType::Utf8, false),
        Field::new("chunk_index", DataType::Int32, false),
        Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                dimension as i32,
            ),
            false,
        ),
    ]))
}

async fn ensure_table(connection: &Connection, table_name: &str) -> Result<()> {
    let table_names = connection.table_names().execute().await?;
    
    if !table_names.contains(&table_name.to_string()) {
        info!("Creating new LanceDB table: {}", table_name);
        
        // Create empty table with schema
        let schema = get_table_schema(EMBEDDING_DIMENSION);
        let empty_batch = RecordBatch::new_empty(schema.clone());
        
        // Create a simple record batch reader
        let batches = vec![empty_batch];
        connection
            .create_table(table_name, batches)
            .execute()
            .await?;
    } else {
        info!("LanceDB table '{}' already exists.", table_name);
    }
    
    Ok(())
}

async fn upsert_batch(
    connection: &Connection,
    table_name: &str,
    points: Vec<(String, Vec<f32>, FileMetadata)>,
) -> Result<()> {
    if points.is_empty() {
        return Ok(());
    }
    
    let table = connection.open_table(table_name).execute().await?;
    
    let mut ids = Vec::new();
    let mut paths = Vec::new();
    let mut hashes = Vec::new();
    let mut chunk_indices = Vec::new();
    let mut vectors = Vec::new();
    
    for (id, vector, metadata) in points {
        ids.push(id);
        paths.push(metadata.path);
        hashes.push(metadata.hash);
        chunk_indices.push(metadata.chunk_index as i32);
        vectors.push(vector);
    }
    
    // Build the FixedSizeListArray properly
    let flat_vectors: Vec<Option<f32>> = vectors
        .into_iter()
        .flat_map(|v| v.into_iter().map(Some))
        .collect();
    
    let vector_array = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
        std::iter::once(Some(flat_vectors)),
        EMBEDDING_DIMENSION as i32,
    );
    
    let schema = get_table_schema(EMBEDDING_DIMENSION);
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(ids)) as ArrayRef,
            Arc::new(StringArray::from(paths)) as ArrayRef,
            Arc::new(StringArray::from(hashes)) as ArrayRef,
            Arc::new(arrow_array::Int32Array::from(chunk_indices)) as ArrayRef,
            Arc::new(vector_array) as ArrayRef,
        ],
    )?;
    
    // Use the simpler add API
    table.add(vec![batch])
        .execute()
        .await
        .context("Failed to upsert batch to LanceDB")?;
    
    Ok(())
}

async fn delete_by_path(connection: &Connection, table_name: &str, path: &str) -> Result<()> {
    let table = connection.open_table(table_name).execute().await?;
    
    // LanceDB delete by filter
    table
        .delete(&format!("path = '{}'", path))
        .await
        .context("Failed to delete entries from LanceDB")?;
    
    Ok(())
}

async fn search_similar(
    connection: &Connection,
    table_name: &str,
    query_vector: Vec<f32>,
    limit: usize,
) -> Result<Vec<(String, f32, FileMetadata)>> {
    let table = connection.open_table(table_name).execute().await?;
    
    let results = table
        .vector_search(query_vector)
        .limit(limit)
        .execute()
        .await?;
    
    let mut search_results = Vec::new();
    
    for batch in results.iter() {
        let ids = batch.column(0).as_any().downcast_ref::<StringArray>().unwrap();
        let paths = batch.column(1).as_any().downcast_ref::<StringArray>().unwrap();
        let hashes = batch.column(2).as_any().downcast_ref::<StringArray>().unwrap();
        let chunk_indices = batch.column(3).as_any().downcast_ref::<arrow_array::Int32Array>().unwrap();
        // Note: scores would need to be extracted from metadata if available
        
        for i in 0..batch.num_rows() {
            let metadata = FileMetadata {
                path: paths.value(i).to_string(),
                hash: hashes.value(i).to_string(),
                chunk_index: chunk_indices.value(i) as usize,
            };
            
            search_results.push((
                ids.value(i).to_string(),
                0.9, // Placeholder score - LanceDB doesn't return scores in the same way
                metadata,
            ));
        }
    }
    
    Ok(search_results)
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt::init();
    
    info!("Starting Groma with LanceDB backend");
    
    // Setup OpenAI client for embeddings
    let openai_api_key = std::env::var("OPENAI_API_KEY")
        .context("OPENAI_API_KEY environment variable not set")?;
    
    let openai_client = openai::Client::new(&openai_api_key);
    let embedding_model = Arc::new(openai_client.embedding_model(TEXT_EMBEDDING_3_SMALL));
    
    // Connect to LanceDB
    let db_path = PathBuf::from(".lancedb");
    let connection = Connection::open(db_path.to_str().unwrap())
        .execute()
        .await
        .context("Failed to connect to LanceDB")?;
    
    // Ensure table exists
    ensure_table(&connection, TABLE_NAME).await?;
    
    // Example: Generate embeddings for some text
    let texts = vec!["Hello world".to_string(), "Test embedding".to_string()];
    let embeddings = embedding_model.embed_texts(texts.clone()).await?;
    
    // Prepare points for insertion
    let mut points = Vec::new();
    for (i, (text, embedding)) in texts.iter().zip(embeddings.iter()).enumerate() {
        let metadata = FileMetadata {
            path: format!("test/file{}.rs", i),
            hash: format!("hash{}", i),
            chunk_index: 0,
        };
        
        let id = format!("id_{}", i);
        let vector: Vec<f32> = embedding.vec.iter().map(|&v| v as f32).collect();
        
        points.push((id, vector, metadata));
    }
    
    // Insert points
    upsert_batch(&connection, TABLE_NAME, points).await?;
    
    info!("Successfully inserted test data into LanceDB");
    
    // Example search
    let query = "Hello";
    let query_embeddings = embedding_model.embed_texts(vec![query.to_string()]).await?;
    let query_vector: Vec<f32> = query_embeddings[0].vec.iter().map(|&v| v as f32).collect();
    
    let results = search_similar(&connection, TABLE_NAME, query_vector, 5).await?;
    
    info!("Search results for '{}': {:?}", query, results);
    
    Ok(())
}
