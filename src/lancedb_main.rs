use anyhow::Result;
use arrow_array::{ArrayRef, FixedSizeListArray, Float32Array, RecordBatch, RecordBatchIterator, RecordBatchReader, StringArray, Int32Array};
use arrow_schema::{DataType, Field, Schema};
use fastembed::{TextEmbedding, EmbeddingModel, InitOptions};
use lancedb::arrow::IntoArrow;
use lancedb::query::ExecutableQuery;
use lancedb::table::Table;
use serde::Serialize;
use std::fs;
use std::path::Path;
use std::sync::Arc;
use tracing::{info, warn};
use walkdir::WalkDir;
use futures::TryStreamExt;
use clap::Parser;

use groma::{Args, ChunkMetadata, normalize_vector};

// LanceDB-specific vector store implementation
struct LanceDBStore {
    table: Table,
    embedding_dimension: usize,
}

impl LanceDBStore {
    async fn new(db_path: &str, embedding_dimension: usize) -> Result<Self> {
        // Create LanceDB connection
        let db = lancedb::connect(db_path).execute().await?;
        
        // Define schema
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("path", DataType::Utf8, false),
            Field::new("hash", DataType::Utf8, false),
            Field::new("chunk_index", DataType::Int32, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    embedding_dimension as i32,
                ),
                false,
            ),
        ]));

        // Create or open table - try to open first, create if it doesn't exist
        let table = match db.open_table("code_chunks").execute().await {
            Ok(table) => table,
            Err(_) => {
                // Table doesn't exist, create it with RecordBatchIterator
                let empty_batch = RecordBatch::new_empty(schema.clone());
                let batch_iter = RecordBatchIterator::new(
                    vec![Ok(empty_batch)],
                    schema,
                );
                db.create_table("code_chunks", Box::new(batch_iter) as Box<dyn RecordBatchReader + Send>)
                    .execute()
                    .await?
            }
        };

        Ok(Self {
            table,
            embedding_dimension,
        })
    }

    async fn insert_points(
        &self,
        points: Vec<(String, Vec<f32>, ChunkMetadata)>,
    ) -> Result<()> {
        if points.is_empty() {
            return Ok(());
        }

        // Prepare data
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

        // Convert vectors to FixedSizeListArray
        let flat_values: Vec<f32> = vectors.into_iter().flatten().collect();
        let values_array = Float32Array::from(flat_values);
        
        let vector_array = FixedSizeListArray::new(
            Arc::new(Field::new("item", DataType::Float32, true)),
            self.embedding_dimension as i32,
            Arc::new(values_array) as ArrayRef,
            None,
        );

        // Create RecordBatch
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("path", DataType::Utf8, false),
            Field::new("hash", DataType::Utf8, false),
            Field::new("chunk_index", DataType::Int32, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    self.embedding_dimension as i32,
                ),
                false,
            ),
        ]));

        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(ids)) as ArrayRef,
                Arc::new(StringArray::from(paths)) as ArrayRef,
                Arc::new(StringArray::from(hashes)) as ArrayRef,
                Arc::new(Int32Array::from(chunk_indices)) as ArrayRef,
                Arc::new(vector_array) as ArrayRef,
            ],
        )?;

        // Add to table using RecordBatchIterator as a Box
        let batch_iter = RecordBatchIterator::new(
            vec![Ok(batch.clone())],
            batch.schema(),
        );
        self.table.add(Box::new(batch_iter) as Box<dyn RecordBatchReader + Send>).execute().await?;

        Ok(())
    }

    async fn search(
        &self,
        query_vector: Vec<f32>,
        _limit: usize,
    ) -> Result<Vec<SearchResult>> {
        // LanceDB vector search automatically limits to 10 results by default
        // We'll use the default for now
        let query = self.table
            .query()
            .nearest_to(query_vector)?
            .execute()
            .await?;

        let mut results = Vec::new();
        let batch = query.try_collect::<Vec<_>>().await?;
        
        for record_batch in batch {
            let ids = record_batch
                .column_by_name("id")
                .unwrap()
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();
            
            let paths = record_batch
                .column_by_name("path")
                .unwrap()
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();
            
            let distances = record_batch
                .column_by_name("_distance")
                .unwrap()
                .as_any()
                .downcast_ref::<Float32Array>()
                .unwrap();

            for i in 0..record_batch.num_rows() {
                results.push(SearchResult {
                    id: ids.value(i).to_string(),
                    path: paths.value(i).to_string(),
                    distance: distances.value(i),
                });
            }
        }

        Ok(results)
    }
}

#[derive(Debug, Serialize)]
struct SearchResult {
    id: String,
    path: String,
    distance: f32,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let args = Args::parse();

    // Initialize the embedding model
    let mut model = TextEmbedding::try_new(
        InitOptions::new(EmbeddingModel::AllMiniLML6V2)
            .with_show_download_progress(true)
    )?;

    let embedding_dimension = 384; // AllMiniLML6V2 dimension

    // Initialize LanceDB
    let db_path = args.db_path.unwrap_or_else(|| "lancedb_data".to_string());
    let store = LanceDBStore::new(&db_path, embedding_dimension).await?;

    match args.command.as_deref() {
        Some("index") => {
            let target = args.target.unwrap_or_else(|| ".".to_string());
            info!("Indexing directory: {}", target);
            index_directory(&target, &mut model, &store).await?;
        }
        Some("search") => {
            let query = args.query.expect("Query is required for search");
            info!("Searching for: {}", query);
            search_query(&query, &mut model, &store).await?;
        }
        _ => {
            println!("Usage: groma-lancedb <index|search> [options]");
            println!("  index --target <directory>");
            println!("  search --query <query>");
        }
    }

    Ok(())
}

async fn index_directory(
    directory: &str,
    model: &mut TextEmbedding,
    store: &LanceDBStore,
) -> Result<()> {
    let extensions = vec!["rs", "py", "js", "ts", "java", "c", "cpp", "go"];
    let mut file_count = 0;
    let mut chunk_count = 0;

    for entry in WalkDir::new(directory)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
    {
        let path = entry.path();
        let extension = path.extension().and_then(|e| e.to_str());
        
        if let Some(ext) = extension {
            if extensions.contains(&ext) {
                match index_file(path, model, store).await {
                    Ok(chunks) => {
                        file_count += 1;
                        chunk_count += chunks;
                        if file_count % 10 == 0 {
                            info!("Indexed {} files, {} chunks", file_count, chunk_count);
                        }
                    }
                    Err(e) => {
                        warn!("Failed to index {}: {}", path.display(), e);
                    }
                }
            }
        }
    }

    info!("Indexing complete: {} files, {} chunks", file_count, chunk_count);
    Ok(())
}

async fn index_file(
    path: &Path,
    model: &mut TextEmbedding,
    store: &LanceDBStore,
) -> Result<usize> {
    let content = fs::read_to_string(path)?;
    let chunks = chunk_code(&content, 500, 50);
    
    if chunks.is_empty() {
        return Ok(0);
    }

    // Generate embeddings
    let embeddings = model.embed(chunks.clone(), None)?;
    
    // Prepare points for insertion
    let mut points = Vec::new();
    let file_hash = format!("{:x}", md5::compute(&content));
    
    for (i, (_chunk, embedding)) in chunks.iter().zip(embeddings.iter()).enumerate() {
        let id = format!("{}_{}", path.display(), i);
        let normalized = normalize_vector(embedding.to_vec());
        let metadata = ChunkMetadata {
            path: path.display().to_string(),
            hash: file_hash.clone(),
            chunk_index: i,
        };
        points.push((id, normalized, metadata));
    }
    
    // Insert into LanceDB
    store.insert_points(points).await?;
    
    Ok(chunks.len())
}

async fn search_query(
    query: &str,
    model: &mut TextEmbedding,
    store: &LanceDBStore,
) -> Result<()> {
    // Generate query embedding
    let embeddings = model.embed(vec![query], None)?;
    let query_vector = normalize_vector(embeddings[0].to_vec());
    
    // Search
    let results = store.search(query_vector, 5).await?;
    
    // Display results
    println!("\nSearch Results:");
    println!("{}", "=".repeat(80));
    
    for (i, result) in results.iter().enumerate() {
        println!("\n{}. {} (distance: {:.4})", i + 1, result.path, result.distance);
        
        // Load and display the relevant chunk
        if let Ok(content) = fs::read_to_string(&result.path) {
            let chunk_index = result.id.split('_').last()
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(0);
            
            let chunks = chunk_code(&content, 500, 50);
            if chunk_index < chunks.len() {
                println!("---");
                println!("{}", chunks[chunk_index]);
            }
        }
    }
    
    Ok(())
}

fn chunk_code(content: &str, chunk_size: usize, overlap: usize) -> Vec<String> {
    let lines: Vec<&str> = content.lines().collect();
    let mut chunks = Vec::new();
    let mut i = 0;
    
    while i < lines.len() {
        let end = (i + chunk_size).min(lines.len());
        let chunk = lines[i..end].join("\n");
        
        if !chunk.trim().is_empty() {
            chunks.push(chunk);
        }
        
        i += chunk_size - overlap;
        if i + overlap >= lines.len() {
            break;
        }
    }
    
    chunks
}