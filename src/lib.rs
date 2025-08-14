pub mod common;

// Re-export commonly used types for binary crates
pub use clap::Parser;

// Re-export ChunkMetadata from common
pub use common::FileMetadata as ChunkMetadata;

// Helper function to normalize vectors
pub fn normalize_vector(vector: Vec<f32>) -> Vec<f32> {
    let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        vector.iter().map(|x| x / norm).collect()
    } else {
        vector
    }
}
