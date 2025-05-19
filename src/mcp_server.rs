use anyhow::Result;
use mcp_core::{
    content::Content,
    handler::{PromptError, ResourceError, ToolError},
    protocol::ServerCapabilities,
    resource::Resource,
    tool::{Tool, ToolAnnotations},
};
use mcp_server::{
    router::{CapabilitiesBuilder, RouterService},
    ByteTransport, Router, Server,
};
use serde_json::Value;
use std::{
    future::Future,
    pin::Pin,
    sync::Mutex,
    collections::HashSet,
};
use tokio::io::{stdin, stdout};
use tracing::{info, error};
use once_cell::sync::Lazy;

// Simple set to track which folders are currently being indexed
static INDEXING_FOLDERS: Lazy<Mutex<HashSet<String>>> = Lazy::new(|| Mutex::new(HashSet::new()));

/// A router that wraps our existing Groma functionality to expose it via MCP
#[derive(Clone)]
pub struct GromaRouter {}

impl GromaRouter {
    pub fn new() -> Self {
        Self {}
    }

    /// Process a query and return the results
    async fn process_query(&self, query: String, folder: String, cutoff: f32) -> Result<String, ToolError> {
        use std::path::PathBuf;
        
        // Check if this folder is already being indexed
        let is_indexing = {
            let indexing_folders = INDEXING_FOLDERS.lock().unwrap();
            indexing_folders.contains(&folder)
        };
        
        // If we're already indexing, return a message
        if is_indexing {
            let json_output = serde_json::json!({
                "status": "indexing",
                "message": format!("The codebase '{}' is currently being indexed. Please check back in a few minutes.", folder),
                "files_by_relevance": []
            });
            
            return serde_json::to_string_pretty(&json_output)
                .map_err(|e| ToolError::ExecutionError(format!("Failed to serialize message: {}", e)));
        }
        
        // Get config from environment
        let config = crate::GromaConfig::from_env()
            .map_err(|e| ToolError::ExecutionError(format!("Failed to get configuration: {}", e)))?;
        
        // Prepare for query
        let folder_path = PathBuf::from(&folder);
        let prepare_result = crate::prepare_for_query(&folder_path, cutoff, &config).await;
        
        let (embedding_model, qdrant_client, collection_name, canonical_folder_path) = match prepare_result {
            Ok(result) => result,
            Err(e) => {
                // Return a friendly message if we can't prepare for the query
                let error_str = e.to_string();
                let message = if error_str.contains("canonicalize") {
                    format!("The folder '{}' does not exist or is not accessible.", folder)
                } else {
                    format!("Failed to prepare for query: {}", e)
                };
                
                let json_output = serde_json::json!({
                    "status": "error",
                    "message": message,
                    "files_by_relevance": []
                });
                
                return serde_json::to_string_pretty(&json_output)
                    .map_err(|e| ToolError::ExecutionError(format!("Failed to serialize message: {}", e)));
            }
        };
        
        // Check if the collection exists
        let collections = match qdrant_client.list_collections().await {
            Ok(collections) => collections,
            Err(e) => {
                // Return a friendly message if we can't list collections
                let json_output = serde_json::json!({
                    "status": "error",
                    "message": format!("Failed to connect to Qdrant: {}. Please check your Qdrant configuration.", e),
                    "files_by_relevance": []
                });
                
                return serde_json::to_string_pretty(&json_output)
                    .map_err(|e| ToolError::ExecutionError(format!("Failed to serialize message: {}", e)));
            }
        };
        
        let collection_exists = collections.collections.iter().any(|c| c.name == collection_name);
        
        // If collection doesn't exist, start indexing
        if !collection_exists {
            // Start indexing in the background
            start_background_indexing(&folder).await;
            
            // Return a message that indexing has started
            let json_output = serde_json::json!({
                "status": "indexing_started",
                "message": format!("Started indexing codebase '{}'. Please check back in a few minutes.", folder),
                "files_by_relevance": []
            });
            
            return serde_json::to_string_pretty(&json_output)
                .map_err(|e| ToolError::ExecutionError(format!("Failed to serialize message: {}", e)));
        }
        
        // Try to run the query
        match crate::process_query_core(
            &query,
            qdrant_client,
            embedding_model,
            &collection_name,
            &canonical_folder_path,
            cutoff
        ).await {
            Ok(json_output) => {
                // Return the JSON as a string
                serde_json::to_string_pretty(&json_output)
                    .map_err(|e| ToolError::ExecutionError(format!("Failed to serialize results: {}", e)))
            },
            Err(e) => {
                // If the error indicates the collection doesn't exist, start indexing
                let error_str = e.to_string();
                if error_str.contains("Collection not found") || 
                   error_str.contains("does not exist") || 
                   error_str.contains("Failed to search Qdrant") {
                    
                    // Start indexing in the background
                    start_background_indexing(&folder).await;
                    
                    let json_output = serde_json::json!({
                        "status": "indexing_started",
                        "message": format!("Started indexing codebase '{}'. Please check back in a few minutes.", folder),
                        "files_by_relevance": []
                    });
                    
                    serde_json::to_string_pretty(&json_output)
                        .map_err(|e| ToolError::ExecutionError(format!("Failed to serialize message: {}", e)))
                } else {
                    // For other errors, return a friendly error message
                    let json_output = serde_json::json!({
                        "status": "error",
                        "message": format!("Query processing failed: {}. Please try again later.", e),
                        "files_by_relevance": []
                    });
                    
                    serde_json::to_string_pretty(&json_output)
                        .map_err(|e| ToolError::ExecutionError(format!("Failed to serialize error message: {}", e)))
                }
            }
        }
    }
}

/// Helper function to start indexing a folder in the background
async fn start_background_indexing(folder: &str) {
    use std::path::PathBuf;
    use tokio::task;
    use tokio::process::Command;
    
    info!("Starting indexing for {}...", folder);
    
    // Mark this folder as being indexed
    {
        let mut indexing_folders = INDEXING_FOLDERS.lock().unwrap();
        indexing_folders.insert(folder.to_string());
    }
    
    // Clone what we need for the background task
    let folder_clone = folder.to_string();
    
    // Get the current executable path
    let current_exe = std::env::current_exe().unwrap_or_else(|_| PathBuf::from("groma"));
    
    // Spawn a background process to run the indexing
    task::spawn(async move {
        info!("Running indexing process for {}", folder_clone);
        
        // Run the CLI as a separate process
        let status = Command::new(current_exe)
            .arg(&folder_clone)
            .status()
            .await;
        
        if let Err(e) = &status {
            error!("Failed to start indexing for {}: {}", folder_clone, e);
        } else if let Ok(exit_status) = status {
            if exit_status.success() {
                info!("Indexing completed successfully for {}", folder_clone);
            } else {
                error!("Indexing failed for {}: {:?}", folder_clone, exit_status);
            }
        }
        
        // Remove from indexing folders
        let mut indexing_folders = INDEXING_FOLDERS.lock().unwrap();
        indexing_folders.remove(&folder_clone);
    });
}

impl Router for GromaRouter {
    fn name(&self) -> String {
        "groma".to_string()
    }

    fn instructions(&self) -> String {
        "Use this for finding semantically similar files in a given repository. \
        The results will be returned as a JSON object with relevant files listed.
        It is highly recommended to use this along with rg for searching".to_string()
    }

    fn capabilities(&self) -> ServerCapabilities {
        CapabilitiesBuilder::new()
            .with_tools(false)  // We don't need tool change notifications
            .with_resources(false, false)  // We don't need resource capabilities
            .with_prompts(false)  // We don't need prompt capabilities
            .build()
    }

    fn list_tools(&self) -> Vec<Tool> {
        vec![
            Tool::new(
                "query".to_string(),
                "pass in search terms to find related files that are similar in concept".to_string(),
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "what to search for, terms, concepts, snippets etc"
                        },
                        "folder": {
                            "type": "string",
                            "description": "The path to the repository to search"
                        },
                        "cutoff": {
                            "type": "number",
                            "description": "Relevance cutoff (0.0-1.0)",
                            "default": 0.3
                        }
                    },
                    "required": ["query", "folder"]
                }),
                Some(ToolAnnotations {
                    title: Some("Search Repository".to_string()),
                    read_only_hint: true,
                    destructive_hint: false,
                    idempotent_hint: true,
                    open_world_hint: false,
                }),
            ),
        ]
    }

    fn call_tool(
        &self,
        tool_name: &str,
        arguments: Value,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<Content>, ToolError>> + Send + 'static>> {
        let this = self.clone();
        let tool_name = tool_name.to_string();
        let arguments = arguments.clone();

        Box::pin(async move {
            match tool_name.as_str() {
                "query" => {
                    // Extract arguments
                    let query = arguments
                        .get("query")
                        .and_then(|v| v.as_str())
                        .ok_or_else(|| ToolError::InvalidParameters("Missing 'query' argument".to_string()))?
                        .to_string();
                    
                    let folder = arguments
                        .get("folder")
                        .and_then(|v| v.as_str())
                        .ok_or_else(|| ToolError::InvalidParameters("Missing 'folder' argument".to_string()))?
                        .to_string();
                    
                    let cutoff = arguments
                        .get("cutoff")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.3) as f32;
                    
                    // Process the query and return results directly
                    let result = this.process_query(query, folder, cutoff).await?;
                    
                    // Return the result as text content
                    Ok(vec![Content::text(result)])
                },
                _ => Err(ToolError::NotFound(format!("Tool {} not found", tool_name))),
            }
        })
    }

    // Implement the required resource methods with empty implementations
    fn list_resources(&self) -> Vec<Resource> {
        vec![]
    }

    fn read_resource(
        &self,
        _uri: &str,
    ) -> Pin<Box<dyn Future<Output = Result<String, ResourceError>> + Send + 'static>> {
        Box::pin(async {
            Err(ResourceError::NotFound("Resources not supported".to_string()))
        })
    }

    // Implement required prompt methods with empty implementations
    fn list_prompts(&self) -> Vec<mcp_core::prompt::Prompt> {
        vec![]
    }

    fn get_prompt(
        &self,
        _prompt_name: &str,
    ) -> Pin<Box<dyn Future<Output = Result<String, PromptError>> + Send + 'static>> {
        Box::pin(async {
            Err(PromptError::NotFound("Prompts not supported".to_string()))
        })
    }
}

/// Run the MCP server with our Groma router
pub async fn run_mcp_server() -> Result<()> {
    tracing::info!("Starting Groma MCP server");

    // Create an instance of our router
    let router = RouterService(GromaRouter::new());

    // Create and run the server
    let server = Server::new(router);
    let transport = ByteTransport::new(stdin(), stdout());

    tracing::info!("MCP server initialized and ready to handle requests");
    server.run(transport).await?;

    Ok(())
}