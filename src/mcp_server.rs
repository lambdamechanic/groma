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
};
use tokio::io::{stdin, stdout};

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
        
        // Get config from environment
        let config = crate::GromaConfig::from_env()
            .map_err(|e| ToolError::ExecutionError(format!("Failed to get configuration: {}", e)))?;
        
        // Prepare for query
        let folder_path = PathBuf::from(&folder);
        let (embedding_model, qdrant_client, collection_name, canonical_folder_path) = 
            crate::prepare_for_query(&folder_path, cutoff, &config).await
            .map_err(|e| ToolError::ExecutionError(format!("Failed to prepare for query: {}", e)))?;
        
        // Use the core query processing function from main.rs
        let json_output = crate::process_query_core(
            &query,
            qdrant_client,
            embedding_model,
            &collection_name,
            &canonical_folder_path,
            cutoff
        ).await
        .map_err(|e| ToolError::ExecutionError(format!("Query processing failed: {}", e)))?;
        
        // Return the JSON as a string
        serde_json::to_string_pretty(&json_output)
            .map_err(|e| ToolError::ExecutionError(format!("Failed to serialize results: {}", e)))
    }
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