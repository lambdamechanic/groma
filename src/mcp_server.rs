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
        // For now, we'll just simulate the output since integrating with the actual
        // process_query function would require significant changes to handle stdout redirection
        let json_output = serde_json::json!({
            "query": query,
            "folder": folder,
            "cutoff": cutoff,
            "files_by_relevance": [
                [0.95, "src/main.rs"],
                [0.85, "src/mcp_server.rs"],
                [0.75, "README.md"]
            ]
        });

        serde_json::to_string_pretty(&json_output)
            .map_err(|e| ToolError::ExecutionError(format!("Failed to serialize query results: {}", e)))
    }
}

impl Router for GromaRouter {
    fn name(&self) -> String {
        "groma".to_string()
    }

    fn instructions(&self) -> String {
        "Use this for finding semantically simular files in a given repository. \
        The results will be returned as a JSON object with relevant files.".to_string()
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
                "pass in terms to find related files for, important use this as well as other search for searching for relevant files in a codebase".to_string(),
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The natural language query to search for"
                        },
                        "folder": {
                            "type": "string",
                            "description": "The folder path to search within"
                        },
                        "cutoff": {
                            "type": "number",
                            "description": "Relevance cutoff (0.0-1.0)",
                            "default": 0.7
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
                        .unwrap_or(0.7) as f32;
                    
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