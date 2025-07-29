// src/routes/chat.rs
use axum::{extract::State, Json, http::StatusCode};
use serde::{Deserialize, Serialize};
use std::{sync::Arc, time::Duration};
use crate::services::model_service::ModelService;

#[derive(Deserialize)]
pub struct ChatRequest {
    pub message: String,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
}

fn default_max_tokens() -> usize { 20 }

#[derive(Serialize)]
pub struct ChatResponse {
    pub reply: String,
    pub tokens_generated: usize,
    pub generation_time_ms: u128,
    pub tokens_per_second: f64,
    pub cached: bool,
}

pub async fn chat_handler(
    State(model_service): State<Arc<ModelService>>,
    Json(req): Json<ChatRequest>,
) -> Result<Json<ChatResponse>, (StatusCode, String)> {
    println!("🔵 Chat handler started with message: '{}'", req.message);
    
    // Validate input
    if req.message.trim().is_empty() {
        println!("❌ Empty message rejected");
        return Err((StatusCode::BAD_REQUEST, "Message cannot be empty".to_string()));
    }
    
    if req.max_tokens > 15 {
        println!("❌ Too many tokens requested: {}", req.max_tokens);
        return Err((StatusCode::BAD_REQUEST, "Max tokens cannot exceed 15".to_string()));
    }
    
    // Format prompt for TinyLlama
    let prompt = format!("Human: {}\nAssistant:", req.message);
    println!("🔵 Using prompt: '{}'", prompt);
    
    // Generate response with timeout
    println!("🔵 Starting generation...");
    let start_time = std::time::Instant::now();
    
    let response = tokio::time::timeout(
        Duration::from_secs(25), // 25 second timeout
        model_service.generate(prompt, req.max_tokens)
    ).await;
    
    let response = match response {
        Ok(Ok(resp)) => {
            println!("✅ Generation completed: '{}'", resp.text);
            resp
        },
        Ok(Err(e)) => {
            println!("❌ Generation failed: {}", e);
            return Err((StatusCode::INTERNAL_SERVER_ERROR, format!("Generation failed: {}", e)));
        },
        Err(_) => {
            println!("❌ Generation timed out after 25 seconds");
            return Err((StatusCode::REQUEST_TIMEOUT, "Request timed out".to_string()));
        }
    };
    
    let total_time = start_time.elapsed().as_millis();
    println!("🔵 Total request time: {}ms", total_time);
    
    let tokens_per_second = if response.generation_time_ms > 0 {
        response.tokens as f64 / (response.generation_time_ms as f64 / 1000.0)
    } else {
        0.0
    };
    
    Ok(Json(ChatResponse {
        reply: response.text,
        tokens_generated: response.tokens,
        generation_time_ms: response.generation_time_ms,
        tokens_per_second,
        cached: response.cached,
    }))
}

pub async fn health_handler(
    State(model_service): State<Arc<ModelService>>,
) -> Json<serde_json::Value> {
    let stats = model_service.get_stats();
    Json(serde_json::json!({
        "status": "healthy",
        "cache_size": stats.cache_size,
        "available_permits": stats.available_permits,
        "timestamp": chrono::Utc::now().to_rfc3339()
    }))
}
