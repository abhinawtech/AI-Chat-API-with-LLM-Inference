use axum::{extract::State, Json};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use crate::models::chat::ChatModel;

#[derive(Deserialize)]
pub struct ChatRequest {
    pub message: String,
}

#[derive(Serialize)]
pub struct ChatResponse {
    pub reply: String,
}

#[derive(Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub model_loaded: bool,
}

pub async fn health_handler(
    State(model): State<Arc<ChatModel>>,
) -> Result<Json<HealthResponse>, String> {
    println!("❤️ Health check requested");
    let response = HealthResponse {
        status: "healthy".to_string(),
        model_loaded: true, // If we got here, the model is loaded
    };
    println!("✅ Health check completed: status=healthy, model_loaded=true");
    Ok(Json(response))
}

pub async fn chat_handler(
    State(model): State<Arc<ChatModel>>,
    Json(req): Json<ChatRequest>,
) -> Result<Json<ChatResponse>, String> {
    println!("🚀 New chat request received: '{}'", req.message);
    let request_start = std::time::Instant::now();
    
    // Format the input as a chat prompt
    let formatted_prompt = format!("User: {}\nAssistant:", req.message);
    println!("🔵 Formatted prompt: '{}'", formatted_prompt);
    
    // 1) Tokenize input
    let input_ids = model
        .tokenizer
        .encode(formatted_prompt, true)
        .unwrap()
        .get_ids()
        .to_vec();
    println!("🔢 Input tokens: {:?} (length: {})", input_ids, input_ids.len());

    // 2) Run inference (blocking call)
    // Convert input_ids (Vec<u32>) to a 2D Tensor with batch dimension
    let device = model.device.clone();
    let input_tensor = candle_core::Tensor::from_vec(
        input_ids.clone(), 
        (1, input_ids.len()), // Add batch dimension: (batch_size=1, sequence_length)
        &device
    ).unwrap();
    println!("🧠 Created input tensor with shape: {:?}", input_tensor.shape());

    // Create a cache for the model using F16 to match the model weights
    let mut cache = candle_transformers::models::llama::Cache::new(
        true, 
        candle_core::DType::F16, // Change from F32 to F16 to match model
        &model.config, 
        &device
    ).unwrap();
    println!("💾 Initialized model cache");
    
    println!("⚡ Running model inference...");
    let start = std::time::Instant::now();
    let output = model
        .model
        .forward(&input_tensor, 0, &mut cache)
        .unwrap();
    let inference_time = start.elapsed();
    println!("⚡ Model inference completed in {:?}", inference_time);
    println!("📊 Output tensor shape: {:?}", output.shape());

    // Get the last token's logits and convert to token ID
    println!("🎯 Extracting logits from output...");
    let logits = output.squeeze(0).map_err(|e| format!("Failed to squeeze output: {}", e))?;
    println!("📈 Logits shape after squeeze: {:?}", logits.shape());
    
    println!("🔍 Finding token with highest probability...");
    let next_token = logits.argmax(candle_core::D::Minus1).map_err(|e| format!("Failed to get argmax: {}", e))?; // Remove keepdim
    let token_id = next_token.to_scalar::<u32>().map_err(|e| format!("Failed to convert to scalar: {}", e))?;
    println!("🎲 Generated token ID: {} (probability: highest)", token_id);

    // 3) Decode token back to text
    println!("🔤 Decoding token ID {} to text...", token_id);
    let reply_text = model.tokenizer.decode(&[token_id], true).unwrap();
    println!("✅ Generated text: '{}'", reply_text);

    let total_time = request_start.elapsed();
    println!("🏁 Chat request completed in {:?}", total_time);

    Ok(Json(ChatResponse { reply: reply_text }))
}
