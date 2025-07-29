// src/routes/chat.rs
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

pub async fn chat_handler(
    State(model): State<Arc<ChatModel>>,
    Json(req):   Json<ChatRequest>,
) -> Json<ChatResponse> {
    // 1) Tokenize input
    let input_ids = model
        .tokenizer
        .encode(req.message.clone(), true)
        .unwrap()
        .get_ids()
        .to_vec();

    // 2) Run inference (blocking call)
    // Replace with the correct inference method for your model
    // Convert input_ids (Vec<u32>) to a Tensor
    let device = model.device.clone(); // use model.device if available
    let input_tensor = candle_core::Tensor::from_vec(input_ids.clone(), (input_ids.len(),), &device).unwrap();

    // Provide the required second argument (e.g., None if optional)
    let output = model
        .model.clone()
        .forward(&input_tensor, 0)
        .unwrap();

    // Assuming output is a vector of token ids
    let output_ids = output.to_vec1::<u32>().unwrap();

    // 3) Decode tokens back to text
    let reply_text = model.tokenizer.decode(&output_ids, true).unwrap();

    Json(ChatResponse { reply: reply_text })
}
