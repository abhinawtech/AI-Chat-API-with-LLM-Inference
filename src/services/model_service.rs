use std::sync::Arc;
use tokio::sync::{mpsc, oneshot, Semaphore};
use dashmap::DashMap;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use uuid::Uuid;
use crate::models::chat::ChatModel;
use candle_core::{Tensor, D};
use candle_nn;

pub struct InferenceRequest {
    pub id: String,
    pub prompt: String,
    pub max_tokens: usize,
    pub response_tx: oneshot::Sender<Result<InferenceResponse, String>>,
}

#[derive(Clone, Debug)]
pub struct InferenceResponse {
    pub text: String,
    pub tokens: usize,
    pub generation_time_ms: u128,
    pub cached: bool,
}

#[derive(Clone)]
struct CachedResponse {
    pub response: String,
    pub tokens: usize,
    pub timestamp: u64,
}

pub struct ModelService {
    request_queue: mpsc::Sender<InferenceRequest>,
    cache: Arc<DashMap<String, CachedResponse>>,
    rate_limiter: Arc<Semaphore>,
}

impl ModelService {
    pub async fn new(model: Arc<ChatModel>) -> Result<Self, Box<dyn std::error::Error>> {
        let (tx, rx) = mpsc::channel(1000);
        let cache = Arc::new(DashMap::new());
        let rate_limiter = Arc::new(Semaphore::new(5)); // 5 concurrent requests max
        
        // Start background inference worker
        let worker_cache = cache.clone();
        tokio::spawn(inference_worker(rx, model, worker_cache));
        
        Ok(Self {
            request_queue: tx,
            cache,
            rate_limiter,
        })
    }
    
    pub async fn generate(
        &self, 
        prompt: String, 
        max_tokens: usize
    ) -> Result<InferenceResponse, String> {
        println!("🟡 ModelService::generate called with {} tokens", max_tokens);
        
        // Check cache first
        let cache_key = format!("{}:{}", prompt, max_tokens);
        if let Some(cached) = self.cache.get(&cache_key) {
            if is_cache_valid(&cached) {
                println!("🟡 Cache hit for prompt");
                return Ok(InferenceResponse {
                    text: cached.response.clone(),
                    tokens: cached.tokens,
                    generation_time_ms: 0,
                    cached: true,
                });
            }
        }
        
        println!("🟡 Acquiring rate limit permit...");
        // Rate limiting
        let _permit = self.rate_limiter.acquire().await
            .map_err(|_| "Rate limited - too many concurrent requests")?;
        
        println!("🟡 Queueing inference request...");
        // Queue request
        let (response_tx, response_rx) = oneshot::channel();
        let request = InferenceRequest {
            id: Uuid::new_v4().to_string(),
            prompt: prompt.clone(),
            max_tokens,
            response_tx,
        };
        
        self.request_queue.send(request).await
            .map_err(|_| "Request queue full")?;
        
        println!("🟡 Waiting for inference response...");
        // Wait for response with timeout
        let response = tokio::time::timeout(
            Duration::from_secs(20), // Shorter timeout
            response_rx
        ).await
            .map_err(|_| "Inference timeout")?
            .map_err(|_| "Request cancelled")??;
        
        println!("🟡 Inference completed: {} tokens", response.tokens);
        
        // Cache successful responses
        if !response.text.is_empty() && response.tokens > 0 {
            self.cache.insert(cache_key, CachedResponse {
                response: response.text.clone(),
                tokens: response.tokens,
                timestamp: current_timestamp(),
            });
        }
        
        Ok(response)
    }
    
    pub fn get_stats(&self) -> ServiceStats {
        ServiceStats {
            cache_size: self.cache.len(),
            available_permits: self.rate_limiter.available_permits(),
        }
    }
}

#[derive(serde::Serialize)]
pub struct ServiceStats {
    pub cache_size: usize,
    pub available_permits: usize,
}

// Background worker for processing inference requests
async fn inference_worker(
    mut rx: mpsc::Receiver<InferenceRequest>,
    model: Arc<ChatModel>,
    cache: Arc<DashMap<String, CachedResponse>>,
) {
    println!("🔥 Inference worker started");
    
    while let Some(request) = rx.recv().await {
        let start_time = Instant::now();
        
        // Process single request (can be optimized to batch later)
        let result = process_inference_request(&model, &request).await;
        
        // Send response back
        let _ = request.response_tx.send(result);
        
        // Clean old cache entries periodically
        if rand::random::<f32>() < 0.1 { // 10% chance
            cleanup_cache(&cache);
        }
    }
}

async fn process_inference_request(
    model: &ChatModel,
    request: &InferenceRequest,
) -> Result<InferenceResponse, String> {
    let generation_start = Instant::now();
    
    // Move to blocking task
    let model_clone = Arc::new(ChatModel {
        model: unsafe { std::ptr::read(&model.model as *const _) },
        tokenizer: model.tokenizer.clone(),
        device: model.device.clone(),
        config: model.config.clone(),
    });
    
    let prompt = request.prompt.clone();
    let max_tokens = request.max_tokens.min(50); // Cap at 50 for performance
    
    let result = tokio::task::spawn_blocking(move || {
        // Tokenize
        let input_ids = model_clone
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| format!("Tokenization failed: {}", e))?
            .get_ids()
            .to_vec();

        if input_ids.is_empty() {
            return Err("Empty tokenization result".to_string());
        }

        // Create input tensor
        let device = &model_clone.device;
        let mut input_tensor = Tensor::from_vec(
            input_ids.clone(), 
            (1, input_ids.len()),
            device
        ).map_err(|e| format!("Tensor creation failed: {}", e))?;

        // Create cache
        let mut cache = candle_transformers::models::llama::Cache::new(
            true, 
            candle_core::DType::F16,
            &model_clone.config, 
            device
        ).map_err(|e| format!("Cache creation failed: {}", e))?;
        
        // Fast generation
        let mut generated_tokens = Vec::new();
        
        for step in 0..max_tokens {
            let output = model_clone
                .model
                .forward(&input_tensor, step, &mut cache)
                .map_err(|e| format!("Forward failed: {}", e))?;

            let logits = output.get(0).map_err(|e| format!("Logits failed: {}", e))?;

            let token_id = logits.argmax(D::Minus1)
                .map_err(|e| format!("Argmax failed: {}", e))?
                .to_scalar::<u32>()
                .map_err(|e| format!("Scalar failed: {}", e))?;

            // EOS check
            if token_id == 2 || token_id == 1 {
                break;
            }

            // Stop on period after some tokens
            if token_id == 29889 && generated_tokens.len() > 5 {
                generated_tokens.push(token_id);
                break;
            }

            generated_tokens.push(token_id);

            input_tensor = Tensor::from_vec(vec![token_id], (1, 1), device)
                .map_err(|e| format!("Next input failed: {}", e))?;
        }

        // Decode
        let reply_text = if !generated_tokens.is_empty() {
            model_clone.tokenizer.decode(&generated_tokens, true)
                .map_err(|e| format!("Decode failed: {}", e))?
        } else {
            "No response generated".to_string()
        };

        Ok((reply_text, generated_tokens.len()))
    }).await;

    let generation_time = generation_start.elapsed().as_millis();

    match result {
        Ok(Ok((text, tokens))) => {
            Ok(InferenceResponse {
                text: text.trim().to_string(),
                tokens,
                generation_time_ms: generation_time,
                cached: false,
            })
        }
        Ok(Err(e)) => Err(e),
        Err(e) => Err(format!("Task failed: {}", e)),
    }
}

fn is_cache_valid(cached: &CachedResponse) -> bool {
    let now = current_timestamp();
    let age = now.saturating_sub(cached.timestamp);
    age < 300 // 5 minutes cache
}

fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

fn cleanup_cache(cache: &DashMap<String, CachedResponse>) {
    let now = current_timestamp();
    cache.retain(|_, v| is_cache_valid(v));
}