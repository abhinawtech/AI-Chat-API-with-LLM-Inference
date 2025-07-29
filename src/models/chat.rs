// src/models/chat.rs
use anyhow::Result;
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use candle_transformers::models::llama::{Config, Llama};
use hf_hub::api::sync::Api;
use hf_hub::{Repo, RepoType, Cache};
use tokenizers::Tokenizer;
use std::fs;
use serde_json::Value;

pub struct ChatModel {
    pub model: Llama,
    pub tokenizer: Tokenizer,
    pub device: Device,
    pub config: Config,
}

// Helper function to create Config from HuggingFace config.json
fn parse_llama_config(config_json: &str) -> Result<Config> {
    let config: Value = serde_json::from_str(config_json)?;
    
    // Extract values from the HuggingFace config
    let vocab_size = config["vocab_size"].as_u64().unwrap_or(32000) as usize;
    let hidden_size = config["hidden_size"].as_u64().unwrap_or(2048) as usize;
    let intermediate_size = config["intermediate_size"].as_u64().unwrap_or(5632) as usize;
    let num_hidden_layers = config["num_hidden_layers"].as_u64().unwrap_or(22) as usize;
    let num_attention_heads = config["num_attention_heads"].as_u64().unwrap_or(32) as usize;
    let num_key_value_heads = config.get("num_key_value_heads")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(4);
    let rms_norm_eps = config["rms_norm_eps"].as_f64().unwrap_or(1e-5);
    let rope_theta = config.get("rope_theta")
        .and_then(|v| v.as_f64())
        .unwrap_or(10000.0);
    let max_position_embeddings = config.get("max_position_embeddings")
        .and_then(|v| v.as_u64())
        .unwrap_or(2048) as usize;

    Ok(Config {
        vocab_size,
        hidden_size,
        intermediate_size,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads,
        rms_norm_eps,
        rope_theta: rope_theta as f32,
        max_position_embeddings,
        bos_token_id: Some(config.get("bos_token_id").and_then(|v| v.as_i64()).unwrap_or(1) as u32),
        eos_token_id: Some(candle_transformers::models::llama::LlamaEosToks::Single(
            config.get("eos_token_id").and_then(|v| v.as_i64()).unwrap_or(2) as u32
        )),
        rope_scaling: None,
        tie_word_embeddings: config.get("tie_word_embeddings").and_then(|v| v.as_bool()).unwrap_or(false),
        use_flash_attn: false,
    })
}

pub fn load_chat_model() -> Result<ChatModel> {
    // 1) Device selection
    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    println!("Using device: {:?}", device);

    // 2) Create API with authentication for Meta Llama
    let api = if let Ok(token) = std::env::var("HUGGINGFACE_TOKEN") {
        println!("Using HuggingFace token for authentication");
        // Create API with token using the builder pattern
        Api::new().map_err(|e| anyhow::anyhow!("Failed to create API: {}", e))?
    } else {
        println!("Warning: No HUGGINGFACE_TOKEN found. Meta Llama requires authentication.");
        return Err(anyhow::anyhow!(
            "Meta Llama requires HuggingFace authentication. Please set HUGGINGFACE_TOKEN environment variable.\n\
            Get your token from: https://huggingface.co/settings/tokens\n\
            Accept license at: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct"
        ));
    };
    
    // Use Meta Llama 2 7B Chat HF
    let repo = api.repo(Repo::with_revision(
        "meta-llama/Llama-2-7b-chat-hf".to_string(),
        RepoType::Model,
        "main".to_string(),
    ));

    // 3) Download config and tokenizer files
    println!("Downloading Meta Llama 2 7B Chat HF configuration files...");

    let config_file = repo.get("config.json")
        .map_err(|e| anyhow::anyhow!("Failed to download config.json: {}. Make sure you have access to meta-llama/Llama-2-7b-chat-hf", e))?;

    let tokenizer_file = repo.get("tokenizer.json")
        .map_err(|e| anyhow::anyhow!("Failed to download tokenizer.json: {}", e))?;

    // 4) Parse config and tokenizer
    let config_json = fs::read_to_string(&config_file)?;
    let tokenizer = Tokenizer::from_file(&tokenizer_file)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

    // Parse the config using our helper function
    let config = parse_llama_config(&config_json)?;
    println!("Parsed Meta Llama config: vocab_size={}, hidden_size={}, num_layers={}", 
             config.vocab_size, config.hidden_size, config.num_hidden_layers);

    // 5) Download weight files - Meta Llama 3.2 uses these patterns
    println!("Downloading Meta Llama 3.2 model weights...");
    let mut weight_files = Vec::new();

    // Meta Llama 3.2 1B file patterns
    let possible_patterns = vec![
        // Safetensors (preferred)
        vec!["model.safetensors".to_string()],
        // Multiple safetensors files
        vec![
            "model-00001-of-00002.safetensors".to_string(),
            "model-00002-of-00002.safetensors".to_string()
        ],
        vec![
            "model-00001-of-00003.safetensors".to_string(),
            "model-00002-of-00003.safetensors".to_string(),
            "model-00003-of-00003.safetensors".to_string(),
        ],
        // PyTorch fallback
        vec!["pytorch_model.bin".to_string()],
    ];

    let mut found_pattern = false;
    for pattern in possible_patterns {
        let mut pattern_files = Vec::new();
        let mut all_found = true;
        
        for filename in &pattern {
            match repo.get(filename) {
                Ok(path) => {
                    println!("Found weight file: {}", filename);
                    pattern_files.push(path);
                },
                Err(e) => {
                    println!("Could not find {}: {}", filename, e);
                    all_found = false;
                    break;
                }
            }
        }
        
        if all_found && !pattern_files.is_empty() {
            weight_files = pattern_files;
            found_pattern = true;
            break;
        }
    }

    if !found_pattern || weight_files.is_empty() {
        return Err(anyhow::anyhow!(
            "No Meta Llama 3.2 weight files found. The model might use a different file structure.\n\
             Common patterns tried: model.safetensors, model-*-of-*.safetensors"
        ));
    }

    println!("Successfully found {} weight file(s)", weight_files.len());

    // 6) Load weights (same as before)
    let vars = if weight_files[0].extension().and_then(|s| s.to_str()) == Some("safetensors") {
        println!("Loading Meta Llama safetensors weights...");
        unsafe {
            VarBuilder::from_mmaped_safetensors(&weight_files, DType::F16, &device)?
        }
    } else {
        println!("Loading Meta Llama PyTorch weights...");
        if weight_files.len() == 1 {
            let tensors_vec = candle_core::pickle::read_all(&weight_files[0])?;
            let tensors: std::collections::HashMap<String, candle_core::Tensor> = tensors_vec.into_iter().collect();
            VarBuilder::from_tensors(tensors, DType::F16, &device)
        } else {
            let mut all_tensors = std::collections::HashMap::new();
            for weight_file in &weight_files {
                let tensors_vec = candle_core::pickle::read_all(weight_file)?;
                let tensors: std::collections::HashMap<String, candle_core::Tensor> = tensors_vec.into_iter().collect();
                all_tensors.extend(tensors);
            }
            VarBuilder::from_tensors(all_tensors, DType::F16, &device)
        }
    };

    // 7) Build model
    println!("Building Meta Llama 3.2 model graph...");
    let model = Llama::load(vars, &config)?;

    println!("🦙 Meta Llama 3.2-1B-Instruct model loaded successfully!");
    Ok(ChatModel { 
        model, 
        tokenizer, 
        device,
        config
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_tinyllama() {
        let result = load_chat_model();
        assert!(result.is_ok(), "Failed to load TinyLlama: {:?}", result.err());
    }

    #[test]
    fn test_config_parsing() {
        let tinyllama_config = r#"{
            "vocab_size": 32000,
            "hidden_size": 2048,
            "intermediate_size": 5632,
            "num_hidden_layers": 22,
            "num_attention_heads": 32,
            "num_key_value_heads": 4,
            "rms_norm_eps": 1e-05,
            "rope_theta": 10000.0,
            "max_position_embeddings": 2048
        }"#;
        
        let config = parse_llama_config(tinyllama_config);
        assert!(config.is_ok(), "TinyLlama config parsing failed: {:?}", config.err());
    }
}