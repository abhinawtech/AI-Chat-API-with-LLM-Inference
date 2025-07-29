// src/models/chat.rs
use anyhow::Result;
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use candle_transformers::models::csm::{LlamaConfig, LlamaModel};
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;
use std::fs;

pub struct ChatModel {
    pub model: LlamaModel,
    pub tokenizer: Tokenizer,
    pub device: Device,
}

pub fn load_chat_model() -> Result<ChatModel> {
    // 1) Device selection
    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);

    // 2) Create API with authentication
    // The hf_hub crate automatically uses the token from environment variables or HF cache
    let api = Api::new()?;
    
    // Check if token is available
    if std::env::var("HUGGING_FACE_HUB_TOKEN").is_err() && std::env::var("HF_TOKEN").is_err() {
        println!("Warning: No HUGGING_FACE_HUB_TOKEN or HF_TOKEN found. You may need to set this environment variable or run 'huggingface-cli login'.");
    }

    let repo = api.repo(Repo::with_revision(
        "meta-llama/Llama-2-7b-chat-hf".to_string(),
        RepoType::Model,
        "main".to_string(),
    ));

    // 3) Download files from Hugging Face with better error handling
    println!("Downloading model files...");
    
    let config_file = repo.get("config.json")
        .map_err(|e| anyhow::anyhow!("Failed to download config.json: {}. Make sure you've accepted the Llama 2 license at https://huggingface.co/meta-llama/Llama-2-7b-chat-hf", e))?;
    
    let tokenizer_file = repo.get("tokenizer.json")
        .map_err(|e| anyhow::anyhow!("Failed to download tokenizer.json: {}", e))?;
    
    // Get all weight files - Llama models are split into multiple files
    let mut weight_files = Vec::new();
    
    // Try common patterns for weight files
    let patterns = [
        // Safetensors patterns
        ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"),
        ("model-00001-of-00003.safetensors", "model-00002-of-00003.safetensors"),
        // PyTorch patterns  
        ("pytorch_model-00001-of-00002.bin", "pytorch_model-00002-of-00002.bin"),
        ("pytorch_model-00001-of-00003.bin", "pytorch_model-00002-of-00003.bin"),
    ];
    
    for (first_file, second_file) in patterns.iter() {
        if let Ok(file1) = repo.get(first_file) {
            weight_files.push(file1);
            if let Ok(file2) = repo.get(second_file) {
                weight_files.push(file2);
            }
            // Check for third file if it exists
            let third_file = first_file.replace("00002-of-00002", "00003-of-00003")
                                     .replace("00002-of-00003", "00003-of-00003");
            if let Ok(file3) = repo.get(&third_file) {
                weight_files.push(file3);
            }
            break;
        }
    }
    
    // Try single file formats as fallback
    if weight_files.is_empty() {
        if let Ok(file) = repo.get("model.safetensors") {
            weight_files.push(file);
        } else if let Ok(file) = repo.get("pytorch_model.bin") {
            weight_files.push(file);
        }
    }
    
    if weight_files.is_empty() {
        return Err(anyhow::anyhow!(
            "No model weight files found. Expected files like:\n\
             - model-00001-of-00002.safetensors\n\
             - pytorch_model-00001-of-00002.bin\n\
             - model.safetensors\n\
             - pytorch_model.bin"
        ));
    }
    
    println!("Found {} weight file(s)", weight_files.len());

    // 4) Parse config & tokenizer
    let config_json = fs::read_to_string(&config_file)?;
    let tokenizer = Tokenizer::from_file(&tokenizer_file)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

          let flavor: candle_transformers::models::csm::Flavor = serde_json::from_str(&config_json)?;
    let config = LlamaConfig::from_flavor(flavor);
    // 5) Memory-map weights - handle multiple files
    let vars = if weight_files[0].extension().and_then(|s| s.to_str()) == Some("safetensors") {
        unsafe {
            VarBuilder::from_mmaped_safetensors(&weight_files, DType::F16, &device)?
        }
    } else {
        // For pytorch files, we might need different handling
        // Note: candle primarily works with safetensors for multi-file models
        return Err(anyhow::anyhow!("PyTorch format not fully supported for multi-file models. Please use a model with safetensors format."));
    };

    // 6) Build model graph
    println!("Building model...");
    let model = LlamaModel::new(&config, vars)?;


    println!("Model loaded successfully!");
    Ok(ChatModel { 
        model, 
        tokenizer, 
        device 
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_model() {
        // This test requires HUGGING_FACE_HUB_TOKEN to be set
        if std::env::var("HUGGING_FACE_HUB_TOKEN").is_ok() {
            let result = load_chat_model();
            assert!(result.is_ok(), "Failed to load model: {:?}", result.err());
        }
    }
}