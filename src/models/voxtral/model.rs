use std::path::PathBuf;

use anyhow::{Context, Error, Result};
use byteorder::{LittleEndian, ReadBytesExt};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::voxtral::{
    VoxtralCache, VoxtralConfig, VoxtralEncoderConfig, VoxtralForConditionalGeneration,
    VoxtralGenerationConfig, VoxtralLlamaConfig, audio,
};
use log::{debug, info, warn};
use std::io::Cursor;
use tekken::Tekkenizer;

use crate::models::audio_utils::{ResampleQuality, resample};
use crate::models::stt_model::STTModel;

const SAMPLE_RATE: u32 = 16000;

pub struct VoxtralModel {
    model: VoxtralForConditionalGeneration,
    tokenizer: Tekkenizer,
    device: Device,
    #[allow(dead_code)]
    config: VoxtralConfig,
    audio_token_id: usize,
    mel_filters: Vec<f32>,
    cache: VoxtralCache,
}

impl VoxtralModel {
    pub fn new(stt_model: &STTModel, force_cpu: bool) -> Result<Self> {
        info!("Loading Voxtral {stt_model:?} model...");

        let device = crate::models::select_device(force_cpu)?;

        let file_paths = crate::models::download::get_model_file_paths(stt_model)?;

        let config_path = crate::models::find_model_file(&file_paths, "config.json")?;
        let tokenizer_path = crate::models::find_model_file(&file_paths, "tekken.json")?;

        let safetensors_files: Vec<PathBuf> = file_paths
            .iter()
            .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("safetensors"))
            .cloned()
            .collect();

        info!("Loading model configuration...");
        let config = load_model_config(config_path)?;

        info!("Loading model weights ({} files)...", safetensors_files.len());
        let vb = load_model_weights(&safetensors_files, &device)?;

        info!("Creating Voxtral model...");
        let model = VoxtralForConditionalGeneration::new(&config, vb)?;

        info!("Loading tokenizer...");
        let tokenizer = Tekkenizer::from_file(tokenizer_path).map_err(Error::msg)?;
        debug!("Loaded tokenizer");

        let cache = VoxtralCache::new(true, DType::F16, &config.text_config, &device)?;

        let audio_token_id = config.audio_token_id;

        let mel_bytes = include_bytes!("../data/melfilters128.bytes");
        let mut mel_filters = vec![0f32; mel_bytes.len() / 4];
        let mut cursor = Cursor::new(mel_bytes);
        cursor.read_f32_into::<LittleEndian>(&mut mel_filters)?;

        info!("Voxtral model loaded successfully on {device:?}");

        Ok(Self {
            model,
            tokenizer,
            device,
            config,
            audio_token_id,
            mel_filters,
            cache,
        })
    }

    pub fn transcribe_audio(&mut self, audio_data: &[f32], sample_rate: u32) -> Result<String> {
        let audio = if sample_rate == SAMPLE_RATE {
            audio_data.to_vec()
        } else {
            warn!("Audio sample rate is {sample_rate}Hz, resampling to {SAMPLE_RATE}Hz");
            resample(audio_data, sample_rate, SAMPLE_RATE, ResampleQuality::Fast)?
        };

        let chunk_size = 480_000;
        let padded_audio = if audio.len() % chunk_size != 0 {
            let target_samples = ((audio.len() / chunk_size) + 1) * chunk_size;
            let mut padded = audio.clone();
            padded.resize(target_samples, 0.0);
            padded
        } else {
            audio
        };

        let audio_features = audio::extract_features(&padded_audio, &self.mel_filters, &self.device)?;

        let (result, _tokens) = transcribe_with_voxtral(
            &self.model,
            &self.tokenizer,
            &audio_features,
            self.audio_token_id,
            &self.device,
            &self.cache.clone(),
        )?;

        Ok(result)
    }
}

fn transcribe_with_voxtral(
    model: &VoxtralForConditionalGeneration,
    tokenizer: &Tekkenizer,
    audio_features: &Tensor,
    audio_token_id: usize,
    device: &Device,
    cache: &VoxtralCache,
) -> Result<(String, Vec<u32>)> {
    let audio_dims = audio_features.dims();
    if audio_dims.len() != 3 {
        return Err(anyhow::anyhow!(
            "Audio features must be 3D tensor, got shape: {audio_dims:?}"
        ));
    }
    if audio_dims[1] != 128 {
        return Err(anyhow::anyhow!(
            "Audio features must have 128 mel bins, got {}",
            audio_dims[1]
        ));
    }

    let mut input_tokens = Vec::new();
    input_tokens.push(1u32); // BOS <s>
    input_tokens.push(3u32); // [INST]
    input_tokens.push(25u32); // [BEGIN_AUDIO]

    let batch_size = audio_features.dim(0)?;
    let tokens_per_chunk = 375;
    let num_audio_tokens = batch_size * tokens_per_chunk;

    for _ in 0..num_audio_tokens {
        #[allow(clippy::cast_possible_truncation)]
        input_tokens.push(audio_token_id as u32);
    }

    input_tokens.push(4u32); // [/INST]
    input_tokens.push(9909u32); // lang
    input_tokens.push(1058u32); // :
    input_tokens.push(1262u32); // en
    input_tokens.push(34u32); // [TRANSCRIBE]

    let input_len = input_tokens.len();
    let input_ids = Tensor::new(input_tokens, device)?.unsqueeze(0)?;

    let config = VoxtralGenerationConfig {
        max_new_tokens: 1000,
        temperature: 0.0,
        top_p: None,
        device: device.clone(),
        cache: Some(cache.clone()),
    };

    let generated_tokens = model
        .generate(
            &input_ids,
            Some(audio_features),
            config,
        )
        .map_err(|e| anyhow::anyhow!("Failed to generate tokens: {e}"))?;

    let new_tokens = if generated_tokens.len() > input_len {
        &generated_tokens[input_len..]
    } else {
        &generated_tokens
    };

    let decoded_text = tokenizer
        .decode(new_tokens, tekken::SpecialTokenPolicy::Ignore)
        .map_err(|e| anyhow::anyhow!("Failed to decode tokens: {e}"))?;

    let transcription = post_process_transcription(&decoded_text).unwrap_or(decoded_text);

    Ok((transcription, new_tokens.to_vec()))
}

fn post_process_transcription(text: &str) -> Option<String> {
    let mut cleaned = text.trim().to_string();

    if cleaned.starts_with("\"'") || cleaned.starts_with("'\"") {
        cleaned = cleaned
            .trim_start_matches("\"'")
            .trim_start_matches("'\"")
            .trim()
            .to_string();
    }

    if cleaned.starts_with('\'') {
        cleaned = cleaned[1..].trim().to_string();
    }

    cleaned = cleaned.split_whitespace().collect::<Vec<_>>().join(" ");

    cleaned = cleaned.replace(" \"' ", " ").replace(" '\" ", " ");

    if cleaned == "." || cleaned.trim().is_empty() {
        return None;
    }

    cleaned = cleaned
        .trim_end_matches('\'')
        .trim_end_matches('"')
        .to_string();

    Some(cleaned)
}

fn load_model_weights<'a>(model_files: &'a [PathBuf], device: &Device) -> Result<VarBuilder<'a>> {
    let dtype = DType::F16;
    info!("Loading {} safetensors files...", model_files.len());
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(model_files, dtype, device)? };
    Ok(vb)
}

fn load_model_config(config_file: &PathBuf) -> Result<VoxtralConfig> {
    let config_str = std::fs::read_to_string(config_file)?;
    let json: serde_json::Value =
        serde_json::from_str(&config_str).context("Failed to parse config.json")?;

    let audio_token_id = json
        .get("audio_token_id")
        .and_then(serde_json::Value::as_u64)
        .and_then(|v| usize::try_from(v).ok())
        .unwrap_or(24);

    let audio_config = parse_audio_config(&json)?;
    let text_config = parse_text_config(&json)?;

    let projector_hidden_act = json
        .get("projector_hidden_act")
        .and_then(|v| v.as_str())
        .unwrap_or("gelu")
        .to_string();

    Ok(VoxtralConfig {
        audio_config,
        text_config,
        audio_token_id,
        projector_hidden_act,
    })
}

fn parse_audio_config(json: &serde_json::Value) -> Result<VoxtralEncoderConfig> {
    let audio_json = json
        .get("audio_config")
        .ok_or_else(|| anyhow::anyhow!("Missing audio_config in configuration"))?;

    Ok(VoxtralEncoderConfig {
        vocab_size: audio_json
            .get("vocab_size")
            .and_then(serde_json::Value::as_u64)
            .and_then(|v| usize::try_from(v).ok())
            .unwrap_or(51866),
        hidden_size: audio_json
            .get("hidden_size")
            .and_then(serde_json::Value::as_u64)
            .and_then(|v| usize::try_from(v).ok())
            .unwrap_or(1280),
        num_hidden_layers: audio_json
            .get("num_hidden_layers")
            .and_then(serde_json::Value::as_u64)
            .and_then(|v| usize::try_from(v).ok())
            .unwrap_or(32),
        num_attention_heads: audio_json
            .get("num_attention_heads")
            .and_then(serde_json::Value::as_u64)
            .and_then(|v| usize::try_from(v).ok())
            .unwrap_or(20),
        num_key_value_heads: audio_json
            .get("num_key_value_heads")
            .and_then(serde_json::Value::as_u64)
            .and_then(|v| usize::try_from(v).ok())
            .unwrap_or(20),
        intermediate_size: audio_json
            .get("intermediate_size")
            .and_then(serde_json::Value::as_u64)
            .and_then(|v| usize::try_from(v).ok())
            .unwrap_or(5120),
        dropout: audio_json
            .get("dropout")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(0.0),
        attention_dropout: audio_json
            .get("attention_dropout")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(0.0),
        activation_dropout: audio_json
            .get("activation_dropout")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(0.0),
        activation_function: audio_json
            .get("activation_function")
            .and_then(|v| v.as_str())
            .unwrap_or("gelu")
            .to_string(),
        max_source_positions: audio_json
            .get("max_source_positions")
            .and_then(serde_json::Value::as_u64)
            .and_then(|v| usize::try_from(v).ok())
            .unwrap_or(1500),
        layerdrop: audio_json
            .get("layerdrop")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(0.0),
        initializer_range: audio_json
            .get("initializer_range")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(0.02),
        scale_embedding: audio_json
            .get("scale_embedding")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false),
        num_mel_bins: audio_json
            .get("num_mel_bins")
            .and_then(serde_json::Value::as_u64)
            .and_then(|v| usize::try_from(v).ok())
            .unwrap_or(128),
        head_dim: audio_json
            .get("head_dim")
            .and_then(serde_json::Value::as_u64)
            .and_then(|v| usize::try_from(v).ok())
            .unwrap_or(64),
    })
}

#[cfg(feature = "flash-attn")]
const fn use_flash_attn() -> bool {
    true
}

#[cfg(not(feature = "flash-attn"))]
const fn use_flash_attn() -> bool {
    false
}

fn parse_text_config(json: &serde_json::Value) -> Result<VoxtralLlamaConfig> {
    let text_json = json
        .get("text_config")
        .ok_or_else(|| anyhow::anyhow!("Missing text_config in configuration"))?;

    let use_flash_attn = use_flash_attn();

    Ok(VoxtralLlamaConfig {
        vocab_size: text_json
            .get("vocab_size")
            .and_then(serde_json::Value::as_u64)
            .and_then(|v| usize::try_from(v).ok())
            .unwrap_or(131_072),
        hidden_size: text_json
            .get("hidden_size")
            .and_then(serde_json::Value::as_u64)
            .and_then(|v| usize::try_from(v).ok())
            .unwrap_or(3072),
        intermediate_size: text_json
            .get("intermediate_size")
            .and_then(serde_json::Value::as_u64)
            .and_then(|v| usize::try_from(v).ok())
            .unwrap_or(8192),
        num_hidden_layers: text_json
            .get("num_hidden_layers")
            .and_then(serde_json::Value::as_u64)
            .and_then(|v| usize::try_from(v).ok())
            .unwrap_or(30),
        num_attention_heads: text_json
            .get("num_attention_heads")
            .and_then(serde_json::Value::as_u64)
            .and_then(|v| usize::try_from(v).ok())
            .unwrap_or(32),
        num_key_value_heads: text_json
            .get("num_key_value_heads")
            .and_then(serde_json::Value::as_u64)
            .and_then(|v| usize::try_from(v).ok())
            .unwrap_or(8),
        head_dim: text_json
            .get("head_dim")
            .and_then(serde_json::Value::as_u64)
            .and_then(|v| usize::try_from(v).ok()),
        rms_norm_eps: text_json
            .get("rms_norm_eps")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(1e-5),
        rope_theta: {
            let v = text_json
                .get("rope_theta")
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(100_000_000.0);
            #[allow(clippy::cast_possible_truncation)]
            { v as f32 }
        },
        max_position_embeddings: text_json
            .get("max_position_embeddings")
            .and_then(serde_json::Value::as_u64)
            .and_then(|v| usize::try_from(v).ok())
            .unwrap_or(131_072),
        use_flash_attn,
        tie_word_embeddings: text_json
            .get("attention_bias")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false),
    })
}
