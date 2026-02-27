#[cfg(feature = "candle")]
pub mod audio_utils;
pub mod download;
pub mod stt_model;
#[cfg(feature = "candle")]
pub mod voxtral;
#[cfg(feature = "candle")]
pub mod whisper;

use anyhow::Result;
#[cfg(feature = "candle")]
use candle_core::{
    Device,
    utils::{cuda_is_available, metal_is_available},
};
#[cfg(feature = "candle")]
use log::info;
#[cfg(feature = "candle")]
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Mutex;

use stt_model::STTModel;
#[cfg(feature = "candle")]
use voxtral::VoxtralModel;
#[cfg(feature = "candle")]
use whisper::WhisperModel;

/// Select the best available compute device.
#[cfg(feature = "candle")]
pub fn select_device(force_cpu: bool) -> Result<Device> {
    if force_cpu {
        info!("Using CPU (forced by user)");
        return Ok(Device::Cpu);
    }
    if metal_is_available() {
        info!("Using Metal device (Apple Silicon)");
        return Device::new_metal(0)
            .map_err(|e| anyhow::anyhow!("Failed to create Metal device: {e}"));
    }
    if cuda_is_available() {
        info!("Using CUDA device");
        return Device::new_cuda(0)
            .map_err(|e| anyhow::anyhow!("Failed to create CUDA device: {e}"));
    }
    info!("Using CPU (no GPU acceleration available)");
    Ok(Device::Cpu)
}

/// Find a model file by name within the list of cached paths.
/// Returns an error rather than panicking on non-UTF-8 or missing names.
#[cfg(feature = "candle")]
pub fn find_model_file<'a>(paths: &'a [PathBuf], filename: &str) -> Result<&'a PathBuf> {
    paths
        .iter()
        .find(|p| p.file_name().and_then(|n| n.to_str()) == Some(filename))
        .ok_or_else(|| anyhow::anyhow!("{filename} not found in model files"))
}

#[cfg(feature = "candle")]
pub enum ModelInstance {
    Whisper(Box<WhisperModel>),
    Voxtral(Box<VoxtralModel>),
}

#[cfg(feature = "candle")]
impl ModelInstance {
    /// Transcribe raw PCM audio samples.
    pub fn transcribe(&mut self, audio: &[f32], sample_rate: u32) -> Result<String> {
        match self {
            Self::Whisper(m) => m.transcribe_audio(audio, sample_rate),
            Self::Voxtral(m) => m.transcribe_audio(audio, sample_rate),
        }
    }
}

/// Load a model synchronously (blocking). Call from a `spawn_blocking` context.
#[cfg(feature = "candle")]
pub fn load_model_blocking(model: &STTModel, force_cpu: bool) -> Result<ModelInstance> {
    if model.is_voxtral() {
        let m = VoxtralModel::new(model, force_cpu)?;
        Ok(ModelInstance::Voxtral(Box::new(m)))
    } else {
        let m = WhisperModel::new(model, force_cpu)?;
        Ok(ModelInstance::Whisper(Box::new(m)))
    }
}

#[cfg(feature = "candle")]
pub type SharedModel = Arc<Mutex<ModelInstance>>;

#[cfg(not(feature = "candle"))]
pub enum ModelInstance {
    Disabled,
}

#[cfg(not(feature = "candle"))]
impl ModelInstance {
    /// Transcribe raw PCM audio samples.
    pub fn transcribe(&mut self, _audio: &[f32], _sample_rate: u32) -> Result<String> {
        Err(anyhow::anyhow!(
            "Model support is disabled. Rebuild with the `candle` feature enabled."
        ))
    }
}

/// Load a model synchronously (blocking). Call from a `spawn_blocking` context.
#[cfg(not(feature = "candle"))]
pub fn load_model_blocking(_model: &STTModel, _force_cpu: bool) -> Result<ModelInstance> {
    Err(anyhow::anyhow!(
        "Model support is disabled. Rebuild with the `candle` feature enabled."
    ))
}

#[cfg(not(feature = "candle"))]
pub type SharedModel = Arc<Mutex<ModelInstance>>;
