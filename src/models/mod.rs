pub mod download;
pub mod stt_model;
#[cfg(feature = "candle")]
pub mod audio_utils;
#[cfg(feature = "candle")]
pub mod voxtral;
#[cfg(feature = "candle")]
pub mod whisper;

use anyhow::Result;
use std::sync::Arc;
use tokio::sync::Mutex;

use stt_model::STTModel;
#[cfg(feature = "candle")]
use voxtral::VoxtralModel;
#[cfg(feature = "candle")]
use whisper::WhisperModel;

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
