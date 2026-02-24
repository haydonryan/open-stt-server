pub mod audio_utils;
pub mod download;
pub mod stt_model;
pub mod voxtral;
pub mod whisper;

use anyhow::Result;
use std::sync::Arc;
use tokio::sync::Mutex;

use stt_model::STTModel;
use voxtral::VoxtralModel;
use whisper::WhisperModel;

pub enum ModelInstance {
    Whisper(Box<WhisperModel>),
    Voxtral(Box<VoxtralModel>),
}

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
pub fn load_model_blocking(model: &STTModel, force_cpu: bool) -> Result<ModelInstance> {
    if model.is_voxtral() {
        let m = VoxtralModel::new(model, force_cpu)?;
        Ok(ModelInstance::Voxtral(Box::new(m)))
    } else {
        let m = WhisperModel::new(model, force_cpu)?;
        Ok(ModelInstance::Whisper(Box::new(m)))
    }
}

pub type SharedModel = Arc<Mutex<ModelInstance>>;
