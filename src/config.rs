use crate::models::stt_model::STTModel;
use clap::Parser;

/// A Whisper-compatible STT API server.
///
/// All options can also be set via environment variables (shown in brackets).
#[derive(Parser, Debug, Clone)]
#[command(name = "open-stt-server", version, about)]
pub struct Config {
    /// Port to listen on.
    #[arg(long, env = "OPEN_STT_PORT", default_value_t = 8080)]
    pub port: u16,

    /// Model(s) to load at startup. Can be specified multiple times.
    /// Set OPEN_STT_MODELS as a comma-separated list (e.g. "whisper-base,whisper-small").
    #[arg(
        long = "model",
        env = "OPEN_STT_MODELS",
        value_delimiter = ',',
        required = true
    )]
    pub models: Vec<STTModel>,

    /// Default model to use when the request does not specify one,
    /// or when the requested model is not loaded. Defaults to the first
    /// model in --model list.
    #[arg(long, env = "OPEN_STT_DEFAULT_MODEL")]
    pub default_model: Option<STTModel>,

    /// Force CPU inference even if CUDA is available.
    #[arg(long, env = "OPEN_STT_FORCE_CPU", default_value_t = false)]
    pub force_cpu: bool,

    /// Download model files from HuggingFace if not already cached.
    /// Without this flag, the server will fail to start if any model is missing.
    #[arg(long, env = "OPEN_STT_DOWNLOAD", default_value_t = false)]
    pub download: bool,

    /// Optional API key. If set, all requests must include
    /// "Authorization: Bearer <key>" header.
    #[arg(long, env = "OPEN_STT_API_KEY")]
    pub api_key: Option<String>,

    /// Log level (error, warn, info, debug, trace).
    #[arg(long, env = "RUST_LOG", default_value = "info")]
    pub log_level: String,
}

impl Config {
    /// Returns the effective default model (first in list if not explicitly set).
    pub fn effective_default_model(&self) -> STTModel {
        self.default_model.unwrap_or_else(|| self.models[0])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn effective_default_model_falls_back_to_first_model() {
        let config = Config {
            port: 8080,
            models: vec![STTModel::WhisperTiny, STTModel::WhisperBase],
            default_model: None,
            force_cpu: false,
            download: false,
            api_key: None,
            log_level: "info".to_string(),
        };

        assert_eq!(config.effective_default_model(), STTModel::WhisperTiny);
    }

    #[test]
    fn effective_default_model_prefers_explicit_setting() {
        let config = Config {
            port: 8080,
            models: vec![STTModel::WhisperTiny, STTModel::WhisperBase],
            default_model: Some(STTModel::WhisperBase),
            force_cpu: false,
            download: false,
            api_key: None,
            log_level: "info".to_string(),
        };

        assert_eq!(config.effective_default_model(), STTModel::WhisperBase);
    }
}
