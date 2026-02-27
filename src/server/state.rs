use std::collections::HashMap;
use std::sync::Arc;

use crate::models::SharedModel;
use crate::models::stt_model::STTModel;

#[derive(Clone)]
pub struct AppState {
    /// Loaded models keyed by their string name.
    pub models: Arc<HashMap<String, SharedModel>>,
    /// Name of the default model.
    pub default_model: String,
    /// Optional API key for authentication.
    pub api_key: Option<String>,
}

impl AppState {
    pub fn new(
        models: HashMap<STTModel, SharedModel>,
        default_model: STTModel,
        api_key: Option<String>,
    ) -> Self {
        let models_str: HashMap<String, SharedModel> = models
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect();

        Self {
            models: Arc::new(models_str),
            default_model: default_model.to_string(),
            api_key,
        }
    }

    /// Resolve a model by name, falling back to the default.
    pub fn resolve_model(&self, requested: Option<&str>) -> Option<&SharedModel> {
        let name = requested.unwrap_or(&self.default_model);
        // Try exact match first, then try parsing the name to handle aliases
        if let Some(m) = self.models.get(name) {
            return Some(m);
        }
        // Try alias resolution via STTModel::from_str
        if let Ok(parsed) = name.parse::<STTModel>() {
            return self.models.get(parsed.as_str());
        }
        None
    }

    /// List all loaded model names.
    pub fn model_names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.models.keys().map(String::as_str).collect();
        names.sort();
        names
    }
}
