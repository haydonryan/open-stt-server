use axum::{
    Json,
    body::Bytes,
    extract::{Multipart, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use log::{info, warn};
use serde::Serialize;
use serde_json::json;

use crate::audio::decode_audio_bytes;
use crate::server::state::AppState;

// ──────────────────────────────────────────────────────────────────────────────
// Auth helper
// ──────────────────────────────────────────────────────────────────────────────

fn check_auth(state: &AppState, headers: &HeaderMap) -> Result<(), impl IntoResponse> {
    let Some(ref expected_key) = state.api_key else {
        return Ok(());
    };

    let provided = headers
        .get("Authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.strip_prefix("Bearer "));

    if provided == Some(expected_key.as_str()) {
        Ok(())
    } else {
        Err((
            StatusCode::UNAUTHORIZED,
            Json(json!({"error": {"message": "Invalid or missing API key", "type": "authentication_error"}})),
        ))
    }
}

fn api_error(status: StatusCode, message: impl std::fmt::Display, error_type: &str) -> Response {
    (
        status,
        Json(json!({"error": {"message": message.to_string(), "type": error_type}})),
    )
        .into_response()
}

// ──────────────────────────────────────────────────────────────────────────────
// POST /v1/audio/transcriptions
// ──────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct TranscriptionResponse {
    pub text: String,
}

/// OpenAI-compatible transcription endpoint.
///
/// Accepts multipart/form-data with fields:
/// - `file`            – audio file (required)
/// - `model`           – model name (optional, uses default if absent)
/// - `language`        – language hint (accepted but not used yet)
/// - `prompt`          – initial prompt (accepted but not used yet)
/// - `response_format` – "json" | "text" (default: "json")
/// - `temperature`     – float (accepted but not used yet)
pub async fn transcribe(
    State(state): State<AppState>,
    headers: HeaderMap,
    mut multipart: Multipart,
) -> impl IntoResponse {
    if let Err(e) = check_auth(&state, &headers) {
        return e.into_response();
    }

    let mut audio_bytes: Option<Bytes> = None;
    let mut model_name: Option<String> = None;
    let mut response_format = "json".to_string();

    // Parse multipart fields
    loop {
        match multipart.next_field().await {
            Ok(Some(field)) => {
                let name = field.name().unwrap_or("").to_string();
                match name.as_str() {
                    "file" => {
                        match field.bytes().await {
                            Ok(b) => audio_bytes = Some(b),
                            Err(e) => {
                                return api_error(StatusCode::BAD_REQUEST, format!("Failed to read audio file: {e}"), "invalid_request_error");
                            }
                        }
                    }
                    "model" => {
                        if let Ok(v) = field.text().await {
                            model_name = Some(v);
                        }
                    }
                    "response_format" => {
                        if let Ok(v) = field.text().await {
                            response_format = v;
                        }
                    }
                    // Accept but ignore: language, prompt, temperature
                    _ => {
                        let _ = field.bytes().await;
                    }
                }
            }
            Ok(None) => break,
            Err(e) => {
                return api_error(StatusCode::BAD_REQUEST, format!("Multipart error: {e}"), "invalid_request_error");
            }
        }
    }

    let audio_bytes = match audio_bytes {
        Some(b) => b,
        None => return api_error(StatusCode::BAD_REQUEST, "Missing required field: file", "invalid_request_error"),
    };

    // Resolve model
    let shared_model = match state.resolve_model(model_name.as_deref()) {
        Some(m) => m.clone(),
        None => {
            let requested = model_name.as_deref().unwrap_or("<default>");
            let available = state.model_names().join(", ");
            warn!("Requested model '{requested}' is not loaded. Loaded: {available}");
            return api_error(
                StatusCode::BAD_REQUEST,
                format!("Model '{requested}' is not loaded. Available: {available}"),
                "invalid_request_error",
            );
        }
    };

    // Decode audio
    let (pcm, sample_rate) = match decode_audio_bytes(&audio_bytes) {
        Ok(v) => v,
        Err(e) => return api_error(StatusCode::BAD_REQUEST, format!("Failed to decode audio: {e}"), "invalid_request_error"),
    };

    info!(
        "Transcribing {} samples @ {}Hz with model '{}'",
        pcm.len(),
        sample_rate,
        model_name.as_deref().unwrap_or(&state.default_model)
    );

    // Run inference (blocking, inside spawn_blocking to avoid blocking the async runtime)
    let text = match tokio::task::spawn_blocking(move || {
        let mut guard = shared_model.blocking_lock();
        guard.transcribe(&pcm, sample_rate)
    })
    .await
    {
        Ok(Ok(t)) => t,
        Ok(Err(e)) => return api_error(StatusCode::INTERNAL_SERVER_ERROR, format!("Transcription failed: {e}"), "server_error"),
        Err(e) => return api_error(StatusCode::INTERNAL_SERVER_ERROR, format!("Task panicked: {e}"), "server_error"),
    };

    match response_format.as_str() {
        "text" => (StatusCode::OK, text).into_response(),
        _ => (StatusCode::OK, Json(TranscriptionResponse { text })).into_response(),
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// GET /v1/models
// ──────────────────────────────────────────────────────────────────────────────

#[derive(Serialize)]
struct ModelObject {
    id: String,
    object: &'static str,
    owned_by: &'static str,
}

#[derive(Serialize)]
struct ModelsResponse {
    object: &'static str,
    data: Vec<ModelObject>,
}

pub async fn list_models(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> impl IntoResponse {
    if let Err(e) = check_auth(&state, &headers) {
        return e.into_response();
    }

    let data: Vec<ModelObject> = state
        .model_names()
        .into_iter()
        .map(|id| ModelObject {
            id: id.to_string(),
            object: "model",
            owned_by: "open-stt-server",
        })
        .collect();

    Json(ModelsResponse { object: "list", data }).into_response()
}

// ──────────────────────────────────────────────────────────────────────────────
// GET /health
// ──────────────────────────────────────────────────────────────────────────────

pub async fn health() -> impl IntoResponse {
    Json(json!({"status": "ok"}))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderValue;
    use std::collections::HashMap;
    use std::sync::Arc;

    fn state_with_key(api_key: Option<&str>) -> AppState {
        let models: HashMap<String, crate::models::SharedModel> = HashMap::new();
        AppState {
            models: Arc::new(models),
            default_model: "whisper-tiny".to_string(),
            api_key: api_key.map(|v| v.to_string()),
        }
    }

    #[test]
    fn check_auth_allows_when_no_key_configured() {
        let state = state_with_key(None);
        let headers = HeaderMap::new();

        assert!(check_auth(&state, &headers).is_ok());
    }

    #[test]
    fn check_auth_rejects_missing_or_invalid_key() {
        let state = state_with_key(Some("secret"));
        let headers = HeaderMap::new();

        let response = check_auth(&state, &headers)
            .expect_err("missing key should be rejected")
            .into_response();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);

        let mut headers = HeaderMap::new();
        headers.insert("Authorization", HeaderValue::from_static("Bearer wrong"));
        let response = check_auth(&state, &headers)
            .expect_err("invalid key should be rejected")
            .into_response();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[test]
    fn check_auth_accepts_valid_key() {
        let state = state_with_key(Some("secret"));
        let mut headers = HeaderMap::new();
        headers.insert("Authorization", HeaderValue::from_static("Bearer secret"));

        assert!(check_auth(&state, &headers).is_ok());
    }
}
