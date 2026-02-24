use anyhow::Result;
use axum::{Router, routing::{get, post}};
use clap::Parser;
use log::info;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::net::TcpListener;
use tokio::sync::Mutex;

mod audio;
mod config;
mod models;
mod server;

use config::Config;
use models::{ModelInstance, SharedModel, load_model_blocking, stt_model::STTModel};
use server::{routes, state::AppState};

#[tokio::main]
async fn main() -> Result<()> {
    let cfg = Config::parse();

    // Initialise logging
    unsafe { std::env::set_var("RUST_LOG", &cfg.log_level) };
    env_logger::init();

    info!("open-stt-server starting up");
    info!("Models to load: {:?}", cfg.models);

    // Optionally download model files
    if cfg.download {
        info!("--download flag set, ensuring all models are available...");
        for model in &cfg.models {
            models::download::ensure_model_downloaded(model).await?;
        }
    }

    // Load models (blocking, done before serving)
    let mut loaded: HashMap<STTModel, SharedModel> = HashMap::new();

    for model in &cfg.models {
        info!("Loading model: {model}");
        let model_clone = *model;
        let force_cpu = cfg.force_cpu;

        let instance: ModelInstance = tokio::task::spawn_blocking(move || {
            load_model_blocking(&model_clone, force_cpu)
        })
        .await??;

        let shared: SharedModel = Arc::new(Mutex::new(instance));
        loaded.insert(*model, shared);
        info!("Model loaded: {model}");
    }

    let default_model = cfg.effective_default_model();
    info!("Default model: {default_model}");

    let state = AppState::new(loaded, default_model, cfg.api_key.clone());

    // Build router
    let app = Router::new()
        .route("/health", get(routes::health))
        .route("/v1/models", get(routes::list_models))
        .route("/v1/audio/transcriptions", post(routes::transcribe))
        .with_state(state);

    let addr = format!("0.0.0.0:{}", cfg.port);
    let listener = TcpListener::bind(&addr).await?;
    info!("Listening on http://{addr}");
    info!("Endpoints:");
    info!("  GET  /health");
    info!("  GET  /v1/models");
    info!("  POST /v1/audio/transcriptions");

    axum::serve(listener, app).await?;

    Ok(())
}
