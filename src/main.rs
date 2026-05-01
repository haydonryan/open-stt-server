#![allow(
    clippy::default_trait_access,
    clippy::let_underscore_must_use,
    clippy::manual_let_else,
    clippy::needless_pass_by_ref_mut,
    clippy::too_many_lines,
    clippy::trivially_copy_pass_by_ref
)]

use anyhow::Result;
use axum::{
    Router,
    routing::{get, post},
};
use clap::Parser;
use flate2::read::GzDecoder;
use log::info;
use semver::Version;
use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::Path;
use std::sync::Arc;
use tar::Archive;
use tempfile::tempdir;
use tokio::net::TcpListener;
use tokio::sync::Mutex;
use zip::ZipArchive;

mod audio;
mod config;
mod models;
mod server;

use config::Config;
use models::{ModelInstance, SharedModel, load_model_blocking, stt_model::STTModel};
use server::{routes, state::AppState};

const GITHUB_OWNER: &str = "haydonryan";
const GITHUB_REPO: &str = "open-stt-server";

#[derive(serde::Deserialize)]
struct GitHubRelease {
    tag_name: String,
    assets: Vec<GitHubAsset>,
}

#[derive(serde::Deserialize)]
struct GitHubAsset {
    name: String,
    browser_download_url: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    if std::env::args().nth(1).as_deref() == Some("update") {
        update().await?;
        return Ok(());
    }
    let cfg = Config::parse();

    // Initialise logging
    env_logger::Builder::new()
        .parse_filters(&cfg.log_level)
        .init();

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

        let instance: ModelInstance =
            tokio::task::spawn_blocking(move || load_model_blocking(&model_clone, force_cpu))
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

const fn binary_name() -> &'static str {
    env!("CARGO_PKG_NAME")
}

fn target_triplet_and_archive() -> (&'static str, &'static str) {
    if cfg!(all(
        target_os = "linux",
        target_arch = "x86_64",
        target_env = "gnu"
    )) {
        ("x86_64-unknown-linux-gnu", "tar.gz")
    } else if cfg!(all(
        target_os = "linux",
        target_arch = "x86_64",
        target_env = "musl"
    )) {
        ("x86_64-unknown-linux-musl", "tar.gz")
    } else if cfg!(all(target_os = "linux", target_arch = "aarch64")) {
        ("aarch64-unknown-linux-gnu", "tar.gz")
    } else if cfg!(all(target_os = "macos", target_arch = "x86_64")) {
        ("x86_64-apple-darwin", "tar.gz")
    } else if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        ("aarch64-apple-darwin", "tar.gz")
    } else if cfg!(all(target_os = "windows", target_arch = "x86_64")) {
        ("x86_64-pc-windows-msvc", "zip")
    } else if cfg!(all(target_os = "windows", target_arch = "aarch64")) {
        ("aarch64-pc-windows-msvc", "zip")
    } else {
        panic!("unsupported target platform for self-update");
    }
}

fn parse_release_version(tag_name: &str) -> Result<Version> {
    let normalized = tag_name.strip_prefix('v').unwrap_or(tag_name);
    Ok(Version::parse(normalized)?)
}

async fn fetch_latest_release() -> Result<GitHubRelease> {
    let url = format!("https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/releases/latest");
    Ok(reqwest::Client::new()
        .get(url)
        .header(reqwest::header::USER_AGENT, "open-stt-server")
        .header(reqwest::header::ACCEPT, "application/vnd.github+json")
        .send()
        .await?
        .error_for_status()?
        .json::<GitHubRelease>()
        .await?)
}

async fn download_file(url: &str, dest: &Path) -> Result<()> {
    let bytes = reqwest::Client::new()
        .get(url)
        .header(reqwest::header::USER_AGENT, "open-stt-server")
        .send()
        .await?
        .error_for_status()?
        .bytes()
        .await?;
    tokio::fs::write(dest, bytes).await?;
    Ok(())
}

fn extract_release_asset(archive_path: &Path, output_path: &Path) -> Result<()> {
    if archive_path.extension().and_then(|s| s.to_str()) == Some("zip") {
        let file = fs::File::open(archive_path)?;
        let mut archive = ZipArchive::new(file)?;
        for i in 0..archive.len() {
            let mut entry = archive.by_index(i)?;
            let name = entry.name().to_string();
            if Path::new(&name).file_name().and_then(|s| s.to_str()) == Some(binary_name()) {
                let mut output = fs::File::create(output_path)?;
                io::copy(&mut entry, &mut output)?;
                return Ok(());
            }
        }
    } else {
        let file = fs::File::open(archive_path)?;
        let decompressed = GzDecoder::new(file);
        let mut archive = Archive::new(decompressed);
        for entry in archive.entries()? {
            let mut entry = entry?;
            let path = entry.path()?;
            if path.file_name().and_then(|s| s.to_str()) == Some(binary_name()) {
                entry.unpack(output_path)?;
                return Ok(());
            }
        }
    }
    anyhow::bail!("Binary not found in release archive");
}

fn install_binary(new_binary: &Path) -> Result<()> {
    let current_exe = std::env::current_exe()?;
    let current_dir = current_exe
        .parent()
        .ok_or_else(|| anyhow::anyhow!("Failed to determine executable directory"))?;
    let temp_path = current_dir.join(format!("{}.new", binary_name()));
    fs::copy(new_binary, &temp_path)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = fs::metadata(&temp_path)?.permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&temp_path, perms)?;
    }
    fs::rename(&temp_path, &current_exe)?;
    Ok(())
}

async fn update() -> Result<()> {
    if cfg!(target_os = "windows") {
        anyhow::bail!("Self-update is not supported on Windows for this binary.");
    }
    let current = Version::parse(env!("CARGO_PKG_VERSION"))?;
    let release = fetch_latest_release().await?;
    let latest = parse_release_version(&release.tag_name)?;
    if latest <= current {
        println!("Already on the latest version ({current})");
        return Ok(());
    }
    let (target, archive_ext) = target_triplet_and_archive();
    let asset_name = format!("{}-{}.{}", binary_name(), target, archive_ext);
    let asset = release
        .assets
        .iter()
        .find(|a| a.name == asset_name)
        .ok_or_else(|| anyhow::anyhow!("missing release asset"))?;
    let dir = tempdir()?;
    let archive_path = dir.path().join(&asset.name);
    let extracted = dir.path().join(binary_name());
    download_file(&asset.browser_download_url, &archive_path).await?;
    extract_release_asset(&archive_path, &extracted)?;
    install_binary(&extracted)?;
    println!("Updated {} from {} to {}", binary_name(), current, latest);
    Ok(())
}
