use anyhow::Result;
use futures::StreamExt;
use log::{info, warn};
use ring::digest::{Context, SHA256};
use std::fmt::Write as FmtWrite;
use std::path::{Path, PathBuf};
use tokio::fs;
use tokio::io::AsyncWriteExt;

use crate::models::stt_model::STTModel;

fn get_hf_url(model_id: &str, revision: &str, filename: &str) -> String {
    format!("https://huggingface.co/{model_id}/resolve/{revision}/{filename}")
}

fn get_cache_paths(model_id: &str, revision: &str, filename: &str) -> Result<(PathBuf, PathBuf)> {
    let cache_dir = dirs::cache_dir()
        .ok_or_else(|| anyhow::anyhow!("Cannot determine cache directory"))?
        .join("huggingface")
        .join("hub");

    let model_folder = format!("models--{}", model_id.replace('/', "--"));
    let snapshot_dir = cache_dir
        .join(&model_folder)
        .join("snapshots")
        .join(revision);

    let symlink_path = snapshot_dir.join(filename);
    let blobs_dir = cache_dir.join(&model_folder).join("blobs");

    Ok((symlink_path, blobs_dir))
}

async fn download_file(model_id: &str, revision: &str, filename: &str) -> Result<PathBuf> {
    let (symlink_path, blobs_dir) = get_cache_paths(model_id, revision, filename)?;

    if symlink_path.exists() {
        info!("File already cached: {filename}");
        return Ok(symlink_path);
    }

    info!("Downloading: {filename}");
    let url = get_hf_url(model_id, revision, filename);

    fs::create_dir_all(&blobs_dir).await?;

    let final_blob_path = download_and_hash(&url, &blobs_dir).await?;

    if let Some(parent) = symlink_path.parent() {
        fs::create_dir_all(parent).await?;
    }

    let blob_relative_path = {
        let mut relative = PathBuf::new();
        relative.push("..");
        relative.push("..");
        relative.push("blobs");
        relative.push(final_blob_path.file_name().unwrap());
        relative
    };

    #[cfg(unix)]
    {
        use std::os::unix::fs::symlink;
        if let Err(e) = symlink(&blob_relative_path, &symlink_path) {
            warn!("Failed to create symlink for {filename}: {e}");
            return Ok(final_blob_path);
        }
    }

    #[cfg(not(unix))]
    {
        warn!("Symlinks not supported on this platform, using blob path directly");
        return Ok(final_blob_path);
    }

    info!("Downloaded and cached: {filename}");
    Ok(symlink_path)
}

async fn download_and_hash(url: &str, blobs_dir: &Path) -> Result<PathBuf> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(3600))
        .connect_timeout(std::time::Duration::from_secs(30))
        .build()?;

    let response = client.get(url).send().await?;
    if !response.status().is_success() {
        return Err(anyhow::anyhow!(
            "Download failed with status {}: {}",
            response.status(),
            url
        ));
    }

    let total = response.content_length();
    let tmp_name = format!(".tmp-{}", uuid::Uuid::new_v4());
    let temp_path = blobs_dir.join(tmp_name);
    let mut file = fs::File::create(&temp_path).await?;
    let mut downloaded: u64 = 0;
    let mut stream = response.bytes_stream();
    let mut hasher = Context::new(&SHA256);

    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result?;
        hasher.update(&chunk);
        file.write_all(&chunk).await?;
        downloaded += chunk.len() as u64;

        if let Some(total) = total {
            if downloaded % (10 * 1024 * 1024) < chunk.len() as u64 {
                let pct = downloaded * 100 / total;
                info!("  {pct}% ({} MB / {} MB)", downloaded / 1024 / 1024, total / 1024 / 1024);
            }
        }
    }

    file.flush().await?;
    file.sync_all().await?;
    drop(file);

    let digest = hasher.finish();
    let hash_hex = digest.as_ref().iter().fold(String::new(), |mut output, b| {
        let _ = write!(output, "{b:02x}");
        output
    });
    let final_path = blobs_dir.join(hash_hex);

    match fs::metadata(&final_path).await {
        Ok(md) if md.len() > 0 => {
            let _ = fs::remove_file(&temp_path).await;
        }
        _ => {
            if let Some(parent) = final_path.parent() {
                fs::create_dir_all(parent).await?;
            }
            fs::rename(&temp_path, &final_path).await?;
        }
    }

    Ok(final_path)
}

fn model_files(model: &STTModel) -> Vec<&'static str> {
    let mut files: Vec<&'static str> = if model.is_voxtral() {
        vec!["config.json", "tekken.json"]
    } else {
        vec!["config.json", "tokenizer.json"]
    };

    let safetensors: &[&'static str] = match model {
        STTModel::VoxtralMini => &[
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ],
        STTModel::VoxtralSmall => &[
            "model-00001-of-00011.safetensors",
            "model-00002-of-00011.safetensors",
            "model-00003-of-00011.safetensors",
            "model-00004-of-00011.safetensors",
            "model-00005-of-00011.safetensors",
            "model-00006-of-00011.safetensors",
            "model-00007-of-00011.safetensors",
            "model-00008-of-00011.safetensors",
            "model-00009-of-00011.safetensors",
            "model-00010-of-00011.safetensors",
            "model-00011-of-00011.safetensors",
        ],
        _ => &["model.safetensors"],
    };

    files.extend_from_slice(safetensors);
    files
}

/// Download a model from HuggingFace Hub if not already cached.
pub async fn ensure_model_downloaded(model: &STTModel) -> Result<()> {
    let (model_id, revision) = model.model_and_revision();
    let files = model_files(model);

    info!("Ensuring model {} is downloaded ({} files)...", model, files.len());

    for filename in &files {
        download_file(model_id, revision, filename).await?;
    }

    info!("Model {} is ready.", model);
    Ok(())
}

/// Get cached file paths for a model. Returns an error if any file is missing.
pub fn get_model_file_paths(model: &STTModel) -> Result<Vec<PathBuf>> {
    let (model_id, revision) = model.model_and_revision();
    let files = model_files(model);

    let mut paths = Vec::new();
    for filename in &files {
        let (symlink_path, _) = get_cache_paths(model_id, revision, filename)?;
        if symlink_path.exists() {
            paths.push(symlink_path);
        } else {
            return Err(anyhow::anyhow!(
                "Model file not found: {filename}. Run with --download to fetch it."
            ));
        }
    }

    Ok(paths)
}
