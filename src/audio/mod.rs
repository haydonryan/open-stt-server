use anyhow::Result;
use symphonia::core::audio::GenericAudioBufferRef;
use symphonia::core::codecs::audio::{AudioDecoderOptions, CODEC_ID_NULL_AUDIO};
use symphonia::core::errors::Error;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::{FormatOptions, TrackType};
use symphonia::core::io::{MediaSourceStream, MediaSourceStreamOptions};
use symphonia::core::meta::MetadataOptions;

fn append_mono_samples(samples: &mut Vec<f32>, data: &GenericAudioBufferRef<'_>) {
    // Downmix to mono (channel 0 only)
    let mut channels = Vec::<Vec<f32>>::new();
    data.copy_to_vecs_planar(&mut channels);

    let Some(converted) = channels.into_iter().next() else {
        return;
    };

    let max_abs = converted.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
    if max_abs > 0.0 && max_abs < 1e-3 {
        let scale = 0.5 / max_abs;
        samples.extend(converted.iter().map(|&x| x * scale));
    } else {
        samples.extend(converted);
    }
}

/// Decode audio bytes (any format supported by symphonia) to mono PCM f32 samples.
pub fn decode_audio_bytes(data: &[u8]) -> Result<(Vec<f32>, u32)> {
    let cursor = std::io::Cursor::new(data);
    let mss = MediaSourceStream::new(Box::new(cursor), MediaSourceStreamOptions::default());

    let hint = Hint::new();
    let mut format = symphonia::default::get_probe().probe(
        &hint,
        mss,
        FormatOptions::default(),
        MetadataOptions::default(),
    )?;

    let track = format
        .default_track(TrackType::Audio)
        .or_else(|| {
            format.tracks().iter().find(|track| {
                track
                    .codec_params
                    .as_ref()
                    .and_then(|params| params.audio())
                    .is_some_and(|params| params.codec != CODEC_ID_NULL_AUDIO)
            })
        })
        .ok_or_else(|| anyhow::anyhow!("No supported audio tracks found"))?;

    let audio_params = track
        .codec_params
        .as_ref()
        .and_then(|params| params.audio())
        .ok_or_else(|| anyhow::anyhow!("Audio track missing codec parameters"))?;

    let dec_opts = AudioDecoderOptions::default();
    let mut decoder =
        symphonia::default::get_codecs().make_audio_decoder(audio_params, &dec_opts)?;

    let track_id = track.id;
    let sample_rate = audio_params.sample_rate.unwrap_or(16000);
    let mut pcm_data = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(Some(packet)) => packet,
            Ok(None) => break,
            Err(Error::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e.into()),
        };

        if packet.track_id != track_id {
            continue;
        }

        let decoded_audio = decoder.decode(&packet)?;
        append_mono_samples(&mut pcm_data, &decoded_audio);
    }

    if pcm_data.is_empty() {
        return Err(anyhow::anyhow!("Audio file contained no PCM samples"));
    }

    Ok((pcm_data, sample_rate))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal mono 16-bit PCM WAV in memory.
    /// Build a minimal mono 16-bit PCM WAV in memory.
    // WAV header fields are 16-bit; the truncating casts are inherent to the format.
    #[allow(clippy::cast_possible_truncation)]
    fn build_wav(sample_rate: u32, sample_count: usize) -> Vec<u8> {
        let bits_per_sample: u32 = 16;
        let channels: u16 = 1;
        let block_align = u32::from(channels) * bits_per_sample / 8;
        let byte_rate = sample_rate * block_align;
        let data_size = (sample_count as u32) * block_align;

        let mut wav = Vec::with_capacity(44 + data_size as usize);
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&(36 + data_size).to_le_bytes());
        wav.extend_from_slice(b"WAVE");
        wav.extend_from_slice(b"fmt ");
        wav.extend_from_slice(&16u32.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes()); // PCM
        wav.extend_from_slice(&channels.to_le_bytes());
        wav.extend_from_slice(&sample_rate.to_le_bytes());
        wav.extend_from_slice(&byte_rate.to_le_bytes());
        wav.extend_from_slice(&(block_align as u16).to_le_bytes());
        wav.extend_from_slice(&(bits_per_sample as u16).to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&data_size.to_le_bytes());
        let mut phase: i16 = 0;
        for _ in 0..sample_count {
            phase = phase.wrapping_add(997);
            wav.extend_from_slice(&phase.to_le_bytes());
        }
        wav
    }

    #[test]
    fn decodes_generated_wav_with_expected_sample_count_and_rate() {
        let sample_rate = 8000;
        let sample_count = 8000; // 1 second of audio
        let wav = build_wav(sample_rate, sample_count);

        let (pcm, decoded_rate) =
            decode_audio_bytes(&wav).expect("generated WAV should decode successfully");

        assert_eq!(
            decoded_rate, sample_rate,
            "sample rate should match the WAV header"
        );
        assert_eq!(
            pcm.len(),
            sample_count,
            "sample count should match the generated audio"
        );
        assert!(
            pcm.iter().any(|&s| s != 0.0),
            "decoded audio should contain non-zero samples"
        );
    }
}
