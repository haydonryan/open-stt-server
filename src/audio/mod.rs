use anyhow::Result;
use symphonia::core::audio::GenericAudioBufferRef;
use symphonia::core::codecs::audio::{AudioDecoderOptions, CODEC_ID_NULL_AUDIO};
use symphonia::core::errors::Error;
use symphonia::core::formats::TrackType;
use symphonia::core::formats::probe::Hint;
use symphonia::core::io::MediaSourceStream;
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
    let cursor = std::io::Cursor::new(data.to_vec());
    let mss = MediaSourceStream::new(Box::new(cursor), Default::default());

    let hint = Hint::new();
    let mut format = symphonia::default::get_probe().probe(
        &hint,
        mss,
        Default::default(),
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
