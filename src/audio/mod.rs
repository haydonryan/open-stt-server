use anyhow::Result;
use symphonia::core::audio::{AudioBufferRef, Signal};
use symphonia::core::codecs::{CODEC_TYPE_NULL, DecoderOptions};
use symphonia::core::conv::FromSample;
use symphonia::core::io::MediaSourceStream;

fn append_mono_samples<T>(samples: &mut Vec<f32>, data: &symphonia::core::audio::AudioBuffer<T>)
where
    T: symphonia::core::sample::Sample,
    f32: symphonia::core::conv::FromSample<T>,
{
    // Downmix to mono (channel 0 only)
    let converted: Vec<f32> = data.chan(0).iter().map(|v| f32::from_sample(*v)).collect();

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

    let hint = symphonia::core::probe::Hint::new();
    let probed = symphonia::default::get_probe().format(
        &hint,
        mss,
        &Default::default(),
        &Default::default(),
    )?;

    let mut format = probed.format;

    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| anyhow::anyhow!("No supported audio tracks found"))?;

    let dec_opts = DecoderOptions::default();
    let mut decoder = symphonia::default::get_codecs().make(&track.codec_params, &dec_opts)?;

    let track_id = track.id;
    let sample_rate = track.codec_params.sample_rate.unwrap_or(16000);
    let mut pcm_data = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(symphonia::core::errors::Error::IoError(e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(e) => return Err(e.into()),
        };

        if packet.track_id() != track_id {
            continue;
        }

        match decoder.decode(&packet)? {
            AudioBufferRef::F64(buf) => append_mono_samples(&mut pcm_data, &buf),
            AudioBufferRef::F32(buf) => append_mono_samples(&mut pcm_data, &buf),
            AudioBufferRef::S32(buf) => append_mono_samples(&mut pcm_data, &buf),
            AudioBufferRef::S16(buf) => append_mono_samples(&mut pcm_data, &buf),
            AudioBufferRef::S8(buf) => append_mono_samples(&mut pcm_data, &buf),
            AudioBufferRef::U32(buf) => append_mono_samples(&mut pcm_data, &buf),
            AudioBufferRef::U16(buf) => append_mono_samples(&mut pcm_data, &buf),
            AudioBufferRef::U8(buf) => append_mono_samples(&mut pcm_data, &buf),
            AudioBufferRef::U24(buf) => append_mono_samples(&mut pcm_data, &buf),
            AudioBufferRef::S24(buf) => append_mono_samples(&mut pcm_data, &buf),
        }
    }

    if pcm_data.is_empty() {
        return Err(anyhow::anyhow!("Audio file contained no PCM samples"));
    }

    Ok((pcm_data, sample_rate))
}
