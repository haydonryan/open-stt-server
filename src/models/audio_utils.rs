use anyhow::Result;
use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub enum ResampleQuality {
    Fast,
    Balanced,
    HighQuality,
}

/// Resample audio from one sample rate to another.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
pub fn resample(
    samples: &[f32],
    from_sr: u32,
    to_sr: u32,
    quality: ResampleQuality,
) -> Result<Vec<f32>> {
    if from_sr == to_sr {
        return Ok(samples.to_vec());
    }

    let params = match quality {
        ResampleQuality::Fast => SincInterpolationParameters {
            sinc_len: 64,
            f_cutoff: 0.95,
            interpolation: SincInterpolationType::Nearest,
            oversampling_factor: 16,
            window: WindowFunction::Hann,
        },
        ResampleQuality::Balanced => SincInterpolationParameters {
            sinc_len: 128,
            f_cutoff: 0.95,
            interpolation: SincInterpolationType::Linear,
            oversampling_factor: 128,
            window: WindowFunction::Blackman,
        },
        ResampleQuality::HighQuality => SincInterpolationParameters {
            sinc_len: 256,
            f_cutoff: 0.95,
            interpolation: SincInterpolationType::Cubic,
            oversampling_factor: 256,
            window: WindowFunction::BlackmanHarris2,
        },
    };

    let mut resampler = SincFixedIn::<f32>::new(
        f64::from(to_sr) / f64::from(from_sr),
        2.0,
        params,
        samples.len(),
        1,
    )?;

    let waves_in = vec![samples.to_vec()];
    let waves_out = resampler.process(&waves_in, None)?;

    Ok(waves_out.into_iter().next().unwrap())
}
