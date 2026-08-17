// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Robust WAV IR loader: orchestration of parsing, resampling, and normalization.
//!
//! Loading, resampling, and normalization happen **outside** the audio thread.
//! The resulting `CabSimIr` is transferred to the RT callback via lock-free SPSC,
//! following the same pattern as `NamResampler` swapping.

use super::ir_parse;
use super::ir_resample;
use log::{debug, info};

use std::io;
use std::path::Path;

/// A loaded impulse response ready for convolution.
///
/// All memory is pre-allocated at load time (outside the audio thread).
/// The struct holds the resampled, normalized mono samples alongside
/// the original metadata needed for convolution setup.
pub struct CabSimIr {
    /// Mono IR samples (f32).
    pub samples: Vec<f32>,
    /// Sample rate of the IR after resampling (i.e., the effective rate of `samples`).
    pub sample_rate: u32,
    /// Sample rate of the original WAV file (before resampling, if applicable).
    pub original_rate: u32,
    /// Whether the IR was normalized (peak = 1.0).
    pub normalized: bool,
}

impl CabSimIr {
    /// Loads a mono WAV impulse response from `path`.
    ///
    /// Supported formats: PCM16, PCM24, IEEE float32.
    ///
    /// # Parameters
    /// - `path`: path to the `.wav` file.
    /// - `target_rate`: desired sample rate after resampling. If 0, no resampling.
    /// - `normalize`: if true, normalize samples so peak amplitude = 1.0.
    ///
    /// # Errors
    /// Returns `io::Error` on I/O failures, invalid headers, unsupported formats,
    /// or resampling errors. Never panics.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::path::Path;
    /// use neural_amp_modeler_rs::dsp::cabsim::loader::CabSimIr;
    ///
    /// let ir = CabSimIr::load(Path::new("path/to/ir.wav"), 48000, true)
    ///     .expect("Failed to load impulse response");
    /// assert_eq!(ir.sample_rate, 48000);
    /// ```
    #[cold]
    pub fn load(path: &Path, target_rate: u32, normalize: bool) -> io::Result<Box<Self>> {
        info!(
            "[Loader] Loading IR from \"{}\" (target_rate={} Hz, normalize={})",
            path.display(),
            target_rate,
            normalize
        );
        let data = ir_parse::read_file(path)?;
        let (samples, original_rate) = ir_parse::parse_wav(&data)?;

        let mut samples = if target_rate != 0 && target_rate != original_rate {
            info!(
                "[Loader] IR resampling: {} Hz -> {} Hz",
                original_rate, target_rate
            );
            ir_resample::resample(&samples, original_rate, target_rate)?
        } else {
            samples
        };

        let effective_rate = if target_rate != 0 && target_rate != original_rate {
            target_rate
        } else {
            original_rate
        };

        let normalized = if normalize {
            let was_normalized = Self::normalize_in_place(&mut samples);
            if was_normalized {
                info!("[Loader] IR normalized to peak 1.0");
            } else {
                debug!("[Loader] IR normalization skipped (peak already ~1.0 or zero)");
            }
            was_normalized
        } else {
            false
        };

        info!(
            "[Loader] IR loaded: {} samples, {} Hz, normalized={}",
            samples.len(),
            effective_rate,
            normalized
        );

        Ok(Box::new(Self {
            samples,
            sample_rate: effective_rate,
            original_rate,
            normalized,
        }))
    }

    /// Normalizes samples in-place so the absolute peak becomes 1.0.
    ///
    /// Returns `true` if normalization was applied (peak > 0).
    /// If all samples are zero, they are left unchanged and returns `false`.
    fn normalize_in_place(samples: &mut [f32]) -> bool {
        let peak = samples.iter().fold(0.0f32, |acc, &s| acc.max(s.abs()));

        if peak <= 0.0 || peak >= 1.0 && (peak - 1.0).abs() < f32::EPSILON {
            return false;
        }

        let gain = 1.0 / peak;
        for s in samples.iter_mut() {
            *s *= gain;
        }
        true
    }

    /// Resamples `input` from `input_rate` to `output_rate` using the polyphase resampler.
    ///
    /// Public delegate to `ir_resample::resample` — preserved for external consumers
    /// (e.g. downstream integration test suites).
    pub fn resample(input: &[f32], input_rate: u32, output_rate: u32) -> io::Result<Vec<f32>> {
        ir_resample::resample(input, input_rate, output_rate)
    }
}

#[cfg(test)]
#[path = "loader_test.rs"]
mod loader_test;
