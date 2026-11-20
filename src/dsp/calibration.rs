// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Amp model input calibration and output level mode semantics.
//!
//! Aligns gain equations and bounds with the reference NAM calibration contract:
//! - `DEFAULT_CALIBRATION_DBU` (`12.0` dBu) within `[-60.0, 60.0]` dBu
//! - `OutputMode`: `Raw`, `Normalized` (default, `-18.0` dB target), `Calibrated`
//!
//! When a loaded model omits `input_level_dbu`, `output_level_dbu`, or `loudness`,
//! the corresponding compensation contributes `0.0 dB` (`1.0x`) without fabricating
//! missing metadata.

use crate::loader::LoadedModelPair;
use std::fmt;

/// Default input calibration reference level in dBu.
pub const DEFAULT_CALIBRATION_DBU: f32 = 12.0;
/// Minimum allowed input calibration reference level in dBu.
pub const MIN_CALIBRATION_DBU: f32 = -60.0;
/// Maximum allowed input calibration reference level in dBu.
pub const MAX_CALIBRATION_DBU: f32 = 60.0;
/// Target normalized loudness in LUFS/dB.
pub const TARGET_LOUDNESS_DB: f32 = -18.0;
/// Minimum plausible model loudness in dB LUFS/dB.
pub const MIN_LOUDNESS_DB: f32 = -120.0;
/// Maximum plausible model loudness in dB LUFS/dB.
pub const MAX_LOUDNESS_DB: f32 = 60.0;

/// Output gain compensation mode for neural amp models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u32)]
pub enum OutputMode {
    /// Unity model output (`0.0 dB` compensation).
    Raw = 0,
    /// Normalize output loudness to `-18.0 dB` using model `loudness` metadata.
    #[default]
    Normalized = 1,
    /// Scale output to match `input_calibration_level_dbu` using model `output_level_dbu`.
    Calibrated = 2,
}

impl OutputMode {
    /// Converts a discrete numeric index into [`OutputMode`].
    #[inline]
    pub const fn from_u32(val: u32) -> Self {
        match val {
            0 => Self::Raw,
            2 => Self::Calibrated,
            _ => Self::Normalized,
        }
    }

    /// Converts a floating-point parameter value into [`OutputMode`].
    #[inline]
    pub fn from_f32(val: f32) -> Self {
        if !val.is_finite() {
            return Self::Normalized;
        }
        Self::from_u32(val.round().clamp(0.0, 2.0) as u32)
    }

    /// Converts [`OutputMode`] into its discrete numeric value.
    #[inline]
    pub const fn to_u32(self) -> u32 {
        self as u32
    }

    /// Converts [`OutputMode`] into its floating-point parameter value.
    #[inline]
    pub const fn to_f32(self) -> f32 {
        (self as u32) as f32
    }

    /// Canonical display label (`"Raw"`, `"Normalized"`, `"Calibrated"`).
    #[inline]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Raw => "Raw",
            Self::Normalized => "Normalized",
            Self::Calibrated => "Calibrated",
        }
    }

    /// Parses text entered by a host or user (case-insensitive).
    pub fn from_text(text: &str) -> Option<Self> {
        match text.trim().to_ascii_lowercase().as_str() {
            "raw" | "0" => Some(Self::Raw),
            "normalized" | "norm" | "1" => Some(Self::Normalized),
            "calibrated" | "cal" | "2" => Some(Self::Calibrated),
            _ => None,
        }
    }
}

impl fmt::Display for OutputMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// RT-safe snapshot of amp model calibration metadata (`Copy`, zero allocations).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ModelCalibrationMeta {
    /// True only when an amp model is active.
    pub has_model: bool,
    /// Model capture input reference level in dBu, when present and valid.
    pub input_level_dbu: Option<f32>,
    /// Model capture output reference level in dBu, when present and valid.
    pub output_level_dbu: Option<f32>,
    /// Measured model loudness in dB, when present and finite.
    pub loudness: Option<f32>,
}

impl ModelCalibrationMeta {
    /// Extracts validated calibration metadata from a loaded neural amp model pair.
    pub fn from_loaded_model(pair: &LoadedModelPair) -> Self {
        let input_level_dbu = pair
            .has_input_level_dbu()
            .then(|| pair.input_level_dbu())
            .flatten()
            .filter(|v| is_valid_dbu(*v));
        let output_level_dbu = pair
            .has_output_level_dbu()
            .then(|| pair.output_level_dbu())
            .flatten()
            .filter(|v| is_valid_dbu(*v));
        let loudness = pair
            .has_loudness()
            .then(|| pair.loudness())
            .flatten()
            .filter(|v| is_valid_loudness(*v));
        Self {
            has_model: true,
            input_level_dbu,
            output_level_dbu,
            loudness,
        }
    }

    /// Empty metadata when no amp model is active.
    #[inline]
    pub const fn empty() -> Self {
        Self {
            has_model: false,
            input_level_dbu: None,
            output_level_dbu: None,
            loudness: None,
        }
    }
}

/// Resolved linear input and output calibration multipliers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CalibrationGains {
    /// Pre-model input calibration multiplier.
    pub input_mult: f32,
    /// Post-model output calibration multiplier.
    pub output_mult: f32,
}

impl Default for CalibrationGains {
    fn default() -> Self {
        Self {
            input_mult: 1.0,
            output_mult: 1.0,
        }
    }
}

/// Validates and clamps an input calibration dBu level to `[MIN_CALIBRATION_DBU, MAX_CALIBRATION_DBU]`.
#[inline]
pub fn sanitize_calibration_level_dbu(val: f32) -> f32 {
    if !val.is_finite() {
        return DEFAULT_CALIBRATION_DBU;
    }
    val.clamp(MIN_CALIBRATION_DBU, MAX_CALIBRATION_DBU)
}

#[inline]
fn is_valid_dbu(val: f32) -> bool {
    val.is_finite() && (MIN_CALIBRATION_DBU..=MAX_CALIBRATION_DBU).contains(&val)
}

#[inline]
fn is_valid_loudness(val: f32) -> bool {
    val.is_finite() && (MIN_LOUDNESS_DB..=MAX_LOUDNESS_DB).contains(&val)
}

#[inline]
fn db_to_linear_exact(db: f32) -> f32 {
    if db == 0.0 || !db.is_finite() {
        return 1.0;
    }
    let linear = 10.0_f64.powf(f64::from(db) / 20.0) as f32;
    if linear.is_finite() && linear > 0.0 {
        linear
    } else {
        1.0
    }
}

/// Computes linear `(input_mult, output_mult)` calibration gains in `f64` precision
/// without clamping valid calibration deltas at user gain LUT bounds.
///
/// Missing metadata never synthesizes fake dBu values and yields unity (`1.0`).
#[inline]
pub fn compute_calibration_gains(
    meta: ModelCalibrationMeta,
    calibrate_input: bool,
    input_calibration_level_dbu: f32,
    output_mode: OutputMode,
) -> CalibrationGains {
    if !meta.has_model {
        return CalibrationGains::default();
    }

    let cal_dbu = sanitize_calibration_level_dbu(input_calibration_level_dbu);
    let input_db = if calibrate_input {
        match meta.input_level_dbu {
            Some(model_in_dbu) => cal_dbu - model_in_dbu,
            None => 0.0,
        }
    } else {
        0.0
    };

    let output_db = match output_mode {
        OutputMode::Raw => 0.0,
        OutputMode::Normalized => match meta.loudness {
            Some(loudness_db) if is_valid_loudness(loudness_db) => TARGET_LOUDNESS_DB - loudness_db,
            _ => 0.0,
        },
        OutputMode::Calibrated => match meta.output_level_dbu {
            Some(model_out_dbu) if is_valid_dbu(model_out_dbu) => model_out_dbu - cal_dbu,
            _ => 0.0,
        },
    };

    CalibrationGains {
        input_mult: db_to_linear_exact(input_db),
        output_mult: db_to_linear_exact(output_db),
    }
}

#[cfg(test)]
#[path = "calibration_test.rs"]
mod tests;
