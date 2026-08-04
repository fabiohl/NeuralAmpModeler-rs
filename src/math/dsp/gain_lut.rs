// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Look-Up Table (LUT) for ultra-fast dB to linear gain conversion.
//!
//! Designed to be instantiated via `OnceLock` and accessed in RT threads.
//! Migrated from the fastmath module for structural organization.

use crate::math::constants::*;
use std::sync::OnceLock;

/// Look-Up Table for ultra-fast dB to linear gain conversion.
/// Designed to be instantiated via `OnceLock` and accessed in RT threads.
pub struct GainLUT {
    table: [f32; GAIN_LUT_SIZE],
}

impl GainLUT {
    /// Initializes the LUT by precomputing the linear gain values.
    pub fn new() -> Self {
        let mut table = [0.0f32; GAIN_LUT_SIZE];
        for (i, item) in table.iter_mut().enumerate() {
            let db =
                (i as f32) * (GAIN_MAX_DB - GAIN_MIN_DB) / (GAIN_LUT_SIZE as f32) + GAIN_MIN_DB;
            *item = 10.0f32.powf(db / 20.0);
        }
        Self { table }
    }

    /// Converts dB to linear gain using the LUT with linear interpolation.
    #[inline(always)]
    pub fn db_to_linear(&self, db: f32) -> f32 {
        let db_clamped = db.clamp(GAIN_MIN_DB, GAIN_MAX_DB - 0.001);
        let idx_f =
            (db_clamped - GAIN_MIN_DB) * (GAIN_LUT_SIZE as f32) / (GAIN_MAX_DB - GAIN_MIN_DB);
        let idx = idx_f as usize;
        let frac = idx_f - (idx as f32);

        let v0 = self.table[idx];
        let v1 = self.table[(idx + 1).min(GAIN_LUT_SIZE - 1)];
        v0 + frac * (v1 - v0)
    }
}

impl Default for GainLUT {
    fn default() -> Self {
        Self::new()
    }
}

/// Safe global access to the gain LUT.
pub static GAIN_LUT: OnceLock<GainLUT> = OnceLock::new();

/// Returns the global reference to the gain LUT, initializing it if necessary.
pub fn get_gain_lut() -> &'static GainLUT {
    GAIN_LUT.get_or_init(GainLUT::new)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gain_lut_initialization() {
        let lut = GainLUT::new();
        // Boundary test: minimum gain
        assert!((lut.db_to_linear(GAIN_MIN_DB) - 10.0f32.powf(GAIN_MIN_DB / 20.0)).abs() < 1e-6);
        // Boundary test: maximum gain within the safe interpolation range
        let max_db = GAIN_MAX_DB - GAIN_DB_STEP;
        let max_val = lut.db_to_linear(max_db);
        let expected_max = 10.0f32.powf(max_db / 20.0);
        assert!(
            (max_val - expected_max).abs() < 1e-3,
            "GAIN_MAX_DB={GAIN_MAX_DB}, step={GAIN_DB_STEP}: lut={max_val}, expected={expected_max}"
        );
    }

    #[test]
    fn test_gain_lut_interpolation() {
        let lut = GainLUT::new();
        let mid_db = (GAIN_MIN_DB + GAIN_MAX_DB) / 2.0;
        let linear = lut.db_to_linear(mid_db);
        let expected = 10.0f32.powf(mid_db / 20.0);
        // Linear interpolation on an exponential curve introduces small error,
        // but should be very close with a 0.1dB step.
        assert!((linear - expected).abs() < 1e-3);
    }

    #[test]
    fn test_gain_lut_clamping() {
        let lut = GainLUT::new();
        let linear_min = lut.db_to_linear(GAIN_MIN_DB - 10.0);
        let linear_max = lut.db_to_linear(GAIN_MAX_DB + 10.0);
        assert_eq!(linear_min, lut.db_to_linear(GAIN_MIN_DB));
        assert_eq!(linear_max, lut.db_to_linear(GAIN_MAX_DB));
    }
}
