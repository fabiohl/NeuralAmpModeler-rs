// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! WeightCursor — Deterministic sequential reading of flattened weights
//!
//! Weights are consumed sequentially by a forward-only cursor,
//! with an exhaustion check at the end to detect inconsistent models.

use anyhow::bail;

/// Forward-only read cursor over the flattened weight vector.
///
/// Ensures:
/// - No weight is read out of bounds (`read_slice` / `read_f32`)
/// - All weights have been consumed at the end (`verify_exhausted`)
pub(crate) struct WeightCursor<'a> {
    /// Reference to the full model weight slice.
    data: &'a [f32],
    /// Current cursor position (advances with each read).
    pos: usize,
    /// Weight layout reported in the binary header.
    pub layout: crate::loader::nam_json::WeightsLayout,
}

impl<'a> WeightCursor<'a> {
    /// Creates a new cursor over the weight slice with the specified layout.
    #[cold]
    pub fn new(data: &'a [f32], layout: crate::loader::nam_json::WeightsLayout) -> Self {
        Self {
            data,
            pos: 0,
            layout,
        }
    }

    /// Checks whether the weights are in interleaved format (WaveNet v2).
    pub fn is_interleaved4(&self) -> bool {
        self.layout == crate::loader::nam_json::WeightsLayout::Interleaved4WaveNet
    }

    /// Checks whether the weights are in Gate-Major format (LSTM v2).
    pub fn is_gate_major_lstm(&self) -> bool {
        self.layout == crate::loader::nam_json::WeightsLayout::GateMajorLstm
    }

    /// Reads a contiguous slice of `len` weights, advancing the cursor.
    pub(crate) fn read_slice(&mut self, len: usize) -> anyhow::Result<&'a [f32]> {
        let end = self
            .pos
            .checked_add(len)
            .filter(|&e| e <= self.data.len())
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Insufficient weights: required {} starting from position {}, available {}",
                    len,
                    self.pos,
                    self.data.len()
                )
            })?;
        let slice = &self.data[self.pos..end];
        self.pos = end;
        Ok(slice)
    }

    /// Reads a single `f32` scalar, advancing the cursor.
    pub(crate) fn read_f32(&mut self) -> anyhow::Result<f32> {
        let s = self.read_slice(1)?;
        Ok(s[0])
    }

    /// Reads a single `f32` scalar and validates finiteness, advancing the cursor.
    /// Use this for critical scalars such as `head_scale`, `head_bias`, etc.
    pub(crate) fn read_f32_finite(&mut self) -> anyhow::Result<f32> {
        let val = self.read_f32()?;
        if !val.is_finite() {
            bail!(
                "Non-finite f32 scalar at weight position {}: {:e}",
                self.pos.wrapping_sub(1),
                val
            );
        }
        Ok(val)
    }

    /// Verifies that all weights have been consumed. Fails if weights remain.
    pub(crate) fn verify_exhausted(&self) -> anyhow::Result<()> {
        if self.pos != self.data.len() {
            bail!(
                "Model with inconsistent weights: consumed {}, total {}",
                self.pos,
                self.data.len()
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loader::nam_json::WeightsLayout;

    #[test]
    fn read_slice_normal_reads_correct_range() {
        let data = &[0.0f32, 1.0, 2.0, 3.0, 4.0];
        let mut cursor = WeightCursor::new(data, WeightsLayout::Original);
        let slice = cursor.read_slice(3).unwrap();
        assert_eq!(slice, &[0.0, 1.0, 2.0]);
        assert_eq!(cursor.pos, 3);
    }

    #[test]
    fn read_slice_exceeding_data_len_returns_err() {
        let data = &[0.0f32, 1.0, 2.0];
        let mut cursor = WeightCursor::new(data, WeightsLayout::Original);
        let result = cursor.read_slice(10);
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(
            err_msg.contains("Insufficient weights"),
            "Expected 'Insufficient weights' in error message, got: {err_msg}"
        );
    }

    #[test]
    fn read_slice_usize_max_returns_err_no_panic() {
        let data = &[0.0f32, 1.0, 2.0];
        let mut cursor = WeightCursor::new(data, WeightsLayout::Original);
        let result = cursor.read_slice(usize::MAX);
        assert!(result.is_err());
    }

    #[test]
    fn read_slice_exhausts_cursor_pos() {
        let data = &[0.0f32, 1.0, 2.0, 3.0];
        let mut cursor = WeightCursor::new(data, WeightsLayout::Original);
        cursor.read_slice(2).unwrap();
        cursor.read_slice(1).unwrap();
        let slice = cursor.read_slice(1).unwrap();
        assert_eq!(slice, &[3.0]);
        assert_eq!(cursor.pos, 4);
    }

    #[test]
    fn verify_exhausted_passes_when_all_read() {
        let data = &[0.0f32, 1.0, 2.0];
        let mut cursor = WeightCursor::new(data, WeightsLayout::Original);
        cursor.read_slice(3).unwrap();
        assert!(cursor.verify_exhausted().is_ok());
    }

    #[test]
    fn verify_exhausted_fails_when_not_all_read() {
        let data = &[0.0f32, 1.0, 2.0];
        let mut cursor = WeightCursor::new(data, WeightsLayout::Original);
        cursor.read_slice(1).unwrap();
        let result = cursor.verify_exhausted();
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(
            err_msg.contains("inconsistent weights"),
            "Expected 'inconsistent weights' in error message, got: {err_msg}"
        );
    }

    #[test]
    fn read_f32_advances_cursor() {
        let data = &[7.0f32, 8.0, 9.0];
        let mut cursor = WeightCursor::new(data, WeightsLayout::Original);
        let val = cursor.read_f32().unwrap();
        assert_eq!(val, 7.0);
        assert_eq!(cursor.pos, 1);
    }

    #[test]
    fn read_f32_finite_ok_for_finite_values() {
        let data = &[42.0f32];
        let mut cursor = WeightCursor::new(data, WeightsLayout::Original);
        let val = cursor.read_f32_finite().unwrap();
        assert_eq!(val, 42.0);
    }

    #[test]
    fn read_f32_finite_err_for_nan() {
        let data = &[f32::NAN];
        let mut cursor = WeightCursor::new(data, WeightsLayout::Original);
        let result = cursor.read_f32_finite();
        assert!(result.is_err());
        assert!(format!("{}", result.unwrap_err()).contains("Non-finite"));
    }

    #[test]
    fn read_f32_finite_err_for_inf() {
        let data = &[f32::INFINITY];
        let mut cursor = WeightCursor::new(data, WeightsLayout::Original);
        let result = cursor.read_f32_finite();
        assert!(result.is_err());
    }

    #[test]
    fn is_interleaved4_returns_true_for_correct_layout() {
        let data = &[0.0f32];
        let cursor = WeightCursor::new(data, WeightsLayout::Interleaved4WaveNet);
        assert!(cursor.is_interleaved4());
    }

    #[test]
    fn is_interleaved4_returns_false_for_original_layout() {
        let data = &[0.0f32];
        let cursor = WeightCursor::new(data, WeightsLayout::Original);
        assert!(!cursor.is_interleaved4());
    }

    #[test]
    fn is_gate_major_lstm_returns_true_for_correct_layout() {
        let data = &[0.0f32];
        let cursor = WeightCursor::new(data, WeightsLayout::GateMajorLstm);
        assert!(cursor.is_gate_major_lstm());
    }

    #[test]
    fn is_gate_major_lstm_returns_false_for_original_layout() {
        let data = &[0.0f32];
        let cursor = WeightCursor::new(data, WeightsLayout::Original);
        assert!(!cursor.is_gate_major_lstm());
    }
}
