// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! WaveNet Module — Dilated Causal Neural Architecture for amplifier and pedal emulation.
//!
//! This module provides WaveNet inference engines optimized for real-time DSP execution
//! using Const Generics for fixed dimensions, eliminating bounds checks and maximizing throughput.
//!
//! ## Sub-modules
//!
//! | Module        | Description                                                             |
//! | ------------- | ----------------------------------------------------------------------- |
//! | `common`      | Fundamental constants and types (`WaveNetLayerState`, `WavenetProcessContext`) |
//! | `conv1d`      | Static causal 1D convolution (`Conv1d`)                                |
//! | `conv1d_dyn`  | Runtime-dimensional causal 1D convolution (`Conv1dDyn`)                 |
//! | `dense`       | Static 1x1 dense layer (`DenseLayer`)                                   |
//! | `dense_dyn`   | Runtime-dimensional 1x1 dense layer (`DenseLayerDyn`)                   |
//! | `layer`       | Static WaveNet layer (`WaveNetLayer`)                                   |
//! | `layer_dyn`   | Runtime-dimensional WaveNet layer (`WaveNetLayerDyn`)                   |
//! | `layer_array` | Static WaveNet layer array (`WaveNetLayerArray`)                        |
//! | `layer_array_dyn` | Runtime-dimensional WaveNet layer array (`WaveNetLayerArrayDyn`)    |
//! | `model`           | Complete static model (`WaveNetModel`)                                  |
//! | `model_dyn`       | Runtime-dimensional model (`WaveNetModelDyn`)                           |
//! | `post_stack_head` | Post-stack head sub-object (Conv1D + activation)                        |

pub mod common;
pub mod conv1d;
pub mod conv1d_dual;
pub mod conv1d_dyn;
/// Dual-frame kernel for Conv1dDyn (Temporal-Tiling dual-frame processing).
pub mod conv1d_dyn_dual;
/// F32-native dot-product helpers (standalone functions).
pub mod conv_input;
/// Static 1x1 dense layer (`DenseLayer<IN, OUT>`).
pub mod dense;
/// Runtime-dimensional 1x1 dense layer (`DenseLayerDyn`).
pub mod dense_dyn;
/// Static WaveNet layer (`WaveNetLayer<COND, CH, K>`).
pub mod layer;
/// Static WaveNet layer array (`WaveNetLayerArray<IN, COND, CH, K, HEAD>`).
pub mod layer_array;
/// Runtime-dimensional WaveNet layer array (`WaveNetLayerArrayDyn`).
pub mod layer_array_dyn;
/// Runtime-dimensional WaveNet layer (`WaveNetLayerDyn`).
pub mod layer_dyn;
/// Static model (`WaveNetModel<CH, K, HEAD>`).
pub mod model;
/// Runtime-dimensional model (`WaveNetModelDyn`).
pub mod model_dyn;
/// Post-stack head sub-object (Conv1D + activation).
pub mod post_stack_head;

use super::NamModel;
use super::sealed;

// =============================================================================
// sealed::Sealed for WaveNet
// =============================================================================

impl<const CH: usize, const K: usize, const HEAD: usize> sealed::Sealed
    for model::WaveNetModel<CH, K, HEAD>
{
}

impl sealed::Sealed for model_dyn::WaveNetModelDyn {}

// =============================================================================
// NamModel for WaveNet (Const Generics)
// =============================================================================

impl<const CH: usize, const K: usize, const HEAD: usize> NamModel
    for model::WaveNetModel<CH, K, HEAD>
{
    fn process(&mut self, input: &[f32], output: &mut [f32]) {
        // Delegates to the inherent WaveNetModel::process method (inherent methods have priority)
        self.process(input, output);
    }

    fn prewarm(&mut self, _num_samples: usize) {
        // WaveNet prewarm is one-shot: fills the receptive field via copy_buffer.
        // C++ runs `model->Prewarm()` without a parameter (unlike LSTM).
        self.prewarm();
    }

    fn prewarm_samples(&self) -> usize {
        self.array1.receptive_field_size + self.array2.receptive_field_size
    }

    fn prewarm_on_reset(&self) -> bool {
        self.prewarm_on_reset
    }

    fn set_prewarm_on_reset(&mut self, val: bool) {
        self.prewarm_on_reset = val;
    }
}

// =============================================================================
// NamModel for WaveNet (Dynamic / Runtime-Dimensional)
// =============================================================================

impl NamModel for model_dyn::WaveNetModelDyn {
    fn process(&mut self, input: &[f32], output: &mut [f32]) {
        self.process(input, output);
    }

    fn prewarm(&mut self, _num_samples: usize) {
        self.prewarm();
    }

    fn prewarm_samples(&self) -> usize {
        let mut rf: usize = self.arrays.iter().map(|a| a.receptive_field_size).sum();
        if let Some(ref cond_dsp) = self.condition_dsp {
            rf += cond_dsp.prewarm_samples();
        }
        if let Some(ref head_proc) = self.post_stack_head {
            rf += head_proc.receptive_field() - 1;
        }
        rf
    }

    fn set_max_buffer_size(&mut self, max_buf: usize) -> anyhow::Result<()> {
        if let Some(ref mut cond_dsp) = self.condition_dsp {
            cond_dsp.set_max_buffer_size(max_buf)?;
        }
        Ok(())
    }

    fn prewarm_on_reset(&self) -> bool {
        self.prewarm_on_reset
    }

    fn set_prewarm_on_reset(&mut self, val: bool) {
        self.prewarm_on_reset = val;
        if let Some(ref mut cond_dsp) = self.condition_dsp {
            cond_dsp.set_prewarm_on_reset(val);
        }
    }
}

// =============================================================================
// Public re-exports
// =============================================================================

pub use common::{
    LAYER_ARRAY_BUFFER_PADDING, MAX_KERNEL, WAVENET_MAX_NUM_FRAMES, WaveNetLayerState,
    WavenetProcessContext,
};
pub use conv1d::Conv1d;
pub use conv1d_dyn::Conv1dDyn;
pub use dense::DenseLayer;
pub use dense_dyn::DenseLayerDyn;
pub use layer::WaveNetLayer;
pub use layer_array::WaveNetLayerArray;
pub use layer_array_dyn::WaveNetLayerArrayDyn;
pub use layer_dyn::WaveNetLayerDyn;
pub use model::WaveNetModel;
pub use model_dyn::WaveNetModelDyn;
pub use post_stack_head::PostStackHead;

#[cfg(test)]
mod wavenet_test;
