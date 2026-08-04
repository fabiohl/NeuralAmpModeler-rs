// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Neural Inference Architectures (Brain Engines) module for NAM-rs.
//!
//! This module contains the acoustic brains of the program: neural networks that have learned how,
//! for example, a real amplifier or pedal distorts and colors a guitar sound.

pub mod a2;
pub mod container;
/// ConvNet feed-forward architecture (F4).
pub mod convnet;
pub mod linear;
pub mod linear_fft;
pub mod lstm;
pub mod slimmable;
pub mod wavenet;

/// NamModel trait implementation for StaticModel (dispatch methods).
mod nam_model;
mod static_model;

// =============================================================================
// Sealed Pattern — Prevents external implementations of NamModel
// =============================================================================

mod sealed {
    pub trait Sealed {}
}

// =============================================================================
// Trait NamModel — Public Contract
// =============================================================================

/// The interface (standard connector) for any neural model (amplifiers, pedals, etc.).
///
/// Sealed via private supertrait — only types within this crate can implement `NamModel`.
pub trait NamModel: Send + Sync + sealed::Sealed {
    /// Invoked by the DSP RT-Thread to process acoustic sample blocks (Float32).
    fn process(&mut self, input: &[f32], output: &mut [f32]);

    /// "Heats up" the virtual tubes of the neural engine (`prewarm`).
    fn prewarm(&mut self, num_samples: usize);

    /// Returns whether prewarm should be executed on `reset()`.
    ///
    /// Default: `true` (prewarm on every reset).
    fn prewarm_on_reset(&self) -> bool {
        true
    }

    /// Sets whether prewarm should be executed on `reset()`.
    ///
    /// Default: no-op (fixed-size models ignore this flag).
    fn set_prewarm_on_reset(&mut self, _val: bool) {}

    /// Resets the model's internal state with a new sample rate and max buffer size.
    ///
    /// The default implementation calls `prewarm(max_buffer_size)` if `prewarm_on_reset()`
    /// returns `true`. Architectures with recurrent state (LSTM) may override this
    /// for a lighter reset (only zero the internal states without reprocessing a full prewarm).
    fn reset(&mut self, _sample_rate: u32, max_buffer_size: usize) -> anyhow::Result<()> {
        if self.prewarm_on_reset() {
            self.prewarm(max_buffer_size);
        }
        Ok(())
    }

    /// Reallocates internal buffers to support the given maximum block size.
    ///
    /// Models with fixed (const-generic) buffer sizes can use the default no-op.
    ///
    /// Default: no-op (suitable for static models and LSTM).
    fn set_max_buffer_size(&mut self, _max_buf: usize) -> anyhow::Result<()> {
        Ok(())
    }

    /// Returns the number of samples needed to fully stabilize the model's internal
    /// state (receptive field / recurrent memory depth).
    ///
    /// Default: `0` (suitable for LSTM, which stabilizes via recurrence).
    /// WaveNet variants override this to return the sum of all array receptive
    /// fields plus any condition_dsp prewarm samples.
    fn prewarm_samples(&self) -> usize {
        0
    }

    /// Returns the breakpoints at which slimmable quality transitions occur.
    ///
    /// Each breakpoint represents a normalized value in `[0.0, 1.0]` where the
    /// model switches to a different submodel or internal quality tier. Hosts
    /// and plugins can use these to map and snap discrete quality parameters.
    ///
    /// Delegates to the inner model when applicable (e.g., `ContainerModel`).
    /// Defaults to an empty vector for models without discrete breakpoints.
    fn slimmable_breakpoints(&self) -> Vec<f64> {
        vec![]
    }
}

/// Wrapper enum for trained model variants.
/// Enables static dispatch of DSP calls to the concrete variant, avoiding vtable overhead.
///
/// Named `StaticModel` because all variants are compile-time-fixed geometries.
/// The legacy "Dynamic" mode (arbitrary geometry at runtime) has been retired.
pub enum StaticModel {
    /// WaveNet Standard (16 channels, kernel 3, dilation 8).
    WavenetStandard(Box<wavenet::WaveNetModel<16, 3, 8>>),
    /// WaveNet Lite (12 channels, kernel 3, dilation 6).
    WavenetLite(Box<wavenet::WaveNetModel<12, 3, 6>>),
    /// WaveNet Feather (8 channels, kernel 3, dilation 4).
    WavenetFeather(Box<wavenet::WaveNetModel<8, 3, 4>>),
    /// WaveNet Nano (4 channels, kernel 3, dilation 2).
    WavenetNano(Box<wavenet::WaveNetModel<4, 3, 2>>),
    /// WaveNet A2 Full (8 channels, real inference).
    WavenetA2Full(Box<a2::WaveNetA2<8>>),
    /// WaveNet A2 Lite (3 channels, real inference).
    WavenetA2Lite(Box<a2::WaveNetA2<3>>),
    /// WaveNet A2 Dynamic (runtime-dimensioned, full topology spectrum).
    WavenetA2Dyn(Box<a2::WaveNetA2Dyn>),
    /// WaveNet A2 Cascade (multi-array chain of Dynamic engines).
    WavenetA2Cascade(Box<a2::WaveNetA2Cascade>),
    /// WaveNet Dynamic (runtime-dimensioned, free geometry).
    WavenetDyn(Box<wavenet::WaveNetModelDyn>),
    /// LSTM 1 Layer × 3 hidden units.
    Lstm1x3(Box<lstm::Lstm1x3>),
    /// LSTM 1 Layer × 8 hidden units.
    Lstm1x8(Box<lstm::Lstm1x8>),
    /// LSTM 1 Layer × 12 hidden units.
    Lstm1x12(Box<lstm::Lstm1x12>),
    /// LSTM 1 Layer × 16 hidden units.
    Lstm1x16(Box<lstm::Lstm1x16>),
    /// LSTM 1 Layer × 24 hidden units.
    Lstm1x24(Box<lstm::Lstm1x24>),
    /// LSTM 2 Layers × 8 hidden units.
    Lstm2x8(Box<lstm::Lstm2x8>),
    /// LSTM 2 Layers × 12 hidden units.
    Lstm2x12(Box<lstm::Lstm2x12>),
    /// LSTM 2 Layers × 16 hidden units.
    Lstm2x16(Box<lstm::Lstm2x16>),
    /// LSTM 1 Layer × 40 hidden units.
    Lstm1x40(Box<lstm::Lstm1x40>),
    /// LSTM 2 Layers × 24 hidden units.
    Lstm2x24(Box<lstm::Lstm2x24>),
    /// LSTM Dynamic — runtime-dimensioned, free geometry (F7 fallback).
    LstmDyn(Box<lstm::LstmModelDyn>),
    /// SlimmableContainer — bundle of submodels selected by quality threshold.
    Container(Box<container::ContainerModel>),
    /// Linear — FIR-based model (dot product of input history with weights + bias).
    Linear(Box<linear::LinearModel>),
    /// ConvNet feed-forward model (F4).
    ConvNet(Box<convnet::ConvNetModel>),
}

impl sealed::Sealed for StaticModel {}

pub(crate) use static_model::clone_condition_dsp;
