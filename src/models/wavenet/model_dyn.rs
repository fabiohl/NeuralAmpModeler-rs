// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::common::WAVENET_MAX_NUM_FRAMES;
use super::layer_array_dyn::WaveNetLayerArrayDyn;
use super::post_stack_head::PostStackHead;
use crate::math::common::{AlignedVec, SimdMath};
use crate::models::{NamModel, StaticModel};

/// Complete WaveNet Model with runtime dimensions.
///
/// Mirrors the const-generic `WaveNetModel<CH, K, HEAD>` but accepts dimensions
/// at runtime, enabling arbitrary geometry loading without compile-time
/// specialization.
///
/// # Performance Characteristics — Dynamic vs. Static Path (2026-06-24)
///
/// The dynamic path has a **structural performance gap** relative to the static
/// const-generic path (`WaveNetModel<CH, K, HEAD>`). This gap was characterized and
/// partially addressed, but a residual overhead remains.
///
/// ## What was measured and evaluated
///
/// Benchmark: `WaveNet_Dynamic_CH5_64samp_48kHz` vs `A2Full_CH8_64samp_48kHz`:
/// - Dynamic CH5: **37.58 µs** (2.82% of RT budget @ 48kHz/64samp)
/// - A2Full CH8:  **27.16 µs** (2.04% of RT budget) — **38% faster** with more channels
///
/// **Root cause of the residual gap:** the dynamic convolution kernels
/// (`conv1d_dyn.rs`, `conv1d_dyn_dual.rs`) cannot use the fused SIMD K-tap accumulation
/// kernel (`dot_product_*x_accumulate`) because the channel count is not known at
/// compile time. Instead, they use runtime-dimensioned scalar accumulators
/// (`[0.0f32; 16]` per block, scalar init and accumulate), which prevents the compiler
/// from generating fully-vectorized code without the LLVM const-propagation that
/// drives the static path.
///
/// **What the optimization fixed:** eliminated the 4KB intermediate tap-copy buffer that was
/// copying each tap's data into a contiguous stack array before processing — that
/// was a +53-57% regression. The fix switched to direct pointer
/// accumulation, recovering most of the loss.
///
/// **What remains unfixed:** the accumulator scalar init + K sequential scalar
/// accumulations across K taps (instead of 1 fused SIMD pass). Addressable via:
///   - Option C: const-generic specializations for CH3/CH5/CH8 at the dispatcher level
///     (similar to how LSTM static profiles work for common hidden sizes)
///
/// ## Decision: not a priority
///
/// The dynamic path (`WaveNetModelDyn`) is a **fallback** for models whose geometry
/// does not match any of the static profiles. In practice, the standard NAM trainers
/// produce models with standard shapes (CH=4/8/12/16 for A1, CH=3/8 for A2), so
/// `WaveNetModelDyn` handles rare edge cases.
///
/// At 37.58 µs / 2.82% of RT budget on x86-64-v3 (AVX2), the dynamic path is
/// still very capable for real-time use. This issue should only be revisited if:
///   - Models with non-standard channel counts become common in the NAM ecosystem
///   - A new A2-Free trainer variant produces CH5/CH6/CH7 models at scale
///   - A user-reported use case requires better dynamic model performance
///
/// ## C++ Parity
///
/// Head scale is the **last** f32 read from the weights file
/// (`model.cpp:623-644`). The f32-native head_rechannel path is preserved
/// (`layer_array.rs:220`) for mixed-precision head projection fidelity.
///
/// ## Array Topology
///
/// The model is composed of `N` dynamically-sized layer arrays chained sequentially:
///   - Array 0: IN=1, COND=condition_size, CH channels, HEAD outputs, no HeadBias
///   - Arrays 1..N-2: IN=CH, COND=condition_size, CH channels, HEAD outputs, no HeadBias
///   - Array N-1: IN=CH, COND=condition_size, HEAD channels, 1 output, with HeadBias
///
/// The `output` of array `i` is used as the `input` of array `i+1`, and the
/// `head_outputs` of array `i` seed the `head_accum` of array `i+1` (cascaded head
/// pattern from C++).
///
/// ## condition_dsp (C++ `_process_condition`)
///
/// When `condition_dsp` is `Some`, the raw audio input is first processed by
/// the nested DSP sub-model before reaching the main layer arrays. The sub-model's
/// output channels replace the raw input as the `condition` parameter for both
/// arrays (layer_inputs remain the raw audio). This mirrors C++ `model.cpp:692-722`.
/// The sub-model is built eagerly during model construction and its `set_weights`
/// and `prewarm` are consumed independently from the main weight stream.
pub struct WaveNetModelDyn {
    /// Internal channel count (e.g., 16 for Standard).
    pub ch: usize,
    /// Kernel size (always 3 for WaveNet).
    pub k: usize,
    /// Head projection size (e.g., 8 for Standard).
    pub head: usize,
    /// Dynamically-sized layer arrays chained sequentially.
    /// - Array 0: IN=1, COND=cond, CH channels, HEAD outputs, no HeadBias
    /// - Middle arrays: IN=CH, COND=cond, CH channels, HEAD outputs, no HeadBias
    /// - Last array: IN=CH, COND=cond, HEAD channels, 1 output, with HeadBias
    pub arrays: Vec<WaveNetLayerArrayDyn>,
    /// Final voltage compensation scale (Target Output Scale).
    pub head_scale: f32,
    /// Largest circular buffer required at the Kernel's temporal root.
    pub receptive_field_size: usize,
    /// Optional nested condition DSP sub-model (C++ `_condition_dsp`).
    ///
    /// Built eagerly during model construction from the `condition_dsp` JSON
    /// object. Its `process()` is called with mono audio input; its multi-channel
    /// output replaces the raw input as the `condition` parameter passed to the
    /// layer arrays. When `None`, the raw input is used as both `layer_inputs`
    /// and `condition` (passthrough, cond≤1).
    pub condition_dsp: Option<Box<StaticModel>>,
    /// Pre-allocated output buffer for condition_dsp processing.
    ///
    /// Size: `cond × WAVENET_MAX_NUM_FRAMES`, where `cond` is the main model's
    /// `condition_size`. This matches the sub-model's `NumOutputChannels()`.
    pub condition_dsp_output: AlignedVec<f32>,
    /// Optional post-stack head sub-object (Conv1D + activation).
    ///
    /// Processes the signal after the last layer array and before `head_scale`.
    /// When `None`, the output of the last array is used directly (standard behavior).
    /// Mirrors the `_Head` structure in NAMCore's `convnet.h:108-118`.
    pub post_stack_head: Option<PostStackHead>,
    /// Scratch buffer for post-stack head output.
    ///
    /// Size: `out_ch × WAVENET_MAX_NUM_FRAMES`, where `out_ch` is the head's
    /// output channel count (defaults to 1 for mono).
    pub head_output_scratch: AlignedVec<f32>,
    /// Whether to execute prewarm during `reset()`. Default: `true`.
    pub prewarm_on_reset: bool,
    /// Whether this dynamic WaveNet model explicitly supports dynamic channel slimming.
    pub slimmable_capable: bool,
    /// Allowed channel breakpoints for slimmable channel slicing.
    ///
    /// When `Some`, contains the valid channel counts for this model in
    /// ascending order (e.g. `[1, 2, 3]` for a CH=3 model with 3 tiers).
    /// `None` means no slimmable breakpoints are defined.
    pub allowed_channels: Option<Vec<usize>>,
    /// Pending slimmable channel target set by `set_slimmable_size`.
    ///
    /// Consumed by `try_slimmable_rebuild_single` before DSP processing.
    /// `None` means no pending change.
    pub pending_slim_channel: Option<usize>,
}

impl Clone for WaveNetModelDyn {
    fn clone(&self) -> Self {
        let condition_dsp = crate::models::clone_condition_dsp(&self.condition_dsp);
        Self {
            ch: self.ch,
            k: self.k,
            head: self.head,
            arrays: self.arrays.clone(),
            head_scale: self.head_scale,
            receptive_field_size: self.receptive_field_size,
            condition_dsp,
            condition_dsp_output: self.condition_dsp_output.clone(),
            post_stack_head: self.post_stack_head.clone(),
            head_output_scratch: self.head_output_scratch.clone(),
            prewarm_on_reset: self.prewarm_on_reset,
            slimmable_capable: self.slimmable_capable,
            allowed_channels: self.allowed_channels.clone(),
            pending_slim_channel: self.pending_slim_channel,
        }
    }
}

impl WaveNetModelDyn {
    /// Returns whether this dynamic WaveNet model was explicitly constructed as slimmable capable.
    #[inline(always)]
    pub fn is_slimmable_capable(&self) -> bool {
        self.slimmable_capable
    }

    /// Sets the effective number of layers on all arrays for soft-degrade.
    #[inline(always)]
    pub fn set_effective_layers(&mut self, n: usize) {
        for array in &mut self.arrays {
            array.set_effective_layers(n);
        }
    }

    /// Backs up the `buffer_start` pointers of all layer arrays into a slice.
    #[inline(always)]
    pub fn backup_buffer_starts(&self, starts: &mut [usize], offset: &mut usize) {
        for array in &self.arrays {
            for state in &array.states {
                if *offset < starts.len() {
                    starts[*offset] = state.buffer_start;
                    *offset += 1;
                }
            }
        }
    }

    /// Restores the `buffer_start` pointers of all layer arrays from a slice.
    #[inline(always)]
    pub fn restore_buffer_starts(&mut self, starts: &[usize], offset: &mut usize) {
        for array in &mut self.arrays {
            for state in &mut array.states {
                if *offset < starts.len() {
                    state.buffer_start = starts[*offset];
                    *offset += 1;
                }
            }
        }
    }

    /// Creates a new `WaveNetModelDyn` with all internal channels reduced to
    /// `new_ch`. Delegates to [`crate::models::slimmable::slice_wavenet_model`].
    ///
    /// This is the entry point for the SPSC GC swap pipeline: the async thread
    /// calls this to produce a lightweight copy that can be atomically swapped
    /// in via `gc_cascade`.
    ///
    /// # Errors
    /// Returns [`crate::models::slimmable::SlicingError`] if `new_ch` is zero,
    /// exceeds the original channel count, or exceeds array minimum channels.
    pub fn slice_channels(
        &self,
        new_ch: usize,
    ) -> Result<Self, crate::models::slimmable::SlicingError> {
        crate::models::slimmable::slice_wavenet_model(self, new_ch)
    }

    /// Creates a dedicated exact structural clone of the WaveNet model,
    /// duplicating arrays, weights, topology, and state buffers without invoking
    /// `slice_wavenet_model` or applying any channel reductions.
    #[inline]
    pub fn clone_exact(&self) -> Self {
        self.clone()
    }

    /// Resolves the full forward pass and produces waveform samples in zero allocation (DSP).
    ///
    /// Combines the outputs of both arrays: sum(head1) + sum(head2) × `head_scale`.
    ///
    /// **For Scientists and Devs:** This is where the performance "magic trick" happens (SIMD Dispatch).
    /// Instead of using slow `if/else` per frame to check the CPU (AVX2 vs AVX-512),
    /// the `dispatch_simd!` macro evaluates the hardware once and "teleports" execution
    /// to a cloned (monomorphized) version of this function strictly optimized for your processor.
    pub fn process(&mut self, input: &[f32], output: &mut [f32]) {
        unsafe { crate::math::common::dispatch_simd!(self, process_internal, input, output) };
    }

    /// Fast, generic routine that implements the neural network (WaveNet).
    /// The `<M: SimdMath>` constraint forces the compiler to generate assembly focused on
    /// large registers (256-bit or 512-bit) without branches (branchless).
    ///
    /// ## Array chaining
    /// Array 0 receives the raw input and condition. Each subsequent array `i`
    /// receives the `array_outputs` of array `i-1` as layer inputs, and the
    /// `head_outputs` of array `i-1` seed its head accumulator (cascaded head
    /// pattern). The final output is the last array's `head_outputs × head_scale`.
    ///
    /// When `condition_dsp` is present, the raw input is first processed by the sub-DSP
    /// to produce multi-channel conditioning. The sub-model's output is used as the
    /// `condition` parameter for all arrays, while the raw input remains as
    /// `layer_inputs` for the first array. This mirrors C++ `model.cpp:737-825`.
    #[inline(always)]
    unsafe fn process_internal<M: SimdMath>(&mut self, input: &[f32], output: &mut [f32]) {
        let total_frames = input.len();
        if total_frames == 0 {
            return;
        }

        let cond = self.arrays[0].cond;
        let mut pos = 0;

        while pos < total_frames {
            let num_frames = (total_frames - pos).min(WAVENET_MAX_NUM_FRAMES);
            let in_slice = &input[pos..pos + num_frames];

            let condition_slice: &[f32] = if let Some(ref mut cond_dsp) = self.condition_dsp {
                let cond_out = &mut self.condition_dsp_output[0..num_frames * cond];
                cond_dsp.process(in_slice, cond_out);
                let dsp_ch = cond_dsp.num_output_channels();
                if dsp_ch > 0 && dsp_ch < cond {
                    for f in (0..num_frames).rev() {
                        let val = cond_out[f];
                        for c in 1..cond {
                            cond_out[f * cond + c] = val;
                        }
                    }
                }
                cond_out
            } else {
                in_slice
            };

            unsafe {
                let num_arrays = self.arrays.len();
                let arrays_ptr = self.arrays.as_mut_ptr();

                // Array 0: layer_inputs = raw input, no prev_head_outputs
                (*arrays_ptr).process_block_internal::<M, false>(
                    in_slice,
                    condition_slice,
                    num_frames,
                    None,
                );

                // Chain arrays 1..N-1
                for i in 1..num_arrays {
                    let prev = &*arrays_ptr.add(i - 1);
                    let curr = &mut *arrays_ptr.add(i);
                    let prev_head_out = &prev.head_outputs[0..num_frames * prev.head];
                    let prev_outputs = &prev.array_outputs[0..num_frames * prev.ch];
                    curr.process_block_internal::<M, false>(
                        prev_outputs,
                        condition_slice,
                        num_frames,
                        Some(prev_head_out),
                    );
                }
            }

            let last = &self.arrays[self.arrays.len() - 1];
            let head_dim = last.head;
            let last_head = &last.head_outputs[0..num_frames * head_dim];

            if let Some(ref mut head_proc) = self.post_stack_head {
                let out_ch = head_proc.out_channels();
                let scratch = &mut self.head_output_scratch[0..num_frames * out_ch];
                unsafe {
                    head_proc.process_block(last_head, scratch, num_frames);
                }
                let out_start = pos * out_ch;
                let out_slice = &mut output[out_start..out_start + num_frames * out_ch];
                unsafe {
                    core::ptr::copy_nonoverlapping(
                        scratch.as_ptr(),
                        out_slice.as_mut_ptr(),
                        num_frames * out_ch,
                    );
                }
                unsafe {
                    M::apply_gain(out_slice, self.head_scale);
                }
            } else {
                let out_start = pos * head_dim;
                let out_slice = &mut output[out_start..out_start + num_frames * head_dim];
                unsafe {
                    core::ptr::copy_nonoverlapping(
                        last_head.as_ptr(),
                        out_slice.as_mut_ptr(),
                        num_frames * head_dim,
                    );
                }
                unsafe {
                    M::apply_gain(out_slice, self.head_scale);
                }
            }
            pos += num_frames;
        }
    }

    /// Stabilizes the model by processing silence (Zero Input) for pre-warm.
    ///
    /// AVX-512 vs AVX2 dispatch is done via `SimdMathConfig::get().is_avx512` —
    /// Relaxed atomic read of a `LazyLock` initialized at startup, without calling
    /// `is_x86_feature_detected!` per invocation (cold-path, but consistent with
    /// the dispatch pattern of the rest of the codebase).
    #[cold]
    pub fn prewarm(&mut self) {
        unsafe {
            crate::math::common::dispatch_simd!(self, prewarm_internal);
        }
    }

    /// Prewarm strictly optimized for AVX-512 architecture.
    ///
    /// # Safety
    /// Requires a supported processor (AVX-512).
    #[target_feature(enable = "avx512f,avx512vl")]
    #[cold]
    pub unsafe fn prewarm_avx512(&mut self) {
        unsafe { self.prewarm_internal::<crate::math::common::Avx512Math>() };
    }

    /// Prewarm strictly optimized for AVX2 architecture.
    ///
    /// # Safety
    /// Requires an x86-64-v3 (AVX2) processor.
    #[cold]
    pub unsafe fn prewarm_avx2(&mut self) {
        unsafe { self.prewarm_internal::<crate::math::common::Avx2Math>() };
    }

    /// # Safety
    /// Call this via `dispatch_simd!` macro only.
    #[inline(always)]
    #[cold]
    unsafe fn prewarm_internal<M: SimdMath>(&mut self) {
        if let Some(ref mut cond_dsp) = self.condition_dsp {
            cond_dsp.prewarm(cond_dsp.prewarm_samples());
        }

        let zero_input = [0.0f32];
        let cond = self.arrays[0].cond;

        let condition: &[f32] = if let Some(ref mut cond_dsp) = self.condition_dsp {
            cond_dsp.process(&zero_input, &mut self.condition_dsp_output[0..cond]);
            let dsp_ch = cond_dsp.num_output_channels();
            if dsp_ch > 0 && dsp_ch < cond {
                let val = self.condition_dsp_output[0];
                for c in 1..cond {
                    self.condition_dsp_output[c] = val;
                }
            }
            &self.condition_dsp_output[0..cond]
        } else {
            &zero_input
        };

        unsafe {
            let num_arrays = self.arrays.len();
            let arrays_ptr = self.arrays.as_mut_ptr();

            // Array 0: layer_inputs = silence, no prev_head_outputs
            (*arrays_ptr).prewarm_internal::<M>(&zero_input, condition, None);

            for i in 1..num_arrays {
                let prev = &*arrays_ptr.add(i - 1);
                let curr = &mut *arrays_ptr.add(i);
                let prev_outputs = &prev.array_outputs[0..prev.ch];
                let prev_head_out = &prev.head_outputs[0..prev.head];
                curr.prewarm_internal::<M>(prev_outputs, condition, Some(prev_head_out));
            }
        }

        if let Some(ref mut head_proc) = self.post_stack_head {
            head_proc.prewarm();
        }
    }
}

use crate::models::slimmable::SlimmableModel;

impl SlimmableModel for WaveNetModelDyn {
    fn set_slimmable_size(
        &mut self,
        val: f32,
        _rt_status: Option<&crate::common::spsc::RtStatusFlags>,
    ) {
        let Some(ref allowed) = self.allowed_channels else {
            return;
        };
        if allowed.is_empty() {
            return;
        }
        let val = val.clamp(0.0, 1.0);
        let num_tiers = allowed.len();
        let tier = ((val * (num_tiers as f32 - 1.0)).round() as usize).min(num_tiers - 1);
        self.pending_slim_channel = Some(allowed[tier]);
    }

    fn slimmable_breakpoints(&self) -> Box<[f64]> {
        let Some(ref allowed) = self.allowed_channels else {
            return Box::new([]);
        };
        if allowed.len() < 2 {
            return Box::new([]);
        }
        let max_ch = *allowed.last().unwrap_or(&1) as f64;
        if max_ch <= 0.0 {
            return Box::new([]);
        }
        allowed[..allowed.len() - 1]
            .iter()
            .map(|&ch| (ch as f64) / max_ch)
            .collect()
    }
}
