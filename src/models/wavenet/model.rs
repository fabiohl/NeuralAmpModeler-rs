// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::common::WAVENET_MAX_NUM_FRAMES;
use super::layer_array::WaveNetLayerArray;
use crate::math::common::SimdMath;

/// Complete WaveNet Model containing Two Heterogeneous Layer Array Blocks.
///
/// **Scientific Reference:** van den Oord, A., et al. (2016). *"WaveNet: A Generative Model for Raw Audio."* DeepMind.
///
/// `CH` = Array1 channels (layer 0 of the JSON, e.g., 16 for Standard)
/// `K`  = kernel size (always 3)
/// `HEAD` = Array1 head_size = Array2 channels (e.g., 8 for Standard)
///
/// Array2 uses `HEAD` channels and projects to 1 output (`HEAD2=1`),
/// following the C++ pattern: `WaveNetLayerArrayT<CH, 1, 1, HEAD, K, Dilations, true>`.
#[derive(Clone)]
pub struct WaveNetModel<const CH: usize, const K: usize, const HEAD: usize> {
    /// Inner array 01: IN=1, COND=1, CH channels, HEAD outputs, no HeadBias.
    pub array1: WaveNetLayerArray<1, 1, CH, K, HEAD>,
    /// Inner array 02: IN=CH, COND=1, HEAD channels, 1 output, with HeadBias.
    pub array2: WaveNetLayerArray<CH, 1, HEAD, K, 1>,
    /// Final voltage compensation scale (Target Output Scale).
    pub head_scale: f32,
    /// Largest circular buffer required at the Kernel's temporal root.
    pub receptive_field_size: usize,
    /// Whether to execute prewarm during `reset()`. Default: `true`.
    pub prewarm_on_reset: bool,
}

impl<const CH: usize, const K: usize, const HEAD: usize> WaveNetModel<CH, K, HEAD> {
    /// Creates a dedicated exact structural clone of the WaveNet model,
    /// duplicating arrays, weights, topology, and state buffers without invoking
    /// channel slicing.
    #[inline]
    pub fn clone_exact(&self) -> Self {
        self.clone()
    }
    /// Sets the effective number of layers on both arrays for soft-degrade.
    #[inline(always)]
    pub fn set_effective_layers(&mut self, n: usize) {
        self.array1.set_effective_layers(n);
        self.array2.set_effective_layers(n);
    }

    /// Backs up the `buffer_start` pointers of both layer arrays into a slice.
    #[inline(always)]
    pub fn backup_buffer_starts(&self, starts: &mut [usize], offset: &mut usize) {
        for state in &self.array1.states {
            if *offset < starts.len() {
                starts[*offset] = state.buffer_start;
                *offset += 1;
            }
        }
        for state in &self.array2.states {
            if *offset < starts.len() {
                starts[*offset] = state.buffer_start;
                *offset += 1;
            }
        }
    }

    /// Restores the `buffer_start` pointers of both layer arrays from a slice.
    #[inline(always)]
    pub fn restore_buffer_starts(&mut self, starts: &[usize], offset: &mut usize) {
        for state in &mut self.array1.states {
            if *offset < starts.len() {
                state.buffer_start = starts[*offset];
                *offset += 1;
            }
        }
        for state in &mut self.array2.states {
            if *offset < starts.len() {
                state.buffer_start = starts[*offset];
                *offset += 1;
            }
        }
    }

    /// Resolves the full forward pass and produces waveform samples in zero allocation (DSP).
    ///
    /// Combines the outputs of both arrays: `sum(head1) + sum(head2)` × `head_scale`.
    ///
    /// **For Scientists and Devs:** The `dispatch_simd!` macro monomorphizes
    /// this function via the `SimdMath` trait. Inference is strictly `f32`.
    /// Production policy is the AVX2 backend; the AVX-512 match arm is not a
    /// promoted product path (see `docs/architecture.md` §1.2).
    pub fn process(&mut self, input: &[f32], output: &mut [f32]) {
        // SAFETY: `dispatch_simd!` expands to a runtime-CPUID-matched call of
        // `process_internal::<M>`, guaranteeing `M`'s `#[target_feature]` backend is available on
        // this host; `input`/`output` are the same validated slices passed to this safe wrapper.
        unsafe { crate::math::common::dispatch_simd!(self, process_internal, input, output) };
    }

    #[inline(always)]
    /// Fast, generic routine that implements the neural network (WaveNet).
    /// The `<M: SimdMath>` constraint forces the compiler to generate assembly focused on
    /// large registers (256-bit or 512-bit) without branches (branchless).
    ///
    /// The frame count is clamped to `min(input.len(), output.len())` so the
    /// output is never indexed beyond its actual length.
    unsafe fn process_internal<M: SimdMath>(&mut self, input: &[f32], output: &mut [f32]) {
        let total_frames = input.len().min(output.len());
        if total_frames == 0 {
            return;
        }

        let mut pos = 0;
        // [PROCESSING IN CHUNKS (BLOCKS)]
        // To maintain zero-allocation invariants (no temporary RAM vector allocations)
        // and respect the restricted L1/L2 Cache hierarchy, we limit processing
        // to `WAVENET_MAX_NUM_FRAMES` (typically 64 samples) at a time.
        // This loop iterates until it consumes the entire buffer (e.g., 256, 512, 1024 frames).
        while pos < total_frames {
            let num_frames = (total_frames - pos).min(WAVENET_MAX_NUM_FRAMES);
            let in_slice = &input[pos..pos + num_frames];

            // SAFETY: `num_frames <= WAVENET_MAX_NUM_FRAMES` (chunk loop), so the arrays'
            // pre-allocated scratch and ring buffers (sized for `WAVENET_MAX_NUM_FRAMES` frames)
            // cover every access; `in_slice` has exactly `num_frames` frames, and the head/array
            // output slices (`num_frames * HEAD` / `num_frames * CH`) match the array geometry.
            unsafe {
                // [STEP 1: Array1 Forward]
                // Conditioning and Input (1D: 1 channel) -> formatted as blocks of IN frames.
                // In the standard NAM topology, this Array performs convolutions using huge dilations
                // (e.g., from 1 to 512, 1 to 512 successively) to capture amplifier sub-bass.
                // Its output enters `array1.array_outputs` and the skips enter `array1.head_outputs`.
                self.array1
                    .process_block_internal::<M, false>(in_slice, in_slice, num_frames, None);

                // [STEP 2: Array2 Forward (Cascaded Head)]
                // C++ parity: array2 seeds its head_accum with array1's post-head_rechannel
                // output — all layers then accumulate on top of this seed. The final output
                // is head_scale × array2.head_outputs (only the last array's head).
                let array1_head_out = &self.array1.head_outputs[0..num_frames * HEAD];
                let array1_outputs = &self.array1.array_outputs[0..num_frames * CH];
                self.array2.process_block_internal::<M, false>(
                    array1_outputs,
                    in_slice,
                    num_frames,
                    Some(array1_head_out),
                );
            }

            // [STEP 3: Final Scale]
            // C++ reference: output = head_scale × last_array.head_outputs
            let array2_head = &self.array2.head_outputs[0..num_frames];
            let out_slice = &mut output[pos..pos + num_frames];
            // SAFETY: `array2_head` and `out_slice` both have exactly `num_frames` f32s, are
            // non-null/`f32`-aligned, and refer to distinct buffers (`head_outputs` vs `output`,
            // no overlap).
            unsafe {
                core::ptr::copy_nonoverlapping(
                    array2_head.as_ptr(),
                    out_slice.as_mut_ptr(),
                    num_frames,
                );
            }
            // SAFETY: `M` is guaranteed by the runtime CPUID dispatch to support the executed
            // backend's instructions, and `out_slice` has `num_frames` elements as the kernel
            // requires.
            unsafe {
                M::apply_gain(out_slice, self.head_scale);
            }
            pos += num_frames;
        }
    }

    /// Stabilizes the model by processing silence (Zero Input) for pre-warm.
    ///
    /// Dispatches via `dispatch_simd!` (static match, `f32`). Production
    /// policy is AVX2; the AVX-512 arm is not a promoted product path.
    #[cold]
    pub fn prewarm(&mut self) {
        // SAFETY: `dispatch_simd!` dispatches on runtime CPUID feature checks to a matching
        // `#[target_feature]` backend `M`; `prewarm_internal` only touches the model's own
        // pre-allocated buffers.
        unsafe {
            crate::math::common::dispatch_simd!(self, prewarm_internal);
        }
    }

    /// Prewarm strictly optimized for AVX-512 architecture.
    ///
    /// # Safety
    /// Requires a supported processor (AVX-512).
    #[cfg(feature = "avx512")]
    #[target_feature(enable = "avx512f,avx512vl")]
    #[cold]
    pub unsafe fn prewarm_avx512(&mut self) {
        // SAFETY: this body runs under `#[target_feature(enable = "avx512f,avx512vl")]` and the fn
        // is documented to require a supported AVX-512 processor, so the `Avx512Math` kernel
        // instructions are available; `prewarm_internal` only touches the model's own
        // pre-allocated buffers.
        unsafe { self.prewarm_internal::<crate::math::common::Avx512Math>() };
    }

    /// Prewarm strictly optimized for AVX2 architecture.
    ///
    /// # Safety
    /// Requires an x86-64-v3 (AVX2) processor.
    #[cold]
    pub unsafe fn prewarm_avx2(&mut self) {
        // SAFETY: `prewarm_avx2` is documented to require an x86-64-v3 (AVX2) host and
        // `Avx2Math` is selected only on such hosts, so its instructions are available;
        // `prewarm_internal` only touches the model's own pre-allocated buffers.
        unsafe { self.prewarm_internal::<crate::math::common::Avx2Math>() };
    }

    /// # Safety
    /// Call this via `dispatch_simd!` macro only.
    #[cold]
    unsafe fn prewarm_internal<M: SimdMath>(&mut self) {
        let condition = [0.0f32];
        let layer_inputs_1 = [0.0f32];

        // SAFETY: `array1.prewarm_internal` is an `unsafe fn` only reachable via `dispatch_simd!`
        // (per its docs) with `M`'s features guaranteed by the dispatch; it processes a single
        // frame against the array's own pre-allocated buffers.
        unsafe {
            self.array1
                .prewarm_internal::<M>(&layer_inputs_1, &condition, None);
        }
        let array1_outputs = &self.array1.array_outputs[0..CH];
        let array1_head_out = &self.array1.head_outputs[0..HEAD];
        // SAFETY: same dispatch precondition as the array1 call above; `array1_outputs` and
        // `array1_head_out` are array1's outputs sized `CH`/`HEAD` for one frame, matching
        // array2's `IN=CH`/head geometry.
        unsafe {
            self.array2
                .prewarm_internal::<M>(array1_outputs, &condition, Some(array1_head_out));
        }
    }
}
