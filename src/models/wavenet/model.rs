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
    /// this function via the `SimdMath` trait, matching the global hardware
    /// configuration once at the call site. The compiler generates branchless
    /// assembly optimized for your processor (AVX2, AVX-512, or AVX-512 VNNI BF16)
    /// without v-table or dynamic dispatch overhead.
    pub fn process(&mut self, input: &[f32], output: &mut [f32]) {
        unsafe { crate::math::common::dispatch_simd!(self, process_internal, input, output) };
    }

    #[inline(always)]
    /// Fast, generic routine that implements the neural network (WaveNet).
    /// The `<M: SimdMath>` constraint forces the compiler to generate assembly focused on
    /// large registers (256-bit or 512-bit) without branches (branchless).
    unsafe fn process_internal<M: SimdMath>(&mut self, input: &[f32], output: &mut [f32]) {
        let total_frames = input.len();
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
            unsafe {
                core::ptr::copy_nonoverlapping(
                    array2_head.as_ptr(),
                    out_slice.as_mut_ptr(),
                    num_frames,
                );
            }
            unsafe {
                M::apply_gain(out_slice, self.head_scale);
            }
            pos += num_frames;
        }
    }

    /// Stabilizes the model by processing silence (Zero Input) for pre-warm.
    ///
    /// Dispatches via `dispatch_simd!` macro — a single static `match` on the
    /// globally detected `SIMD_MATH.instruction_set` — to run the AVX2, AVX-512,
    /// or AVX-512 VNNI BF16 kernel without runtime feature detection per call.
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
        let condition = [0.0f32];
        let layer_inputs_1 = [0.0f32];

        unsafe {
            self.array1
                .prewarm_internal::<M>(&layer_inputs_1, &condition, None);
        }
        let array1_outputs = &self.array1.array_outputs[0..CH];
        let array1_head_out = &self.array1.head_outputs[0..HEAD];
        unsafe {
            self.array2
                .prewarm_internal::<M>(array1_outputs, &condition, Some(array1_head_out));
        }
    }
}
