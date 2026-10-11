// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Zero-allocation boundary adapter between planar host buffers and the
//! engine's interleaved internal convention.
//!
//! Multi-output families (WaveNet A1/A2 with `head_size > 1`, ConvNet with
//! `out_channels > 1`, LSTM multi-output) write their samples
//! **interleaved frame-major** in the mono `process(input, output)` entry
//! point, while the raw-pointer contract hands over one plane per channel.
//! [`run_staged`] reconciles the two without a heap allocation and, at the
//! same time, removes the exact-aliasing undefined behaviour that previous
//! boundary code had when `input == output`.
//!
//! # Aliasing rules
//!
//! - **Exact alias (`input == outputs[0]`) is supported.** The input is copied
//!   into a stack staging buffer before any output is written, so no
//!   `&[f32]`/`&mut [f32]` pair is ever created over the same memory.
//! - **Partial overlap is a contract violation** (e.g. an input plane offset
//!   by a few samples into a destination plane). Staging cannot undo a shifted
//!   aliasing, so it is detected in debug builds and documented here.

/// Staging block length in `f32` samples: each of the two scratch buffers
/// holds `STAGE_SAMPLES` elements. The block is `STAGE_SAMPLES / out_ch`
/// frames, so the interleaved output fits alongside the input copy.
// Measured: 4 KiB per buffer (2 × 1024 f32), L1-resident on x86-64
// (32 KiB L1D). The staged loop is chunk-bound by `STAGE_SAMPLES / max_ch`
// frames, so a 1→2 staged block moves 512 frames per closure call — well
// above the 64-frame WAVENET_MAX_NUM_FRAMES engine block and below the
// 16 KiB stack budget enforced by the heap-audit (16 KB stack limit).
// The mono fast path (`run_staged_io` callers bypass staging when
// `in_ch == 1 && out_ch == 1` without alias) keeps conv1d_hotpath-class
// hot loops untouched; a dedicated staged-path bench is tracked alongside
// `benches/inference` multi-output geometries.
pub(crate) const STAGE_SAMPLES: usize = 1024;

/// Runs `f(in_chunk, out_interleaved_chunk)` over `frames` frames, feeding
/// `chunk` frames at a time, and deinterleaves the result into the per-channel
/// output planes.
///
/// # Safety
///
/// - `input` must point to at least `frames` readable `f32` samples.
/// - `outputs` must contain `out_ch` valid, writable `*mut f32` pointers, each
///   covering at least `frames` samples.
/// - `input` may be exactly equal to `outputs[0]` (exact in-place alias is
///   supported). Any other overlap between `input` and an output plane
///   (partial/shifted alias) violates this contract; the behaviour is
///   undefined and only detected by a `debug_assert!`.
/// - `f` must write exactly `n * out_ch` interleaved samples into its output
///   slice and must not allocate, lock, or perform I/O.
pub(crate) unsafe fn run_staged(
    input: *const f32,
    outputs: &[*mut f32],
    frames: usize,
    out_ch: usize,
    f: impl FnMut(&[f32], &mut [f32]),
) {
    // SAFETY: forwarded with mono-in plane.
    unsafe {
        run_staged_io(&[input], outputs, frames, 1, out_ch, f);
    }
}

/// Runs `f(in_interleaved_chunk, out_interleaved_chunk)` over `frames` frames,
/// reading from `inputs` planar channels and deinterleaving to `outputs` planar channels.
///
/// Staged with zero heap allocations using stack scratch buffers, eliminating
/// exact-aliasing hazards even when any input plane aliases an output plane.
///
/// # Safety
///
/// - `inputs` must contain `in_ch` valid `*const f32` pointers, each covering `frames` samples.
/// - `outputs` must contain `out_ch` valid `*mut f32` pointers, each covering `frames` samples.
/// - Exact in-place aliasing is supported. Partial/shifted overlap is undefined behaviour.
/// - `f` must not allocate, lock, or perform I/O.
pub(crate) unsafe fn run_staged_io(
    inputs: &[*const f32],
    outputs: &[*mut f32],
    frames: usize,
    in_ch: usize,
    out_ch: usize,
    mut f: impl FnMut(&[f32], &mut [f32]),
) {
    if frames == 0 || in_ch == 0 || out_ch == 0 || inputs.is_empty() || outputs.is_empty() {
        return;
    }
    debug_assert!(
        in_ch <= STAGE_SAMPLES,
        "run_staged_io: in_ch ({in_ch}) exceeds staging capacity ({STAGE_SAMPLES})"
    );
    debug_assert!(
        out_ch <= STAGE_SAMPLES,
        "run_staged_io: out_ch ({out_ch}) exceeds staging capacity ({STAGE_SAMPLES})"
    );

    // Partial/shifted overlap between an input plane and an output plane is a
    // contract violation: staging only covers the exact-alias case. Detect it
    // in debug builds before any read/write so shifted aliasing fails loudly
    // instead of silently corrupting the staged copy.
    debug_assert!(
        {
            let mut ok = true;
            for &inp in inputs.iter().take(in_ch.min(inputs.len())) {
                let in_start = inp as usize;
                let in_end = in_start.wrapping_add(frames * size_of::<f32>());
                for &outp in outputs.iter().take(out_ch.min(outputs.len())) {
                    let out_start = outp as usize;
                    let out_end = out_start.wrapping_add(frames * size_of::<f32>());
                    let overlaps = in_start < out_end && out_start < in_end;
                    let exact = core::ptr::eq(inp, outp as *const f32);
                    if overlaps && !exact {
                        ok = false;
                        break;
                    }
                }
                if !ok {
                    break;
                }
            }
            ok
        },
        "run_staged_io: partial/shifted input↔output overlap is a contract violation (only exact alias is supported)"
    );

    let max_ch = in_ch.max(out_ch);
    let chunk = (STAGE_SAMPLES / max_ch).max(1);

    let mut in_buf = [0.0f32; STAGE_SAMPLES];
    let mut out_buf = [0.0f32; STAGE_SAMPLES];
    let mut done = 0usize;

    while done < frames {
        let n = chunk.min(frames - done);

        // Copy and interleave input frames into in_buf on the stack before writing any output.
        if in_ch == 1 {
            // SAFETY: caller guarantees inputs[0] covers at least done + n samples.
            unsafe {
                core::ptr::copy_nonoverlapping(inputs[0].add(done), in_buf.as_mut_ptr(), n);
            }
        } else {
            for i in 0..n {
                for c in 0..in_ch {
                    // SAFETY: caller guarantees each input plane covers frames samples.
                    unsafe {
                        in_buf[i * in_ch + c] = *inputs[c].add(done + i);
                    }
                }
            }
        }

        f(&in_buf[..n * in_ch], &mut out_buf[..n * out_ch]);

        // Deinterleave out_buf to output planes.
        for c in 0..out_ch {
            let dst = outputs[c];
            for i in 0..n {
                // SAFETY: caller guarantees each output plane covers frames samples.
                unsafe {
                    *dst.add(done + i) = out_buf[i * out_ch + c];
                }
            }
        }

        done += n;
    }
}
