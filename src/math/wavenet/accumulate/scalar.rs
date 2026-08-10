// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Scalar fallback accumulation and activation kernels for WaveNet.

/// Accumulation of the network "Head".
/// In WaveNet-like architectures, the various layers (blocks) of the network contribute
/// to a common result called "head". This function simply sums those contributions.
pub unsafe fn accumulate_head_fallback(dest: &mut [f32], src: &[f32]) {
    let len = core::cmp::min(dest.len(), src.len());
    for i in 0..len {
        unsafe {
            // Sum the contents of 'src' into 'dest'.
            let acc = *dest.get_unchecked_mut(i) as f64 + *src.get_unchecked(i) as f64;
            *dest.get_unchecked_mut(i) = acc as f32;
        }
    }
}

/// Applies the 'Tanh' activation and accumulates into the main output.
/// Tanh (Hyperbolic Tangent) is a function that "squashes" any number to
/// be between -1.0 and 1.0. It is very common in guitar amplifier modeling.
pub unsafe fn tanh_and_accumulate_block_fallback(head_input: &mut [f32], block: &mut [f32]) {
    let len = head_input.len();
    for i in 0..len {
        let v = block[i];
        let activated = v.tanh(); // Apply the "squashing".
        block[i] = activated; // Save the squashed value in the block.
        let acc = head_input[i] as f64 + activated as f64;
        head_input[i] = acc as f32; // Add the same value to the "head" accumulator.
    }
}

/// Gated Activation + Accumulation.
/// This technique uses two signals (z1 and z2):
/// 1. z1 goes through a Tanh (holds the "information").
/// 2. z2 goes through a Sigmoid (acts as a "volume" or "gate" for z1).
///
/// At the end, we multiply the two. It's as if z2 decides how much of z1 will pass through.
pub unsafe fn gated_activation_and_accumulate_block_fallback(
    head_input: &mut [f32],
    block: &mut [f32],
    ch: usize, // Number of channels.
) {
    let num_frames = head_input.len() / ch;
    for f in 0..num_frames {
        let block_offset = f * 2 * ch;
        let head_offset = f * ch;
        for c in 0..ch {
            // The block contains z1 and z2 side by side.
            let z1 = block[block_offset + c];
            let z2 = block[block_offset + ch + c];

            // Complex activation: tanh(z1) * sigmoid(z2).
            let activated = z1.tanh() * (1.0 / (1.0 + (-z2).exp()));

            block[block_offset + c] = activated;
            let acc = head_input[head_offset + c] as f64 + activated as f64;
            head_input[head_offset + c] = acc as f32;
        }
    }
}

/// Applies the 'Tanh' activation and overwrites the main output.
pub unsafe fn tanh_and_overwrite_block_fallback(head_input: &mut [f32], block: &mut [f32]) {
    let len = head_input.len();
    for i in 0..len {
        let v = block[i];
        let activated = v.tanh();
        block[i] = activated;
        head_input[i] = activated;
    }
}

/// Fused Seed + Tanh + Head Accumulate.
///
/// Computes `head_input[i] = seed[i] + tanh(block[i])`.
/// Equivalent to `copy_from_slice(seed)` followed by `tanh_and_accumulate_block`,
/// fused into a single pass.
pub unsafe fn tanh_and_accumulate_with_seed_fallback(
    head_input: &mut [f32],
    block: &mut [f32],
    seed: &[f32],
) {
    let len = head_input.len();
    for i in 0..len {
        let v = block[i];
        let activated = v.tanh();
        block[i] = activated;
        let acc = seed[i] as f64 + activated as f64;
        head_input[i] = acc as f32;
    }
}

/// Gated Activation + Overwrite.
pub unsafe fn gated_activation_and_overwrite_block_fallback(
    head_input: &mut [f32],
    block: &mut [f32],
    ch: usize,
) {
    let num_frames = head_input.len() / ch;
    for f in 0..num_frames {
        let block_offset = f * 2 * ch;
        let head_offset = f * ch;
        for c in 0..ch {
            let z1 = block[block_offset + c];
            let z2 = block[block_offset + ch + c];
            let activated = z1.tanh() * (1.0 / (1.0 + (-z2).exp()));
            block[block_offset + c] = activated;
            head_input[head_offset + c] = activated;
        }
    }
}

/// Applies the 'ReLU' activation and accumulates into the main output.
pub unsafe fn relu_and_accumulate_block_fallback(head_input: &mut [f32], block: &mut [f32]) {
    let len = head_input.len();
    for i in 0..len {
        let v = block[i];
        let activated = if v > 0.0 { v } else { 0.0 };
        block[i] = activated;
        let acc = head_input[i] as f64 + activated as f64;
        head_input[i] = acc as f32;
    }
}

/// Applies the 'ReLU' activation and overwrites the main output.
pub unsafe fn relu_and_overwrite_block_fallback(head_input: &mut [f32], block: &mut [f32]) {
    let len = head_input.len();
    for i in 0..len {
        let v = block[i];
        let activated = if v > 0.0 { v } else { 0.0 };
        block[i] = activated;
        head_input[i] = activated;
    }
}

/// Fused Seed + ReLU + Head Accumulate.
pub unsafe fn relu_and_accumulate_with_seed_fallback(
    head_input: &mut [f32],
    block: &mut [f32],
    seed: &[f32],
) {
    let len = head_input.len();
    for i in 0..len {
        let v = block[i];
        let activated = if v > 0.0 { v } else { 0.0 };
        block[i] = activated;
        let acc = seed[i] as f64 + activated as f64;
        head_input[i] = acc as f32;
    }
}
