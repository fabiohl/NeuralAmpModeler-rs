// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! A2 layer topology resolution and validation before dynamic allocation.

use crate::loader::nam_json::model::NamLayerConfig;
use crate::loader::nam_json::validation::{MAX_DILATION, MAX_DILATIONS_PER_ARRAY, MAX_KERNEL_SIZE};
use crate::models::a2::params::{A2_DILATIONS, A2_KERNEL_SIZES};

/// Resolved A2 layer-array topology vectors, mirroring the resolution performed
/// by `build_wavenet_a2_dynamic` (scalar `kernel_size`, plural `kernel_sizes`,
/// or the canonical A2 constant table).
#[derive(Debug, Clone, PartialEq)]
pub struct A2TopologyVectors {
    /// Effective per-layer kernel sizes.
    pub kernel_sizes: Vec<usize>,
    /// Effective per-layer dilations.
    pub dilations: Vec<usize>,
    /// Number of layers (`kernel_sizes.len()`).
    pub num_layers: usize,
    /// Head convolution kernel size (`layer_raw.head.kernel_size`, default 1).
    pub head_kernel_size: usize,
}

/// Validates the A2 layer-array topology *before any allocation*.
///
/// Fail-closed hardening:
/// - F-02: diverging `kernel_sizes`/`dilations`/activation-array lengths are
///   rejected here instead of panicking on `assert_eq!` in `WaveNetA2Dyn::new`
///   or silently falling back to default activations.
/// - F-03: zero kernel sizes are rejected (would underflow `kernel - 1`).
/// - F-04: unbounded layer counts are rejected (each layer allocates a
///   page-mapped `MirroredBuffer`; a hostile 100k-element array would exhaust
///   VMAs/mmap).
pub fn validate_a2_layer_topology(layer: &NamLayerConfig) -> Result<A2TopologyVectors, String> {
    let kernel_sizes = if let Some(ks) = layer.kernel_sizes.clone() {
        ks
    } else if let Some(ks_scalar) = layer.kernel_size {
        let dils = layer
            .dilations
            .clone()
            .unwrap_or_else(|| A2_DILATIONS.to_vec());
        vec![ks_scalar; dils.len()]
    } else {
        A2_KERNEL_SIZES.to_vec()
    };
    let dilations = layer
        .dilations
        .clone()
        .unwrap_or_else(|| A2_DILATIONS.to_vec());
    let num_layers = kernel_sizes.len();

    if dilations.len() != num_layers {
        return Err(format!(
            "A2 layer-array kernel_sizes length ({num_layers}) != dilations length ({}) — \
             topology divergence rejected before allocation (F-02)",
            dilations.len()
        ));
    }
    if num_layers == 0 {
        return Err("A2 layer-array has zero layers".to_string());
    }
    if num_layers > MAX_DILATIONS_PER_ARRAY {
        return Err(format!(
            "A2 layer-array has {num_layers} layers, exceeding maximum {} — \
             DoS/OOM protection (F-04)",
            MAX_DILATIONS_PER_ARRAY
        ));
    }
    for (i, &k) in kernel_sizes.iter().enumerate() {
        if k == 0 {
            return Err(format!(
                "A2 layer-array kernel_sizes[{i}] is 0 — must be >= 1 (F-03)"
            ));
        }
        if k > MAX_KERNEL_SIZE {
            return Err(format!(
                "A2 layer-array kernel_sizes[{i}] ({k}) exceeds maximum {MAX_KERNEL_SIZE}"
            ));
        }
    }
    for (i, &d) in dilations.iter().enumerate() {
        if d == 0 {
            return Err(format!("A2 layer-array dilations[{i}] is 0 — must be >= 1"));
        }
        if d > MAX_DILATION {
            return Err(format!(
                "A2 layer-array dilations[{i}] ({d}) exceeds maximum {MAX_DILATION}"
            ));
        }
    }

    let head_kernel_size = layer
        .head_kernel_size
        .or_else(|| {
            layer
                .layer_raw
                .as_ref()
                .and_then(|raw| raw.get("head"))
                .and_then(|h| h.get("kernel_size"))
                .and_then(|k| k.as_u64())
                .map(|k| k as usize)
        })
        .unwrap_or(1);
    if head_kernel_size == 0 {
        return Err("A2 layer-array head kernel_size must be >= 1".to_string());
    }
    if head_kernel_size > MAX_KERNEL_SIZE {
        return Err(format!(
            "A2 layer-array head kernel_size ({head_kernel_size}) exceeds maximum {MAX_KERNEL_SIZE}"
        ));
    }

    // C++ model.cpp:1228-1232: layer1x1_post_film cannot be active when layer1x1.active is false
    let layer1x1_active = layer
        .layer_raw
        .as_ref()
        .and_then(|raw| raw.get("layer1x1"))
        .and_then(|l| l.get("active"))
        .and_then(|a| a.as_bool())
        .unwrap_or(false);

    let layer1x1_post_film_active = layer
        .layer_raw
        .as_ref()
        .and_then(|raw| raw.get("layer1x1_post_film"))
        .and_then(|l| l.get("active"))
        .and_then(|a| a.as_bool())
        .unwrap_or(false);

    if layer1x1_post_film_active && !layer1x1_active {
        return Err(
            "layer1x1_post_film cannot be active when layer1x1.active is false".to_string(),
        );
    }

    // F-02: when activation/gating/secondary vectors are present as arrays,
    // their length must match num_layers — no silent fallback on divergence.
    if let Some(ref raw) = layer.layer_raw {
        for key in ["activation", "gating_mode", "secondary_activation"] {
            if let Some(arr) = raw.get(key).and_then(|v| v.as_array())
                && arr.len() != num_layers
            {
                return Err(format!(
                    "A2 layer-array `{key}` array length ({}) != num_layers ({num_layers}) — \
                     topology divergence rejected before allocation (F-02)",
                    arr.len()
                ));
            }
        }
    }

    Ok(A2TopologyVectors {
        kernel_sizes,
        dilations,
        num_layers,
        head_kernel_size,
    })
}
