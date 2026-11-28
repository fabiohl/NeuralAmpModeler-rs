// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! `.namb` format matrix: version × architecture × layout × geometry.
//!
//! Cells covered here:
//! - v2 + `Original` + every supported architecture (LSTM, WaveNet A1/A2,
//!   Sequential, SlimmableContainer, ConvNet, Linear): encode → decode →
//!   weights/config/rate preserved and audio parity against the source model.
//! - v1 + legacy fallback (no JSON metadata): accepted as WaveNet/Original.
//! - Negative cells with a *recomputed* CRC (unknown layout, layout/arch
//!   mismatch, A2 geometry under Interleaved-4).
//!
//! Optimized layouts (`GateMajorLstm`, `Interleaved4WaveNet`) are exercised by
//! `namb_v2_roundtrip` and `namb_v2_validation`; this file adds the
//! cross-architecture/format cells those suites do not cover.

use neural_amp_modeler_rs::loader::dispatcher::build_model;
use neural_amp_modeler_rs::loader::nam_json::{NamModelData, WeightsLayout, parse_nam_json};
use neural_amp_modeler_rs::loader::namb::{
    FLAG_HAS_CRC32, NambError, NambHeader, crc32_ieee, parse_namb_typed, test_fix_v2_crc,
};
use neural_amp_modeler_rs::loader::namb_encoder::encode_namb;
use neural_amp_modeler_rs::math::activations::ActivationPrecision;
use neural_amp_modeler_rs::models::NamModel;
use std::fs;

use crate::common::PrecisionGuard;
use crate::common::generate_sine_440hz as generate_sine;

fn model_path(filename: &str) -> std::path::PathBuf {
    neural_amp_modeler_rs::testing::fixtures::model_path(filename)
}

fn fixture_data(filename: &str) -> NamModelData {
    let json = fs::read_to_string(model_path(filename)).expect("fixture must be readable");
    parse_nam_json(&json).expect("fixture must parse")
}

fn compute_mse(a: &[f32], b: &[f32]) -> f64 {
    let sum: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = (*x as f64) - (*y as f64);
            d * d
        })
        .sum();
    sum / (a.len() as f64)
}

/// `(fixture, architecture cell)` matrix. Every entry must be a valid `.nam`
/// model in the shipped fixture set.
fn matrix() -> Vec<(&'static str, &'static str)> {
    vec![
        ("BossLSTM-1x16.nam", "LSTM 1x16"),
        ("BossLSTM-2x8.nam", "LSTM 2x8"),
        ("BossWN-standard.nam", "WaveNet A1 Standard"),
        ("BossWN-lite.nam", "WaveNet A1 Lite"),
        ("BossWN-feather.nam", "WaveNet A1 Feather"),
        ("BossWN-nano.nam", "WaveNet A1 Nano"),
        ("wavenet_a1_standard.nam", "WaveNet A1 two-array"),
        (
            "wavenet_a2_full.nam",
            "WaveNet A2 Full (heterogeneous kernels)",
        ),
        (
            "wavenet_a2_lite.nam",
            "WaveNet A2 Lite (heterogeneous kernels)",
        ),
        ("sequential_linear_chain.nam", "Sequential Linear chain"),
        ("sequential_linear_wavenet.nam", "Sequential Linear+WaveNet"),
        ("slimmable_container.nam", "SlimmableContainer"),
        ("convnet_test.nam", "ConvNet"),
        ("linear_2x2_shared.nam", "Linear"),
    ]
}

#[test]
fn test_namb_v2_original_matrix_roundtrip() {
    for (filename, cell) in matrix() {
        let source = fixture_data(filename);
        let encoded = encode_namb(&source, 2, WeightsLayout::Original)
            .unwrap_or_else(|e| panic!("[{cell}] encode must succeed: {e}"));
        let decoded = parse_namb_typed(&encoded)
            .unwrap_or_else(|e| panic!("[{cell}] decode must succeed: {e}"));

        assert_eq!(decoded.weights_layout, WeightsLayout::Original, "[{cell}]");
        assert_eq!(decoded.architecture, source.architecture, "[{cell}]");
        // The header always carries a concrete rate; a source without an
        // explicit rate resolves to the documented 48 kHz default.
        assert_eq!(
            decoded.sample_rate,
            Some(source.sample_rate.unwrap_or(48000.0)),
            "[{cell}]"
        );
        assert_eq!(
            decoded.weights, source.weights,
            "[{cell}] bit-exact weights"
        );
        assert_eq!(
            decoded.config.layers.len(),
            source.config.layers.len(),
            "[{cell}] layer count"
        );
    }
}

#[test]
fn test_namb_v2_original_matrix_audio_parity() {
    let _guard = PrecisionGuard::new(ActivationPrecision::Standard);
    let input = generate_sine(256);

    for (filename, cell) in matrix() {
        let source = fixture_data(filename);
        let encoded = encode_namb(&source, 2, WeightsLayout::Original)
            .unwrap_or_else(|e| panic!("[{cell}] encode must succeed: {e}"));
        let decoded = parse_namb_typed(&encoded)
            .unwrap_or_else(|e| panic!("[{cell}] decode must succeed: {e}"));

        let mut model_src =
            build_model(&source).unwrap_or_else(|e| panic!("[{cell}] source build: {e}"));
        let mut model_dec =
            build_model(&decoded).unwrap_or_else(|e| panic!("[{cell}] decoded build: {e}"));
        model_src.prewarm(256);
        model_dec.prewarm(256);

        let mut out_src = vec![0.0f32; input.len()];
        let mut out_dec = vec![0.0f32; input.len()];
        model_src.process(&input, &mut out_src);
        model_dec.process(&input, &mut out_dec);

        let mse = compute_mse(&out_src, &out_dec);
        // Measured: identical weight/config round-trips are bit-exact (MSE = 0);
        // the 1e-12 guard only absorbs a non-deterministic activation-precision flip.
        assert!(mse < 1e-12, "[{cell}] audio divergence: MSE={mse:e}");
    }
}

/// Builds a v1 file with the given weights and a valid v1 CRC (weights only).
fn build_v1(weights: &[f32]) -> Vec<u8> {
    let header_size = std::mem::size_of::<NambHeader>();
    let mut data = vec![0u8; header_size + weights.len() * 4];
    // SAFETY: the buffer is at least `size_of::<NambHeader>()` bytes and is
    // aligned for `NambHeader`; it outlives the borrow.
    let header = unsafe { &mut *data.as_mut_ptr().cast::<NambHeader>() };
    header.magic = 0x4E414D42;
    header.version = 1;
    header.weights_offset = header_size as u32;
    header.sample_rate = 48000.0;
    header.input_level_dbu = 12.0;
    header.output_level_dbu = -6.0;
    for (i, &w) in weights.iter().enumerate() {
        let at = header_size + i * 4;
        data[at..at + 4].copy_from_slice(&w.to_le_bytes());
    }
    header.crc32 = crc32_ieee(&data[header_size..]);
    data
}

#[test]
fn test_namb_v1_fallback_matrix_cell() {
    // v1 has no embedded JSON: the decoder must synthesize the documented
    // WaveNet Standard fallback. A real v1 corpus is not available locally, so
    // audio/oracle comparison for legacy files remains a GAP (see docs).
    let data = build_v1(&[0.0, 0.1, -0.1, 0.2]);
    let parsed = parse_namb_typed(&data).expect("v1 fallback must parse");
    assert_eq!(parsed.architecture, "WaveNet");
    assert_eq!(parsed.weights_layout, WeightsLayout::Original);
    assert_eq!(parsed.config.layers.len(), 2);
    assert_eq!(parsed.sample_rate, Some(48000.0));
}

#[test]
fn test_namb_negative_matrix_with_recomputed_crc() {
    let source = fixture_data("BossWN-nano.nam");

    // Unknown layout byte.
    let mut unknown_layout = encode_namb(&source, 2, WeightsLayout::Original).expect("encode");
    unknown_layout[6] = 7;
    test_fix_v2_crc(&mut unknown_layout);
    let err = parse_namb_typed(&unknown_layout).expect_err("unknown layout must fail");
    assert!(matches!(
        err,
        NambError::UnsupportedLayout { layout_type: 7, .. }
    ));

    // Layout/architecture mismatch: WaveNet weights declared as GateMajorLstm.
    let mut arch_mismatch = encode_namb(&source, 2, WeightsLayout::Original).expect("encode");
    arch_mismatch[6] = WeightsLayout::GateMajorLstm as u8;
    test_fix_v2_crc(&mut arch_mismatch);
    let err = parse_namb_typed(&arch_mismatch).expect_err("layout/arch mismatch must fail");
    assert!(matches!(
        err,
        NambError::LayoutArchitectureMismatch {
            layout: "GateMajorLstm",
            ..
        }
    ));

    // A2 geometry declared as Interleaved-4: canonical kernel_sizes form.
    let a2 = fixture_data("wavenet_a2_full.nam");
    let mut a2_mismatch = encode_namb(&a2, 2, WeightsLayout::Original).expect("encode A2");
    a2_mismatch[6] = WeightsLayout::Interleaved4WaveNet as u8;
    test_fix_v2_crc(&mut a2_mismatch);
    let err = parse_namb_typed(&a2_mismatch).expect_err("A2 under Interleaved-4 must fail");
    assert!(matches!(
        err,
        NambError::LayoutGeometryMismatch {
            layout: "Interleaved4WaveNet",
            ..
        }
    ));

    // Scalar-form A2-Dynamic (`kernel_size` + dilations, no `kernel_sizes`)
    // must fail closed too: the A2 engine ignores `weights_layout`.
    let a2_scalar = fixture_data("wavenet_a2_max.nam");
    assert!(a2_scalar.config.layers[0].kernel_sizes.is_none());
    let mut scalar_mismatch =
        encode_namb(&a2_scalar, 2, WeightsLayout::Original).expect("encode scalar A2");
    scalar_mismatch[6] = WeightsLayout::Interleaved4WaveNet as u8;
    test_fix_v2_crc(&mut scalar_mismatch);
    let err =
        parse_namb_typed(&scalar_mismatch).expect_err("scalar A2 under Interleaved-4 must fail");
    assert!(matches!(
        err,
        NambError::LayoutGeometryMismatch {
            layout: "Interleaved4WaveNet",
            ..
        }
    ));

    // CRC still wins over metadata interpretation when the CRC is stale.
    let mut stale = encode_namb(&source, 2, WeightsLayout::Original).expect("encode");
    stale[std::mem::size_of::<NambHeader>()] ^= 0xFF;
    let err = parse_namb_typed(&stale).expect_err("stale CRC must fail");
    assert!(matches!(err, NambError::CrcMismatch { .. }));

    // And a valid v2 header must declare FLAG_HAS_CRC32.
    let mut no_flag = encode_namb(&source, 2, WeightsLayout::Original).expect("encode");
    no_flag[7] &= !FLAG_HAS_CRC32;
    test_fix_v2_crc(&mut no_flag);
    let err = parse_namb_typed(&no_flag).expect_err("missing CRC flag must fail");
    assert!(matches!(err, NambError::CrcMissing { version: 2 }));
}

#[test]
fn test_namb_reencode_same_layout_does_not_double_transpose() {
    // The parser returns weights in the *declared* layout. Re-encoding those
    // weights into the same layout must be a pure copy: transposing again
    // would corrupt an already-optimized tensor.
    for (filename, layout) in [
        ("BossLSTM-1x16.nam", WeightsLayout::GateMajorLstm),
        ("BossWN-standard.nam", WeightsLayout::Interleaved4WaveNet),
    ] {
        let source = fixture_data(filename);
        let optimized = encode_namb(&source, 2, layout)
            .unwrap_or_else(|e| panic!("[{filename}] forward encode: {e}"));
        let decoded =
            parse_namb_typed(&optimized).unwrap_or_else(|e| panic!("[{filename}] decode: {e}"));
        assert_eq!(decoded.weights_layout, layout, "[{filename}]");

        let reencoded = encode_namb(&decoded, 2, layout)
            .unwrap_or_else(|e| panic!("[{filename}] same-layout re-encode: {e}"));
        let again =
            parse_namb_typed(&reencoded).unwrap_or_else(|e| panic!("[{filename}] re-decode: {e}"));
        assert_eq!(
            again.weights, decoded.weights,
            "[{filename}] same-layout re-encode must copy bit-exactly"
        );
        assert_eq!(again.weights_layout, layout, "[{filename}]");
    }
}

#[test]
fn test_namb_reencode_optimized_source_without_inverse_is_rejected() {
    // Only forward transposition (Original -> optimized) is implemented.
    // Optimized -> different target has no inverse and must fail before any
    // bytes are emitted, never silently relabel weights.
    let source = fixture_data("BossLSTM-1x16.nam");
    let optimized = encode_namb(&source, 2, WeightsLayout::GateMajorLstm).expect("encode");
    let decoded = parse_namb_typed(&optimized).expect("decode");
    assert_eq!(decoded.weights_layout, WeightsLayout::GateMajorLstm);

    let err = encode_namb(&decoded, 2, WeightsLayout::Original)
        .expect_err("optimized -> Original must be rejected");
    let msg = err.to_string();
    assert!(
        msg.contains("no inverse layout"),
        "rejection must explain the missing inverse, got: {msg}"
    );
    assert!(
        encode_namb(&decoded, 2, WeightsLayout::Interleaved4WaveNet).is_err(),
        "optimized -> different optimized layout must be rejected"
    );
}

#[test]
fn test_namb_v1_empty_weight_stream_is_rejected_by_encoder() {
    // CRC32(empty) == 0 is the v1 parser's missing-checksum sentinel. The
    // encoder must refuse to emit a file its own reader rejects.
    let mut data = fixture_data("BossWN-nano.nam");
    data.weights.clear();
    let err = encode_namb(&data, 1, WeightsLayout::Original)
        .expect_err("v1 empty weights must be rejected");
    assert!(
        err.to_string().contains("empty weight stream"),
        "unexpected rejection message: {err}"
    );
}
