// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;
use crate::models::{NamModel, StaticModel};

fn append(bytes: &mut Vec<u8>, value: u32, count: usize) {
    for i in 0..count {
        bytes.push((value >> (8 * i)) as u8);
    }
}

fn chunk(bytes: &mut Vec<u8>, id: &[u8; 4], payload: &[u8]) {
    bytes.extend_from_slice(id);
    append(bytes, payload.len() as u32, 4);
    bytes.extend_from_slice(payload);
    if !payload.len().is_multiple_of(2) {
        bytes.push(0);
    }
}

fn make_wav(format: u32, bits: u16, samples: &[u8], extensible: bool, channels: u16) -> Vec<u8> {
    let mut fmt = Vec::new();
    append(&mut fmt, if extensible { 65534 } else { format }, 2);
    append(&mut fmt, channels as u32, 2);
    append(&mut fmt, 44100, 4);
    append(&mut fmt, 44100 * (channels as u32) * ((bits / 8) as u32), 4);
    append(&mut fmt, (channels * (bits / 8)) as u32, 2);
    append(&mut fmt, bits as u32, 2);
    if extensible {
        append(&mut fmt, 22, 2);
        append(&mut fmt, bits as u32, 2);
        append(&mut fmt, 0, 4);
        append(&mut fmt, format, 4);
        let guid_tail: [u8; 12] = [0, 0, 0x10, 0, 0x80, 0, 0, 0xaa, 0, 0x38, 0x9b, 0x71];
        fmt.extend_from_slice(&guid_tail);
    }
    let mut body = Vec::from(b"WAVE".as_slice());
    chunk(&mut body, b"fmt ", &fmt);
    chunk(&mut body, b"JUNK", &[42]); // Odd-sized unknown chunk requires padding byte
    chunk(&mut body, b"data", samples);
    let mut result = Vec::from(b"RIFF".as_slice());
    append(&mut result, body.len() as u32, 4);
    result.extend_from_slice(&body);
    result
}

fn check_dsp_response(model: &mut StaticModel, channels: usize) {
    assert_eq!(model.in_channels(), 1);
    assert_eq!(model.num_output_channels(), channels);

    let _ = model.reset(44100, 2);

    let input_1 = [1.0f32, 0.0f32];
    let mut output_ch0 = [0.0f32; 2];
    let mut output_ch1 = [0.0f32; 2];

    if channels == 1 {
        model.process_multichannel(&[&input_1], &mut [&mut output_ch0]);
        assert!((output_ch0[0] - 0.5).abs() < 1e-6);
        assert!((output_ch0[1] - (-0.25)).abs() < 1e-6);
    } else {
        model.process_multichannel(&[&input_1], &mut [&mut output_ch0, &mut output_ch1]);
        assert!((output_ch0[0] - 0.5).abs() < 1e-6);
        assert!((output_ch0[1] - (-0.25)).abs() < 1e-6);
        assert!((output_ch1[0] - (-0.5)).abs() < 1e-6);
        assert!((output_ch1[1] - 0.25).abs() < 1e-6);
    }

    let input_2 = [0.0f32, 0.0f32];
    if channels == 1 {
        model.process_multichannel(&[&input_2], &mut [&mut output_ch0]);
        assert!((output_ch0[0] - 0.125).abs() < 1e-6);
        assert!(output_ch0[1].abs() < 1e-6);
    } else {
        model.process_multichannel(&[&input_2], &mut [&mut output_ch0, &mut output_ch1]);
        assert!((output_ch0[0] - 0.125).abs() < 1e-6);
        assert!(output_ch0[1].abs() < 1e-6);
        assert!((output_ch1[0] - (-0.125)).abs() < 1e-6);
        assert!(output_ch1[1].abs() < 1e-6);
    }
}

#[test]
fn test_formats_and_configuration() {
    for format in [1u32, 3u32] {
        for bits in [16u16, 24u16, 32u16] {
            for channels in [1u16, 2u16] {
                for extensible in [false, true] {
                    if format == 3 && bits != 32 {
                        continue;
                    }

                    let mut samples = Vec::new();
                    if format == 3 {
                        for value in [0x3f000000u32, 0xbe800000u32, 0x3e000000u32] {
                            append(&mut samples, value, 4);
                            if channels == 2 {
                                append(&mut samples, value ^ 0x80000000u32, 4);
                            }
                        }
                    } else {
                        let b = bits as i32;
                        let val_0 = 1i32 << (b - 2);
                        let val_1 = -(1i32 << (b - 3));
                        let val_2 = 1i32 << (b - 4);

                        for value in [val_0, val_1, val_2] {
                            append(&mut samples, value as u32, (bits / 8) as usize);
                            if channels == 2 {
                                append(&mut samples, (-value) as u32, (bits / 8) as usize);
                            }
                        }
                    }

                    let wav_bytes = make_wav(format, bits, &samples, extensible, channels);
                    let wav_data = parse_wav_ir(&wav_bytes).expect("Failed to parse valid WAV IR");

                    assert_eq!(wav_data.sample_rate, 44100.0);
                    assert_eq!(wav_data.channels, channels as usize);
                    assert_eq!(wav_data.receptive_field, 3);

                    let mut expected_weights = vec![0.5f32, -0.25f32, 0.125f32];
                    if channels == 2 {
                        expected_weights.extend_from_slice(&[-0.5f32, 0.25f32, -0.125f32]);
                    }
                    assert_eq!(wav_data.weights, expected_weights);

                    let model_data = wav_ir_to_model_data(wav_data);
                    assert_eq!(model_data.architecture, "Linear");
                    assert_eq!(model_data.config.in_channels, Some(1));
                    assert_eq!(model_data.config.out_channels, Some(channels as usize));
                    assert_eq!(model_data.config.receptive_field, Some(3));
                    assert_eq!(model_data.config.bias, Some(false));
                    assert_eq!(model_data.sample_rate, Some(44100.0));

                    let mut model = crate::loader::dispatcher::build_model(&model_data)
                        .expect("Failed to build LinearModel from WAV");
                    check_dsp_response(&mut model, channels as usize);
                }
            }
        }
    }
}

#[test]
fn test_invalid_files() {
    let valid = make_wav(1, 16, &[0, 64, 0, 0], false, 1);

    // Every truncated prefix must fail without reading outside bounds.
    for n in 0..valid.len() {
        assert!(
            parse_wav_ir(&valid[..n]).is_err(),
            "Prefix of length {n} should fail"
        );
    }

    // Empty data chunk
    assert!(parse_wav_ir(&make_wav(1, 16, &[], false, 1)).is_err());
    // Truncated sample byte
    assert!(parse_wav_ir(&make_wav(1, 16, &[1], false, 1)).is_err());
    // Unsupported format code (A-law = 6)
    assert!(parse_wav_ir(&make_wav(6, 16, &[0, 0], false, 1)).is_err());
    // Unsupported 8-bit PCM
    assert!(parse_wav_ir(&make_wav(1, 8, &[0], false, 1)).is_err());
    // Infinity float sample
    assert!(matches!(
        parse_wav_ir(&make_wav(3, 32, &[0, 0, 0x80, 0x7f], false, 1)),
        Err(WavError::NonFiniteSample)
    ));
    // NaN float sample
    assert!(matches!(
        parse_wav_ir(&make_wav(3, 32, &[0, 0, 0xc0, 0x7f], false, 1)),
        Err(WavError::NonFiniteSample)
    ));

    // Zeroing key header offsets
    for offset in [0usize, 8, 20, 22, 24, 28, 32] {
        let mut invalid = valid.clone();
        invalid[offset] = 0;
        assert!(
            parse_wav_ir(&invalid).is_err(),
            "Zeroed offset {offset} must fail"
        );
    }

    // 3 channels (only 1 or 2 supported)
    assert!(matches!(
        parse_wav_ir(&make_wav(1, 16, &[0, 0, 0, 0, 0, 0], false, 3)),
        Err(WavError::UnsupportedChannelCount(3))
    ));
    // Partial stereo frame
    assert!(matches!(
        parse_wav_ir(&make_wav(1, 16, &[0, 0], false, 2)),
        Err(WavError::InvalidDataSize { .. })
    ));
    // Empty stereo data
    assert!(parse_wav_ir(&make_wav(1, 16, &[], false, 2)).is_err());

    // Stereo header with inconsistent mono alignment/byte rate
    let mut stereo_bad_align = valid.clone();
    stereo_bad_align[22] = 2; // Declare 2 channels but keep mono align/byte_rate
    assert!(matches!(
        parse_wav_ir(&stereo_bad_align),
        Err(WavError::InvalidRateOrAlignment { .. })
    ));

    // Corrupted extensible GUID tail
    let mut bad_guid = make_wav(1, 16, &[0, 0], true, 1);
    bad_guid[59] = 0;
    assert!(matches!(
        parse_wav_ir(&bad_guid),
        Err(WavError::UnsupportedExtensibleSubtype)
    ));
}
