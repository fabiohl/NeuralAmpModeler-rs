// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;

/// Builds a synthetic mono PCM16 WAV: `junk_chunks` zero-sized JUNK chunks
/// followed by a valid `fmt ` chunk and an 8-sample `data` chunk.
fn build_synthetic_wav(junk_chunks: usize) -> Vec<u8> {
    let mut data = Vec::new();
    data.extend_from_slice(b"RIFF");
    data.extend_from_slice(&0u32.to_le_bytes()); // patched below
    data.extend_from_slice(b"WAVE");

    for _ in 0..junk_chunks {
        data.extend_from_slice(b"JUNK");
        data.extend_from_slice(&0u32.to_le_bytes()); // size = 0 (junk)
    }

    // fmt chunk: PCM16, mono, 48000 Hz
    data.extend_from_slice(b"fmt ");
    data.extend_from_slice(&16u32.to_le_bytes());
    data.extend_from_slice(&1u16.to_le_bytes()); // PCM
    data.extend_from_slice(&1u16.to_le_bytes()); // mono
    data.extend_from_slice(&48000u32.to_le_bytes());
    data.extend_from_slice(&(48000u32 * 2).to_le_bytes()); // byte rate
    data.extend_from_slice(&2u16.to_le_bytes()); // block align
    data.extend_from_slice(&16u16.to_le_bytes()); // bits per sample

    // data chunk: 8 samples
    data.extend_from_slice(b"data");
    data.extend_from_slice(&16u32.to_le_bytes());
    for _ in 0..8 {
        data.extend_from_slice(&0i16.to_le_bytes());
    }

    let total = (data.len() - 8) as u32;
    data[4..8].copy_from_slice(&total.to_le_bytes());
    data
}

#[test]
fn junk_zero_size_chunk_flood_aborts_scan() {
    // F-20 / T2.6: 10,000 zero-sized junk chunks must abort the scan at
    // MAX_CHUNKS_SCANNED — a typed error, in bounded time, never a hang.
    let wav = build_synthetic_wav(10_000);
    let result = parse_wav(&wav);
    assert!(result.is_err(), "junk-chunk flood must be rejected");
    let msg = result.err().unwrap().to_string();
    assert!(
        msg.contains("fmt ") || msg.contains("not found"),
        "unexpected error: {msg}"
    );
}

#[test]
fn find_chunk_returns_none_after_scan_cap() {
    // F-20 / T2.6: the cap aborts exactly after MAX_CHUNKS_SCANNED chunks.
    let junk_id = *b"JUNK";
    let mut data = Vec::new();
    data.extend_from_slice(b"RIFF");
    data.extend_from_slice(&0u32.to_le_bytes());
    data.extend_from_slice(b"WAVE");
    for _ in 0..(MAX_CHUNKS_SCANNED + 16) {
        data.extend_from_slice(&junk_id);
        data.extend_from_slice(&0u32.to_le_bytes());
    }

    assert!(find_chunk(&data, b"fmt ", 12).is_none());
    assert!(find_chunk(&data, &junk_id, 12).is_some());
}

#[test]
fn junk_chunks_within_cap_still_parse() {
    // F-20 / T2.6: a valid WAV with 100 junk chunks (well within the cap)
    // must parse normally — the guard must not reject legitimate files.
    let wav = build_synthetic_wav(100);
    let (samples, rate) = parse_wav(&wav).expect("valid WAV with junk must parse");
    assert_eq!(rate, 48000);
    assert_eq!(samples.len(), 8);
}

#[test]
fn scan_cap_boundary_exact() {
    // F-20 / T2.6: each chunk scan (fmt, then data) restarts the counter.
    // A file with N junk chunks requires N + 2 iterations to find `data`
    // (junk × N, fmt, data), so N = MAX - 2 is the largest parseable junk
    // count; N = MAX - 1 finds `fmt` but aborts before `data`.
    let below = build_synthetic_wav(MAX_CHUNKS_SCANNED - 2);
    assert!(parse_wav(&below).is_ok());

    let at_cap = build_synthetic_wav(MAX_CHUNKS_SCANNED - 1);
    let err = parse_wav(&at_cap).expect_err("scan cap must abort before data");
    assert!(err.to_string().contains("not found"), "unexpected: {err}");
}
