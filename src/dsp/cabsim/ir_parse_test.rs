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
    // F-20: 10,000 zero-sized junk chunks must abort the scan at
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
    // F-20: the cap aborts exactly after MAX_CHUNKS_SCANNED chunks.
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
    // F-20: a valid WAV with 100 junk chunks (well within the cap)
    // must parse normally — the guard must not reject legitimate files.
    let wav = build_synthetic_wav(100);
    let (samples, rate) = parse_wav(&wav).expect("valid WAV with junk must parse");
    assert_eq!(rate, 48000);
    assert_eq!(samples.len(), 8);
}

#[test]
fn scan_cap_boundary_exact() {
    // F-20: each chunk scan (fmt, then data) restarts the counter.
    // A file with N junk chunks requires N + 2 iterations to find `data`
    // (junk × N, fmt, data), so N = MAX - 2 is the largest parseable junk
    // count; N = MAX - 1 finds `fmt` but aborts before `data`.
    let below = build_synthetic_wav(MAX_CHUNKS_SCANNED - 2);
    assert!(parse_wav(&below).is_ok());

    let at_cap = build_synthetic_wav(MAX_CHUNKS_SCANNED - 1);
    let err = parse_wav(&at_cap).expect_err("scan cap must abort before data");
    assert!(err.to_string().contains("not found"), "unexpected: {err}");
}

#[test]
fn parse_wav_rejects_sub_12_bytes_without_panic() {
    // R-17: 0 to 11 bytes must return typed InvalidData without panicking on header slice indexing.
    let full = build_synthetic_wav(0);
    for len in 0..12 {
        let truncated = &full[..len];
        let res = parse_wav(truncated);
        assert!(
            res.is_err(),
            "buffer of length {len} must be rejected, not parsed"
        );
        let err = res.unwrap_err();
        assert_eq!(
            err.kind(),
            io::ErrorKind::InvalidData,
            "error kind must be InvalidData for length {len}"
        );
    }
}

#[test]
fn parse_wav_rejects_truncated_headers() {
    // R-17: headers truncated between 12 and WAV_HEADER_MIN (44 bytes) must return InvalidData.
    let full = build_synthetic_wav(0);
    for len in 12..WAV_HEADER_MIN {
        let truncated = &full[..len];
        let res = parse_wav(truncated);
        assert!(
            res.is_err(),
            "truncated header of length {len} must be rejected"
        );
        let err = res.unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }
}

/// Mock reader that yields pre-configured chunks to simulate deterministic stream mutation.
struct MockStream {
    chunks: Vec<Vec<u8>>,
    idx: usize,
}

impl std::io::Read for MockStream {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if self.idx >= self.chunks.len() {
            return Ok(0);
        }
        let chunk = &self.chunks[self.idx];
        self.idx += 1;
        let to_copy = chunk.len().min(buf.len());
        buf[..to_copy].copy_from_slice(&chunk[..to_copy]);
        Ok(to_copy)
    }
}

#[test]
fn read_ir_reader_detects_controlled_stream_truncation() {
    // R-17: hint specifies 1000 bytes, but reader truncates after 20 bytes (less than WAV_HEADER_MIN).
    let mut stream = MockStream {
        chunks: vec![vec![0u8; 20]],
        idx: 0,
    };
    let res = read_ir_reader(&mut stream, Some(1000));
    assert!(res.is_err(), "truncated stream must be rejected");
    let err = res.unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    assert!(
        err.to_string().contains("too small"),
        "error must report size shortfall: {err}"
    );
}

/// Mock reader that yields an endless stream of bytes to verify bounded read behavior.
struct EndlessStream {
    remaining: usize,
}

impl std::io::Read for EndlessStream {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if self.remaining == 0 {
            return Ok(0);
        }
        let to_fill = buf.len().min(self.remaining);
        buf[..to_fill].fill(0x55);
        self.remaining -= to_fill;
        Ok(to_fill)
    }
}

#[test]
fn read_ir_reader_detects_oversized_stream() {
    // R-17: stream exceeding MAX_IR_FILE_SIZE must abort with typed error, not allocate unbounded memory.
    let mut stream = EndlessStream {
        remaining: (MAX_IR_FILE_SIZE as usize) + 1024,
    };
    let res = read_ir_reader(&mut stream, Some(4096));
    assert!(res.is_err(), "oversized stream must be rejected");
    let err = res.unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    assert!(
        err.to_string().contains("too large"),
        "error must report oversized stream: {err}"
    );
}

#[test]
fn read_ir_reader_valid_wav_roundtrip() {
    let wav = build_synthetic_wav(0);
    let mut cursor = io::Cursor::new(&wav);
    let read_bytes = read_ir_reader(&mut cursor, Some(wav.len()))
        .expect("valid synthetic WAV must read successfully");
    assert_eq!(read_bytes.len(), wav.len());
    let (samples, rate) = parse_wav(&read_bytes).expect("valid WAV must parse");
    assert_eq!(rate, 48000);
    assert_eq!(samples.len(), 8);
}
