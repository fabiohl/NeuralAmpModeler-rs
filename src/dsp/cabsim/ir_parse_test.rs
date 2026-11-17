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

/// Builds a mono WAV whose `data` chunk declares `declared_size` bytes while
/// carrying only `payload`. Used to reproduce truncated/partial RIFF payloads
/// without external fixtures.
fn build_wav_with_data(fmt: u16, bits: u16, declared_size: u32, payload: &[u8]) -> Vec<u8> {
    let mut data = Vec::new();
    data.extend_from_slice(b"RIFF");
    data.extend_from_slice(&0u32.to_le_bytes()); // patched below
    data.extend_from_slice(b"WAVE");

    let block_align = (bits / 8).max(1);
    data.extend_from_slice(b"fmt ");
    data.extend_from_slice(&16u32.to_le_bytes());
    data.extend_from_slice(&fmt.to_le_bytes());
    data.extend_from_slice(&1u16.to_le_bytes()); // mono
    data.extend_from_slice(&48000u32.to_le_bytes());
    data.extend_from_slice(&(48000u32 * block_align as u32).to_le_bytes());
    data.extend_from_slice(&block_align.to_le_bytes());
    data.extend_from_slice(&bits.to_le_bytes());

    data.extend_from_slice(b"data");
    data.extend_from_slice(&declared_size.to_le_bytes());
    data.extend_from_slice(payload);

    let total = (data.len() - 8) as u32;
    data[4..8].copy_from_slice(&total.to_le_bytes());
    data
}

/// A `data` chunk that declares more bytes than the file carries must be
/// rejected before any sample conversion, for every supported format. Before
/// the fix the readers silently clamped to the available bytes and returned a
/// shorter IR as if the file were valid.
#[test]
fn truncated_data_payload_rejected_for_all_formats() {
    let cases: [(u16, u16, &[u8], u32); 4] = [
        (WAV_FORMAT_PCM, 16, &[0x00, 0x40], 8),
        (WAV_FORMAT_PCM, 24, &[0x00, 0x00, 0x40], 6),
        (WAV_FORMAT_PCM, 32, &[0x00, 0x00, 0x00, 0x40], 8),
        (WAV_FORMAT_IEEE_FLOAT, 32, &[0x00, 0x00, 0x00, 0x3F], 8),
    ];

    for (fmt, bits, payload, declared) in cases {
        let wav = build_wav_with_data(fmt, bits, declared, payload);
        let err = parse_wav(&wav)
            .expect_err("truncated data payload must not be accepted as a shorter IR");
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(
            err.to_string().contains("data"),
            "error must identify the data chunk: {err}"
        );
    }
}

/// A payload whose declared size is not a whole number of samples (partial
/// trailing sample) must be rejected instead of silently dropping the bytes.
#[test]
fn partial_trailing_sample_rejected_for_all_formats() {
    let cases: [(u16, u16, &[u8]); 4] = [
        (WAV_FORMAT_PCM, 16, &[0x00, 0x40, 0x11]),
        (WAV_FORMAT_PCM, 24, &[0x00, 0x00, 0x40, 0x11]),
        (WAV_FORMAT_PCM, 32, &[0x00, 0x00, 0x00, 0x40, 0x11]),
        (WAV_FORMAT_IEEE_FLOAT, 32, &[0x00, 0x00, 0x00, 0x3F, 0x11]),
    ];

    for (fmt, bits, payload) in cases {
        let wav = build_wav_with_data(fmt, bits, payload.len() as u32, payload);
        let err = parse_wav(&wav).expect_err("partial trailing sample must be rejected");
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(
            err.to_string().contains("multiple") || err.to_string().contains("aligned"),
            "error must report sample alignment: {err}"
        );
    }
}

/// An empty `data` payload remains an error (no audio samples), and must never
/// be reported as a successfully loaded zero-length IR.
#[test]
fn empty_data_payload_rejected() {
    let wav = build_wav_with_data(WAV_FORMAT_PCM, 16, 0, &[]);
    let err = parse_wav(&wav).expect_err("empty data payload must be rejected");
    assert_eq!(err.kind(), io::ErrorKind::InvalidData);
}

/// Valid mono payloads for every supported format keep their exact sample
/// values after the bounds/alignment validation.
#[test]
fn valid_data_payload_preserved_for_all_formats() {
    let cases: [(u16, u16, &[u8]); 4] = [
        (WAV_FORMAT_PCM, 16, &[0x00, 0x40]),
        (WAV_FORMAT_PCM, 24, &[0x00, 0x00, 0x40]),
        (WAV_FORMAT_PCM, 32, &[0x00, 0x00, 0x00, 0x40]),
        (WAV_FORMAT_IEEE_FLOAT, 32, &[0x00, 0x00, 0x00, 0x3F]),
    ];

    for (fmt, bits, payload) in cases {
        let wav = build_wav_with_data(fmt, bits, payload.len() as u32, payload);
        let (samples, rate) = parse_wav(&wav).expect("valid mono payload must parse");
        assert_eq!(rate, 48000);
        assert_eq!(samples.len(), 1, "fmt={fmt}, bits={bits}");
        assert!(
            (samples[0] - 0.5).abs() < 1e-4,
            "fmt={fmt}, bits={bits}: unexpected sample {}",
            samples[0]
        );
    }
}

/// A final odd-sized `data` chunk without the RIFF pad byte is still accepted:
/// the padding is optional at EOF and must not reject otherwise valid audio.
#[test]
fn odd_sized_data_chunk_without_padding_accepted() {
    let payload = [0x00u8, 0x00, 0x40]; // one PCM24 sample = 3 bytes
    let wav = build_wav_with_data(WAV_FORMAT_PCM, 24, payload.len() as u32, &payload);
    let (samples, rate) = parse_wav(&wav).expect("unpadded final data chunk must parse");
    assert_eq!(rate, 48000);
    assert_eq!(samples.len(), 1);
}

/// A `fmt ` chunk that declares more bytes than the file carries must be
/// rejected instead of reading a partially materialized header.
#[test]
fn fmt_chunk_declared_larger_than_file_rejected() {
    let mut wav = build_wav_with_data(WAV_FORMAT_PCM, 16, 2, &[0x00, 0x40]);
    // RIFF(12) + "fmt "(4) → declared size lives at bytes 16..20.
    wav[16..20].copy_from_slice(&64u32.to_le_bytes());
    let err = parse_wav(&wav).expect_err("truncated fmt payload must be rejected");
    assert_eq!(err.kind(), io::ErrorKind::InvalidData);
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

struct ExtensibleWavSpec<'a> {
    fmt_size: u32,
    sample_rate: u32,
    byte_rate: u32,
    block_align: u16,
    bits_per_sample: u16,
    cb_size: u16,
    valid_bits: u16,
    subformat_u32: u32,
    guid_tail: [u8; 12],
    payload: &'a [u8],
}

/// Builds a mono `WAVE_FORMAT_EXTENSIBLE` (`0xFFFE`) WAV with full control over
/// `fmt_size`, `cb_size`, `valid_bits`, `byte_rate`, `block_align`, `subformat_u32`,
/// and the 12-byte `guid_tail`.
fn build_extensible_wav(spec: ExtensibleWavSpec<'_>) -> Vec<u8> {
    let mut data = Vec::new();
    data.extend_from_slice(b"RIFF");
    data.extend_from_slice(&0u32.to_le_bytes());
    data.extend_from_slice(b"WAVE");

    data.extend_from_slice(b"fmt ");
    data.extend_from_slice(&spec.fmt_size.to_le_bytes());
    data.extend_from_slice(&WAV_FORMAT_EXTENSIBLE.to_le_bytes());
    data.extend_from_slice(&1u16.to_le_bytes()); // mono
    data.extend_from_slice(&spec.sample_rate.to_le_bytes());
    data.extend_from_slice(&spec.byte_rate.to_le_bytes());
    data.extend_from_slice(&spec.block_align.to_le_bytes());
    data.extend_from_slice(&spec.bits_per_sample.to_le_bytes());

    let mut ext = Vec::new();
    ext.extend_from_slice(&spec.cb_size.to_le_bytes());
    ext.extend_from_slice(&spec.valid_bits.to_le_bytes());
    ext.extend_from_slice(&0u32.to_le_bytes()); // dwChannelMask
    ext.extend_from_slice(&spec.subformat_u32.to_le_bytes());
    ext.extend_from_slice(&spec.guid_tail);
    let ext_take = (spec.fmt_size as usize).saturating_sub(16).min(ext.len());
    data.extend_from_slice(&ext[..ext_take]);
    if (spec.fmt_size as usize) > 16 + ext_take {
        data.resize(data.len() + (spec.fmt_size as usize - 16 - ext_take), 0);
    }

    data.extend_from_slice(b"data");
    data.extend_from_slice(&(spec.payload.len() as u32).to_le_bytes());
    data.extend_from_slice(spec.payload);
    if spec.payload.len() % 2 == 1 {
        data.push(0); // RIFF even-byte pad
    }

    let total = (data.len() - 8) as u32;
    data[4..8].copy_from_slice(&total.to_le_bytes());
    data
}

const VALID_GUID_TAIL: [u8; 12] = [0, 0, 0x10, 0, 0x80, 0, 0, 0xaa, 0, 0x38, 0x9b, 0x71];

#[test]
fn extensible_valid_pcm_and_float_and_mutated_guid_bytes() {
    let cases: [(u32, u16, &[u8]); 4] = [
        (1, 16, &[0x00, 0x40]),
        (1, 24, &[0x00, 0x00, 0x40]),
        (1, 32, &[0x00, 0x00, 0x00, 0x40]),
        (3, 32, &[0x00, 0x00, 0x00, 0x3F]),
    ];

    for (subformat, bits, payload) in cases {
        let align = bits / 8;
        let wav = build_extensible_wav(ExtensibleWavSpec {
            fmt_size: 40,
            sample_rate: 48_000,
            byte_rate: 48_000 * align as u32,
            block_align: align,
            bits_per_sample: bits,
            cb_size: 22,
            valid_bits: bits,
            subformat_u32: subformat,
            guid_tail: VALID_GUID_TAIL,
            payload,
        });
        let (samples, rate) = parse_wav(&wav).expect("valid extensible WAV must parse");
        assert_eq!(rate, 48_000);
        assert_eq!(samples.len(), 1);
        assert!((samples[0] - 0.5).abs() < 1e-4);
        assert!(crate::loader::parse_wav_ir(&wav).is_ok());

        // Mutate each of the 12 bytes in EXTENSIBLE_GUID_TAIL.
        for byte_idx in 0..12 {
            let mut bad_tail = VALID_GUID_TAIL;
            bad_tail[byte_idx] ^= 0xFF;
            let bad_wav = build_extensible_wav(ExtensibleWavSpec {
                fmt_size: 40,
                sample_rate: 48_000,
                byte_rate: 48_000 * align as u32,
                block_align: align,
                bits_per_sample: bits,
                cb_size: 22,
                valid_bits: bits,
                subformat_u32: subformat,
                guid_tail: bad_tail,
                payload,
            });
            assert!(
                parse_wav(&bad_wav).is_err(),
                "mutated GUID tail byte {byte_idx} must be rejected by cabsim"
            );
            assert!(
                crate::loader::parse_wav_ir(&bad_wav).is_err(),
                "mutated GUID tail byte {byte_idx} must be rejected by loader::wav"
            );
        }

        // Mutate upper 2 bytes of Data1 (subformat u32).
        for shift in [16u32, 24] {
            let bad_subformat = subformat ^ (0x01 << shift);
            let bad_wav = build_extensible_wav(ExtensibleWavSpec {
                fmt_size: 40,
                sample_rate: 48_000,
                byte_rate: 48_000 * align as u32,
                block_align: align,
                bits_per_sample: bits,
                cb_size: 22,
                valid_bits: bits,
                subformat_u32: bad_subformat,
                guid_tail: VALID_GUID_TAIL,
                payload,
            });
            assert!(parse_wav(&bad_wav).is_err());
            assert!(crate::loader::parse_wav_ir(&bad_wav).is_err());
        }
    }
}

#[test]
fn extensible_rejects_26_byte_fmt_and_invalid_cb_size_and_valid_bits() {
    let payload = [0x00u8, 0x40];

    // 26-byte complete extensible fmt chunk is invalid by spec (must be >= 40).
    let wav_26 = build_extensible_wav(ExtensibleWavSpec {
        fmt_size: 26,
        sample_rate: 48_000,
        byte_rate: 96_000,
        block_align: 2,
        bits_per_sample: 16,
        cb_size: 22,
        valid_bits: 16,
        subformat_u32: 1,
        guid_tail: VALID_GUID_TAIL,
        payload: &payload,
    });
    assert!(parse_wav(&wav_26).is_err());
    assert!(matches!(
        crate::loader::parse_wav_ir(&wav_26),
        Err(crate::loader::WavError::InvalidExtensibleFormat)
    ));

    // cbSize < 22
    let wav_cb_small = build_extensible_wav(ExtensibleWavSpec {
        fmt_size: 40,
        sample_rate: 48_000,
        byte_rate: 96_000,
        block_align: 2,
        bits_per_sample: 16,
        cb_size: 21,
        valid_bits: 16,
        subformat_u32: 1,
        guid_tail: VALID_GUID_TAIL,
        payload: &payload,
    });
    assert!(parse_wav(&wav_cb_small).is_err());
    assert!(matches!(
        crate::loader::parse_wav_ir(&wav_cb_small),
        Err(crate::loader::WavError::InvalidExtensibleFormat)
    ));

    // cbSize > fmt_size - 18 (e.g. cbSize = 24 with fmt_size = 40 -> 24 + 18 = 42 > 40)
    let wav_cb_large = build_extensible_wav(ExtensibleWavSpec {
        fmt_size: 40,
        sample_rate: 48_000,
        byte_rate: 96_000,
        block_align: 2,
        bits_per_sample: 16,
        cb_size: 24,
        valid_bits: 16,
        subformat_u32: 1,
        guid_tail: VALID_GUID_TAIL,
        payload: &payload,
    });
    assert!(parse_wav(&wav_cb_large).is_err());
    assert!(matches!(
        crate::loader::parse_wav_ir(&wav_cb_large),
        Err(crate::loader::WavError::InvalidExtensibleFormat)
    ));

    // valid_bits = 0, > bits, and reduced (< bits) for PCM & float
    for (subfmt, bits, valid_bits, p) in [
        (1u32, 24u16, 0u16, &[0x00u8, 0x00, 0x40][..]),
        (1, 24, 32, &[0x00, 0x00, 0x40]),
        (1, 24, 20, &[0x00, 0x00, 0x40]),
        (3, 32, 24, &[0x00, 0x00, 0x00, 0x3F]),
    ] {
        let align = bits / 8;
        let bad = build_extensible_wav(ExtensibleWavSpec {
            fmt_size: 40,
            sample_rate: 48_000,
            byte_rate: 48_000 * align as u32,
            block_align: align,
            bits_per_sample: bits,
            cb_size: 22,
            valid_bits,
            subformat_u32: subfmt,
            guid_tail: VALID_GUID_TAIL,
            payload: p,
        });
        assert!(
            parse_wav(&bad).is_err(),
            "subfmt={subfmt} bits={bits} valid_bits={valid_bits} must fail"
        );
        assert!(crate::loader::parse_wav_ir(&bad).is_err());
    }

    // Wrong block_align / byte_rate
    let bad_align = build_extensible_wav(ExtensibleWavSpec {
        fmt_size: 40,
        sample_rate: 48_000,
        byte_rate: 96_000,
        block_align: 4,
        bits_per_sample: 16,
        cb_size: 22,
        valid_bits: 16,
        subformat_u32: 1,
        guid_tail: VALID_GUID_TAIL,
        payload: &payload,
    });
    assert!(parse_wav(&bad_align).is_err());
    assert!(crate::loader::parse_wav_ir(&bad_align).is_err());

    let bad_rate = build_extensible_wav(ExtensibleWavSpec {
        fmt_size: 40,
        sample_rate: 48_000,
        byte_rate: 48_000,
        block_align: 2,
        bits_per_sample: 16,
        cb_size: 22,
        valid_bits: 16,
        subformat_u32: 1,
        guid_tail: VALID_GUID_TAIL,
        payload: &payload,
    });
    assert!(parse_wav(&bad_rate).is_err());
    assert!(crate::loader::parse_wav_ir(&bad_rate).is_err());
}

#[test]
fn odd_inter_chunk_padding_and_cabsim_load_from_memory_convolution_parity() {
    // Odd-sized JUNK chunk (3 bytes + 1 pad byte) before fmt, plus odd-sized data at EOF.
    let mut wav = Vec::new();
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&0u32.to_le_bytes());
    wav.extend_from_slice(b"WAVE");
    wav.extend_from_slice(b"JUNK");
    wav.extend_from_slice(&3u32.to_le_bytes());
    wav.extend_from_slice(&[0xAA, 0xBB, 0xCC, 0x00]); // 3 payload + 1 pad byte

    // 40-byte extensible PCM24 mono
    wav.extend_from_slice(b"fmt ");
    wav.extend_from_slice(&40u32.to_le_bytes());
    wav.extend_from_slice(&WAV_FORMAT_EXTENSIBLE.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes());
    wav.extend_from_slice(&48_000u32.to_le_bytes());
    wav.extend_from_slice(&(48_000u32 * 3).to_le_bytes());
    wav.extend_from_slice(&3u16.to_le_bytes());
    wav.extend_from_slice(&24u16.to_le_bytes());
    wav.extend_from_slice(&22u16.to_le_bytes());
    wav.extend_from_slice(&24u16.to_le_bytes());
    wav.extend_from_slice(&0u32.to_le_bytes());
    wav.extend_from_slice(&1u32.to_le_bytes());
    wav.extend_from_slice(&VALID_GUID_TAIL);

    // 1 PCM24 sample (3 bytes = odd) at EOF without pad byte: value ~ 0.5
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&3u32.to_le_bytes());
    wav.extend_from_slice(&[0x00, 0x00, 0x40]);
    let total = (wav.len() - 8) as u32;
    wav[4..8].copy_from_slice(&total.to_le_bytes());

    let ir = crate::prelude::CabSimIr::load_from_memory(&wav, 0, false)
        .expect("load_from_memory must accept valid extensible WAV after padded JUNK");
    assert_eq!(ir.sample_rate, 48_000);
    assert_eq!(ir.samples.len(), 1);
    assert!((ir.samples[0] - 0.5).abs() < 1e-4);

    // Validate convolution output with partition_size = 64 (zero-latency UPOLS).
    let mut conv =
        crate::prelude::ConvEngine::new(&ir.samples, 64).expect("ConvEngine::new must succeed");
    let mut input = [0.0f32; 64];
    input[0] = 1.0;
    let mut output = [0.0f32; 64];
    conv.process(&input, &mut output, None);
    assert!(
        (output[0] - ir.samples[0]).abs() < 1e-5,
        "impulse convolution output[0]={} != ir.samples[0]={}",
        output[0],
        ir.samples[0]
    );
}
