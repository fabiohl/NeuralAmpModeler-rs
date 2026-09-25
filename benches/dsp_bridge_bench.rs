// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Micro-benchmarks for `DspBridge` lock-free double-buffered channel.
//!
//! Evaluates writer latency (`write_block`), reader latency (`read_block`),
//! full producer-consumer roundtrips, and off-RT dropped frame telemetry drain.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use neural_amp_modeler_rs::dsp::pipeline::{DspBridge, DspBridgeReader, DspBridgeWriter};
use std::hint::black_box;

fn bench_bridge_write(c: &mut Criterion) {
    let mut group = c.benchmark_group("dsp_bridge_write");

    for &block_size in &[64, 128, 256] {
        let input_l = vec![0.5f32; block_size];
        let input_r = vec![0.5f32; block_size];

        group.bench_with_input(
            BenchmarkId::new("write_block", block_size),
            &block_size,
            |b, &_bs| {
                let mut bridge = DspBridge::new_boxed();
                let bridge_ptr = &mut *bridge as *mut DspBridge;
                let writer = unsafe { DspBridgeWriter::new(bridge_ptr) };

                b.iter(|| {
                    // Force consumed_gen forward to avoid artificial frame drops during benchmark
                    bridge.consumed_gen.store(
                        bridge.generation.load(std::sync::atomic::Ordering::Relaxed),
                        std::sync::atomic::Ordering::Release,
                    );
                    writer.write_block(
                        black_box(&input_l),
                        black_box(&input_r),
                        black_box(block_size),
                        black_box(false),
                    );
                });
            },
        );
    }
    group.finish();
}

fn bench_bridge_read(c: &mut Criterion) {
    let mut group = c.benchmark_group("dsp_bridge_read");

    for &block_size in &[64, 128, 256] {
        let input_l = vec![0.5f32; block_size];
        let input_r = vec![0.5f32; block_size];

        group.bench_with_input(
            BenchmarkId::new("read_block", block_size),
            &block_size,
            |b, &_bs| {
                let mut bridge = DspBridge::new_boxed();
                let bridge_ptr = &mut *bridge as *mut DspBridge;
                let writer = unsafe { DspBridgeWriter::new(bridge_ptr) };
                let reader = unsafe { DspBridgeReader::new(bridge_ptr) };
                let mut last_gen = 0u64;

                b.iter(|| {
                    bridge.consumed_gen.store(
                        bridge.generation.load(std::sync::atomic::Ordering::Relaxed),
                        std::sync::atomic::Ordering::Release,
                    );
                    writer.write_block(&input_l, &input_r, block_size, false);

                    let samples_read = reader
                        .read_block(&mut last_gen, |l, _r| black_box(l.len()))
                        .unwrap_or(0);
                    black_box(samples_read);
                });
            },
        );
    }
    group.finish();
}

fn bench_bridge_drain_dropped(c: &mut Criterion) {
    c.bench_function("dsp_bridge_drain_dropped", |b| {
        let bridge = DspBridge::new_boxed();
        b.iter(|| {
            black_box(bridge.drain_dropped_frames());
        });
    });
}

criterion_group!(
    benches,
    bench_bridge_write,
    bench_bridge_read,
    bench_bridge_drain_dropped
);
criterion_main!(benches);
