// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;
use crate::common::diagnostics::NamErrorCode;

struct GeometryModel {
    in_ch: usize,
    out_ch: usize,
    processed: bool,
}

impl sealed::Sealed for GeometryModel {}

impl NamModel for GeometryModel {
    fn process(&mut self, _input: &[f32], output: &mut [f32]) {
        self.processed = true;
        output.fill(0.0);
    }

    fn num_input_channels(&self) -> usize {
        self.in_ch
    }

    fn num_output_channels(&self) -> usize {
        self.out_ch
    }

    fn prewarm(&mut self, _num_samples: usize) {}
}

fn planes(count: usize, frames: usize, value: f32) -> Vec<Vec<f32>> {
    (0..count).map(|_| vec![value; frames]).collect()
}

#[test]
fn process_multichannel_accepts_geometry_at_staging_capacity() {
    let mut model = GeometryModel {
        in_ch: 1,
        out_ch: plane_adapter::STAGE_SAMPLES,
        processed: false,
    };
    let input = vec![0.5f32; 4];
    let mut out_bufs = planes(plane_adapter::STAGE_SAMPLES, 4, 7.0);
    let mut outs: Vec<&mut [f32]> = out_bufs.iter_mut().map(|b| b.as_mut_slice()).collect();

    assert_eq!(
        model.process_multichannel(&[input.as_slice()], &mut outs),
        Ok(4)
    );
    assert!(model.processed);
    assert!(outs.iter().all(|p| p.iter().all(|&v| v == 0.0)));
}

#[test]
fn process_multichannel_rejects_geometry_above_staging_capacity_before_touching_planes() {
    let wide = plane_adapter::STAGE_SAMPLES + 1;
    let mut model = GeometryModel {
        in_ch: 1,
        out_ch: wide,
        processed: false,
    };
    let input = vec![0.5f32; 4];
    let mut out_bufs = planes(wide, 4, 7.0);
    let mut outs: Vec<&mut [f32]> = out_bufs.iter_mut().map(|b| b.as_mut_slice()).collect();

    assert_eq!(
        model.process_multichannel(&[input.as_slice()], &mut outs),
        Err(NamErrorCode::HostFormatContractViolation)
    );
    assert!(!model.processed);
    assert!(outs.iter().all(|p| p.iter().all(|&v| v == 7.0)));
}

#[test]
fn process_multichannel_in_place_rejects_geometry_above_staging_capacity() {
    let wide = plane_adapter::STAGE_SAMPLES + 1;
    let mut model = GeometryModel {
        in_ch: wide,
        out_ch: 1,
        processed: false,
    };
    let mut bufs = planes(wide, 4, 3.0);
    let mut planes_mut: Vec<&mut [f32]> = bufs.iter_mut().map(|b| b.as_mut_slice()).collect();

    assert_eq!(
        model.process_multichannel_in_place(&mut planes_mut),
        Err(NamErrorCode::HostFormatContractViolation)
    );
    assert!(!model.processed);
    assert!(planes_mut.iter().all(|p| p.iter().all(|&v| v == 3.0)));
}
