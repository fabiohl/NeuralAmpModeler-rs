// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Generic DSP utilities: RT-safe building blocks shared by any host.

/// Generic RT-safe variable delay line.
pub mod delay_line;

pub use delay_line::DelayLine;
