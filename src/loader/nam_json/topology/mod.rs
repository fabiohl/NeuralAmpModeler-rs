// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Model topology detection submodules (wavenet, lstm, linear, convnet, a2).

mod a2;
mod convnet;
mod linear;
mod lstm;
mod wavenet;

pub use a2::{A2TopologyResult, is_a2_shape};
pub use convnet::{ConvNetFormat, ConvNetTopology, get_convnet_topology};
pub use linear::get_linear_topology;
pub use lstm::get_lstm_topology;
pub(crate) use wavenet::parse_semver;
pub use wavenet::{
    FreeWavenetGeometry, NamWavenetTopology, WavenetTopologyResult, get_wavenet_topology,
};
