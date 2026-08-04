// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Detection of Linear topologies from model data.

use super::super::data::{LinearImplementation, NamModelData};
use super::super::validation::MAX_RECEPTIVE_FIELD;

/// Checks and returns the Linear geometry (receptive_field, has_bias, implementation).
pub fn get_linear_topology(data: &NamModelData) -> Option<(usize, bool, LinearImplementation)> {
    if data.architecture != "Linear" {
        return None;
    }

    let receptive_field = data.config.receptive_field?;
    if receptive_field > MAX_RECEPTIVE_FIELD {
        log::warn!(
            "Linear receptive_field ({receptive_field}) exceeds maximum \
             {MAX_RECEPTIVE_FIELD} — OOM/DoS protection"
        );
        return None;
    }
    let has_bias = data.config.bias.unwrap_or(false);
    let implementation = data
        .config
        .implementation
        .as_deref()
        .and_then(|s| s.parse().ok())
        .unwrap_or_default();
    Some((receptive_field, has_bias, implementation))
}
