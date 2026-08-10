// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Data structures, validation, and errors for the `.nam` format (JSON).
//!
//! Re-exports from sibling modules: model structs, typed errors, and validation helpers.

pub use super::error::JsonError;
pub use super::model::{
    LinearImplementation, NamConfig, NamDate, NamLayerConfig, NamMetadata, NamModelData,
    SlimmableConfig, SlimmableKwargs, WeightsLayout,
};
