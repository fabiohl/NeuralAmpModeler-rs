// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Structural (per-field) JSON deserialization guards for `.nam` format.
//!
//! Enforces field presence, type, and range bounds at parse time via
//! custom serde visitors — preventing DoS/memory exhaustion before any
//! semantic or topology-level validation runs.

use serde::{Deserialize, Deserializer};

use super::super::error::JsonError;

/// Maximum number of floats in the `weights` array (MAX_MODEL_BYTES / 4).
const MAX_WEIGHTS: usize = (256 * 1024 * 1024 / 4) as usize; // 64 Mi floats

/// Maximum number of layers across any architecture at parse time.
/// Universal OOM guard applied immediately after JSON deserialization.
pub const MAX_LAYERS: usize = 8;

/// Maximum hidden size across any architecture at parse time.
/// Universal OOM guard applied immediately after JSON deserialization.
pub const MAX_HIDDEN_SIZE: usize = 512;

/// Maximum size of the `metadata.training` field in bytes.
const MAX_TRAINING_BYTES: usize = 1024 * 1024; // 1 MiB

/// Maximum depth of the JSON tree in `metadata.training`.
const MAX_TRAINING_DEPTH: usize = 16;

/// Maximum number of submodels in a SlimmableContainer.
const MAX_SUBMODELS: usize = 8;

// ── Custom serde visitors ──

/// Custom visitor for `Vec<f32>` that aborts upon exceeding MAX_WEIGHTS floats.
#[cfg(not(test))]
struct WeightsVisitor;
#[cfg(test)]
pub(crate) struct WeightsVisitor;

impl<'de> serde::de::Visitor<'de> for WeightsVisitor {
    type Value = Vec<f32>;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("a sequence of f32 floats within the size limit")
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<Vec<f32>, A::Error>
    where
        A: serde::de::SeqAccess<'de>,
    {
        let mut weights = Vec::new();
        let mut index: usize = 0;
        loop {
            match seq.next_element::<f32>() {
                Ok(Some(val)) => {
                    if !val.is_finite() {
                        return Err(serde::de::Error::custom(JsonError::WeightNotFinite {
                            index,
                            value: val,
                        }));
                    }
                    if weights.len() >= MAX_WEIGHTS {
                        return Err(serde::de::Error::custom(JsonError::WeightsExceedLimit {
                            got: weights.len() + 1,
                            max: MAX_WEIGHTS,
                        }));
                    }
                    weights.push(val);
                    index += 1;
                }
                Ok(None) => break,
                Err(e) => return Err(e),
            }
        }
        Ok(weights)
    }
}

/// Custom deserializer for `weights: Vec<f32>` with cap at MAX_WEIGHTS.
pub(crate) fn deserialize_weights<'de, D>(deserializer: D) -> Result<Vec<f32>, D::Error>
where
    D: Deserializer<'de>,
{
    deserializer.deserialize_seq(WeightsVisitor)
}

/// JSON tree visitor for `metadata.training` with depth and size limits.
///
/// Limits are enforced during **deserialization** (not afterwards) to prevent
/// memory exhaustion before validation completes.  The 1 MiB aggregate limit
/// protects against DoS via deeply nested or overly verbose training metadata.
///
/// Uses `std::cell::Cell<usize>` so that child visitors share the size counter
/// with the parent, avoiding bypass of the 1 MiB aggregate limit.
struct LimitedValueVisitor {
    depth: usize,
    max_depth: usize,
    max_size: usize,
    current_size: std::cell::Cell<usize>,
}

impl LimitedValueVisitor {
    fn root(max_depth: usize, max_size: usize) -> Self {
        Self {
            depth: 0,
            max_depth,
            max_size,
            current_size: std::cell::Cell::new(0),
        }
    }

    fn child(&self) -> Self {
        Self {
            depth: self.depth + 1,
            max_depth: self.max_depth,
            max_size: self.max_size,
            current_size: self.current_size.clone(),
        }
    }

    fn add_size(&self, bytes: usize) -> Result<(), serde_json::Error> {
        let new = self.current_size.get() + bytes;
        self.current_size.set(new);
        if new > self.max_size {
            return Err(serde::de::Error::custom(JsonError::TrainingTooLarge {
                size: new,
                max_size: self.max_size,
            }));
        }
        Ok(())
    }

    fn check_depth(&self) -> Result<(), serde_json::Error> {
        if self.depth > self.max_depth {
            return Err(serde::de::Error::custom(JsonError::TrainingTooDeep {
                depth: self.depth,
                max_depth: self.max_depth,
            }));
        }
        Ok(())
    }
}

impl<'de> serde::de::Visitor<'de> for LimitedValueVisitor {
    type Value = serde_json::Value;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("a JSON value within depth and size limits")
    }

    // Each visit_* method charges a conservative upper-bound byte count
    // against the 1 MiB aggregate limit.  The estimates approximate the size
    // of the JSON *text* that produced the value, not the in-memory
    // representation, to remain invariant across different JSON parsers.

    fn visit_bool<E>(self, v: bool) -> Result<serde_json::Value, E>
    where
        E: serde::de::Error,
    {
        // "true" = 4 bytes, "false" = 5 bytes
        self.add_size(if v { 4 } else { 5 }).map_err(E::custom)?;
        Ok(serde_json::Value::Bool(v))
    }

    fn visit_i64<E>(self, v: i64) -> Result<serde_json::Value, E>
    where
        E: serde::de::Error,
    {
        // 16 bytes: conservative max for any i64 JSON representation
        // (e.g. "-9223372036854775808")
        self.add_size(16).map_err(E::custom)?;
        Ok(serde_json::Value::Number(serde_json::Number::from(v)))
    }

    fn visit_u64<E>(self, v: u64) -> Result<serde_json::Value, E>
    where
        E: serde::de::Error,
    {
        // 16 bytes: same upper-bound as i64
        self.add_size(16).map_err(E::custom)?;
        Ok(serde_json::Value::Number(serde_json::Number::from(v)))
    }

    fn visit_f64<E>(self, v: f64) -> Result<serde_json::Value, E>
    where
        E: serde::de::Error,
    {
        // 16 bytes: covers the vast majority of f64 text forms
        // (e.g. "-1.7976931348623157e308" would be longer, but
        // training metadata never reaches that precision)
        self.add_size(16).map_err(E::custom)?;
        Ok(serde_json::Value::Number(
            serde_json::Number::from_f64(v).unwrap_or(serde_json::Number::from(0)),
        ))
    }

    fn visit_str<E>(self, v: &str) -> Result<serde_json::Value, E>
    where
        E: serde::de::Error,
    {
        // +2 for the surrounding double-quotes "…"
        self.add_size(v.len() + 2).map_err(E::custom)?;
        Ok(serde_json::Value::String(v.to_string()))
    }

    fn visit_string<E>(self, v: String) -> Result<serde_json::Value, E>
    where
        E: serde::de::Error,
    {
        // +2 for the surrounding double-quotes "…"
        self.add_size(v.len() + 2).map_err(E::custom)?;
        Ok(serde_json::Value::String(v))
    }

    fn visit_unit<E>(self) -> Result<serde_json::Value, E> {
        // null costs 4 bytes but the discriminator (type tag) covers it
        Ok(serde_json::Value::Null)
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<serde_json::Value, A::Error>
    where
        A: serde::de::SeqAccess<'de>,
    {
        self.check_depth().map_err(serde::de::Error::custom)?;
        // +2 for the opening and closing brackets "[…]"
        self.add_size(2).map_err(serde::de::Error::custom)?;
        let mut arr = Vec::new();
        loop {
            match seq.next_element_seed(self.child()) {
                Ok(Some(val)) => {
                    // +1 for the comma between elements
                    self.add_size(1).map_err(serde::de::Error::custom)?;
                    arr.push(val);
                }
                Ok(None) => break,
                Err(e) => return Err(e),
            }
        }
        Ok(serde_json::Value::Array(arr))
    }

    fn visit_map<A>(self, mut map: A) -> Result<serde_json::Value, A::Error>
    where
        A: serde::de::MapAccess<'de>,
    {
        self.check_depth().map_err(serde::de::Error::custom)?;
        // +2 for the opening and closing braces "{…}"
        self.add_size(2).map_err(serde::de::Error::custom)?;
        let mut obj = serde_json::Map::new();
        loop {
            match map.next_key::<String>() {
                Ok(Some(key)) => {
                    // len + 4: key string chars + 2 quotes + 1 colon + 1 space
                    let key_len = key.len() + 4;
                    self.add_size(key_len).map_err(serde::de::Error::custom)?;
                    let val: serde_json::Value = map.next_value_seed(self.child())?;
                    obj.insert(key, val);
                    // +1 for the comma between key-value pairs
                    self.add_size(1).map_err(serde::de::Error::custom)?;
                }
                Ok(None) => break,
                Err(e) => return Err(e),
            }
        }
        Ok(serde_json::Value::Object(obj))
    }
}

impl<'de> serde::de::DeserializeSeed<'de> for LimitedValueVisitor {
    type Value = serde_json::Value;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_any(self)
    }
}

/// External visitor for `Option<serde_json::Value>`: returns `None` for null/absent,
/// and `Some(value)` with depth/size limits for present values.
struct TrainingOptionVisitor;

impl<'de> serde::de::Visitor<'de> for TrainingOptionVisitor {
    type Value = Option<serde_json::Value>;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("an optional JSON value")
    }

    fn visit_none<E>(self) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        Ok(None)
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        Ok(None)
    }

    fn visit_some<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let visitor = LimitedValueVisitor::root(MAX_TRAINING_DEPTH, MAX_TRAINING_BYTES);
        let value: serde_json::Value = deserializer.deserialize_any(visitor)?;
        Ok(Some(value))
    }
}

/// Custom deserializer for `metadata.training` with depth and size limits.
pub(crate) fn deserialize_training<'de, D>(
    deserializer: D,
) -> Result<Option<serde_json::Value>, D::Error>
where
    D: Deserializer<'de>,
{
    deserializer.deserialize_option(TrainingOptionVisitor)
}

/// Custom deserializer for `config.submodels: Option<Vec<serde_json::Value>>`
/// that enforces a maximum of 8 submodels. Nested containers are allowed —
/// recursion depth is enforced by the dispatcher.
pub(crate) fn deserialize_submodels<'de, D>(
    deserializer: D,
) -> Result<Option<Vec<serde_json::Value>>, D::Error>
where
    D: Deserializer<'de>,
{
    deserializer.deserialize_option(SubmodelsOptionVisitor)
}

struct SubmodelsOptionVisitor;

impl<'de> serde::de::Visitor<'de> for SubmodelsOptionVisitor {
    type Value = Option<Vec<serde_json::Value>>;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("an optional array of submodel entries")
    }

    fn visit_none<E>(self) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        Ok(None)
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        Ok(None)
    }

    fn visit_some<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let arr: Vec<serde_json::Value> = Vec::deserialize(deserializer)?;

        if arr.len() > MAX_SUBMODELS {
            return Err(serde::de::Error::custom(JsonError::SubmodelsExceedLimit {
                got: arr.len(),
                max: MAX_SUBMODELS,
            }));
        }

        Ok(Some(arr))
    }
}

/// Custom deserializer for `sample_rate: Option<f32>`.
///
/// RFC 7.1: `sample_rate = -1.0` is the NAMcore C++ sentinel for "unknown"
/// (see `get_dsp.cpp`). All finite ≤ 0.0 values are treated as "unknown" (`None`)
/// — matching C++ lenience. NaN and ±Inf are rejected with `InvalidSampleRate`.
pub(crate) fn deserialize_sample_rate<'de, D>(deserializer: D) -> Result<Option<f32>, D::Error>
where
    D: Deserializer<'de>,
{
    struct SampleRateOptionVisitor;

    impl<'de> serde::de::Visitor<'de> for SampleRateOptionVisitor {
        type Value = Option<f32>;

        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter.write_str("an optional f32 sample rate")
        }

        fn visit_none<E>(self) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            Ok(None)
        }

        fn visit_unit<E>(self) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            Ok(None)
        }

        fn visit_some<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
        where
            D: serde::Deserializer<'de>,
        {
            let val = f32::deserialize(deserializer)?;
            if !val.is_finite() {
                return Err(serde::de::Error::custom(JsonError::InvalidSampleRate {
                    value: val,
                    reason: "must be finite",
                }));
            }
            if val <= 0.0 {
                return Ok(None);
            }
            Ok(Some(val))
        }
    }

    deserializer.deserialize_option(SampleRateOptionVisitor)
}
