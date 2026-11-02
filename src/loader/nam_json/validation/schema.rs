// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Structural (per-field) JSON deserialization guards for `.nam` format.
//!
//! Enforces field presence, type, and range bounds at parse time via
//! custom serde visitors — preventing DoS/memory exhaustion before any
//! semantic or topology-level validation runs.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use serde::{Deserialize, Deserializer};

use super::super::error::JsonError;
use super::record_last_typed_parse_error;
use super::semantic::MAX_SEQUENTIAL_TOTAL_CHILDREN;

/// Maximum number of floats in the `weights` array (MAX_MODEL_BYTES / 4).
pub(crate) const MAX_WEIGHTS: usize = (256 * 1024 * 1024 / 4) as usize; // 64 Mi floats

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
pub(crate) const MAX_SUBMODELS: usize = 8;

/// Aggregate cost ceiling for JSON materialized as `serde_json::Value`
/// (metadata, raw layers, `head`, `condition_dsp`, submodel/sequential
/// envelopes).
///
/// A1: the root `weights` array is no longer materialized as a generic JSON
/// tree — it is streamed straight into `Vec<f32>` under [`MAX_WEIGHTS`]. The
/// remaining `Value` fields share one per-document budget so nested models
/// cannot amplify memory before the typed error is raised. Cost is a
/// conservative estimate of in-memory footprint (see [`BudgetedValueVisitor`]),
/// so the ceiling is compared against estimated bytes, not text bytes.
pub(crate) const MAX_JSON_VALUE_COST: usize = 256 * 1024 * 1024;

/// Maximum structural depth accepted while materializing a `serde_json::Value`.
pub(crate) const MAX_JSON_VALUE_DEPTH: usize = 64;

/// Shared, per-document budget for `serde_json::Value` materialization.
///
/// Parsing is off-RT and single-threaded per call, so the active budget lives
/// in a thread-local slot installed by `parse_nam_json` (mirroring the typed
/// error side channel). Deserializers invoked outside an active parse (for
/// example `serde_json::from_value` in the dispatcher, operating on a Value
/// that was already budgeted) create a fresh local budget.
pub(crate) struct JsonBudget {
    max: usize,
    used: Cell<usize>,
}

impl JsonBudget {
    fn new(max: usize) -> Rc<Self> {
        Rc::new(Self {
            max,
            used: Cell::new(0),
        })
    }

    fn charge(&self, units: usize) -> Result<(), serde_json::Error> {
        let new = self.used.get().saturating_add(units);
        if new > self.max {
            log::warn!(
                "[Loader] Invalid field rejected: field='json_value_budget', cost={}, max={}",
                new,
                self.max
            );
            record_last_typed_parse_error(JsonError::JsonValueExceedsLimit {
                cost: new,
                max: self.max,
            });
            return Err(serde::de::Error::custom(JsonError::JsonValueExceedsLimit {
                cost: new,
                max: self.max,
            }));
        }
        self.used.set(new);
        Ok(())
    }
}

thread_local! {
    static ACTIVE_JSON_BUDGET: RefCell<Option<Rc<JsonBudget>>> = const { RefCell::new(None) };
}

/// Restores the previous thread-local budget when dropped.
pub(crate) struct JsonBudgetGuard {
    previous: Option<Rc<JsonBudget>>,
}

impl Drop for JsonBudgetGuard {
    fn drop(&mut self) {
        ACTIVE_JSON_BUDGET.with(|slot| *slot.borrow_mut() = self.previous.take());
    }
}

/// Installs a fresh per-document budget. The returned guard restores the
/// previous slot on drop (supports the nested `.namb` metadata-JSON path).
pub(crate) fn install_json_budget(max: usize) -> JsonBudgetGuard {
    let budget = JsonBudget::new(max);
    let previous = ACTIVE_JSON_BUDGET.with(|slot| slot.borrow_mut().replace(budget));
    JsonBudgetGuard { previous }
}

fn active_budget() -> Rc<JsonBudget> {
    ACTIVE_JSON_BUDGET
        .with(|slot| slot.borrow().clone())
        .unwrap_or_else(|| JsonBudget::new(MAX_JSON_VALUE_COST))
}

/// Temporarily removes the active budget so a value that was already
/// materialized under it can be converted to a typed struct without charging
/// the same nodes twice. The returned guard restores the previous budget.
pub(crate) fn suspend_json_budget() -> JsonBudgetGuard {
    let previous = ACTIVE_JSON_BUDGET.with(|slot| slot.borrow_mut().take());
    JsonBudgetGuard { previous }
}

// ── Custom serde visitors ──

/// Custom visitor for `Vec<f32>` that aborts upon exceeding MAX_WEIGHTS floats.
///
/// Kept for the visitor-level product tests; production parsing streams the
/// root array through [`collect_weights`] directly.
#[cfg(test)]
pub(crate) struct WeightsVisitor;

#[cfg(test)]
impl<'de> serde::de::Visitor<'de> for WeightsVisitor {
    type Value = Vec<f32>;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("a sequence of f32 floats within the size limit")
    }

    fn visit_seq<A>(self, seq: A) -> Result<Vec<f32>, A::Error>
    where
        A: serde::de::SeqAccess<'de>,
    {
        collect_weights(seq)
    }
}

/// Streams a JSON float sequence into `Vec<f32>` while enforcing
/// [`MAX_WEIGHTS`] and finiteness **during** deserialization.
///
/// Shared by [`WeightsVisitor`] and the root-model envelope visitor so the
/// root `weights` array is never materialized as a generic JSON tree (A1).
pub(crate) fn collect_weights<'de, A>(mut seq: A) -> Result<Vec<f32>, A::Error>
where
    A: serde::de::SeqAccess<'de>,
{
    let mut weights = Vec::new();
    let mut index: usize = 0;
    loop {
        match seq.next_element::<f32>() {
            Ok(Some(val)) => {
                if !val.is_finite() {
                    // Structured rejection diagnostic for off-RT triage.
                    // `offset_bytes` is the byte offset of the offending slot
                    // within the weights section (index × 4), matching the
                    // coordinate system used by the `.namb` binary weights.
                    log::warn!(
                        "[Loader] Invalid field rejected: field='weights[{}]', value={:?}, offset_bytes={}",
                        index,
                        val,
                        index * 4
                    );
                    record_last_typed_parse_error(JsonError::WeightNotFinite { index, value: val });
                    return Err(serde::de::Error::custom(JsonError::WeightNotFinite {
                        index,
                        value: val,
                    }));
                }
                if weights.len() >= MAX_WEIGHTS {
                    record_last_typed_parse_error(JsonError::WeightsExceedLimit {
                        got: weights.len() + 1,
                        max: MAX_WEIGHTS,
                    });
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

/// Custom deserializer for `weights: Vec<f32>` with cap at MAX_WEIGHTS.
#[cfg(test)]
pub(crate) fn deserialize_weights<'de, D>(deserializer: D) -> Result<Vec<f32>, D::Error>
where
    D: Deserializer<'de>,
{
    deserializer.deserialize_seq(WeightsVisitor)
}

/// Budgeted generic JSON tree visitor.
///
/// Unlike [`LimitedValueVisitor`] (which guards only `metadata.training` with
/// its own 1 MiB ceiling), this visitor draws from the document-wide
/// [`JsonBudget`] so every `serde_json::Value` field shares one aggregate
/// ceiling. Cost estimates approximate the in-memory `Value` footprint
/// (`Number`/tag/map-entry overhead), not the source text.
pub(crate) struct BudgetedValueVisitor {
    budget: Rc<JsonBudget>,
    depth: usize,
}

impl BudgetedValueVisitor {
    fn root() -> Self {
        Self {
            budget: active_budget(),
            depth: 0,
        }
    }

    fn check_depth<E>(&self) -> Result<(), E>
    where
        E: serde::de::Error,
    {
        if self.depth > MAX_JSON_VALUE_DEPTH {
            record_last_typed_parse_error(JsonError::JsonValueTooDeep {
                depth: self.depth,
                max_depth: MAX_JSON_VALUE_DEPTH,
            });
            return Err(serde::de::Error::custom(JsonError::JsonValueTooDeep {
                depth: self.depth,
                max_depth: MAX_JSON_VALUE_DEPTH,
            }));
        }
        Ok(())
    }
}

impl<'de> serde::de::Visitor<'de> for BudgetedValueVisitor {
    type Value = serde_json::Value;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("a JSON value within the document budget")
    }

    fn visit_bool<E>(self, v: bool) -> Result<serde_json::Value, E>
    where
        E: serde::de::Error,
    {
        self.budget.charge(8).map_err(E::custom)?;
        Ok(serde_json::Value::Bool(v))
    }

    fn visit_i64<E>(self, v: i64) -> Result<serde_json::Value, E>
    where
        E: serde::de::Error,
    {
        self.budget.charge(16).map_err(E::custom)?;
        Ok(serde_json::Value::Number(serde_json::Number::from(v)))
    }

    fn visit_u64<E>(self, v: u64) -> Result<serde_json::Value, E>
    where
        E: serde::de::Error,
    {
        self.budget.charge(16).map_err(E::custom)?;
        Ok(serde_json::Value::Number(serde_json::Number::from(v)))
    }

    fn visit_f64<E>(self, v: f64) -> Result<serde_json::Value, E>
    where
        E: serde::de::Error,
    {
        self.budget.charge(16).map_err(E::custom)?;
        Ok(serde_json::Value::Number(
            serde_json::Number::from_f64(v).unwrap_or(serde_json::Number::from(0)),
        ))
    }

    fn visit_str<E>(self, v: &str) -> Result<serde_json::Value, E>
    where
        E: serde::de::Error,
    {
        self.budget
            .charge(v.len().saturating_add(16))
            .map_err(E::custom)?;
        Ok(serde_json::Value::String(v.to_string()))
    }

    fn visit_string<E>(self, v: String) -> Result<serde_json::Value, E>
    where
        E: serde::de::Error,
    {
        self.budget
            .charge(v.len().saturating_add(16))
            .map_err(E::custom)?;
        Ok(serde_json::Value::String(v))
    }

    fn visit_unit<E>(self) -> Result<serde_json::Value, E>
    where
        E: serde::de::Error,
    {
        self.budget.charge(8).map_err(E::custom)?;
        Ok(serde_json::Value::Null)
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<serde_json::Value, A::Error>
    where
        A: serde::de::SeqAccess<'de>,
    {
        self.check_depth::<A::Error>()?;
        self.budget.charge(16).map_err(serde::de::Error::custom)?;
        let mut arr = Vec::new();
        loop {
            match seq.next_element_seed(BudgetedValueSeed::child(&self)) {
                Ok(Some(val)) => {
                    self.budget.charge(8).map_err(serde::de::Error::custom)?;
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
        self.check_depth::<A::Error>()?;
        self.budget.charge(16).map_err(serde::de::Error::custom)?;
        let mut obj = serde_json::Map::new();
        loop {
            match map.next_key::<String>() {
                Ok(Some(key)) => {
                    self.budget
                        .charge(key.len().saturating_add(24))
                        .map_err(serde::de::Error::custom)?;
                    let val: serde_json::Value =
                        map.next_value_seed(BudgetedValueSeed::child(&self))?;
                    obj.insert(key, val);
                }
                Ok(None) => break,
                Err(e) => return Err(e),
            }
        }
        Ok(serde_json::Value::Object(obj))
    }
}

/// `DeserializeSeed` wrapper so children share the parent's budget/depth.
pub(crate) struct BudgetedValueSeed {
    budget: Rc<JsonBudget>,
    depth: usize,
}

impl BudgetedValueSeed {
    pub(crate) fn root() -> Self {
        Self {
            budget: active_budget(),
            depth: 0,
        }
    }

    fn child(parent: &BudgetedValueVisitor) -> Self {
        Self {
            budget: Rc::clone(&parent.budget),
            depth: parent.depth + 1,
        }
    }
}

impl<'de> serde::de::DeserializeSeed<'de> for BudgetedValueSeed {
    type Value = serde_json::Value;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(BudgetedValueVisitor {
            budget: self.budget,
            depth: self.depth,
        })
    }
}

/// Custom deserializer for bounded `Option<serde_json::Value>` fields.
/// `null`/absent becomes `None` (matching derived `Option<Value>` semantics).
pub(crate) fn deserialize_bounded_value_opt<'de, D>(
    deserializer: D,
) -> Result<Option<serde_json::Value>, D::Error>
where
    D: Deserializer<'de>,
{
    struct BoundedOptionVisitor;

    impl<'de> serde::de::Visitor<'de> for BoundedOptionVisitor {
        type Value = Option<serde_json::Value>;

        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter.write_str("an optional JSON value within the document budget")
        }

        fn visit_none<E>(self) -> Result<Self::Value, E> {
            Ok(None)
        }

        fn visit_unit<E>(self) -> Result<Self::Value, E> {
            Ok(None)
        }

        fn visit_some<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
        where
            D: Deserializer<'de>,
        {
            let value = deserializer.deserialize_any(BudgetedValueVisitor::root())?;
            Ok(Some(value))
        }
    }

    deserializer.deserialize_option(BoundedOptionVisitor)
}

/// Custom deserializer for bounded `Value` fields that preserve explicit
/// `null` as `Some(Value::Null)` (used by `config.head`).
pub(crate) fn deserialize_bounded_value_present<'de, D>(
    deserializer: D,
) -> Result<Option<serde_json::Value>, D::Error>
where
    D: Deserializer<'de>,
{
    let value = deserializer.deserialize_any(BudgetedValueVisitor::root())?;
    Ok(Some(value))
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
            record_last_typed_parse_error(JsonError::TrainingTooLarge {
                size: new,
                max_size: self.max_size,
            });
            return Err(serde::de::Error::custom(JsonError::TrainingTooLarge {
                size: new,
                max_size: self.max_size,
            }));
        }
        Ok(())
    }

    fn check_depth(&self) -> Result<(), serde_json::Error> {
        if self.depth > self.max_depth {
            record_last_typed_parse_error(JsonError::TrainingTooDeep {
                depth: self.depth,
                max_depth: self.max_depth,
            });
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
            record_last_typed_parse_error(JsonError::SubmodelsExceedLimit {
                got: arr.len(),
                max: MAX_SUBMODELS,
            });
            return Err(serde::de::Error::custom(JsonError::SubmodelsExceedLimit {
                got: arr.len(),
                max: MAX_SUBMODELS,
            }));
        }

        Ok(Some(arr))
    }
}

/// Custom deserializer for `config.models: Option<Vec<serde_json::Value>>`
/// (Sequential architecture) that enforces the per-level child-count cap.
/// The aggregate tree budget (total children and total weights across nested
/// `Sequential` descendants) is enforced by the topology scan
/// (`topology/sequential.rs`) with checked arithmetic.
pub(crate) fn deserialize_models<'de, D>(
    deserializer: D,
) -> Result<Option<Vec<serde_json::Value>>, D::Error>
where
    D: Deserializer<'de>,
{
    deserializer.deserialize_option(ModelsOptionVisitor)
}

struct ModelsOptionVisitor;

impl<'de> serde::de::Visitor<'de> for ModelsOptionVisitor {
    type Value = Option<Vec<serde_json::Value>>;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("an optional array of sequential child models")
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
        D: Deserializer<'de>,
    {
        let arr: Vec<serde_json::Value> = Vec::deserialize(deserializer)?;

        if arr.len() > MAX_SEQUENTIAL_TOTAL_CHILDREN {
            record_last_typed_parse_error(JsonError::SequentialChildrenExceedLimit {
                got: arr.len(),
                max: MAX_SEQUENTIAL_TOTAL_CHILDREN,
            });
            return Err(serde::de::Error::custom(
                JsonError::SequentialChildrenExceedLimit {
                    got: arr.len(),
                    max: MAX_SEQUENTIAL_TOTAL_CHILDREN,
                },
            ));
        }

        Ok(Some(arr))
    }
}

/// Custom deserializer for `sample_rate: Option<f32>`.
///
/// RFC 7.1: `sample_rate = -1.0` is the NAMcore C++ sentinel for "unknown"
/// (see `get_dsp.cpp`). All finite ≤ 0.0 values are treated as "unknown" (`None`)
/// — matching C++ lenience. NaN and ±Inf are rejected with `InvalidSampleRate`,
/// as are positive values that are not integers exactly representable as `u32`
/// (DEC-01d); the original literal is validated as `f64` before reducing to `f32`.
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
            // Read the original number as f64 so the value can be validated
            // before the lossy reduction to f32 (A3).
            let num = f64::deserialize(deserializer)?;
            match crate::loader::sample_rate::declared_rate_from_f64(num) {
                Ok(rate) => Ok(rate),
                Err(_) => {
                    let val = num as f32;
                    let reason = if num.is_finite() {
                        "must be a positive integer exactly representable as u32"
                    } else {
                        "must be finite"
                    };
                    // Structured rejection diagnostic for off-RT triage.
                    // `sample_rate` is a top-level scalar field, so it has no
                    // per-element offset within a section — `offset_bytes=0`
                    // signals "document-level field" (no array slot index).
                    log::warn!(
                        "[Loader] Invalid field rejected: field='sample_rate', value={:?}, offset_bytes={}",
                        val,
                        0
                    );
                    record_last_typed_parse_error(JsonError::InvalidSampleRate {
                        value: val,
                        reason,
                    });
                    Err(serde::de::Error::custom(JsonError::InvalidSampleRate {
                        value: val,
                        reason,
                    }))
                }
            }
        }
    }

    deserializer.deserialize_option(SampleRateOptionVisitor)
}

/// Validates the top-level or child envelope of a `.nam` model JSON representation.
///
/// Ensures that:
/// - The value is a JSON object.
/// - Required fields `version`, `architecture`, `config`, and `weights` are present and non-null.
/// - `version` is a JSON string.
/// - `architecture` is a JSON string.
/// - `config` is a JSON object.
/// - `weights` is a JSON array.
///
/// This validation is reusable for both top-level model files and child model
/// definitions in `Sequential` containers (NC-3). Error messages never leak
/// local file system paths.
pub fn validate_envelope(value: &serde_json::Value) -> Result<(), JsonError> {
    let Some(obj) = value.as_object() else {
        return Err(JsonError::InvalidEnvelope {
            reason: "model envelope must be a JSON object".to_string(),
        });
    };

    match obj.get("version") {
        None => {
            return Err(JsonError::InvalidEnvelope {
                reason: "missing required field 'version'".to_string(),
            });
        }
        Some(serde_json::Value::Null) => {
            return Err(JsonError::InvalidEnvelope {
                reason: "field 'version' cannot be null".to_string(),
            });
        }
        Some(v) if !v.is_string() => {
            return Err(JsonError::InvalidEnvelope {
                reason: "field 'version' must be a JSON string".to_string(),
            });
        }
        _ => {}
    }

    match obj.get("architecture") {
        None => {
            return Err(JsonError::InvalidEnvelope {
                reason: "missing required field 'architecture'".to_string(),
            });
        }
        Some(serde_json::Value::Null) => {
            return Err(JsonError::InvalidEnvelope {
                reason: "field 'architecture' cannot be null".to_string(),
            });
        }
        Some(v) if !v.is_string() => {
            return Err(JsonError::InvalidEnvelope {
                reason: "field 'architecture' must be a JSON string".to_string(),
            });
        }
        _ => {}
    }

    match obj.get("config") {
        None => {
            return Err(JsonError::InvalidEnvelope {
                reason: "missing required field 'config'".to_string(),
            });
        }
        Some(serde_json::Value::Null) => {
            return Err(JsonError::InvalidEnvelope {
                reason: "field 'config' cannot be null".to_string(),
            });
        }
        Some(v) if !v.is_object() => {
            return Err(JsonError::InvalidEnvelope {
                reason: "field 'config' must be a JSON object".to_string(),
            });
        }
        _ => {}
    }

    match obj.get("weights") {
        None => {
            return Err(JsonError::InvalidEnvelope {
                reason: "missing required field 'weights'".to_string(),
            });
        }
        Some(serde_json::Value::Null) => {
            return Err(JsonError::InvalidEnvelope {
                reason: "field 'weights' cannot be null".to_string(),
            });
        }
        Some(v) if !v.is_array() => {
            return Err(JsonError::InvalidEnvelope {
                reason: "field 'weights' must be a JSON array".to_string(),
            });
        }
        _ => {}
    }

    Ok(())
}
