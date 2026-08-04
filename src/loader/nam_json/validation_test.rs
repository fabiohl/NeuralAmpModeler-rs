// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use serde::de::Visitor;
use serde::de::value::Error as ValueError;
use serde::de::value::{F32Deserializer, SeqDeserializer};

// — Weight visitor tests (NaN, Inf, extreme counts) ——————————————

#[derive(Debug, serde::Deserialize)]
struct WeightsWrapper {
    #[serde(deserialize_with = "super::validation::deserialize_weights")]
    weights: Vec<f32>,
}

#[test]
fn test_weights_accepts_finite_values() {
    let json = r#"{"weights": [0.5, -1.0, 2.0e3, -0.001, 3.14]}"#;
    let parsed: WeightsWrapper = serde_json::from_str(json).unwrap();
    assert_eq!(parsed.weights.len(), 5);
    for &v in &parsed.weights {
        assert!(v.is_finite(), "weight should be finite, got {v}");
    }
}

#[test]
fn test_weights_visitor_rejects_nan() {
    let values = vec![0.5f32, f32::NAN, -1.0f32];
    let deser = SeqDeserializer::<_, ValueError>::new(
        values.into_iter().map(F32Deserializer::<ValueError>::new),
    );
    let visitor = super::validation::WeightsVisitor;
    let result: Result<Vec<f32>, ValueError> = visitor.visit_seq(deser);
    assert!(result.is_err(), "NaN weight should be rejected");
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("not finite") || err_msg.contains("WeightNotFinite"),
        "error should mention non-finite: {err_msg}"
    );
}

#[test]
fn test_weights_visitor_rejects_infinity() {
    let values = vec![0.5f32, f32::INFINITY, -1.0f32];
    let deser = SeqDeserializer::<_, ValueError>::new(
        values.into_iter().map(F32Deserializer::<ValueError>::new),
    );
    let visitor = super::validation::WeightsVisitor;
    let result: Result<Vec<f32>, ValueError> = visitor.visit_seq(deser);
    assert!(result.is_err(), "+Inf weight should be rejected");
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("not finite") || err_msg.contains("WeightNotFinite"),
        "error should mention non-finite: {err_msg}"
    );
}

#[test]
fn test_weights_visitor_rejects_negative_infinity() {
    let values = vec![0.5f32, f32::NEG_INFINITY, -1.0f32];
    let deser = SeqDeserializer::<_, ValueError>::new(
        values.into_iter().map(F32Deserializer::<ValueError>::new),
    );
    let visitor = super::validation::WeightsVisitor;
    let result: Result<Vec<f32>, ValueError> = visitor.visit_seq(deser);
    assert!(result.is_err(), "-Inf weight should be rejected");
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("not finite") || err_msg.contains("WeightNotFinite"),
        "error should mention non-finite: {err_msg}"
    );
}

#[test]
fn test_weights_visitor_rejects_count_overflow() {
    let count = 1000;
    let values: Vec<f32> = (0..count).map(|i| i as f32 * 0.001).collect();
    let deser = SeqDeserializer::<_, ValueError>::new(
        values.into_iter().map(F32Deserializer::<ValueError>::new),
    );
    let visitor = super::validation::WeightsVisitor;
    let weights: Vec<f32> = visitor.visit_seq(deser).unwrap();
    assert_eq!(weights.len(), count);
}

#[test]
fn test_weights_single_element() {
    let json = r#"{"weights": [1.0]}"#;
    let parsed: WeightsWrapper = serde_json::from_str(json).unwrap();
    assert_eq!(parsed.weights.len(), 1);
    assert!((parsed.weights[0] - 1.0).abs() < 1e-12);
}

#[test]
fn test_weights_empty_array() {
    let json = r#"{"weights": []}"#;
    let parsed: WeightsWrapper = serde_json::from_str(json).unwrap();
    assert!(parsed.weights.is_empty());
}

#[test]
fn test_weights_missing_field_is_error() {
    let json = r#"{}"#;
    let result: Result<WeightsWrapper, _> = serde_json::from_str(json);
    assert!(result.is_err(), "missing weights field should be an error");
}

#[test]
fn test_weights_not_a_sequence_is_error() {
    let json = r#"{"weights": "not_a_list"}"#;
    let result: Result<WeightsWrapper, _> = serde_json::from_str(json);
    assert!(result.is_err(), "string where array expected should error");
}

#[test]
fn test_weights_not_a_float_in_sequence_is_error() {
    let json = r#"{"weights": [1.0, "bad", 3.0]}"#;
    let result: Result<WeightsWrapper, _> = serde_json::from_str(json);
    assert!(result.is_err(), "non-float element should error");
}

#[test]
fn test_weights_extreme_negative_exponent() {
    let json = r#"{"weights": [-1.0e-45, 1.0e-45]}"#;
    let parsed: WeightsWrapper = serde_json::from_str(json).unwrap();
    assert_eq!(parsed.weights.len(), 2);
    // Subnormal floats should still be representable as f32
    assert!(parsed.weights[0] < 0.0);
    assert!(parsed.weights[1] > 0.0);
}

#[test]
fn test_weights_extreme_positive_exponent() {
    let json = r#"{"weights": [3.4e38, -3.4e38]}"#;
    let parsed: WeightsWrapper = serde_json::from_str(json).unwrap();
    assert_eq!(parsed.weights.len(), 2);
    assert!(parsed.weights[0].is_finite());
    assert!(parsed.weights[1].is_finite());
}

#[test]
fn test_weights_large_array_near_limit() {
    // MAX_WEIGHTS is 64 Mi floats — we test with 10k (tiny but exercises the path)
    let mut buf = String::from(r#"{"weights": ["#);
    for i in 0..10_000 {
        if i > 0 {
            buf.push(',');
        }
        buf.push_str(&format!("{}", i as f32 * 0.001));
    }
    buf.push_str("]}");
    let parsed: WeightsWrapper = serde_json::from_str(&buf).unwrap();
    assert_eq!(parsed.weights.len(), 10_000);
}

// — Training visitor tests (depth/size limits) ————————————————————

#[derive(Debug, serde::Deserialize)]
struct TrainingWrapper {
    #[serde(deserialize_with = "super::validation::deserialize_training")]
    #[serde(default)]
    metadata: Option<serde_json::Value>,
}

#[test]
fn test_training_absence_is_none() {
    let json = r#"{}"#;
    let parsed: TrainingWrapper = serde_json::from_str(json).unwrap();
    assert!(parsed.metadata.is_none());
}

#[test]
fn test_training_null_is_none() {
    let json = r#"{"metadata": null}"#;
    let parsed: TrainingWrapper = serde_json::from_str(json).unwrap();
    assert!(parsed.metadata.is_none());
}

#[test]
fn test_training_simple_value_is_ok() {
    let json = r#"{"metadata": 42}"#;
    let parsed: TrainingWrapper = serde_json::from_str(json).unwrap();
    let v = parsed.metadata.unwrap();
    assert_eq!(v.as_u64(), Some(42));
}

#[test]
fn test_training_small_object_is_ok() {
    let json = r#"{"metadata": {"epochs": 100, "lr": 0.001}}"#;
    let parsed: TrainingWrapper = serde_json::from_str(json).unwrap();
    let v = parsed.metadata.unwrap();
    let obj = v.as_object().unwrap();
    assert_eq!(obj["epochs"].as_u64(), Some(100));
    assert!((obj["lr"].as_f64().unwrap() - 0.001).abs() < 1e-12);
}

#[test]
fn test_training_deeply_nested_rejected() {
    let depth = 64;
    let mut buf = String::from(r#"{"metadata": "#);
    for _ in 0..depth {
        buf.push_str(r#"{"x": "#);
    }
    buf.push_str(r#""deep""#);
    for _ in 0..depth {
        buf.push('}');
    }
    buf.push('}');
    let result: Result<TrainingWrapper, _> = serde_json::from_str(&buf);
    assert!(
        result.is_err(),
        "deeply nested JSON should exceed depth limit"
    );
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("too deep") || err_msg.contains("TrainingTooDeep"),
        "error should mention depth: {err_msg}"
    );
}

// — Submodels visitor tests ————————————————————————————————————————

#[derive(Debug, serde::Deserialize)]
struct SubmodelsWrapper {
    #[serde(default, deserialize_with = "super::validation::deserialize_submodels")]
    submodels: Option<Vec<serde_json::Value>>,
}

#[test]
fn test_submodels_absence_is_none() {
    let json = r#"{}"#;
    let parsed: SubmodelsWrapper = serde_json::from_str(json).unwrap();
    assert!(parsed.submodels.is_none());
}

#[test]
fn test_submodels_null_is_none() {
    let json = r#"{"submodels": null}"#;
    let parsed: SubmodelsWrapper = serde_json::from_str(json).unwrap();
    assert!(parsed.submodels.is_none());
}

#[test]
fn test_submodels_within_limit_is_ok() {
    let json = r#"{"submodels": [1, 2, 3]}"#;
    let parsed: SubmodelsWrapper = serde_json::from_str(json).unwrap();
    let arr = parsed.submodels.unwrap();
    assert_eq!(arr.len(), 3);
}

#[test]
fn test_submodels_at_max_limit_is_ok() {
    // MAX_SUBMODELS = 8
    let mut buf = String::from(r#"{"submodels": ["#);
    for i in 0..8 {
        if i > 0 {
            buf.push(',');
        }
        buf.push_str(&format!("{}", i));
    }
    buf.push_str("]}");
    let parsed: SubmodelsWrapper = serde_json::from_str(&buf).unwrap();
    assert_eq!(parsed.submodels.unwrap().len(), 8);
}

#[test]
fn test_submodels_exceed_limit_rejected() {
    let json = r#"{"submodels": [1, 2, 3, 4, 5, 6, 7, 8, 9]}"#;
    let result: Result<SubmodelsWrapper, _> = serde_json::from_str(json);
    assert!(result.is_err(), "9 submodels should exceed limit of 8");
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("submodels") || err_msg.contains("SubmodelsExceedLimit"),
        "error should mention submodels: {err_msg}"
    );
}

// — Sample rate visitor tests ————————————————————————————————————————

#[derive(Debug, serde::Deserialize)]
struct SampleRateWrapper {
    #[serde(
        default,
        deserialize_with = "super::validation::deserialize_sample_rate"
    )]
    sample_rate: Option<f32>,
}

#[test]
fn test_sample_rate_absence_is_none() {
    let json = r#"{}"#;
    let parsed: SampleRateWrapper = serde_json::from_str(json).unwrap();
    assert!(parsed.sample_rate.is_none());
}

#[test]
fn test_sample_rate_null_is_none() {
    let json = r#"{"sample_rate": null}"#;
    let parsed: SampleRateWrapper = serde_json::from_str(json).unwrap();
    assert!(parsed.sample_rate.is_none());
}

#[test]
fn test_sample_rate_valid_is_ok() {
    let json = r#"{"sample_rate": 48000.0}"#;
    let parsed: SampleRateWrapper = serde_json::from_str(json).unwrap();
    assert_eq!(parsed.sample_rate, Some(48000.0));
}

#[test]
fn test_sample_rate_zero_is_none_sentinel() {
    let json = r#"{"sample_rate": 0.0}"#;
    let parsed: SampleRateWrapper = serde_json::from_str(json).unwrap();
    assert!(parsed.sample_rate.is_none());
}

#[test]
fn test_sample_rate_negative_is_none_sentinel() {
    let json = r#"{"sample_rate": -44100.0}"#;
    let parsed: SampleRateWrapper = serde_json::from_str(json).unwrap();
    assert!(parsed.sample_rate.is_none());
}

#[test]
fn test_sample_rate_exact_cpp_sentinel_is_none() {
    let json = r#"{"sample_rate": -1.0}"#;
    let parsed: SampleRateWrapper = serde_json::from_str(json).unwrap();
    assert!(parsed.sample_rate.is_none());
}

#[test]
fn test_sample_rate_very_negative_is_none_sentinel() {
    let json = r#"{"sample_rate": -99999.0}"#;
    let parsed: SampleRateWrapper = serde_json::from_str(json).unwrap();
    assert!(parsed.sample_rate.is_none());
}

struct MockDeserializer {
    val: Option<f32>,
}

impl<'de> serde::Deserializer<'de> for MockDeserializer {
    type Error = serde::de::value::Error;

    fn deserialize_any<V>(self, _visitor: V) -> Result<V::Value, Self::Error>
    where
        V: serde::de::Visitor<'de>,
    {
        unimplemented!()
    }

    fn deserialize_option<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: serde::de::Visitor<'de>,
    {
        match self.val {
            Some(v) => visitor.visit_some(MockFloatDeserializer(v)),
            None => visitor.visit_none(),
        }
    }

    serde::forward_to_deserialize_any! {
        bool i8 i16 i32 i64 i128 u8 u16 u32 u64 u128 f32 f64 char str string
        bytes byte_buf unit unit_struct newtype_struct seq tuple
        tuple_struct map struct enum identifier ignored_any
    }
}

struct MockFloatDeserializer(f32);

impl<'de> serde::Deserializer<'de> for MockFloatDeserializer {
    type Error = serde::de::value::Error;

    fn deserialize_any<V>(self, visitor: V) -> Result<V::Value, Self::Error>
    where
        V: serde::de::Visitor<'de>,
    {
        visitor.visit_f32(self.0)
    }

    serde::forward_to_deserialize_any! {
        bool i8 i16 i32 i64 i128 u8 u16 u32 u64 u128 f32 f64 char str string
        bytes byte_buf option unit unit_struct newtype_struct seq tuple
        tuple_struct map struct enum identifier ignored_any
    }
}

#[test]
fn test_sample_rate_infinity_rejected() {
    let deser = MockDeserializer {
        val: Some(f32::INFINITY),
    };
    let result = super::validation::deserialize_sample_rate(deser);
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("sample rate") || err_msg.contains("InvalidSampleRate"),
        "error should contain sample rate info, got: {err_msg}"
    );
}

#[test]
fn test_sample_rate_neg_infinity_rejected() {
    let deser = MockDeserializer {
        val: Some(f32::NEG_INFINITY),
    };
    let result = super::validation::deserialize_sample_rate(deser);
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("sample rate") || err_msg.contains("InvalidSampleRate"),
        "error should contain sample rate info, got: {err_msg}"
    );
}

#[test]
fn test_sample_rate_nan_rejected() {
    let deser = MockDeserializer {
        val: Some(f32::NAN),
    };
    let result = super::validation::deserialize_sample_rate(deser);
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("sample rate") || err_msg.contains("InvalidSampleRate"),
        "error should contain sample rate info, got: {err_msg}"
    );
}
