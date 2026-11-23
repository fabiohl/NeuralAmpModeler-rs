// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;

#[test]
fn render_header_distinguishes_active_and_host_isa_when_different() {
    let with_host = r#"{"kind":"build_metadata","effective_isa":"x86-64-v3 (AVX2/FMA/F16C/BMI)","host_isa":"AVX-512","engine_backend":"AVX2"}"#;
    let report = parse_quality_report(with_host).expect("parse build_metadata");
    assert_eq!(report.header.host_isa.as_deref(), Some("AVX-512"));
    assert_eq!(report.header.engine_backend.as_deref(), Some("AVX2"));
    let plain = render_quality_report(&report, RenderStyle::Plain);
    assert!(
        plain.contains(
            "   ISA:    x86-64-v3 (AVX2/FMA/F16C/BMI) (active engine dispatch; host CPU: AVX-512)\n"
        ),
        "unexpected header:\n{plain}"
    );

    let without_host =
        r#"{"kind":"build_metadata","effective_isa":"x86-64-v3 (AVX2/FMA/F16C/BMI)"}"#;
    let baseline_report = parse_quality_report(without_host).expect("parse baseline");
    let baseline = render_quality_report(&baseline_report, RenderStyle::Plain);
    assert!(baseline.contains("   ISA:    x86-64-v3 (AVX2/FMA/F16C/BMI)\n"));
    assert!(!baseline.contains("host CPU"));
}
