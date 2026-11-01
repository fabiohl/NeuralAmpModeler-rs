// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::ids::f64_oracle_fixture_is_substitute;

#[test]
fn container_families_are_marked_as_f64_substitutes() {
    for family in [
        "Container A2-Full (CH=8) C++ cross-reference",
        "Container A2-Lite (CH=3) C++ cross-reference",
        "Container File A2-Lite (CH=3) C++ cross-reference",
        "Container File A2-Full (CH=8) C++ cross-reference",
        "SlimmableContainer A2 Example (CH=3→6) C++ cross-reference",
    ] {
        assert!(
            f64_oracle_fixture_is_substitute(family),
            "{family} must be classified as a proxy"
        );
    }
}

#[test]
fn direct_fixture_families_are_not_marked_as_substitutes() {
    for family in [
        "BossWN-standard",
        "WaveNet A2-Lite (CH=3) C++ cross-reference",
        "BossLSTM-1x16",
    ] {
        assert!(
            !f64_oracle_fixture_is_substitute(family),
            "{family} must not be classified as a proxy"
        );
    }
}
