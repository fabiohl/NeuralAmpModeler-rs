// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;

#[test]
fn test_shutdown_in_progress_lifecycle_and_reset() {
    // Ensure initial or reset state
    clear_shutdown_in_progress();
    assert!(
        !is_shutdown_in_progress(),
        "Shutdown latch should be false initially or after clear"
    );

    // Simulate instance teardown (0 remaining instances)
    set_shutdown_in_progress();
    assert!(
        is_shutdown_in_progress(),
        "Shutdown latch should be true after set_shutdown_in_progress"
    );

    // Simulate new instance created in the same process (0 -> 1 reload)
    clear_shutdown_in_progress();
    assert!(
        !is_shutdown_in_progress(),
        "Shutdown latch should be reset to false on new instance initialization"
    );
}
