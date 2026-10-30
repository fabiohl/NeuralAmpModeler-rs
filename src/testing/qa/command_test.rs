// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;
use std::thread;
use std::time::Instant;

#[cfg(unix)]
#[test]
fn returns_the_child_exit_code_and_captures_both_streams() {
    let mut cmd = Command::new("sh");
    cmd.args(["-c", "printf out; printf err >&2; exit 7"]);
    let result = run_command_with_timeout(&mut cmd, Duration::from_secs(5));
    assert_eq!(result.exit_code, 7);
    assert!(!result.timed_out);
    assert_eq!(result.output, b"outerr");
}

#[cfg(unix)]
#[test]
fn timeout_returns_124_and_kills_the_child_tree() {
    let mut cmd = Command::new("sh");
    cmd.args(["-c", "sleep 3600 & echo $!; wait"]);
    let started = Instant::now();
    let result = run_command_with_timeout(&mut cmd, Duration::from_secs(1));
    assert_eq!(result.exit_code, 124);
    assert!(result.timed_out);
    assert!(started.elapsed() < Duration::from_secs(5));

    let child_pid: i32 = String::from_utf8_lossy(&result.output)
        .lines()
        .next()
        .and_then(|line| line.trim().parse().ok())
        .expect("shell must report the background child pid");
    let deadline = Instant::now() + Duration::from_secs(2);
    while process_alive(child_pid) && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(10));
    }
    assert!(
        !process_alive(child_pid),
        "grandchild {child_pid} survived the process-group kill"
    );
}

#[cfg(unix)]
fn process_alive(pid: i32) -> bool {
    // SAFETY: signal 0 performs the error check without delivering a signal.
    unsafe { libc::kill(pid, 0) == 0 }
}

#[cfg(unix)]
#[test]
fn missing_binary_reports_spawn_failure() {
    let mut cmd = Command::new("nam-qa-command-that-does-not-exist");
    let result = run_command_with_timeout(&mut cmd, Duration::from_secs(1));
    assert_eq!(result.exit_code, 127);
    assert!(!result.timed_out);
    assert!(!result.output.is_empty());
}
