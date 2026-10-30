// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Bounded subprocess execution for the quality dashboard.
//!
//! The dashboard must never abandon a phase because one command hung: a
//! time-out has to kill the whole child tree (libtest spawns the test binary,
//! cargo spawns libtest) and still drain the captured pipes, otherwise a
//! runaway process survives the run and keeps the workspace busy.

use std::io::Read;
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

/// Polling interval of the child-exit wait loop.
const POLL_INTERVAL: Duration = Duration::from_millis(10);

/// Result of a command executed under a wall-clock deadline.
#[derive(Debug)]
pub struct TimedCommandOutput {
    /// Captured bytes: stdout followed by stderr.
    pub output: Vec<u8>,
    /// Process exit code, or the synthetic 124 (time-out) / 127 (spawn error).
    pub exit_code: u32,
    /// `true` when the deadline expired and the child tree was killed.
    pub timed_out: bool,
}

/// Runs `cmd` with captured stdout/stderr, killing its whole process group if
/// it exceeds `timeout`.
///
/// On Unix the child becomes a process-group leader, so a hung `cargo test`
/// cannot leave the actual test binary (or any grandchild) behind. On other
/// platforms only the direct child is killed.
pub fn run_command_with_timeout(cmd: &mut Command, timeout: Duration) -> TimedCommandOutput {
    cmd.stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }

    let mut child = match cmd.spawn() {
        Ok(child) => child,
        Err(e) => {
            return TimedCommandOutput {
                output: format!("failed to spawn {:?}: {e}\n", cmd.get_program()).into_bytes(),
                exit_code: 127,
                timed_out: false,
            };
        }
    };

    let (Some(stdout), Some(stderr)) = (child.stdout.take(), child.stderr.take()) else {
        kill_child_tree(&mut child);
        let _ = child.wait();
        return TimedCommandOutput {
            output: b"failed to capture child pipes\n".to_vec(),
            exit_code: 127,
            timed_out: false,
        };
    };

    let stdout_reader = thread::spawn(move || read_all(stdout));
    let stderr_reader = thread::spawn(move || read_all(stderr));

    let deadline = Instant::now() + timeout;
    let mut timed_out = false;
    let exit_code: u32;
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                exit_code = status.code().unwrap_or(1) as u32;
                break;
            }
            Ok(None) => {
                let now = Instant::now();
                if now >= deadline {
                    timed_out = true;
                    exit_code = 124;
                    kill_child_tree(&mut child);
                    let _ = child.wait();
                    break;
                }
                thread::sleep(POLL_INTERVAL.min(deadline - now));
            }
            Err(_) => {
                exit_code = 1;
                kill_child_tree(&mut child);
                let _ = child.wait();
                break;
            }
        }
    }

    let mut output = stdout_reader.join().unwrap_or_default();
    output.extend(stderr_reader.join().unwrap_or_default());
    TimedCommandOutput {
        output,
        exit_code,
        timed_out,
    }
}

fn read_all(mut reader: impl Read) -> Vec<u8> {
    let mut buf = Vec::new();
    let _ = reader.read_to_end(&mut buf);
    buf
}

fn kill_child_tree(child: &mut Child) {
    #[cfg(unix)]
    {
        // SAFETY: `process_group(0)` made the child its own process-group
        // leader, so its pid is also the group id; a negative pid targets the
        // whole group. SIGKILL cannot be ignored and the group id is still
        // valid while the child has not been reaped.
        unsafe {
            libc::kill(-(child.id() as i32), libc::SIGKILL);
        }
    }
    let _ = child.kill();
}

#[cfg(test)]
#[path = "command_test.rs"]
mod tests;
