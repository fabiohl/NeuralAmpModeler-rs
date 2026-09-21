// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::DiagnosticBundle;
use std::fs::File;
use std::io::Write;

#[test]
fn test_purge_old_reports_in_dir() {
    let mut temp_dir = std::env::temp_dir();
    let unique_id = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(12345);
    temp_dir.push(format!("nam_purge_test_{unique_id}"));

    std::fs::create_dir_all(&temp_dir).expect("Failed to create temporary directory for test");

    let crash_txt = temp_dir.join("crash-1000-clap.txt");
    let crash_tmp = temp_dir.join("crash-1001-pipe.tmp");
    let non_crash_file = temp_dir.join("keep_me.txt");
    let regular_log = temp_dir.join("normal.log");

    File::create(&crash_txt)
        .and_then(|mut f| f.write_all(b"panic 1"))
        .expect("create crash_txt");
    File::create(&crash_tmp)
        .and_then(|mut f| f.write_all(b"panic tmp"))
        .expect("create crash_tmp");
    File::create(&non_crash_file)
        .and_then(|mut f| f.write_all(b"do not delete"))
        .expect("create non_crash_file");
    File::create(&regular_log)
        .and_then(|mut f| f.write_all(b"log data"))
        .expect("create regular_log");

    // With max_age_secs = 3600, newly created files should NOT be purged.
    let count_preserved = DiagnosticBundle::purge_old_reports_in_dir(&temp_dir, 3600)
        .expect("purge with high threshold failed");
    assert_eq!(count_preserved, 0, "Recent files must not be purged");
    assert!(crash_txt.exists());
    assert!(crash_tmp.exists());
    assert!(non_crash_file.exists());
    assert!(regular_log.exists());

    // Sleep briefly to ensure duration_since(modified) >= 0 (or elapsed duration).
    std::thread::sleep(std::time::Duration::from_millis(15));

    // With max_age_secs = 0, all crash files (txt and tmp) must be purged.
    let count_purged = DiagnosticBundle::purge_old_reports_in_dir(&temp_dir, 0)
        .expect("purge with zero threshold failed");
    assert_eq!(count_purged, 2, "Both crash files should have been purged");
    assert!(!crash_txt.exists(), "crash-1000-clap.txt must be removed");
    assert!(!crash_tmp.exists(), "crash-1001-pipe.tmp must be removed");
    assert!(non_crash_file.exists(), "keep_me.txt must be retained");
    assert!(regular_log.exists(), "normal.log must be retained");

    // Purging a non-existent directory must return Ok(0) without error.
    let nonexistent = temp_dir.join("nonexistent_subfolder");
    let res = DiagnosticBundle::purge_old_reports_in_dir(&nonexistent, 0);
    assert_eq!(res.unwrap(), 0);

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_purge_old_reports_default_cache_invocable() {
    // Calling purge_old_reports with a reasonable threshold must succeed without error.
    let res = DiagnosticBundle::purge_old_reports(86400 * 30);
    assert!(res.is_ok(), "purge_old_reports must execute cleanly");
}
