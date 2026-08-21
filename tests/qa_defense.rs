// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Defense harness (EP-05 / R-05) — the Rust home of the long
//! suite's "Defense scripts + libm + oversample" phase.
//!
//! `utils/tests-long.sh` used to run an inline bash unit suite
//! (`run_bash_scripts_unit_tests`) that extracted
//! functions from `utils/quality-dashboard.sh` / `utils/_lib.sh` with
//! `sed`+`eval` (`extract_define`) and asserted their failure modes. This
//! harness replaces that block with one `cargo test --features testing
//! --test qa_defense` invocation covering the same acceptance cases, now
//! against the Rust implementations the defense surface was ported to:
//!
//! - F-01 / F-28 — metric sanitization: `is_finite_num` accept/reject lists
//!   and the fail-closed `as_finite` accessor (`src/testing/qa/metrics.rs`).
//! - F-08 — single `NOT_VERIFIED` classifier (`src/testing/qa/classify.rs`).
//! - F-21 — executed-tests counter (`count_tests_executed_from_log` +
//!   `nam_long_receipt count-log`, `src/testing/receipt.rs`).
//! - F-22 — freshness gate sandbox: consistent / stale / missing / orphan
//!   (`src/testing/freshness.rs` + the `nam_freshness` stdout tokens the
//!   bash wrapper classifies into `FRESHNESS_REASON`).
//! - F-24 — baseline coverage cross-check (`src/testing/qa/coverage.rs` +
//!   `nam_perf_gate coverage`).
//! - F-27 — canonical JSONL fidelity parse (`src/testing/qa/metrics.rs`).
//! - `utils/ensure_namcore_render.sh` exit-code contract and
//!   idempotency flows via subprocess (the function deliberately stays in
//!   bash; these tests never `extract_define` it).
//!
//! The harness is structural (no float oracles), so the long suite runs it
//! in debug like the old bash block. Every test is fast and deterministic:
//! the C++ build scenarios use a fake `cmake` + fake compilers, and all
//! sandboxes live under the OS temp dir.

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};

use neural_amp_modeler_rs::testing::freshness::{FreshnessMode, FreshnessReason, check_freshness};
use neural_amp_modeler_rs::testing::qa::classify::{
    RegressionOutcome, classify_regression_outcome,
};
use neural_amp_modeler_rs::testing::qa::coverage::{executed_bench_ids, missing_baseline_coverage};
use neural_amp_modeler_rs::testing::qa::metrics::{
    MetricValue, is_finite_num, parse_fidelity_jsonl,
};
use neural_amp_modeler_rs::testing::receipt::count_tests_executed_from_log;

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn temp_dir() -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("nam-qa-defense-{}-{n}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("temp dir must be creatable");
    dir
}

fn write_file(path: &Path, content: &str) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("parent dirs must be creatable");
    }
    fs::write(path, content).expect("file must be writable");
}

fn stdout(out: &Output) -> String {
    String::from_utf8_lossy(&out.stdout).to_string()
}

fn stderr(out: &Output) -> String {
    String::from_utf8_lossy(&out.stderr).to_string()
}

/// Reads a file trimming the trailing newline — `ensure_namcore_render`
/// persists `.build_config` with `printf '%s\n'`.
fn read_trim(path: &Path) -> String {
    fs::read_to_string(path)
        .expect("file must be readable")
        .trim()
        .to_string()
}

fn project_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

// ── F-01 / F-28: metric sanitization ─────────────────────────────────────────

/// The `_is_finite_num` acceptance list of `utils/tests-long.sh` (F-01),
/// verbatim.
#[test]
fn f01_is_finite_num_accepts_canonical_forms() {
    for value in [
        "0",
        "0.0",
        "0.5",
        ".5",
        "1.",
        "1.5e-3",
        "-1.5E3",
        "+3.14",
        "3.14e2",
        "42",
        "12345678901234567890",
    ] {
        assert!(is_finite_num(value), "expected '{value}' to be accepted");
    }
}

/// The `_is_finite_num` rejection list of `utils/tests-long.sh` (F-01,
/// F-28), verbatim — including the case-insensitive non-finite sentinels.
#[test]
fn f01_is_finite_num_rejects_sentinels_and_garbage() {
    for value in [
        "",
        " ",
        "inf",
        "-inf",
        "+inf",
        "Infinity",
        "-infinity",
        "nan",
        "-nan",
        "NaN",
        "null",
        "N/A",
        "abc",
        "1.2.3",
        "0x10",
        "1e",
        "e5",
        ".",
    ] {
        assert!(!is_finite_num(value), "expected '{value}' to be rejected");
    }
}

/// Fail-closed verify path: non-finite sentinels and `N/A` never coerce to
/// `0.0`; finite e-notation passes through verbatim.
#[test]
fn f01_finite_accessor_is_fail_closed() {
    assert!(is_finite_num(".5"), "_is_numeric_esr accepts '.5'");
    assert!(!is_finite_num("inf"), "_is_numeric_esr rejects 'inf'");
    assert_eq!(MetricValue::Na.as_finite(), None);
    assert_eq!(MetricValue::Raw("inf".into()).as_finite(), None);
    assert_eq!(
        MetricValue::Raw("1.5e-3".into()).as_finite(),
        Some("1.5e-3")
    );
}

// ── F-08: single NOT_VERIFIED classifier ────────────────────────────────────

/// The 7 classifier cases of the removed bash block (F-08), verbatim.
#[test]
fn f08_single_classifier_cases() {
    assert_eq!(
        classify_regression_outcome("PASS", ""),
        RegressionOutcome::Pass
    );
    assert_eq!(
        classify_regression_outcome("FAIL", "MISSING_BASELINE"),
        RegressionOutcome::NotVerified
    );
    assert_eq!(
        classify_regression_outcome("FAIL", "INCOMPARABLE_ENVIRONMENT"),
        RegressionOutcome::NotVerified
    );
    assert_eq!(
        classify_regression_outcome("FAIL", "REGRESSION_DETECTED"),
        RegressionOutcome::Fail
    );
    assert_eq!(
        classify_regression_outcome("FAIL", "Benchmark run failed"),
        RegressionOutcome::Fail
    );
    assert_eq!(classify_regression_outcome("", ""), RegressionOutcome::Fail);
    assert_eq!(
        classify_regression_outcome("SKIP_CAPABILITY", "whatever"),
        RegressionOutcome::Fail
    );
}

// ── F-21: executed-tests counter ─────────────────────────────────────────────

/// The 6 `assert_ran_tests` cases of the removed bash block (F-21),
/// verbatim — a 0 counter (100% skip / missing log) must FAIL the gate.
#[test]
fn f21_executed_tests_counter_cases() {
    let work = temp_dir();
    let logs = work.join("logs");
    fs::create_dir_all(&logs).unwrap();
    let counter = |name: &str, content: &str| {
        let path = logs.join(name);
        write_file(&path, content);
        count_tests_executed_from_log(&path)
    };

    assert_eq!(
        counter("pass.log", "test result: ok. 50 passed. 2 failed.\n"),
        52
    );
    assert_eq!(
        counter("zero.log", "test result: ok. 0 passed. 0 failed.\n"),
        0
    );
    assert_eq!(
        counter(
            "skip.log",
            "running tests...\nall filtered out (early return)\n"
        ),
        0
    );
    assert_eq!(
        count_tests_executed_from_log(&logs.join("absent.log")),
        0,
        "missing file must count 0"
    );
    assert_eq!(
        counter("bench.log", "bench time: [1.2 ms]\nbench time: [3.4 ms]\n"),
        2
    );
    assert_eq!(counter("meas.log", "x 5 measured\n"), 5);
}

/// The same fixtures through the `nam_long_receipt count-log` process — the
/// path `_lib.sh::assert_ran_tests` delegates to.
#[test]
fn f21_count_log_bin_mirrors_cases() {
    let bin = env!("CARGO_BIN_EXE_nam_long_receipt");
    let work = temp_dir();
    let cases: &[(&str, &str, &str)] = &[
        ("pass.log", "test result: ok. 50 passed. 2 failed.\n", "52"),
        ("zero.log", "test result: ok. 0 passed. 0 failed.\n", "0"),
        (
            "skip.log",
            "running tests...\nall filtered out (early return)\n",
            "0",
        ),
        (
            "bench.log",
            "bench time: [1.2 ms]\nbench time: [3.4 ms]\n",
            "2",
        ),
    ];
    for (name, content, expected) in cases {
        let path = work.join(name);
        write_file(&path, content);
        let out = Command::new(bin)
            .args(["count-log", "--log", path.to_str().unwrap()])
            .output()
            .expect("nam_long_receipt count-log must run");
        assert!(
            out.status.success(),
            "count-log failed on {name}: {}",
            stderr(&out)
        );
        assert_eq!(stdout(&out).trim(), *expected, "count-log on {name}");
    }
}

// ── F-22: freshness gate sandbox ─────────────────────────────────────────────

/// Minimal self-consistent sandbox: manifest + one registered model.
fn freshness_sandbox(extra_manifest: &str) -> PathBuf {
    let sb = temp_dir();
    let model = sb.join("tests/fixtures/models/model_a.nam");
    write_file(&model, "model-a\n");
    let sha = sha256_of_file(&model);
    write_file(
        &sb.join("tests/fixtures/.golden_manifest.sha256"),
        &format!(
            "# Golden freshness manifest — test fixture\n\
             {sha} 0000000000000000000000000000000000000000000000000000000000000000 model_a.nam golden_a.bin\n\
             {extra_manifest}"
        ),
    );
    sb
}

/// SHA-256 of a model file via `sha256sum` — exactly how the removed bash
/// block built its sandbox manifest (coreutils, present on the Linux hosts
/// the long suite certifies).
fn sha256_of_file(path: &Path) -> String {
    let out = Command::new("sha256sum")
        .arg(path)
        .output()
        .expect("sha256sum must run");
    assert!(out.status.success(), "sha256sum failed on {:?}", path);
    stdout(&out)
        .split_whitespace()
        .next()
        .unwrap_or_default()
        .to_string()
}

/// The 4 `run_freshness_gate` sandbox cases of the removed bash block
/// (F-22), verbatim: OK / STALE_FIXTURES / MISSING_FIXTURES /
/// ORPHAN_FIXTURE.
#[test]
fn f22_freshness_sandbox_reasons() {
    let ok = check_freshness(
        freshness_sandbox("# MODEL-REGISTRY: model_a.nam\n"),
        FreshnessMode::ArtifactsHard,
    )
    .unwrap();
    assert!(ok.is_ok());
    assert_eq!(ok.reason, FreshnessReason::Ok);

    let stale = check_freshness(
        {
            let sb = freshness_sandbox("# MODEL-REGISTRY: model_a.nam\n");
            write_file(&sb.join("tests/fixtures/models/model_a.nam"), "tamper\n");
            sb
        },
        FreshnessMode::ArtifactsHard,
    )
    .unwrap();
    assert_eq!(stale.reason, FreshnessReason::StaleFixtures);

    let missing = check_freshness(
        freshness_sandbox("# EXPECTED: missing_golden.bin\n# MODEL-REGISTRY: model_a.nam\n"),
        FreshnessMode::ArtifactsHard,
    )
    .unwrap();
    assert_eq!(missing.reason, FreshnessReason::MissingFixtures);

    let orphan = check_freshness(
        {
            let sb = freshness_sandbox("# MODEL-REGISTRY: model_a.nam\n");
            write_file(&sb.join("tests/fixtures/models/orphan.nam"), "orphan\n");
            sb
        },
        FreshnessMode::ArtifactsHard,
    )
    .unwrap();
    assert_eq!(orphan.reason, FreshnessReason::OrphanFixture);
}

/// The `nam_freshness` stdout tokens the bash `run_freshness_gate` wrapper
/// classifies into `FRESHNESS_REASON` (grep STALE/MISSING/ORPHAN) — asserted
/// through the real bin so the wrapper semantics survive the bash block
/// removal.
#[test]
fn f22_freshness_bin_classifies_tokens() {
    let bin = env!("CARGO_BIN_EXE_nam_freshness");
    let run = |root: &Path| {
        Command::new(bin)
            .arg("--root")
            .arg(root)
            .arg("artifacts-hard")
            .output()
            .expect("nam_freshness must run")
    };

    let out = run(&freshness_sandbox("# MODEL-REGISTRY: model_a.nam\n"));
    assert_eq!(out.status.code(), Some(0), "stderr: {}", stderr(&out));

    let sb = freshness_sandbox("# MODEL-REGISTRY: model_a.nam\n");
    write_file(&sb.join("tests/fixtures/models/model_a.nam"), "tamper\n");
    let out = run(&sb);
    assert_eq!(out.status.code(), Some(1));
    assert!(stdout(&out).contains("STALE:"), "got: {}", stdout(&out));

    let out = run(&freshness_sandbox(
        "# EXPECTED: missing_golden.bin\n# MODEL-REGISTRY: model_a.nam\n",
    ));
    assert_eq!(out.status.code(), Some(1));
    assert!(stdout(&out).contains("MISSING:"), "got: {}", stdout(&out));

    let sb = freshness_sandbox("# MODEL-REGISTRY: model_a.nam\n");
    write_file(&sb.join("tests/fixtures/models/orphan.nam"), "orphan\n");
    let out = run(&sb);
    assert_eq!(out.status.code(), Some(1));
    assert!(stdout(&out).contains("ORPHAN:"), "got: {}", stdout(&out));
}

// ── F-24: baseline coverage cross-check ──────────────────────────────────────

/// The 4 `nam_perf_gate coverage` cases of the removed bash block (F-24),
/// verbatim: gap listed + `BASELINE_COVERAGE_GAP`, full coverage ok, and the
/// blind gate on unparseable/absent logs.
#[test]
fn f24_baseline_coverage_cases() {
    let work = temp_dir();
    let crit = work.join("crit-root");
    for id in ["RT_A", "RT_B"] {
        fs::create_dir_all(crit.join(id).join("ci-baseline")).unwrap();
    }
    let crit_log = work.join("crit.log");
    write_file(
        &crit_log,
        "Benchmarking RT_A: Warming up for 1.0000 s\n\
         Benchmarking RT_A: Collecting 100 samples\n\
         Benchmarking RT_B: Warming up for 1.0000 s\n\
         Benchmarking RT_C: Warming up for 1.0000 s\n",
    );

    // Lib level: ids dedup + missing list + fail-closed parse errors.
    assert_eq!(
        executed_bench_ids(&fs::read_to_string(&crit_log).unwrap()),
        ["RT_A", "RT_B", "RT_C"]
    );
    assert_eq!(
        missing_baseline_coverage(
            &fs::read_to_string(&crit_log).unwrap(),
            &crit,
            "ci-baseline"
        )
        .unwrap(),
        vec!["RT_C".to_string()]
    );
    assert!(missing_baseline_coverage("garbage\n", &crit, "ci-baseline").is_err());
    assert!(missing_baseline_coverage("", &crit, "ci-baseline").is_err());

    // Process level: the bin is the gate the long suite calls.
    let bin = env!("CARGO_BIN_EXE_nam_perf_gate");
    let run = |log: &Path| {
        Command::new(bin)
            .args([
                "coverage",
                "--log",
                log.to_str().unwrap(),
                "--root",
                crit.to_str().unwrap(),
                "--baseline",
                "ci-baseline",
            ])
            .output()
            .expect("nam_perf_gate must run")
    };

    let out = run(&crit_log);
    assert_eq!(out.status.code(), Some(1), "RT_C without series must fail");
    assert!(stdout(&out).contains("RT_C"), "got: {}", stdout(&out));
    assert!(
        stdout(&out).contains("BASELINE_COVERAGE_GAP"),
        "got: {}",
        stdout(&out)
    );

    fs::create_dir_all(crit.join("RT_C").join("ci-baseline")).unwrap();
    let out = run(&crit_log);
    assert_eq!(out.status.code(), Some(0), "stderr: {}", stderr(&out));
    assert!(
        stdout(&out).contains("coverage ok"),
        "got: {}",
        stdout(&out)
    );

    let garbage = work.join("nobench.log");
    write_file(&garbage, "garbage log with no criterion lines\n");
    let out = run(&garbage);
    assert_eq!(
        out.status.code(),
        Some(1),
        "unparseable log is the blind gate"
    );
    assert!(stdout(&out).contains("BASELINE_COVERAGE_GAP"));

    let out = run(&work.join("absent-crit.log"));
    assert_eq!(out.status.code(), Some(1), "absent log is the blind gate");
}

// ── F-27: canonical JSONL fidelity parse ─────────────────────────────────────

/// The canonical fixture of the removed bash block (F-27), verbatim.
const CANONICAL_JSONL: &str = r#"{"kind":"fidelity","label":"Model A @48000 Live","esr":null,"esr_db":"","snr_db":"1.5e2","mse":"1.2e-5","mrstft":"inf"}
{"kind":"fidelity","label":"Model B","esr":"","esr_db":null,"snr_db":"-inf","mse":null,"mrstft":"nan"}
{"label":"Model C @44100","esr":"0.0001","esr_db":"-40.0","snr_db":"50.0","mse":"3.0e-7","mrstft":"0.001"}
{"label":null,"esr":"1","esr_db":"2","snr_db":"3","mse":"4","mrstft":"5"}"#;

#[test]
fn f27_canonical_jsonl_parse() {
    let records = parse_fidelity_jsonl(CANONICAL_JSONL).expect("canonical JSONL must parse");
    assert_eq!(records.len(), 3, "null-label record must be dropped");

    let model_a = &records[0];
    assert_eq!(model_a.label, "Model A @48000 Live");
    assert_eq!(model_a.esr, MetricValue::Null, "null esr -> Null");
    assert_eq!(model_a.esr_db, MetricValue::Na, "empty esr_db -> N/A");
    assert_eq!(model_a.snr_db, MetricValue::Raw("1.5e2".into()));
    assert_eq!(model_a.mrstft, MetricValue::Raw("inf".into()));
    assert!(
        model_a.mrstft.as_finite().is_none(),
        "'inf' sentinel rejected"
    );
    assert_eq!(
        model_a.snr_db.as_finite(),
        Some("1.5e2"),
        "e-notation from JSONL accepted"
    );

    let model_b = &records[1];
    assert_eq!(model_b.mrstft, MetricValue::Raw("nan".into()));
    assert!(
        model_b.mrstft.as_finite().is_none(),
        "'nan' sentinel rejected"
    );

    let model_c = &records[2];
    assert_eq!(model_c.label, "Model C @44100");
    assert_eq!(model_c.esr, MetricValue::Raw("0.0001".into()));
    assert_eq!(model_c.mrstft, MetricValue::Raw("0.001".into()));
}

// ── S3-T01: unified C++ render build (subprocess, no extract_define) ─────────

/// Fake `cmake`: logs every invocation to `CMAKE_CALL_LOG` and fabricates a
/// render binary at `tools/render` (the Makefiles-generator layout). Unlike
/// the original bash-block fake, `--build` also populates the build dir so
/// the second (build) invocation never touches `/tools` on the host.
const FAKE_CMAKE: &str = r#"#!/bin/bash
echo "invoked: $*" >> "${CMAKE_CALL_LOG:-/dev/null}"
BUILD_D=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        -B|--build) shift; BUILD_D="$1" ;;
    esac
    shift
done
mkdir -p "$BUILD_D/tools"
printf '#!/bin/bash\nexit 0\n' > "$BUILD_D/tools/render"
chmod +x "$BUILD_D/tools/render"
exit 0
"#;

fn set_executable(path: &Path) {
    let mut perms = fs::metadata(path).expect("tool must exist").permissions();
    perms.set_mode(0o755);
    fs::set_permissions(path, perms).expect("chmod must succeed");
}

fn fake_toolchain_bin(work: &Path) -> PathBuf {
    let bin_dir = work.join("bin");
    fs::create_dir_all(&bin_dir).unwrap();
    write_file(&bin_dir.join("cmake"), FAKE_CMAKE);
    write_file(&bin_dir.join("fake-cxx"), "#!/bin/bash\nexit 0\n");
    write_file(&bin_dir.join("fake-clang"), "#!/bin/bash\nexit 0\n");
    for tool in ["cmake", "fake-cxx", "fake-clang"] {
        set_executable(&bin_dir.join(tool));
    }
    bin_dir
}

fn path_with_prefix(prefix: &Path) -> String {
    let existing = std::env::var("PATH").unwrap_or_default();
    if existing.is_empty() {
        prefix.display().to_string()
    } else {
        format!("{}:{}", prefix.display(), existing)
    }
}

fn render_script() -> PathBuf {
    project_root().join("utils/ensure_namcore_render.sh")
}

/// Absolute `bash` path: some S3-T01 scenarios replace `PATH` entirely, so
/// the interpreter itself must not be resolved through it.
fn bash_abs() -> String {
    let out = Command::new("bash")
        .arg("-c")
        .arg("command -v bash")
        .output()
        .expect("bash lookup must run");
    let path = String::from_utf8_lossy(&out.stdout).trim().to_string();
    assert!(!path.is_empty(), "bash must exist on the host");
    path
}

/// Exit-code contract of `ensure_namcore_render` (S3-T01), asserted through
/// `utils/ensure_namcore_render.sh` so the script wrapper itself stays
/// certified without `extract_define`:
/// 0 = binary ensured, 1 = no C++ compiler, 2 = cmake not found,
/// 3 = NAMCore vendor tree missing.
#[test]
fn s3t01_ensure_namcore_render_exit_codes() {
    let work = temp_dir();
    let script = render_script();
    let bash = bash_abs();
    let bin_dir = fake_toolchain_bin(&work);

    // rc 2: cmake absent from PATH. Keep only `dirname` (via symlink) so the
    // script still resolves its own location.
    let emptybin = work.join("emptybin");
    fs::create_dir_all(&emptybin).unwrap();
    let dirname = Command::new(&bash)
        .arg("-c")
        .arg("command -v dirname")
        .output()
        .expect("dirname lookup must run");
    let dirname = String::from_utf8_lossy(&dirname.stdout).trim().to_string();
    assert!(!dirname.is_empty(), "dirname must exist on the host");
    std::os::unix::fs::symlink(&dirname, emptybin.join("dirname")).unwrap();
    let out = Command::new(&bash)
        .arg(&script)
        .env("PATH", &emptybin)
        .output()
        .expect("render script must run");
    assert_eq!(
        out.status.code(),
        Some(2),
        "cmake missing must exit 2 (stderr: {})",
        stderr(&out)
    );

    // rc 1: CXX points at a missing compiler.
    let out = Command::new(&bash)
        .arg(&script)
        .env("PATH", path_with_prefix(&bin_dir))
        .env("CXX", work.join("missing-cxx"))
        .output()
        .expect("render script must run");
    assert_eq!(
        out.status.code(),
        Some(1),
        "invalid CXX must exit 1 (stderr: {})",
        stderr(&out)
    );

    // rc 3: NAMCore vendor tree missing.
    let out = Command::new(&bash)
        .arg(&script)
        .env("PATH", path_with_prefix(&bin_dir))
        .env("CXX", "fake-cxx")
        .env("NAM_CORE_DIR", work.join("nonexistent-core"))
        .output()
        .expect("render script must run");
    assert_eq!(
        out.status.code(),
        Some(3),
        "missing NAMCore must exit 3 (stderr: {})",
        stderr(&out)
    );
}

/// The fake-toolchain build flows of the removed bash block (S3-T01):
/// cold build, warm idempotency, compiler switch, `NAM_RENDER_FORCE`,
/// build-type change — all deterministic, no real C++ toolchain needed.
#[test]
fn s3t01_ensure_namcore_render_fake_toolchain_flows() {
    let work = temp_dir();
    let script = render_script();
    let bash = bash_abs();
    let bin_dir = fake_toolchain_bin(&work);
    let core = work.join("namcore-mock");
    fs::create_dir_all(&core).unwrap();
    let rb = work.join("rb");
    let call_log = work.join("cmake-calls.log");

    let run = |extra: &[(&str, &str)]| {
        let mut cmd = Command::new(&bash);
        cmd.arg(&script)
            .env("PATH", path_with_prefix(&bin_dir))
            .env("CXX", "fake-cxx")
            .env("NAM_CORE_DIR", &core)
            .env("NAM_RENDER_BUILD_DIR", &rb)
            .env("CMAKE_CALL_LOG", &call_log)
            .env_remove("NAM_RENDER_FORCE")
            .env_remove("NAM_RENDER_BUILD_TYPE");
        for (key, value) in extra {
            cmd.env(key, value);
        }
        cmd.output().expect("render script must run")
    };
    let cmake_calls = || {
        fs::read_to_string(&call_log)
            .map(|s| s.lines().count())
            .unwrap_or(0)
    };
    let render_bin = rb.join("tools/render");
    let fingerprint = |suffix: &str| format!("{suffix}:-w -fno-fast-math -ffp-contract=off");

    // Cold build: two cmake invocations (configure + build), stdout = the
    // render binary path, fingerprint persisted.
    let out = run(&[]);
    assert_eq!(
        out.status.code(),
        Some(0),
        "cold build must exit 0 (stderr: {})",
        stderr(&out)
    );
    assert_eq!(
        stdout(&out).trim(),
        render_bin.display().to_string(),
        "cold build prints the binary path"
    );
    assert_eq!(
        read_trim(&rb.join(".build_config")),
        "fake-cxx:Release:-w -fno-fast-math -ffp-contract=off"
    );
    let cold = cmake_calls();
    assert_eq!(cold, 2, "cold build invokes cmake twice");

    // Warm run: fingerprint matches -> no cmake invocation at all.
    let out = run(&[]);
    assert_eq!(out.status.code(), Some(0));
    assert_eq!(
        cmake_calls(),
        cold,
        "warm run skips cmake entirely (idempotent)"
    );

    // Compiler switch: fingerprint follows CXX, fresh configure + build.
    let out = run(&[("CXX", "fake-clang")]);
    assert_eq!(out.status.code(), Some(0));
    let switched = cmake_calls();
    assert_eq!(switched, cold + 2, "compiler change triggers fresh build");
    assert_eq!(
        read_trim(&rb.join(".build_config")),
        fingerprint("fake-clang:Release")
    );

    // NAM_RENDER_FORCE=1: wipes the build dir and rebuilds from scratch.
    let out = run(&[("CXX", "fake-clang"), ("NAM_RENDER_FORCE", "1")]);
    assert_eq!(out.status.code(), Some(0));
    let forced = cmake_calls();
    assert_eq!(
        forced,
        switched + 2,
        "NAM_RENDER_FORCE forces reconfigure+build"
    );

    // Build-type change: fingerprint follows NAM_RENDER_BUILD_TYPE.
    let out = run(&[("CXX", "fake-clang"), ("NAM_RENDER_BUILD_TYPE", "Debug")]);
    assert_eq!(out.status.code(), Some(0));
    assert_eq!(
        cmake_calls(),
        forced + 2,
        "build-type change triggers fresh build"
    );
    assert_eq!(
        read_trim(&rb.join(".build_config")),
        fingerprint("fake-clang:Debug")
    );
}
