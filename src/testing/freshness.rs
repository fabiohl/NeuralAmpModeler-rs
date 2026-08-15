// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Freshness verification for golden vectors and build artifacts.
//!
//! Centralizes the logic that used to live in `utils/_lib.sh::check_freshness`,
//! making it robust against shell locale/timestamp/portability differences.
//! Provides:
//!
//! - SHA-256 manifest verification against models, fixtures, and generators.
//! - Toolchain drift detection against `# TOOLCHAIN:` manifest annotations.
//! - mtime-based freshness check for source trees vs compiled artifacts.

use std::collections::HashSet;
use std::fmt;
use std::fs;
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::str::FromStr;
use std::time::SystemTime;

use sha2::{Digest, Sha256};

/// Gating mode for the freshness check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FreshnessMode {
    /// Emit warnings but always succeed (exit 0).
    WarnOnly,
    /// Fail on artifact integrity issues; warn on generator drift.
    ArtifactsHard,
    /// Fail on any drift, including generator/toolchain provenance.
    HardFail,
}

impl FromStr for FreshnessMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "warn-only" | "soft" => Ok(Self::WarnOnly),
            "artifacts-hard" => Ok(Self::ArtifactsHard),
            "hard-fail" => Ok(Self::HardFail),
            _ => Err(format!("unknown freshness mode: {s}")),
        }
    }
}

impl FreshnessMode {
    /// Whether generator/toolchain drift is a hard failure.
    pub fn generator_hard(self) -> bool {
        matches!(self, Self::HardFail)
    }

    /// Whether artifact integrity issues are a hard failure.
    pub fn artifact_hard(self) -> bool {
        matches!(self, Self::ArtifactsHard | Self::HardFail)
    }
}

impl fmt::Display for FreshnessMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WarnOnly => write!(f, "warn-only"),
            Self::ArtifactsHard => write!(f, "artifacts-hard"),
            Self::HardFail => write!(f, "hard-fail"),
        }
    }
}

/// Machine-readable outcome classification for `run_freshness_gate`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FreshnessReason {
    /// No drift detected.
    Ok,
    /// At least one fixture/model hash changed.
    StaleFixtures,
    /// At least one expected fixture/golden is missing on disk.
    MissingFixtures,
    /// A model file exists outside the manifest registry.
    OrphanFixture,
    /// Any other failure.
    FreshnessFailed,
}

impl fmt::Display for FreshnessReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ok => write!(f, "OK"),
            Self::StaleFixtures => write!(f, "STALE_FIXTURES"),
            Self::MissingFixtures => write!(f, "MISSING_FIXTURES"),
            Self::OrphanFixture => write!(f, "ORPHAN_FIXTURE"),
            Self::FreshnessFailed => write!(f, "FRESHNESS_FAILED"),
        }
    }
}

/// Result of a freshness gate run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FreshnessOutcome {
    /// Fixture/golden files that were expected but not found on disk.
    pub missing: Vec<PathBuf>,
    /// Files whose SHA-256 no longer matches the manifest.
    pub stale: Vec<PathBuf>,
    /// `.nam` model files not registered in the manifest.
    pub orphans: Vec<PathBuf>,
    /// Generator scripts whose SHA-256 changed.
    pub generator_drift: Vec<PathBuf>,
    /// Human-readable toolchain drift descriptions.
    pub toolchain_drift: Vec<String>,
    /// True when no artifact is missing, stale, or orphaned.
    pub artifact_integrity_ok: bool,
    /// True when no generator script drifted.
    pub generator_provenance_ok: bool,
    /// True when the toolchain fingerprint matches the manifest.
    pub toolchain_provenance_ok: bool,
    /// Canonical reason code for the bash wrapper.
    pub reason: FreshnessReason,
}

impl FreshnessOutcome {
    /// True when the gate passed in every dimension.
    pub fn is_ok(&self) -> bool {
        self.artifact_integrity_ok && self.generator_provenance_ok && self.toolchain_provenance_ok
    }
}

/// Errors that can occur while running the freshness gate.
#[derive(Debug, thiserror::Error)]
pub enum FreshnessError {
    /// Freshness manifest is missing.
    #[error("freshness manifest not found: {0}")]
    ManifestNotFound(PathBuf),
    /// I/O error while reading a file or directory.
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
    /// A non-comment manifest line has an unexpected number of fields.
    #[error("invalid manifest line {line}: {message}")]
    InvalidManifestLine {
        /// 1-based line number.
        line: usize,
        /// Human-readable explanation.
        message: String,
    },
}

/// Run the SHA-256 freshness gate against `.golden_manifest.sha256` under `root`.
///
/// The layout expected under `root` mirrors the crate checkout:
///
/// - `tests/fixtures/.golden_manifest.sha256` — the manifest.
/// - `tests/fixtures/models/*.nam` — model files.
/// - `tests/fixtures/*` — fixture/golden files.
///
/// Returns an error only for I/O or manifest parsing failures. Drift is reported
/// inside the returned [`FreshnessOutcome`].
pub fn check_freshness(
    root: impl AsRef<Path>,
    mode: FreshnessMode,
) -> Result<FreshnessOutcome, FreshnessError> {
    let root = root.as_ref();
    let manifest_path = root.join("tests/fixtures/.golden_manifest.sha256");
    let models_dir = root.join("tests/fixtures/models");
    let fixtures_dir = root.join("tests/fixtures");

    if !manifest_path.exists() {
        return Err(FreshnessError::ManifestNotFound(manifest_path));
    }

    let manifest_text = fs::read_to_string(&manifest_path)?;
    let manifest_toolchain = ToolchainFingerprint::from_manifest(&manifest_text);
    let current_toolchain = ToolchainFingerprint::current();
    let toolchain_drift = manifest_toolchain.drift_against(&current_toolchain);

    let mut missing = Vec::new();
    let mut stale = Vec::new();
    let mut orphans = Vec::new();
    let mut generator_drift = Vec::new();
    let mut registered_models: HashSet<String> = HashSet::new();

    enum Section {
        Catalog,
        Fixtures,
        Generators,
    }
    let mut section = Section::Catalog;

    for (line_no, line) in manifest_text.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        if let Some(expected) = trimmed.strip_prefix("# EXPECTED:") {
            let expected_file = expected.trim();
            if !fixtures_dir.join(expected_file).exists() {
                missing.push(PathBuf::from(expected_file));
            }
            continue;
        }

        if let Some(model) = trimmed.strip_prefix("# MODEL-REGISTRY:") {
            registered_models.insert(model.trim().to_string());
            continue;
        }

        if trimmed.starts_with('#') {
            if trimmed.contains("FIXTURES") {
                section = Section::Fixtures;
            } else if trimmed.contains("GENERATORS") {
                section = Section::Generators;
            }
            continue;
        }

        let fields: Vec<&str> = trimmed.split_whitespace().collect();
        match section {
            Section::Catalog => {
                if fields.len() != 4 {
                    return Err(FreshnessError::InvalidManifestLine {
                        line: line_no + 1,
                        message: format!("expected 4 fields, got {}", fields.len()),
                    });
                }
                let expected_model_sha = fields[0].to_lowercase();
                let nam_file = fields[2];

                registered_models.insert(nam_file.to_string());

                let model_path = models_dir.join(nam_file);
                if model_path.exists() {
                    let current_sha = sha256_file(&model_path)?;
                    if current_sha != expected_model_sha {
                        stale.push(PathBuf::from(nam_file));
                    }
                }
            }
            Section::Fixtures => {
                if fields.len() != 2 {
                    return Err(FreshnessError::InvalidManifestLine {
                        line: line_no + 1,
                        message: format!("expected 2 fields, got {}", fields.len()),
                    });
                }
                let expected_sha = fields[0].to_lowercase();
                let file_path = fields[1];
                let fixture_path = fixtures_dir.join(file_path);
                if fixture_path.exists() {
                    let current_sha = sha256_file(&fixture_path)?;
                    if current_sha != expected_sha {
                        stale.push(PathBuf::from(file_path));
                    }
                } else {
                    missing.push(PathBuf::from(file_path));
                }
            }
            Section::Generators => {
                if fields.len() != 2 {
                    return Err(FreshnessError::InvalidManifestLine {
                        line: line_no + 1,
                        message: format!("expected 2 fields, got {}", fields.len()),
                    });
                }
                let expected_sha = fields[0].to_lowercase();
                let file_path = fields[1];
                let gen_path = root.join(file_path);
                if gen_path.exists() {
                    let current_sha = sha256_file(&gen_path)?;
                    if current_sha != expected_sha {
                        generator_drift.push(PathBuf::from(file_path));
                    }
                }
            }
        }
    }

    // Reverse-check: scan models/ for .nam files not registered in the manifest.
    if models_dir.is_dir() {
        for entry in fs::read_dir(&models_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("nam") {
                let name = path.file_name().unwrap().to_string_lossy().to_string();
                if !registered_models.contains(&name) {
                    orphans.push(PathBuf::from(name));
                }
            }
        }
    }

    let artifact_integrity_ok = missing.is_empty() && stale.is_empty() && orphans.is_empty();
    let generator_provenance_ok = generator_drift.is_empty();
    let toolchain_provenance_ok = toolchain_drift.is_empty();

    let natural_reason = if !stale.is_empty() {
        FreshnessReason::StaleFixtures
    } else if !missing.is_empty() {
        FreshnessReason::MissingFixtures
    } else if !orphans.is_empty() {
        FreshnessReason::OrphanFixture
    } else if !generator_drift.is_empty() || !toolchain_drift.is_empty() {
        FreshnessReason::StaleFixtures
    } else {
        FreshnessReason::Ok
    };

    let should_fail = mode.artifact_hard() && !artifact_integrity_ok
        || mode.generator_hard() && (!generator_provenance_ok || !toolchain_provenance_ok);

    let reason = if should_fail {
        natural_reason
    } else {
        FreshnessReason::Ok
    };

    Ok(FreshnessOutcome {
        missing,
        stale,
        orphans,
        generator_drift,
        toolchain_drift,
        artifact_integrity_ok,
        generator_provenance_ok,
        toolchain_provenance_ok,
        reason,
    })
}

/// Compute the SHA-256 hex digest of a file, lower-case.
fn sha256_file(path: &Path) -> io::Result<String> {
    let mut file = fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 8192];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }
    Ok(hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect())
}

/// Toolchain fingerprint extracted from manifest comments or the current host.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct ToolchainFingerprint {
    /// C++ compiler version string.
    pub cxx: Option<String>,
    /// CMake version string.
    pub cmake: Option<String>,
    /// GNU libc version string.
    pub glibc: Option<String>,
    /// Kernel release string.
    pub os: Option<String>,
    /// C++ compiler flags stored in the manifest (not compared).
    pub cxx_flags: Option<String>,
}

impl ToolchainFingerprint {
    /// Parse `# TOOLCHAIN:` annotations from the manifest text.
    pub fn from_manifest(text: &str) -> Self {
        let mut fp = Self::default();
        for line in text.lines() {
            let trimmed = line.trim();
            if let Some(v) = trimmed.strip_prefix("# TOOLCHAIN: cxx:") {
                fp.cxx = Some(v.trim().to_string());
            } else if let Some(v) = trimmed.strip_prefix("# TOOLCHAIN: cmake:") {
                fp.cmake = Some(v.trim().to_string());
            } else if let Some(v) = trimmed.strip_prefix("# TOOLCHAIN: glibc:") {
                fp.glibc = Some(v.trim().to_string());
            } else if let Some(v) = trimmed.strip_prefix("# TOOLCHAIN: os:") {
                fp.os = Some(v.trim().to_string());
            } else if let Some(v) = trimmed.strip_prefix("# TOOLCHAIN: cxx-flags:") {
                fp.cxx_flags = Some(v.trim().to_string());
            }
        }
        fp
    }

    /// Probe the current host for the same toolchain fields.
    pub fn current() -> Self {
        Self {
            cxx: first_line_of_command(&[cxx_compiler().as_str(), "--version"]),
            cmake: first_line_of_command(&["cmake", "--version"]),
            glibc: first_line_of_command(&["ldd", "--version"])
                .or_else(|| first_line_of_command(&["getconf", "GNU_LIBC_VERSION"])),
            os: first_line_of_command(&["uname", "-r"]),
            cxx_flags: None,
        }
    }

    /// Compare this manifest fingerprint against the current host.
    ///
    /// Returns a list of human-readable drift descriptions. Only fields present
    /// in the manifest are compared; missing manifest fields are ignored.
    pub fn drift_against(&self, current: &Self) -> Vec<String> {
        let mut drifts = Vec::new();
        if let (Some(manifest), Some(now)) = (&self.cxx, &current.cxx)
            && manifest != now
        {
            drifts.push(format!("compiler changed: manifest={manifest} now={now}"));
        }
        if let (Some(manifest), Some(now)) = (&self.cmake, &current.cmake)
            && manifest != now
        {
            drifts.push(format!("cmake changed: manifest={manifest} now={now}"));
        }
        if let (Some(manifest), Some(now)) = (&self.glibc, &current.glibc)
            && manifest != now
        {
            drifts.push(format!("glibc changed: manifest={manifest} now={now}"));
        }
        if let (Some(manifest), Some(now)) = (&self.os, &current.os)
            && manifest != now
        {
            drifts.push(format!("kernel changed: manifest={manifest} now={now}"));
        }
        drifts
    }
}

fn cxx_compiler() -> String {
    std::env::var("CXX").unwrap_or_else(|_| "g++".to_string())
}

fn first_line_of_command(parts: &[&str]) -> Option<String> {
    if parts.is_empty() {
        return None;
    }
    let output = Command::new(parts[0]).args(&parts[1..]).output().ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .next()
        .map(|s| s.trim().to_string())
}

/// Compare the newest modification time among `sources` against each existing
/// `artifact`, returning the paths of artifacts older than the newest source.
///
/// `sources` may be files or directories; directories are recursively scanned.
/// Non-existent artifacts are ignored. If no source exists, the result is
/// empty.
pub fn check_artifact_freshness_mtime(
    sources: &[impl AsRef<Path>],
    artifacts: &[impl AsRef<Path>],
) -> io::Result<Vec<PathBuf>> {
    let newest = newest_mtime(sources)?;
    let Some(newest) = newest else {
        return Ok(Vec::new());
    };

    let mut stale = Vec::new();
    for artifact in artifacts {
        let artifact = artifact.as_ref();
        if !artifact.exists() {
            continue;
        }
        if let Some(mtime) = mtime(artifact)?
            && mtime < newest
        {
            stale.push(artifact.to_path_buf());
        }
    }
    Ok(stale)
}

/// Newest modification time among a list of files or directories.
fn newest_mtime(paths: &[impl AsRef<Path>]) -> io::Result<Option<SystemTime>> {
    let mut newest: Option<SystemTime> = None;
    for path in paths {
        let path = path.as_ref();
        if path.is_dir() {
            for entry in fs::read_dir(path)? {
                let entry = entry?;
                let sub = entry.path();
                if let Some(candidate) = newest_mtime(&[sub])? {
                    newest = Some(newest.map_or(candidate, |n| n.max(candidate)));
                }
            }
        } else if path.is_file()
            && let Some(mtime) = mtime(path)?
        {
            newest = Some(newest.map_or(mtime, |n| n.max(mtime)));
        }
    }
    Ok(newest)
}

fn mtime(path: &Path) -> io::Result<Option<SystemTime>> {
    match fs::metadata(path) {
        Ok(meta) => Ok(Some(meta.modified()?)),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e),
    }
}

#[cfg(test)]
#[path = "freshness_test.rs"]
mod tests;
