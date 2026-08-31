// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Fail-closed binary surface scanner for the default-build AVX-512 absence
//! certificate.
//!
//! The scanner proves the absence of EVEX-encoded instructions in the
//! machine code of a default (`not(feature = "avx512")`) build by parsing the
//! *encoding bytes* that [`llvm-objdump`](https://llvm.org/docs/CommandGuide/llvm-objdump.html)
//! emits for every disassembled instruction — never by register names or
//! mnemonics (those can miss AVX-512VL code that uses only low `xmm0..15` /
//! `ymm0..15` registers and opmasks, which was the previous guard's blind
//! spot).
//!
//! Every EVEX-encoded x86-64 instruction starts with the `0x62` prefix byte.
//! Therefore a clean certificate requires that no disassembled instruction
//! line in any executable section of any ELF object begins with `0x62`.
//!
//! # Fail-closed contract
//!
//! The scanner returns an error instead of a clean report when:
//!
//! - the inspection tool (`llvm-objdump` / `llvm-nm`) cannot be resolved;
//! - the tool exits with a non-zero status;
//! - the tool produces unexpectedly empty output;
//! - the artifact cannot be read or is empty;
//! - the artifact is neither an ELF file nor a GNU `ar` archive;
//! - an archive member is neither an ELF object nor a recognized
//!   metadata/bitcode member;
//! - an archive contains no disassemblable ELF member at all.
//!
//! A successful scan reports the number of inspected executable sections,
//! instructions and archive members so a human can audit that the scan was
//! not vacuous.

use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::Duration;

const ELF_MAGIC: &[u8] = b"\x7fELF";
const AR_MAGIC: &[u8] = b"!<arch>\n";
const BITCODE_MAGIC: &[u8] = b"BC\xc0\xde";

/// The EVEX encoding prefix: every EVEX-encoded instruction begins with this byte.
const EVEX_PREFIX: u8 = 0x62;

/// Pairs of tool-kind identifiers used to resolve a concrete binary.
#[derive(Debug)]
pub enum ToolKind {
    /// The disassembler used for the authoritative EVEX opcode scan.
    LlvmObjdump,
    /// The symbol-table reader used for the defensive symbol scan.
    LlvmNm,
}

impl ToolKind {
    fn tool_name(&self) -> &'static str {
        match self {
            ToolKind::LlvmObjdump => "llvm-objdump",
            ToolKind::LlvmNm => "llvm-nm",
        }
    }

    fn path_names(&self) -> &'static [&'static str] {
        match self {
            ToolKind::LlvmObjdump => &[
                "llvm-objdump",
                "llvm-objdump-21",
                "llvm-objdump-20",
                "llvm-objdump-19",
                "llvm-objdump-18",
                "objdump",
            ],
            ToolKind::LlvmNm => &[
                "llvm-nm",
                "llvm-nm-21",
                "llvm-nm-20",
                "llvm-nm-19",
                "llvm-nm-18",
                "nm",
            ],
        }
    }
}

impl std::fmt::Display for ToolKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.tool_name())
    }
}

/// A single EVEX-encoded instruction found in an executable section.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvexViolation {
    /// Archive member the violation was found in, when scanning an archive.
    pub member: Option<String>,
    /// Executable section the instruction lives in.
    pub section: String,
    /// Virtual offset of the instruction within the object.
    pub address: u64,
    /// The raw disassembly line (for human auditing).
    pub line: String,
}

/// Machine-readable outcome of an EVEX scan.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EvexScanReport {
    /// Path of the scanned artifact.
    pub artifact: PathBuf,
    /// Number of executable sections disassembled (aggregated over members).
    pub sections_scanned: usize,
    /// Number of instructions decoded from encoding bytes.
    pub instructions_scanned: usize,
    /// Number of archive members that are ELF objects and were disassembled.
    pub members_scanned: usize,
    /// Number of archive members that are LLVM bitcode (thin-LTO IR only).
    pub bitcode_members: usize,
    /// Number of archive members recognized as rustc metadata.
    pub metadata_members: usize,
    /// Every EVEX-prefixed instruction found.
    pub evex_violations: Vec<EvexViolation>,
}

impl EvexScanReport {
    /// True when no EVEX instruction was found in any disassembled section.
    pub fn is_clean(&self) -> bool {
        self.evex_violations.is_empty()
    }
}

/// Errors produced by the fail-closed scanner.
#[derive(Debug, thiserror::Error)]
pub enum BinGuardError {
    /// The artifact could not be read.
    #[error("cannot read artifact '{path}': {source}")]
    ReadFailed {
        /// Artifact path.
        path: PathBuf,
        /// Underlying I/O error.
        source: std::io::Error,
    },
    /// The artifact is empty.
    #[error("artifact '{path}' is empty")]
    EmptyInput {
        /// Artifact path.
        path: PathBuf,
    },
    /// The artifact is neither ELF nor an `ar` archive.
    #[error("unrecognized binary format in '{path}' (first bytes: {magic:02X?})")]
    UnsupportedFormat {
        /// Artifact path.
        path: PathBuf,
        /// Leading bytes that did not match any known format.
        magic: Vec<u8>,
    },
    /// A required inspection tool could not be resolved.
    #[error(
        "required tool '{tool}' not found in rustc sysroot or PATH — \
         cannot certify binary absence of AVX-512"
    )]
    ToolMissing {
        /// Tool kind that is missing.
        tool: ToolKind,
    },
    /// A tool exited with a non-zero status.
    #[error("tool '{tool}' failed on '{path}' with exit code {code}: {stderr}")]
    ToolFailed {
        /// Tool that failed.
        tool: String,
        /// Artifact being inspected.
        path: PathBuf,
        /// Non-zero exit code.
        code: i32,
        /// Captured stderr.
        stderr: String,
    },
    /// A tool produced empty output where non-empty output was required.
    #[error("tool '{tool}' produced unexpectedly empty output for '{path}'")]
    EmptyOutput {
        /// Tool that produced the empty output.
        tool: String,
        /// Artifact being inspected.
        path: PathBuf,
    },
    /// An archive member is neither ELF, bitcode nor known rustc metadata.
    #[error("archive member '{member}' in '{archive}' is not a disassemblable ELF object")]
    UnsupportedArchiveMember {
        /// Archive containing the member.
        archive: PathBuf,
        /// Member name.
        member: String,
    },
    /// An archive has no members at all.
    #[error("archive '{path}' has no members")]
    EmptyArchive {
        /// Archive path.
        path: PathBuf,
    },
    /// An archive contains members but none of them is a disassemblable ELF
    /// object, so no machine-code absence could be proven.
    #[error(
        "archive '{path}' has no disassemblable ELF members (bitcode/metadata \
         only) — nothing to certify"
    )]
    NoDisassemblableMembers {
        /// Archive path.
        path: PathBuf,
    },
}

/// Resolve an inspection tool, preferring the `rustc` sysroot LLVM tools and
/// falling back to PATH. Fails closed when no usable tool exists.
pub fn resolve_llvm_tool(kind: ToolKind) -> Result<PathBuf, BinGuardError> {
    if let Ok(sysroot_out) = Command::new("rustc").arg("--print").arg("sysroot").output()
        && sysroot_out.status.success()
    {
        let sysroot = String::from_utf8_lossy(&sysroot_out.stdout)
            .trim()
            .to_string();
        if let Ok(vv) = Command::new("rustc").arg("-vV").output()
            && vv.status.success()
        {
            let vv_str = String::from_utf8_lossy(&vv.stdout);
            for line in vv_str.lines() {
                if let Some(host) = line.strip_prefix("host: ") {
                    let candidate = PathBuf::from(&sysroot)
                        .join("lib/rustlib")
                        .join(host.trim())
                        .join("bin")
                        .join(kind.tool_name());
                    if candidate.is_file() {
                        return Ok(candidate);
                    }
                }
            }
        }
    }

    for name in kind.path_names() {
        if let Ok(output) = Command::new(name).arg("--version").output()
            && output.status.success()
        {
            return Ok(PathBuf::from(name));
        }
    }

    Err(BinGuardError::ToolMissing { tool: kind })
}

/// Run a tool with the artifact path as its last argument, requiring exit code
/// zero and (when `require_output`) non-empty stdout. Fail-closed on any
/// violation of those contracts.
fn run_tool_on_path(
    tool: &Path,
    args: &[&OsStr],
    artifact: &Path,
    require_output: bool,
) -> Result<String, BinGuardError> {
    let mut attempts = 0;
    let output = loop {
        match Command::new(tool).args(args).arg(artifact).output() {
            Ok(output) => break output,
            Err(source) if source.raw_os_error() == Some(libc::ETXTBSY) && attempts < 20 => {
                attempts += 1;
                thread::sleep(Duration::from_millis(5));
            }
            Err(source) => {
                return Err(BinGuardError::ReadFailed {
                    path: artifact.to_path_buf(),
                    source,
                });
            }
        }
    };
    if !output.status.success() {
        return Err(BinGuardError::ToolFailed {
            tool: tool.display().to_string(),
            path: artifact.to_path_buf(),
            code: output.status.code().unwrap_or(-1),
            stderr: String::from_utf8_lossy(&output.stderr).trim().to_string(),
        });
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    if require_output && stdout.trim().is_empty() {
        return Err(BinGuardError::EmptyOutput {
            tool: tool.display().to_string(),
            path: artifact.to_path_buf(),
        });
    }
    Ok(stdout.into_owned())
}

/// Scan a file (ELF object/executable or GNU `ar` archive) for EVEX-prefixed
/// instructions. Returns a structured report, or a fail-closed error.
pub fn scan_for_evex(objdump: &Path, artifact: &Path) -> Result<EvexScanReport, BinGuardError> {
    let bytes = fs::read(artifact).map_err(|source| BinGuardError::ReadFailed {
        path: artifact.to_path_buf(),
        source,
    })?;
    if bytes.is_empty() {
        return Err(BinGuardError::EmptyInput {
            path: artifact.to_path_buf(),
        });
    }

    let mut report = EvexScanReport {
        artifact: artifact.to_path_buf(),
        ..EvexScanReport::default()
    };

    if bytes.starts_with(ELF_MAGIC) {
        let stdout = run_tool_on_path(objdump, &[OsStr::new("-d")], artifact, true)?;
        parse_disassembly(&stdout, None, &mut report);
        return Ok(report);
    }

    if bytes.starts_with(AR_MAGIC) {
        return scan_archive_bytes(objdump, artifact, &bytes, &mut report);
    }

    let magic = bytes.iter().take(16).copied().collect();
    Err(BinGuardError::UnsupportedFormat {
        path: artifact.to_path_buf(),
        magic,
    })
}

/// A single member parsed out of a GNU `ar` archive.
struct ArMember {
    name: String,
    data: Vec<u8>,
}

/// Parse a GNU `ar` archive (as produced for `.rlib` files), supporting the
/// extended name table (`//`), BSD `#1/<len>` long names and the archive
/// symbol index (`/`).
fn parse_archive(bytes: &[u8]) -> Result<Vec<ArMember>, BinGuardError> {
    if bytes.len() < AR_MAGIC.len() || !bytes.starts_with(AR_MAGIC) {
        return Err(BinGuardError::UnsupportedFormat {
            path: PathBuf::from("<archive>"),
            magic: bytes.iter().take(16).copied().collect(),
        });
    }
    let mut members = Vec::new();
    let mut extended_names: Option<Vec<u8>> = None;
    let mut pos = AR_MAGIC.len();

    while pos + 60 <= bytes.len() {
        let header = &bytes[pos..pos + 60];
        pos += 60;
        if &header[58..60] != b"`\n" {
            return Err(BinGuardError::UnsupportedFormat {
                path: PathBuf::from("<archive>"),
                magic: header[58..60].to_vec(),
            });
        }
        let name_cow = String::from_utf8_lossy(&header[0..16]);
        let name_field = name_cow.trim().to_string();
        let size_cow = String::from_utf8_lossy(&header[48..58]);
        let size_field = size_cow.trim();
        let size: usize = size_field
            .parse()
            .map_err(|_| BinGuardError::UnsupportedFormat {
                path: PathBuf::from("<archive>"),
                magic: size_field.as_bytes().to_vec(),
            })?;
        if pos + size > bytes.len() {
            return Err(BinGuardError::UnsupportedFormat {
                path: PathBuf::from("<archive>"),
                magic: bytes[pos..].iter().take(16).copied().collect(),
            });
        }
        let mut data = bytes[pos..pos + size].to_vec();
        pos += size;
        if size % 2 == 1 {
            pos += 1; // GNU ar pads each member to an even boundary.
        }

        if name_field == "//" {
            extended_names = Some(data);
            continue;
        }
        if name_field == "/" || name_field == "/SYM64/" {
            continue;
        }

        let name = if let Some(idx_str) = name_field.strip_prefix('/') {
            let offset: usize = idx_str.parse().unwrap_or(0);
            let table =
                extended_names
                    .as_deref()
                    .ok_or_else(|| BinGuardError::UnsupportedFormat {
                        path: PathBuf::from("<archive>"),
                        magic: b"<no-name-table>".to_vec(),
                    })?;
            name_from_extended_table(table, offset)?
        } else if let Some(len_str) = name_field.strip_prefix("#1/") {
            let len: usize = len_str.parse().unwrap_or(0);
            if len > data.len() {
                return Err(BinGuardError::UnsupportedFormat {
                    path: PathBuf::from("<archive>"),
                    magic: b"<bad-bsd-name>".to_vec(),
                });
            }
            let name = String::from_utf8_lossy(&data[..len]).trim().to_string();
            data.drain(..len);
            name
        } else {
            name_field.trim_end_matches('/').to_string()
        };

        members.push(ArMember { name, data });
    }

    Ok(members)
}

/// Resolve a member name from the GNU extended name table starting at `offset`.
fn name_from_extended_table(table: &[u8], offset: usize) -> Result<String, BinGuardError> {
    let tail = table
        .get(offset..)
        .ok_or_else(|| BinGuardError::UnsupportedFormat {
            path: PathBuf::from("<archive>"),
            magic: b"<bad-name-offset>".to_vec(),
        })?;
    let end = tail.iter().position(|&b| b == b'\n').unwrap_or(tail.len());
    let name = String::from_utf8_lossy(&tail[..end]);
    Ok(name.trim_end_matches('/').to_string())
}

/// Unique counter for temporary member files (tests may run in parallel).
static TEMP_SEQ: AtomicU64 = AtomicU64::new(0);

fn scan_archive_bytes(
    objdump: &Path,
    artifact: &Path,
    bytes: &[u8],
    report: &mut EvexScanReport,
) -> Result<EvexScanReport, BinGuardError> {
    let members = parse_archive(bytes)?;
    if members.is_empty() {
        return Err(BinGuardError::EmptyArchive {
            path: artifact.to_path_buf(),
        });
    }

    for member in &members {
        if member.data.starts_with(ELF_MAGIC) {
            scan_elf_member(objdump, artifact, member, report)?;
        } else if member.data.starts_with(BITCODE_MAGIC) {
            report.bitcode_members += 1;
        } else if member.name == "lib.rmeta" || member.name == "lib.rmeta-link" {
            report.metadata_members += 1;
        } else {
            return Err(BinGuardError::UnsupportedArchiveMember {
                archive: artifact.to_path_buf(),
                member: member.name.clone(),
            });
        }
    }

    if report.members_scanned == 0 {
        return Err(BinGuardError::NoDisassemblableMembers {
            path: artifact.to_path_buf(),
        });
    }

    Ok(std::mem::take(report))
}

/// Disassemble one archive member by materializing it to a temporary file and
/// scanning it like a standalone ELF object.
fn scan_elf_member(
    objdump: &Path,
    artifact: &Path,
    member: &ArMember,
    report: &mut EvexScanReport,
) -> Result<(), BinGuardError> {
    let seq = TEMP_SEQ.fetch_add(1, Ordering::Relaxed);
    let tmp = std::env::temp_dir().join(format!(
        "nam-bin-guard-{}-{seq}-{}",
        std::process::id(),
        member.name
    ));
    fs::write(&tmp, &member.data).map_err(|source| BinGuardError::ReadFailed {
        path: artifact.to_path_buf(),
        source,
    })?;
    let result = run_tool_on_path(objdump, &[OsStr::new("-d")], &tmp, true);
    let _ = fs::remove_file(&tmp);
    let stdout = result?;

    report.members_scanned += 1;
    parse_disassembly(&stdout, Some(member.name.clone()), report);
    Ok(())
}

/// Parse `llvm-objdump -d` output, counting sections and instructions and
/// flagging every instruction whose encoding starts with the EVEX `0x62` byte.
fn parse_disassembly(output: &str, member: Option<String>, report: &mut EvexScanReport) {
    let mut current_section = String::from("<no-section>");
    for line in output.lines() {
        if let Some(name) = line.strip_prefix("Disassembly of section ") {
            current_section = name.trim_end_matches(':').trim().to_string();
            report.sections_scanned += 1;
            continue;
        }
        if let Some((address, bytes, mnemonic)) = parse_instruction_line(line) {
            report.instructions_scanned += 1;
            if bytes.first() == Some(&EVEX_PREFIX) {
                report.evex_violations.push(EvexViolation {
                    member: member.clone(),
                    section: current_section.clone(),
                    address,
                    line: format!(
                        "0x{address:x}: {} ({})",
                        mnemonic,
                        bytes
                            .iter()
                            .map(|b| format!("{b:02x}"))
                            .collect::<Vec<_>>()
                            .join(" ")
                    ),
                });
            }
        }
    }
}

/// Data-directive mnemonics that `llvm-objdump` can emit for embedded data in
/// executable sections. They are not instructions and must not count towards
/// the instruction scan.
fn is_data_directive(mnemonic: &str) -> bool {
    matches!(
        mnemonic,
        ".byte"
            | ".short"
            | ".word"
            | ".long"
            | ".quad"
            | ".8byte"
            | ".4byte"
            | ".2byte"
            | ".zero"
            | ".ascii"
            | ".asciz"
            | ".string"
            | ".text"
            | ".globl"
            | ".size"
    )
}

/// Parse a single disassembly instruction line of the form
/// `      addr: 62 f1 7c a9 10 06    mnemonic operands`.
///
/// Returns the address, the raw encoding bytes and the mnemonic+operands text.
/// Lines that do not describe a decoded instruction (section/symbol headers,
/// data directives, relocation notes) yield `None`.
fn parse_instruction_line(line: &str) -> Option<(u64, Vec<u8>, String)> {
    let trimmed = line.trim_start();
    let colon = trimmed.find(':')?;
    let addr_str = &trimmed[..colon];
    if addr_str.is_empty() || !addr_str.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    let address = u64::from_str_radix(addr_str, 16).ok()?;
    let rest = trimmed[colon + 1..].trim_start();
    if rest.is_empty() {
        return None;
    }

    let mut tokens = rest.split_whitespace();
    let mut bytes = Vec::new();
    for token in tokens.by_ref() {
        if token.len() == 2 && token.bytes().all(|b| b.is_ascii_hexdigit()) {
            bytes.push(u8::from_str_radix(token, 16).ok()?);
        } else {
            if is_data_directive(token) {
                return None;
            }
            let tail = tokens.collect::<Vec<_>>().join(" ");
            let mnemonic = if tail.is_empty() {
                token.to_string()
            } else {
                format!("{token} {tail}")
            };
            return Some((address, bytes, mnemonic));
        }
    }
    None
}

/// Symbol-table defense scan: run `llvm-nm --demangle` on the artifact and
/// return every line containing any forbidden symbol fragment. Fail-closed on
/// tool resolution/exit errors; an empty symbol table is legitimate (stripped
/// or metadata-only objects) and is reported as clean.
pub fn scan_symbols(
    nm: &Path,
    artifact: &Path,
    forbidden: &[&str],
) -> Result<Vec<String>, BinGuardError> {
    // An empty symbol table is legitimate (stripped or metadata-only objects),
    // so `require_output` is false here; a non-zero exit still fails closed.
    let stdout = run_tool_on_path(nm, &[OsStr::new("--demangle")], artifact, false)?;
    let mut matches = Vec::new();
    for line in stdout.lines() {
        if forbidden.iter().any(|frag| line.contains(frag)) {
            matches.push(line.to_string());
        }
    }
    Ok(matches)
}

#[cfg(test)]
#[path = "bin_guard_test.rs"]
mod bin_guard_test;
