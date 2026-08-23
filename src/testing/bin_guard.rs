// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Fail-closed binary surface scanner for the default-build AVX-512 absence
//! certificate (Sprint 1 / F-ROB-01).
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
    let output = Command::new(tool)
        .args(args)
        .arg(artifact)
        .output()
        .map_err(|source| BinGuardError::ReadFailed {
            path: artifact.to_path_buf(),
            source,
        })?;
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
mod tests {
    use super::*;
    use std::io::Write;

    fn sysroot_objdump() -> PathBuf {
        resolve_llvm_tool(ToolKind::LlvmObjdump)
            .expect("bin_guard: llvm-objdump required for scanner tests")
    }

    fn sysroot_nm() -> PathBuf {
        resolve_llvm_tool(ToolKind::LlvmNm).expect("bin_guard: llvm-nm required for scanner tests")
    }

    /// Build a minimal ELF64 relocatable object containing `text` bytes in a
    /// single executable `.text` section.
    fn mini_elf(text: &[u8]) -> Vec<u8> {
        const ELF_HDR: usize = 64;
        const SHDR_SIZE: usize = 64;
        const SECTION_COUNT: usize = 3; // null + .text + .shstrtab
        const SHDRTAB_OFF: usize = ELF_HDR; // section headers start after the ELF header
        const TEXT_OFF: usize = SHDRTAB_OFF + SHDR_SIZE * SECTION_COUNT;

        // Section header string table: "\0.text\0.shstrtab\0"
        let mut shstrtab = vec![0u8];
        shstrtab.extend_from_slice(b".text\0");
        shstrtab.extend_from_slice(b".shstrtab\0");
        let text_name_idx = 1usize; // ".text" offset within .shstrtab
        let shstrtab_name_idx = 7usize; // ".shstrtab" offset within .shstrtab
        let shstrtab_off = TEXT_OFF + text.len();

        let mut elf = Vec::with_capacity(shstrtab_off + shstrtab.len());

        // ELF64 header.
        let mut hdr = [0u8; ELF_HDR];
        hdr[0..4].copy_from_slice(b"\x7fELF");
        hdr[4] = 2; // ELFCLASS64
        hdr[5] = 1; // ELFDATA2LSB
        hdr[6] = 1; // EV_CURRENT
        hdr[7] = 0; // ELFOSABI_NONE
        hdr[16..18].copy_from_slice(&1u16.to_le_bytes()); // ET_REL
        hdr[18..20].copy_from_slice(&62u16.to_le_bytes()); // EM_X86_64
        hdr[20..24].copy_from_slice(&1u32.to_le_bytes()); // e_version
        hdr[40..48].copy_from_slice(&(SHDRTAB_OFF as u64).to_le_bytes()); // e_shoff
        hdr[52..54].copy_from_slice(&(ELF_HDR as u16).to_le_bytes()); // e_ehsize
        hdr[54..56].copy_from_slice(&0u16.to_le_bytes()); // e_phentsize (no program headers)
        hdr[56..58].copy_from_slice(&0u16.to_le_bytes()); // e_phnum
        hdr[58..60].copy_from_slice(&(SHDR_SIZE as u16).to_le_bytes()); // e_shentsize
        hdr[60..62].copy_from_slice(&(SECTION_COUNT as u16).to_le_bytes()); // e_shnum
        hdr[62..64].copy_from_slice(&2u16.to_le_bytes()); // e_shstrndx
        elf.extend_from_slice(&hdr);

        // Section 0: null (all zeros).
        elf.extend_from_slice(&[0u8; SHDR_SIZE]);

        // Section 1: .text.
        let mut sh_text = [0u8; SHDR_SIZE];
        sh_text[0..4].copy_from_slice(&(text_name_idx as u32).to_le_bytes()); // sh_name
        sh_text[4..8].copy_from_slice(&1u32.to_le_bytes()); // SHT_PROGBITS
        sh_text[8..16].copy_from_slice(&6u64.to_le_bytes()); // SHF_ALLOC | SHF_EXECINSTR
        sh_text[24..32].copy_from_slice(&(TEXT_OFF as u64).to_le_bytes()); // sh_offset
        sh_text[32..40].copy_from_slice(&(text.len() as u64).to_le_bytes()); // sh_size
        sh_text[48..56].copy_from_slice(&16u64.to_le_bytes()); // sh_addralign
        elf.extend_from_slice(&sh_text);

        // Section 2: .shstrtab.
        let mut sh_shstr = [0u8; SHDR_SIZE];
        sh_shstr[0..4].copy_from_slice(&(shstrtab_name_idx as u32).to_le_bytes()); // sh_name
        sh_shstr[4..8].copy_from_slice(&3u32.to_le_bytes()); // SHT_STRTAB
        sh_shstr[24..32].copy_from_slice(&(shstrtab_off as u64).to_le_bytes());
        sh_shstr[32..40].copy_from_slice(&(shstrtab.len() as u64).to_le_bytes());
        sh_shstr[48..56].copy_from_slice(&1u64.to_le_bytes()); // sh_addralign
        elf.extend_from_slice(&sh_shstr);

        elf.extend_from_slice(text);
        elf.extend_from_slice(&shstrtab);
        elf
    }

    /// Build a GNU `ar` archive from named members.
    fn mini_archive(members: &[(&str, &[u8])]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"!<arch>\n");
        for (name, data) in members {
            let mut name_field = [b' '; 16];
            let n = name.len().min(16);
            name_field[..n].copy_from_slice(&name.as_bytes()[..n]);
            let mut header = Vec::with_capacity(60);
            header.extend_from_slice(&name_field);
            header.extend_from_slice(b"            "); // mtime
            header.extend_from_slice(b"      "); // uid
            header.extend_from_slice(b"      "); // gid
            header.extend_from_slice(b"        "); // mode
            header.extend_from_slice(format!("{:>10}", data.len()).as_bytes()); // size
            header.extend_from_slice(b"`\n");
            out.extend_from_slice(&header);
            out.extend_from_slice(data);
            if data.len() % 2 == 1 {
                out.push(b'\n');
            }
        }
        out
    }

    fn write_temp(bytes: &[u8], name: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!("nam-bin-guard-{}-{name}", std::process::id()));
        let mut f = fs::File::create(&path).expect("write fixture");
        f.write_all(bytes).expect("write fixture bytes");
        path
    }

    fn write_fake_tool(script: &str, label: &str) -> PathBuf {
        use std::os::unix::fs::PermissionsExt;
        let mut path = std::env::temp_dir();
        path.push(format!("nam-fake-objdump-{}-{label}", std::process::id()));
        let mut f = fs::File::create(&path).expect("write fake tool");
        f.write_all(script.as_bytes()).expect("write fake tool");
        fs::set_permissions(&path, fs::Permissions::from_mode(0o755)).expect("chmod");
        path
    }

    // EVEX byte sequences (from real AVX-512 compilations).
    const EVEX_VL256_MASK_ZERO: &[u8] = &[0x62, 0xf1, 0x7c, 0xa9, 0x10, 0x06]; // vmovups (%rsi),%ymm0{%k1}{z}
    const EVEX_VL256_MASK: &[u8] = &[0x62, 0xf1, 0x7c, 0x29, 0x11, 0x07]; // vmovups %ymm0,(%rdi){%k1}
    const EVEX_ZMM0: &[u8] = &[0x62, 0xf1, 0x7c, 0x48, 0x10, 0x07]; // vmovups (%rdi),%zmm0
    const EVEX_ZMM16_31: &[u8] = &[0x62, 0xa1, 0x74, 0x40, 0x58, 0xd0]; // vaddps %zmm16,%zmm17,%zmm18
    // Clean x86-64-v3 (VEX/SSE only) byte sequences.
    const CLEAN_VEX: &[u8] = &[0xc5, 0xfc, 0x28, 0x02]; // vmovaps (%rdx),%ymm0
    const CLEAN_SSE: &[u8] = &[0x48, 0x8b, 0x44, 0x24, 0x08]; // movq 0x8(%rsp),%rax

    #[test]
    fn resolves_llvm_tools_from_sysroot() {
        let objdump = resolve_llvm_tool(ToolKind::LlvmObjdump)
            .expect("llvm-objdump must resolve via rustc sysroot");
        assert!(objdump.is_file());
        let nm = resolve_llvm_tool(ToolKind::LlvmNm).expect("llvm-nm must resolve");
        assert!(nm.is_file());
    }

    #[test]
    fn rejects_evex_vl256_low_registers() {
        let obj = mini_elf(EVEX_VL256_MASK_ZERO);
        let path = write_temp(&obj, "evex-vl256.o");
        let report = scan_for_evex(&sysroot_objdump(), &path).expect("scan must not error");
        assert!(!report.is_clean());
        assert_eq!(report.evex_violations.len(), 1);
        let v = &report.evex_violations[0];
        assert!(v.line.starts_with("0x0:"));
        assert_eq!(report.sections_scanned, 1);
        assert_eq!(report.instructions_scanned, 1);
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn rejects_evex_with_opmask() {
        let obj = mini_elf(EVEX_VL256_MASK);
        let path = write_temp(&obj, "evex-opmask.o");
        let report = scan_for_evex(&sysroot_objdump(), &path).expect("scan must not error");
        assert!(!report.is_clean());
        assert_eq!(report.evex_violations.len(), 1);
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn rejects_evex_zmm0() {
        let obj = mini_elf(EVEX_ZMM0);
        let path = write_temp(&obj, "evex-zmm0.o");
        let report = scan_for_evex(&sysroot_objdump(), &path).expect("scan must not error");
        assert!(!report.is_clean());
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn rejects_evex_zmm16_31() {
        let obj = mini_elf(EVEX_ZMM16_31);
        let path = write_temp(&obj, "evex-zmm16.o");
        let report = scan_for_evex(&sysroot_objdump(), &path).expect("scan must not error");
        assert!(!report.is_clean());
        assert_eq!(report.evex_violations.len(), 1);
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn accepts_clean_x86_64_v3_elf() {
        let mut text = Vec::new();
        text.extend_from_slice(CLEAN_VEX);
        text.extend_from_slice(CLEAN_SSE);
        let obj = mini_elf(&text);
        let path = write_temp(&obj, "clean-avx2.o");
        let report = scan_for_evex(&sysroot_objdump(), &path).expect("scan must not error");
        assert!(report.is_clean());
        assert_eq!(report.evex_violations.len(), 0);
        assert_eq!(report.sections_scanned, 1);
        assert_eq!(report.instructions_scanned, 2);
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn rejects_evex_inside_archive() {
        let obj = mini_elf(EVEX_VL256_MASK);
        let archive = mini_archive(&[("clean.o", &mini_elf(CLEAN_VEX)), ("evil.o", &obj)]);
        let path = write_temp(&archive, "evil.rlib");
        let report = scan_for_evex(&sysroot_objdump(), &path).expect("scan must not error");
        assert!(!report.is_clean());
        assert_eq!(report.members_scanned, 2);
        assert_eq!(report.evex_violations.len(), 1);
        assert_eq!(report.evex_violations[0].member.as_deref(), Some("evil.o"));
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn accepts_clean_archive() {
        let archive = mini_archive(&[
            ("clean-a.o", &mini_elf(CLEAN_VEX)),
            ("clean-b.o", &mini_elf(CLEAN_SSE)),
        ]);
        let path = write_temp(&archive, "clean.rlib");
        let report = scan_for_evex(&sysroot_objdump(), &path).expect("scan must not error");
        assert!(report.is_clean());
        assert_eq!(report.members_scanned, 2);
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn fails_closed_on_unknown_archive_member() {
        let archive = mini_archive(&[("junk.bin", b"definitely not an object file")]);
        let path = write_temp(&archive, "junk.rlib");
        let err = scan_for_evex(&sysroot_objdump(), &path).expect_err("must fail");
        assert!(matches!(
            err,
            BinGuardError::UnsupportedArchiveMember { .. }
        ));
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn fails_closed_on_empty_archive() {
        let path = write_temp(b"!<arch>\n", "empty.rlib");
        let err = scan_for_evex(&sysroot_objdump(), &path).expect_err("must fail");
        assert!(matches!(err, BinGuardError::EmptyArchive { .. }));
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn fails_closed_on_bitcode_only_archive() {
        // Thin-LTO rlibs store LLVM bitcode members; a machine-code absence
        // certificate cannot be produced from IR, so the scan must fail.
        let bitcode = b"BC\xc0\xde\x35\x14\x00\x00\x00\x00\x00\x00";
        let archive = mini_archive(&[("lib.rmeta", b"meta"), ("code.rcgu.o", bitcode)]);
        let path = write_temp(&archive, "thinlto.rlib");
        let err = scan_for_evex(&sysroot_objdump(), &path).expect_err("must fail");
        assert!(matches!(err, BinGuardError::NoDisassemblableMembers { .. }));
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn fails_closed_on_corrupted_file() {
        let path = write_temp(b"garbage bytes that are not a binary", "corrupt.bin");
        let err = scan_for_evex(&sysroot_objdump(), &path).expect_err("must fail");
        assert!(matches!(err, BinGuardError::UnsupportedFormat { .. }));
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn fails_closed_on_empty_file() {
        let path = write_temp(b"", "empty.bin");
        let err = scan_for_evex(&sysroot_objdump(), &path).expect_err("must fail");
        assert!(matches!(err, BinGuardError::EmptyInput { .. }));
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn fails_closed_on_missing_artifact() {
        let missing = std::env::temp_dir().join("nam-bin-guard-does-not-exist.o");
        let err = scan_for_evex(&sysroot_objdump(), &missing).expect_err("must fail");
        assert!(matches!(err, BinGuardError::ReadFailed { .. }));
    }

    #[test]
    fn fails_closed_when_tool_exits_nonzero() {
        let obj = mini_elf(CLEAN_VEX);
        let path = write_temp(&obj, "tool-fail.o");
        let fake = write_fake_tool("#!/bin/sh\nexit 1\n", "nonzero");
        let err = scan_for_evex(&fake, &path).expect_err("must fail");
        assert!(matches!(err, BinGuardError::ToolFailed { .. }));
        let _ = fs::remove_file(&path);
        let _ = fs::remove_file(&fake);
    }

    #[test]
    fn fails_closed_when_tool_outputs_nothing() {
        let obj = mini_elf(CLEAN_VEX);
        let path = write_temp(&obj, "empty-out.o");
        let fake = write_fake_tool("#!/bin/sh\nexit 0\n", "empty");
        let err = scan_for_evex(&fake, &path).expect_err("must fail");
        assert!(matches!(err, BinGuardError::EmptyOutput { .. }));
        let _ = fs::remove_file(&path);
        let _ = fs::remove_file(&fake);
    }

    #[test]
    fn symbol_scan_reports_forbidden_symbols() {
        let obj = mini_elf(CLEAN_VEX);
        let path = write_temp(&obj, "symscan.o");
        let forbidden = ["definitely_not_present"];
        let matches = scan_symbols(&sysroot_nm(), &path, &forbidden).expect("nm must run");
        assert!(matches.is_empty());
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn symbol_scan_fails_closed_on_tool_error() {
        let obj = mini_elf(CLEAN_VEX);
        let path = write_temp(&obj, "symscan-fail.o");
        let fake = write_fake_tool("#!/bin/sh\nexit 3\n", "nm-fail");
        let err = scan_symbols(&fake, &path, &["x"]).expect_err("must fail");
        assert!(matches!(err, BinGuardError::ToolFailed { .. }));
        let _ = fs::remove_file(&path);
        let _ = fs::remove_file(&fake);
    }

    #[test]
    fn archive_parser_handles_gnu_extended_names() {
        // Build an archive whose member name exceeds 16 bytes, forcing the
        // GNU extended name table (`//`).
        let long_name = "neural_amp_modeler_rs-cafebabe.neural_amp_modeler_rs.c0ffee-cgu.0.rcgu.o";
        let obj = mini_elf(CLEAN_VEX);

        let mut out = Vec::new();
        out.extend_from_slice(b"!<arch>\n");

        let table = format!("{long_name}/\n");
        let mut name_field = [b' '; 16];
        name_field[..2].copy_from_slice(b"//");
        let mut header = Vec::new();
        header.extend_from_slice(&name_field);
        header.extend_from_slice(b"            ");
        header.extend_from_slice(b"      ");
        header.extend_from_slice(b"      ");
        header.extend_from_slice(b"        ");
        header.extend_from_slice(format!("{:>10}", table.len()).as_bytes());
        header.extend_from_slice(b"`\n");
        out.extend_from_slice(&header);
        out.extend_from_slice(table.as_bytes());
        if table.len() % 2 == 1 {
            out.push(b'\n');
        }

        let mut name_field = [b' '; 16];
        name_field[..2].copy_from_slice(b"/0");
        let mut header = Vec::new();
        header.extend_from_slice(&name_field);
        header.extend_from_slice(b"            ");
        header.extend_from_slice(b"      ");
        header.extend_from_slice(b"      ");
        header.extend_from_slice(b"        ");
        header.extend_from_slice(format!("{:>10}", obj.len()).as_bytes());
        header.extend_from_slice(b"`\n");
        out.extend_from_slice(&header);
        out.extend_from_slice(&obj);
        if obj.len() % 2 == 1 {
            out.push(b'\n');
        }

        let path = write_temp(&out, "longname.rlib");
        let report = scan_for_evex(&sysroot_objdump(), &path).expect("scan must not error");
        assert!(report.is_clean());
        assert_eq!(report.members_scanned, 1);
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn multi_member_clean_archive_aggregates_counts() {
        let archive = mini_archive(&[
            ("clean-a.o", &mini_elf(CLEAN_VEX)),
            ("clean-b.o", &mini_elf(CLEAN_SSE)),
        ]);
        let path = write_temp(&archive, "two-clean.rlib");
        let report = scan_for_evex(&sysroot_objdump(), &path).expect("scan must not error");
        assert!(report.is_clean());
        assert_eq!(report.members_scanned, 2);
        assert_eq!(report.sections_scanned, 2);
        assert_eq!(report.instructions_scanned, 2);
        let _ = fs::remove_file(&path);
    }
}
