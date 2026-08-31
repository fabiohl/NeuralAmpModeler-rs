// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

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

static UNIT_TEMP_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn write_temp(bytes: &[u8], name: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    let seq = UNIT_TEMP_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    path.push(format!("nam-bin-guard-{}-{seq}-{name}", std::process::id()));
    let mut f = fs::File::create(&path).expect("write fixture");
    f.write_all(bytes).expect("write fixture bytes");
    f.sync_all().expect("sync fixture bytes");
    drop(f);
    path
}

fn write_fake_tool(script: &str, label: &str) -> PathBuf {
    use std::os::unix::fs::PermissionsExt;
    let mut path = std::env::temp_dir();
    let seq = UNIT_TEMP_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    path.push(format!(
        "nam-fake-objdump-{}-{seq}-{label}",
        std::process::id()
    ));
    let mut f = fs::File::create(&path).expect("write fake tool");
    f.write_all(script.as_bytes()).expect("write fake tool");
    f.sync_all().expect("sync fake tool");
    drop(f);
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
