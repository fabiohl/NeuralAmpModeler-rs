// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Synthetic binary fixtures for the EVEX absence guard (Sprint 1 / F-ROB-01).
//!
//! These helpers build minimal ELF64 relocatable objects and GNU `ar`
//! archives in-memory so the guard can be mutation-tested without depending
//! on an external assembler or on stale `target/` artifacts.

#![allow(dead_code)]

use neural_amp_modeler_rs::testing::bin_guard::ToolKind;
use neural_amp_modeler_rs::testing::bin_guard::resolve_llvm_tool;
use std::fs;
use std::io::Write;
use std::path::PathBuf;

/// EVEX VL256 with a low `ymm0..15` register and zeroing mask:
/// `vmovups (%rsi), %ymm0 {%k1} {z}`.
pub const EVEX_VL256_MASK_ZERO: &[u8] = &[0x62, 0xf1, 0x7c, 0xa9, 0x10, 0x06];
/// EVEX VL256 with a low register and opmask (no zeroing):
/// `vmovups %ymm0, (%rdi) {%k1}`.
pub const EVEX_VL256_MASK: &[u8] = &[0x62, 0xf1, 0x7c, 0x29, 0x11, 0x07];
/// EVEX ZMM0: `vmovups (%rdi), %zmm0`.
pub const EVEX_ZMM0: &[u8] = &[0x62, 0xf1, 0x7c, 0x48, 0x10, 0x07];
/// EVEX ZMM16-31 high registers: `vaddps %zmm16, %zmm17, %zmm18`.
pub const EVEX_ZMM16_31: &[u8] = &[0x62, 0xa1, 0x74, 0x40, 0x58, 0xd0];
/// Clean x86-64-v3 VEX instruction: `vmovaps (%rdx), %ymm0`.
pub const CLEAN_VEX: &[u8] = &[0xc5, 0xfc, 0x28, 0x02];
/// Clean x86-64 legacy instruction: `movq 0x8(%rsp), %rax`.
pub const CLEAN_SSE: &[u8] = &[0x48, 0x8b, 0x44, 0x24, 0x08];

/// Resolve the sysroot/PATH `llvm-objdump` for the guard tests.
pub fn sysroot_objdump() -> PathBuf {
    resolve_llvm_tool(ToolKind::LlvmObjdump).expect("bin_guard: llvm-objdump required")
}

/// Resolve the sysroot/PATH `llvm-nm` for the guard tests.
pub fn sysroot_nm() -> PathBuf {
    resolve_llvm_tool(ToolKind::LlvmNm).expect("bin_guard: llvm-nm required")
}

/// Build a minimal ELF64 relocatable object containing `text` bytes in a
/// single executable `.text` section.
pub fn mini_elf(text: &[u8]) -> Vec<u8> {
    const ELF_HDR: usize = 64;
    const SHDR_SIZE: usize = 64;
    const SECTION_COUNT: usize = 3; // null + .text + .shstrtab
    const SHDRTAB_OFF: usize = ELF_HDR; // section headers start after the ELF header
    const TEXT_OFF: usize = SHDRTAB_OFF + SHDR_SIZE * SECTION_COUNT;

    let mut shstrtab = vec![0u8];
    shstrtab.extend_from_slice(b".text\0");
    shstrtab.extend_from_slice(b".shstrtab\0");
    let text_name_idx = 1usize;
    let shstrtab_name_idx = 7usize;
    let shstrtab_off = TEXT_OFF + text.len();

    let mut elf = Vec::with_capacity(shstrtab_off + shstrtab.len());

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
    hdr[54..56].copy_from_slice(&0u16.to_le_bytes()); // e_phentsize
    hdr[56..58].copy_from_slice(&0u16.to_le_bytes()); // e_phnum
    hdr[58..60].copy_from_slice(&(SHDR_SIZE as u16).to_le_bytes()); // e_shentsize
    hdr[60..62].copy_from_slice(&(SECTION_COUNT as u16).to_le_bytes()); // e_shnum
    hdr[62..64].copy_from_slice(&2u16.to_le_bytes()); // e_shstrndx
    elf.extend_from_slice(&hdr);

    elf.extend_from_slice(&[0u8; SHDR_SIZE]); // section 0: null

    let mut sh_text = [0u8; SHDR_SIZE];
    sh_text[0..4].copy_from_slice(&(text_name_idx as u32).to_le_bytes());
    sh_text[4..8].copy_from_slice(&1u32.to_le_bytes()); // SHT_PROGBITS
    sh_text[8..16].copy_from_slice(&6u64.to_le_bytes()); // SHF_ALLOC | SHF_EXECINSTR
    sh_text[24..32].copy_from_slice(&(TEXT_OFF as u64).to_le_bytes());
    sh_text[32..40].copy_from_slice(&(text.len() as u64).to_le_bytes());
    sh_text[48..56].copy_from_slice(&16u64.to_le_bytes());
    elf.extend_from_slice(&sh_text);

    let mut sh_shstr = [0u8; SHDR_SIZE];
    sh_shstr[0..4].copy_from_slice(&(shstrtab_name_idx as u32).to_le_bytes());
    sh_shstr[4..8].copy_from_slice(&3u32.to_le_bytes()); // SHT_STRTAB
    sh_shstr[24..32].copy_from_slice(&(shstrtab_off as u64).to_le_bytes());
    sh_shstr[32..40].copy_from_slice(&(shstrtab.len() as u64).to_le_bytes());
    sh_shstr[48..56].copy_from_slice(&1u64.to_le_bytes());
    elf.extend_from_slice(&sh_shstr);

    elf.extend_from_slice(text);
    elf.extend_from_slice(&shstrtab);
    elf
}

/// Build a GNU `ar` archive from named members.
pub fn mini_archive(members: &[(&str, &[u8])]) -> Vec<u8> {
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

/// Write bytes to a unique temporary file and return its path.
pub fn write_temp(bytes: &[u8], name: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    path.push(format!("nam-avx512-guard-{}-{name}", std::process::id()));
    let mut f = fs::File::create(&path).expect("write fixture");
    f.write_all(bytes).expect("write fixture bytes");
    path
}

/// Write an executable fake tool script (for failure-injection tests).
pub fn write_fake_tool(script: &str, label: &str) -> PathBuf {
    use std::os::unix::fs::PermissionsExt;
    let mut path = std::env::temp_dir();
    path.push(format!("nam-fake-objdump-{}-{label}", std::process::id()));
    let mut f = fs::File::create(&path).expect("write fake tool");
    f.write_all(script.as_bytes()).expect("write fake tool");
    fs::set_permissions(&path, fs::Permissions::from_mode(0o755)).expect("chmod");
    path
}

/// Remove a temporary file, ignoring errors (best-effort cleanup).
pub fn cleanup(path: &std::path::Path) {
    let _ = fs::remove_file(path);
}
