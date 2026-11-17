// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! `nam_preflight` — Safe-baseline host preflight probe.
//!
//! Designed to be compiled and executed *before* any `x86-64-v3` binary so a
//! host lacking AVX2/FMA/BMI2/MOVBE/LZCNT/F16C never crashes with `SIGILL` or
//! `STATUS_ILLEGAL_INSTRUCTION` (`0xC000001D`). When `x86-64-v3` features are
//! incomplete, emits `RESULT: GAP cpu-missing-x86-64-v3` and exits with code 3.
//!
//! Invoke in safe-baseline mode (overriding workspace `-Ctarget-cpu=x86-64-v3`):
//! ```sh
//! RUSTFLAGS="" cargo run --locked -p nam-preflight --bin nam_preflight
//! ```

use std::process::ExitCode;

/// Exit code emitted when a mandatory host capability is missing (`GAP`).
const EXIT_GAP: u8 = 3;

#[cfg(any(test, target_arch = "x86_64"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FeatureCheck {
    name: &'static str,
    present: bool,
}

#[cfg(any(test, target_arch = "x86_64"))]
#[derive(Debug, Clone, PartialEq, Eq)]
struct V3Summary {
    passed: bool,
    missing: Vec<&'static str>,
}

#[cfg(any(test, target_arch = "x86_64"))]
fn evaluate_x86_64_v3_features(checks: &[FeatureCheck]) -> V3Summary {
    let missing: Vec<&'static str> = checks
        .iter()
        .filter(|c| !c.present)
        .map(|c| c.name)
        .collect();
    V3Summary {
        passed: missing.is_empty(),
        missing,
    }
}

#[cfg(any(test, target_arch = "aarch64"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Arm64Summary {
    neon_ok: bool,
    lse_present: bool,
}

#[cfg(any(test, target_arch = "aarch64"))]
fn evaluate_arm64_features(
    neon_arch: bool,
    neon_os: Option<bool>,
    lse_arch: bool,
    lse_os: Option<bool>,
) -> Arm64Summary {
    let neon_ok = neon_arch && neon_os.unwrap_or(true);
    let lse_present = match lse_os {
        Some(os) => lse_arch && os,
        None => lse_arch,
    };
    Arm64Summary {
        neon_ok,
        lse_present,
    }
}

#[cfg(any(test, target_os = "windows"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WindowsSupportStatus {
    /// Windows 11 Client on an actively supported build (`>= 26100`, e.g. 24H2/25H2/26H2+).
    SupportedWindows11Client,
    /// Windows 11 Client on an older build (`22000..26100`, serviced/EOS check needed).
    LegacyWindows11Client,
    /// Windows Server runner (e.g. `windows-2025` CI — not Windows 11 Client evidence).
    WindowsServerCiOnly,
    /// Windows 10 or older — outside supported target scope.
    OutOfScope,
}

#[cfg(any(test, target_os = "windows"))]
impl WindowsSupportStatus {
    fn as_str(self) -> &'static str {
        match self {
            Self::SupportedWindows11Client => "supported-windows-11-client",
            Self::LegacyWindows11Client => "legacy-windows-11-check-eol",
            Self::WindowsServerCiOnly => "windows-server-not-win11-client",
            Self::OutOfScope => "out-of-scope-pre-win11",
        }
    }
}

#[cfg(any(test, target_os = "windows"))]
fn classify_windows_support(major: u32, build: u32, is_workstation: bool) -> WindowsSupportStatus {
    if major < 10 {
        return WindowsSupportStatus::OutOfScope;
    }
    if !is_workstation {
        return if build >= 26100 {
            WindowsSupportStatus::WindowsServerCiOnly
        } else {
            WindowsSupportStatus::OutOfScope
        };
    }
    if build >= 26100 {
        WindowsSupportStatus::SupportedWindows11Client
    } else if build >= 22000 {
        WindowsSupportStatus::LegacyWindows11Client
    } else {
        WindowsSupportStatus::OutOfScope
    }
}

fn main() -> ExitCode {
    println!("=======================================================");
    println!(" NeuralAmpModeler-rs Safe-Baseline Host Preflight");
    println!("=======================================================");
    println!("os:   {}", std::env::consts::OS);
    println!("arch: {}", std::env::consts::ARCH);

    print_os_details();
    print_cpu_model();
    print_fp_state();

    let exit_code = check_arch_capabilities();
    println!("=======================================================");
    exit_code
}

#[cfg(target_os = "windows")]
mod win_probe {
    #[cfg(target_arch = "aarch64")]
    use core::ffi::c_void;
    #[cfg(target_arch = "aarch64")]
    use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use windows_sys::Wdk::System::SystemServices::RtlGetVersion;
    use windows_sys::Win32::Foundation::ERROR_SUCCESS;
    #[cfg(target_arch = "aarch64")]
    use windows_sys::Win32::Foundation::EXCEPTION_ILLEGAL_INSTRUCTION;
    use windows_sys::Win32::System::Registry::{
        HKEY, HKEY_LOCAL_MACHINE, KEY_READ, REG_DWORD, REG_SZ, RegCloseKey, RegOpenKeyExW,
        RegQueryValueExW,
    };
    use windows_sys::Win32::System::SystemInformation::{OSVERSIONINFOEXW, OSVERSIONINFOW};

    const VER_NT_WORKSTATION: u8 = 1;

    pub(super) struct WinVersionInfo {
        pub(super) major: u32,
        pub(super) minor: u32,
        pub(super) build: u32,
        pub(super) ubr: Option<u32>,
        pub(super) is_workstation: bool,
        pub(super) product_name: String,
        pub(super) edition_id: String,
        pub(super) display_version: String,
    }

    fn to_wide_z(s: &str) -> Vec<u16> {
        s.encode_utf16().chain(core::iter::once(0)).collect()
    }

    fn read_reg_string(hkey: HKEY, name: &str) -> Option<String> {
        let wide_name = to_wide_z(name);
        let mut reg_type: u32 = 0;
        let mut bytes_len: u32 = 0;
        // SAFETY: `hkey` is an open registry key; `wide_name` is NUL-terminated UTF-16.
        let rc = unsafe {
            RegQueryValueExW(
                hkey,
                wide_name.as_ptr(),
                core::ptr::null(),
                &raw mut reg_type,
                core::ptr::null_mut(),
                &raw mut bytes_len,
            )
        };
        if rc != ERROR_SUCCESS || reg_type != REG_SZ || bytes_len == 0 || bytes_len > 4096 {
            return None;
        }
        let mut buf = vec![0u8; bytes_len as usize];
        // SAFETY: `buf` is sized to `bytes_len` reported by the first query.
        let rc2 = unsafe {
            RegQueryValueExW(
                hkey,
                wide_name.as_ptr(),
                core::ptr::null(),
                &raw mut reg_type,
                buf.as_mut_ptr(),
                &raw mut bytes_len,
            )
        };
        if rc2 != ERROR_SUCCESS || reg_type != REG_SZ {
            return None;
        }
        let words: Vec<u16> = buf[..(bytes_len as usize).min(buf.len())]
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| u16::from_le_bytes(*c))
            .take_while(|&w| w != 0)
            .collect();
        Some(String::from_utf16_lossy(&words))
    }

    fn read_reg_dword(hkey: HKEY, name: &str) -> Option<u32> {
        let wide_name = to_wide_z(name);
        let mut reg_type: u32 = 0;
        let mut value: u32 = 0;
        let mut bytes_len: u32 = core::mem::size_of::<u32>() as u32;
        // SAFETY: `value` is a 4-byte aligned `u32` buffer matching `bytes_len`.
        let rc = unsafe {
            RegQueryValueExW(
                hkey,
                wide_name.as_ptr(),
                core::ptr::null(),
                &raw mut reg_type,
                (&raw mut value).cast::<u8>(),
                &raw mut bytes_len,
            )
        };
        if rc == ERROR_SUCCESS && reg_type == REG_DWORD && bytes_len == 4 {
            Some(value)
        } else {
            None
        }
    }

    #[cfg(target_arch = "aarch64")]
    pub(super) fn read_registry_string(subkey: &str, value_name: &str) -> Option<String> {
        let wide_key = to_wide_z(subkey);
        let mut hkey: HKEY = core::ptr::null_mut();
        // SAFETY: Opens a read-only handle to `HKEY_LOCAL_MACHINE\<subkey>`.
        let rc = unsafe {
            RegOpenKeyExW(
                HKEY_LOCAL_MACHINE,
                wide_key.as_ptr(),
                0,
                KEY_READ,
                &raw mut hkey,
            )
        };
        if rc != ERROR_SUCCESS || hkey.is_null() {
            return None;
        }
        let val = read_reg_string(hkey, value_name);
        // SAFETY: `hkey` was opened by `RegOpenKeyExW`.
        unsafe {
            RegCloseKey(hkey);
        }
        val
    }

    pub(super) fn query_windows_version() -> WinVersionInfo {
        let mut info = OSVERSIONINFOEXW {
            dwOSVersionInfoSize: core::mem::size_of::<OSVERSIONINFOEXW>() as u32,
            ..Default::default()
        };
        // SAFETY: `OSVERSIONINFOEXW` shares its prefix layout with `OSVERSIONINFOW`
        // and `dwOSVersionInfoSize` is set to the extended struct size per MSDN.
        let status = unsafe { RtlGetVersion((&raw mut info).cast::<OSVERSIONINFOW>()) };
        let (major, minor, build, is_workstation) = if status == 0 {
            (
                info.dwMajorVersion,
                info.dwMinorVersion,
                info.dwBuildNumber,
                info.wProductType == VER_NT_WORKSTATION,
            )
        } else {
            (0, 0, 0, true)
        };

        let subkey = r"SOFTWARE\Microsoft\Windows NT\CurrentVersion";
        let wide_key = to_wide_z(subkey);
        let mut hkey: HKEY = core::ptr::null_mut();
        let mut ubr = None;
        let mut product_name = String::from("unknown");
        let mut edition_id = String::from("unknown");
        let mut display_version = String::from("unknown");

        // SAFETY: Opens `HKLM\SOFTWARE\Microsoft\Windows NT\CurrentVersion` read-only.
        if unsafe {
            RegOpenKeyExW(
                HKEY_LOCAL_MACHINE,
                wide_key.as_ptr(),
                0,
                KEY_READ,
                &raw mut hkey,
            )
        } == ERROR_SUCCESS
            && !hkey.is_null()
        {
            ubr = read_reg_dword(hkey, "UBR");
            if let Some(s) = read_reg_string(hkey, "ProductName") {
                product_name = s;
            }
            if let Some(s) = read_reg_string(hkey, "EditionID") {
                edition_id = s;
            }
            if let Some(s) = read_reg_string(hkey, "DisplayVersion") {
                display_version = s;
            }
            // SAFETY: `hkey` was opened by `RegOpenKeyExW`.
            unsafe {
                RegCloseKey(hkey);
            }
        }

        WinVersionInfo {
            major,
            minor,
            build,
            ubr,
            is_workstation,
            product_name,
            edition_id,
            display_version,
        }
    }

    #[cfg(target_arch = "aarch64")]
    pub(super) fn probe_arm64_generic_timer() -> (bool, u64, i64) {
        use windows_sys::Win32::System::Diagnostics::Debug::{
            AddVectoredExceptionHandler, EXCEPTION_CONTINUE_EXECUTION, EXCEPTION_CONTINUE_SEARCH,
            EXCEPTION_POINTERS, RemoveVectoredExceptionHandler,
        };
        use windows_sys::Win32::System::Performance::QueryPerformanceFrequency;

        const EXCEPTION_PRIVILEGED_INSTRUCTION: i32 = 0xC000_0096_u32 as i32;
        static FAULT_PC: AtomicUsize = AtomicUsize::new(0);
        static TRAPPED: AtomicBool = AtomicBool::new(false);

        unsafe extern "system" fn handler(info: *mut EXCEPTION_POINTERS) -> i32 {
            if info.is_null() {
                return EXCEPTION_CONTINUE_SEARCH;
            }
            // SAFETY: Validates record/context pointers provided by SEH dispatch and
            // advances `Pc` by 4 bytes only on the exact `FAULT_PC` `mrs` instruction.
            unsafe {
                let rec = (*info).ExceptionRecord;
                let ctx = (*info).ContextRecord;
                if rec.is_null() || ctx.is_null() {
                    return EXCEPTION_CONTINUE_SEARCH;
                }
                let code = (*rec).ExceptionCode;
                if code != EXCEPTION_ILLEGAL_INSTRUCTION && code != EXCEPTION_PRIVILEGED_INSTRUCTION
                {
                    return EXCEPTION_CONTINUE_SEARCH;
                }
                let expected = FAULT_PC.load(Ordering::Acquire);
                if expected == 0 || (*ctx).Pc as usize != expected {
                    return EXCEPTION_CONTINUE_SEARCH;
                }
                TRAPPED.store(true, Ordering::Release);
                (*ctx).Pc = (*ctx).Pc.wrapping_add(4);
                EXCEPTION_CONTINUE_EXECUTION
            }
        }

        struct Guard(*mut c_void);
        impl Drop for Guard {
            fn drop(&mut self) {
                if !self.0.is_null() {
                    // SAFETY: `self.0` was returned by `AddVectoredExceptionHandler`.
                    unsafe {
                        RemoveVectoredExceptionHandler(self.0);
                    }
                }
                FAULT_PC.store(0, Ordering::Release);
            }
        }

        let mut qpf: i64 = 0;
        // SAFETY: `qpf` is a stack `i64` out-parameter.
        let _ = unsafe { QueryPerformanceFrequency(&raw mut qpf) };

        // SAFETY: Registers a temporary vectored exception handler scoped to `Guard`.
        let veh = unsafe { AddVectoredExceptionHandler(1, Some(handler)) };
        let _guard = Guard(veh);
        if veh.is_null() {
            return (false, 0, qpf);
        }

        let mut cntfrq: u64 = 0;
        TRAPPED.store(false, Ordering::Release);
        let slot = FAULT_PC.as_ptr();
        // SAFETY: Writes the exact address of label `1:` to `FAULT_PC` around `mrs`.
        unsafe {
            core::arch::asm!(
                "adr {pc}, 1f",
                "str {pc}, [{slot}]",
                "dmb ish",
                "1:",
                "mrs {val}, cntfrq_el0",
                "str xzr, [{slot}]",
                pc = out(reg) _,
                slot = in(reg) slot,
                val = inout(reg) cntfrq,
                options(nostack, preserves_flags),
            );
        }
        let frq_ok = !TRAPPED.load(Ordering::Acquire);

        let mut cntvct: u64 = 0;
        TRAPPED.store(false, Ordering::Release);
        // SAFETY: Writes the exact address of label `2:` to `FAULT_PC` around `mrs`.
        unsafe {
            core::arch::asm!(
                "isb",
                "adr {pc}, 2f",
                "str {pc}, [{slot}]",
                "dmb ish",
                "2:",
                "mrs {val}, cntvct_el0",
                "str xzr, [{slot}]",
                pc = out(reg) _,
                slot = in(reg) slot,
                val = inout(reg) cntvct,
                options(nostack, preserves_flags),
            );
        }
        let vct_ok = !TRAPPED.load(Ordering::Acquire);
        let _ = cntvct;

        (frq_ok && vct_ok, cntfrq, qpf)
    }
}

fn print_os_details() {
    #[cfg(target_os = "windows")]
    {
        let ver = win_probe::query_windows_version();
        let support = classify_windows_support(ver.major, ver.build, ver.is_workstation);
        let ubr_str = ver.ubr.map(|u| format!(".{u}")).unwrap_or_default();
        println!(
            "windows.version:   {}.{}.{}{}",
            ver.major, ver.minor, ver.build, ubr_str
        );
        println!("windows.product:   {}", ver.product_name);
        println!("windows.edition:   {}", ver.edition_id);
        println!("windows.display:   {}", ver.display_version);
        println!(
            "windows.role:      {}",
            if ver.is_workstation {
                "workstation"
            } else {
                "server"
            }
        );
        println!("windows.lifecycle: {}", support.as_str());
    }
    #[cfg(target_os = "linux")]
    {
        if let Ok(text) = std::fs::read_to_string("/etc/os-release") {
            for line in text.lines() {
                if let Some(pretty) = line.strip_prefix("PRETTY_NAME=") {
                    println!("linux.distro: {}", pretty.trim_matches('"'));
                    break;
                }
            }
        }
    }
}

fn print_cpu_model() {
    #[cfg(target_arch = "x86_64")]
    {
        let brand = read_x86_brand_string().unwrap_or_else(|| "unknown".to_string());
        println!("cpu.model: {brand}");
    }
    #[cfg(all(target_arch = "aarch64", target_os = "windows"))]
    {
        let model = win_probe::read_registry_string(
            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            "ProcessorNameString",
        )
        .unwrap_or_else(|| "unknown".to_string());
        println!("cpu.model: {}", model.trim());
    }
    #[cfg(all(target_arch = "aarch64", not(target_os = "windows")))]
    {
        let model = std::fs::read_to_string("/proc/cpuinfo")
            .ok()
            .and_then(|text| {
                text.lines().find_map(|line| {
                    if line.starts_with("model name")
                        || line.starts_with("Model")
                        || line.starts_with("Hardware")
                    {
                        line.split_once(':').map(|(_, v)| v.trim().to_string())
                    } else {
                        None
                    }
                })
            })
            .unwrap_or_else(|| "unknown".to_string());
        println!("cpu.model: {model}");
    }
}

#[cfg(target_arch = "x86_64")]
fn read_x86_brand_string() -> Option<String> {
    let max_ext = core::arch::x86_64::__cpuid(0x8000_0000).eax;
    if max_ext < 0x8000_0004 {
        return None;
    }
    let mut bytes = Vec::with_capacity(48);
    for leaf in 0x8000_0002u32..=0x8000_0004u32 {
        let res = core::arch::x86_64::__cpuid(leaf);
        bytes.extend_from_slice(&res.eax.to_le_bytes());
        bytes.extend_from_slice(&res.ebx.to_le_bytes());
        bytes.extend_from_slice(&res.ecx.to_le_bytes());
        bytes.extend_from_slice(&res.edx.to_le_bytes());
    }
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    let brand = String::from_utf8_lossy(&bytes[..end]).trim().to_string();
    if brand.is_empty() { None } else { Some(brand) }
}

fn print_fp_state() {
    #[cfg(target_arch = "x86_64")]
    {
        let mut mxcsr: u32 = 0;
        // SAFETY: `stmxcsr` is part of SSE/SSE2 baseline on all x86-64 CPUs and
        // writes the 32-bit MXCSR control/status register to the stack slot.
        unsafe {
            core::arch::asm!("stmxcsr [{0}]", in(reg) &raw mut mxcsr, options(nostack, preserves_flags));
        }
        let daz = (mxcsr >> 6) & 1 != 0;
        let ftz = (mxcsr >> 15) & 1 != 0;
        let rc = (mxcsr >> 13) & 3;
        println!("fp.mxcsr:  0x{mxcsr:08x} (DAZ={daz}, FTZ={ftz}, RC={rc})");
    }
    #[cfg(target_arch = "aarch64")]
    {
        let fpcr: u64;
        // SAFETY: `mrs fpcr` reads the EL0 floating-point control register without side effects.
        unsafe {
            core::arch::asm!("mrs {0}, fpcr", out(reg) fpcr, options(nomem, nostack, preserves_flags));
        }
        let fz = (fpcr >> 24) & 1 != 0;
        let fz16 = (fpcr >> 19) & 1 != 0;
        let rmode = (fpcr >> 22) & 3;
        println!("fp.fpcr:   0x{fpcr:016x} (FZ={fz}, FZ16={fz16}, RMode={rmode})");
    }
}

fn check_arch_capabilities() -> ExitCode {
    #[cfg(target_arch = "x86_64")]
    {
        let v3_checks = [
            FeatureCheck {
                name: "avx",
                present: std::arch::is_x86_feature_detected!("avx"),
            },
            FeatureCheck {
                name: "avx2",
                present: std::arch::is_x86_feature_detected!("avx2"),
            },
            FeatureCheck {
                name: "bmi1",
                present: std::arch::is_x86_feature_detected!("bmi1"),
            },
            FeatureCheck {
                name: "bmi2",
                present: std::arch::is_x86_feature_detected!("bmi2"),
            },
            FeatureCheck {
                name: "f16c",
                present: std::arch::is_x86_feature_detected!("f16c"),
            },
            FeatureCheck {
                name: "fma",
                present: std::arch::is_x86_feature_detected!("fma"),
            },
            FeatureCheck {
                name: "lzcnt",
                present: std::arch::is_x86_feature_detected!("lzcnt"),
            },
            FeatureCheck {
                name: "movbe",
                present: std::arch::is_x86_feature_detected!("movbe"),
            },
        ];

        println!("[x86-64-v3 Required Features]");
        for item in &v3_checks {
            println!(
                "  - {:<8} {}",
                item.name,
                if item.present { "YES" } else { "NO" }
            );
        }

        let avx512f = std::arch::is_x86_feature_detected!("avx512f");
        let avx512vl = std::arch::is_x86_feature_detected!("avx512vl");
        let avx512bw = std::arch::is_x86_feature_detected!("avx512bw");
        let avx512dq = std::arch::is_x86_feature_detected!("avx512dq");
        println!(
            "avx512.opt_in: {}",
            avx512f && avx512vl && avx512bw && avx512dq
        );

        let summary = evaluate_x86_64_v3_features(&v3_checks);
        if summary.passed {
            println!("RESULT: PASS x86-64-v3-ready");
            ExitCode::SUCCESS
        } else {
            println!(
                "RESULT: GAP cpu-missing-x86-64-v3 missing={}",
                summary.missing.join(",")
            );
            ExitCode::from(EXIT_GAP)
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        let neon_arch = std::arch::is_aarch64_feature_detected!("neon");
        let lse_arch = std::arch::is_aarch64_feature_detected!("lse");
        #[cfg(target_os = "windows")]
        let (neon_os, lse_os) = {
            use windows_sys::Win32::System::Threading::{
                IsProcessorFeaturePresent, PF_ARM_NEON_INSTRUCTIONS_AVAILABLE,
                PF_ARM_V81_ATOMIC_INSTRUCTIONS_AVAILABLE,
            };
            // SAFETY: `IsProcessorFeaturePresent` takes a value enum and performs no pointer deref.
            unsafe {
                (
                    Some(IsProcessorFeaturePresent(PF_ARM_NEON_INSTRUCTIONS_AVAILABLE) != 0),
                    Some(IsProcessorFeaturePresent(PF_ARM_V81_ATOMIC_INSTRUCTIONS_AVAILABLE) != 0),
                )
            }
        };
        #[cfg(not(target_os = "windows"))]
        let (neon_os, lse_os): (Option<bool>, Option<bool>) = (None, None);

        let summary = evaluate_arm64_features(neon_arch, neon_os, lse_arch, lse_os);
        println!("[ARM64 Features]");
        println!("  - neon (baseline):        {}", summary.neon_ok);
        println!(
            "  - lse (v8.1-a atomics):   {} (informational; not assumed enabled)",
            summary.lse_present
        );

        #[cfg(target_os = "windows")]
        {
            let (timer_ok, cntfrq, qpf) = win_probe::probe_arm64_generic_timer();
            println!(
                "  - generic_timer (EL0):    {} (CNTFRQ_EL0={} Hz, QPF={} Hz)",
                if timer_ok {
                    "ACCESSIBLE"
                } else {
                    "GAP arm64-generic-timer-unavailable (Instant/QPC fallback)"
                },
                cntfrq,
                qpf
            );
        }

        if summary.neon_ok {
            println!("RESULT: PASS aarch64-neon-ready");
            ExitCode::SUCCESS
        } else {
            println!("RESULT: GAP cpu-missing-aarch64-neon");
            ExitCode::from(EXIT_GAP)
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        println!("RESULT: GAP unsupported-architecture");
        ExitCode::from(EXIT_GAP)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn x86_64_v3_evaluator_detects_missing_features() {
        let all_yes = [
            FeatureCheck {
                name: "avx",
                present: true,
            },
            FeatureCheck {
                name: "avx2",
                present: true,
            },
            FeatureCheck {
                name: "bmi1",
                present: true,
            },
            FeatureCheck {
                name: "bmi2",
                present: true,
            },
            FeatureCheck {
                name: "f16c",
                present: true,
            },
            FeatureCheck {
                name: "fma",
                present: true,
            },
            FeatureCheck {
                name: "lzcnt",
                present: true,
            },
            FeatureCheck {
                name: "movbe",
                present: true,
            },
        ];
        let ok = evaluate_x86_64_v3_features(&all_yes);
        assert!(ok.passed);
        assert!(ok.missing.is_empty());

        let mut missing_fma_movbe = all_yes;
        missing_fma_movbe[5].present = false;
        missing_fma_movbe[7].present = false;
        let gap = evaluate_x86_64_v3_features(&missing_fma_movbe);
        assert!(!gap.passed);
        assert_eq!(gap.missing, vec!["fma", "movbe"]);
    }

    #[test]
    fn windows_support_classification_covers_client_server_and_eos() {
        assert_eq!(
            classify_windows_support(10, 26300, true).as_str(),
            "supported-windows-11-client"
        );
        assert_eq!(
            classify_windows_support(10, 26100, true),
            WindowsSupportStatus::SupportedWindows11Client
        );
        assert_eq!(
            classify_windows_support(10, 22631, true),
            WindowsSupportStatus::LegacyWindows11Client
        );
        assert_eq!(
            classify_windows_support(10, 26100, false),
            WindowsSupportStatus::WindowsServerCiOnly
        );
        assert_eq!(
            classify_windows_support(10, 19045, true),
            WindowsSupportStatus::OutOfScope
        );
    }

    #[test]
    fn arm64_evaluator_requires_neon_and_keeps_lse_informational() {
        let ok = evaluate_arm64_features(true, Some(true), false, Some(false));
        assert!(ok.neon_ok);
        assert!(!ok.lse_present);

        let missing = evaluate_arm64_features(true, Some(false), true, Some(true));
        assert!(!missing.neon_ok);
    }
}
