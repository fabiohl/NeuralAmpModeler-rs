<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# Postmortem: ELF Symbol-Interposition Hang

This document records the durable knowledge from the investigation and fix of an
indefinite hang in `--release` builds, confirmed in test binaries, standalone
executables, and plugin `cdylib` outputs. It deliberately omits the
investigation's chronology and dead ends. What remains is the root cause, why it
was hard to see, and the checks and defaults that now prevent this class of bug
from recurring.

## 1. Summary

A test (`test_x2_aliasing_rejection`) and, potentially, any code calling a
standard libm function (`log10f`, `atan2f`, `acosf`, and other transcendentals)
could hang **indefinitely** in `--release` builds. The hang was a pure CPU spin
with zero syscalls and zero computation: not a deadlock, not an allocator issue,
and not a bug in any DSP algorithm.

**Root cause.** Part of the build (a dependency, or the toolchain's
`compiler_builtins`/libm fallback configuration at the time) caused libm-shaped
fallback symbols (`log10f`, `atan2f`, `acosf`, ...) to be linked into the final
binary with **global (exported) visibility** and the **same C name** as the real
functions in the system's `libm.so.6`. Under the standard ELF rule that a main
executable's own global symbols take priority over same-named symbols in its
dynamic dependencies (*symbol interposition*), `ld.so` resolved the `JUMP_SLOT`
relocation for `log10f@GLIBC_2.2.5` back to the binary's own fallback definition
instead of the glibc implementation. That definition jumped to the PLT, which
jumped back through the GOT slot that had been resolved to the same definition,
forming a self-referential two-instruction loop:

```text
call log10f
   -> in-binary definition "log10f"
   -> PLT "log10f@plt"
   -> GOT[log10f@GLIBC]  (should point into libm.so.6; instead points
                          back to the in-binary definition)
   -> back to the in-binary definition, forever
```

Two `jmp` instructions, no computation, no syscalls, 100% CPU, forever. This
explains the "CPU spin, zero syscalls" signature observed from the first
diagnostic attempt.

**Fix.** A linker version script
([`.cargo/hide-libm-shadow.map`](../.cargo/hide-libm-shadow.map)), applied by
[`build.rs`](../build.rs), forces every standard libm C symbol name to `local`
binding in the final binary. A `local` symbol can never win `ld.so`'s global
interposition, so calls to these names always fall through to the real dynamic
`libm.so.6`, regardless of which dependency (now or in the future) defines a
same-named fallback. The fix is structural, not a point patch for `log10f`: it
closes the entire class of bug.

**Verification.** The fix was confirmed by reading the *runtime value* written
into the GOT slot of a live, attached process, not just the relocation *type*
(which is insufficient; see §2). It was confirmed clean across the production
link targets: the engine test suite (`--lib`), the integration suites (for
example `--features testing,dual-mono`), and downstream dynamic-library and
standalone-host builds.

**Scope of the protection (consumers).** `build.rs` applies the version script
through `cargo:rustc-link-arg`, and only when the target is Linux with the GNU
environment (`target_os = "linux"`, `target_env = "gnu"`); the step is skipped
for docs.rs builds. Cargo passes that argument only to this package's own link
targets (its tests, binaries, examples, and dynamic-library outputs). It does
not propagate to the final binary of a downstream crate. A consumer that links a
same-named libm shadow from another dependency is **not** protected by this
crate and must apply an equivalent version script in its own build. No public
helper exists for this today.

## 2. Why This Was Hard to Find

The DSP code exercised by the failing test (`X2Stage::upsample`/`downsample`,
`HalfBandFilter::design`) was innocent, and a thorough static analysis of it
(loop bounds, indexing, allocation invariants) pointed nowhere, because the bug
was never in that code. The actual obstacle was that **the standard tools' first
answers were misleading**, in specific, generalizable ways:

- **A relocation's *type* is not its *value*.** `readelf -r` showing
  `R_X86_64_JUMP_SLOT log10f@GLIBC_2.2.5` looks like proof of a correct dynamic
  import, but it only states what `ld.so` was *asked* to resolve, not what it
  *wrote* into the GOT slot at load time. Under interposition, the written value
  can silently be wrong. The only reliable check is to read the slot's contents
  from a live process (`gdb -p <pid>`, `print/x *(void**)<runtime-slot-address>`)
  and confirm that the address falls inside the expected library's mapped range,
  ideally by resolving it back to a real symbol name (for example
  `__log10_compatf` inside `libm.so.6`).
- **`nm -C | grep <RustPath>` cannot see `#[no_mangle]` C-ABI symbols.** Functions
  exported with `#[no_mangle]` (as libm fallbacks typically are) keep their bare
  C name, so there is no Rust path for a demangler to produce. A search for a
  Rust-shaped substring returns zero matches whether or not the symbol is
  present, a structurally blind check that was twice mistaken for "confirmed
  absent".
- **Constant-folded probes prove nothing.** An isolated reproduction must reach
  the suspect code path at *runtime*. A literal (`let x: f32 = 0.5; x.log10()`)
  can be constant-folded away under `opt-level=3`/LTO, so the probe silently
  never executes the real call. Force every operand through
  `std::hint::black_box`.
- **"It no longer reproduces" requires re-running the original failing command.**
  Twice, an indirect check was reported as "the bug is gone", and both times a
  fresh run of the exact original scenario still hung. Any claim that a
  previously reproducing failure is fixed must be backed by re-running the
  original failing command or test, not a proxy measurement.
- **Loop-iteration canaries have a blind spot.** A guard that is checked once per
  *outer*-loop iteration never fires if the infinite loop is nested *inside* a
  single outer iteration. Instrument the innermost loop, not just the outermost.
- **A reproduction wrapper must kill the whole process tree.** Wrapping a
  suspect command in `timeout -s KILL` alone can leave a grandchild (for example
  the real test binary spawned by `cargo test`) spinning after the wrapper
  reports termination. Prefer a cgroup-scoped limit
  (`systemd-run --scope -p RuntimeMaxSec=<n> ...`, no inner `timeout`) enforced
  by the kernel against the whole cgroup, and add an automated post-run check
  (for example `pgrep`) that fails loudly if anything from the run is still
  alive.

## 3. Verification Checklist for This Class of Bug

If a program hangs in `--release` at 100% CPU with no syscalls (`strace -f`
shows nothing after process and thread startup), and the hang is near a call to
a libm-style function (anything in `<math.h>`: `log`, `log10`, `log2`, `exp`,
`pow`, `sin`, `cos`, `tan`, `atan2`, `acos`, `asin`, `sqrt`, `cbrt`, `hypot`,
`erf`, `tgamma`, `ldexp`, `fmod`, and their `f`/no-suffix `f32`/`f64` variants),
suspect symbol interposition first. Confirm or rule it out with:

```bash
# 1. Quick, static, informative-but-not-definitive signal: does the binary
#    export a definition (T/W) whose bare C name is also imported from glibc
#    (U ...@GLIBC_*)? Unlike grepping for a Rust-mangled path, this has no
#    `#[no_mangle]` blind spot. A hit is suspicious; a miss is not proof of
#    safety (see step 2).
nm -D --defined-only <binary> | grep -E ' [TW] (log10f|atan2f|acosf|cbrt|fma|fmod)'
nm -D <binary> | grep -E '@GLIBC'

# 2. Definitive, runtime check: read the value written into the GOT slot of
#    the suspect symbol from a live process, and confirm that it resolves to a
#    real symbol inside the expected shared library.
gdb -p <pid> -batch \
  -ex "print/x *(void**)<runtime-GOT-slot-address>" \
  -ex "x/3i *(void**)<runtime-GOT-slot-address>"
# Runtime slot address = (binary load base, from /proc/<pid>/maps)
#                      + (static file offset of the relocation, from
#                         `readelf -r <binary> | grep <symbol>@GLIBC`).
```

If step 2 shows the slot resolving to an address inside the binary's own mapped
range instead of the target shared library's, symbol interposition is confirmed.

## 4. What Is in Place to Prevent Recurrence

- **[`.cargo/hide-libm-shadow.map`](../.cargo/hide-libm-shadow.map)**: a linker
  version script listing the standard libm C symbol surface (`f32`/`f64`: trig,
  exp/log family, `hypot`/`erf`/`tgamma`/`ldexp`/`fmod`, and others) as `local`.
  Any of these names that ends up defined in the binary again, from any
  dependency, is never eligible for `ld.so`'s global interposition. This is the
  durable fix; it does not depend on knowing which dependency caused the problem.
- **[`build.rs`](../build.rs)**: applies the version script (plus
  `-Wl,--undefined-version`, required because `lld` otherwise errors on any
  version-script entry for a symbol that a given binary does not define) via
  `cargo:rustc-link-arg`, scoped to this crate's own link targets on Linux GNU.
  This is deliberately *not* a global `[build] rustflags` entry in
  `.cargo/config.toml`: such a flag also reaches every dependency's build-script
  helper binaries, which resolve the version script's relative path from a
  different working directory and fail to find it.
- **[`tests/libm_export_guard.rs`](../tests/libm_export_guard.rs)**: the
  canonical, fail-closed ELF surface gate. It runs `nm`/`llvm-nm` (always under
  `LC_ALL=C`) over the *linked* test binary (`current_exe`) and fails on any
  global or weak definition (`T`/`W`) of a representative libm sample
  (`log10f`, `atan2f`, `acosf`, `cbrt`, `cbrtf`, `fma`, `fmod`) in the dynamic
  table and, when present, the static table. It requires a `__libc_start_main`
  import as proof that `nm -D` actually read `.dynsym`, and it panics rather
  than skipping when no `nm` is available. It runs in the tests phase of
  `utils/tests-quick.sh` and in the `catch-all` phase of `utils/tests-long.sh`
  (see [testing.md](testing.md)). The former standalone wrapper
  `utils/debug/verify_no_libm_exports.sh` was removed: it scanned `.rlib`
  archives, the wrong surface (object archives still carry `T` exports before
  the version script applies), and exited 0 when the artifact was missing.
- **[`utils/debug/repro_oversample_hang.sh`](../utils/debug/repro_oversample_hang.sh)**:
  a general-purpose safety wrapper for reproducing any suspected hang. It applies
  a cgroup-scoped `RuntimeMaxSec` (kills the whole process tree, not just the
  direct child), keeps the wrapped command and the status capture free of pipes
  (a `| tee` pipeline once produced a false "exit 0" for a run that had hung),
  and performs an automatic post-run residual-process check. Reuse it for future
  hang investigations instead of writing a new wrapper.
- **The reactivated test** (`test_x2_aliasing_rejection` in
  [`src/dsp/oversample_test.rs`](../src/dsp/oversample_test.rs)) runs unignored,
  in debug and release, as part of the `--lib` pass of both `utils/tests-quick.sh`
  and `utils/tests-long.sh`. It exercises exactly the code path (a runtime,
  non-constant `f32::log10()` call in its attenuation evaluation) that this class
  of interposition bug affects, so it doubles as a living regression check.

## 5. Guidance for Future Dependency and Toolchain Changes

- Adding a dependency with a `no_std`, `libm`, or portable-math feature (common
  in numeric, embedded, and `no_std`-compatible crates) is the most likely way
  this class of bug could reappear if the version-script protection were ever
  removed or bypassed. Do not remove it without replacing it with an equivalent
  guard.
- If a libm function used in a hot path is missing from
  `.cargo/hide-libm-shadow.map`, add its name to the `local:` list. The list is
  intentionally broad (the standard C89/C99 libm surface, not just the symbols
  observed to be affected) so that it does not need reactive extension.
- Whenever the toolchain (`rustc`/`compiler_builtins`) or the linker
  (`ld`/`lld`/`bfd`) changes, run the test suite, including
  `test_x2_aliasing_rejection` and `libm_export_guard`, before trusting a release
  build.
