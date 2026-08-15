<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# Postmortem: ELF symbol-interposition hang

This document replaces the working notes used during the investigation and fix of a real,
indefinite hang in `--release` builds (also confirmed in the standalone
binary and the CLAP `cdylib`). It intentionally does **not** preserve the
investigation's chronology, dead ends, or every experiment run — that
detail no longer has operational value. What follows is the durable
knowledge: what the bug actually was, why it was hard to see, and the
concrete checks and defaults now in place to stop this class of bug from
recurring.

## 1. Summary

A test (`test_x2_aliasing_rejection`) and, potentially, any other code
calling standard libm functions (`log10f`, `atan2f`, `acosf`, and other
transcendentals), could hang **indefinitely** in `--release` builds. The
hang was a pure CPU spin with zero syscalls and zero computation — not a
deadlock, not an allocator issue, and not a bug in any DSP algorithm.

**Root cause:** some part of the crate's build (a dependency, or the exact
toolchain's `compiler_builtins`/libm fallback configuration at the time)
caused libm-shaped fallback symbols (`log10f`, `atan2f`, `acosf`, ...) to
be linked into the final binary with **global (exported) visibility** and
the **same C name** as the real functions in the system's `libm.so.6`.
Under the standard ELF rule that a main executable's own global symbols
take priority over same-named symbols in its dynamic dependencies
("symbol interposition"), `ld.so` resolved the `JUMP_SLOT` relocation for
`log10f@GLIBC_2.2.5` back to the binary's own local stub instead of the
real glibc implementation. That stub, in turn, jumped to the PLT, which
jumped back to the GOT slot that had been resolved to the same local stub
— a self-referential two-instruction loop:

```text
call log10f
   -> local trampoline "log10f"
   -> PLT "log10f@plt"
   -> GOT[log10f@GLIBC]  (should point into libm.so.6; instead points
                           back to the local trampoline)
   -> back to the local trampoline, forever
```

Two `jmp` instructions, no computation, no syscalls, 100% CPU, forever.
This explains the "CPU spin, zero syscalls" signature observed from the
very first diagnostic attempt.

**Fix:** a linker version script ([`.cargo/hide-libm-shadow.map`](../.cargo/hide-libm-shadow.map)), applied
via [`build.rs`](../build.rs), forces every standard libm C symbol name to `local`
binding in the final binary. A `local` symbol can never win `ld.so`'s
global symbol interposition, so calls to these names always fall through
to the real dynamic `libm.so.6`, regardless of which dependency (now or in
the future) happens to also define a same-named fallback. This is a
structural fix, not a point patch for `log10f` — it closes the entire
class of bug.

**Verification:** confirmed by reading the *actual runtime value* written
into the GOT slot from a live, attached process (not just the relocation
*type*, which is not sufficient — see §3). Confirmed clean across all
production link targets: the engine test suite (`--lib`), integration suites
(`--features testing,stereo`), and downstream targets (CLAP cdylib plugins
and standalone hosts).

## 2. Why this was hard to find

The DSP algorithm exercised by the failing test (`X2Stage::upsample`/
`downsample`, `HalfBandFilter::design`) was innocent, and static analysis
of it (loop bounds, indexing, allocation invariants) was thorough and
correct — and still pointed nowhere, because the bug was never in that
code. The actual blocker was that **the standard tools' first answers were
misleading**, in specific, generalizable ways:

- **A relocation's *type* is not its *value*.** `readelf -r` showing
  `R_X86_64_JUMP_SLOT log10f@GLIBC_2.2.5` looks like proof of a correct
  dynamic import. It only says what `ld.so` was *asked* to resolve, not
  what it actually *wrote* into that GOT slot at load time. Under symbol
  interposition, the written value can silently be the wrong one. The
  only reliable check is to read the GOT slot's contents from a live
  process (`gdb -p <pid>`, `print/x *(void**)<runtime-slot-address>`) and
  confirm the address falls inside the expected library's mapped range —
  ideally by resolving it back to a real symbol name (e.g.
  `__log10_compatf` inside `libm.so.6`), not just an address range.
- **`nm -C | grep <RustPath>` cannot see `#[no_mangle]` C-ABI symbols.**
  Functions exported with `#[no_mangle]` (as libm fallbacks typically are)
  keep their bare C name; there is no Rust path for a demangler to
  produce. A search for a Rust-shaped substring will return zero matches
  whether or not the symbol is present — a structurally blind check that,
  in this investigation, was mistaken for "confirmed absent" twice.
- **Constant-folded probes prove nothing.** An isolated test built to
  reproduce a runtime bug must actually reach the suspect code path at
  *runtime*. A literal value (`let x: f32 = 0.5; x.log10()`) can be
  constant-folded away entirely under `opt-level=3`/LTO, so the "probe"
  silently never executes the real call. Any isolation test for this
  class of bug must force the value through `std::hint::black_box` on
  every operand, or it is not testing anything.
- **"It doesn't reproduce anymore" needs the original failing command run
  again, not an indirect signal.** Twice during this investigation, a
  static/indirect check was reported as "the bug is gone," and both times
  a fresh, direct re-run of the exact original failing scenario showed the
  hang was still there. The generalizable rule: any claim that a
  previously-reproducing failure is fixed must be backed by literally
  re-running the original failing command/test, not by a proxy
  measurement, however plausible that proxy looks.
- **Loop-iteration canaries have a blind spot.** An instrumentation
  technique that only increments/checks a guard counter once per
  *outer*-loop iteration will never fire if the actual infinite loop is
  nested *inside* a single outer iteration — the outer loop simply never
  advances far enough to be checked again. If you add iteration canaries
  to localize a hang, check the innermost loop, not just the outermost.
- **A resource-isolation wrapper for reproducing a suspected hang must
  kill the whole process tree, not just its direct child.** Wrapping a
  suspect command in `timeout -s KILL` alone can leave a grandchild
  process (e.g. the real test binary, spawned by `cargo test`) running
  and consuming CPU indefinitely after the wrapper script has already
  reported the run as terminated. Prefer a cgroup-scoped time limit
  (`systemd-run --scope -p RuntimeMaxSec=<n> ...`, no inner `timeout`)
  that the kernel enforces against the entire cgroup, and always add an
  automated post-run check (e.g. `pgrep`) that fails loudly if anything
  from the run is still alive.

## 3. Verification checklist for this class of bug

If a program hangs in `--release` with 100% CPU and no syscalls (visible
via `strace -f`, which will show nothing after process/thread startup),
and the hang is near a call to a libm-style function (anything in
`<math.h>`: `log`, `log10`, `log2`, `exp`, `pow`, `sin`, `cos`, `tan`,
`atan2`, `acos`, `asin`, `sqrt`, `cbrt`, `hypot`, `erf`, `tgamma`,
`ldexp`, `fmod`, and their `f`/no-suffix f32/f64 variants), suspect symbol
interposition first. Confirm or rule it out with:

```bash
# 1. Quick, static, informative-but-not-definitive signal: is there a
#    LOCAL (defined, "T"/"t") symbol with the exact same bare C name as an
#    imported ("U ...@GLIBC_*") one? This is a real, valid heuristic —
#    unlike grepping for a Rust-mangled path, it does not have the
#    `#[no_mangle]` blind spot described above. A hit here is suspicious;
#    the absence of a hit is *not* proof of safety (see below).
nm -D <binary> | grep -E '^[0-9a-f]+ T '
nm -D <binary> | grep -E '@GLIBC'

# 2. Definitive, runtime check: read the actual value written into the
#    GOT slot for the suspect symbol from a live process, and confirm it
#    resolves to a real symbol inside the expected shared library.
gdb -p <pid> -batch \
  -ex "print/x *(void**)<runtime-GOT-slot-address>" \
  -ex "x/3i *(void**)<runtime-GOT-slot-address>"
# The GOT slot's runtime address = (binary's load base, from
# /proc/<pid>/maps) + (static file offset of the relocation, from
# `readelf -r <binary> | grep <symbol>@GLIBC`).
```

If step 2 shows the slot resolving to an address inside the binary's own
mapped range (rather than inside the target shared library's), symbol
interposition is confirmed.

## 4. What is now in place to prevent recurrence

- **[`.cargo/hide-libm-shadow.map`](../.cargo/hide-libm-shadow.map)** — a linker version script listing the
  entire standard libm C symbol surface (f32/f64: trig, exp/log family,
  hypot/erf/tgamma/ldexp/fmod, etc.) as `local`. Any of these names that
  ever end up defined in our own binary again — from any dependency, now
  or in the future — will never again be eligible for `ld.so`'s global
  symbol interposition. This is the actual, durable fix; it does not
  depend on knowing which dependency caused the problem.
- **[`build.rs`](../build.rs)** — applies the version script (plus `-Wl,--undefined-version`,
  required because this toolchain's linker, `lld`, otherwise errors on any
  version-script entry for a symbol not defined in a given binary) via
  `cargo:rustc-link-arg`, scoped to this crate's own link targets only.
  This is deliberately *not* a global `[build] rustflags` entry in
  `.cargo/config.toml` — that also applies to every dependency's own
  build-script helper binaries, which resolve the version script's
  relative path from a different working directory and fail to find it.
- **[`tests/libm_export_guard.rs`](../tests/libm_export_guard.rs)** — the canonical,
  fail-closed ELF surface gate: it runs `nm`/`llvm-nm` over the *linked*
  test binary (`current_exe`) and fails on any GLOBAL/WEAK export of the
  libm symbol surface (or panics if no `nm` is available — never a silent
  skip). It is wired into `utils/tests-quick.sh` Phase 1 and
  `utils/tests-long.sh` Defense phase. The former standalone wrapper
  `utils/debug/verify_no_libm_exports.sh` was removed: it scanned
  `.rlib` archives — the wrong surface, since object archives still carry
  `T` exports before the version script applies — and exited 0 when the
  artifact was missing.
- **[`utils/debug/repro_oversample_hang.sh`](../utils/debug/repro_oversample_hang.sh)** — a general-purpose safety
  wrapper for reproducing any suspected hang: cgroup-scoped
  `RuntimeMaxSec` (kills the whole process tree, not just a direct
  child), no pipe between the wrapped command and the status capture (a
  `| tee` pipeline previously produced a false "exit 0" for a run that
  had actually hung), and an automatic post-run residual-process check.
  Reuse this wrapper — don't write a new one — for any future hang
  investigation.
- **The reactivated test** ([`test_x2_aliasing_rejection`](../src/dsp/oversample_test.rs#L74),
  [`src/dsp/oversample_test.rs`](../src/dsp/oversample_test.rs)) now runs unignored, in both debug and
  release, as part of the normal `--lib` unit-test pass exercised by both
  `utils/tests-quick.sh` and `utils/tests-long.sh`. It exercises exactly
  the code path (a runtime, non-const `f32::log10()` call inside a hot
  DSP function) that this whole class of bug affects, so it doubles as a
  living regression check integrated into the normal test suite.

## 5. Guidance for future dependency/toolchain changes

- Adding a new dependency that has a `no_std`/`libm`/portable-math feature
  (common in numeric, embedded, or `no_std`-compatible crates) is the most
  likely way this class of bug could reappear if the version-script
  protection in `.cargo/hide-libm-shadow.map` were ever removed or
  bypassed. It should not be removed without replacing it with an
  equivalent guard.
- If `.cargo/hide-libm-shadow.map`'s list needs extending (e.g. a new
  libm function is used in a hot path and the fix should cover it too),
  add the name to both the `local:` list and, if relevant, note it here —
  but the list is intentionally already broad (the full standard libm
  C89/C99 surface), not just the handful of symbols empirically observed
  to be affected, precisely so it does not need to be extended reactively
  every time a new function is used.
- Any time the toolchain (`rustc`/`compiler_builtins`) or the linker
  (`ld`/`lld`/`bfd`) changes, run the test suite (including the
  reactivated `test_x2_aliasing_rejection`) as a sanity check before
  trusting a release build.
