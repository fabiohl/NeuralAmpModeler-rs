<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# Human Certification Protocol for QA Runners

This document establishes the official operator protocol, execution checklists, defense log verification steps, and evidence records for running the test suites: **Agile Quick Suite** ([`utils/tests-quick.sh`](../utils/tests-quick.sh)) and **Nightly / Pre-Release Long Suite** ([`utils/tests-long.sh`](../utils/tests-long.sh)).

> [!IMPORTANT]
> **AI Agent Execution Prohibition:** AI agents must **never** execute `utils/tests-long.sh`, `utils/tests-performance-regression.sh --bootstrap-baseline`, or `utils/quality-dashboard.sh --save`. These operations are exclusively reserved for human operators under controlled hardware environments.

---

## 1. Environmental Prerequisites (Operator Machine)

Before executing pre-release audits or baseline updates, verify the host environment:

1. **CPU Frequency Scaling Governor:**
   Must be set to `performance` across all physical cores to prevent frequency throttling and timer jitter during micro-benchmarking and deadline testing.

   ```bash
   # Check governor status
   cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor | sort -u
   # Expected: performance
   ```

2. **Low System Load & Thermal Stability:**
   Close all heavy background services, IDE indexers, browsers, and background compilers.

3. **Core Pinning & Priority:**
   The runner scripts automatically configure CPU affinity (`taskset`) and nice priority (`nice -n 19 ionice -c 3`) unless `NAM_NO_LOW_PRIORITY=1` is specified.

4. **Upstream Vendor Mirrors & Fixtures:**
   Ensure `third-party/NeuralAmpModelerCore/` is populated and matches the commit pinned in [`variables.env`](../variables.env).

   ```bash
   ./utils/setup-third-party.sh
   ```

---

## 2. Quick Suite Protocol (`utils/tests-quick.sh`)

The quick suite serves as the agile first line of defense (~2–3 minutes runtime).

### 2.1 Execution Modes

```bash
# Standard interactive / agile verification
./utils/tests-quick.sh

# Strict release-gate mode (promotes any missing fixture or toolchain gap to hard failure)
NAM_QUICK_STRICT=1 ./utils/tests-quick.sh
```

### 2.2 Phase Layout

- **Phase 1 (Structural & Logic, Debug):** Core unit tests, DSP logic, parsers, FSM, and SPSC channels.
- **Phase 2 (Measurement Oracles & Parity, Release):** Evaluates production float codegen across `golden_vectors`, `reference_oracle_f64`, `spectral_fidelity`, `linear_fft_test`, and C++ live parity (`quick_parity`).
- **Phase 3 (Parser Fuzzing, Release `--ignored`):** Capped `proptest` sweeps on `.nam` and `.namb` formats.

### 2.3 Receipt & Log Verification

Every execution emits machine-readable logs and a receipt in `target/logs/`:

- Receipt file: `target/logs/quick-receipt.txt`
- Phase logs: `target/logs/quick-phase1.log`, `target/logs/quick-phase2.log`, `target/logs/quick-phase3.log`

**Expected Outcome:**

- Interactive / Agile: `FIDELITY: OK` or `FIDELITY: INCOMPLETE` (with documented `GAP:` entries if vendor mirrors are absent) and `OVERALL: PASSED` (exit 0).
- Release Certification: Must run with `NAM_QUICK_STRICT=1` and yield `FIDELITY: OK` and `OVERALL: PASSED` (exit 0) with zero gaps.

---

## 3. Long Audit Suite Protocol (`utils/tests-long.sh`)

The long audit suite provides exhaustive, multi-phase pre-release validation (~45–60 minutes unattended).

### 3.1 Preflight Defense Verification

Ahead of any timed test phase, the runner executes blocking preflight gates:

1. `preflight-render`: Builds or validates the C++ `render` binary via `utils/ensure_namcore_render.sh`.
2. `preflight-catalog`: Validates fixture presence against `src/testing/catalog.rs`.
3. `preflight-freshness`: Enforces SHA-256 freshness against `tests/fixtures/.golden_manifest.sha256`.
4. `preflight-meta`: Asserts catalog↔test metadata coherence (`meta_coherence.rs`).

### 3.2 Execution

```bash
# Full nightly / pre-release audit
./utils/tests-long.sh

# Strict pre-release mode (turns any optional capability gap into failure)
./utils/tests-long.sh --strict-pre-release
```

### 3.3 Phase Breakdown

1. **Phase 1 — Soak & Concurrency:** 10M+ frames continuous endurance, lock-free SPSC contention sweeps.
2. **Phase 2 — Defense Scripts & Invariant Tests:** Shell unit tests (`run_bash_scripts_unit_tests`), ELF export guard (`libm_export_guard`).
3. **Phase 3 — Exhaustive Matrix & Parity:** Full C++ live parity matrix, v2 multi-SR goldens, cross-ISA validation, spectral baselines, full proptest sweeps (up to 100k cases).
4. **Phase 4 — Heap Audit:** Strict verification of zero heap allocations on the audio processing hot path (`CountingAllocator`).
5. **Phase 5 — RT Deadline & Constraints:** Validates $p99 < 1.33\text{ ms}$ processing budget per 64-sample block at 48 kHz.
6. **Phase 6 — RT Jitter Telemetry:** Measures processing latency distribution under simulated CPU contention.
7. **Phase 7 — Concurrency Model Checking:** Loom model verification for lock-free queues and atomic bitmasks.

### 3.4 Structured Audit Receipt (`long-audit-receipt.jsonl`)

The suite writes a structured JSONL receipt (`target/logs/long-audit-receipt.jsonl`). Each phase appends a record:

```json
{"phase_id":"phase1","name":"Soak Tests (Numerical Stability)","status":"PASSED","duration_ms":42000,"tests_executed":26,"gaps":[],"timestamp":"2026-08-14T03:00:00Z"}
```

Final verification:

```bash
# Verify receipt integrity and summary verdict
cargo run --locked --bin nam_long_receipt -- validate target/logs/long-audit-receipt.jsonl
```

---

## 4. Human Pre-Release Certification Record Template

When certifying a release candidate or significant milestone, the human operator completes and archives the following record:

```markdown
### Release / Audit Certification Record

- **Date (UTC):** YYYY-MM-DD HH:MM:SS
- **Operator Name:** Fábio Henrique de Lima Silva
- **Git Commit:** <commit-sha> (clean working tree required)
- **Rustc Version:** rustc X.Y.Z (Edition 2024)
- **CPU Architecture & Model:** <lscpu output summary>
- **CPU Governor:** performance

#### Verification Checklist:
- [ ] `utils/lints.sh` executed cleanly (zero warnings, formatting intact, SPDX headers verified).
- [ ] `NAM_QUICK_STRICT=1 utils/tests-quick.sh` passed with `FIDELITY: OK` and `OVERALL: PASSED` (zero gaps).
- [ ] `utils/tests-long.sh --strict-pre-release` executed with `OVERALL: PASSED` (receipt validated via `nam_long_receipt validate`).
- [ ] `utils/tests-performance-regression.sh --check` passed without regression against baseline.
- [ ] `utils/quality-dashboard.sh --check docs/quality-contract.txt` satisfied all fidelity and latency envelopes.

#### Certification Verdict:
[ APPROVED FOR RELEASE / REJECTED ]
```
