#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
"""Unified RT-codegen disassembly filter and static gate (NAM ecosystem).

Reads a full ``objdump -d --demangle --no-show-raw-insn`` dump, keeps only the
``.text`` / ``.text.cold*`` sections (excluding pre-BOLT ``.bolt.org.text``
duplicates) and only the symbols selected by a per-repository *profile*, writes
a filtered report (``--filtered``) plus a per-symbol summary (``--summary``),
and enforces fail-closed static gates.

This single implementation replaces the previously divergent filters:
``NAM-Audio-Pipe/utils/dsp-hotpath-filter.py`` and the inline Python inside
``NAM-Plug/utils/verify_rt_codegen.sh``. It is driven by a per-repo profile
(``--profile``) so no host/plugin/application names leak into the code.

Profile JSON schema::

    {
      "sections": [".text", ".text.cold"],          # optional (default)
      "gates":    {"zmm": 0, "libm": 0, "alloc": 0},# optional simple-mode gates
      "symbols":  "regex",                          # simple single-target mode
      "targets": [                                  # advanced mode
        {"id": "...", "pattern": "regex", "required": true,
         "gates": {"zmm": 0, "libm": 0, "alloc": 0, "div": 0, "tls": 0}}
      ]
    }

A gate value is the maximum allowed count (``0`` = forbid, absent/``null`` =
disabled). ``--symbols REGEX`` selects simple mode inline (equivalent to a
profile with a single required target). ``--test-synthetic-violation`` injects
one violation so callers can prove the gate actually fails closed.

Exit 0 when every gate passes, 1 when any gate fails (fail-closed).
"""

from __future__ import annotations

import argparse
import json
import re
import sys


SECTION_RE = re.compile(r"^\s*Disassembly of section (\S+?):\s*$")
SYMBOL_RE = re.compile(r"^([0-9a-fA-F]+)\s+<(.+)>:\s*$")
INSTR_RE = re.compile(r"^\s+[0-9a-fA-F]+:\s+")
VFMADD_RE = re.compile(r"vfmadd", re.IGNORECASE)
CALL_RE = re.compile(r"\bcallq?\b")
SPILL_RE = re.compile(r"\(%rsp\)")
ZMM_RE = re.compile(r"zmm\d+", re.IGNORECASE)
# Bare libm symbols only (``<tanhf@GLIBC_...>``); namespaced Rust symbols such
# as ``<...::tanh_and_...>`` never match because ``<`` must directly precede
# the libm name.
LIBM_RE = re.compile(
    r"call\w*\s+.*<(tanhf?|expf?|logf?|powf?|sinf?|cosf?|atan2?f?|"
    r"atanf?|sqrtf?|fmodf?|sincosf?)(@|>)"
)
# Heap allocation / deallocation calls on the audio path (Rust and libc forms).
ALLOC_RE = re.compile(
    r"\bcall\w*\s+.*(__rust_alloc|__rust_dealloc|__rust_realloc|__rg_alloc|"
    r"<alloc::|\bmalloc\b|\bcalloc\b|\brealloc\b|\bfree\b)"
)
# Integer division (div/idiv, all operand widths); excludes FP `divss`/`divps`.
DIV_RE = re.compile(r"\b(i?div[bwdlq]?)\b")
# Thread-local storage accessor call (allocates/locks in some toolchains).
TLS_RE = re.compile(r"\bcall\w*\s+.*__tls_get_addr")

DEFAULT_SECTIONS = (".text", ".text.cold")
GATE_KEYS = ("zmm", "libm", "alloc", "div", "tls")
SYNTHETIC_VIOLATION = "SYNTHETIC: violation injected by --test-synthetic-violation"


def is_kept_section(section: str, sections: tuple[str, ...]) -> bool:
    """Accept ``.text`` and ``.text.cold*`` (reject pre-BOLT duplicates)."""
    for wanted in sections:
        if section == wanted or section.startswith(wanted + "."):
            return True
    return False


def parse_args() -> argparse.Namespace:
    """Parse the filter CLI (dump paths plus profile/symbols selector)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", required=True, help="Full objdump dump input")
    parser.add_argument("--filtered", required=True, help="Filtered hotspot output")
    parser.add_argument("--summary", required=True, help="Per-symbol summary output")
    parser.add_argument(
        "--symbols",
        help="Python regex selecting hotspot symbols (simple inline mode)",
    )
    parser.add_argument(
        "--profile",
        help="JSON profile selecting symbols/targets and gate budgets",
    )
    parser.add_argument(
        "--test-synthetic-violation",
        action="store_true",
        help="Inject a synthetic violation to prove the gate fails closed",
    )
    args = parser.parse_args()
    if not args.symbols and not args.profile:
        parser.error("one of --symbols or --profile is required")
    return args


def load_profile(args: argparse.Namespace) -> tuple[list[dict], tuple[str, ...]]:
    """Resolve the profile into a list of target specs and kept sections."""
    sections = DEFAULT_SECTIONS
    targets: list[dict] = []

    if args.profile:
        try:
            with open(args.profile, "r", encoding="utf-8") as handle:
                doc = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[FATAL] cannot read profile {args.profile}: {exc}", file=sys.stderr)
            raise SystemExit(1)
        if not isinstance(doc, dict):
            print(f"[FATAL] profile must be a JSON object: {args.profile}", file=sys.stderr)
            raise SystemExit(1)
        if isinstance(doc.get("sections"), list) and doc["sections"]:
            sections = tuple(str(s) for s in doc["sections"])
        if isinstance(doc.get("targets"), list) and doc["targets"]:
            for raw in doc["targets"]:
                if not isinstance(raw, dict) or "pattern" not in raw:
                    print(f"[FATAL] malformed target in profile {args.profile}", file=sys.stderr)
                    raise SystemExit(1)
                targets.append(
                    {
                        "id": str(raw.get("id") or raw["pattern"]),
                        "pattern": str(raw["pattern"]),
                        "required": bool(raw.get("required", True)),
                        "gates": dict(raw.get("gates") or {}),
                        "simple": False,
                    }
                )
        elif doc.get("symbols"):
            targets.append(
                {
                    "id": str(doc.get("id") or "hotspot"),
                    "pattern": str(doc["symbols"]),
                    "required": True,
                    "gates": dict(doc.get("gates") or {}),
                    "simple": True,
                }
            )
        else:
            print(f"[FATAL] profile has neither 'targets' nor 'symbols': {args.profile}", file=sys.stderr)
            raise SystemExit(1)

    if args.symbols:
        targets.append(
            {
                "id": "hotspot",
                "pattern": args.symbols,
                "required": True,
                "gates": {"zmm": 0, "libm": 0, "alloc": 0},
                "simple": True,
            }
        )

    # Simple mode forbids zmm/libm/alloc by default; explicit target specs are
    # left exactly as authored (an absent gate is disabled, not defaulted).
    for target in targets:
        gates = target["gates"]
        if target["simple"]:
            for key in ("zmm", "libm", "alloc"):
                gates.setdefault(key, 0)
        for key in GATE_KEYS:
            value = gates.get(key)
            if value is not None:
                try:
                    gates[key] = int(value)
                except (TypeError, ValueError):
                    print(f"[FATAL] gate '{key}' must be an integer or null", file=sys.stderr)
                    raise SystemExit(1)
        # Pre-compile the pattern (fail-closed on an invalid regex).
        try:
            target["_regex"] = re.compile(target["pattern"])
        except re.error as exc:
            print(f"[FATAL] invalid target regex for '{target['id']}': {exc}", file=sys.stderr)
            raise SystemExit(1)
    return targets, sections


def main() -> int:
    """Filter, summarize and gate the disassembly; return 0 on pass, 1 on fail."""
    args = parse_args()
    targets, sections = load_profile(args)

    try:
        with open(args.full, "r", encoding="utf-8", errors="replace") as handle:
            lines = handle.readlines()
    except OSError as exc:
        print(f"[FATAL] cannot read full dump {args.full}: {exc}", file=sys.stderr)
        return 1

    kept: list[str] = []
    # symbol -> {"target": spec, "stats": {...}, "hits": {gate: [lines]}}
    matched: dict[str, dict] = {}
    cur_section = ""
    cur_symbol: str | None = None
    cur_target: dict | None = None

    for raw in lines:
        line = raw.rstrip("\n")
        section_match = SECTION_RE.match(line)
        if section_match:
            cur_section = section_match.group(1)
            cur_symbol = None
            cur_target = None
            if is_kept_section(cur_section, sections):
                kept.append(raw)
            continue
        if not is_kept_section(cur_section, sections):
            continue
        symbol_match = SYMBOL_RE.match(line.strip()) or SYMBOL_RE.match(line)
        if symbol_match:
            cur_symbol = symbol_match.group(2)
            cur_target = None
            for target in targets:
                if target["_regex"].search(cur_symbol):
                    cur_target = target
                    break
            if cur_target is not None:
                kept.append(raw)
                matched.setdefault(
                    cur_symbol,
                    {
                        "target": cur_target,
                        "stats": {"instructions": 0, "vfmadd": 0, "call": 0, "spills": 0},
                        "hits": {key: [] for key in GATE_KEYS},
                    },
                )
            continue
        if cur_symbol is None or cur_target is None:
            continue
        if not INSTR_RE.match(line):
            continue
        kept.append(raw)
        entry = matched[cur_symbol]
        entry["stats"]["instructions"] += 1
        if VFMADD_RE.search(line):
            entry["stats"]["vfmadd"] += 1
        if CALL_RE.search(line):
            entry["stats"]["call"] += 1
        if SPILL_RE.search(line):
            entry["stats"]["spills"] += 1
        for key, regex in (("zmm", ZMM_RE), ("libm", LIBM_RE), ("alloc", ALLOC_RE),
                           ("div", DIV_RE), ("tls", TLS_RE)):
            if regex.search(line):
                entry["hits"][key].append(line.strip())

    # --- Filtered report -----------------------------------------------------
    header = (
        "# DSP hotspot report (filtered) — critical DSP loops only\n"
        f"# Source full dump: {args.full}\n"
        f"# Sections: {', '.join(sections)} (pre-BOLT duplicates excluded)\n"
        f"# Targets: {', '.join(t['id'] for t in targets)}\n"
        "# Generator: utils/dsp-hotpath-filter.py (unified NAM codegen filter)\n"
    )
    try:
        with open(args.filtered, "w", encoding="utf-8") as handle:
            handle.write(header)
            handle.writelines(kept)
    except OSError as exc:
        print(f"[FATAL] cannot write filtered report: {exc}", file=sys.stderr)
        return 1

    # --- Summary + gates -----------------------------------------------------
    total_instr = sum(v["stats"]["instructions"] for v in matched.values())
    total_vfmadd = sum(v["stats"]["vfmadd"] for v in matched.values())
    total_call = sum(v["stats"]["call"] for v in matched.values())

    failures: list[str] = []
    target_symbol_count = {t["id"]: 0 for t in targets}
    for entry in matched.values():
        target_symbol_count[entry["target"]["id"]] += 1

    gate_lines: list[str] = []
    for target in targets:
        if target["required"] and target_symbol_count[target["id"]] == 0:
            failures.append(f"target '{target['id']}': zero matching symbols (required, fail-closed)")
            gate_lines.append(f"target {target['id']}: MISSING (required)")
            continue
        for key in GATE_KEYS:
            budget = target["gates"].get(key)
            if budget is None:
                continue
            hits = [
                line
                for sym, entry in matched.items()
                if entry["target"] is target
                for line in entry["hits"][key]
            ]
            status = "PASS" if len(hits) <= budget else "FAIL"
            gate_lines.append(
                f"target {target['id']} gate {key}: {status} ({len(hits)} <= {budget})"
            )
            if len(hits) > budget:
                failures.append(
                    f"target '{target['id']}': {len(hits)} {key} hit(s) > budget {budget}"
                )

    if args.test_synthetic_violation:
        failures.append(SYNTHETIC_VIOLATION)

    try:
        with open(args.summary, "w", encoding="utf-8") as handle:
            handle.write("DSP hotspot summary (per symbol)\n")
            handle.write(
                f"symbols={len(matched)} instructions={total_instr} "
                f"vfmadd={total_vfmadd} call={total_call}\n"
            )
            handle.write("symbol | target | instructions | vfmadd | call | spills(%rsp)\n")
            for symbol in sorted(matched):
                entry = matched[symbol]
                handle.write(
                    f"{symbol} | {entry['target']['id']} | {entry['stats']['instructions']} | "
                    f"{entry['stats']['vfmadd']} | {entry['stats']['call']} | "
                    f"{entry['stats']['spills']}\n"
                )
            for line in gate_lines:
                handle.write(line + "\n")
            for failure in failures:
                handle.write(f"FAIL: {failure}\n")
    except OSError as exc:
        print(f"[FATAL] cannot write summary: {exc}", file=sys.stderr)
        return 1

    if failures:
        print(f"[FATAL] RT codegen gate FAILED ({args.full}):", file=sys.stderr)
        for failure in failures:
            print(f"  ❌ {failure}", file=sys.stderr)
            if failure == SYNTHETIC_VIOLATION:
                continue
        # Emit up to 10 representative offending instructions for triage
        # (only gates that are actually enabled for the symbol's target).
        shown = 0
        for entry in matched.values():
            for key in GATE_KEYS:
                if entry["target"]["gates"].get(key) is None:
                    continue
                for line in entry["hits"][key][:1]:
                    print(f"  hit[{key}]: {line}", file=sys.stderr)
                    shown += 1
                    if shown >= 10:
                        break
            if shown >= 10:
                break
        return 1

    print(
        f"  OK hotspot gates PASS: {len(matched)} symbols, {total_instr} instructions, "
        f"vfmadd={total_vfmadd}, targets={len(targets)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
