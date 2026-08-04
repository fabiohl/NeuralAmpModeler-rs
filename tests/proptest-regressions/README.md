<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# Proptest Regressions

This directory contains the failure persistence seed files from the `proptest` framework.

## Purpose and How It Works

When a property-based test fails, `proptest` generates and saves a seed containing the exact input that caused the panic or assertion failure.

Our project is configured to save these regressions in `tests/proptest-regressions/` in an organized manner (via `FileFailurePersistence::SourceParallel`), avoiding clutter at the repository root.

## Importance of Version Control

Tracking these failure seeds in version control (Git) is a **best practice recommended by `proptest`** for two main reasons:

1. **Repeatability in CI:** Ensures that continuous integration (CI) and other developers immediately re-run the specific test cases that failed in the past, preventing fixed bugs from reappearing (regression).
2. **Determinism:** Tests with random inputs can be hard to reproduce without the exact failure seed. The persistence file removes that randomness for known errors.

If a test that previously failed now passes consistently and the fix has been consolidated, the corresponding file will continue to serve as a permanent baseline to attest the stability of that logic.
