# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

# Maintainer entry point. Thin delegation only: every recipe invokes the
# canonical script in utils/ with the same arguments. No logic duplicated here.

# Run SIMD capability and hardware pre-flight probe
probe *args:
	./utils/simd-probe.sh {{args}}

# Setup vendor mirrors and third-party dependencies
setup *args:
	./utils/setup-third-party.sh {{args}}

# Run code formatting, cargo clippy, and static analysis lints
lint *args:
	./utils/lints.sh {{args}}

# Run agile quick QA test suite (cargo test gate)
test *args:
	./utils/tests-quick.sh {{args}}

# Verify quality and fidelity against the baseline contract
check:
	./utils/quality-dashboard.sh --check docs/quality-contract.json

# Run quality dashboard report (supports --fidelity-only, --bench-only, etc.)
dashboard *args:
	./utils/quality-dashboard.sh {{args}}

# Run performance regression gate against saved Criterion baseline
bench *args:
	./utils/tests-performance-regression.sh {{args}}

# Operator-only: long suite is human-owned (±50 min, unattended). Do not
# execute it from automation.
long:
	@echo 'Operator-only task: run ./utils/tests-long.sh manually (approx. 50 min, unattended).'
	@echo 'This recipe intentionally does not execute the script; automation must not run it.'
