# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

# Maintainer entry point. Thin delegation only: every recipe invokes the
# canonical script in utils/ with the same arguments. No logic duplicated here.

setup:
	./utils/setup-third-party.sh

lint:
	./utils/lints.sh

test:
	./utils/tests-quick.sh

check:
	./utils/quality-dashboard.sh --check docs/quality-contract.json

bench:
	./utils/tests-performance-regression.sh --check

# Operator-only: long suite is human-owned (±50 min, unattended). Do not
# execute it from automation.
long:
	@echo 'Operator-only task: run ./utils/tests-long.sh manually (approx. 50 min, unattended).'
	@echo 'This recipe intentionally does not execute the script; automation must not run it.'
