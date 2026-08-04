#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

import hashlib
import json
import os
import sys

# Check terminal color support (respecting NO_COLOR and TERM=dumb env variables)
USE_COLOR = (
    sys.stdout.isatty()
    and os.environ.get("NO_COLOR") is None
    and os.environ.get("TERM") != "dumb"
)


def color(code, text):
    if USE_COLOR:
        return f"\033[{code}m{text}\033[0m"
    return text


# Standard A1 WaveNet topologies
STD_DILATIONS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
LITE_DILATIONS_1 = [1, 2, 4, 8, 16, 32, 64]
LITE_DILATIONS_2 = [128, 256, 512, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

# Standard LSTM configs from model mappings
STANDARD_LSTMS = {
    1: {3, 8, 12, 16, 24, 40},
    2: {8, 12, 16, 24},
}


def classify_model(data, filename=""):
    """Classifies a .nam neural model file by topology and architectural parameters.

    Returns a metadata dictionary containing:
      - name, author, version: descriptive properties
      - arch: model architecture label
      - details: descriptive topology details
      - status: short classification summary
      - is_goal: boolean indicating non-standard target topology
    """
    if not isinstance(data, dict):
        return {
            "name": os.path.basename(filename) if filename else "Unknown",
            "author": "Unknown Author",
            "version": "unknown",
            "arch": "Invalid JSON",
            "details": "Root JSON element is not a dictionary object",
            "status": "Invalid model structure",
            "is_goal": False,
        }

    arch = data.get("architecture")
    config = data.get("config") or {}
    if not isinstance(config, dict):
        config = {}

    version = data.get("version", "unknown")
    metadata = data.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}

    modeled_by = metadata.get("modeled_by")
    model_name = metadata.get("name")

    author_str = str(modeled_by) if modeled_by is not None else "Unknown Author"
    name_str = (
        str(model_name)
        if model_name is not None
        else (os.path.basename(filename) if filename else "Unknown Model")
    )

    if arch == "WaveNet":
        # Check if it is a multi-model container (SlimmableContainer)
        submodels = config.get("submodels")
        if submodels is not None and isinstance(submodels, list):
            return {
                "name": name_str,
                "author": author_str,
                "version": version,
                "arch": "SlimmableContainer",
                "details": f"SlimmableContainer with {len(submodels)} submodels",
                "status": "Slimmable Multi-Model Container",
                "is_goal": True,
            }

        layers = config.get("layers", [])
        if not isinstance(layers, list):
            layers = []

        # Check for A2 shape criteria
        is_a2 = False
        a2_reason = ""
        if len(layers) == 1:
            l0 = layers[0]
            if isinstance(l0, dict):
                if l0.get("kernel_sizes") is not None or l0.get("bottleneck") is not None:
                    is_a2 = True
                    a2_reason = "A2 Shape (1 layer, bottleneck/kernel_sizes present)"
                elif isinstance(l0.get("activation"), list):
                    is_a2 = True
                    a2_reason = "A2 activation array"

        # Check for multi-condition / FiLM (condition_size > 1)
        for i, l in enumerate(layers):
            if isinstance(l, dict):
                cond_size = l.get("condition_size")
                if (
                    cond_size is not None
                    and isinstance(cond_size, (int, float))
                    and cond_size > 1
                ):
                    return {
                        "name": name_str,
                        "author": author_str,
                        "version": version,
                        "arch": "WaveNet A2 (FiLM)",
                        "details": f"Layer {i} has condition_size={cond_size}",
                        "status": "WaveNet A2 FiLM / Multi-Condition model",
                        "is_goal": True,
                    }

        if is_a2:
            return {
                "name": name_str,
                "author": author_str,
                "version": version,
                "arch": "WaveNet A2",
                "details": a2_reason,
                "status": "WaveNet A2 General topology",
                "is_goal": True,
            }

        # Check standard A1 topologies (expects 2 layers)
        if len(layers) == 2:
            l0 = layers[0] if isinstance(layers[0], dict) else {}
            l1 = layers[1] if isinstance(layers[1], dict) else {}
            ch0 = l0.get("channels")
            dils0 = l0.get("dilations", [])
            dils1 = l1.get("dilations", [])
            if not isinstance(dils0, list):
                dils0 = []
            if not isinstance(dils1, list):
                dils1 = []

            # Lite topology check: 12 channels, standard lite dilation arrays
            if ch0 == 12 and dils0 == LITE_DILATIONS_1 and dils1 == LITE_DILATIONS_2:
                return {
                    "name": name_str,
                    "author": author_str,
                    "version": version,
                    "arch": "WaveNet A1 Lite (CH=12)",
                    "details": "Standard A1 Lite topology",
                    "status": "WaveNet A1 Lite topology (CH=12)",
                    "is_goal": True,
                }

            # Standard geometries: 16 (Standard), 8 (Feather), 4 (Nano)
            is_standard = False
            if ch0 == 16 and dils0 == STD_DILATIONS and dils1 == STD_DILATIONS:
                is_standard = True
            elif ch0 == 8 and dils0 == LITE_DILATIONS_1 and dils1 == LITE_DILATIONS_2:
                is_standard = True
            elif ch0 == 4 and dils0 == LITE_DILATIONS_1 and dils1 == LITE_DILATIONS_2:
                is_standard = True

            if not is_standard:
                return {
                    "name": name_str,
                    "author": author_str,
                    "version": version,
                    "arch": "WaveNet A1 (Custom)",
                    "details": f"Non-standard shape: CH={ch0}, dilations_len={len(dils0)}/{len(dils1)}",
                    "status": "WaveNet A1 Non-standard geometry",
                    "is_goal": True,
                }
            else:
                return {
                    "name": name_str,
                    "author": author_str,
                    "version": version,
                    "arch": f"WaveNet A1 (Standard CH={ch0})",
                    "details": "Standard A1 topology",
                    "status": "Standard Supported Model",
                    "is_goal": False,
                }
        else:
            return {
                "name": name_str,
                "author": author_str,
                "version": version,
                "arch": "WaveNet (Custom Layers)",
                "details": f"Number of layers is {len(layers)} (expected 2)",
                "status": "WaveNet Custom layer count",
                "is_goal": True,
            }

    elif arch == "LSTM":
        num_layers = config.get("num_layers")
        hidden_size = config.get("hidden_size")

        if num_layers is None or hidden_size is None:
            return {
                "name": name_str,
                "author": author_str,
                "version": version,
                "arch": "LSTM (Invalid Config)",
                "details": "Missing num_layers or hidden_size",
                "status": "Invalid model structure",
                "is_goal": False,
            }

        allowed = STANDARD_LSTMS.get(num_layers, set())
        if hidden_size not in allowed:
            return {
                "name": name_str,
                "author": author_str,
                "version": version,
                "arch": f"LSTM {num_layers}x{hidden_size} (Custom)",
                "details": f"Non-standard geometry: layers={num_layers}, hidden={hidden_size}",
                "status": "LSTM Non-standard shape",
                "is_goal": True,
            }
        else:
            return {
                "name": name_str,
                "author": author_str,
                "version": version,
                "arch": f"LSTM {num_layers}x{hidden_size} (Standard)",
                "details": "Standard LSTM topology",
                "status": "Standard Supported Model",
                "is_goal": False,
            }

    elif arch == "Linear":
        rf = config.get("receptive_field", 0)
        bias = config.get("bias", False)
        return {
            "name": name_str,
            "author": author_str,
            "version": version,
            "arch": f"Linear (RF={rf}, bias={bias})",
            "details": "Linear model structure",
            "status": "Standard Supported Model",
            "is_goal": False,
        }

    elif arch == "ConvNet":
        channels = config.get("channels")
        return {
            "name": name_str,
            "author": author_str,
            "version": version,
            "arch": f"ConvNet (CH={channels})",
            "details": "ConvNet architecture",
            "status": "Standard Supported Model",
            "is_goal": False,
        }

    else:
        return {
            "name": name_str,
            "author": author_str,
            "version": version,
            "arch": f"Unknown ({arch})",
            "details": "Unsupported architecture",
            "status": "Unknown / Unsupported",
            "is_goal": False,
        }


def main():
    manifest_mode = False
    file_args = []

    for arg in sys.argv[1:]:
        if arg == "--manifest":
            manifest_mode = True
        else:
            file_args.append(arg)

    if not file_args:
        print("Usage: utils/check-model.py [--manifest] <path_to_model.nam> [path_to_model2.nam ...]")
        sys.exit(1)

    if manifest_mode:
        manifest_entries = []
        for filepath in file_args:
            if not os.path.exists(filepath):
                continue
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue

            info = classify_model(data, filepath)

            try:
                with open(filepath, "rb") as f:
                    sha256_hash = hashlib.sha256(f.read()).hexdigest()
            except Exception:
                sha256_hash = ""

            manifest_entries.append({
                "filename": os.path.basename(filepath),
                "sha256": sha256_hash,
                "expected_class": info["arch"],
                "is_goal_target": info["is_goal"],
                "name": info["name"],
                "author": info["author"],
            })
        print(json.dumps(manifest_entries, indent=2))
        return

    for filepath in file_args:
        if not os.path.exists(filepath):
            print(color("91", f"Error: File not found: {filepath}"))
            continue

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(color("91", f"Error: Failed to parse JSON for {filepath}: {e}"))
            continue

        info = classify_model(data, filepath)

        if info["is_goal"]:
            status_color = "92"
            tag = "[TARGET MATCH]"
        else:
            status_color = "94"
            tag = "[STANDARD/SUPPORTED]"

        print("================================================================")
        print(f"File: {filepath}")
        print(f"Name: {info['name']} (by {info['author']})")
        print(f"Version: {info['version']}")
        print(f"Architecture: {info['arch']}")
        print(f"Details: {info['details']}")
        print(color(status_color, f"Status: {tag} {info['status']}"))
        print("================================================================")


if __name__ == "__main__":
    main()

