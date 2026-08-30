// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Structured catalog and SHA-256 registry for test fixtures and community models.
//!
//! Provides a canonical, deduplicated registry of all 61 catalog paths mapped to
//! 51 unique SHA-256 identities. Classifies support status (45 supported, 3 intentional
//! negative fixtures, 3 known gaps) and tracks all 10 redundant file aliases.
//!
//! Also hosts the **V2 golden catalog** — the single source of truth for which
//! models participate in the V2 multi-SR golden vector matrix, which sample
//! rates each model requires, and which fixture files are expected on disk.
//! `validate_v2_catalog` powers the `catalog_preflight` gate; the shell
//! generator (`tests/fixtures/golden_gen_build.sh`) consumes the same registry
//! through `golden_gen_catalog_lines` (via the `nam_golden_catalog` binary), so
//! no bash array defines the catalog anymore.
//!
//! Also hosts the **V1 golden catalog** (`V1_GOLDEN_CATALOG`) — the single
//! source of truth for the 48 kHz v1 golden vectors (DistributedCore model
//! goldens, the LocalNonDistributable WaveNet Lite golden, and the CabSim
//! convolution goldens). `validate_v1_goldens` powers the same
//! `catalog_preflight` gate; the former bash lists `REQUIRED_GOLDEN_MODELS`,
//! `NONDIST_GOLDEN_MODELS` and `REQUIRED_CABSIM_GOLDENS` in
//! `utils/tests-long.sh` were removed.

use std::fs::File;
use std::io::{self, Read};
use std::path::Path;

use sha2::{Digest, Sha256};

use serde::{Deserialize, Serialize};

/// Support classification for a model identity in the test catalog.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ModelSupportKind {
    /// Fully supported model topology; parses and builds successfully.
    Supported,
    /// Intentional negative test fixture (e.g. invalid activation, mock topology).
    IntentionalNegative,
    /// Known architectural gap or unhandled feature (e.g. WaveNet LSTM condition, max CH).
    KnownGap,
}

/// A single unique SHA-256 identity in the model catalog.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelCatalogEntry {
    /// SHA-256 digest in lowercase hexadecimal (64 chars).
    pub sha256: &'static str,
    /// Primary canonical file path relative to workspace root.
    pub canonical_path: &'static str,
    /// Redundant alias file paths pointing to this same content identity.
    pub aliases: &'static [&'static str],
    /// Primary architecture family (WaveNet, LSTM, ConvNet, Linear, SlimmableContainer).
    pub architecture: &'static str,
    /// Support classification status.
    pub support: ModelSupportKind,
    /// Description of the model fixture.
    pub description: &'static str,
}

/// Canonical catalog array of all 51 unique SHA-256 model identities.
pub static MODEL_CATALOG: &[ModelCatalogEntry] = &[
    ModelCatalogEntry {
        sha256: "a5cadee4badb70ceb1a0b680b90d1ca1455bd26f7a05e0e7c4e05f854d530f69",
        canonical_path: "tests/fixtures/models/a2_dynamic_blended_ch3.nam",
        aliases: &[],
        architecture: "{'channels': 3, 'topology': 'A2-Dynamic', 'type': 'WaveNet', 'weights_layout': 'Original'}",
        support: ModelSupportKind::Supported,
        description: "Supported {'channels': 3, 'topology': 'A2-Dynamic', 'type': 'WaveNet', 'weights_layout': 'Original'} model (a2_dynamic_blended_ch3.nam)",
    },
    ModelCatalogEntry {
        sha256: "f027af37b5a7feff90564b750eb623c76147c2489d7df4408bea5cbbd8ea0b6e",
        canonical_path: "tests/fixtures/models/a2_dynamic_gated_ch8.nam",
        aliases: &[],
        architecture: "{'channels': 8, 'topology': 'A2-Dynamic', 'type': 'WaveNet', 'weights_layout': 'Original'}",
        support: ModelSupportKind::Supported,
        description: "Supported {'channels': 8, 'topology': 'A2-Dynamic', 'type': 'WaveNet', 'weights_layout': 'Original'} model (a2_dynamic_gated_ch8.nam)",
    },
    ModelCatalogEntry {
        sha256: "2d2d744516dc0197737a2c5001010429692d4cb20d72b08264781de626fcf4ca",
        canonical_path: "tests/fixtures/models/a2_example.nam",
        aliases: &["third-party/NeuralAmpModelerCore/example_models/A2.nam"],
        architecture: "{'channels': 8, 'topology': 'Unknown', 'type': 'SlimmableContainer', 'weights_layout': 'Original'}",
        support: ModelSupportKind::Supported,
        description: "Supported {'channels': 8, 'topology': 'Unknown', 'type': 'SlimmableContainer', 'weights_layout': 'Original'} model (a2_example.nam)",
    },
    ModelCatalogEntry {
        sha256: "4918e0525790541caba90a0ffeb8f73c407f8ecbb8da88dd44182d9ca7e08be9",
        canonical_path: "tests/fixtures/models/BossLSTM-1x16.nam",
        aliases: &[],
        architecture: "{'channels': 16, 'topology': '1x16', 'type': 'LSTM', 'weights_layout': 'Original'}",
        support: ModelSupportKind::Supported,
        description: "Supported {'channels': 16, 'topology': '1x16', 'type': 'LSTM', 'weights_layout': 'Original'} model (BossLSTM-1x16.nam)",
    },
    ModelCatalogEntry {
        sha256: "7c1a93c6ad9cfcf8fd3b186f3e72b21c4ad602b0b9c85d03e521fc6b7e375817",
        canonical_path: "tests/fixtures/models/BossLSTM-2x8.nam",
        aliases: &[],
        architecture: "{'channels': 8, 'topology': '2x8', 'type': 'LSTM', 'weights_layout': 'Original'}",
        support: ModelSupportKind::Supported,
        description: "Supported {'channels': 8, 'topology': '2x8', 'type': 'LSTM', 'weights_layout': 'Original'} model (BossLSTM-2x8.nam)",
    },
    ModelCatalogEntry {
        sha256: "4c4906a50e7b050517b47d64d6b90a5ed20174869a01a3cc0361c6b5544b38a5",
        canonical_path: "tests/fixtures/models/BossWN-feather.nam",
        aliases: &[],
        architecture: "{'channels': 8, 'topology': 'Feather', 'type': 'WaveNet', 'weights_layout': 'Original'}",
        support: ModelSupportKind::Supported,
        description: "Supported {'channels': 8, 'topology': 'Feather', 'type': 'WaveNet', 'weights_layout': 'Original'} model (BossWN-feather.nam)",
    },
    ModelCatalogEntry {
        sha256: "68d69b90053301d5f5efee511217cadec299ebb919885801528ffe36d9baadbe",
        canonical_path: "tests/fixtures/models/BossWN-lite.nam",
        aliases: &[],
        architecture: "{'channels': 12, 'topology': 'Lite', 'type': 'WaveNet', 'weights_layout': 'Original'}",
        support: ModelSupportKind::Supported,
        description: "Supported {'channels': 12, 'topology': 'Lite', 'type': 'WaveNet', 'weights_layout': 'Original'} model (BossWN-lite.nam)",
    },
    ModelCatalogEntry {
        sha256: "747bd1d2afd1efe3aa84a851112984dca3f4082e36af4db9f74ffe1e94d57f11",
        canonical_path: "tests/fixtures/models/BossWN-nano.nam",
        aliases: &[],
        architecture: "{'channels': 4, 'topology': 'Nano', 'type': 'WaveNet', 'weights_layout': 'Original'}",
        support: ModelSupportKind::Supported,
        description: "Supported {'channels': 4, 'topology': 'Nano', 'type': 'WaveNet', 'weights_layout': 'Original'} model (BossWN-nano.nam)",
    },
    ModelCatalogEntry {
        sha256: "0474d8e1593f9063b268c4d1636ce84ff62186ad1dfcc66533eceba39f952d65",
        canonical_path: "tests/fixtures/models/BossWN-standard.nam",
        aliases: &[],
        architecture: "{'channels': 16, 'topology': 'Standard', 'type': 'WaveNet', 'weights_layout': 'Original'}",
        support: ModelSupportKind::Supported,
        description: "Supported {'channels': 16, 'topology': 'Standard', 'type': 'WaveNet', 'weights_layout': 'Original'} model (BossWN-standard.nam)",
    },
    ModelCatalogEntry {
        sha256: "74bd7bf577faba99fdb9aff92339c87c8d55b56ac8b2e5e6f1a6f60d63fdbada",
        canonical_path: "tests/fixtures/models/convnet_nobn.nam",
        aliases: &[],
        architecture: "{'channels': 1, 'topology': 'B6', 'type': 'ConvNet', 'weights_layout': 'Original'}",
        support: ModelSupportKind::Supported,
        description: "Supported {'channels': 1, 'topology': 'B6', 'type': 'ConvNet', 'weights_layout': 'Original'} model (convnet_nobn.nam)",
    },
    ModelCatalogEntry {
        sha256: "a882e42ad351ac2bfbdec3d30a7af048d7390f1b749f56b6b3e4a361650175f3",
        canonical_path: "tests/fixtures/models/convnet_relu.nam",
        aliases: &[],
        architecture: "{'channels': 1, 'topology': 'B6', 'type': 'ConvNet', 'weights_layout': 'Original'}",
        support: ModelSupportKind::Supported,
        description: "Supported {'channels': 1, 'topology': 'B6', 'type': 'ConvNet', 'weights_layout': 'Original'} model (convnet_relu.nam)",
    },
    ModelCatalogEntry {
        sha256: "bb570f853c5d88ba4f74a993dd7939c72976184226bc401e728c9f0a99a007f6",
        canonical_path: "tests/fixtures/models/convnet_silu.nam",
        aliases: &[],
        architecture: "{'channels': 1, 'topology': 'B6', 'type': 'ConvNet', 'weights_layout': 'Original'}",
        support: ModelSupportKind::Supported,
        description: "Supported {'channels': 1, 'topology': 'B6', 'type': 'ConvNet', 'weights_layout': 'Original'} model (convnet_silu.nam)",
    },
    ModelCatalogEntry {
        sha256: "af46f05270ffac608b47707d89e059bb380aff675ecef8ad2093a909ae6b1221",
        canonical_path: "tests/fixtures/models/convnet_test.nam",
        aliases: &[],
        architecture: "{'channels': 1, 'topology': 'B6', 'type': 'ConvNet', 'weights_layout': 'Original'}",
        support: ModelSupportKind::Supported,
        description: "Supported {'channels': 1, 'topology': 'B6', 'type': 'ConvNet', 'weights_layout': 'Original'} model (convnet_test.nam)",
    },
    ModelCatalogEntry {
        sha256: "86c4ebf2be1cfdc80939423b3c0815ed2f18335c4f748b8cea935ef188a127f7",
        canonical_path: "tests/fixtures/models/linear_fft_rf2048.nam",
        aliases: &[],
        architecture: "{'channels': 1, 'topology': 'RF2048 (biased)', 'type': 'Linear', 'weights_layout': 'Original'}",
        support: ModelSupportKind::Supported,
        description: "Supported {'channels': 1, 'topology': 'RF2048 (biased)', 'type': 'Linear', 'weights_layout': 'Original'} model (linear_fft_rf2048.nam)",
    },
    ModelCatalogEntry {
        sha256: "feb40950b6ca02e9a3194ad4aed6f4dbfb823f9c81f64423c438695bfc0dc5f0",
        canonical_path: "tests/fixtures/models/linear_fft_rf320.nam",
        aliases: &[],
        architecture: "{'channels': 1, 'topology': 'RF320 (biased)', 'type': 'Linear', 'weights_layout': 'Original'}",
        support: ModelSupportKind::Supported,
        description: "Supported {'channels': 1, 'topology': 'RF320 (biased)', 'type': 'Linear', 'weights_layout': 'Original'} model (linear_fft_rf320.nam)",
    },
    ModelCatalogEntry {
        sha256: "2224517b2a3cd1cfddd8698d75f048cdec2d6adcf35bb58a39f54dad7af2cde6",
        canonical_path: "tests/fixtures/models/linear_fft_rf4096.nam",
        aliases: &[],
        architecture: "{'channels': 1, 'topology': 'RF4096 (biased)', 'type': 'Linear', 'weights_layout': 'Original'}",
        support: ModelSupportKind::Supported,
        description: "Supported {'channels': 1, 'topology': 'RF4096 (biased)', 'type': 'Linear', 'weights_layout': 'Original'} model (linear_fft_rf4096.nam)",
    },
    ModelCatalogEntry {
        sha256: "303af21a1f83dc24c0bdd0932d62716cf740cc73ee21ab34bba6e35e17aa6445",
        canonical_path: "tests/fixtures/models/linear_fft_rf8192.nam",
        aliases: &[],
        architecture: "{'channels': 1, 'topology': 'RF8192 (biased)', 'type': 'Linear', 'weights_layout': 'Original'}",
        support: ModelSupportKind::Supported,
        description: "Supported {'channels': 1, 'topology': 'RF8192 (biased)', 'type': 'Linear', 'weights_layout': 'Original'} model (linear_fft_rf8192.nam)",
    },
    ModelCatalogEntry {
        sha256: "95449b7337b9e5f74ae4ba09a59de1e6774a65fd3a9699b8039fcbb3524284bb",
        canonical_path: "tests/fixtures/models/linear_nobias.nam",
        aliases: &[],
        architecture: "{'channels': 1, 'topology': 'RF4', 'type': 'Linear', 'weights_layout': 'Original'}",
        support: ModelSupportKind::Supported,
        description: "Supported {'channels': 1, 'topology': 'RF4', 'type': 'Linear', 'weights_layout': 'Original'} model (linear_nobias.nam)",
    },
    ModelCatalogEntry {
        sha256: "f61b07a9e6f91793538f418bff0242d44e584c75e1c8a3f27a3b2fa873f21f4b",
        canonical_path: "tests/fixtures/models/linear_test.nam",
        aliases: &[],
        architecture: "{'channels': 1, 'topology': 'RF4 (biased)', 'type': 'Linear', 'weights_layout': 'Original'}",
        support: ModelSupportKind::Supported,
        description: "Supported {'channels': 1, 'topology': 'RF4 (biased)', 'type': 'Linear', 'weights_layout': 'Original'} model (linear_test.nam)",
    },
    ModelCatalogEntry {
        sha256: "945fa6780ebc38b3ae298d46de120e4ca46b2099be0c6b7ea2e9c7cd3f463cc6",
        canonical_path: "tests/fixtures/models/lstm_1x10.nam",
        aliases: &[],
        architecture: "{'channels': 10, 'topology': '1x10', 'type': 'LSTM', 'weights_layout': 'Original'}",
        support: ModelSupportKind::Supported,
        description: "Supported {'channels': 10, 'topology': '1x10', 'type': 'LSTM', 'weights_layout': 'Original'} model (lstm_1x10.nam)",
    },
    ModelCatalogEntry {
        sha256: "17ab500dc329362ea1baa1e8588622bfebd393eba8277fac502c1922c1e74a1d",
        canonical_path: "tests/fixtures/models/lstm_2x24.nam",
        aliases: &[],
        architecture: "{'channels': 24, 'topology': '2x24', 'type': 'LSTM', 'weights_layout': 'Original'}",
        support: ModelSupportKind::Supported,
        description: "Supported {'channels': 24, 'topology': '2x24', 'type': 'LSTM', 'weights_layout': 'Original'} model (lstm_2x24.nam)",
    },
    ModelCatalogEntry {
        sha256: "00afa60799616ddfa186ed34e35f32e45568ce6727beb4030cb06930ca82493f",
        canonical_path: "tests/fixtures/models/lstm_3x8.nam",
        aliases: &[],
        architecture: "{'channels': 8, 'topology': '3x8', 'type': 'LSTM', 'weights_layout': 'Original'}",
        support: ModelSupportKind::Supported,
        description: "Supported {'channels': 8, 'topology': '3x8', 'type': 'LSTM', 'weights_layout': 'Original'} model (lstm_3x8.nam)",
    },
    ModelCatalogEntry {
        sha256: "002291f4d3708ca5758d8c921f26a003198601012c71ccfcac0a1da48d57b655",
        canonical_path: "tests/fixtures/models/lstm_dyn_test.nam",
        aliases: &[],
        architecture: "{'channels': 7, 'topology': '1x7', 'type': 'LSTM', 'weights_layout': 'Original'}",
        support: ModelSupportKind::Supported,
        description: "Supported {'channels': 7, 'topology': '1x7', 'type': 'LSTM', 'weights_layout': 'Original'} model (lstm_dyn_test.nam)",
    },
    ModelCatalogEntry {
        sha256: "df9f78c49f49c2bb32411df47e3f53746075adb206b92d017e06379d1e56234a",
        canonical_path: "tests/fixtures/models/lstm.nam",
        aliases: &["third-party/NeuralAmpModelerCore/example_models/lstm.nam"],
        architecture: "{'channels': 3, 'topology': '1x3', 'type': 'LSTM', 'weights_layout': 'Original'}",
        support: ModelSupportKind::Supported,
        description: "Supported {'channels': 3, 'topology': '1x3', 'type': 'LSTM', 'weights_layout': 'Original'} model (lstm.nam)",
    },
    ModelCatalogEntry {
        sha256: "39910aa5988d2d6cba4fa26799e0184106d5a1ee056310174fce0c7d48aca5af",
        canonical_path: "tests/fixtures/models/mock_a2.nam",
        aliases: &[],
        architecture: "{'channels': None, 'topology': 'Custom', 'type': 'WaveNet', 'weights_layout': 'Original'}",
        support: ModelSupportKind::IntentionalNegative,
        description: "Intentional negative fixture (mock_a2.nam)",
    },
    ModelCatalogEntry {
        sha256: "78eba4fc17c39bba0fef375ee9cd3865d8ffefb76f53920037a874dcb2d2fbdc",
        canonical_path: "tests/fixtures/models/slimmable_container.nam",
        aliases: &["third-party/NeuralAmpModelerCore/example_models/slimmable_container.nam"],
        architecture: "{'channels': None, 'topology': 'Unknown', 'type': 'SlimmableContainer', 'weights_layout': 'Original'}",
        support: ModelSupportKind::Supported,
        description: "Slimmable Container 3 submodels (LSTM, WaveNetDyn, Nano) with ReLU — permanent regression fixture",
    },
    ModelCatalogEntry {
        sha256: "735c1a86e18140b7cfe90c08427ca6a85f62c32d34cc4048997933652aa774b4",
        canonical_path: "tests/fixtures/models/slimmable_wavenet.nam",
        aliases: &["third-party/NeuralAmpModelerCore/example_models/slimmable_wavenet.nam"],
        architecture: "{'channels': None, 'topology': 'A2-Dynamic', 'type': 'WaveNet', 'weights_layout': 'Original'}",
        support: ModelSupportKind::KnownGap,
        description: "Known architectural gap (slimmable_wavenet.nam)",
    },
    ModelCatalogEntry {
        sha256: "dde903b304fd55f705b8608760237bda0eddfbe287265dea4b6e1927272ffa1f",
        canonical_path: "tests/fixtures/models/wavenet_a1_secondary_act.nam",
        aliases: &[],
        architecture: "{'channels': None, 'topology': 'Custom', 'type': 'WaveNet', 'weights_layout': 'Original'}",
        support: ModelSupportKind::IntentionalNegative,
        description: "Intentional negative fixture (wavenet_a1_secondary_act.nam)",
    },
    ModelCatalogEntry {
        sha256: "ceb53469a19ce278e2235da982ae676cb8d5451a8de22a7ecc7a2617d07224d1",
        canonical_path: "tests/fixtures/models/wavenet_a1_standard.nam",
        aliases: &[
            "third-party/NeuralAmpModelerCore/example_models/my_model.nam",
            "third-party/NeuralAmpModelerCore/example_models/wavenet_a1_standard.nam",
        ],
        architecture: "{'channels': 16, 'topology': 'Standard', 'type': 'WaveNet', 'weights_layout': 'Original'}",
        support: ModelSupportKind::Supported,
        description: "Supported {'channels': 16, 'topology': 'Standard', 'type': 'WaveNet', 'weights_layout': 'Original'} model (wavenet_a1_standard.nam)",
    },
    ModelCatalogEntry {
        sha256: "d483d9a561e2dcb23cc6634a31ce3424d5e47131e9dcbde8aef350e0280d1777",
        canonical_path: "tests/fixtures/models/wavenet_a2_container.nam",
        aliases: &[],
        architecture: "{'channels': 8, 'topology': 'Unknown', 'type': 'SlimmableContainer', 'weights_layout': 'Original'}",
        support: ModelSupportKind::Supported,
        description: "Supported {'channels': 8, 'topology': 'Unknown', 'type': 'SlimmableContainer', 'weights_layout': 'Original'} model (wavenet_a2_container.nam)",
    },
    ModelCatalogEntry {
        sha256: "dade5331de3be210f653ce491c0f8ce32b9a0ae79cf2e2695f29b94fbeab2112",
        canonical_path: "tests/fixtures/models/wavenet_a2_film_chaos_stress.nam",
        aliases: &[],
        architecture: "{'channels': 3, 'topology': 'A2-Dynamic', 'type': 'WaveNet', 'weights_layout': 'Original'}",
        support: ModelSupportKind::Supported,
        description: "Supported {'channels': 3, 'topology': 'A2-Dynamic', 'type': 'WaveNet', 'weights_layout': 'Original'} model (wavenet_a2_film_chaos_stress.nam)",
    },
    ModelCatalogEntry {
        sha256: "c1e76c1930942965b6020fb76ef4363995f85df3f6947d142bd7e2801b8d218e",
        canonical_path: "tests/fixtures/models/wavenet_a2_film_full.nam",
        aliases: &[],
        architecture: "{'channels': 8, 'topology': 'A2-Dynamic', 'type': 'WaveNet', 'weights_layout': 'Original'}",
        support: ModelSupportKind::Supported,
        description: "Supported {'channels': 8, 'topology': 'A2-Dynamic', 'type': 'WaveNet', 'weights_layout': 'Original'} model (wavenet_a2_film_full.nam)",
    },
    ModelCatalogEntry {
        sha256: "2c8186e44f20a073a399b260fa94cad95f568a6040f8e734206509c4bc9c7c10",
        canonical_path: "tests/fixtures/models/wavenet_a2_film_input_mixin_pre.nam",
        aliases: &[],
        architecture: "{'channels': 3, 'topology': 'A2-Dynamic', 'type': 'WaveNet', 'weights_layout': 'Original'}",
        support: ModelSupportKind::Supported,
        description: "Supported {'channels': 3, 'topology': 'A2-Dynamic', 'type': 'WaveNet', 'weights_layout': 'Original'} model (wavenet_a2_film_input_mixin_pre.nam)",
    },
    ModelCatalogEntry {
        sha256: "a836899e2c10e902ef943ed133a2de6d5262ed1e739d2e152f19acea96b4c865",
        canonical_path: "tests/fixtures/models/wavenet_a2_film_lite.nam",
        aliases: &[],
        architecture: "{'channels': 3, 'topology': 'A2-Dynamic', 'type': 'WaveNet', 'weights_layout': 'Original'}",
        support: ModelSupportKind::Supported,
        description: "Supported {'channels': 3, 'topology': 'A2-Dynamic', 'type': 'WaveNet', 'weights_layout': 'Original'} model (wavenet_a2_film_lite.nam)",
    },
    ModelCatalogEntry {
        sha256: "533e86004ff04ad3a2fd2c840e02f5a1ae29d44faabcd008e3770b0d57e8f151",
        canonical_path: "tests/fixtures/models/wavenet_a2_full.nam",
        aliases: &[],
        architecture: "{'channels': 8, 'topology': 'A2-Full', 'type': 'WaveNet', 'weights_layout': 'Original'}",
        support: ModelSupportKind::Supported,
        description: "Supported {'channels': 8, 'topology': 'A2-Full', 'type': 'WaveNet', 'weights_layout': 'Original'} model (wavenet_a2_full.nam)",
    },
    ModelCatalogEntry {
        sha256: "f5359bba18c55a0259d6eee616206ac2c67fb0e02b884dcfbfa7ff0bacf56480",
        canonical_path: "tests/fixtures/models/wavenet_a2_lite.nam",
        aliases: &[],
        architecture: "{'channels': 3, 'topology': 'A2-Lite', 'type': 'WaveNet', 'weights_layout': 'Original'}",
        support: ModelSupportKind::Supported,
        description: "Supported {'channels': 3, 'topology': 'A2-Lite', 'type': 'WaveNet', 'weights_layout': 'Original'} model (wavenet_a2_lite.nam)",
    },
    ModelCatalogEntry {
        sha256: "12384c6640e1126907b366584024c4abb129ac5920b3dc2d31b29e39315e820d",
        canonical_path: "tests/fixtures/models/wavenet_a2_max.nam",
        aliases: &["third-party/NeuralAmpModelerCore/example_models/wavenet_a2_max.nam"],
        architecture: "{'channels': None, 'topology': 'A2-Dynamic', 'type': 'WaveNet', 'weights_layout': 'Original'}",
        support: ModelSupportKind::KnownGap,
        description: "Known architectural gap (wavenet_a2_max.nam)",
    },
    ModelCatalogEntry {
        sha256: "1af5a5d4eb079b894e095882738c102fd2d9eeced387a16d0d24cd73a07de718",
        canonical_path: "tests/fixtures/models/wavenet_condition_dsp.nam",
        aliases: &["third-party/NeuralAmpModelerCore/example_models/wavenet_condition_dsp.nam"],
        architecture: "{'channels': 3, 'topology': 'WaveNet-Dynamic', 'type': 'WaveNet', 'weights_layout': 'Original'}",
        support: ModelSupportKind::Supported,
        description: "Supported {'channels': 3, 'topology': 'WaveNet-Dynamic', 'type': 'WaveNet', 'weights_layout': 'Original'} model (wavenet_condition_dsp.nam)",
    },
    ModelCatalogEntry {
        sha256: "f2b394d5c30725bc426afee88ee182ff9d6770a45a6a16570f31666b2f27afc5",
        canonical_path: "tests/fixtures/models/wavenet_condition_lstm.nam",
        aliases: &[],
        architecture: "{'channels': None, 'topology': 'WaveNet-Dynamic', 'type': 'WaveNet', 'weights_layout': 'Original'}",
        support: ModelSupportKind::KnownGap,
        description: "Known architectural gap (wavenet_condition_lstm.nam)",
    },
    ModelCatalogEntry {
        sha256: "b78568bb0b042ef6aa09600b4b62153c0bcd7e3694f131aa85ef38083df18fe6",
        canonical_path: "tests/fixtures/models/wavenet_dyn_free.nam",
        aliases: &[],
        architecture: "{'channels': 7, 'topology': 'WaveNet-Dynamic', 'type': 'WaveNet', 'weights_layout': 'Original'}",
        support: ModelSupportKind::Supported,
        description: "Supported {'channels': 7, 'topology': 'WaveNet-Dynamic', 'type': 'WaveNet', 'weights_layout': 'Original'} model (wavenet_dyn_free.nam)",
    },
    ModelCatalogEntry {
        sha256: "66bda2b379289eff079c0755588bc9a92760654d9cc9af1b97cf30d0e92b167d",
        canonical_path: "tests/fixtures/models/wavenet.nam",
        aliases: &[
            "tests/fixtures/models/wavenet_official.nam",
            "third-party/NeuralAmpModelerCore/example_models/wavenet.nam",
        ],
        architecture: "{'channels': 3, 'topology': 'WaveNet-Dynamic', 'type': 'WaveNet', 'weights_layout': 'Original'}",
        support: ModelSupportKind::Supported,
        description: "Supported {'channels': 3, 'topology': 'WaveNet-Dynamic', 'type': 'WaveNet', 'weights_layout': 'Original'} model (wavenet.nam)",
    },
    ModelCatalogEntry {
        sha256: "7c0b6a15058c7a6d8d9025c7bd96a4563a637dbe4b2506fbfe003ed157b97616",
        canonical_path: "third-party/community_models/APP-EVH-Stealth100-Dialled-xSTD.nam",
        aliases: &[],
        architecture: "{'channels': 8, 'topology': 'WaveNet-Dynamic', 'type': 'WaveNet', 'weights_layout': 'Original'}",
        support: ModelSupportKind::Supported,
        description: "Supported {'channels': 8, 'topology': 'WaveNet-Dynamic', 'type': 'WaveNet', 'weights_layout': 'Original'} non-distributable community model",
    },
    ModelCatalogEntry {
        sha256: "cddd9e9e8fdeccb9f51b34f4c23f0f0d444c3faed307e962f76db7c6d287e1bf",
        canonical_path: "third-party/community_models/BOG UU II Gain BAL CAB.nam",
        aliases: &[],
        architecture: "{'channels': 8, 'topology': 'Unknown', 'type': 'SlimmableContainer', 'weights_layout': 'Original'}",
        support: ModelSupportKind::Supported,
        description: "Supported {'channels': 8, 'topology': 'Unknown', 'type': 'SlimmableContainer', 'weights_layout': 'Original'} non-distributable community model",
    },
    ModelCatalogEntry {
        sha256: "7ffd11c244664c737363d8d8753c3843f124b76a7afac7f15bcaebe0b6fd59f0",
        canonical_path: "third-party/community_models/Boss BD-2 H2O Mod T-12_00 G-12_00.nam",
        aliases: &[],
        architecture: "{'channels': 9, 'topology': 'WaveNet-Dynamic', 'type': 'WaveNet', 'weights_layout': 'Original'}",
        support: ModelSupportKind::Supported,
        description: "Supported {'channels': 9, 'topology': 'WaveNet-Dynamic', 'type': 'WaveNet', 'weights_layout': 'Original'} non-distributable community model",
    },
    ModelCatalogEntry {
        sha256: "66a4be684f6599c172d406af8f7206539fde8ffda9fd9eaefe5f3a09f388a0b8",
        canonical_path: "third-party/community_models/ChandlerRedd47-Gain34-Standard.nam",
        aliases: &[],
        architecture: "{'channels': 16, 'topology': 'Standard', 'type': 'WaveNet', 'weights_layout': 'Original'}",
        support: ModelSupportKind::Supported,
        description: "Supported {'channels': 16, 'topology': 'Standard', 'type': 'WaveNet', 'weights_layout': 'Original'} non-distributable community model",
    },
    ModelCatalogEntry {
        sha256: "4404e56fbe20a30d57735a3294c8023c22a27bc63d0f4994b3d889605c445ad6",
        canonical_path: "third-party/community_models/EVH-5150-Lite.nam",
        aliases: &[],
        architecture: "{'channels': 12, 'topology': 'Lite', 'type': 'WaveNet', 'weights_layout': 'Original'}",
        support: ModelSupportKind::Supported,
        description: "Supported {'channels': 12, 'topology': 'Lite', 'type': 'WaveNet', 'weights_layout': 'Original'} non-distributable community model",
    },
    ModelCatalogEntry {
        sha256: "d3f6c9e6f08cdd2a2f99ef910abdd5e37a6e1a83f79f47f40011aefcb7ec1f66",
        canonical_path: "third-party/community_models/little-bear-t7_phono-aux-tube-preamp_line-in_Standard.nam",
        aliases: &[],
        architecture: "{'channels': 16, 'topology': 'Standard', 'type': 'WaveNet', 'weights_layout': 'Original'}",
        support: ModelSupportKind::Supported,
        description: "Supported {'channels': 16, 'topology': 'Standard', 'type': 'WaveNet', 'weights_layout': 'Original'} non-distributable community model",
    },
    ModelCatalogEntry {
        sha256: "4257fc55a0cf105292613a7ed8864b933f809cc265694e8f14b5b522223697fe",
        canonical_path: "third-party/community_models/NEVE1073-Standard.nam",
        aliases: &[],
        architecture: "{'channels': 16, 'topology': 'Standard', 'type': 'WaveNet', 'weights_layout': 'Original'}",
        support: ModelSupportKind::Supported,
        description: "Supported {'channels': 16, 'topology': 'Standard', 'type': 'WaveNet', 'weights_layout': 'Original'} non-distributable community model",
    },
    ModelCatalogEntry {
        sha256: "c76c0666945213deb8f43b53690ef64ce72e5b8c562725d75a1b106372e81a61",
        canonical_path: "third-party/community_models/SLAMMIN_MARSHALL_J45_VN9_TREBLEBOOSTER_P4_C.nam",
        aliases: &[],
        architecture: "{'channels': 32, 'topology': 'WaveNet-Dynamic', 'type': 'WaveNet', 'weights_layout': 'Original'}",
        support: ModelSupportKind::Supported,
        description: "Supported {'channels': 32, 'topology': 'WaveNet-Dynamic', 'type': 'WaveNet', 'weights_layout': 'Original'} non-distributable community model",
    },
    ModelCatalogEntry {
        sha256: "203fac43316573ecb56e4e060d361d29669084259362b09b6e728e3f3d548a2d",
        canonical_path: "third-party/community_models/UA610B-Gain+10-Standard.nam",
        aliases: &[],
        architecture: "{'channels': 16, 'topology': 'Standard', 'type': 'WaveNet', 'weights_layout': 'Original'}",
        support: ModelSupportKind::Supported,
        description: "Supported {'channels': 16, 'topology': 'Standard', 'type': 'WaveNet', 'weights_layout': 'Original'} non-distributable community model",
    },
    ModelCatalogEntry {
        sha256: "8d5d62626945e079a34a09e91d6ddef575beaea3bc17efdb4dee4afed9be81c7",
        canonical_path: "third-party/NeuralAmpModelerPlugin/REAPER/model.nam",
        aliases: &[],
        architecture: "{'channels': 16, 'topology': 'Standard', 'type': 'WaveNet', 'weights_layout': 'Original'}",
        support: ModelSupportKind::Supported,
        description: "Supported {'channels': 16, 'topology': 'Standard', 'type': 'WaveNet', 'weights_layout': 'Original'} model (model.nam)",
    },
];

/// Computes the SHA-256 hex digest of a byte buffer.
pub fn compute_sha256_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let result = hasher.finalize();
    result.iter().map(|b| format!("{b:02x}")).collect()
}

/// Computes the SHA-256 hex digest of a file at the given path.
pub fn compute_sha256_file(path: &Path) -> io::Result<String> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 8192];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    let result = hasher.finalize();
    Ok(result.iter().map(|b| format!("{b:02x}")).collect())
}

/// Returns all entries in the static catalog.
pub fn catalog_entries() -> &'static [ModelCatalogEntry] {
    MODEL_CATALOG
}

/// Finds a catalog entry by its exact SHA-256 hash.
pub fn find_by_sha256(sha: &str) -> Option<&'static ModelCatalogEntry> {
    let target = sha.to_lowercase();
    MODEL_CATALOG.iter().find(|e| e.sha256 == target)
}

/// Finds a catalog entry by matching either canonical path or any alias path.
pub fn find_by_path(path: &Path) -> Option<&'static ModelCatalogEntry> {
    let p_str = path.to_string_lossy();
    MODEL_CATALOG
        .iter()
        .find(|e| e.canonical_path == p_str || e.aliases.iter().any(|alias| *alias == p_str))
}

/// Total number of unique SHA-256 identities in the catalog (51).
pub fn unique_sha_count() -> usize {
    MODEL_CATALOG.len()
}

/// Total number of mapped paths across canonical paths and aliases (61).
pub fn total_catalog_paths() -> usize {
    MODEL_CATALOG.iter().map(|e| 1 + e.aliases.len()).sum()
}

/// Total number of alias file paths in the catalog (10).
pub fn alias_count() -> usize {
    MODEL_CATALOG.iter().map(|e| e.aliases.len()).sum()
}

/// Total number of supported model identities in the catalog (45).
pub fn supported_count() -> usize {
    MODEL_CATALOG
        .iter()
        .filter(|e| e.support == ModelSupportKind::Supported)
        .count()
}

/// Total number of unsupported model identities in the catalog (6).
pub fn unsupported_count() -> usize {
    MODEL_CATALOG
        .iter()
        .filter(|e| e.support != ModelSupportKind::Supported)
        .count()
}

/// Total number of intentional negative fixtures (3).
pub fn intentional_negative_count() -> usize {
    MODEL_CATALOG
        .iter()
        .filter(|e| e.support == ModelSupportKind::IntentionalNegative)
        .count()
}

/// Total number of known architectural gaps (3).
pub fn known_gap_count() -> usize {
    MODEL_CATALOG
        .iter()
        .filter(|e| e.support == ModelSupportKind::KnownGap)
        .count()
}

// =============================================================================
// V2 Golden Catalog — single source of truth
// =============================================================================
//
// This registry defines the canonical V2 multi-SR golden matrix: which models
// participate, which sample rates each model requires, which fixture files are
// expected on disk, and the distribution policy for each file. It is the ONLY
// definition — `utils/tests-long.sh` no longer carries bash arrays and
// `tests/fixtures/golden_gen_build.sh` sources its catalog from here (via the
// `nam_golden_catalog` binary), so model lists never appear twice in shell.
//
// The v2_scope column semantics preserved from the former generator CATALOG:
//   all       — v2 golden at all 5 sample rates (44100, 48000, 88200, 96000, 192000)
//   all:192000 — all rates except 192 kHz (LSTM recurrent drift over the 5s stress)
//   48k_only  — only 48000 Hz (model declares expected_sample_rate=48000, or the
//                C++ render tool rejects other SRs)
//   none      — no v2 golden generation for this model
//
// Models with `in_v2_catalog == true` (24) are the validated V2 subset checked
// by `validate_v2_catalog`/`catalog_preflight`. The remaining generator
// registry entries (15) are v1-only models (v2_scope=none) or v1 models that
// carry an incidental unrequired v2@48k golden (KB-A2-MAX + dyn/convnet v1
// fixtures): they keep generating but are never preflight gates.

/// Sample-rate scope of a golden catalog entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum V2GenScope {
    /// No v2 golden generation for this model (v1-only).
    NoV2,
    /// All 5 sample rates (44100, 48000, 88200, 96000, 192000).
    AllRates,
    /// All rates except 192 kHz (LSTM recurrent drift over the 5s stress).
    Exclude192k,
    /// Only 48000 Hz (model declares `expected_sample_rate=48000`).
    Sr48kOnly,
}

/// Distribution policy for a V2 catalog file (model fixture or golden binary).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum V2Distribution {
    /// Must exist on disk — preflight hard-fail when absent.
    RequiredLocal,
    /// Non-distributable community model — skip gracefully when absent.
    OptionalExternal,
}

/// Single entry of the golden generation registry.
///
/// Fields map 1:1 to the former colon-format entries of
/// `tests/fixtures/golden_gen_build.sh`'s bash CATALOG (which no longer
/// exists): `nam_file:golden_name:label:v2_scope[:skip_srs[:skip_reason]]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GoldenGenEntry {
    /// Model fixture basename (resolved via `fixtures::model_path`).
    pub nam_file: &'static str,
    /// Golden name prefix used to derive `golden_{name}[_v2_{sr}].bin`.
    pub golden_name: &'static str,
    /// Human-readable model label (generator echo output).
    pub label: &'static str,
    /// Sample-rate scope for v2 golden generation.
    pub v2_scope: V2GenScope,
    /// Whether this entry is part of the validated V2 catalog (preflight gate).
    pub in_v2_catalog: bool,
    /// Distribution policy of the model fixture file itself.
    pub model_distribution: V2Distribution,
    /// Distribution policy of the v2 golden binary files.
    pub golden_distribution: V2Distribution,
    /// Non-empty when the model is intentionally skipped in generation loops
    /// (upstream limitation / rejection fixture). Must carry a `(YYYY-MM-DD)`
    /// review date — enforced by `threshold_calibration::test_catalog_anti_placebo_audit`.
    pub skip_reason: Option<&'static str>,
}

/// All 5 sample rates of the v2 multi-SR matrix.
pub const V2_ALL_SAMPLE_RATES: &[u32] = &[44100, 48000, 88200, 96000, 192000];

/// The 4 sample rates of the `Exclude192k` scope.
pub const V2_EX_192K_SAMPLE_RATES: &[u32] = &[44100, 48000, 88200, 96000];

/// The single 48 kHz sample rate of the `Sr48kOnly` scope.
pub const V2_48K_SAMPLE_RATES: &[u32] = &[48000];

/// Canonical golden generation registry — 39 entries (24 validated V2 + 15 v1-only).
pub static GOLDEN_GEN_CATALOG: &[GoldenGenEntry] = &[
    GoldenGenEntry {
        nam_file: "BossWN-standard.nam",
        golden_name: "golden_wavenet_standard",
        label: "WaveNet Standard (CH=16)",
        v2_scope: V2GenScope::Sr48kOnly,
        in_v2_catalog: true,
        model_distribution: V2Distribution::RequiredLocal,
        golden_distribution: V2Distribution::RequiredLocal,
        skip_reason: None,
    },
    GoldenGenEntry {
        nam_file: "EVH-5150-Lite.nam",
        golden_name: "golden_wavenet_lite",
        label: "WaveNet Lite (CH=12)",
        v2_scope: V2GenScope::AllRates,
        in_v2_catalog: true,
        model_distribution: V2Distribution::OptionalExternal,
        golden_distribution: V2Distribution::OptionalExternal,
        skip_reason: None,
    },
    GoldenGenEntry {
        nam_file: "BossWN-feather.nam",
        golden_name: "golden_wavenet_feather",
        label: "WaveNet Feather (CH=8)",
        v2_scope: V2GenScope::AllRates,
        in_v2_catalog: true,
        model_distribution: V2Distribution::RequiredLocal,
        golden_distribution: V2Distribution::RequiredLocal,
        skip_reason: None,
    },
    GoldenGenEntry {
        nam_file: "BossWN-nano.nam",
        golden_name: "golden_wavenet_nano",
        label: "WaveNet Nano (CH=4)",
        v2_scope: V2GenScope::AllRates,
        in_v2_catalog: true,
        model_distribution: V2Distribution::RequiredLocal,
        golden_distribution: V2Distribution::RequiredLocal,
        skip_reason: None,
    },
    GoldenGenEntry {
        nam_file: "wavenet_a1_standard.nam",
        golden_name: "golden_wavenet_a1_standard",
        label: "WaveNet A1 Standard (Official)",
        v2_scope: V2GenScope::AllRates,
        in_v2_catalog: true,
        model_distribution: V2Distribution::RequiredLocal,
        golden_distribution: V2Distribution::RequiredLocal,
        skip_reason: None,
    },
    GoldenGenEntry {
        nam_file: "wavenet_official.nam",
        golden_name: "golden_wavenet_official",
        label: "WaveNet Official (CH=3 free geom)",
        v2_scope: V2GenScope::Sr48kOnly,
        in_v2_catalog: true,
        model_distribution: V2Distribution::RequiredLocal,
        golden_distribution: V2Distribution::RequiredLocal,
        skip_reason: None,
    },
    GoldenGenEntry {
        nam_file: "BossLSTM-1x16.nam",
        golden_name: "golden_lstm_1x16",
        label: "LSTM 1×16",
        v2_scope: V2GenScope::Exclude192k,
        in_v2_catalog: true,
        model_distribution: V2Distribution::RequiredLocal,
        golden_distribution: V2Distribution::RequiredLocal,
        skip_reason: None,
    },
    GoldenGenEntry {
        nam_file: "BossLSTM-2x8.nam",
        golden_name: "golden_lstm_2x8",
        label: "LSTM 2×8",
        v2_scope: V2GenScope::Exclude192k,
        in_v2_catalog: true,
        model_distribution: V2Distribution::RequiredLocal,
        golden_distribution: V2Distribution::RequiredLocal,
        skip_reason: None,
    },
    GoldenGenEntry {
        nam_file: "lstm.nam",
        golden_name: "golden_lstm_official",
        label: "LSTM Official",
        v2_scope: V2GenScope::Sr48kOnly,
        in_v2_catalog: true,
        model_distribution: V2Distribution::RequiredLocal,
        golden_distribution: V2Distribution::RequiredLocal,
        skip_reason: None,
    },
    GoldenGenEntry {
        nam_file: "wavenet_a2_full.nam",
        golden_name: "golden_wavenet_a2_full",
        label: "A2-Full (CH=8)",
        v2_scope: V2GenScope::Sr48kOnly,
        in_v2_catalog: true,
        model_distribution: V2Distribution::RequiredLocal,
        golden_distribution: V2Distribution::RequiredLocal,
        skip_reason: None,
    },
    GoldenGenEntry {
        nam_file: "wavenet_a2_lite.nam",
        golden_name: "golden_wavenet_a2_lite",
        label: "A2-Lite (CH=3)",
        v2_scope: V2GenScope::Sr48kOnly,
        in_v2_catalog: true,
        model_distribution: V2Distribution::RequiredLocal,
        golden_distribution: V2Distribution::RequiredLocal,
        skip_reason: None,
    },
    GoldenGenEntry {
        nam_file: "wavenet_condition_dsp.nam",
        golden_name: "golden_wavenet_condition_dsp",
        label: "Condition DSP (CH=3, cond=3)",
        v2_scope: V2GenScope::Sr48kOnly,
        in_v2_catalog: true,
        model_distribution: V2Distribution::RequiredLocal,
        golden_distribution: V2Distribution::RequiredLocal,
        skip_reason: None,
    },
    GoldenGenEntry {
        nam_file: "wavenet_condition_lstm.nam",
        golden_name: "golden_wavenet_condition_lstm",
        label: "Condition DSP LSTM (CH=3, cond=3, LSTM)",
        v2_scope: V2GenScope::Sr48kOnly,
        in_v2_catalog: true,
        model_distribution: V2Distribution::RequiredLocal,
        golden_distribution: V2Distribution::RequiredLocal,
        skip_reason: Some(
            "C++ upstream limitation: LSTM condition_dsp sub-model channel mismatch \
             (uses input_size=1 instead of hidden_size=3) — golden binary cannot be \
             generated (2026-07-11)",
        ),
    },
    GoldenGenEntry {
        nam_file: "a2_example.nam",
        golden_name: "golden_a2_example",
        label: "SlimmableContainer A2 Example (CH=3→6)",
        v2_scope: V2GenScope::Sr48kOnly,
        in_v2_catalog: true,
        model_distribution: V2Distribution::RequiredLocal,
        golden_distribution: V2Distribution::RequiredLocal,
        skip_reason: None,
    },
    GoldenGenEntry {
        nam_file: "APP-EVH-Stealth100-Dialled-xSTD.nam",
        golden_name: "golden_wavenet_app_evh",
        label: "APP EVH Stealth 100",
        v2_scope: V2GenScope::Sr48kOnly,
        in_v2_catalog: true,
        model_distribution: V2Distribution::OptionalExternal,
        golden_distribution: V2Distribution::RequiredLocal,
        skip_reason: None,
    },
    GoldenGenEntry {
        nam_file: "Boss BD-2 H2O Mod T-12_00 G-12_00.nam",
        golden_name: "golden_wavenet_boss_bd2",
        label: "Boss BD-2 H2O Mod",
        v2_scope: V2GenScope::Sr48kOnly,
        in_v2_catalog: true,
        model_distribution: V2Distribution::OptionalExternal,
        golden_distribution: V2Distribution::RequiredLocal,
        skip_reason: None,
    },
    GoldenGenEntry {
        nam_file: "SLAMMIN_MARSHALL_J45_VN9_TREBLEBOOSTER_P4_C.nam",
        golden_name: "golden_wavenet_slammin_marshall",
        label: "SLAMMIN MARSHALL J45",
        v2_scope: V2GenScope::Sr48kOnly,
        in_v2_catalog: true,
        model_distribution: V2Distribution::OptionalExternal,
        golden_distribution: V2Distribution::RequiredLocal,
        skip_reason: None,
    },
    // ── v1-only models (in_v2_catalog = false) ────────────────────────────
    // wavenet_dyn_free / lstm_dyn_test / convnet_test carry an incidental
    // unrequired v2@48k golden; wavenet_a2_max is excluded by design
    // (KB-A2-MAX, docs/cpp_parity_map.md §4.4.3 — never a preflight gate).
    GoldenGenEntry {
        nam_file: "wavenet_dyn_free.nam",
        golden_name: "golden_wavenet_dyn_free",
        label: "WaveNetDyn Free-Shape (CH=7/4)",
        v2_scope: V2GenScope::Sr48kOnly,
        in_v2_catalog: false,
        model_distribution: V2Distribution::RequiredLocal,
        golden_distribution: V2Distribution::RequiredLocal,
        skip_reason: None,
    },
    GoldenGenEntry {
        nam_file: "lstm_dyn_test.nam",
        golden_name: "golden_lstm_dyn_test",
        label: "LSTM-Dyn 1×7",
        v2_scope: V2GenScope::Sr48kOnly,
        in_v2_catalog: false,
        model_distribution: V2Distribution::RequiredLocal,
        golden_distribution: V2Distribution::RequiredLocal,
        skip_reason: None,
    },
    GoldenGenEntry {
        nam_file: "convnet_test.nam",
        golden_name: "golden_convnet_test",
        label: "ConvNet Test (CH=8, 6 blocks)",
        v2_scope: V2GenScope::Sr48kOnly,
        in_v2_catalog: false,
        model_distribution: V2Distribution::RequiredLocal,
        golden_distribution: V2Distribution::RequiredLocal,
        skip_reason: None,
    },
    GoldenGenEntry {
        nam_file: "wavenet_a2_max.nam",
        golden_name: "golden_wavenet_a2_max",
        label: "WaveNet A2 Max (CH=4, cond=8, FiLM, head1x1)",
        v2_scope: V2GenScope::Sr48kOnly,
        in_v2_catalog: false,
        model_distribution: V2Distribution::RequiredLocal,
        golden_distribution: V2Distribution::RequiredLocal,
        skip_reason: None,
    },
    // Dynamic/FiLM models: v2_scope=none — C++ a2_fast render path rejects
    // FiLM-conditioned models and the generic Eigen engine does not support
    // multi-SR FiLM rendering; dynamic-engine coverage is a superset at test
    // time (see rationale comment in tests/fixtures/golden_gen_build.sh §7).
    GoldenGenEntry {
        nam_file: "a2_dynamic_gated_ch8.nam",
        golden_name: "golden_a2_dynamic_gated_ch8",
        label: "A2 Dynamic Gated (CH=8)",
        v2_scope: V2GenScope::NoV2,
        in_v2_catalog: false,
        model_distribution: V2Distribution::RequiredLocal,
        golden_distribution: V2Distribution::RequiredLocal,
        skip_reason: None,
    },
    GoldenGenEntry {
        nam_file: "a2_dynamic_blended_ch3.nam",
        golden_name: "golden_a2_dynamic_blended_ch3",
        label: "A2 Dynamic Blended (CH=3)",
        v2_scope: V2GenScope::NoV2,
        in_v2_catalog: false,
        model_distribution: V2Distribution::RequiredLocal,
        golden_distribution: V2Distribution::RequiredLocal,
        skip_reason: None,
    },
    GoldenGenEntry {
        nam_file: "wavenet_a2_film_lite.nam",
        golden_name: "golden_wavenet_a2_film_lite",
        label: "A2-FiLM Lite (CH=3)",
        v2_scope: V2GenScope::NoV2,
        in_v2_catalog: false,
        model_distribution: V2Distribution::RequiredLocal,
        golden_distribution: V2Distribution::RequiredLocal,
        skip_reason: None,
    },
    GoldenGenEntry {
        nam_file: "wavenet_a2_film_full.nam",
        golden_name: "golden_wavenet_a2_film_full",
        label: "A2-FiLM Full (CH=8)",
        v2_scope: V2GenScope::NoV2,
        in_v2_catalog: false,
        model_distribution: V2Distribution::RequiredLocal,
        golden_distribution: V2Distribution::RequiredLocal,
        skip_reason: None,
    },
    GoldenGenEntry {
        nam_file: "wavenet_a2_film_chaos_stress.nam",
        golden_name: "golden_wavenet_a2_film_chaos_stress",
        label: "A2-FiLM Chaos Stress (CH=3)",
        v2_scope: V2GenScope::NoV2,
        in_v2_catalog: false,
        model_distribution: V2Distribution::RequiredLocal,
        golden_distribution: V2Distribution::RequiredLocal,
        skip_reason: None,
    },
    GoldenGenEntry {
        nam_file: "wavenet_a2_film_input_mixin_pre.nam",
        golden_name: "golden_wavenet_a2_film_input_mixin_pre",
        label: "A2-FiLM InputMixinPre (CH=3)",
        v2_scope: V2GenScope::NoV2,
        in_v2_catalog: false,
        model_distribution: V2Distribution::RequiredLocal,
        golden_distribution: V2Distribution::RequiredLocal,
        skip_reason: None,
    },
    GoldenGenEntry {
        nam_file: "linear_fft_rf320.nam",
        golden_name: "golden_linear_fft_rf320",
        label: "Linear FFT RF=320",
        v2_scope: V2GenScope::NoV2,
        in_v2_catalog: false,
        model_distribution: V2Distribution::RequiredLocal,
        golden_distribution: V2Distribution::RequiredLocal,
        skip_reason: None,
    },
    GoldenGenEntry {
        nam_file: "linear_fft_rf2048.nam",
        golden_name: "golden_linear_fft_rf2048",
        label: "Linear FFT RF=2048",
        v2_scope: V2GenScope::NoV2,
        in_v2_catalog: false,
        model_distribution: V2Distribution::RequiredLocal,
        golden_distribution: V2Distribution::RequiredLocal,
        skip_reason: None,
    },
    GoldenGenEntry {
        nam_file: "linear_fft_rf4096.nam",
        golden_name: "golden_linear_fft_rf4096",
        label: "Linear FFT RF=4096",
        v2_scope: V2GenScope::NoV2,
        in_v2_catalog: false,
        model_distribution: V2Distribution::RequiredLocal,
        golden_distribution: V2Distribution::RequiredLocal,
        skip_reason: None,
    },
    GoldenGenEntry {
        nam_file: "linear_fft_rf8192.nam",
        golden_name: "golden_linear_fft_rf8192",
        label: "Linear FFT RF=8192",
        v2_scope: V2GenScope::NoV2,
        in_v2_catalog: false,
        model_distribution: V2Distribution::RequiredLocal,
        golden_distribution: V2Distribution::RequiredLocal,
        skip_reason: None,
    },
    // LSTM uncatalogued hidden sizes and 3-layer topology
    GoldenGenEntry {
        nam_file: "lstm_1x10.nam",
        golden_name: "golden_lstm_1x10",
        label: "LSTM 1×10 (uncat.)",
        v2_scope: V2GenScope::Sr48kOnly,
        in_v2_catalog: true,
        model_distribution: V2Distribution::RequiredLocal,
        golden_distribution: V2Distribution::RequiredLocal,
        skip_reason: None,
    },
    GoldenGenEntry {
        nam_file: "lstm_2x24.nam",
        golden_name: "golden_lstm_2x24",
        label: "LSTM 2×24 (uncat.)",
        v2_scope: V2GenScope::Sr48kOnly,
        in_v2_catalog: true,
        model_distribution: V2Distribution::RequiredLocal,
        golden_distribution: V2Distribution::RequiredLocal,
        skip_reason: None,
    },
    GoldenGenEntry {
        nam_file: "lstm_3x8.nam",
        golden_name: "golden_lstm_3x8",
        label: "LSTM 3×8",
        v2_scope: V2GenScope::Sr48kOnly,
        in_v2_catalog: true,
        model_distribution: V2Distribution::RequiredLocal,
        golden_distribution: V2Distribution::RequiredLocal,
        skip_reason: None,
    },
    // ConvNet variants (nobn, ReLU, SiLU)
    GoldenGenEntry {
        nam_file: "convnet_nobn.nam",
        golden_name: "golden_convnet_nobn",
        label: "ConvNet No BatchNorm",
        v2_scope: V2GenScope::Sr48kOnly,
        in_v2_catalog: true,
        model_distribution: V2Distribution::RequiredLocal,
        golden_distribution: V2Distribution::RequiredLocal,
        skip_reason: None,
    },
    GoldenGenEntry {
        nam_file: "convnet_relu.nam",
        golden_name: "golden_convnet_relu",
        label: "ConvNet ReLU",
        v2_scope: V2GenScope::Sr48kOnly,
        in_v2_catalog: true,
        model_distribution: V2Distribution::RequiredLocal,
        golden_distribution: V2Distribution::RequiredLocal,
        skip_reason: None,
    },
    GoldenGenEntry {
        nam_file: "convnet_silu.nam",
        golden_name: "golden_convnet_silu",
        label: "ConvNet SiLU",
        v2_scope: V2GenScope::Sr48kOnly,
        in_v2_catalog: true,
        model_distribution: V2Distribution::RequiredLocal,
        golden_distribution: V2Distribution::RequiredLocal,
        skip_reason: None,
    },
    // Linear without bias
    GoldenGenEntry {
        nam_file: "linear_nobias.nam",
        golden_name: "golden_linear_nobias",
        label: "Linear No Bias",
        v2_scope: V2GenScope::Sr48kOnly,
        in_v2_catalog: true,
        model_distribution: V2Distribution::RequiredLocal,
        golden_distribution: V2Distribution::RequiredLocal,
        skip_reason: None,
    },
    // Rejection test fixture — model parses, golden generation intentionally skipped
    GoldenGenEntry {
        nam_file: "wavenet_a1_secondary_act.nam",
        golden_name: "golden_wavenet_a1_secondary_act",
        label: "WaveNet A1 Secondary Activation Rejection",
        v2_scope: V2GenScope::NoV2,
        in_v2_catalog: false,
        model_distribution: V2Distribution::RequiredLocal,
        golden_distribution: V2Distribution::RequiredLocal,
        skip_reason: Some("Rejection fixture for non-null secondary_activation (2026-08-03)"),
    },
];

/// Returns the full golden generation registry (39 entries).
pub fn golden_gen_entries() -> &'static [GoldenGenEntry] {
    GOLDEN_GEN_CATALOG
}

/// Returns the validated V2 catalog subset (24 entries).
pub fn v2_catalog_entries() -> Vec<&'static GoldenGenEntry> {
    GOLDEN_GEN_CATALOG
        .iter()
        .filter(|e| e.in_v2_catalog)
        .collect()
}

/// Looks up the sample-rate slice for a model from the canonical V2 catalog.
///
/// Falls back to 48 kHz only with a warning if the model is not registered.
pub fn v2_sample_rates_for(nam_file: &str) -> &'static [u32] {
    for entry in GOLDEN_GEN_CATALOG {
        if entry.in_v2_catalog && entry.nam_file == nam_file {
            return match entry.v2_scope {
                V2GenScope::AllRates => V2_ALL_SAMPLE_RATES,
                V2GenScope::Exclude192k => V2_EX_192K_SAMPLE_RATES,
                _ => V2_48K_SAMPLE_RATES,
            };
        }
    }
    eprintln!(
        "WARNING: {nam_file} not in V2 golden catalog — defaulting to 48 kHz only. \
         Add the model to GOLDEN_GEN_CATALOG in src/testing/catalog.rs."
    );
    V2_48K_SAMPLE_RATES
}

/// Serializes the golden generation registry in the shell catalog line format.
///
/// One line per entry, colon-separated:
/// `nam_file:golden_name:label:v2_scope[:skip_srs[:skip_reason]]`.
/// Consumed by `tests/fixtures/golden_gen_build.sh` (via the
/// `nam_golden_catalog` binary), replacing the former static bash CATALOG array.
pub fn golden_gen_catalog_lines() -> String {
    let mut out = String::new();
    for entry in GOLDEN_GEN_CATALOG {
        out.push_str(entry.nam_file);
        out.push(':');
        out.push_str(entry.golden_name);
        out.push(':');
        out.push_str(entry.label);
        out.push(':');
        match entry.v2_scope {
            V2GenScope::NoV2 => out.push_str("none"),
            V2GenScope::AllRates => out.push_str("all"),
            V2GenScope::Exclude192k => out.push_str("all:192000"),
            V2GenScope::Sr48kOnly => out.push_str("48k_only"),
        }
        if let Some(reason) = entry.skip_reason {
            out.push_str("::");
            out.push_str(reason);
        }
        out.push('\n');
    }
    out
}

/// Outcome of a `validate_v2_catalog` run: disk presence of every expected
/// V2 model fixture and golden binary.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CatalogStatus {
    /// Number of catalog entries validated.
    pub entries_checked: usize,
    /// Number of fixture files checked (models + goldens).
    pub fixtures_checked: usize,
    /// Number of fixture files found on disk.
    pub present: usize,
    /// Required fixture paths absent from disk (hard preflight failure).
    pub missing_required: Vec<String>,
    /// Optional (non-distributable) fixture paths absent from disk.
    pub missing_optional: Vec<String>,
    /// Golden names with a documented skip reason — absence is expected.
    pub known_gaps: Vec<&'static str>,
}

impl CatalogStatus {
    /// True when no required fixture is missing.
    pub fn is_ok(&self) -> bool {
        self.missing_required.is_empty()
    }

    /// Emits a typed capability receipt for the V2 golden catalog.
    ///
    /// Uses the `MISSING-REQUIRED:` / `MISSING-OPTIONAL:` markers also emitted
    /// by the fixture catalog receipt, so `utils/tests-long.sh` can count
    /// failures from the log with a single grep.
    pub fn receipt(&self) -> String {
        self.receipt_for(
            "V2",
            "V2 Golden Catalog Capability Receipt",
            v2_catalog_entries().len(),
        )
    }

    /// Emits a typed capability receipt for the V1 golden catalog.
    ///
    /// Same marker contract as [`CatalogStatus::receipt`]: the shell preflight
    /// counts `MISSING-REQUIRED:` lines from the `catalog_preflight` log, so
    /// both receipts must keep that marker verbatim.
    pub fn receipt_v1(&self) -> String {
        self.receipt_for(
            "V1",
            "V1 Golden Catalog Capability Receipt",
            V1_GOLDEN_CATALOG.len(),
        )
    }

    fn receipt_for(&self, version: &str, title: &str, total_entries: usize) -> String {
        let mut lines = Vec::new();
        lines.push(format!("=== {title} ==="));
        lines.push(format!(
            "Validated {}/{} catalog entries, {} fixture files checked, {} present.",
            self.entries_checked, total_entries, self.fixtures_checked, self.present
        ));
        lines.push("-".repeat(100));
        if !self.known_gaps.is_empty() {
            lines.push(format!(
                "  KNOWN-GAP (expected absence, {}): {}",
                self.known_gaps.len(),
                self.known_gaps.join(", ")
            ));
        }
        if !self.missing_optional.is_empty() {
            lines.push(format!(
                "=== Optional (non-distributable) {version} fixtures absent ({} file(s)) ===",
                self.missing_optional.len()
            ));
            for path in &self.missing_optional {
                lines.push(format!(
                    "  MISSING-OPTIONAL: {path} ({version} — OptionalExternal, skip gracefully)"
                ));
            }
        }
        if !self.missing_required.is_empty() {
            lines.push(format!(
                "=== Required {version} fixtures absent ({} file(s)) ===",
                self.missing_required.len()
            ));
            for path in &self.missing_required {
                lines.push(format!(
                    "  MISSING-REQUIRED: {path} ({version} — RequiredLocal, preflight hard-fail)"
                ));
            }
        } else {
            lines.push(format!(
                "=== All {version} RequiredLocal fixtures present ✓ ==="
            ));
        }
        lines.push("-".repeat(100));
        lines.push(
            "Run './tests/fixtures/golden_gen_build.sh' to regenerate missing V2 \
             golden vectors (C++ toolchain + NeuralAmpModelerCore required)."
                .to_string(),
        );
        lines.join("\n")
    }
}

/// Error surfaced when the V2 catalog cannot be validated.
#[derive(Debug)]
pub enum CatalogError {
    /// The fixtures root directory is not available at the resolved path.
    FixturesRootUnavailable(std::path::PathBuf),
    /// Internal catalog invariant violated (e.g. empty registry).
    CatalogInvariant(&'static str),
}

impl std::fmt::Display for CatalogError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CatalogError::FixturesRootUnavailable(path) => write!(
                f,
                "V2 catalog validation aborted: fixtures root not found at {path:?}. \
                 Run the suite from the crate root or restore tests/fixtures/."
            ),
            CatalogError::CatalogInvariant(msg) => {
                write!(f, "V2 catalog validation aborted: {msg}")
            }
        }
    }
}

impl std::error::Error for CatalogError {}

/// Validates the canonical V2 golden catalog against disk.
///
/// For every entry of the validated V2 subset (24):
/// - the model fixture must resolve via `fixtures::model_path` (required or
///   optional per `model_distribution`);
/// - every golden binary expected by the entry's sample-rate scope must exist
///   in `tests/fixtures/` (required or optional per `golden_distribution`);
/// - entries with a documented `skip_reason` (KnownGap) are recorded as
///   expected-absent and never fail the gate.
///
/// Returns `Ok(CatalogStatus)` describing disk integrity, or `Err` when the
/// environment is not a valid crate checkout. Consumable by integration tests
/// (`tests/models/golden_vectors.rs::catalog_preflight`) and binaries.
pub fn validate_v2_catalog() -> Result<CatalogStatus, CatalogError> {
    let entries = v2_catalog_entries();
    if entries.is_empty() {
        return Err(CatalogError::CatalogInvariant(
            "GOLDEN_GEN_CATALOG defines no V2 entries (in_v2_catalog is all false)",
        ));
    }

    let fixtures_root = crate::testing::fixtures::fixture_dir();
    if !fixtures_root.is_dir() {
        return Err(CatalogError::FixturesRootUnavailable(fixtures_root));
    }

    let mut status = CatalogStatus::default();

    for entry in entries {
        status.entries_checked += 1;

        let model_ok = crate::testing::fixtures::model_path(entry.nam_file).exists();
        if !model_ok {
            status.fixtures_checked += 1;
            let path = format!("tests/fixtures/models/{}", entry.nam_file);
            match entry.model_distribution {
                V2Distribution::RequiredLocal => status.missing_required.push(path),
                V2Distribution::OptionalExternal => status.missing_optional.push(path),
            }
        } else {
            status.present += 1;
        }

        if entry.skip_reason.is_some() {
            status.known_gaps.push(entry.golden_name);
            continue;
        }

        for &sr in v2_sample_rates_for(entry.nam_file) {
            status.fixtures_checked += 1;
            let file = format!("{}_v2_{}.bin", entry.golden_name, sr);
            let path = fixtures_root.join(&file);
            if path.exists() {
                status.present += 1;
            } else {
                let rel = format!("tests/fixtures/{file}");
                match entry.golden_distribution {
                    V2Distribution::RequiredLocal => status.missing_required.push(rel),
                    V2Distribution::OptionalExternal => status.missing_optional.push(rel),
                }
            }
        }
    }

    Ok(status)
}

// =============================================================================
// V1 Golden Catalog — single source of truth
// =============================================================================
//
// The former bash lists in `utils/tests-long.sh` (REQUIRED_GOLDEN_MODELS,
// NONDIST_GOLDEN_MODELS and REQUIRED_CABSIM_GOLDENS) are hereby eliminated:
// this registry is the ONLY definition of the v1 golden matrix (48 kHz `.bin`
// vectors) and its distribution policy. Disk presence is validated
// fail-closed by `validate_v1_goldens` / `catalog_preflight` before any timed
// long-suite phase, and by `check_freshness` (nam_freshness) against the
// `.golden_manifest.sha256` integrity manifest.

/// Single v1 golden registry entry: an expected `.bin` file under
/// `tests/fixtures/` and its distribution policy.
///
/// Distribution taxonomy mirrors the V2 registry:
/// `DistributedCore` → `RequiredLocal` (hard preflight failure when absent),
/// `LocalNonDistributable` → `OptionalExternal` (graceful typed skip).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct V1GoldenEntry {
    /// Golden binary basename under `tests/fixtures/`.
    pub golden_file: &'static str,
    /// Distribution policy (RequiredLocal hard-fails preflight when absent).
    pub distribution: V2Distribution,
    /// Human-readable description (source model family / convolution scope).
    pub description: &'static str,
}

/// Canonical v1 golden registry — 13 entries (9 DistributedCore model goldens,
/// 1 LocalNonDistributable model golden, 3 CabSim convolution goldens).
pub static V1_GOLDEN_CATALOG: &[V1GoldenEntry] = &[
    // ── DistributedCore model goldens (v1, 48 kHz) — RequiredLocal ────────
    V1GoldenEntry {
        golden_file: "golden_wavenet_standard.bin",
        distribution: V2Distribution::RequiredLocal,
        description: "WaveNet Standard (CH=16) v1 golden",
    },
    V1GoldenEntry {
        golden_file: "golden_wavenet_feather.bin",
        distribution: V2Distribution::RequiredLocal,
        description: "WaveNet Feather (CH=8) v1 golden",
    },
    V1GoldenEntry {
        golden_file: "golden_wavenet_nano.bin",
        distribution: V2Distribution::RequiredLocal,
        description: "WaveNet Nano (CH=4) v1 golden",
    },
    V1GoldenEntry {
        golden_file: "golden_wavenet_a1_standard.bin",
        distribution: V2Distribution::RequiredLocal,
        description: "WaveNet A1 Standard v1 golden",
    },
    V1GoldenEntry {
        golden_file: "golden_wavenet_a2_full.bin",
        distribution: V2Distribution::RequiredLocal,
        description: "WaveNet A2 Full (CH=8) v1 golden",
    },
    V1GoldenEntry {
        golden_file: "golden_wavenet_a2_lite.bin",
        distribution: V2Distribution::RequiredLocal,
        description: "WaveNet A2 Lite (CH=3) v1 golden",
    },
    V1GoldenEntry {
        golden_file: "golden_lstm_1x16.bin",
        distribution: V2Distribution::RequiredLocal,
        description: "LSTM 1×16 v1 golden",
    },
    V1GoldenEntry {
        golden_file: "golden_lstm_2x8.bin",
        distribution: V2Distribution::RequiredLocal,
        description: "LSTM 2×8 v1 golden",
    },
    V1GoldenEntry {
        golden_file: "golden_lstm_official.bin",
        distribution: V2Distribution::RequiredLocal,
        description: "LSTM Official v1 golden",
    },
    // ── LocalNonDistributable model golden (v1, 48 kHz) — OptionalExternal ─
    V1GoldenEntry {
        golden_file: "golden_wavenet_lite.bin",
        distribution: V2Distribution::OptionalExternal,
        description: "WaveNet Lite (CH=12, community) v1 golden",
    },
    // ── CabSim convolution goldens — RequiredLocal ────────────────────────
    V1GoldenEntry {
        golden_file: "golden_cabsim_cpp_short.bin",
        distribution: V2Distribution::RequiredLocal,
        description: "CabSim short IR convolution golden",
    },
    V1GoldenEntry {
        golden_file: "golden_cabsim_cpp_medium.bin",
        distribution: V2Distribution::RequiredLocal,
        description: "CabSim medium IR convolution golden",
    },
    V1GoldenEntry {
        golden_file: "golden_cabsim_cpp_long.bin",
        distribution: V2Distribution::RequiredLocal,
        description: "CabSim long IR convolution golden",
    },
];

/// Validates the canonical V1 golden catalog against disk.
///
/// For every entry of [`V1_GOLDEN_CATALOG`], checks that the `.bin` file exists
/// under `tests/fixtures/`. RequiredLocal entries (DistributedCore model
/// goldens + CabSim convolution goldens) absent from disk are hard preflight
/// failures; the single OptionalExternal entry (WaveNet Lite,
/// non-distributable) is recorded as a graceful missing-optional.
///
/// Returns `Ok(CatalogStatus)` describing disk integrity, or `Err` when the
/// environment is not a valid crate checkout. Consumable by integration tests
/// (`tests/models/golden_vectors.rs::catalog_preflight`) and binaries.
pub fn validate_v1_goldens() -> Result<CatalogStatus, CatalogError> {
    if V1_GOLDEN_CATALOG.is_empty() {
        return Err(CatalogError::CatalogInvariant(
            "V1_GOLDEN_CATALOG defines no entries",
        ));
    }

    let fixtures_root = crate::testing::fixtures::fixture_dir();
    if !fixtures_root.is_dir() {
        return Err(CatalogError::FixturesRootUnavailable(fixtures_root));
    }

    let mut status = CatalogStatus::default();

    for entry in V1_GOLDEN_CATALOG {
        status.entries_checked += 1;
        status.fixtures_checked += 1;
        let path = fixtures_root.join(entry.golden_file);
        if path.exists() {
            status.present += 1;
        } else {
            let rel = format!("tests/fixtures/{}", entry.golden_file);
            match entry.distribution {
                V2Distribution::RequiredLocal => status.missing_required.push(rel),
                V2Distribution::OptionalExternal => status.missing_optional.push(rel),
            }
        }
    }

    Ok(status)
}

#[cfg(test)]
#[path = "catalog_test.rs"]
mod tests;
