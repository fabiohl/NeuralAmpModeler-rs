// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Structured catalog and SHA-256 registry for test fixtures and community models.
//!
//! Provides a canonical, deduplicated registry of all 61 catalog paths mapped to
//! 51 unique SHA-256 identities. Classifies support status (45 supported, 3 intentional
//! negative fixtures, 3 known gaps) and tracks all 10 redundant file aliases.

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
        description: "Supported {'channels': 8, 'topology': 'WaveNet-Dynamic', 'type': 'WaveNet', 'weights_layout': 'Original'} model (APP-EVH-Stealth100-Dialled-xSTD.nam)",
    },
    ModelCatalogEntry {
        sha256: "cddd9e9e8fdeccb9f51b34f4c23f0f0d444c3faed307e962f76db7c6d287e1bf",
        canonical_path: "third-party/community_models/BOG UU II Gain BAL CAB.nam",
        aliases: &[],
        architecture: "{'channels': 8, 'topology': 'Unknown', 'type': 'SlimmableContainer', 'weights_layout': 'Original'}",
        support: ModelSupportKind::Supported,
        description: "Supported {'channels': 8, 'topology': 'Unknown', 'type': 'SlimmableContainer', 'weights_layout': 'Original'} model (BOG UU II Gain BAL CAB.nam)",
    },
    ModelCatalogEntry {
        sha256: "7ffd11c244664c737363d8d8753c3843f124b76a7afac7f15bcaebe0b6fd59f0",
        canonical_path: "third-party/community_models/Boss BD-2 H2O Mod T-12_00 G-12_00.nam",
        aliases: &[],
        architecture: "{'channels': 9, 'topology': 'WaveNet-Dynamic', 'type': 'WaveNet', 'weights_layout': 'Original'}",
        support: ModelSupportKind::Supported,
        description: "Supported {'channels': 9, 'topology': 'WaveNet-Dynamic', 'type': 'WaveNet', 'weights_layout': 'Original'} model (Boss BD-2 H2O Mod T-12_00 G-12_00.nam)",
    },
    ModelCatalogEntry {
        sha256: "66a4be684f6599c172d406af8f7206539fde8ffda9fd9eaefe5f3a09f388a0b8",
        canonical_path: "third-party/community_models/ChandlerRedd47-Gain34-Standard.nam",
        aliases: &[],
        architecture: "{'channels': 16, 'topology': 'Standard', 'type': 'WaveNet', 'weights_layout': 'Original'}",
        support: ModelSupportKind::Supported,
        description: "Supported {'channels': 16, 'topology': 'Standard', 'type': 'WaveNet', 'weights_layout': 'Original'} model (ChandlerRedd47-Gain34-Standard.nam)",
    },
    ModelCatalogEntry {
        sha256: "4404e56fbe20a30d57735a3294c8023c22a27bc63d0f4994b3d889605c445ad6",
        canonical_path: "third-party/community_models/EVH-5150-Lite.nam",
        aliases: &[],
        architecture: "{'channels': 12, 'topology': 'Lite', 'type': 'WaveNet', 'weights_layout': 'Original'}",
        support: ModelSupportKind::Supported,
        description: "Supported {'channels': 12, 'topology': 'Lite', 'type': 'WaveNet', 'weights_layout': 'Original'} model (EVH-5150-Lite.nam)",
    },
    ModelCatalogEntry {
        sha256: "d3f6c9e6f08cdd2a2f99ef910abdd5e37a6e1a83f79f47f40011aefcb7ec1f66",
        canonical_path: "third-party/community_models/little-bear-t7_phono-aux-tube-preamp_line-in_Standard.nam",
        aliases: &[],
        architecture: "{'channels': 16, 'topology': 'Standard', 'type': 'WaveNet', 'weights_layout': 'Original'}",
        support: ModelSupportKind::Supported,
        description: "Supported {'channels': 16, 'topology': 'Standard', 'type': 'WaveNet', 'weights_layout': 'Original'} model (little-bear-t7_phono-aux-tube-preamp_line-in_Standard.nam)",
    },
    ModelCatalogEntry {
        sha256: "4257fc55a0cf105292613a7ed8864b933f809cc265694e8f14b5b522223697fe",
        canonical_path: "third-party/community_models/NEVE1073-Standard.nam",
        aliases: &[],
        architecture: "{'channels': 16, 'topology': 'Standard', 'type': 'WaveNet', 'weights_layout': 'Original'}",
        support: ModelSupportKind::Supported,
        description: "Supported {'channels': 16, 'topology': 'Standard', 'type': 'WaveNet', 'weights_layout': 'Original'} model (NEVE1073-Standard.nam)",
    },
    ModelCatalogEntry {
        sha256: "c76c0666945213deb8f43b53690ef64ce72e5b8c562725d75a1b106372e81a61",
        canonical_path: "third-party/community_models/SLAMMIN_MARSHALL_J45_VN9_TREBLEBOOSTER_P4_C.nam",
        aliases: &[],
        architecture: "{'channels': 32, 'topology': 'WaveNet-Dynamic', 'type': 'WaveNet', 'weights_layout': 'Original'}",
        support: ModelSupportKind::Supported,
        description: "Supported {'channels': 32, 'topology': 'WaveNet-Dynamic', 'type': 'WaveNet', 'weights_layout': 'Original'} model (SLAMMIN_MARSHALL_J45_VN9_TREBLEBOOSTER_P4_C.nam)",
    },
    ModelCatalogEntry {
        sha256: "203fac43316573ecb56e4e060d361d29669084259362b09b6e728e3f3d548a2d",
        canonical_path: "third-party/community_models/UA610B-Gain+10-Standard.nam",
        aliases: &[],
        architecture: "{'channels': 16, 'topology': 'Standard', 'type': 'WaveNet', 'weights_layout': 'Original'}",
        support: ModelSupportKind::Supported,
        description: "Supported {'channels': 16, 'topology': 'Standard', 'type': 'WaveNet', 'weights_layout': 'Original'} model (UA610B-Gain+10-Standard.nam)",
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
