<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# NAMB Binary Format Specification

Specification of the `.namb` binary format for optimized distribution of NAM (Neural Amp Modeler) models.

## 1. Header (80 bytes, packed)

The header is a `#[repr(C, packed)]` struct — **without implicit padding**. All multi-byte integers are in **little-endian** format.

| Offset | Size | Field              | Type      | Description                                                                                               |
|:------ |:---- |:------------------ |:--------- |:--------------------------------------------------------------------------------------------------------- |
| 0x00   | 4    | `magic`            | `u32`     | Magic number `0x4E414D42` ("NAMB" in ASCII LE). Different values are rejected with `InvalidMagic`.        |
| 0x04   | 2    | `version`          | `u16`     | Format version: `1` = legacy, `2` = pre-transposed                                                        |
| 0x06   | 1    | `layout_type`      | `u8`      | Weight layout (active for `version >= 2`): `0` = Original, `1` = GateMajorLstm, `2` = Interleaved4WaveNet |
| 0x07   | 1    | `flags`            | `u8`      | Bitmask of feature flags (see §2)                                                                         |
| 0x08   | 4    | `reserved_v2`      | `[u8;4]`  | Reserved for future expansion                                                                             |
| 0x0C   | 4    | `weights_offset`   | `u32`     | Offset of the start of the weights block from the beginning of the file. Must be `>= 0x50` (80).          |
| 0x10   | 8    | `reserved1`        | `[u32;2]` | Reserved for future expansion                                                                             |
| 0x18   | 4    | `crc32`            | `u32`     | IEEE 802.3 CRC32 of the file (see §6 for range details). Semantics vary by version.                       |
| 0x1C   | 4    | `reserved2`        | `u32`     | Reserved for future expansion                                                                             |
| 0x20   | 32   | `version_str`      | `[u8;32]` | Informational version string, null-terminated (e.g., `"NAMB v2 (GateMajorLstm)\0"`).                      |
| 0x40   | 4    | `sample_rate`      | `f32`     | Default sample rate (e.g., `48000.0`)                                                                     |
| 0x44   | 4    | `input_level_dbu`  | `f32`     | Input level in dBu (e.g., `12.0`)                                                                         |
| 0x48   | 4    | `output_level_dbu` | `f32`     | Output level in dBu (e.g., `-6.0`)                                                                        |
| 0x4C   | 4    | `reserved3`        | `[u32;1]` | Reserved for future expansion                                                                             |

**Total header size: 80 bytes (0x50).**

### 1.1 Magic number

The canonical value is `0x4E414D42` (bytes `42 4D 41 4E` = `"NAMB"` in little-endian). Any other value is rejected with `NambError::InvalidMagic` (E1202). The encoder **always** writes `0x4E414D42`.

### 1.2 Versioning

| Version | Description                                                                                                                                                                                                                |
|:------- |:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `1`     | Legacy format. No `flags` field — the byte at offset 0x07 is part of `reserved_v2: [u8;5]`. `layout_type` was part of the reserved region, treated as `0` (Original). Optional CRC via sentinel `crc32 == 0` (deprecated). |
| `2`     | Pre-transposed format. Byte 0x07 is `flags: u8`, `reserved_v2` reduced to 4 bytes. `layout_type` active. `FLAG_HAS_CRC32` is mandatory.                                                                                    |

Values of `version` other than `1` and `2` are rejected with `NambError::InvalidVersion`.

### 1.3 Alignment-Safe Header Parsing (`NambHeader::from_slice`)

Because `NambHeader` is `#[repr(C, packed)]` without implicit padding, direct pointer references to unaligned fields risk hardware exceptions on architectures sensitive to misaligned access. The canonical constructor in [`src/loader/namb/header.rs`](../src/loader/namb/header.rs) is [`NambHeader::from_slice`](../src/loader/namb/header.rs):

```rust,ignore
pub fn from_slice(bytes: &[u8]) -> Result<Self, NambError>
```

It copies the header by value using `std::ptr::read_unaligned`, validating:

- Minimum length requirement (`bytes.len() >= 80`);
- Magic signature (`0x4E414D42`);
- Supported version compliance (`version ∈ {1, 2}`);
- Full safety and zero heap allocation even when the source buffer begins at an odd memory address.

## 2. Flags

| Bit | Name                       | Value  | Description                                                                                        |
|:--- |:-------------------------- |:------ |:-------------------------------------------------------------------------------------------------- |
| 0   | `FLAG_HAS_CRC32`           | `0x01` | Indicates that the `crc32` field contains a valid CRC that **must** be verified (mandatory in v2+) |
| 1   | `FLAG_HAS_QUANT_INT8`      | `0x02` | Reserved: SmoothQuant INT8 Quantization                                                            |
| 2   | `FLAG_HAS_QUANT_INT4`      | `0x04` | Reserved: INT4 Quantization                                                                        |
| 3   | `FLAG_HAS_AMX_TILE_LAYOUT` | `0x08` | Reserved: AMX tile layout                                                                          |
| 4   | `FLAG_HAS_SVE2_LAYOUT`     | `0x10` | Reserved: SVE2 layout                                                                              |
| 5-7 | —                          | —      | Reserved for future use. **Must be 0.**                                                            |

## 3. Weight Layouts

### 3.1 `Original` (`layout_type = 0`)

Format inherited from the C++ NAM ecosystem. Weights are stored exactly as they appear in the JSON `.nam` file, without transposition.

#### LSTM — Original Order

```text
For each layer l (0..num_layers-1):
  input_hidden_weights:  [4*H][input+hidden]  f32 row-major  (e.g., [gate][hidden][IH])
  bias:                  [4*H]                f32
  hidden_init:           [H]                  f32
  cell_init:             [H]                  f32

After all layers:
  head_weights:  [H]  f32
  head_bias:     1    f32
```

Where:

- `H` = `hidden_size`
- `input` = `1` (layer 1) or `H` (subsequent layers)
- `IH` = `input + H`
- `4*H` corresponds to the 4 LSTM gates (input, forget, cell, output)

The C++ convention stores `[gate][hidden][IH]` row-major.

#### WaveNet — Original Order

```text
Array 1 (layer 0):
  Rechannel:      [CH][IN=1]          f32 row-major  [OUT][IN]
  For each dilation d:
    Conv1d:       [CO][CH][K]         f32 row-major  [OUT][IN][K]
    Conv1d bias:  [CO]                f32
    Input Mixin:  [CH][COND=1]        f32 row-major  [OUT][IN]
    1x1:          [CH][CH]            f32 row-major  [OUT][IN]
    1x1 bias:     [CH]                f32
  Head Rechannel: [HEAD][CH]          f32 row-major  [OUT][IN]  (no bias in array 1)

Array 2 (layer 1):
  Rechannel:      [HEAD][IN=1]        f32 row-major
  For each dilation d:
    Conv1d:       [CO][HEAD][K]       f32 row-major
    Conv1d bias:  [CO]                f32
    Input Mixin:  [HEAD][COND=1]      f32 row-major
    1x1:          [HEAD][HEAD]        f32 row-major
    1x1 bias:     [HEAD]              f32
  Head Rechannel: [1][HEAD]           f32 row-major  (with bias in array 2)

Head Scale: 1 f32
```

Where:

- `CH` = layer internal channels
- `CO` = convolution output channels (`CH*2` if gated, else `CH`)
- `HEAD` = layer output channels (head_size)
- `K` = convolution kernel size

### 3.2 `GateMajorLstm` (`layout_type = 1`)

Optimized layout for LSTM with transposition `[Gate][Hidden][IH]` → `[Gate][IH][Hidden]`. Allows the decoder to read weights directly without load-time transposition.

**Encoder**: weights for each layer are interleaved in the correct order — after writing `W_l, bias_l`, the states `hidden_init_l, cell_init_l` are written immediately, ensuring the decoder reads the complete sequence per layer.

```text
For each layer l (0..num_layers-1):
  gate_major_weights:  [4*H][IH]  →  transposed to [Gate][IH][H]
    Original: raw[k * H * IH + i * IH + j]  →  [gate=k][hidden=i][input=j]
    Transposed: out[k * H * IH + j * H + i]  →  [gate=k][input=j][hidden=i]
  bias:                [4*H]      f32  (not transposed)
  hidden_init:         [H]        f32  (not transposed)
  cell_init:           [H]        f32  (not transposed)

After all layers:
  head_weights:  [H]  f32  (not transposed)
  head_bias:     1    f32  (not transposed)
```

The decoder, when detecting `layout_type = 1`, reads the weights directly without additional transposition, since they are already in the runtime's consumption format.

### 3.3 `Interleaved4WaveNet` (`layout_type = 2`)

Optimized layout for WaveNet with two transformations:

1. **Conv1d interleaved-4**: weights rearranged to `[BLOCK][K][IN][4]` allowing SIMD processing of 4 channels at a time.
2. **Transposed dense layers**: `[OUT][IN]` → `[IN][OUT]` for efficient sequential access in the decoder.

```text
Array 1 (layer 0):
  Rechannel:      [IN=1][CH]          f32  transposed [OUT][IN]→[IN][OUT]
  For each dilation d:
    Conv1d:       [BLOCK][K][CH][4]   f32  interleaved-4 (see below)
    Conv1d bias:  [CO]                f32  (not transposed)
    Input Mixin:  [COND=1][CH]        f32  transposed [OUT][IN]→[IN][OUT]
    1x1:          [CH][CH]            f32  transposed [OUT][IN]→[IN][OUT]
    1x1 bias:     [CH]                f32  (not transposed)
  Head Rechannel: [CH][HEAD]          f32  transposed [OUT][IN]→[IN][OUT]
  Head bias:      [HEAD]              f32  (if head_bias=true)

Array 2 (layer 1):
  ... (same structure with adapted IN/OUT)
```

#### Conv1d Interleaved-4

Given `CO` = output channels, `K` = kernel, `IN` = input channels:

```text
num_blocks = ceil(CO / 4)
total_padded = num_blocks * 4 * IN * K  f32

Layout: [BLOCK][K][IN][LANE]  (innermost = lane of 4)

For block b in 0..num_blocks-1:
  For kernel ki in 0..K-1:
    For input channel in_c in 0..IN-1:
      For lane in 0..3:
        out_c = b * 4 + lane
        If out_c < CO:
          value = original[(out_c * IN + in_c) * K + ki]
        Else:
          value = 0.0  (zero-padding)
```

**Implicit Padding**: when `CO % 4 != 0`, the encoder pads the remaining slots with `0.0` to keep the interleaved format uniform. The decoder always reads the padded size (`num_blocks * 4 * IN * K`) but only accesses `CO` logical channels. This ensures that future geometries with `CO % 4 != 0` work correctly without conditional code in the decoder.

## 4. Post-Header Sections

### 4.1 Metadata JSON (optional)

UTF-8 between offset 80 (0x50) and `weights_offset`. If `weights_offset > 80`, the intermediate bytes contain the JSON serialization of `NamModelData` **excluding the `weights` field** (serialized via `serde_json`).

The JSON is null-terminated; null bytes after the terminator are ignored. If the JSON is empty or `weights_offset == 80`, the decoder synthesizes a fallback configuration (WaveNet Standard).

### 4.2 Weights Block

Contiguous array of little-endian `f32` starting from `weights_offset` to the end of the file. Each `f32` occupies 4 bytes. The number of floats is `(file_len - weights_offset) / 4`.

Residual bytes that do not complete a 4-byte `f32` boundary (1–3 trailing bytes) are **rejected** with `NambError::Truncated { got, need }`, where `got` is the actual file size and `need` is the minimum size required to form a complete `f32`.

The `f32` values are read via `f32::from_le_bytes()` in 4-byte chunks. There is no alignment or padding within the block — it is a pure contiguous stream.

## 5. File Structure

```text
┌──────────────────────┐
│   NambHeader         │  offset 0x0000 .. 0x004F  (80 bytes)
│   (80 bytes)         │
├──────────────────────┤
│   JSON Metadata      │  offset 0x0050 .. (weights_offset - 1)  OPTIONAL
│   (UTF-8, null-term) │
├──────────────────────┤
│   Weights Blob       │  offset weights_offset .. EOF  REQUIRED
│   (contiguous LE f32)│
└──────────────────────┘
```

## 6. Integrity Validation (CRC32)

### 6.1 Algorithm

- **Standard:** CRC32 IEEE 802.3 (CRC-32/ISO-HDLC, PKZIP, Ethernet)
- **Polynomial:** `0xEDB88320` (reflected)
- **Init:** `0xFFFFFFFF`
- **XOR out:** `0xFFFFFFFF`
- **Implementation:** pure software (no external dependency), 8 bits per iteration

### 6.2 Coverage Range

- **Version 1**: The CRC32 covers **only the weights block** (`data[weights_offset..]`). The header and metadata JSON are **not** covered by the CRC.
- **Version 2+**: The CRC32 covers the **entire file** except the `crc32` field itself (i.e. `data[..24]` concatenated with `data[28..EOF]`).

### 6.3 Semantics by Version

| Condition                     | Behavior                                                                                                            |
|:----------------------------- |:------------------------------------------------------------------------------------------------------------------- |
| v1, `crc32 == 0`              | CRC **ignored** (sentinel). Emits `log::warn!` deprecation warning. Compatibility with pre-flag files (deprecated). |
| v1, `crc32 != 0`              | CRC **validated** normally.                                                                                         |
| v2+, `FLAG_HAS_CRC32` not set | **Rejected** with `NambError::CrcMissing { version }`.                                                              |
| v2+, `FLAG_HAS_CRC32` set     | CRC **validated** always (treating `crc32 == 0` as a legitimate ~1/2³² coincidence case).                           |

### 6.4 Encoder Behavior

- v1: `flags = 0`, CRC always calculated and written.
- v2+: `flags = FLAG_HAS_CRC32` (0x01), CRC always calculated and written.

## 7. Hexadecimal Examples

### 7.1 Minimal Header (v2, Original, no JSON)

```text
Offset  Hex                                           Decoding
------  --------------------------------------------  ------------------------------
0x00    42 4D 41 4E                                   magic = "NAMB" (0x4E414D42) LE
0x04    02 00                                         version = 2
0x06    00                                            layout_type = Original (0)
0x07    01                                            flags = FLAG_HAS_CRC32 (0x01)
0x08    00 00 00 00                                   reserved_v2
0x0C    50 00 00 00                                   weights_offset = 80 (0x50)
0x10    00 00 00 00 00 00 00 00                       reserved1
0x18    XX XX XX XX                                   crc32 (calculated)
0x1C    00 00 00 00                                   reserved2
0x20    4E 41 4D 42 20 76 32 20 28 4F 72 69 67 69     version_str = "NAMB v2 (Original)\0..."
        6E 61 6C 29 00 00 00 00 00 00 00 00 00 00
        00 00 00 00 00 00
0x40    00 00 7F 47                                   sample_rate = 48000.0 (f32 LE: 0x477F0000)
0x44    00 00 40 41                                   input_level_dbu = 12.0 (f32 LE: 0x41400000)
0x48    00 00 C0 C0                                   output_level_dbu = -6.0 (f32 LE: 0xC0C00000)
0x4C    00 00 00 00                                   reserved3
```

### 7.2 GateMajorLstm — Microscopic Example (H=2, 1-layer)

Model with `hidden_size=2`, 1 layer, `input=1`:

```text
Original (Layout=0):
  weights [4*H][IH=3] = 24 f32: [g0h0i0, g0h0i1, g0h0i2, g0h1i0, g0h1i1, g0h1i2,
                                 g1h0i0, g1h0i1, g1h0i2, g1h1i0, g1h1i1, g1h1i2,
                                 g2h0i0, g2h0i1, g2h0i2, g2h1i0, g2h1i1, g2h1i2,
                                 g3h0i0, g3h0i1, g3h0i2, g3h1i0, g3h1i1, g3h1i2]

GateMajorLstm (Layout=1):
  transposed weights [Gate][IH][H] = 24 f32: [g0i0h0, g0i0h1, g0i1h0, g0i1h1, g0i2h0, g0i2h1,
                                              g1i0h0, g1i0h1, g1i1h0, g1i1h1, g1i2h0, g1i2h1,
                                              g2i0h0, g2i0h1, g2i1h0, g2i1h1, g2i2h0, g2i2h1,
                                              g3i0h0, g3i0h1, g3i1h0, g3i1h1, g3i2h0, g3i2h1]
```

If the original weights were `1..24` (f32), the original reading would be row-by-row: `1,2,3,4,5,6, 7,8,...`. The GateMajorLstm would be: `1,4,2,5,3,6, 7,10,...`.

**Complete order in the weights block (1-layer, H=2):**

```text
Offset in block  Content
---------------  --------
+0               gate_major_weights[24 f32]
+96              bias[8 f32]
+128             hidden_init[2 f32]
+136             cell_init[2 f32]
+144             head_weights[2 f32]
+152             head_bias[1 f32]
```

### 7.3 Interleaved4WaveNet — Conv1d Example with padding (CO=6, K=3, IN=2)

`CO=6` output channels is not a multiple of 4. `num_blocks = ceil(6/4) = 2`.

```text
num_blocks = 2, padded_total = 2 * 4 * 2 * 3 = 48 f32

Original (Layout=0): [OUT][IN][K] = 36 f32
  out0: in0k0, in0k1, in0k2, in1k0, in1k1, in1k2
  out1: in0k0, in0k1, in0k2, in1k0, in1k1, in1k2
  ...
  out5: in0k0, in0k1, in0k2, in1k0, in1k1, in1k2

Interleaved4WaveNet (Layout=2): [BLOCK][K][IN][LANE] = 48 f32

  Block 0 (covers out0..out3):
    k=0, in0: out0in0k0, out1in0k0, out2in0k0, out3in0k0  (lane 0..3)
    k=0, in1: out0in1k0, out1in1k0, out2in1k0, out3in1k0
    k=1, in0: out0in0k1, out1in0k1, out2in0k1, out3in0k1
    k=1, in1: out0in1k1, out1in1k1, out2in1k1, out3in1k1
    k=2, in0: out0in0k2, out1in0k2, out2in0k2, out3in0k2
    k=2, in1: out0in1k2, out1in1k2, out2in1k2, out3in1k2

  Block 1 (covers out4..out5, lanes 2-3 are zero-padded):
    k=0, in0: out4in0k0, out5in0k0, 0.0, 0.0
    k=0, in1: out4in1k0, out5in1k0, 0.0, 0.0
    k=1, in0: out4in0k1, out5in0k1, 0.0, 0.0
    k=1, in1: out4in1k1, out5in1k1, 0.0, 0.0
    k=2, in0: out4in0k2, out5in0k2, 0.0, 0.0
    k=2, in1: out4in1k2, out5in1k2, 0.0, 0.0
```

## 8. Typed Errors (`NambError`)

NAMB parsing uses typed errors via `thiserror` for precise diagnostics. The `loader` module performs downcasting via `downcast_ref` to map each variant to the corresponding `NamErrorCode`.

| Variant                                         | Code  | Mnemonic                    | Condition                                                                |
|:----------------------------------------------- |:----- |:--------------------------- |:------------------------------------------------------------------------ |
| `Truncated { got, need }`                       | E1204 | `NAMB_TRUNCATED`            | File < 80 bytes or trailing bytes do not complete 4-byte float boundary  |
| `InvalidMagic(u32)`                             | E1202 | `NAMB_INVALID_MAGIC`        | Magic is not `0x4E414D42`                                                |
| `InvalidVersion(u16)`                           | E1203 | `NAMB_UNSUPPORTED_VERSION`  | Version is not 1 or 2                                                    |
| `WeightsOffsetOutOfBounds { offset, file_len }` | E1204 | `NAMB_TRUNCATED`            | `weights_offset` exceeds file size                                       |
| `InvalidWeightsOffset { offset, header_size }`  | E1204 | `NAMB_TRUNCATED`            | `weights_offset` < 80 (less than header size)                            |
| `CrcMismatch { got, expected }`                 | E1201 | `NAMB_CRC32_MISMATCH`       | Calculated CRC ≠ header CRC                                              |
| `CrcMissing { version }`                        | E1205 | `NAMB_CRC32_MISSING`        | File v2+ without `FLAG_HAS_CRC32` set                                    |
| `CrcMissingV1`                                  | E1205 | `NAMB_CRC32_MISSING`        | Legacy v1 file with `crc32 == 0` sentinel rejected by integrity policy   |
| `WeightsTooLarge { got, max }`                  | E1304 | `MODEL_TOO_LARGE`           | Float count exceeds `MAX_MODEL_BYTES / 4` (67,108,864 floats)            |
| `NonFiniteWeight { index, value }`              | E1212 | `NAMB_NON_FINITE_WEIGHT`    | Non-finite weight float (`NaN`, `+Inf`, `-Inf`) in binary weight section |
| `InvalidHeaderField { field, value, reason }`   | E1213 | `NAMB_INVALID_HEADER_FIELD` | Non-finite or invalid float in header metadata fields                    |

Global limit: `MAX_MODEL_BYTES = 256 MiB`. Files larger than this limit are rejected with `NamErrorCode::ModelTooLarge` (E1304).

## 9. Evolution and Compatibility Policy

### 9.1 Backward Compatibility

- **v1**: maintained for reading legacy files. Optional CRC via sentinel `crc32 == 0` (deprecated). In v1 headers, the byte at offset `0x07` is treated as reserved (part of `reserved_v2`) and must not be interpreted as feature flags (e.g. even if it is non-zero, it does not enable feature flags). This ensures forward-compatibility and prevents false interpretation of flags in legacy files.
- **v2**: `FLAG_HAS_CRC32` mandatory. `layout_type` active.
- **Interleaved-4 zero padding**: transparent change. NAMB v2 models produced before the fix remain valid (only affects geometries where `CO % 4 != 0`, which do not exist in the current catalog). Documented as an implicit bump without a version change.

### 9.2 Future Expansion

- **New `layout_type`**: values `3..=255` are reserved for future layouts. Unknown values are treated as `Original` by the decoder.
- **New flags**: bits `1..=4` are reserved for specific extensions (INT8, INT4, AMX, SVE2) and bits `5..=7` of the `flags` field are reserved for general future use. They must be `0` in current files. Future decoders can define new bits with fallbacks to current behavior when not set.
- **`reserved_*` fields**: must be `0`. Reserved for adding fields without breaking binary header compatibility.
- **New versions**: versions `>= 3` are reserved for structural changes that require revising the header format or weights block.

## 10. Reference Constants

| Constant            | Value                   | Location                                                                       |
|:------------------- |:----------------------- |:------------------------------------------------------------------------------ |
| `FLAG_HAS_CRC32`    | `0x01` (u8)             | [`src/loader/namb/header.rs:54`](../src/loader/namb/header.rs#L54)             |
| `MAX_MODEL_BYTES`   | `268_435_456` (256 MiB) | [`src/loader/loaded_model_pair.rs:15`](../src/loader/loaded_model_pair.rs#L15) |
| Header size         | `80` bytes (0x50)       | `std::mem::size_of::<NambHeader>()`                                            |
| CRC polynomial      | `0xEDB88320`            | [`src/loader/namb/header.rs:15`](../src/loader/namb/header.rs#L15)             |
| CRC init            | `0xFFFFFFFF`            | [`src/loader/namb/header.rs:24`](../src/loader/namb/header.rs#L24)             |
| CRC xorout          | `0xFFFFFFFF`            | [`src/loader/namb/header.rs:24`](../src/loader/namb/header.rs#L24)             |
| Magic LE            | `0x4E414D42`            | [`src/loader/namb/header.rs:95`](../src/loader/namb/header.rs#L95)             |
| Default sample rate | `48000.0` (f32)         | [`src/loader/loaded_model_pair.rs:13`](../src/loader/loaded_model_pair.rs#L13) |
| Default input dBu   | `12.0` (f32)            | [`src/loader/loaded_model_pair.rs:9`](../src/loader/loaded_model_pair.rs#L9)   |
| Default output dBu  | `-6.0` (f32)            | [`src/loader/namb_encoder.rs:86`](../src/loader/namb_encoder.rs#L86)           |
| Default loudness    | `-18.0` (f32)           | [`src/loader/loaded_model_pair.rs:11`](../src/loader/loaded_model_pair.rs#L11) |

## 11. References

- NAMB Decoder: [`src/loader/namb/mod.rs`](../src/loader/namb/mod.rs)
- NAMB Encoder: [`src/loader/namb_encoder.rs`](../src/loader/namb_encoder.rs)
- Layout Definitions: [`src/loader/nam_json/model.rs`](../src/loader/nam_json/model.rs) (`WeightsLayout`)
- LSTM Dispatcher: [`src/loader/dispatcher/lstm/mod.rs`](../src/loader/dispatcher/lstm/mod.rs)
- WaveNet Dispatcher: [`src/loader/dispatcher/wavenet/mod.rs`](../src/loader/dispatcher/wavenet/mod.rs)
- Error Definition: [`src/loader/namb/error.rs`](../src/loader/namb/error.rs) (`NambError`)
- Error Mapping: [`src/loader/build.rs`](../src/loader/build.rs)
- Error Codes: [`src/common/diagnostics/error_codes.rs`](../src/common/diagnostics/error_codes.rs) (`NamErrorCode`)
- Round-trip Tests: [`tests/models/namb_v2_roundtrip.rs`](../tests/models/namb_v2_roundtrip.rs), [`tests/models/namb_v2_validation.rs`](../tests/models/namb_v2_validation.rs)
- Architecture: [`docs/architecture.md`](architecture.md)
