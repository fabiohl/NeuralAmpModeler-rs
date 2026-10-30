// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! CLI tool to generate, inspect, and bitwise-compare cross-platform parity vectors.
//!
//! # Commands
//! - `generate`: Runs real inference with a `.nam` model on a standard stress signal and exports a `NAMPARV1` binary vector.
//! - `compare`: Compares two parity vectors bit-by-bit (`f32::to_bits()`) and evaluates ESR/SNR/MSE.
//! - `inspect`: Displays metadata and statistics of a parity vector file.

use std::fmt::Write as FmtWrite;
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use sha2::{Digest, Sha256};

use neural_amp_modeler_rs::loader::dispatcher::build_model;
use neural_amp_modeler_rs::loader::nam_json::parse_nam_json;
use neural_amp_modeler_rs::math::common::effective_instruction_set;
use neural_amp_modeler_rs::models::NamModel;
use neural_amp_modeler_rs::testing::stress::{
    generate_stress_signal_v1, generate_stress_signal_v2,
};
use neural_amp_modeler_rs::testing::vector_protocol::{
    ChannelMode, DspMode, InitialState, PARITY_VECTOR_MAGIC, PARITY_VECTOR_VERSION, ParityVector,
    ParityVectorHeader, compare_vectors,
};

fn bytes_to_hex(bytes: &[u8]) -> String {
    bytes
        .iter()
        .fold(String::with_capacity(bytes.len() * 2), |mut acc, b| {
            let _ = write!(acc, "{b:02x}");
            acc
        })
}

fn print_usage() {
    eprintln!(
        r#"Uso: nam_parity_vector <comando> [opções]

Comandos:
  generate  Executa inferência real de produto e grava vetor binário NAMPARV1
  compare   Compara dois vetores bit a bit e avalia ESR/SNR/mismatches
  inspect   Exibe metadados e estatísticas de um vetor gravado

Opções de 'generate':
  --model <PATH>          Caminho para arquivo .nam (obrigatório)
  --signal <v1|v2|sine>   Tipo de sinal de teste (default: v1)
  --out <PATH>            Caminho do arquivo binário de saída (obrigatório)
  --sample-rate <HZ>      Taxa de amostragem (default: 48000)
  --block-size <SAMPLES>  Tamanho do bloco de processamento (default: 64)
  --mode <live|hq>        Modo de processamento (default: live)
  --channels <mono|dual>  Canais de processamento (default: mono)
  --prewarm <SAMPLES>     Amostras de pré-aquecimento (default: 4096)
  --run-id <ID>           Identificador de execução (opcional)

Opções de 'compare':
  <VETOR_REF> <VETOR_TEST>  Caminhos dos vetores a comparar (obrigatórios)
  --json                    Exibe relatório detalhado em formato JSON no stdout
  --report-out <PATH>       Grava relatório JSON em arquivo

Opções de 'inspect':
  <VETOR>                   Caminho do arquivo de vetor a inspecionar
"#
    );
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        print_usage();
        return ExitCode::from(2);
    }

    match args[1].as_str() {
        "generate" => cmd_generate(&args[2..]),
        "compare" => cmd_compare(&args[2..]),
        "inspect" => cmd_inspect(&args[2..]),
        "-h" | "--help" | "help" => {
            print_usage();
            ExitCode::SUCCESS
        }
        unknown => {
            eprintln!("Erro: comando desconhecido '{unknown}'");
            print_usage();
            ExitCode::from(2)
        }
    }
}

fn cmd_generate(args: &[String]) -> ExitCode {
    let mut model_path: Option<PathBuf> = None;
    let mut signal_type = String::from("v1");
    let mut out_path: Option<PathBuf> = None;
    let mut sample_rate: u32 = 48000;
    let mut block_size: u32 = 64;
    let mut dsp_mode = DspMode::Live;
    let mut channel_mode = ChannelMode::Mono;
    let mut prewarm_samples: usize = 4096;
    let mut run_id = std::env::var("NAM_RUN_ID").ok();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                i += 1;
                if i < args.len() {
                    model_path = Some(PathBuf::from(&args[i]));
                }
            }
            "--signal" => {
                i += 1;
                if i < args.len() {
                    signal_type = args[i].clone();
                }
            }
            "--out" => {
                i += 1;
                if i < args.len() {
                    out_path = Some(PathBuf::from(&args[i]));
                }
            }
            "--sample-rate" => {
                i += 1;
                if i < args.len() {
                    sample_rate = args[i].parse().unwrap_or(48000);
                }
            }
            "--block-size" => {
                i += 1;
                if i < args.len() {
                    block_size = args[i].parse().unwrap_or(64);
                }
            }
            "--mode" => {
                i += 1;
                if i < args.len() {
                    dsp_mode = match args[i].to_lowercase().as_str() {
                        "hq" => DspMode::Hq,
                        _ => DspMode::Live,
                    };
                }
            }
            "--channels" => {
                i += 1;
                if i < args.len() {
                    channel_mode = match args[i].to_lowercase().as_str() {
                        "dual" | "dual-mono" | "dual_mono" => ChannelMode::DualMono,
                        _ => ChannelMode::Mono,
                    };
                }
            }
            "--prewarm" => {
                i += 1;
                if i < args.len() {
                    prewarm_samples = args[i].parse().unwrap_or(4096);
                }
            }
            "--run-id" => {
                i += 1;
                if i < args.len() {
                    run_id = Some(args[i].clone());
                }
            }
            other => {
                eprintln!("Erro: argumento desconhecido em 'generate': '{other}'");
                return ExitCode::from(2);
            }
        }
        i += 1;
    }

    let model_path = match model_path {
        Some(p) => p,
        None => {
            eprintln!("Erro: argumento obrigatório '--model <PATH>' não fornecido.");
            return ExitCode::from(2);
        }
    };
    let out_path = match out_path {
        Some(p) => p,
        None => {
            eprintln!("Erro: argumento obrigatório '--out <PATH>' não fornecido.");
            return ExitCode::from(2);
        }
    };

    if !model_path.exists() {
        eprintln!(
            "Erro: arquivo de modelo não encontrado: {}",
            model_path.display()
        );
        return ExitCode::from(2);
    }

    let model_bytes = match std::fs::read(&model_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Erro ao ler modelo {}: {e}", model_path.display());
            return ExitCode::from(2);
        }
    };

    let mut model_hasher = Sha256::new();
    model_hasher.update(&model_bytes);
    let model_sha256 = bytes_to_hex(&model_hasher.finalize());

    let model_json = match std::str::from_utf8(&model_bytes) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Erro ao decodificar JSON do modelo: {e}");
            return ExitCode::from(2);
        }
    };

    let model_data = match parse_nam_json(model_json) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Erro ao parsear modelo NAM: {e}");
            return ExitCode::from(2);
        }
    };

    let mut model = match build_model(&model_data) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Erro ao instanciar motor DSP do modelo: {e}");
            return ExitCode::from(2);
        }
    };

    // Prewarm inference state to stabilize recurrent state / history
    model.prewarm(prewarm_samples);

    // Generate deterministic test signal
    let (signal_name, signal_seed, input_signal) = match signal_type.as_str() {
        "v2" => (
            "stress_v2",
            20261029u64,
            generate_stress_signal_v2("nam-rs-stress-v2", sample_rate),
        ),
        "sine" => {
            let n = sample_rate as usize;
            let mut s = Vec::with_capacity(n);
            for idx in 0..n {
                let t = (idx as f32) / (sample_rate as f32);
                let val = (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.5;
                s.push(val);
            }
            ("sine_440hz", 0u64, s)
        }
        _ => ("stress_v1", 42u64, generate_stress_signal_v1()),
    };

    let mut sig_hasher = Sha256::new();
    for &sample in &input_signal {
        sig_hasher.update(sample.to_bits().to_le_bytes());
    }
    let signal_sha256 = bytes_to_hex(&sig_hasher.finalize());

    let num_channels: u16 = match channel_mode {
        ChannelMode::Mono => 1,
        ChannelMode::DualMono => 2,
    };
    let num_samples_per_channel = input_signal.len() as u64;

    let mut output_samples = match channel_mode {
        ChannelMode::Mono => {
            let mut out = vec![0.0f32; input_signal.len()];
            let bs = block_size as usize;
            for (in_chunk, out_chunk) in input_signal.chunks(bs).zip(out.chunks_mut(bs)) {
                model.process(in_chunk, out_chunk);
            }
            out
        }
        ChannelMode::DualMono => {
            let n = input_signal.len();
            let mut out_l = vec![0.0f32; n];
            let bs = block_size as usize;
            for (in_chunk, out_chunk) in input_signal.chunks(bs).zip(out_l.chunks_mut(bs)) {
                model.process(in_chunk, out_chunk);
            }
            // For dual mono, interleave L and R
            let mut interleaved = Vec::with_capacity(n * 2);
            for &sample in &out_l {
                interleaved.push(sample);
                interleaved.push(sample);
            }
            interleaved
        }
    };

    let target_triple = std::env::var("TARGET").unwrap_or_else(|_| {
        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
        {
            "x86_64-unknown-linux-gnu".into()
        }
        #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
        {
            "aarch64-unknown-linux-gnu".into()
        }
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            "aarch64-apple-darwin".into()
        }
        #[cfg(not(any(
            all(target_os = "linux", target_arch = "x86_64"),
            all(target_os = "linux", target_arch = "aarch64"),
            all(target_os = "macos", target_arch = "aarch64")
        )))]
        {
            "unknown".into()
        }
    });

    let instruction_set = format!("{:?}", effective_instruction_set());

    let header = ParityVectorHeader {
        magic: PARITY_VECTOR_MAGIC,
        protocol_version: PARITY_VECTOR_VERSION,
        endianness: "little".into(),
        model_id: model_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .into(),
        model_sha256,
        signal_name: signal_name.into(),
        signal_sha256,
        signal_seed,
        signal_len: input_signal.len() as u64,
        dsp_mode,
        channel_mode,
        sample_rate,
        block_size,
        initial_state: InitialState::Prewarmed,
        num_channels,
        num_samples_per_channel,
        toolchain: std::env::var("RUSTC_VERSION").unwrap_or_else(|_| "rustc 1.99.0".into()),
        compile_flags: std::env::var("RUSTFLAGS").unwrap_or_else(|_| "-C opt-level=3".into()),
        target_triple,
        instruction_set,
        fp_env: "FTZ/DAZ active".into(),
        run_id,
    };

    let vector = ParityVector {
        header,
        samples: std::mem::take(&mut output_samples),
    };

    if let Some(parent) = out_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }

    let file = match File::create(&out_path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Erro ao criar arquivo de saída {}: {e}", out_path.display());
            return ExitCode::from(2);
        }
    };
    let mut writer = BufWriter::new(file);

    if let Err(e) = vector.write_to(&mut writer) {
        eprintln!("Erro ao gravar vetor binário: {e}");
        return ExitCode::from(2);
    }

    println!(
        "[PARITY VECTOR GENERATED] path={} model='{}' samples={} channels={} rate={}Hz target={}",
        out_path.display(),
        vector.header.model_id,
        vector.header.num_samples_per_channel,
        vector.header.num_channels,
        vector.header.sample_rate,
        vector.header.target_triple
    );

    ExitCode::SUCCESS
}

fn cmd_compare(args: &[String]) -> ExitCode {
    let mut paths = Vec::new();
    let mut emit_json = false;
    let mut report_out: Option<PathBuf> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--json" => emit_json = true,
            "--report-out" => {
                i += 1;
                if i < args.len() {
                    report_out = Some(PathBuf::from(&args[i]));
                }
            }
            arg if !arg.starts_with("--") => {
                paths.push(PathBuf::from(arg));
            }
            other => {
                eprintln!("Erro: argumento desconhecido em 'compare': '{other}'");
                return ExitCode::from(2);
            }
        }
        i += 1;
    }

    if paths.len() < 2 {
        eprintln!("Erro: compare requer exatamente 2 caminhos de vetores.");
        eprintln!(
            "Uso: nam_parity_vector compare <VETOR_REF> <VETOR_TEST> [--json] [--report-out <PATH>]"
        );
        return ExitCode::from(2);
    }

    let ref_path = &paths[0];
    let test_path = &paths[1];

    let ref_vec = match load_vector(ref_path) {
        Ok(v) => v,
        Err(e) => {
            eprintln!(
                "FAIL: não foi possível carregar vetor de referência '{}': {e}",
                ref_path.display()
            );
            return ExitCode::from(2);
        }
    };

    let test_vec = match load_vector(test_path) {
        Ok(v) => v,
        Err(e) => {
            eprintln!(
                "FAIL: não foi possível carregar vetor de teste '{}': {e}",
                test_path.display()
            );
            return ExitCode::from(2);
        }
    };

    let report = compare_vectors(&ref_vec, &test_vec);

    if let Some(out_p) = report_out {
        if let Some(parent) = out_p.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Ok(mut f) = File::create(&out_p) {
            let _ = serde_json::to_writer_pretty(&mut f, &report);
            let _ = f.write_all(b"\n");
        }
    }

    if emit_json {
        println!("{}", serde_json::to_string_pretty(&report).unwrap());
    } else {
        println!("============================================================");
        println!("  PARITY VECTOR COMPARISON REPORT (R-07, R-13, S5-T02)      ");
        println!("============================================================");
        println!(
            "Reference:   {} (ISA: {})",
            report.ref_target, report.ref_isa
        );
        println!(
            "Test:        {} (ISA: {})",
            report.test_target, report.test_isa
        );
        println!(
            "Compatible:  {}",
            if report.is_compatible { "YES" } else { "NO" }
        );
        if let Some(ref reason) = report.incompatibility_reason {
            println!("Incompatibility Reason: {reason}");
        }
        println!("Total Samples Compared: {}", report.total_samples_compared);
        println!(
            "Bitwise Mismatches:     {}/{}",
            report.mismatch_count, report.total_samples_compared
        );
        println!(
            "Bitwise Identical:      {}",
            if report.bitwise_identical {
                "YES (100% bit-exact)"
            } else {
                "NO"
            }
        );
        println!("ESR:                    {:.6e}", report.esr);
        println!("SNR:                    {:.2} dB", report.snr_db);
        println!("Max Absolute Diff:      {:.6e}", report.max_abs_diff);
        println!("Mean Squared Error:     {:.6e}", report.mse);
        println!(
            "NaN Counts:             ref={}, test={}",
            report.nan_count_ref, report.nan_count_test
        );
        println!(
            "Subnormal Counts:       ref={}, test={}",
            report.subnormal_count_ref, report.subnormal_count_test
        );

        if let Some(ref m) = report.first_mismatch {
            println!("------------------------------------------------------------");
            println!("FIRST DIVERGENT BIT DETAILS:");
            println!("  Channel:            {}", m.channel);
            println!("  Sample Index:       {}", m.sample_index);
            println!(
                "  Reference Value:    {:.8e} ({})",
                m.ref_value, m.ref_bits_hex
            );
            println!(
                "  Test Value:         {:.8e} ({})",
                m.test_value, m.test_bits_hex
            );
            println!("  Absolute Diff:      {:.8e}", m.abs_diff);
        }
        println!("============================================================");
    }

    if !report.is_compatible {
        ExitCode::from(2)
    } else if report.bitwise_identical {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    }
}

fn cmd_inspect(args: &[String]) -> ExitCode {
    if args.is_empty() {
        eprintln!("Erro: inspect requer o caminho do arquivo de vetor.");
        return ExitCode::from(2);
    }

    let path = Path::new(&args[0]);
    let vector = match load_vector(path) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Erro ao carregar vetor '{}': {e}", path.display());
            return ExitCode::from(2);
        }
    };

    let hdr = &vector.header;
    println!("=== NAMPARV1 PARITY VECTOR METADATA ===");
    println!("File:             {}", path.display());
    println!("Protocol Version: {}", hdr.protocol_version);
    println!("Model ID:         {}", hdr.model_id);
    println!("Model SHA-256:    {}", hdr.model_sha256);
    println!("Signal Name:      {}", hdr.signal_name);
    println!("Signal SHA-256:   {}", hdr.signal_sha256);
    println!("Signal Seed:      {}", hdr.signal_seed);
    println!("Signal Length:    {} samples", hdr.signal_len);
    println!("DSP Mode:         {:?}", hdr.dsp_mode);
    println!("Channel Mode:     {:?}", hdr.channel_mode);
    println!("Sample Rate:      {} Hz", hdr.sample_rate);
    println!("Block Size:       {} samples", hdr.block_size);
    println!("Initial State:    {:?}", hdr.initial_state);
    println!("Channels:         {}", hdr.num_channels);
    println!("Samples/Channel:  {}", hdr.num_samples_per_channel);
    println!("Target Triple:    {}", hdr.target_triple);
    println!("Instruction Set:  {}", hdr.instruction_set);
    println!("Toolchain:        {}", hdr.toolchain);
    println!("Compile Flags:    {}", hdr.compile_flags);
    println!("FP Environment:   {}", hdr.fp_env);
    if let Some(ref rid) = hdr.run_id {
        println!("Run ID:           {rid}");
    }
    println!("Total Samples:    {}", vector.samples.len());

    let mut nan_count = 0;
    let mut subnormal_count = 0;
    let mut min_val = f32::INFINITY;
    let mut max_val = f32::NEG_INFINITY;
    let mut energy = 0.0f64;

    for &s in &vector.samples {
        if s.is_nan() {
            nan_count += 1;
        } else {
            if s < min_val {
                min_val = s;
            }
            if s > max_val {
                max_val = s;
            }
            energy += (s as f64) * (s as f64);
        }
        if s.is_subnormal() {
            subnormal_count += 1;
        }
    }

    println!("Min Sample Value: {:.6e}", min_val);
    println!("Max Sample Value: {:.6e}", max_val);
    println!(
        "RMS Energy:       {:.6e}",
        (energy / vector.samples.len().max(1) as f64).sqrt()
    );
    println!("NaN Samples:      {}", nan_count);
    println!("Subnormal Sample: {}", subnormal_count);

    ExitCode::SUCCESS
}

fn load_vector(path: &Path) -> Result<ParityVector, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let vector = ParityVector::read_from(&mut reader)?;
    Ok(vector)
}
