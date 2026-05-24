use std::path::PathBuf;

use anyhow::{Context, Result};
use candle_core::Device;
use clap::{Parser, Subcommand, ValueEnum};

use encoder::config::EncoderConfig;
use encoder::data::{load_jsonl, TrainExample};
use encoder::tokenizer::CodeTokenizer;
use encoder::train::{TrainConfig, Trainer};

#[derive(Parser)]
#[command(about = "Dual-head encoder: policy + value for MCTS code search")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Build a token vocabulary from all .rs files under a directory.
    BuildVocab {
        #[arg(long)] src_dir:    PathBuf,
        #[arg(long)] out:        PathBuf,
        #[arg(long, default_value_t = 8192)] vocab_size: usize,
    },

    /// Convert raw search dump (JSONL from `search --dump`) into tokenised
    /// TrainExample JSONL ready for `encoder train`.
    Collect {
        /// Raw JSONL written by `search --dump`.
        #[arg(long)] raw:   PathBuf,
        /// Vocabulary JSON built by `encoder build-vocab`.
        #[arg(long)] vocab: PathBuf,
        /// Output path for tokenised TrainExample JSONL.
        #[arg(long)] out:   PathBuf,
        /// Max sequence length (tokens); must match training config.
        #[arg(long, default_value_t = 2048)] max_len: usize,
        /// Token ID used for padding.
        #[arg(long, default_value_t = 0)] pad_id: u32,
    },

    /// Train the encoder on a tokenised TrainExample JSONL dataset.
    Train {
        #[arg(long)] data:        PathBuf,
        #[arg(long)] out:         PathBuf,
        #[arg(long, default_value_t = 1e-4)]  lr:           f64,
        #[arg(long, default_value_t = 32)]    batch_size:   usize,
        #[arg(long, default_value_t = 10)]    epochs:       usize,
        #[arg(long, default_value_t = 1.0)]   value_lambda: f64,
        /// Use small preset (4 layers, d=256) instead of default (6L, d=512).
        #[arg(long)] small: bool,
        /// Must match the number of op templates used during data collection, or `auto`.
        #[arg(long, default_value = "auto")] n_ops: String,
        /// Training device: auto | cpu | cuda | metal.
        #[arg(long, value_enum, default_value_t = DeviceArg::Auto)] device: DeviceArg,
        /// Existing safetensors checkpoint to load before continuing training.
        #[arg(long)] resume: Option<PathBuf>,
    },
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum DeviceArg {
    Auto,
    Cpu,
    Cuda,
    Metal,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::BuildVocab { src_dir, out, vocab_size } => {
            let corpus = collect_corpus(&src_dir)?;
            let tok = CodeTokenizer::build_vocab(&corpus, vocab_size);
            std::fs::write(&out, tok.to_json())?;
            eprintln!("Vocabulary ({} tokens) → {}", tok.vocab_size(), out.display());
        }

        Cmd::Collect { raw, vocab, out, max_len, pad_id } => {
            let vocab_json = std::fs::read_to_string(&vocab)?;
            let tok = CodeTokenizer::from_json(&vocab_json)?;

            let raw_text = std::fs::read_to_string(&raw)?;
            let mut count = 0usize;

            use std::io::Write;
            let mut outfile = std::fs::File::create(&out)?;

            for (lineno, line) in raw_text.lines().enumerate() {
                let line = line.trim();
                if line.is_empty() { continue; }

                let rec: serde_json::Value = serde_json::from_str(line)
                    .map_err(|e| anyhow::anyhow!("line {}: {e}", lineno + 1))?;

                let root = rec["root"].as_str()
                    .ok_or_else(|| anyhow::anyhow!("line {}: missing root", lineno + 1))?;
                let policy_raw = rec["policy"].as_array()
                    .ok_or_else(|| anyhow::anyhow!("line {}: missing policy", lineno + 1))?;
                let value_target = rec["value"].as_f64()
                    .ok_or_else(|| anyhow::anyhow!("line {}: missing value", lineno + 1))? as f32;

                let policy_target: Vec<f32> = policy_raw.iter()
                    .map(|v| v.as_f64().unwrap_or(0.0) as f32)
                    .collect();

                // Op history applied to reach this state (may be empty for root).
                let op_history: Vec<String> = rec["op_history"]
                    .as_array()
                    .map(|arr| arr.iter()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect())
                    .unwrap_or_default();
                let op_refs: Vec<&str> = op_history.iter().map(|s| s.as_str()).collect();

                // Read all .rs files from the crate root.
                let root_path = std::path::Path::new(root);
                let files = read_rs_files(root_path)?;
                let file_refs: Vec<(&str, &str)> = files.iter()
                    .map(|(p, c)| (p.as_str(), c.as_str()))
                    .collect();

                let mut ids = tok.encode_state(&file_refs, &op_refs, max_len);
                // Pad to max_len.
                ids.resize(max_len, pad_id);

                let example = TrainExample { tokens: ids, policy_target, value_target };
                writeln!(outfile, "{}", serde_json::to_string(&example)?)?;
                count += 1;
            }

            eprintln!("{count} examples → {}", out.display());
        }

        Cmd::Train { data, out, lr, batch_size, epochs, value_lambda, small, n_ops, device, resume } => {
            let examples = load_jsonl(&data)?;
            eprintln!("Loaded {} training examples", examples.len());

            let mut cfg = if small { EncoderConfig::small() } else { EncoderConfig::default() };
            cfg.n_ops = resolve_n_ops(&n_ops, &examples)?;
            cfg.vocab_size = cfg.vocab_size.max(infer_vocab_size(&examples));
            warn_policy_lengths(&examples, cfg.n_ops);

            let train_cfg = TrainConfig {
                lr,
                batch_size,
                epochs,
                value_lambda,
                device: select_device(device)?,
            };
            let mut trainer = Trainer::new(cfg, train_cfg)?;
            if let Some(path) = resume {
                trainer.load(&path)
                    .with_context(|| format!("failed to load checkpoint {}", path.display()))?;
                eprintln!("Resumed from {}", path.display());
            }
            trainer.train(&examples)?;
            trainer.save(&out)?;
            eprintln!("Weights → {}", out.display());
        }
    }
    Ok(())
}

/// Read all `.rs` files under `root` (skipping `target/` and hidden dirs).
/// Returns Vec<(relative_path, content)>.
fn read_rs_files(root: &std::path::Path) -> Result<Vec<(String, String)>> {
    let mut files = Vec::new();
    for entry in walkdir::WalkDir::new(root).min_depth(1) {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy();
        if name == "target" || name.starts_with('.') { continue; }
        if entry.file_type().is_file()
            && entry.path().extension().and_then(|e| e.to_str()) == Some("rs")
        {
            let rel = entry.path().strip_prefix(root)
                .unwrap_or(entry.path())
                .to_string_lossy()
                .into_owned();
            let content = std::fs::read_to_string(entry.path())?;
            files.push((rel, content));
        }
    }
    Ok(files)
}

/// Concatenate all .rs files under `dir` for vocabulary building.
fn collect_corpus(dir: &std::path::Path) -> Result<String> {
    let mut corpus = String::new();
    for (_, content) in read_rs_files(dir)? {
        corpus.push_str(&content);
        corpus.push('\n');
    }
    Ok(corpus)
}

fn resolve_n_ops(value: &str, examples: &[TrainExample]) -> Result<usize> {
    if value.eq_ignore_ascii_case("auto") {
        return examples
            .iter()
            .map(|e| e.policy_target.len())
            .max()
            .filter(|n| *n > 0)
            .ok_or_else(|| anyhow::anyhow!("cannot infer --n-ops from empty policy targets"));
    }
    value
        .parse::<usize>()
        .with_context(|| format!("invalid --n-ops value `{value}`; use a positive integer or `auto`"))
}

fn warn_policy_lengths(examples: &[TrainExample], n_ops: usize) {
    let mut lengths: Vec<usize> = examples.iter().map(|e| e.policy_target.len()).collect();
    lengths.sort_unstable();
    lengths.dedup();
    if lengths.len() > 1 {
        eprintln!(
            "warn: inconsistent policy_target lengths {:?}; padding/truncating to n_ops={}",
            lengths, n_ops
        );
    } else if lengths.first().copied().is_some_and(|len| len != n_ops) {
        eprintln!(
            "warn: policy_target length {} differs from n_ops={}; padding/truncating",
            lengths[0], n_ops
        );
    }
}

fn infer_vocab_size(examples: &[TrainExample]) -> usize {
    examples
        .iter()
        .flat_map(|example| example.tokens.iter())
        .copied()
        .max()
        .map(|id| id as usize + 1)
        .unwrap_or(0)
}

fn select_device(device: DeviceArg) -> Result<Device> {
    match device {
        DeviceArg::Cpu => Ok(Device::Cpu),
        DeviceArg::Cuda => Ok(Device::new_cuda(0)?),
        DeviceArg::Metal => Ok(Device::new_metal(0)?),
        DeviceArg::Auto => {
            if candle_core::utils::cuda_is_available() {
                Ok(Device::new_cuda(0)?)
            } else if candle_core::utils::metal_is_available() {
                Ok(Device::new_metal(0)?)
            } else {
                Ok(Device::Cpu)
            }
        }
    }
}
