use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use clap::Parser;
use encoder::config::EncoderConfig;
use encoder::infer::EncoderInfer;

use search::op::Op;
use search::policy::{Candidate, CandidateGen, NoisyTemplatePolicy, Policy};
use search::search::{Mcts, MctsConfig, NodeRecord};
use search::state::RepoState;
use search::value::{
    ConstantValue, DepthValue, JudgementDeltaValue, Value, VerifierCache, VerifyMode, VerifyValue,
};

#[derive(Parser)]
#[command(about = "MCTS-based neurosymbolic code search")]
struct Cli {
    #[arg(long)]
    root: PathBuf,

    #[arg(long, default_value = "structural-editor")]
    editor_bin: PathBuf,

    #[arg(long, default_value_t = 50)]
    simulations: u32,

    #[arg(long, default_value_t = 8)]
    max_depth: usize,

    #[arg(long, default_value_t = 1.5)]
    c_puct: f32,

    /// Value function: stub | depth | check | test | composite | encoder | judgement-delta.
    #[arg(long, default_value = "stub")]
    value: String,

    /// Baseline judgement.jsonl snapshot taken before any edits.
    /// Required when --value judgement-delta.
    #[arg(long)]
    baseline_judgement: Option<PathBuf>,

    /// Path to the canon-rustc-v3 binary (used as RUSTC_WRAPPER).
    /// Required when --value judgement-delta.
    #[arg(long, default_value = "canon-rustc-v3")]
    canon_bin: PathBuf,

    /// Path to the judgement binary.
    /// Required when --value judgement-delta.
    #[arg(long, default_value = "judgement")]
    judgement_bin: PathBuf,

    /// Rustc crate target name, e.g. "ai". Determines the artifact subdir.
    /// Required when --value judgement-delta.
    #[arg(long)]
    crate_name: Option<String>,

    /// Static op template file (JSONL). Ignored when --dynamic is set.
    #[arg(long)]
    templates: Option<PathBuf>,

    /// Use CandidateGen: scan the crate and generate ops dynamically.
    /// Produces a crate-specific action space without needing ops.jsonl.
    #[arg(long)]
    dynamic: bool,

    /// Max candidates when --dynamic is used.
    #[arg(long, default_value_t = 64)]
    max_candidates: usize,

    #[arg(long, default_value_t = 5)]
    seq_len: usize,

    /// Write raw training records (JSONL) for every MCTS node visited.
    /// Each record: { root, op_history, policy, value }
    #[arg(long)]
    dump: Option<PathBuf>,

    /// Persistent cache for expensive verifier scores.
    #[arg(long)]
    value_cache: Option<PathBuf>,

    /// Path to trained encoder weights (safetensors). Enables neural policy priors.
    #[arg(long)]
    encoder_weights: Option<PathBuf>,

    /// Path to encoder vocab (JSON). Required when --encoder-weights is set.
    #[arg(long)]
    encoder_vocab: Option<PathBuf>,

    /// Use the encoder small preset (must match training config).
    #[arg(long)]
    encoder_small: bool,

    /// Policy-head width used when the encoder checkpoint was trained.
    #[arg(long)]
    encoder_n_ops: Option<usize>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let root = cli.root.canonicalize()?;
    let repo = RepoState::new(root);

    let config = MctsConfig {
        c_puct: cli.c_puct,
        n_simulations: cli.simulations,
        max_depth: cli.max_depth,
        editor_bin: cli.editor_bin.clone(),
    };

    macro_rules! run_with {
        ($policy:expr, $value:expr) => {{
            let mut mcts = Mcts::new(config, $policy, $value);
            run(&mut mcts, &repo, cli.seq_len, cli.dump.as_deref());
        }};
    }

    let policy_is_dynamic = cli.dynamic;
    let max_cands = cli.max_candidates;
    let templates = load_templates(cli.templates.as_deref())?;
    let encoder_n_ops = cli.encoder_n_ops.unwrap_or(if policy_is_dynamic {
        max_cands
    } else {
        templates.len()
    });
    let encoder = load_encoder(
        cli.encoder_weights.as_deref(),
        cli.encoder_vocab.as_deref(),
        cli.encoder_small,
        encoder_n_ops,
    )?;
    let cache_path = cli.value_cache.clone();

    match cli.value.as_str() {
        "stub" => {
            if let Some(infer) = encoder.clone() {
                run_with!(
                    EncPolicy::new(
                        infer,
                        policy_source(policy_is_dynamic, max_cands, templates)
                    ),
                    ConstantValue(0.5)
                );
            } else if policy_is_dynamic {
                run_with!(
                    CandidateGen {
                        max_candidates: max_cands
                    },
                    ConstantValue(0.5)
                );
            } else {
                run_with!(
                    NoisyTemplatePolicy {
                        templates,
                        alpha: 0.3
                    },
                    ConstantValue(0.5)
                );
            }
        }
        "depth" => {
            let v = DepthValue {
                max_depth: cli.max_depth,
            };
            if let Some(infer) = encoder.clone() {
                run_with!(
                    EncPolicy::new(
                        infer,
                        policy_source(policy_is_dynamic, max_cands, templates)
                    ),
                    v
                );
            } else if policy_is_dynamic {
                run_with!(
                    CandidateGen {
                        max_candidates: max_cands
                    },
                    v
                );
            } else {
                run_with!(
                    NoisyTemplatePolicy {
                        templates,
                        alpha: 0.3
                    },
                    v
                );
            }
        }
        "check" => {
            let v = verify_value(
                cli.editor_bin.clone(),
                VerifyMode::Check,
                cache_path.clone(),
            );
            if let Some(infer) = encoder.clone() {
                run_with!(
                    EncPolicy::new(
                        infer,
                        policy_source(policy_is_dynamic, max_cands, templates)
                    ),
                    v
                );
            } else if policy_is_dynamic {
                run_with!(
                    CandidateGen {
                        max_candidates: max_cands
                    },
                    v
                );
            } else {
                run_with!(
                    NoisyTemplatePolicy {
                        templates,
                        alpha: 0.3
                    },
                    v
                );
            }
        }
        "test" => {
            let v = verify_value(cli.editor_bin.clone(), VerifyMode::Test, cache_path.clone());
            if let Some(infer) = encoder.clone() {
                run_with!(
                    EncPolicy::new(
                        infer,
                        policy_source(policy_is_dynamic, max_cands, templates)
                    ),
                    v
                );
            } else if policy_is_dynamic {
                run_with!(
                    CandidateGen {
                        max_candidates: max_cands
                    },
                    v
                );
            } else {
                run_with!(
                    NoisyTemplatePolicy {
                        templates,
                        alpha: 0.3
                    },
                    v
                );
            }
        }
        "composite" => {
            let v = verify_value(
                cli.editor_bin.clone(),
                VerifyMode::Composite,
                cache_path.clone(),
            );
            if let Some(infer) = encoder.clone() {
                run_with!(
                    EncPolicy::new(
                        infer,
                        policy_source(policy_is_dynamic, max_cands, templates)
                    ),
                    v
                );
            } else if policy_is_dynamic {
                run_with!(
                    CandidateGen {
                        max_candidates: max_cands
                    },
                    v
                );
            } else {
                run_with!(
                    NoisyTemplatePolicy {
                        templates,
                        alpha: 0.3
                    },
                    v
                );
            }
        }
        "judgement-delta" => {
            let baseline = cli.baseline_judgement.clone().ok_or_else(|| {
                anyhow::anyhow!("--baseline-judgement required with --value judgement-delta")
            })?;
            let crate_name = cli.crate_name.clone().ok_or_else(|| {
                anyhow::anyhow!("--crate-name required with --value judgement-delta")
            })?;
            let v = judgement_delta_value(
                cli.editor_bin.clone(),
                baseline,
                cli.canon_bin.clone(),
                cli.judgement_bin.clone(),
                crate_name,
                cache_path.clone(),
            );
            if let Some(infer) = encoder.clone() {
                run_with!(
                    EncPolicy::new(
                        infer,
                        policy_source(policy_is_dynamic, max_cands, templates)
                    ),
                    v
                );
            } else if policy_is_dynamic {
                run_with!(
                    CandidateGen {
                        max_candidates: max_cands
                    },
                    v
                );
            } else {
                run_with!(
                    NoisyTemplatePolicy {
                        templates,
                        alpha: 0.3
                    },
                    v
                );
            }
        }
        "encoder" => {
            let Some(infer) = encoder else {
                anyhow::bail!("--value encoder requires --encoder-weights and --encoder-vocab");
            };
            run_with!(
                EncPolicy::new(
                    infer.clone(),
                    policy_source(policy_is_dynamic, max_cands, templates)
                ),
                EncValue { inner: infer }
            );
        }
        other => anyhow::bail!("unknown value function: {other}"),
    }

    Ok(())
}

fn verify_value(editor_bin: PathBuf, mode: VerifyMode, cache_path: Option<PathBuf>) -> VerifyValue {
    VerifyValue {
        editor_bin,
        mode,
        cache: cache_path.map(VerifierCache::load),
    }
}

fn judgement_delta_value(
    editor_bin: PathBuf,
    baseline_judgement: PathBuf,
    canon_bin: PathBuf,
    judgement_bin: PathBuf,
    crate_name: String,
    cache_path: Option<PathBuf>,
) -> JudgementDeltaValue {
    JudgementDeltaValue {
        editor_bin,
        baseline_judgement,
        canon_bin,
        judgement_bin,
        crate_name,
        cache: cache_path.map(VerifierCache::load),
    }
}

#[derive(Clone)]
enum PolicySource {
    Dynamic { max_candidates: usize },
    Static(Vec<Op>),
}

fn policy_source(dynamic: bool, max_candidates: usize, templates: Vec<Op>) -> PolicySource {
    if dynamic {
        PolicySource::Dynamic { max_candidates }
    } else {
        PolicySource::Static(templates)
    }
}

struct EncPolicy {
    inner: Arc<EncoderInfer>,
    source: PolicySource,
}

impl EncPolicy {
    fn new(inner: Arc<EncoderInfer>, source: PolicySource) -> Self {
        Self { inner, source }
    }
}

impl Policy for EncPolicy {
    fn propose(&self, state: &RepoState) -> Vec<Candidate> {
        let templates = match &self.source {
            PolicySource::Dynamic { max_candidates } => CandidateGen {
                max_candidates: *max_candidates,
            }
            .propose(state)
            .into_iter()
            .map(|c| c.op)
            .collect(),
            PolicySource::Static(templates) => templates.clone(),
        };
        if templates.is_empty() {
            return Vec::new();
        }

        let files_owned = read_state_files(&state.root);
        let files: Vec<(&str, &str)> = files_owned
            .iter()
            .map(|(path, content)| (path.as_str(), content.as_str()))
            .collect();
        let history: Vec<String> = state.ops.iter().map(|op| op.to_string()).collect();
        let history_refs: Vec<&str> = history.iter().map(|op| op.as_str()).collect();

        let priors = self
            .inner
            .run(&files, &history_refs, &templates)
            .map(|(priors, _)| priors)
            .unwrap_or_else(|err| {
                eprintln!("warn: encoder policy failed: {err:#}");
                vec![1.0 / templates.len() as f32; templates.len()]
            });

        let fallback = 1.0 / templates.len() as f32;
        templates
            .into_iter()
            .enumerate()
            .map(|(i, op)| Candidate {
                op,
                prior: priors.get(i).copied().unwrap_or(fallback),
            })
            .collect()
    }
}

struct EncValue {
    inner: Arc<EncoderInfer>,
}

impl Value for EncValue {
    fn score(&self, state: &RepoState) -> f32 {
        let files_owned = read_state_files(&state.root);
        let files: Vec<(&str, &str)> = files_owned
            .iter()
            .map(|(path, content)| (path.as_str(), content.as_str()))
            .collect();
        let history: Vec<String> = state.ops.iter().map(|op| op.to_string()).collect();
        let history_refs: Vec<&str> = history.iter().map(|op| op.as_str()).collect();

        self.inner
            .run(&files, &history_refs, &[])
            .map(|(_, value)| value)
            .unwrap_or_else(|err| {
                eprintln!("warn: encoder value failed: {err:#}");
                0.5
            })
    }
}

fn read_state_files(root: &std::path::Path) -> Vec<(String, String)> {
    let mut files = Vec::new();
    for entry in walkdir::WalkDir::new(root).min_depth(1) {
        let Ok(entry) = entry else { continue };
        let name = entry.file_name().to_string_lossy();
        if name == "target" || name.starts_with('.') {
            continue;
        }
        if !entry.file_type().is_file() {
            continue;
        }
        if entry.path().extension().and_then(|ext| ext.to_str()) != Some("rs") {
            continue;
        }
        let rel = entry
            .path()
            .strip_prefix(root)
            .unwrap_or(entry.path())
            .to_string_lossy()
            .replace('\\', "/");
        if let Ok(content) = std::fs::read_to_string(entry.path()) {
            files.push((rel, content));
        }
    }
    files
}

fn load_encoder(
    weights: Option<&std::path::Path>,
    vocab: Option<&std::path::Path>,
    small: bool,
    n_ops: usize,
) -> Result<Option<Arc<EncoderInfer>>> {
    match (weights, vocab) {
        (None, None) => Ok(None),
        (Some(_), None) => {
            anyhow::bail!("--encoder-vocab is required when --encoder-weights is set")
        }
        (None, Some(_)) => {
            anyhow::bail!("--encoder-weights is required when --encoder-vocab is set")
        }
        (Some(weights), Some(vocab)) => {
            let mut cfg = if small {
                EncoderConfig::small()
            } else {
                EncoderConfig::default()
            };
            cfg.n_ops = n_ops;
            Ok(Some(Arc::new(EncoderInfer::load(weights, vocab, cfg)?)))
        }
    }
}

fn run<P: search::policy::Policy, V: search::value::Value>(
    mcts: &mut Mcts<P, V>,
    root: &RepoState,
    seq_len: usize,
    dump: Option<&std::path::Path>,
) {
    let probs = mcts.run(root);
    let root_value = mcts.score(root);

    eprintln!("Action probabilities ({} actions):", probs.len());
    for (op, p) in &probs {
        eprintln!("  {:.4}  {}", p, op);
    }
    eprintln!(
        "Root value: {:.4}  |  tree nodes: {}",
        root_value,
        mcts.tree.len()
    );

    let seq = mcts.best_sequence(root, seq_len);
    println!("Best sequence ({} steps):", seq.len());
    for (i, op) in seq.iter().enumerate() {
        println!("  {i}: {op}");
    }

    if let Some(path) = dump {
        let records = mcts.node_records();
        eprintln!(
            "Dumping {} node records → {}",
            records.len(),
            path.display()
        );
        if let Err(e) = write_records(path, &records) {
            eprintln!("warn: dump failed: {e}");
        }
    }
}

fn write_records(path: &std::path::Path, records: &[NodeRecord]) -> Result<()> {
    use std::io::Write;
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    for rec in records {
        let op_history: Vec<String> = rec.state.ops.iter().map(|o| o.to_string()).collect();
        let policy: Vec<f32> = rec.policy.iter().map(|(_, p)| *p).collect();
        let record = serde_json::json!({
            "root":       rec.state.root.to_string_lossy(),
            "op_history": op_history,
            "policy":     policy,
            "value":      rec.value,
        });
        writeln!(file, "{record}")?;
    }
    Ok(())
}

fn load_templates(path: Option<&std::path::Path>) -> Result<Vec<search::op::Op>> {
    let Some(p) = path else {
        return Ok(Vec::new());
    };
    let content = std::fs::read_to_string(p)?;
    let mut ops = Vec::new();
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with("//") {
            continue;
        }
        ops.push(serde_json::from_str(line)?);
    }
    Ok(ops)
}
