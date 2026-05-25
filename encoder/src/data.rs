use anyhow::Result;
use serde::{Deserialize, Serialize};

/// One training example produced by an MCTS run.
///
/// Collected by running `search` with `--value composite` and recording:
///   tokens         — encoded repo state (from CodeTokenizer::encode_state)
///   policy_target  — MCTS visit-count distribution (length = n_ops)
///   value_target   — ground-truth score from the verifier ∈ [0, 1]
#[derive(Debug, Serialize, Deserialize)]
pub struct TrainExample {
    pub tokens: Vec<u32>,
    pub policy_target: Vec<f32>,
    pub value_target: f32,
}

/// Load a JSONL file of TrainExamples.
pub fn load_jsonl(path: &std::path::Path) -> Result<Vec<TrainExample>> {
    let text = std::fs::read_to_string(path)?;
    let mut examples = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with("//") {
            continue;
        }
        let ex: TrainExample =
            serde_json::from_str(line).map_err(|e| anyhow::anyhow!("line {}: {e}", i + 1))?;
        examples.push(ex);
    }
    Ok(examples)
}

/// Pad or truncate a token sequence to exactly `max_len`.
pub fn pad_tokens(tokens: &[u32], max_len: usize, pad_id: u32) -> Vec<u32> {
    if tokens.len() >= max_len {
        tokens[..max_len].to_vec()
    } else {
        let mut v = tokens.to_vec();
        v.resize(max_len, pad_id);
        v
    }
}

/// Collate a batch of examples into (token_ids, policy_targets, value_targets).
/// All sequences are padded to the length of the longest in the batch.
pub fn collate(
    examples: &[&TrainExample],
    max_len: usize,
    pad_id: u32,
) -> (Vec<Vec<u32>>, Vec<Vec<f32>>, Vec<f32>) {
    let seq_len = examples
        .iter()
        .map(|e| e.tokens.len())
        .max()
        .unwrap_or(0)
        .min(max_len);
    let tokens: Vec<Vec<u32>> = examples
        .iter()
        .map(|e| pad_tokens(&e.tokens, seq_len, pad_id))
        .collect();
    let max_policy_len = examples
        .iter()
        .map(|e| e.policy_target.len())
        .max()
        .unwrap_or(0);
    let policy: Vec<Vec<f32>> = examples
        .iter()
        .map(|e| {
            let mut p = e.policy_target.clone();
            p.resize(max_policy_len, 0.0);
            p
        })
        .collect();
    let value: Vec<f32> = examples.iter().map(|e| e.value_target).collect();
    (tokens, policy, value)
}
