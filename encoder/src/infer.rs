use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use serde_json::Value as Op;

use crate::config::EncoderConfig;
use crate::model::EncoderModel;
use crate::tokenizer::CodeTokenizer;

/// Loaded model ready for inference. Implements the interfaces expected by the
/// `search` crate's `Policy` and `Value` traits (kept as plain methods here to
/// avoid a hard dependency on that crate).
pub struct EncoderInfer {
    model:     EncoderModel,
    tokenizer: CodeTokenizer,
    cfg:       EncoderConfig,
    device:    Device,
}

impl EncoderInfer {
    /// Load from a safetensors weights file + a JSON vocab file.
    pub fn load(
        weights_path: &std::path::Path,
        vocab_path: &std::path::Path,
        cfg: EncoderConfig,
    ) -> Result<Self> {
        Self::load_on_device(weights_path, vocab_path, cfg, Device::Cpu)
    }

    /// Load from a safetensors weights file + a JSON vocab file on a caller
    /// selected device.
    pub fn load_on_device(
        weights_path: &std::path::Path,
        vocab_path: &std::path::Path,
        cfg: EncoderConfig,
        device: Device,
    ) -> Result<Self> {
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)?
        };
        let model = EncoderModel::new(&cfg, vb)?;
        let vocab_json = std::fs::read_to_string(vocab_path)?;
        let tokenizer = CodeTokenizer::from_json(&vocab_json)?;
        Ok(Self { model, tokenizer, cfg, device })
    }

    /// Given the current repo state (list of (path, content) pairs and the
    /// JSON-serialised op history), return prior probabilities over `templates`
    /// and a value estimate ∈ [0, 1].
    pub fn run(
        &self,
        files: &[(&str, &str)],
        op_history: &[&str],
        templates: &[Op],
    ) -> Result<(Vec<f32>, f32)> {
        let ids = self.tokenizer.encode_state(files, op_history, self.cfg.max_seq_len);
        let tok_t = Tensor::from_vec(ids, (1, self.cfg.max_seq_len), &self.device)?;

        let (policy_probs, value_t) = self.model.forward(&tok_t)?;

        // policy_probs: (1, n_ops) — take first row, slice to actual template count
        let probs_vec: Vec<f32> = policy_probs.get(0)?.to_vec1()?;
        let n = templates.len().min(probs_vec.len());
        let priors = probs_vec[..n].to_vec();

        let value = value_t.get(0)?.to_scalar::<f32>()?;

        Ok((priors, value))
    }
}

// ── Adapter structs for search::policy::Policy and search::value::Value ──────
//
// To wire into the search crate without a hard dependency:
//
//   use encoder::infer::EncoderInfer;
//   use search::policy::{Policy, Candidate};
//   use search::value::Value;
//   use search::state::RepoState;
//
//   struct EncPolicy { inner: Arc<EncoderInfer>, templates: Vec<Op> }
//   impl Policy for EncPolicy {
//       fn propose(&self, state: &RepoState) -> Vec<Candidate> {
//           let files = read_state_files(state);   // read file contents
//           let history: Vec<String> = state.ops.iter().map(|o| o.to_string()).collect();
//           let history_refs: Vec<&str> = history.iter().map(|s| s.as_str()).collect();
//           let (priors, _) = self.inner.run(&files, &history_refs, &self.templates).unwrap();
//           self.templates.iter().zip(priors)
//               .map(|(op, p)| Candidate { op: op.clone(), prior: p })
//               .collect()
//       }
//   }
//
//   struct EncValue { inner: Arc<EncoderInfer> }
//   impl Value for EncValue {
//       fn score(&self, state: &RepoState) -> f32 {
//           let files = read_state_files(state);
//           let history: Vec<String> = state.ops.iter().map(|o| o.to_string()).collect();
//           let history_refs: Vec<&str> = history.iter().map(|s| s.as_str()).collect();
//           let (_, v) = self.inner.run(&files, &history_refs, &[]).unwrap();
//           v
//       }
//   }
