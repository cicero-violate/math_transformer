use candle_core::{Result, Tensor};
use candle_nn::{ops::softmax, VarBuilder};

use crate::config::EncoderConfig;
use super::encoder::TransformerEncoder;
use super::heads::{PolicyHead, ValueHead};

/// Full dual-head encoder model.
///
/// Given a tokenised repo state it returns:
///   policy  — prior probability distribution over op templates  (batch, n_ops)
///   value   — estimated quality of the state                   (batch,)
pub struct EncoderModel {
    encoder: TransformerEncoder,
    policy:  PolicyHead,
    value:   ValueHead,
}

impl EncoderModel {
    pub fn new(cfg: &EncoderConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            encoder: TransformerEncoder::new(cfg, vb.pp("encoder"))?,
            policy:  PolicyHead::new(cfg.d_model, cfg.n_ops, vb.pp("policy"))?,
            value:   ValueHead::new(cfg.d_model, vb.pp("value"))?,
        })
    }

    /// token_ids: (batch, seq) → (policy_probs, value)
    pub fn forward(&self, token_ids: &Tensor) -> Result<(Tensor, Tensor)> {
        let cls = self.encoder.cls(token_ids)?;                     // (batch, d_model)
        let policy_logits = self.policy.forward(&cls)?;             // (batch, n_ops)
        let policy_probs  = softmax(&policy_logits, 1)?;            // (batch, n_ops)
        let value         = self.value.forward(&cls)?;              // (batch,)
        Ok((policy_probs, value))
    }
}
