use candle_core::{IndexOp, Module, Result, Tensor};
use candle_nn::{embedding, layer_norm, Embedding, LayerNorm, VarBuilder};

use super::layer::EncoderLayer;
use crate::config::EncoderConfig;

/// Transformer encoder backbone.
///
/// Input  : token IDs  (batch, seq)
/// Output : contextualised embeddings (batch, seq, d_model)
///          The [CLS] token at position 0 is used as the pooled representation.
pub struct TransformerEncoder {
    tok_embed: Embedding,
    pos_embed: Embedding,
    layers: Vec<EncoderLayer>,
    norm: LayerNorm,
    max_seq: usize,
}

impl TransformerEncoder {
    pub fn new(cfg: &EncoderConfig, vb: VarBuilder) -> Result<Self> {
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            layers.push(EncoderLayer::new(
                cfg.d_model,
                cfg.n_heads,
                cfg.d_ff,
                vb.pp(format!("layer_{i}")),
            )?);
        }
        Ok(Self {
            tok_embed: embedding(cfg.vocab_size, cfg.d_model, vb.pp("tok_embed"))?,
            pos_embed: embedding(cfg.max_seq_len, cfg.d_model, vb.pp("pos_embed"))?,
            layers,
            norm: layer_norm(cfg.d_model, 1e-5, vb.pp("norm"))?,
            max_seq: cfg.max_seq_len,
        })
    }

    /// Returns (batch, seq, d_model).
    pub fn forward(&self, token_ids: &Tensor) -> Result<Tensor> {
        let (_, seq) = token_ids.dims2()?;
        let seq = seq.min(self.max_seq);

        let token_ids = token_ids.i((.., ..seq))?;
        let positions = Tensor::arange(0u32, seq as u32, token_ids.device())?
            .unsqueeze(0)? // (1, seq)
            .broadcast_as(token_ids.shape())?;

        let mut x = (self.tok_embed.forward(&token_ids)? + self.pos_embed.forward(&positions)?)?;

        for layer in &self.layers {
            x = layer.forward(&x)?;
        }
        self.norm.forward(&x)
    }

    /// Returns the [CLS] embedding (batch, d_model).
    pub fn cls(&self, token_ids: &Tensor) -> Result<Tensor> {
        self.forward(token_ids)?.i((.., 0, ..))
    }
}
