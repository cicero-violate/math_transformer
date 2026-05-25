use candle_core::{Result, Tensor};
use candle_nn::{layer_norm, linear_no_bias, LayerNorm, Linear, Module, VarBuilder};

use super::attention::MultiHeadAttention;

/// Pre-norm transformer encoder layer.
///   x → norm1 → MHA → residual → norm2 → FFN → residual
pub struct EncoderLayer {
    attn: MultiHeadAttention,
    norm1: LayerNorm,
    ff1: Linear,
    ff2: Linear,
    norm2: LayerNorm,
}

impl EncoderLayer {
    pub fn new(d_model: usize, n_heads: usize, d_ff: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            attn: MultiHeadAttention::new(d_model, n_heads, vb.pp("attn"))?,
            norm1: layer_norm(d_model, 1e-5, vb.pp("norm1"))?,
            ff1: linear_no_bias(d_model, d_ff, vb.pp("ff1"))?,
            ff2: linear_no_bias(d_ff, d_model, vb.pp("ff2"))?,
            norm2: layer_norm(d_model, 1e-5, vb.pp("norm2"))?,
        })
    }

    /// xs: (batch, seq, d_model) → (batch, seq, d_model)
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // MHA sublayer
        let residual = xs;
        let xs = self.norm1.forward(xs)?;
        let xs = (self.attn.forward(&xs)? + residual)?;

        // FFN sublayer
        let residual = &xs;
        let xs = self.norm2.forward(&xs)?;
        let xs = self.ff1.forward(&xs)?.gelu()?;
        let xs = self.ff2.forward(&xs)?;
        xs + residual
    }
}
