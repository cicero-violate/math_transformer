use candle_core::{Result, Tensor};
use candle_nn::{linear, linear_no_bias, Linear, Module, VarBuilder};

/// Policy head: [CLS] → logits over op template vocabulary.
/// Apply softmax externally to get priors.
pub struct PolicyHead {
    proj: Linear,
}

impl PolicyHead {
    pub fn new(d_model: usize, n_ops: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            proj: linear(d_model, n_ops, vb.pp("proj"))?,
        })
    }

    /// cls: (batch, d_model) → logits (batch, n_ops)
    pub fn forward(&self, cls: &Tensor) -> Result<Tensor> {
        self.proj.forward(cls)
    }
}

/// Value head: [CLS] → scalar ∈ (0, 1).
pub struct ValueHead {
    fc1: Linear,
    fc2: Linear,
}

impl ValueHead {
    pub fn new(d_model: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            fc1: linear_no_bias(d_model, d_model / 2, vb.pp("fc1"))?,
            fc2: linear(d_model / 2, 1, vb.pp("fc2"))?,
        })
    }

    /// cls: (batch, d_model) → scalar (batch,)
    pub fn forward(&self, cls: &Tensor) -> Result<Tensor> {
        let h = self.fc1.forward(cls)?.relu()?;
        candle_nn::ops::sigmoid(&self.fc2.forward(&h)?)?.squeeze(1) // (batch,)
    }
}
