use candle_core::{Result, Tensor, D};
use candle_nn::{linear_no_bias, Linear, Module, VarBuilder};

pub struct MultiHeadAttention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    n_heads:  usize,
    head_dim: usize,
}

impl MultiHeadAttention {
    pub fn new(d_model: usize, n_heads: usize, vb: VarBuilder) -> Result<Self> {
        let head_dim = d_model / n_heads;
        Ok(Self {
            q: linear_no_bias(d_model, d_model, vb.pp("q"))?,
            k: linear_no_bias(d_model, d_model, vb.pp("k"))?,
            v: linear_no_bias(d_model, d_model, vb.pp("v"))?,
            o: linear_no_bias(d_model, d_model, vb.pp("o"))?,
            n_heads,
            head_dim,
        })
    }

    /// xs: (batch, seq, d_model) → (batch, seq, d_model)
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, s, _) = xs.dims3()?;

        let split = |proj: &Linear| -> Result<Tensor> {
            let x = proj.forward(xs)?;                          // (b, s, d_model)
            let x = x.reshape((b, s, self.n_heads, self.head_dim))?;
            x.transpose(1, 2)?.contiguous()                     // (b, n_heads, s, head_dim)
        };

        let q = split(&self.q)?;
        let k = split(&self.k)?;
        let v = split(&self.v)?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let scores = q.matmul(&k.transpose(D::Minus1, D::Minus2)?)? // (b, h, s, s)
                      .affine(scale, 0.0)?;
        let attn = candle_nn::ops::softmax(&scores, D::Minus1)?;

        let ctx = attn.matmul(&v)?                              // (b, h, s, head_dim)
                      .transpose(1, 2)?.contiguous()?           // (b, s, h, head_dim)
                      .reshape((b, s, self.n_heads * self.head_dim))?; // (b, s, d_model)

        self.o.forward(&ctx)
    }
}
