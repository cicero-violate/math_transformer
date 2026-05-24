/// Hyperparameters for the encoder model.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EncoderConfig {
    /// Token vocabulary size (includes special tokens).
    pub vocab_size: usize,
    /// Maximum input sequence length (tokens).
    pub max_seq_len: usize,
    /// Number of transformer encoder layers.
    pub n_layers: usize,
    /// Model embedding dimension.
    pub d_model: usize,
    /// Number of attention heads. Must divide d_model.
    pub n_heads: usize,
    /// Feed-forward hidden dimension.
    pub d_ff: usize,
    /// Dropout probability (0.0 = disabled).
    pub dropout: f64,
    /// Size of the policy output (number of op templates in the action space).
    pub n_ops: usize,

    // ── Special token IDs ────────────────────────────────────────────────────
    pub pad_id: u32,
    pub cls_id: u32,
    pub sep_id: u32,
    pub unk_id: u32,
    /// Inserted between file contents in the input sequence.
    pub file_id: u32,
}

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            vocab_size:  8_192,
            max_seq_len: 2_048,
            n_layers:    6,
            d_model:     512,
            n_heads:     8,
            d_ff:        2_048,
            dropout:     0.1,
            n_ops:       64,
            pad_id:  0,
            cls_id:  1,
            sep_id:  2,
            unk_id:  3,
            file_id: 4,
        }
    }
}

impl EncoderConfig {
    pub fn head_dim(&self) -> usize {
        self.d_model / self.n_heads
    }

    pub fn small() -> Self {
        Self {
            max_seq_len: 512,
            n_layers:    4,
            d_model:     256,
            n_heads:     4,
            d_ff:        1_024,
            dropout:     0.1,
            n_ops:       16,
            ..Default::default()
        }
    }
}
