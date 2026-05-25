use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarMap};

use crate::config::EncoderConfig;
use crate::data::{collate, TrainExample};
use crate::model::EncoderModel;

pub struct TrainConfig {
    pub lr: f64,
    pub batch_size: usize,
    pub epochs: usize,
    pub device: Device,
    /// Weight on value loss relative to policy loss (total = policy + λ·value).
    pub value_lambda: f64,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            lr: 1e-4,
            batch_size: 32,
            epochs: 10,
            device: Device::Cpu,
            value_lambda: 1.0,
        }
    }
}

pub struct Trainer {
    pub model: EncoderModel,
    pub varmap: VarMap,
    pub cfg: EncoderConfig,
    pub train_cfg: TrainConfig,
}

impl Trainer {
    pub fn new(cfg: EncoderConfig, train_cfg: TrainConfig) -> Result<Self> {
        let varmap = VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &train_cfg.device);
        let model = EncoderModel::new(&cfg, vb)?;
        Ok(Self {
            model,
            varmap,
            cfg,
            train_cfg,
        })
    }

    /// Load an existing checkpoint into this trainer before training resumes.
    pub fn load(&mut self, path: &std::path::Path) -> Result<()> {
        self.varmap.load(path)?;
        Ok(())
    }

    pub fn train(&mut self, data: &[TrainExample]) -> Result<()> {
        let mut opt = AdamW::new(
            self.varmap.all_vars(),
            ParamsAdamW {
                lr: self.train_cfg.lr,
                ..Default::default()
            },
        )?;

        let n = data.len();
        let bs = self.train_cfg.batch_size;

        for epoch in 0..self.train_cfg.epochs {
            let mut total_loss = 0f32;
            let mut steps = 0usize;

            for start in (0..n).step_by(bs) {
                let end = (start + bs).min(n);
                let batch: Vec<&TrainExample> = data[start..end].iter().collect();

                let loss = self.step(&batch)?;
                opt.backward_step(&loss)?;

                total_loss += loss.to_scalar::<f32>()?;
                steps += 1;
            }

            eprintln!("epoch {epoch:>3}  loss={:.4}", total_loss / steps as f32);
        }
        Ok(())
    }

    fn step(&self, batch: &[&TrainExample]) -> Result<Tensor> {
        let dev = &self.train_cfg.device;
        let (tokens, mut policy_tgt, value_tgt) =
            collate(batch, self.cfg.max_seq_len, self.cfg.pad_id);
        for p in &mut policy_tgt {
            p.resize(self.cfg.n_ops, 0.0);
            p.truncate(self.cfg.n_ops);
        }

        // ── Build tensors ─────────────────────────────────────────────────────
        let flat_tok: Vec<u32> = tokens.into_iter().flatten().collect();
        let b = batch.len();
        let s = flat_tok.len() / b;
        let tok_t = Tensor::from_vec(flat_tok, (b, s), dev)?;

        let flat_pol: Vec<f32> = policy_tgt.into_iter().flatten().collect();
        let n_ops = self.cfg.n_ops;
        let pol_t = Tensor::from_vec(flat_pol, (b, n_ops), dev)?;

        let val_t = Tensor::from_vec(value_tgt, (b,), dev)?;

        // ── Forward ───────────────────────────────────────────────────────────
        let (policy_pred, value_pred) = self.model.forward(&tok_t)?;

        // ── Losses ────────────────────────────────────────────────────────────
        // Policy: cross-entropy  H(target, pred) = -Σ target * log(pred)
        let policy_loss = (pol_t * (policy_pred.log()? * -1.0)?)?
            .sum_all()?
            .affine(1.0 / b as f64, 0.0)?;

        // Value: MSE
        let diff = (value_pred - val_t)?;
        let value_loss = (&diff * &diff)?.mean_all()?;

        // Combined loss
        let lambda = self.train_cfg.value_lambda;
        (policy_loss + (value_loss * lambda)?).map_err(Into::into)
    }

    /// Save weights to a safetensors file.
    pub fn save(&self, path: &std::path::Path) -> Result<()> {
        self.varmap.save(path)?;
        Ok(())
    }
}
