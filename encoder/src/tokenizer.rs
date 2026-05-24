use std::collections::HashMap;

// ── Special token IDs (must match EncoderConfig) ──────────────────────────────
pub const PAD_ID:  u32 = 0;
pub const CLS_ID:  u32 = 1;
pub const SEP_ID:  u32 = 2;
pub const UNK_ID:  u32 = 3;
pub const FILE_ID: u32 = 4;
const FIRST_NORMAL: u32 = 5;

/// Minimal code tokeniser.
///
/// Splits on whitespace and non-identifier characters, keeping each piece as a
/// token. Unknown tokens map to UNK_ID. The vocabulary is built from a corpus
/// via `build_vocab` or loaded from a saved JSON map.
///
/// For production, replace with a proper BPE tokeniser (e.g. the `tokenizers`
/// crate) pre-trained on Rust source.
pub struct CodeTokenizer {
    vocab:     HashMap<String, u32>,
    inv_vocab: Vec<String>,
}

impl CodeTokenizer {
    /// Load from a `{ token: id }` JSON map.
    pub fn from_json(json: &str) -> anyhow::Result<Self> {
        let vocab: HashMap<String, u32> = serde_json::from_str(json)?;
        let mut inv_vocab = vec![String::new(); vocab.len() + FIRST_NORMAL as usize];
        for (tok, &id) in &vocab {
            if id as usize >= inv_vocab.len() {
                inv_vocab.resize(id as usize + 1, String::new());
            }
            inv_vocab[id as usize] = tok.clone();
        }
        Ok(Self { vocab, inv_vocab })
    }

    /// Build a vocabulary from raw source text, keeping the `max_vocab` most
    /// frequent tokens. Special tokens are pre-assigned IDs 0–4.
    pub fn build_vocab(corpus: &str, max_vocab: usize) -> Self {
        let mut freq: HashMap<String, u32> = HashMap::new();
        for tok in tokenize_raw(corpus) {
            *freq.entry(tok).or_default() += 1;
        }

        let mut pairs: Vec<(String, u32)> = freq.into_iter().collect();
        pairs.sort_by(|a, b| b.1.cmp(&a.1));

        let mut vocab: HashMap<String, u32> = HashMap::new();
        let mut inv_vocab: Vec<String> = vec![
            "<PAD>".into(), "<CLS>".into(), "<SEP>".into(), "<UNK>".into(), "<FILE>".into(),
        ];

        for (tok, _) in pairs.into_iter().take(max_vocab - FIRST_NORMAL as usize) {
            let id = inv_vocab.len() as u32;
            vocab.insert(tok.clone(), id);
            inv_vocab.push(tok);
        }
        Self { vocab, inv_vocab }
    }

    /// Encode one source text into token IDs (no CLS/SEP — caller adds those).
    pub fn encode(&self, text: &str) -> Vec<u32> {
        tokenize_raw(text)
            .map(|t| self.vocab.get(&t).copied().unwrap_or(UNK_ID))
            .collect()
    }

    /// Encode a full repo state for the model.
    ///
    /// Format: [CLS] [FILE] <file1-tokens> [SEP] [FILE] <file2-tokens> [SEP] ...
    ///         [SEP] <op-history-tokens>
    pub fn encode_state(
        &self,
        files: &[(&str, &str)],   // (path, content)
        op_history: &[&str],       // JSON strings of applied ops
        max_len: usize,
    ) -> Vec<u32> {
        let mut ids = vec![CLS_ID];

        for (path, content) in files {
            ids.push(FILE_ID);
            ids.extend(self.encode(path));
            ids.push(SEP_ID);
            ids.extend(self.encode(content));
            if ids.len() >= max_len { break; }
        }

        if !op_history.is_empty() {
            ids.push(SEP_ID);
            for op in op_history {
                ids.extend(self.encode(op));
            }
        }

        ids.truncate(max_len);
        ids
    }

    pub fn vocab_size(&self) -> usize {
        self.inv_vocab.len()
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string(&self.vocab).unwrap()
    }
}

fn tokenize_raw(text: &str) -> impl Iterator<Item = String> + '_ {
    // Split on non-identifier characters, keeping the separators as tokens too.
    let mut out = Vec::new();
    let mut buf = String::new();
    for ch in text.chars() {
        if ch.is_alphanumeric() || ch == '_' {
            buf.push(ch);
        } else {
            if !buf.is_empty() {
                out.push(std::mem::take(&mut buf));
            }
            if !ch.is_whitespace() {
                out.push(ch.to_string());
            }
        }
    }
    if !buf.is_empty() { out.push(buf); }
    out.into_iter()
}
