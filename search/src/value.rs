use crate::sandbox;
use crate::state::RepoState;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::sync::Mutex;

/// Value function interface. Given a state, return a score ∈ [0, 1].
/// 0 = clearly wrong/broken, 1 = correct/ideal.
/// The transformer value head implements this; stubs are provided for dev.
pub trait Value: Send + Sync {
    fn score(&self, state: &RepoState) -> f32;
}

// ── Stubs ─────────────────────────────────────────────────────────────────────

/// Value via symbolic verification: apply ops to sandbox, run cargo check/test.
/// This is the ground-truth oracle — expensive but correct.
pub struct VerifyValue {
    pub editor_bin: PathBuf,
    pub mode: VerifyMode,
    pub cache: Option<VerifierCache>,
}

#[derive(Clone, Copy, Debug, Hash)]
pub enum VerifyMode {
    /// cargo check only (fast, binary score)
    Check,
    /// cargo test (slower, fractional score)
    Test,
    /// cargo check gate, then cargo test ratio
    Composite,
}

impl Value for VerifyValue {
    fn score(&self, state: &RepoState) -> f32 {
        let key = verify_key(state, self.mode);
        if let Some(cache) = &self.cache {
            if let Some(score) = cache.get(&key) {
                return score;
            }
        }

        match sandbox::apply_and_score(state, &self.editor_bin, self.mode) {
            Ok(s) => {
                if let Some(cache) = &self.cache {
                    cache.insert(key, s);
                }
                s
            }
            Err(e) => { eprintln!("[sandbox] {e:#}"); 0.0 }
        }
    }
}

pub struct VerifierCache {
    path: PathBuf,
    scores: Mutex<HashMap<String, f32>>,
}

impl VerifierCache {
    pub fn load(path: PathBuf) -> Self {
        let mut scores = HashMap::new();
        if let Ok(text) = std::fs::read_to_string(&path) {
            for line in text.lines() {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                if let Ok(entry) = serde_json::from_str::<CacheEntry>(line) {
                    scores.insert(entry.key, entry.score);
                }
            }
        }
        Self { path, scores: Mutex::new(scores) }
    }

    pub fn get(&self, key: &str) -> Option<f32> {
        self.scores.lock().ok()?.get(key).copied()
    }

    pub fn insert(&self, key: String, score: f32) {
        let Ok(mut scores) = self.scores.lock() else {
            return;
        };
        if scores.insert(key.clone(), score).is_some() {
            return;
        }
        drop(scores);

        let entry = CacheEntry { key, score };
        let Ok(line) = serde_json::to_string(&entry) else {
            return;
        };
        if let Some(parent) = self.path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Ok(mut file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
        {
            use std::io::Write;
            let _ = writeln!(file, "{line}");
        }
    }
}

#[derive(Serialize, Deserialize)]
struct CacheEntry {
    key: String,
    score: f32,
}

fn verify_key(state: &RepoState, mode: VerifyMode) -> String {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    mode.hash(&mut h);
    workspace_content_fingerprint(state).hash(&mut h);
    for op in &state.ops {
        op.to_string().hash(&mut h);
    }
    format!("{:016x}", h.finish())
}

fn workspace_content_fingerprint(state: &RepoState) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    for entry in walkdir::WalkDir::new(&state.root).min_depth(1) {
        let Ok(entry) = entry else { continue };
        let name = entry.file_name().to_string_lossy();
        if name == "target" || name.starts_with('.') {
            continue;
        }
        if !entry.file_type().is_file() {
            continue;
        }
        let path = entry.path();
        let is_relevant = matches!(
            path.extension().and_then(|ext| ext.to_str()),
            Some("rs") | Some("toml") | Some("lock")
        );
        if !is_relevant {
            continue;
        }
        let rel = path.strip_prefix(&state.root).unwrap_or(path);
        rel.hash(&mut h);
        if let Ok(bytes) = std::fs::read(path) {
            bytes.hash(&mut h);
        }
    }
    h.finish()
}

/// Constant value stub — every state scores 0.5 (useful for testing MCTS plumbing).
pub struct ConstantValue(pub f32);

impl Value for ConstantValue {
    fn score(&self, _state: &RepoState) -> f32 {
        self.0
    }
}

/// Heuristic: score by depth (deeper = better exploration, score decays).
/// Useful as a fast proxy before the neural value head is wired in.
pub struct DepthValue {
    pub max_depth: usize,
}

impl Value for DepthValue {
    fn score(&self, state: &RepoState) -> f32 {
        (state.depth() as f32 / self.max_depth as f32).min(1.0)
    }
}
