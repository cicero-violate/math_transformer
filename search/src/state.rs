use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;
use std::path::PathBuf;

use crate::op::Op;

/// The search state: a project root plus the sequence of ops applied to reach
/// this point. Applying ops is lazy — the sandbox reconstructs the file system
/// when a score is needed.
#[derive(Debug, Clone)]
pub struct RepoState {
    /// Absolute path to the project being edited.
    pub root: PathBuf,
    /// Ops applied from the root in order.
    pub ops: Vec<Op>,
}

impl RepoState {
    pub fn new(root: PathBuf) -> Self {
        Self { root, ops: Vec::new() }
    }

    /// Return a child state with `op` appended.
    pub fn apply(&self, op: Op) -> Self {
        let mut next = self.clone();
        next.ops.push(op);
        next
    }

    pub fn depth(&self) -> usize {
        self.ops.len()
    }

    /// Stable hash over (root, op sequence). Used as the transposition table key.
    pub fn fingerprint(&self) -> u64 {
        let mut h = DefaultHasher::new();
        self.root.hash(&mut h);
        for op in &self.ops {
            op.to_string().hash(&mut h);
        }
        h.finish()
    }
}
