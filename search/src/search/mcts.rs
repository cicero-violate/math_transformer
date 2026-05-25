use std::collections::HashMap;
use std::path::PathBuf;

use crate::op::Op;
use crate::policy::Policy;
use crate::search::tree::{Edge, SearchTree};
use crate::state::RepoState;
use crate::value::Value;

// ── Config ────────────────────────────────────────────────────────────────────

pub struct MctsConfig {
    pub c_puct: f32,
    pub n_simulations: u32,
    pub max_depth: usize,
    pub editor_bin: PathBuf,
}

impl Default for MctsConfig {
    fn default() -> Self {
        Self {
            c_puct: 1.5,
            n_simulations: 50,
            max_depth: 8,
            editor_bin: PathBuf::from("structural-editor"),
        }
    }
}

// ── Training record for one MCTS node ────────────────────────────────────────

/// One training example extracted from the search tree after a run.
///
/// Every expanded node contributes one record:
///   policy — MCTS-improved visit-count distribution (the AlphaZero training target)
///   value  — mean backed-up value W(s)/N(s) across all simulations through this node
pub struct NodeRecord {
    pub state: RepoState,
    pub policy: Vec<(Op, f32)>,
    pub value: f32,
}

// ── MCTS ──────────────────────────────────────────────────────────────────────

pub struct Mcts<P, V> {
    pub config: MctsConfig,
    pub policy: P,
    pub value: V,
    pub tree: SearchTree,
    /// Maps fingerprint → RepoState for every node visited during search.
    state_cache: HashMap<u64, RepoState>,
}

impl<P: Policy, V: Value> Mcts<P, V> {
    pub fn new(config: MctsConfig, policy: P, value: V) -> Self {
        Self {
            config,
            policy,
            value,
            tree: SearchTree::new(),
            state_cache: HashMap::new(),
        }
    }

    /// Run `n_simulations` from `root`. Returns MCTS-improved action probabilities.
    pub fn run(&mut self, root: &RepoState) -> Vec<(Op, f32)> {
        let root_fp = root.fingerprint();
        self.tree.get_or_insert(root_fp);
        self.state_cache
            .entry(root_fp)
            .or_insert_with(|| root.clone());

        for _ in 0..self.config.n_simulations {
            let mut path: Vec<(u64, usize)> = Vec::new();
            let v = self.simulate(root, &mut path, 0);
            self.backup(&path, v);
        }

        self.action_probs(root_fp)
    }

    /// Score `state` with the value function (ground-truth oracle for training).
    pub fn score(&self, state: &RepoState) -> f32 {
        self.value.score(state)
    }

    /// Return the best op sequence by greedily following the most-visited edge.
    pub fn best_sequence(&self, root: &RepoState, max_len: usize) -> Vec<Op> {
        let mut seq = Vec::new();
        let mut state = root.clone();
        for _ in 0..max_len {
            let fp = state.fingerprint();
            let node = match self.tree.get(fp) {
                Some(n) if n.is_expanded => n,
                _ => break,
            };
            match node.most_visited() {
                Some(idx) => {
                    let op = node.edges[idx].op.clone();
                    state = state.apply(op.clone());
                    seq.push(op);
                }
                None => break,
            }
        }
        seq
    }

    /// Extract one training record per expanded node.
    ///
    /// Call after `run()`. Each record holds the MCTS-improved policy (visit
    /// counts normalised) and the mean backed-up value for that node.
    pub fn node_records(&self) -> Vec<NodeRecord> {
        self.tree
            .all_expanded()
            .filter_map(|(fp, node)| {
                let state = self.state_cache.get(fp)?;
                let total_w: f32 = node.edges.iter().map(|e| e.w).sum();
                let value = if node.total_n > 0 {
                    total_w / node.total_n as f32
                } else {
                    0.0
                };
                let policy = node
                    .edges
                    .iter()
                    .map(|e| (e.op.clone(), e.n as f32 / node.total_n.max(1) as f32))
                    .collect();
                Some(NodeRecord {
                    state: state.clone(),
                    policy,
                    value,
                })
            })
            .collect()
    }

    // ── Private ───────────────────────────────────────────────────────────────

    fn simulate(&mut self, state: &RepoState, path: &mut Vec<(u64, usize)>, depth: usize) -> f32 {
        let fp = state.fingerprint();
        // Cache every visited state for trace extraction.
        self.state_cache.entry(fp).or_insert_with(|| state.clone());

        if depth >= self.config.max_depth {
            return self.value.score(state);
        }

        let is_expanded = self.tree.get(fp).map(|n| n.is_expanded).unwrap_or(false);
        let is_terminal = self.tree.get(fp).map(|n| n.is_terminal).unwrap_or(false);

        if is_terminal {
            return self.value.score(state);
        }

        if !is_expanded {
            return self.expand_and_eval(state, fp);
        }

        let edge_idx = self.tree.get(fp).unwrap().best_edge(self.config.c_puct);
        let op = self.tree.get(fp).unwrap().edges[edge_idx].op.clone();
        path.push((fp, edge_idx));
        let next = state.apply(op);
        self.simulate(&next, path, depth + 1)
    }

    fn expand_and_eval(&mut self, state: &RepoState, fp: u64) -> f32 {
        let candidates = self.policy.propose(state);
        let edges: Vec<Edge> = candidates
            .into_iter()
            .map(|c| Edge {
                op: c.op,
                child_fp: None,
                n: 0,
                w: 0.0,
                p: c.prior,
            })
            .collect();
        let is_terminal = edges.is_empty();
        let node = self.tree.get_or_insert(fp);
        node.edges = edges;
        node.is_expanded = true;
        node.is_terminal = is_terminal;
        self.value.score(state)
    }

    fn backup(&mut self, path: &[(u64, usize)], value: f32) {
        for &(node_fp, edge_idx) in path.iter().rev() {
            if let Some(node) = self.tree.get_mut(node_fp) {
                node.total_n += 1;
                if let Some(edge) = node.edges.get_mut(edge_idx) {
                    edge.n += 1;
                    edge.w += value;
                }
            }
        }
    }

    fn action_probs(&self, root_fp: u64) -> Vec<(Op, f32)> {
        let node = match self.tree.get(root_fp) {
            Some(n) => n,
            None => return Vec::new(),
        };
        let total = node.total_n.max(1) as f32;
        node.edges
            .iter()
            .map(|e| (e.op.clone(), e.n as f32 / total))
            .collect()
    }
}
