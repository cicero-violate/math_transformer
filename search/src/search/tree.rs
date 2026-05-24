use std::collections::HashMap;
use crate::op::Op;

// ── Edge ──────────────────────────────────────────────────────────────────────

/// An edge in the search tree: one action and its accumulated statistics.
#[derive(Debug, Clone)]
pub struct Edge {
    /// The op this edge represents.
    pub op: Op,
    /// Fingerprint of the child state (None until first visit).
    pub child_fp: Option<u64>,
    /// Visit count N(s, a).
    pub n: u32,
    /// Total value W(s, a).
    pub w: f32,
    /// Prior probability P(s, a) from the policy network.
    pub p: f32,
}

impl Edge {
    /// Mean value Q(s, a) = W / N.
    pub fn q(&self) -> f32 {
        if self.n == 0 { 0.0 } else { self.w / self.n as f32 }
    }

    /// PUCT score used for selection.
    /// score = Q(s,a) + c_puct · P(s,a) · √N(s) / (1 + N(s,a))
    pub fn puct(&self, c_puct: f32, sqrt_parent_n: f32) -> f32 {
        self.q() + c_puct * self.p * sqrt_parent_n / (1.0 + self.n as f32)
    }
}

// ── Node ──────────────────────────────────────────────────────────────────────

/// A node in the search tree: one state and its outgoing edges.
#[derive(Debug, Clone)]
pub struct Node {
    /// Outgoing edges, one per action proposed by the policy.
    pub edges: Vec<Edge>,
    /// Sum of all edge visit counts = N(s).
    pub total_n: u32,
    /// True once the policy has been called and edges populated.
    pub is_expanded: bool,
    /// True if the policy returned no candidates (terminal state).
    pub is_terminal: bool,
}

impl Node {
    pub fn new() -> Self {
        Self { edges: Vec::new(), total_n: 0, is_expanded: false, is_terminal: false }
    }

    /// Index of the edge with the highest PUCT score.
    pub fn best_edge(&self, c_puct: f32) -> usize {
        let sqrt_n = (self.total_n as f32).sqrt();
        self.edges
            .iter()
            .enumerate()
            .map(|(i, e)| (i, e.puct(c_puct, sqrt_n)))
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Index of the edge with the highest visit count (used for best-path extraction).
    pub fn most_visited(&self) -> Option<usize> {
        self.edges
            .iter()
            .enumerate()
            .max_by_key(|(_, e)| e.n)
            .filter(|(_, e)| e.n > 0)
            .map(|(i, _)| i)
    }
}

// ── SearchTree ────────────────────────────────────────────────────────────────

/// Transposition table: maps state fingerprints to nodes.
/// States that produce the same fingerprint share statistics (transposition).
pub struct SearchTree {
    nodes: HashMap<u64, Node>,
}

impl SearchTree {
    pub fn new() -> Self {
        Self { nodes: HashMap::new() }
    }

    pub fn get(&self, fp: u64) -> Option<&Node> {
        self.nodes.get(&fp)
    }

    pub fn get_mut(&mut self, fp: u64) -> Option<&mut Node> {
        self.nodes.get_mut(&fp)
    }

    pub fn get_or_insert(&mut self, fp: u64) -> &mut Node {
        self.nodes.entry(fp).or_insert_with(Node::new)
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Iterate over all expanded (non-terminal) nodes.
    pub fn all_expanded(&self) -> impl Iterator<Item = (&u64, &Node)> {
        self.nodes.iter().filter(|(_, n)| n.is_expanded && !n.is_terminal)
    }
}
