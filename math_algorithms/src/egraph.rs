use std::collections::HashMap;

use crate::disjoint_set::DisjointSet;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Expr {
    Const(i64),
    Add(usize, usize),
    Mul(usize, usize),
}

#[derive(Debug, Clone)]
pub struct EGraph {
    nodes: Vec<Expr>,
    classes: DisjointSet,
    memo: HashMap<Expr, usize>,
}

impl EGraph {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            classes: DisjointSet::new(0),
            memo: HashMap::new(),
        }
    }

    pub fn add(&mut self, expr: Expr) -> usize {
        if let Some(&id) = self.memo.get(&expr) {
            return id;
        }
        let id = self.nodes.len();
        self.nodes.push(expr.clone());
        let mut next = DisjointSet::new(id + 1);
        for i in 0..id {
            let root = self.classes.find(i);
            next.union(i, root);
        }
        self.classes = next;
        self.memo.insert(expr, id);
        id
    }

    pub fn union(&mut self, a: usize, b: usize) {
        self.classes.union(a, b);
    }

    pub fn equivalent(&mut self, a: usize, b: usize) -> bool {
        self.classes.find(a) == self.classes.find(b)
    }

    pub fn saturate_commutativity(&mut self) {
        let snapshot = self.nodes.clone();
        for (id, expr) in snapshot.into_iter().enumerate() {
            match expr {
                Expr::Add(a, b) => {
                    let swapped = self.add(Expr::Add(b, a));
                    self.union(id, swapped);
                }
                Expr::Mul(a, b) => {
                    let swapped = self.add(Expr::Mul(b, a));
                    self.union(id, swapped);
                }
                Expr::Const(_) => {}
            }
        }
    }
}

impl Default for EGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equality_saturation_adds_commuted_form() {
        let mut eg = EGraph::new();
        let a = eg.add(Expr::Const(1));
        let b = eg.add(Expr::Const(2));
        let ab = eg.add(Expr::Add(a, b));
        eg.saturate_commutativity();
        let ba = eg.add(Expr::Add(b, a));
        assert!(eg.equivalent(ab, ba));
    }
}
