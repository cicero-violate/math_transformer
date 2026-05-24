#[derive(Debug, Clone)]
pub struct DisjointSet {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl DisjointSet {
    pub fn new(n: usize) -> Self {
        Self { parent: (0..n).collect(), rank: vec![0; n] }
    }

    pub fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    pub fn union(&mut self, a: usize, b: usize) -> bool {
        let mut ra = self.find(a);
        let mut rb = self.find(b);
        if ra == rb {
            return false;
        }
        if self.rank[ra] < self.rank[rb] {
            std::mem::swap(&mut ra, &mut rb);
        }
        self.parent[rb] = ra;
        if self.rank[ra] == self.rank[rb] {
            self.rank[ra] += 1;
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::DisjointSet;

    #[test]
    fn union_find_connects_components() {
        let mut ds = DisjointSet::new(4);
        assert!(ds.union(0, 1));
        assert!(ds.union(2, 3));
        assert_ne!(ds.find(0), ds.find(2));
        assert!(ds.union(1, 3));
        assert_eq!(ds.find(0), ds.find(2));
    }
}
