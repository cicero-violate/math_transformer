use std::cmp::Reverse;
use std::collections::{BinaryHeap, VecDeque};

pub type WeightedGraph = Vec<Vec<(usize, u64)>>;

pub fn dijkstra(graph: &WeightedGraph, source: usize) -> Vec<u64> {
    let mut dist = vec![u64::MAX; graph.len()];
    let mut heap = BinaryHeap::new();
    dist[source] = 0;
    heap.push((Reverse(0), source));
    while let Some((Reverse(cost), node)) = heap.pop() {
        if cost != dist[node] {
            continue;
        }
        for &(next, weight) in &graph[node] {
            let alt = cost.saturating_add(weight);
            if alt < dist[next] {
                dist[next] = alt;
                heap.push((Reverse(alt), next));
            }
        }
    }
    dist
}

pub fn astar<F>(graph: &WeightedGraph, start: usize, goal: usize, h: F) -> Option<u64>
where
    F: Fn(usize) -> u64,
{
    let mut best = vec![u64::MAX; graph.len()];
    let mut heap = BinaryHeap::new();
    best[start] = 0;
    heap.push((Reverse(h(start)), 0, start));
    while let Some((_, cost, node)) = heap.pop() {
        if node == goal {
            return Some(cost);
        }
        if cost != best[node] {
            continue;
        }
        for &(next, weight) in &graph[node] {
            let alt = cost.saturating_add(weight);
            if alt < best[next] {
                best[next] = alt;
                heap.push((Reverse(alt.saturating_add(h(next))), alt, next));
            }
        }
    }
    None
}

pub fn topological_sort(graph: &[Vec<usize>]) -> Option<Vec<usize>> {
    let mut indegree = vec![0usize; graph.len()];
    for edges in graph {
        for &v in edges {
            indegree[v] += 1;
        }
    }
    let mut queue: VecDeque<usize> = indegree.iter()
        .enumerate()
        .filter_map(|(i, &d)| (d == 0).then_some(i))
        .collect();
    let mut out = Vec::with_capacity(graph.len());
    while let Some(u) = queue.pop_front() {
        out.push(u);
        for &v in &graph[u] {
            indegree[v] -= 1;
            if indegree[v] == 0 {
                queue.push_back(v);
            }
        }
    }
    (out.len() == graph.len()).then_some(out)
}

pub fn tarjan_scc(graph: &[Vec<usize>]) -> Vec<Vec<usize>> {
    struct Tarjan<'a> {
        graph: &'a [Vec<usize>],
        index: usize,
        stack: Vec<usize>,
        on_stack: Vec<bool>,
        indices: Vec<Option<usize>>,
        low: Vec<usize>,
        sccs: Vec<Vec<usize>>,
    }

    impl<'a> Tarjan<'a> {
        fn visit(&mut self, v: usize) {
            self.indices[v] = Some(self.index);
            self.low[v] = self.index;
            self.index += 1;
            self.stack.push(v);
            self.on_stack[v] = true;
            for &w in &self.graph[v] {
                if self.indices[w].is_none() {
                    self.visit(w);
                    self.low[v] = self.low[v].min(self.low[w]);
                } else if self.on_stack[w] {
                    self.low[v] = self.low[v].min(self.indices[w].unwrap());
                }
            }
            if self.low[v] == self.indices[v].unwrap() {
                let mut scc = Vec::new();
                loop {
                    let w = self.stack.pop().unwrap();
                    self.on_stack[w] = false;
                    scc.push(w);
                    if w == v {
                        break;
                    }
                }
                self.sccs.push(scc);
            }
        }
    }

    let n = graph.len();
    let mut t = Tarjan {
        graph,
        index: 0,
        stack: Vec::new(),
        on_stack: vec![false; n],
        indices: vec![None; n],
        low: vec![0; n],
        sccs: Vec::new(),
    };
    for v in 0..n {
        if t.indices[v].is_none() {
            t.visit(v);
        }
    }
    t.sccs
}

pub fn bfs(graph: &[Vec<usize>], start: usize) -> Vec<usize> {
    let mut seen = vec![false; graph.len()];
    let mut queue = VecDeque::from([start]);
    let mut order = Vec::new();
    seen[start] = true;
    while let Some(u) = queue.pop_front() {
        order.push(u);
        for &v in &graph[u] {
            if !seen[v] {
                seen[v] = true;
                queue.push_back(v);
            }
        }
    }
    order
}

pub fn dfs(graph: &[Vec<usize>], start: usize) -> Vec<usize> {
    let mut seen = vec![false; graph.len()];
    let mut stack = vec![start];
    let mut order = Vec::new();
    while let Some(u) = stack.pop() {
        if seen[u] {
            continue;
        }
        seen[u] = true;
        order.push(u);
        for &v in graph[u].iter().rev() {
            stack.push(v);
        }
    }
    order
}

pub fn max_flow(capacity: Vec<Vec<i64>>, source: usize, sink: usize) -> i64 {
    let n = capacity.len();
    let mut residual = capacity;
    let mut flow = 0;
    loop {
        let mut parent = vec![None; n];
        let mut queue = VecDeque::from([source]);
        parent[source] = Some(source);
        while let Some(u) = queue.pop_front() {
            for v in 0..n {
                if parent[v].is_none() && residual[u][v] > 0 {
                    parent[v] = Some(u);
                    queue.push_back(v);
                }
            }
        }
        if parent[sink].is_none() {
            return flow;
        }
        let mut add = i64::MAX;
        let mut v = sink;
        while v != source {
            let u = parent[v].unwrap();
            add = add.min(residual[u][v]);
            v = u;
        }
        v = sink;
        while v != source {
            let u = parent[v].unwrap();
            residual[u][v] -= add;
            residual[v][u] += add;
            v = u;
        }
        flow += add;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graph_basics_work() {
        let g = vec![vec![(1, 4), (2, 1)], vec![(3, 1)], vec![(1, 2), (3, 5)], vec![]];
        assert_eq!(dijkstra(&g, 0)[3], 4);
        assert_eq!(astar(&g, 0, 3, |_| 0), Some(4));
        assert_eq!(topological_sort(&[vec![1], vec![2], vec![]]).unwrap(), vec![0, 1, 2]);
        assert!(tarjan_scc(&[vec![1], vec![0], vec![]]).iter().any(|c| c.len() == 2));
        assert_eq!(bfs(&[vec![1, 2], vec![3], vec![], vec![]], 0), vec![0, 1, 2, 3]);
        assert_eq!(dfs(&[vec![1, 2], vec![3], vec![], vec![]], 0), vec![0, 1, 3, 2]);
        assert_eq!(max_flow(vec![vec![0, 3, 2, 0], vec![0, 0, 1, 2], vec![0, 0, 0, 4], vec![0, 0, 0, 0]], 0, 3), 5);
    }
}
