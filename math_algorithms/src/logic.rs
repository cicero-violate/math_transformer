use std::collections::{BTreeSet, HashMap, HashSet};

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Fact {
    pub pred: String,
    pub args: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct Rule {
    pub head: Fact,
    pub body: Vec<Fact>,
}

pub fn datalog_closure(mut facts: BTreeSet<Fact>, rules: &[Rule]) -> BTreeSet<Fact> {
    loop {
        let mut changed = false;
        for rule in rules {
            if rule.body.iter().all(|f| facts.contains(f)) && facts.insert(rule.head.clone()) {
                changed = true;
            }
        }
        if !changed {
            return facts;
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Interval {
    pub lo: i64,
    pub hi: i64,
}

impl Interval {
    pub fn join(self, rhs: Self) -> Self {
        Self { lo: self.lo.min(rhs.lo), hi: self.hi.max(rhs.hi) }
    }

    pub fn add(self, rhs: Self) -> Self {
        Self { lo: self.lo + rhs.lo, hi: self.hi + rhs.hi }
    }

    pub fn widen(self, rhs: Self) -> Self {
        Self {
            lo: if rhs.lo < self.lo { i64::MIN / 4 } else { self.lo },
            hi: if rhs.hi > self.hi { i64::MAX / 4 } else { self.hi },
        }
    }
}

pub fn abstract_interpret_add(env: &HashMap<String, Interval>, dst: &str, a: &str, b: &str) -> HashMap<String, Interval> {
    let mut next = env.clone();
    if let (Some(x), Some(y)) = (env.get(a), env.get(b)) {
        next.insert(dst.to_string(), x.add(*y));
    }
    next
}

pub fn cegis<Candidate, Generate, Verify>(
    mut generate: Generate,
    verify: Verify,
    max_iters: usize,
) -> Option<Candidate>
where
    Generate: FnMut(&[String]) -> Candidate,
    Verify: Fn(&Candidate) -> Result<(), String>,
{
    let mut counterexamples = Vec::new();
    for _ in 0..max_iters {
        let candidate = generate(&counterexamples);
        match verify(&candidate) {
            Ok(()) => return Some(candidate),
            Err(example) => counterexamples.push(example),
        }
    }
    None
}

pub fn property_fuzz<F>(cases: usize, property: F) -> Option<u64>
where
    F: Fn(u64) -> bool,
{
    let mut x = 0x9e37_79b9_7f4a_7c15u64;
    for _ in 0..cases {
        x ^= x << 7;
        x ^= x >> 9;
        x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
        if !property(x) {
            return Some(x);
        }
    }
    None
}

pub fn reachable_pairs(edges: &[(String, String)]) -> BTreeSet<Fact> {
    let mut facts = BTreeSet::new();
    for (a, b) in edges {
        facts.insert(Fact { pred: "edge".into(), args: vec![a.clone(), b.clone()] });
        facts.insert(Fact { pred: "path".into(), args: vec![a.clone(), b.clone()] });
    }
    let nodes: HashSet<String> = edges.iter().flat_map(|(a, b)| [a.clone(), b.clone()]).collect();
    let mut rules = Vec::new();
    for a in &nodes {
        for b in &nodes {
            for c in &nodes {
                rules.push(Rule {
                    head: Fact { pred: "path".into(), args: vec![a.clone(), c.clone()] },
                    body: vec![
                        Fact { pred: "path".into(), args: vec![a.clone(), b.clone()] },
                        Fact { pred: "path".into(), args: vec![b.clone(), c.clone()] },
                    ],
                });
            }
        }
    }
    datalog_closure(facts, &rules)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logic_tools_work() {
        let paths = reachable_pairs(&[("a".into(), "b".into()), ("b".into(), "c".into())]);
        assert!(paths.contains(&Fact { pred: "path".into(), args: vec!["a".into(), "c".into()] }));
        let mut env = HashMap::new();
        env.insert("x".into(), Interval { lo: 1, hi: 2 });
        env.insert("y".into(), Interval { lo: 3, hi: 5 });
        assert_eq!(abstract_interpret_add(&env, "z", "x", "y")["z"], Interval { lo: 4, hi: 7 });
        let found = cegis(|bad| bad.len(), |n| if *n >= 2 { Ok(()) } else { Err("too small".into()) }, 4);
        assert_eq!(found, Some(2));
        assert!(property_fuzz(32, |x| x == x).is_none());
    }
}
