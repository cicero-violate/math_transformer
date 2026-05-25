use std::cmp::Ordering;

pub fn beam_search<S, Expand, Score>(
    start: S,
    width: usize,
    steps: usize,
    expand: Expand,
    score: Score,
) -> Vec<S>
where
    S: Clone,
    Expand: Fn(&S) -> Vec<S>,
    Score: Fn(&S) -> f64,
{
    let mut beam = vec![start];
    for _ in 0..steps {
        let mut next = Vec::new();
        for state in &beam {
            next.extend(expand(state));
        }
        next.sort_by(|a, b| score(b).partial_cmp(&score(a)).unwrap_or(Ordering::Equal));
        next.truncate(width);
        if next.is_empty() {
            break;
        }
        beam = next;
    }
    beam
}

#[cfg(test)]
mod tests {
    use super::beam_search;

    #[test]
    fn keeps_best_frontier() {
        let out = beam_search(0, 2, 3, |x| vec![x + 1, x + 2], |x| *x as f64);
        assert_eq!(out, vec![6, 5]);
    }
}
