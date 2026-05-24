pub fn kmp_prefix(pattern: &[u8]) -> Vec<usize> {
    let mut pi = vec![0; pattern.len()];
    for i in 1..pattern.len() {
        let mut j = pi[i - 1];
        while j > 0 && pattern[i] != pattern[j] {
            j = pi[j - 1];
        }
        if pattern[i] == pattern[j] {
            j += 1;
        }
        pi[i] = j;
    }
    pi
}

pub fn kmp_search(haystack: &[u8], needle: &[u8]) -> Vec<usize> {
    if needle.is_empty() {
        return (0..=haystack.len()).collect();
    }
    let pi = kmp_prefix(needle);
    let mut out = Vec::new();
    let mut j = 0;
    for (i, &b) in haystack.iter().enumerate() {
        while j > 0 && b != needle[j] {
            j = pi[j - 1];
        }
        if b == needle[j] {
            j += 1;
        }
        if j == needle.len() {
            out.push(i + 1 - needle.len());
            j = pi[j - 1];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::kmp_search;

    #[test]
    fn finds_matches() {
        assert_eq!(kmp_search(b"ababa", b"aba"), vec![0, 2]);
    }
}
