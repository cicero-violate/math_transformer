pub type Hash = [u8; 32];

pub fn toy_hash(data: &[u8]) -> Hash {
    let mut h = [0u8; 32];
    for (i, &b) in data.iter().enumerate() {
        let j = i % 32;
        h[j] = h[j].wrapping_mul(31).wrapping_add(b).rotate_left((i % 8) as u32);
    }
    h
}

pub fn merkle_parent(left: Hash, right: Hash) -> Hash {
    let mut bytes = Vec::with_capacity(64);
    bytes.extend_from_slice(&left);
    bytes.extend_from_slice(&right);
    toy_hash(&bytes)
}

pub fn merkle_root(leaves: &[Hash]) -> Hash {
    if leaves.is_empty() {
        return toy_hash(&[]);
    }
    let mut level = leaves.to_vec();
    while level.len() > 1 {
        let mut next = Vec::new();
        for pair in level.chunks(2) {
            let right = *pair.get(1).unwrap_or(&pair[0]);
            next.push(merkle_parent(pair[0], right));
        }
        level = next;
    }
    level[0]
}

pub fn verify_merkle_proof(leaf: Hash, mut index: usize, proof: &[(Hash, bool)], root: Hash) -> bool {
    let mut acc = leaf;
    for &(sibling, sibling_is_left) in proof {
        acc = if sibling_is_left {
            merkle_parent(sibling, acc)
        } else {
            merkle_parent(acc, sibling)
        };
        index /= 2;
    }
    let _ = index;
    acc == root
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verifies_merkle_path() {
        let leaves = [toy_hash(b"a"), toy_hash(b"b")];
        let root = merkle_root(&leaves);
        assert!(verify_merkle_proof(leaves[0], 0, &[(leaves[1], false)], root));
    }
}
