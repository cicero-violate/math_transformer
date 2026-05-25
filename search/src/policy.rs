use std::path::Path;

use crate::op::Op;
use crate::state::RepoState;

/// One action proposed by the policy, with its prior probability.
pub struct Candidate {
    pub op: Op,
    pub prior: f32,
}

/// Neural policy interface.
pub trait Policy: Send + Sync {
    fn propose(&self, state: &RepoState) -> Vec<Candidate>;
}

// ── Static template policies ──────────────────────────────────────────────────

pub struct TemplatePolicy {
    pub templates: Vec<Op>,
}

impl Policy for TemplatePolicy {
    fn propose(&self, _state: &RepoState) -> Vec<Candidate> {
        let n = self.templates.len();
        let prior = if n == 0 { 1.0 } else { 1.0 / n as f32 };
        self.templates
            .iter()
            .map(|op| Candidate {
                op: op.clone(),
                prior,
            })
            .collect()
    }
}

pub struct NoisyTemplatePolicy {
    pub templates: Vec<Op>,
    pub alpha: f64,
}

impl Policy for NoisyTemplatePolicy {
    fn propose(&self, _state: &RepoState) -> Vec<Candidate> {
        use rand::distributions::Distribution;
        let n = self.templates.len();
        if n == 0 {
            return Vec::new();
        }
        let rng = &mut rand::thread_rng();
        let gamma = rand::distributions::Uniform::new(0.0f64, 1.0);
        let mut noise: Vec<f64> = (0..n)
            .map(|_| (-gamma.sample(rng).ln()).powf(self.alpha))
            .collect();
        let sum: f64 = noise.iter().sum();
        for x in &mut noise {
            *x /= sum;
        }
        self.templates
            .iter()
            .zip(noise)
            .map(|(op, p)| Candidate {
                op: op.clone(),
                prior: p as f32,
            })
            .collect()
    }
}

// ── Dynamic candidate generator ───────────────────────────────────────────────

/// Scans the target crate's `.rs` files and generates structural ops for every
/// public item found. Replaces the fixed `ops.jsonl` template list.
///
/// Each call to `propose` rescans the current state root, so it adapts to
/// whichever crate MCTS is running against and to edits already applied.
pub struct CandidateGen {
    /// Maximum candidates to return per call (keeps the action space bounded).
    pub max_candidates: usize,
}

impl Policy for CandidateGen {
    fn propose(&self, state: &RepoState) -> Vec<Candidate> {
        let ops = scan_crate(&state.root, self.max_candidates);
        let n = ops.len();
        if n == 0 {
            return Vec::new();
        }
        let prior = 1.0 / n as f32;
        ops.into_iter().map(|op| Candidate { op, prior }).collect()
    }
}

// ── Scanner ───────────────────────────────────────────────────────────────────

fn scan_crate(root: &Path, max: usize) -> Vec<Op> {
    let mut ops: Vec<Op> = Vec::new();

    for entry in walkdir::WalkDir::new(root).min_depth(1) {
        let Ok(entry) = entry else { continue };
        let name = entry.file_name().to_string_lossy();
        if name == "target" || name.starts_with('.') {
            continue;
        }
        if !entry.file_type().is_file() {
            continue;
        }
        if entry.path().extension().and_then(|e| e.to_str()) != Some("rs") {
            continue;
        }

        let rel = entry
            .path()
            .strip_prefix(root)
            .unwrap_or(entry.path())
            .to_string_lossy()
            .replace('\\', "/");

        let Ok(src) = std::fs::read_to_string(entry.path()) else {
            continue;
        };

        for item in scan_items(&src) {
            ops.extend(ops_for_item(&rel, &item));
            if ops.len() >= max {
                ops.truncate(max);
                return ops;
            }
        }
    }
    ops
}

struct Item {
    kind: &'static str,
    name: String,
}

/// Extract public items from Rust source using simple line-by-line scanning.
fn scan_items(src: &str) -> Vec<Item> {
    let mut items = Vec::new();
    for line in src.lines() {
        let t = line.trim();
        let (kind, rest) = if let Some(r) = t.strip_prefix("pub fn ") {
            ("fn", r)
        } else if let Some(r) = t.strip_prefix("pub struct ") {
            ("struct", r)
        } else if let Some(r) = t.strip_prefix("pub enum ") {
            ("enum", r)
        } else if let Some(r) = t.strip_prefix("pub trait ") {
            ("trait", r)
        } else if let Some(r) = t.strip_prefix("pub type ") {
            ("type", r)
        } else if let Some(r) = t.strip_prefix("impl ") {
            ("impl", r)
        } else {
            continue;
        };

        // Extract the identifier — stop at whitespace, `(`, `<`, `{`, `:`
        let name: String = rest
            .chars()
            .take_while(|&c| c.is_alphanumeric() || c == '_')
            .collect();
        if name.is_empty() || name == "for" {
            continue;
        }

        items.push(Item { kind, name });
    }
    items
}

/// Generate structural ops for a single item.
fn ops_for_item(rel_path: &str, item: &Item) -> Vec<Op> {
    let selector = format!("{} {}", item.kind, item.name);
    let at = serde_json::json!({
        "loc": "selector",
        "path": rel_path,
        "selector": selector,
    });
    let kind = node_kind(item.kind);

    let mut ops: Vec<Op> = Vec::new();

    // Verify the symbol exists — cheap gate op, always useful.
    ops.push(serde_json::json!({
        "op": "verify",
        "predicate": { "predicate": "symbol_exists", "path": rel_path, "selector": selector }
    }));

    match item.kind {
        "fn" => {
            ops.push(serde_json::json!({"op":"set_attr","kind":kind,"at":at.clone(),"key":"must_use","value":""}));
            ops.push(serde_json::json!({"op":"set_attr","kind":kind,"at":at.clone(),"key":"inline","value":""}));
            ops.push(serde_json::json!({"op":"set_attr","kind":kind,"at":at.clone(),"key":"doc","value":format!("{}.", &item.name)}));
        }
        "struct" | "enum" => {
            ops.push(serde_json::json!({"op":"set_attr","kind":kind,"at":at.clone(),"key":"derive","value":"Clone"}));
            ops.push(serde_json::json!({"op":"set_attr","kind":kind,"at":at.clone(),"key":"derive","value":"Debug"}));
            ops.push(serde_json::json!({"op":"set_attr","kind":kind,"at":at.clone(),"key":"doc","value":format!("{}.", &item.name)}));
        }
        "trait" | "type" => {
            ops.push(serde_json::json!({"op":"set_attr","kind":kind,"at":at.clone(),"key":"doc","value":format!("{}.", &item.name)}));
        }
        "impl" => {
            // Add a minimal helper method to the impl block.
            let insert_at = serde_json::json!({
                "loc": "selector", "path": rel_path,
                "selector": format!("impl {}", &item.name),
            });
            ops.push(serde_json::json!({
                "op": "create_node", "kind": "function",
                "at": insert_at,
                "text": format!("    pub fn is_empty(&self) -> bool {{ false }}")
            }));
        }
        _ => {}
    }
    ops
}

fn node_kind(kind: &str) -> &'static str {
    match kind {
        "fn" => "function",
        "struct" => "struct",
        "enum" => "enum",
        "trait" => "trait",
        "type" => "type_alias",
        "impl" => "impl_block",
        _ => "function",
    }
}
