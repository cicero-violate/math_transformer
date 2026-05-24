use anyhow::Result;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use crate::compiler;
use crate::op::OpBatch;
use crate::output::{EditResult, FileAction, FileDelta};

/// Apply `batch` rooted at `root`. All ops are buffered in memory; files are
/// written only when every op succeeds (all-or-nothing within the batch).
pub fn apply(batch: &OpBatch, root: &Path) -> EditResult {
    match try_apply(batch, root) {
        Ok(deltas) => EditResult {
            ok: true,
            label: batch.label.clone(),
            deltas,
            error: None,
        },
        Err(e) => EditResult {
            ok: false,
            label: batch.label.clone(),
            deltas: vec![],
            error: Some(e.to_string()),
        },
    }
}

fn try_apply(batch: &OpBatch, root: &Path) -> Result<Vec<FileDelta>> {
    let mut buffers: HashMap<PathBuf, Option<Vec<u8>>> = HashMap::new();

    for op in &batch.ops {
        compiler::compile(op, root, &mut buffers)?;
    }

    commit(buffers, root)
}

/// Write buffered changes to disk and build the delta list.
fn commit(buffers: HashMap<PathBuf, Option<Vec<u8>>>, root: &Path) -> Result<Vec<FileDelta>> {
    let mut deltas = Vec::new();

    for (abs_path, new_content) in buffers {
        let rel = abs_path.strip_prefix(root).unwrap_or(&abs_path);
        let path_str = rel.to_string_lossy().into_owned();
        let existed = abs_path.exists();
        let old_bytes = if existed {
            fs::read(&abs_path).ok()
        } else {
            None
        };

        match new_content {
            None => {
                if existed {
                    fs::remove_file(&abs_path)?;
                }
                deltas.push(FileDelta {
                    path: path_str,
                    action: FileAction::Deleted,
                    diff: String::new(),
                });
            }
            Some(bytes) => {
                if let Some(parent) = abs_path.parent() {
                    fs::create_dir_all(parent)?;
                }
                fs::write(&abs_path, &bytes)?;
                let new_str = String::from_utf8_lossy(&bytes).into_owned();
                let old_str = old_bytes
                    .as_deref()
                    .map(|b| String::from_utf8_lossy(b).into_owned())
                    .unwrap_or_default();
                let diff = unified_diff(&old_str, &new_str, &path_str);
                let action = if existed {
                    FileAction::Modified
                } else {
                    FileAction::Created
                };
                deltas.push(FileDelta {
                    path: path_str,
                    action,
                    diff,
                });
            }
        }
    }

    Ok(deltas)
}

fn unified_diff(old: &str, new: &str, path: &str) -> String {
    if old == new {
        return String::new();
    }
    let old_lines: Vec<&str> = old.lines().collect();
    let new_lines: Vec<&str> = new.lines().collect();
    let mut out = format!("--- a/{path}\n+++ b/{path}\n");
    out.push_str(&format!(
        "@@ -{},{} +{},{} @@\n",
        1,
        old_lines.len(),
        1,
        new_lines.len()
    ));
    for l in &old_lines {
        out.push_str(&format!("-{l}\n"));
    }
    for l in &new_lines {
        out.push_str(&format!("+{l}\n"));
    }
    out
}
