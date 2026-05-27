use anyhow::Result;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use super::text::{load_path_text, store_text};
use crate::op::RenameSymbol;

pub fn apply(
    op: &RenameSymbol,
    root: &Path,
    buffers: &mut HashMap<PathBuf, Option<Vec<u8>>>,
) -> Result<()> {
    // Collect all files to scan: definition file + explicit scope.
    let mut files: Vec<String> = vec![op.at.path().to_string()];
    for f in &op.scope {
        if !files.contains(f) {
            files.push(f.clone());
        }
    }

    for file in files {
        let (abs, text) = load_path_text(root, &file, buffers)?;
        // Replace all whole-word occurrences of old_name with new_name.
        let updated = replace_word(&text, &op.old_name, &op.new_name);
        if updated != text {
            store_text(abs, updated, buffers);
        }
    }

    Ok(())
}

/// Replace whole-word occurrences: `old` must be bounded by non-alphanumeric/non-underscore.
fn replace_word(text: &str, old: &str, new: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(pos) = rest.find(old) {
        let before = if pos == 0 {
            true
        } else {
            let ch = rest.as_bytes()[pos - 1] as char;
            !ch.is_alphanumeric() && ch != '_'
        };
        let after = {
            let end = pos + old.len();
            if end >= rest.len() {
                true
            } else {
                let ch = rest.as_bytes()[end] as char;
                !ch.is_alphanumeric() && ch != '_'
            }
        };
        if before && after {
            out.push_str(&rest[..pos]);
            out.push_str(new);
            rest = &rest[pos + old.len()..];
        } else {
            out.push_str(&rest[..pos + 1]);
            rest = &rest[pos + 1..];
        }
    }
    out.push_str(rest);
    out
}
