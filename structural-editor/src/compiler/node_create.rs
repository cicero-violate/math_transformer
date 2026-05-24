use anyhow::{bail, Result};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use crate::op::{CreateNode, NodeKind, NodeLocator};

pub fn apply(
    op: &CreateNode,
    root: &Path,
    buffers: &mut HashMap<PathBuf, Option<Vec<u8>>>,
) -> Result<()> {
    match op.kind {
        NodeKind::File => {
            // Create the whole file.
            let abs = root.join(op.at.path());
            buffers.insert(abs, Some(op.text.as_bytes().to_vec()));
        }
        _ => {
            // Insert `text` into an existing file at the locator position.
            match &op.at {
                NodeLocator::Anchor {
                    path, byte_from, ..
                } => {
                    let abs = root.join(path);
                    let mut content = load_buf(&abs, buffers)?;
                    let at = (*byte_from).min(content.len());
                    content.splice(at..at, op.text.bytes());
                    buffers.insert(abs, Some(content));
                }
                NodeLocator::Selector { path, selector } => {
                    let abs = root.join(path);
                    let mut content = load_buf(&abs, buffers)?;
                    let text = String::from_utf8_lossy(&content).into_owned();
                    // Find the selector's closing brace and insert before it,
                    // or append to the file if the selector resolves to the file itself.
                    let insert_pos = find_selector_insert(&text, selector).unwrap_or(content.len());
                    content.splice(insert_pos..insert_pos, op.text.bytes());
                    buffers.insert(abs, Some(content));
                }
            }
        }
    }
    Ok(())
}

/// Best-effort: find the byte offset to insert a new member inside a selector's
/// body. Returns None if the selector is not found (caller falls back to append).
fn find_selector_insert(text: &str, selector: &str) -> Option<usize> {
    // Strip member qualifiers to get the parent item name.
    let item_name = selector.split("::").next()?;
    // Find the item header (e.g. "struct Config", "impl Config").
    let header_pos = text.find(item_name)?;
    // Find the matching opening brace.
    let open_brace = text[header_pos..].find('{')? + header_pos;
    // Find the matching closing brace (shallow scan, ignores nesting).
    let close_brace = text[open_brace..].rfind('}')? + open_brace;
    // Insert just before the closing brace.
    Some(close_brace)
}

fn load_buf(abs: &Path, buffers: &mut HashMap<PathBuf, Option<Vec<u8>>>) -> Result<Vec<u8>> {
    if let Some(entry) = buffers.get(abs) {
        return match entry {
            Some(v) => Ok(v.clone()),
            None => bail!("file {} was deleted earlier in this batch", abs.display()),
        };
    }
    if abs.exists() {
        Ok(fs::read(abs)?)
    } else {
        Ok(Vec::new())
    }
}
