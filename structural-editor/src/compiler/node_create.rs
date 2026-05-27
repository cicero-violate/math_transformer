use anyhow::Result;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use super::text::{
    abs_path, bytes_to_text, load_path_bytes, parent_selector_item, splice_bytes, store_bytes,
};
use crate::op::{CreateNode, NodeKind, NodeLocator};

pub fn apply(
    op: &CreateNode,
    root: &Path,
    buffers: &mut HashMap<PathBuf, Option<Vec<u8>>>,
) -> Result<()> {
    match op.kind {
        NodeKind::File => {
            // Create the whole file.
            let abs = abs_path(root, op.at.path());
            store_bytes(abs, op.text.as_bytes().to_vec(), buffers);
        }
        _ => {
            // Insert `text` into an existing file at the locator position.
            match &op.at {
                NodeLocator::Anchor {
                    path, byte_from, ..
                } => {
                    let (abs, content) = load_path_bytes(root, path, buffers)?;
                    let at = (*byte_from).min(content.len());
                    splice_bytes(abs, buffers, at..at, &op.text)?;
                }
                NodeLocator::Selector { path, selector } => {
                    let (abs, content) = load_path_bytes(root, path, buffers)?;
                    let text = bytes_to_text(&content);
                    // Find the selector's closing brace and insert before it,
                    // or append to the file if the selector resolves to the file itself.
                    let insert_pos = find_selector_insert(&text, selector).unwrap_or(content.len());
                    splice_bytes(abs, buffers, insert_pos..insert_pos, &op.text)?;
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
    let item_name = parent_selector_item(selector)?;
    // Find the item header (e.g. "struct Config", "impl Config").
    let header_pos = text.find(item_name)?;
    // Find the matching opening brace.
    let open_brace = text[header_pos..].find('{')? + header_pos;
    // Find the matching closing brace (shallow scan, ignores nesting).
    let close_brace = text[open_brace..].rfind('}')? + open_brace;
    // Insert just before the closing brace.
    Some(close_brace)
}
