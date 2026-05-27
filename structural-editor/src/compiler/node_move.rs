use anyhow::Result;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use super::text::{
    append_bytes, checked_range, cut_bytes, delete_path, insert_bytes, item_range, load_path_bytes,
    load_path_text, store_bytes, store_text,
};
use crate::op::{MoveNode, NodeKind, NodeLocator};

pub fn apply(
    op: &MoveNode,
    root: &Path,
    buffers: &mut HashMap<PathBuf, Option<Vec<u8>>>,
) -> Result<()> {
    if matches!(op.kind, NodeKind::File) {
        move_file(op, root, buffers)
    } else {
        move_item(op, root, buffers)
    }
}

fn move_file(
    op: &MoveNode,
    root: &Path,
    buffers: &mut HashMap<PathBuf, Option<Vec<u8>>>,
) -> Result<()> {
    let (src_abs, content) = load_path_bytes(root, op.from.path(), buffers)?;
    store_bytes(root.join(op.to.path()), content, buffers);

    if op.preserve_facade {
        let facade = op
            .facade_text
            .clone()
            .unwrap_or_else(|| default_facade(op.to.path()));
        store_text(src_abs, facade, buffers);
    } else {
        delete_path(root, op.from.path(), buffers);
    }
    Ok(())
}

fn move_item(
    op: &MoveNode,
    root: &Path,
    buffers: &mut HashMap<PathBuf, Option<Vec<u8>>>,
) -> Result<()> {
    let item_text = cut_item_text(op, root, buffers)?;
    paste_item_text(op, root, buffers, item_text)
}

fn cut_item_text(
    op: &MoveNode,
    root: &Path,
    buffers: &mut HashMap<PathBuf, Option<Vec<u8>>>,
) -> Result<Vec<u8>> {
    match &op.from {
        NodeLocator::Anchor {
            path,
            byte_from,
            byte_to,
        } => {
            let (abs, content) = load_path_bytes(root, path, buffers)?;
            let (from, to) =
                checked_range(path, *byte_from, *byte_to, content.len(), "move source")?;
            cut_bytes(abs, buffers, from..to)
        }
        NodeLocator::Selector { path, selector } => {
            let (abs, text) = load_path_text(root, path, buffers)?;
            let (from, to) = item_range(&text, selector)
                .ok_or_else(|| anyhow::anyhow!("selector {:?} not found in {path}", selector))?;
            cut_bytes(abs, buffers, from..to)
        }
    }
}

fn paste_item_text(
    op: &MoveNode,
    root: &Path,
    buffers: &mut HashMap<PathBuf, Option<Vec<u8>>>,
    item_text: Vec<u8>,
) -> Result<()> {
    match &op.to {
        NodeLocator::Anchor {
            path, byte_from, ..
        } => insert_bytes(root.join(path), buffers, *byte_from, item_text),
        NodeLocator::Selector { path, .. } => append_bytes(root.join(path), buffers, &item_text),
    }
}

fn default_facade(to_path: &str) -> String {
    let module = to_path
        .trim_start_matches("src/")
        .trim_end_matches(".rs")
        .replace('/', "::");
    format!("pub use crate::{module}::*;\n")
}
