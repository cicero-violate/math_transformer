use anyhow::{anyhow, bail, Result};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use super::text::{
    bytes_to_text, checked_range, delete_path, item_range, load_path_bytes, remove_bytes,
};
use crate::op::{DeleteNode, NodeKind, NodeLocator};

pub fn apply(
    op: &DeleteNode,
    root: &Path,
    buffers: &mut HashMap<PathBuf, Option<Vec<u8>>>,
) -> Result<()> {
    if matches!(op.kind, NodeKind::File) {
        delete_path(root, op.at.path(), buffers);
        return Ok(());
    }

    match &op.at {
        NodeLocator::Anchor {
            path,
            byte_from,
            byte_to,
        } => {
            let (abs, content) = load_path_bytes(root, path, buffers)?;
            let (from, to) = checked_range(path, *byte_from, *byte_to, content.len(), "delete")?;
            remove_bytes(abs, buffers, from..to)?;
        }
        NodeLocator::Selector { path, selector } => {
            if !op.compiler_proven_unused {
                bail!(
                    "selector-addressed delete of {:?} in {path} requires compiler_proven_unused=true",
                    selector
                );
            }
            let (abs, content) = load_path_bytes(root, path, buffers)?;
            let text = bytes_to_text(&content);
            let (from, to) = item_range(&text, selector)
                .ok_or_else(|| anyhow!("selector {:?} not found in {path}", selector))?;
            remove_bytes(abs, buffers, from..to)?;
        }
    }
    Ok(())
}
