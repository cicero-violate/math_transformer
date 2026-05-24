use anyhow::{bail, Result};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

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
    let src_abs = root.join(op.from.path());
    let dst_abs = root.join(op.to.path());

    let content = load_buf(&src_abs, buffers)?;
    buffers.insert(dst_abs, Some(content));

    if op.preserve_facade {
        let facade = op
            .facade_text
            .clone()
            .unwrap_or_else(|| default_facade(op.to.path()));
        buffers.insert(src_abs, Some(facade.into_bytes()));
    } else {
        buffers.insert(src_abs, None);
    }
    Ok(())
}

fn move_item(
    op: &MoveNode,
    root: &Path,
    buffers: &mut HashMap<PathBuf, Option<Vec<u8>>>,
) -> Result<()> {
    // Cut item text from source.
    let item_text = match &op.from {
        NodeLocator::Anchor {
            path,
            byte_from,
            byte_to,
        } => {
            let abs = root.join(path);
            let mut content = load_buf(&abs, buffers)?;
            let from = (*byte_from).min(content.len());
            let to = (*byte_to).min(content.len());
            if from > to {
                bail!("move source anchor [{from},{to}) invalid in {path}");
            }
            let item: Vec<u8> = content[from..to].to_vec();
            content.drain(from..to);
            buffers.insert(abs, Some(content));
            item
        }
        NodeLocator::Selector { path, selector } => {
            let abs = root.join(path);
            let mut content = load_buf(&abs, buffers)?;
            let text = String::from_utf8_lossy(&content).into_owned();
            let (from, to) = find_item_range(&text, selector)
                .ok_or_else(|| anyhow::anyhow!("selector {:?} not found in {path}", selector))?;
            let item = content[from..to].to_vec();
            content.drain(from..to);
            buffers.insert(abs, Some(content));
            item
        }
    };

    // Paste item text at destination.
    match &op.to {
        NodeLocator::Anchor {
            path, byte_from, ..
        } => {
            let abs = root.join(path);
            let mut content = load_buf(&abs, buffers)?;
            let at = (*byte_from).min(content.len());
            content.splice(at..at, item_text);
            buffers.insert(abs, Some(content));
        }
        NodeLocator::Selector { path, .. } => {
            // Append to destination file if no precise position.
            let abs = root.join(path);
            let mut content = load_buf(&abs, buffers)?;
            content.extend_from_slice(&item_text);
            buffers.insert(abs, Some(content));
        }
    }
    Ok(())
}

fn find_item_range(text: &str, selector: &str) -> Option<(usize, usize)> {
    let item = selector.split("::").last()?;
    let start = text.find(item)?;
    let rest = &text[start..];
    let end = rest
        .find("\npub ")
        .or_else(|| rest.find("\nfn "))
        .or_else(|| rest.find("\nstruct "))
        .or_else(|| rest.find("\nimpl "))
        .unwrap_or(rest.len());
    Some((start, start + end))
}

fn default_facade(to_path: &str) -> String {
    let module = to_path
        .trim_start_matches("src/")
        .trim_end_matches(".rs")
        .replace('/', "::");
    format!("pub use crate::{module}::*;\n")
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
