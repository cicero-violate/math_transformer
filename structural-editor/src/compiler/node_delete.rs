use anyhow::{bail, Result};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use crate::op::{DeleteNode, NodeKind, NodeLocator};

pub fn apply(
    op: &DeleteNode,
    root: &Path,
    buffers: &mut HashMap<PathBuf, Option<Vec<u8>>>,
) -> Result<()> {
    if matches!(op.kind, NodeKind::File) {
        let abs = root.join(op.at.path());
        buffers.insert(abs, None);
        return Ok(());
    }

    match &op.at {
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
                bail!("delete anchor [{from},{to}) invalid in {path}");
            }
            content.drain(from..to);
            buffers.insert(abs, Some(content));
        }
        NodeLocator::Selector { path, selector } => {
            if !op.compiler_proven_unused {
                bail!(
                    "selector-addressed delete of {:?} in {path} requires compiler_proven_unused=true",
                    selector
                );
            }
            let abs = root.join(path);
            let mut content = load_buf(&abs, buffers)?;
            let text = String::from_utf8_lossy(&content).into_owned();
            let (from, to) = resolve_selector(&text, selector)
                .ok_or_else(|| anyhow::anyhow!("selector {:?} not found in {path}", selector))?;
            content.drain(from..to);
            buffers.insert(abs, Some(content));
        }
    }
    Ok(())
}

/// Resolve a selector to a byte range by finding the item name and its full span.
fn resolve_selector(text: &str, selector: &str) -> Option<(usize, usize)> {
    let item = selector.split("::").last()?;
    let start = text.find(item)?;
    // Find end: next item-level keyword or end of file.
    let rest = &text[start..];
    let len = rest
        .find("\npub ")
        .or_else(|| rest.find("\nfn "))
        .or_else(|| rest.find("\nstruct "))
        .or_else(|| rest.find("\nimpl "))
        .unwrap_or(rest.len());
    Some((start, start + len))
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
