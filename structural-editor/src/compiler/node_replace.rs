use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use super::text::{
    bytes_to_text, checked_range, item_range, load_path_bytes, selector_item, splice_bytes,
};
use crate::op::{NodeLocator, ReplaceNode, ReplaceTarget};

pub fn apply(
    op: &ReplaceNode,
    root: &Path,
    buffers: &mut HashMap<PathBuf, Option<Vec<u8>>>,
) -> Result<()> {
    match &op.at {
        NodeLocator::Anchor {
            path,
            byte_from,
            byte_to,
        } => {
            let (abs, content) = load_path_bytes(root, path, buffers)?;
            let (from, to) = checked_range(path, *byte_from, *byte_to, content.len(), "replace")?;
            // For Whole/Body/Signature we replace the entire anchor range.
            // Callers use the compiler index to pass the precise sub-range for
            // Body or Signature; without compiler data the whole range is used.
            splice_bytes(abs, buffers, from..to, &op.text)?;
        }
        NodeLocator::Selector { path, selector } => {
            let (abs, content) = load_path_bytes(root, path, buffers)?;
            let text = bytes_to_text(&content);
            let (from, to) = find_replace_range(&text, selector, &op.target)
                .ok_or_else(|| anyhow!("selector {:?} not found in {path}", selector))?;
            splice_bytes(abs, buffers, from..to, &op.text)?;
        }
    }
    Ok(())
}

fn find_replace_range(
    text: &str,
    selector: &str,
    target: &ReplaceTarget,
) -> Option<(usize, usize)> {
    let item = selector_item(selector);
    let start = text.find(item)?;
    match target {
        ReplaceTarget::Whole => item_range(text, selector),
        ReplaceTarget::Signature => {
            // Signature = start up to (not including) the opening brace or `=`.
            let rest = &text[start..];
            let sig_end = rest
                .find('{')
                .or_else(|| rest.find('='))
                .unwrap_or(rest.len());
            Some((start, start + sig_end))
        }
        ReplaceTarget::Body => {
            let rest = &text[start..];
            let body_start = rest.find('{')? + 1;
            let body_end = rest.rfind('}')?;
            Some((start + body_start, start + body_end))
        }
        ReplaceTarget::Type => {
            // Type annotation: between `:` and `,` or end of item.
            let rest = &text[start..];
            let colon = rest.find(':')?;
            let type_end = rest[colon..]
                .find([',', '\n', '}'].as_ref())
                .unwrap_or(rest.len() - colon);
            Some((start + colon + 1, start + colon + type_end))
        }
        ReplaceTarget::Value => {
            let rest = &text[start..];
            let eq = rest.find('=')?;
            let val_end = rest[eq..]
                .find([',', '\n', ';', '}'].as_ref())
                .unwrap_or(rest.len() - eq);
            Some((start + eq + 1, start + eq + val_end))
        }
    }
}
