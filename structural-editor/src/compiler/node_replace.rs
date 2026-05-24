use anyhow::{bail, Result};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

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
            let abs = root.join(path);
            let mut content = load_buf(&abs, buffers)?;
            let from = (*byte_from).min(content.len());
            let to = (*byte_to).min(content.len());
            if from > to {
                bail!("replace anchor [{from},{to}) invalid in {path}");
            }
            // For Whole/Body/Signature we replace the entire anchor range.
            // Callers use the compiler index to pass the precise sub-range for
            // Body or Signature; without compiler data the whole range is used.
            content.splice(from..to, op.text.bytes());
            buffers.insert(abs, Some(content));
        }
        NodeLocator::Selector { path, selector } => {
            let abs = root.join(path);
            let mut content = load_buf(&abs, buffers)?;
            let text = String::from_utf8_lossy(&content).into_owned();
            let (from, to) = find_replace_range(&text, selector, &op.target)
                .ok_or_else(|| anyhow::anyhow!("selector {:?} not found in {path}", selector))?;
            content.splice(from..to, op.text.bytes());
            buffers.insert(abs, Some(content));
        }
    }
    Ok(())
}

fn find_replace_range(
    text: &str,
    selector: &str,
    target: &ReplaceTarget,
) -> Option<(usize, usize)> {
    let item = selector.split("::").last()?;
    let start = text.find(item)?;
    match target {
        ReplaceTarget::Whole => {
            let rest = &text[start..];
            let end = rest
                .find("\npub ")
                .or_else(|| rest.find("\nfn "))
                .or_else(|| rest.find("\nstruct "))
                .or_else(|| rest.find("\nimpl "))
                .unwrap_or(rest.len());
            Some((start, start + end))
        }
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
