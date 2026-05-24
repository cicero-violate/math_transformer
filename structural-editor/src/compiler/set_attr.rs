use anyhow::{bail, Result};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use crate::op::{AttrKey, NodeLocator, SetAttr};

pub fn apply(
    op: &SetAttr,
    root: &Path,
    buffers: &mut HashMap<PathBuf, Option<Vec<u8>>>,
) -> Result<()> {
    let path = op.at.path();
    let abs = root.join(path);
    let mut content = load_buf(&abs, buffers)?;
    let text = String::from_utf8_lossy(&content).into_owned();

    let item_start = match &op.at {
        NodeLocator::Anchor { byte_from, .. } => *byte_from,
        NodeLocator::Selector { selector, .. } => {
            let item = selector.split("::").last().unwrap_or(selector);
            let item_start = text
                .find(item)
                .ok_or_else(|| anyhow::anyhow!("selector {:?} not found in {path}", selector))?;
            item_line_start(&text, item_start)
        }
    };

    let updated = match &op.key {
        AttrKey::Visibility => set_visibility(&text, item_start, &op.value),
        AttrKey::Derive => set_derive(&text, item_start, &op.value),
        AttrKey::Doc => set_doc(&text, item_start, &op.value),
        AttrKey::Cfg => insert_attr(&text, item_start, &format!("#[cfg({})]", op.value)),
        AttrKey::Allow => insert_attr(&text, item_start, &format!("#[allow({})]", op.value)),
        AttrKey::MustUse => {
            let attr = if op.value.is_empty() {
                "#[must_use]".to_string()
            } else {
                format!("#[must_use = {:?}]", op.value)
            };
            insert_attr(&text, item_start, &attr)
        }
        AttrKey::Inline => {
            let attr = if op.value.is_empty() {
                "#[inline]".to_string()
            } else {
                format!("#[inline({})]", op.value)
            };
            insert_attr(&text, item_start, &attr)
        }
        AttrKey::Deprecated => {
            let attr = if op.value.is_empty() {
                "#[deprecated]".to_string()
            } else {
                format!("#[deprecated = {:?}]", op.value)
            };
            insert_attr(&text, item_start, &attr)
        }
        AttrKey::Repr => insert_attr(&text, item_start, &format!("#[repr({})]", op.value)),
        AttrKey::Custom => insert_attr(&text, item_start, &op.value),
    };

    content = updated.into_bytes();
    buffers.insert(abs, Some(content));
    Ok(())
}

fn item_line_start(text: &str, item_start: usize) -> usize {
    let line_start = text[..item_start]
        .rfind('\n')
        .map(|idx| idx + 1)
        .unwrap_or(0);
    let line = &text[line_start..item_start];
    line_start + line.len() - line.trim_start().len()
}

fn set_visibility(text: &str, item_start: usize, vis: &str) -> String {
    let before = &text[..item_start];
    let after = &text[item_start..];

    // Remove existing visibility keyword.
    let after = after
        .trim_start_matches("pub(crate) ")
        .trim_start_matches("pub(super) ")
        .trim_start_matches("pub ");

    let prefix = if vis.is_empty() {
        String::new()
    } else {
        format!("{vis} ")
    };
    format!("{before}{prefix}{after}")
}

fn set_derive(text: &str, item_start: usize, derives: &str) -> String {
    let before = &text[..item_start];
    let after = &text[item_start..];

    // Check for existing derive attribute just before item_start.
    if let Some(derive_pos) = before.rfind("#[derive(") {
        let derive_end = before[derive_pos..]
            .find(")]")
            .map(|i| derive_pos + i + 2)
            .unwrap_or(before.len());
        if derives.is_empty() {
            // Remove entire derive.
            let mut t = text.to_string();
            t.drain(derive_pos..derive_end);
            return t;
        }
        // Merge: append to existing list.
        let insert_pos = derive_end - 2; // before ")]"
        let mut t = text.to_string();
        t.insert_str(insert_pos, &format!(", {derives}"));
        return t;
    }

    // No existing derive: insert a new one before the item.
    format!("{before}#[derive({derives})]\n{after}")
}

fn set_doc(text: &str, item_start: usize, doc: &str) -> String {
    let before = &text[..item_start];
    let after = &text[item_start..];
    let doc_line = format!("/// {doc}\n");
    format!("{before}{doc_line}{after}")
}

fn insert_attr(text: &str, item_start: usize, attr: &str) -> String {
    let before = &text[..item_start];
    let after = &text[item_start..];
    format!("{before}{attr}\n{after}")
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
