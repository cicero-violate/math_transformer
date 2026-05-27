use anyhow::Result;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use super::text::{
    insert_line_before, line_start_with_indent, load_path_text, selector_item, store_text,
};
use crate::op::{AttrKey, NodeLocator, SetAttr};

pub fn apply(
    op: &SetAttr,
    root: &Path,
    buffers: &mut HashMap<PathBuf, Option<Vec<u8>>>,
) -> Result<()> {
    let path = op.at.path();
    let (abs, text) = load_path_text(root, path, buffers)?;

    let item_start = match &op.at {
        NodeLocator::Anchor { byte_from, .. } => *byte_from,
        NodeLocator::Selector { selector, .. } => {
            let item = selector_item(selector);
            let item_start = text
                .find(item)
                .ok_or_else(|| anyhow::anyhow!("selector {:?} not found in {path}", selector))?;
            line_start_with_indent(&text, item_start)
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

    store_text(abs, updated, buffers);
    Ok(())
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
    insert_line_before(text, item_start, &format!("/// {doc}"))
}

fn insert_attr(text: &str, item_start: usize, attr: &str) -> String {
    insert_line_before(text, item_start, attr)
}
