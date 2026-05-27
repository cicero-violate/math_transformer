use anyhow::Result;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use super::text::{filter_lines_to_string, load_path_text, store_path_text};
use crate::op::{AddEdge, EdgeKind, RemoveEdge};

pub fn add(
    op: &AddEdge,
    root: &Path,
    buffers: &mut HashMap<PathBuf, Option<Vec<u8>>>,
) -> Result<()> {
    let (_abs, text) = load_path_text(root, &op.file, buffers)?;

    let line = edge_to_text(&op.edge);

    let updated = match &op.edge {
        EdgeKind::Uses { .. } | EdgeKind::Declares { .. } | EdgeKind::ExternCrate(_) => {
            // Insert at the top of the use/mod block, or after the last existing use.
            insert_near_uses(&text, &line)
        }
        EdgeKind::Implements {
            trait_path,
            type_path,
        } => {
            // Append an impl block to the file.
            format!("{text}\nimpl {trait_path} for {type_path} {{\n}}\n")
        }
        EdgeKind::Bound { param, bound } => {
            // Insert into an existing where clause, or append one.
            insert_where_bound(&text, param, bound)
        }
        EdgeKind::PathRef { old_path, new_path } => {
            text.replace(old_path.as_str(), new_path.as_str())
        }
    };

    store_path_text(root, &op.file, updated, buffers);
    Ok(())
}

pub fn remove(
    op: &RemoveEdge,
    root: &Path,
    buffers: &mut HashMap<PathBuf, Option<Vec<u8>>>,
) -> Result<()> {
    let (_abs, text) = load_path_text(root, &op.file, buffers)?;

    let line = edge_to_text(&op.edge);
    let updated = match &op.edge {
        EdgeKind::PathRef { old_path, new_path } => {
            // PathRef removal = revert: new_path → old_path.
            text.replace(new_path.as_str(), old_path.as_str())
        }
        EdgeKind::Bound { param, bound } => remove_where_bound(&text, param, bound),
        _ => {
            // Remove the exact line.
            text.lines()
                .filter(|l| l.trim() != line.trim())
                .map(|l| format!("{l}\n"))
                .collect()
        }
    };

    store_path_text(root, &op.file, updated, buffers);
    Ok(())
}

fn edge_to_text(edge: &EdgeKind) -> String {
    match edge {
        EdgeKind::Uses { use_path } => format!("use {use_path};"),
        EdgeKind::Declares { module, inline } => {
            if *inline {
                format!("mod {module} {{}}")
            } else {
                format!("mod {module};")
            }
        }
        EdgeKind::Implements {
            trait_path,
            type_path,
        } => {
            format!("impl {trait_path} for {type_path} {{}}")
        }
        EdgeKind::Bound { param, bound } => format!("{param}: {bound}"),
        EdgeKind::PathRef { old_path, .. } => old_path.clone(),
        EdgeKind::ExternCrate(name) => format!("extern crate {name};"),
    }
}

fn insert_near_uses(text: &str, line: &str) -> String {
    // Find the last `use` or `mod` line and insert after it.
    let mut last_use_end = 0;
    for (i, _ch) in text.char_indices() {
        let rest = &text[i..];
        if rest.starts_with("use ") || rest.starts_with("mod ") || rest.starts_with("extern crate")
        {
            if let Some(nl) = rest.find('\n') {
                last_use_end = i + nl + 1;
            }
        }
        let _ = i; // suppress unused-variable for loop counter shadowed by char_indices
    }
    if last_use_end == 0 {
        format!("{line}\n{text}")
    } else {
        format!("{}{line}\n{}", &text[..last_use_end], &text[last_use_end..])
    }
}

fn insert_where_bound(text: &str, param: &str, bound: &str) -> String {
    let clause = format!("{param}: {bound}");
    if let Some(pos) = text.find("where") {
        // Find the end of the where list (opening brace).
        let after_where = &text[pos..];
        if let Some(brace) = after_where.find('{') {
            let insert = pos + brace;
            return format!("{},\n    {}{}", &text[..insert], clause, &text[insert..]);
        }
    }
    // No where clause: insert one before the opening brace.
    if let Some(brace) = text.find('{') {
        format!(
            "{}where\n    {}\n{}",
            &text[..brace],
            clause,
            &text[brace..]
        )
    } else {
        format!("{text}\nwhere\n    {clause}\n")
    }
}

fn remove_where_bound(text: &str, param: &str, bound: &str) -> String {
    let clause = format!("{param}: {bound}");
    filter_lines_to_string(text, |line| !line.contains(&clause))
}
