use anyhow::{bail, Result};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use crate::op::CargoChange;

pub fn apply(
    op: &CargoChange,
    root: &Path,
    buffers: &mut HashMap<PathBuf, Option<Vec<u8>>>,
) -> Result<()> {
    match op {
        CargoChange::AddDependency {
            manifest,
            name,
            version,
            features,
        } => {
            append_dep(
                root,
                buffers,
                manifest,
                "[dependencies]",
                name,
                version,
                features,
            )?;
        }
        CargoChange::RemoveDependency { manifest, name } => {
            remove_dep(root, buffers, manifest, "[dependencies]", name)?;
        }
        CargoChange::AddDevDependency {
            manifest,
            name,
            version,
            features,
        } => {
            append_dep(
                root,
                buffers,
                manifest,
                "[dev-dependencies]",
                name,
                version,
                features,
            )?;
        }
        CargoChange::RemoveDevDependency { manifest, name } => {
            remove_dep(root, buffers, manifest, "[dev-dependencies]", name)?;
        }
        CargoChange::AddBuildDependency {
            manifest,
            name,
            version,
            features,
        } => {
            append_dep(
                root,
                buffers,
                manifest,
                "[build-dependencies]",
                name,
                version,
                features,
            )?;
        }
        CargoChange::AddFeature {
            manifest,
            name,
            members,
        } => {
            let abs = root.join(manifest);
            let mut text = load_text(&abs, buffers)?;
            let entry = if members.is_empty() {
                format!("{name} = []\n")
            } else {
                let list = members
                    .iter()
                    .map(|m| format!("\"{m}\""))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("{name} = [{list}]\n")
            };
            if let Some(pos) = find_section(&text, "[features]") {
                let insert = pos + "[features]".len() + 1;
                text.insert_str(insert, &entry);
            } else {
                text.push_str(&format!("\n[features]\n{entry}"));
            }
            buffers.insert(abs, Some(text.into_bytes()));
        }
        CargoChange::RemoveFeature { manifest, name } => {
            let abs = root.join(manifest);
            let text = load_text(&abs, buffers)?;
            let updated = remove_key_line(&text, name);
            buffers.insert(abs, Some(updated.into_bytes()));
        }
        CargoChange::SetPackageField {
            manifest,
            field,
            value,
        } => {
            let abs = root.join(manifest);
            let text = load_text(&abs, buffers)?;
            let key_eq = format!("{field} =");
            let updated = if let Some(line_start) = find_key_line(&text, &key_eq) {
                let line_end = text[line_start..]
                    .find('\n')
                    .map(|i| line_start + i + 1)
                    .unwrap_or(text.len());
                let mut t = text.clone();
                t.replace_range(line_start..line_end, &format!("{field} = \"{value}\"\n"));
                t
            } else {
                // Append under [package].
                if let Some(pos) = find_section(&text, "[package]") {
                    let insert = pos + "[package]".len() + 1;
                    let mut t = text.clone();
                    t.insert_str(insert, &format!("{field} = \"{value}\"\n"));
                    t
                } else {
                    format!("{text}\n{field} = \"{value}\"\n")
                }
            };
            buffers.insert(abs, Some(updated.into_bytes()));
        }
        CargoChange::AddBinTarget {
            manifest,
            name,
            path,
        } => {
            append_target(root, buffers, manifest, "bin", name, path)?;
        }
        CargoChange::AddLibTarget {
            manifest,
            path,
            crate_type,
        } => {
            let abs = root.join(manifest);
            let mut text = load_text(&abs, buffers)?;
            let crate_type_line = crate_type
                .as_deref()
                .map(|t| format!("crate-type = [\"{t}\"]\n"))
                .unwrap_or_default();
            text.push_str(&format!("\n[lib]\npath = \"{path}\"\n{crate_type_line}"));
            buffers.insert(abs, Some(text.into_bytes()));
        }
        CargoChange::AddTestTarget {
            manifest,
            name,
            path,
        } => {
            append_target(root, buffers, manifest, "test", name, path)?;
        }
        CargoChange::AddExampleTarget {
            manifest,
            name,
            path,
        } => {
            append_target(root, buffers, manifest, "example", name, path)?;
        }
        CargoChange::RemoveTarget {
            manifest,
            kind,
            name,
        } => {
            let abs = root.join(manifest);
            let text = load_text(&abs, buffers)?;
            let updated = remove_target_block(&text, kind, name);
            buffers.insert(abs, Some(updated.into_bytes()));
        }
        CargoChange::InsertSnippet { manifest, snippet } => {
            let abs = root.join(manifest);
            let mut text = load_text(&abs, buffers)?;
            text.push('\n');
            text.push_str(snippet);
            text.push('\n');
            buffers.insert(abs, Some(text.into_bytes()));
        }
    }

    Ok(())
}

// ── helpers ───────────────────────────────────────────────────────────────────

fn append_dep(
    root: &Path,
    buffers: &mut HashMap<PathBuf, Option<Vec<u8>>>,
    manifest: &str,
    section: &str,
    name: &str,
    version: &str,
    features: &[String],
) -> Result<()> {
    let abs = root.join(manifest);
    let mut text = load_text(&abs, buffers)?;
    let entry = if features.is_empty() {
        format!("{name} = \"{version}\"\n")
    } else {
        let feats = features
            .iter()
            .map(|f| format!("\"{f}\""))
            .collect::<Vec<_>>()
            .join(", ");
        format!("{name} = {{ version = \"{version}\", features = [{feats}] }}\n")
    };
    if let Some(pos) = find_section(&text, section) {
        let insert = pos + section.len() + 1;
        text.insert_str(insert, &entry);
    } else {
        text.push_str(&format!("\n{section}\n{entry}"));
    }
    buffers.insert(abs, Some(text.into_bytes()));
    Ok(())
}

fn remove_dep(
    root: &Path,
    buffers: &mut HashMap<PathBuf, Option<Vec<u8>>>,
    manifest: &str,
    _section: &str,
    name: &str,
) -> Result<()> {
    let abs = root.join(manifest);
    let text = load_text(&abs, buffers)?;
    let updated = remove_key_line(&text, name);
    buffers.insert(abs, Some(updated.into_bytes()));
    Ok(())
}

fn append_target(
    root: &Path,
    buffers: &mut HashMap<PathBuf, Option<Vec<u8>>>,
    manifest: &str,
    kind: &str,
    name: &str,
    path: &str,
) -> Result<()> {
    let abs = root.join(manifest);
    let mut text = load_text(&abs, buffers)?;
    text.push_str(&format!(
        "\n[[{kind}]]\nname = \"{name}\"\npath = \"{path}\"\n"
    ));
    buffers.insert(abs, Some(text.into_bytes()));
    Ok(())
}

fn find_section(text: &str, section: &str) -> Option<usize> {
    text.find(section)
}

fn find_key_line(text: &str, key_eq: &str) -> Option<usize> {
    text.find(key_eq)
}

fn remove_key_line(text: &str, key: &str) -> String {
    text.lines()
        .filter(|l| !l.trim_start().starts_with(&format!("{key} =")))
        .map(|l| format!("{l}\n"))
        .collect()
}

fn remove_target_block(text: &str, kind: &str, name: &str) -> String {
    let header = format!("[[{kind}]]");
    let name_line = format!("name = \"{name}\"");
    let mut out = String::new();
    let mut skip = false;
    for line in text.lines() {
        if line.trim() == header {
            skip = true;
        }
        if skip && line.trim() == name_line {
            // Confirmed: skip this block until next section.
            continue;
        }
        if skip && line.starts_with('[') && line.trim() != header {
            skip = false;
        }
        if !skip {
            out.push_str(line);
            out.push('\n');
        }
    }
    out
}

fn load_text(abs: &Path, buffers: &mut HashMap<PathBuf, Option<Vec<u8>>>) -> Result<String> {
    let bytes = load_buf(abs, buffers)?;
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

fn load_buf(abs: &Path, buffers: &mut HashMap<PathBuf, Option<Vec<u8>>>) -> Result<Vec<u8>> {
    if let Some(entry) = buffers.get(abs) {
        return match entry {
            Some(v) => Ok(v.clone()),
            None => bail!(
                "manifest {} was already deleted in this batch",
                abs.display()
            ),
        };
    }
    if abs.exists() {
        Ok(fs::read(abs)?)
    } else {
        Ok(Vec::new())
    }
}
