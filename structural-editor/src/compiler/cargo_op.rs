use anyhow::Result;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use super::text::{filter_lines_to_string, load_text, store_text};
use crate::op::CargoChange;

type Buffers = HashMap<PathBuf, Option<Vec<u8>>>;

pub fn apply(op: &CargoChange, root: &Path, buffers: &mut Buffers) -> Result<()> {
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
            remove_dep(root, buffers, manifest, name)?;
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
            remove_dep(root, buffers, manifest, name)?;
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
            update_manifest(root, buffers, manifest, |text| {
                let entry = feature_entry(name, members);
                insert_under_section_or_append(text, "[features]", &entry);
            })?;
        }
        CargoChange::RemoveFeature { manifest, name } => {
            update_manifest(root, buffers, manifest, |text| {
                *text = remove_key_line(text, name);
            })?;
        }
        CargoChange::SetPackageField {
            manifest,
            field,
            value,
        } => {
            update_manifest(root, buffers, manifest, |text| {
                set_package_field(text, field, value);
            })?;
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
            update_manifest(root, buffers, manifest, |text| {
                let crate_type_line = crate_type
                    .as_deref()
                    .map(|t| format!("crate-type = [\"{t}\"]\n"))
                    .unwrap_or_default();
                append_lines(
                    text,
                    &format!("[lib]\npath = \"{path}\"\n{crate_type_line}"),
                );
            })?;
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
            update_manifest(root, buffers, manifest, |text| {
                *text = remove_target_block(text, kind, name);
            })?;
        }
        CargoChange::InsertSnippet { manifest, snippet } => {
            update_manifest(root, buffers, manifest, |text| append_lines(text, snippet))?;
        }
    }

    Ok(())
}

// ── helpers ───────────────────────────────────────────────────────────────────

fn update_manifest<F>(root: &Path, buffers: &mut Buffers, manifest: &str, edit: F) -> Result<()>
where
    F: FnOnce(&mut String),
{
    let abs = root.join(manifest);
    let mut text = load_text(&abs, buffers)?;
    edit(&mut text);
    store_text(abs, text, buffers);
    Ok(())
}

fn append_dep(
    root: &Path,
    buffers: &mut Buffers,
    manifest: &str,
    section: &str,
    name: &str,
    version: &str,
    features: &[String],
) -> Result<()> {
    update_manifest(root, buffers, manifest, |text| {
        let entry = dependency_entry(name, version, features);
        insert_under_section_or_append(text, section, &entry);
    })
}

fn remove_dep(root: &Path, buffers: &mut Buffers, manifest: &str, name: &str) -> Result<()> {
    update_manifest(root, buffers, manifest, |text| {
        *text = remove_key_line(text, name);
    })
}

fn append_target(
    root: &Path,
    buffers: &mut Buffers,
    manifest: &str,
    kind: &str,
    name: &str,
    path: &str,
) -> Result<()> {
    update_manifest(root, buffers, manifest, |text| {
        append_lines(
            text,
            &format!("[[{kind}]]\nname = \"{name}\"\npath = \"{path}\"\n"),
        );
    })
}

fn dependency_entry(name: &str, version: &str, features: &[String]) -> String {
    if features.is_empty() {
        format!("{name} = \"{version}\"\n")
    } else {
        let feats = quoted_list(features);
        format!("{name} = {{ version = \"{version}\", features = [{feats}] }}\n")
    }
}

fn feature_entry(name: &str, members: &[String]) -> String {
    if members.is_empty() {
        format!("{name} = []\n")
    } else {
        let list = quoted_list(members);
        format!("{name} = [{list}]\n")
    }
}

fn quoted_list(items: &[String]) -> String {
    items
        .iter()
        .map(|item| format!("\"{item}\""))
        .collect::<Vec<_>>()
        .join(", ")
}

fn insert_under_section_or_append(text: &mut String, section: &str, entry: &str) {
    if let Some(pos) = find_section(text, section) {
        let insert = pos + section.len() + 1;
        text.insert_str(insert, entry);
    } else {
        text.push_str(&format!("\n{section}\n{entry}"));
    }
}

fn append_lines(text: &mut String, body: &str) {
    text.push('\n');
    text.push_str(body);
    if !body.ends_with('\n') {
        text.push('\n');
    }
}

fn set_package_field(text: &mut String, field: &str, value: &str) {
    let replacement = package_field_line(field, value);
    let key_eq = format!("{field} =");
    if let Some(line_start) = find_key_line(text, &key_eq) {
        let line_end = line_end_after(text, line_start);
        text.replace_range(line_start..line_end, &replacement);
    } else if let Some(pos) = find_section(text, "[package]") {
        let insert = pos + "[package]".len() + 1;
        text.insert_str(insert, &replacement);
    } else {
        append_lines(text, replacement.trim_end());
    }
}

fn package_field_line(field: &str, value: &str) -> String {
    format!("{field} = \"{value}\"\n")
}

fn line_end_after(text: &str, line_start: usize) -> usize {
    text[line_start..]
        .find('\n')
        .map(|i| line_start + i + 1)
        .unwrap_or(text.len())
}

fn find_section(text: &str, section: &str) -> Option<usize> {
    text.find(section)
}

fn find_key_line(text: &str, key_eq: &str) -> Option<usize> {
    let mut offset = 0;
    for line in text.lines() {
        if line.trim_start().starts_with(key_eq) {
            return Some(offset);
        }
        offset += line.len() + 1;
    }
    None
}

fn remove_key_line(text: &str, key: &str) -> String {
    let key_eq = format!("{key} =");
    filter_lines_to_string(text, |line| !line.trim_start().starts_with(&key_eq))
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
