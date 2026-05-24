use anyhow::{bail, Result};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use tempfile::TempDir;

use crate::op::{Batch, Op};
use crate::state::RepoState;
use crate::value::VerifyMode;
use crate::verify;

/// Copy the crate to a temp dir, apply ops, score.
///
/// Only the target crate is copied (not the whole workspace), keeping disk
/// usage proportional to the crate, not the workspace.  If the crate is a
/// workspace member, a minimal workspace Cargo.toml is generated at the
/// tempdir root so that `{ workspace = true }` dependencies resolve.
pub fn apply_and_score(state: &RepoState, editor_bin: &Path, mode: VerifyMode) -> Result<f32> {
    let (tmp, crate_in_tmp, pkg) = prepare_sandbox(&state.root)?;

    if !state.ops.is_empty() {
        apply_ops(&crate_in_tmp, &state.ops, editor_bin)?;
    }

    let score = match mode {
        VerifyMode::Check     => verify::cargo_check_pkg(tmp.path(), pkg.as_deref()),
        VerifyMode::Test      => verify::cargo_test_pkg(tmp.path(), pkg.as_deref()),
        VerifyMode::Composite => verify::composite_pkg(tmp.path(), pkg.as_deref()),
    };
    drop(tmp); // keep alive until scoring done
    Ok(score)
}

/// Apply `ops` to a directory via the structural-editor JSONL subprocess.
pub fn apply_ops(dir: &Path, ops: &[Op], editor_bin: &Path) -> Result<()> {
    let batch = Batch { root: dir, label: None, ops };
    let line = serde_json::to_string(&batch)? + "\n";

    let mut child = Command::new(editor_bin)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()?;

    child.stdin.as_mut().unwrap().write_all(line.as_bytes())?;
    drop(child.stdin.take());

    let output = child.wait_with_output()?;
    if output.stdout.is_empty() {
        bail!("structural-editor produced no output");
    }
    let response: serde_json::Value = serde_json::from_slice(&output.stdout)
        .map_err(|e| anyhow::anyhow!("bad JSON from structural-editor: {e}"))?;
    if !response["ok"].as_bool().unwrap_or(false) {
        bail!("structural-editor: {}", response["error"].as_str().unwrap_or("unknown"));
    }
    Ok(())
}

// ── Sandbox preparation ───────────────────────────────────────────────────────

/// Returns (tempdir, crate_path_inside_tempdir, package_name).
fn prepare_sandbox(crate_root: &Path) -> Result<(TempDir, PathBuf, Option<String>)> {
    let tmp = tempfile::tempdir()?;
    let pkg = package_name(crate_root);

    match find_workspace_root(crate_root) {
        Some(ws_root) => {
            // Relative path of crate inside the workspace (e.g. "search").
            let rel = crate_root.strip_prefix(&ws_root).unwrap_or(Path::new("crate"));
            let crate_dst = tmp.path().join(rel);

            // Copy the crate directory and any local path dependencies it needs
            // in the same workspace layout.
            copy_dir(crate_root, &crate_dst)?;
            let mut members = vec![rel.to_string_lossy().replace('\\', "/")];
            for dep in path_dependencies(crate_root) {
                if !dep.starts_with(&ws_root) {
                    continue;
                }
                let dep_rel = dep.strip_prefix(&ws_root).unwrap_or(dep.as_path());
                let dep_dst = tmp.path().join(dep_rel);
                copy_dir(&dep, &dep_dst)?;
                members.push(dep_rel.to_string_lossy().replace('\\', "/"));
            }
            members.sort();
            members.dedup();

            // Generate a minimal workspace Cargo.toml so workspace deps resolve.
            let ws_toml = std::fs::read_to_string(ws_root.join("Cargo.toml"))
                .unwrap_or_default();
            let minimal = minimal_workspace_toml(&ws_toml, &members);
            std::fs::write(tmp.path().join("Cargo.toml"), minimal)?;

            Ok((tmp, crate_dst, pkg))
        }
        None => {
            // Standalone crate — copy it directly.
            copy_dir(crate_root, tmp.path())?;
            let crate_dst = tmp.path().to_path_buf();
            Ok((tmp, crate_dst, pkg))
        }
    }
}

/// Build a minimal workspace Cargo.toml that contains only `member` and
/// preserves the original `[workspace.dependencies]` block verbatim.
fn minimal_workspace_toml(original: &str, members: &[String]) -> String {
    // Extract everything from [workspace.dependencies] to the next top-level section.
    let mut ws_deps = String::new();
    let mut in_ws_deps = false;
    for line in original.lines() {
        let trimmed = line.trim();
        if trimmed == "[workspace.dependencies]" {
            in_ws_deps = true;
            ws_deps.push_str(line);
            ws_deps.push('\n');
        } else if in_ws_deps {
            if trimmed.starts_with('[') && trimmed != "[workspace.dependencies]" {
                break; // next top-level section
            }
            ws_deps.push_str(line);
            ws_deps.push('\n');
        }
    }
    let members = members
        .iter()
        .map(|member| format!("\"{member}\""))
        .collect::<Vec<_>>()
        .join(", ");
    format!("[workspace]\nmembers = [{members}]\nresolver = \"2\"\n\n{ws_deps}")
}

// ── Workspace detection ───────────────────────────────────────────────────────

fn find_workspace_root(crate_root: &Path) -> Option<PathBuf> {
    let mut dir = crate_root.parent()?;
    loop {
        let toml = dir.join("Cargo.toml");
        if toml.exists() {
            if let Ok(txt) = std::fs::read_to_string(&toml) {
                if txt.contains("[workspace]") {
                    return Some(dir.to_path_buf());
                }
            }
        }
        dir = dir.parent()?;
    }
}

fn package_name(crate_root: &Path) -> Option<String> {
    let txt = std::fs::read_to_string(crate_root.join("Cargo.toml")).ok()?;
    let mut in_package = false;
    for line in txt.lines() {
        let line = line.trim();
        if line == "[package]" { in_package = true; continue; }
        if line.starts_with('[') { in_package = false; }
        if in_package {
            if let Some(rest) = line.strip_prefix("name") {
                if let Some(val) = rest.trim().strip_prefix('=') {
                    return Some(val.trim().trim_matches('"').to_string());
                }
            }
        }
    }
    None
}

fn path_dependencies(crate_root: &Path) -> Vec<PathBuf> {
    let Ok(txt) = std::fs::read_to_string(crate_root.join("Cargo.toml")) else {
        return Vec::new();
    };
    let mut deps = Vec::new();
    for line in txt.lines() {
        let Some(path_idx) = line.find("path") else { continue };
        let after_path = &line[path_idx + "path".len()..];
        let Some(eq_idx) = after_path.find('=') else { continue };
        let after_eq = after_path[eq_idx + 1..].trim_start();
        let Some(rest) = after_eq.strip_prefix('"') else { continue };
        let Some(end_idx) = rest.find('"') else { continue };
        let path = &rest[..end_idx];
        let dep = crate_root.join(path);
        if dep.join("Cargo.toml").exists() {
            deps.push(dep);
        }
    }
    deps
}

// ── Filesystem helpers ────────────────────────────────────────────────────────

fn copy_dir(src: &Path, dst: &Path) -> Result<()> {
    for entry in walkdir::WalkDir::new(src).min_depth(1) {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy();
        if name == "target" || name.starts_with('.') { continue; }
        let rel = entry.path().strip_prefix(src)?;
        let dst_path = dst.join(rel);
        if entry.file_type().is_dir() {
            std::fs::create_dir_all(&dst_path)?;
        } else if entry.file_type().is_file() {
            if let Some(p) = dst_path.parent() { std::fs::create_dir_all(p)?; }
            std::fs::copy(entry.path(), &dst_path)?;
        }
        // Skip sockets, pipes, broken symlinks, etc.
    }
    Ok(())
}
