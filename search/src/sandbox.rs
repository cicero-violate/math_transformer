use anyhow::{bail, Result};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use tempfile::TempDir;

use crate::op::{Batch, Op};
use crate::state::RepoState;
use crate::value::VerifyMode;
use crate::verify;

// ── Judgement-delta scorer ────────────────────────────────────────────────────

/// Apply ops to a sandbox, re-run `canon-rustc-v3` + `judgement`, and return
/// a normalized reward in `[0, 1]` where 0.5 = no structural change.
///
/// Requires pre-built `canon_bin` and `judgement_bin` binaries.
/// `crate_name` must match the rustc crate target name (e.g. `"ai"`) so the
/// correct artifact subdirectory is passed to the judgement pass.
pub fn apply_and_score_delta(
    state: &RepoState,
    editor_bin: &Path,
    baseline_judgement: &Path,
    canon_bin: &Path,
    judgement_bin: &Path,
    crate_name: &str,
) -> Result<f32> {
    // No-op states have delta = 0 by definition.
    if state.ops.is_empty() {
        return Ok(0.5);
    }

    let (tmp, crate_in_tmp, pkg) = prepare_sandbox(&state.root)?;
    apply_ops(&crate_in_tmp, &state.ops, editor_bin)?;

    // Re-capture compiler artifacts in the temp workspace.
    let artifact_dir = tmp.path().join("state").join("rustc");
    std::fs::create_dir_all(&artifact_dir)?;

    // Shared target dir for sandbox dep compilation.  Lives under
    // <workspace>/target/sandbox-target/ so dep artifacts are reused across
    // simulations while being separate from the real workspace build.
    let sandbox_target = find_workspace_root(&state.root)
        .map(|ws| ws.join("target").join("sandbox-target"))
        .unwrap_or_else(|| tmp.path().join("target"));
    std::fs::create_dir_all(&sandbox_target)?;

    // Clean just the target crate from the shared sandbox target so cargo
    // is forced to invoke rustc (and canon-rustc-v3) for it specifically.
    // Dep artifacts remain cached.
    if let Some(p) = pkg.as_deref() {
        let _ = Command::new("cargo")
            .args(["clean", "--manifest-path"])
            .arg(tmp.path().join("Cargo.toml"))
            .args(["--target-dir"])
            .arg(&sandbox_target)
            .args(["-p", p])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    }

    let mut check_cmd = Command::new("cargo");
    check_cmd
        .args(["check", "--manifest-path"])
        .arg(tmp.path().join("Cargo.toml"))
        .env("RUSTC_WRAPPER", canon_bin)
        .env("CANON_RUSTC_V3_ARTIFACT_DIR", &artifact_dir)
        .env("CANON_RUSTC_V3_SKIP_CST", "1")  // judgement only needs semantic_index
        .env("CARGO_TARGET_DIR", &sandbox_target)
        .stdout(Stdio::null());
    if let Some(p) = pkg.as_deref() {
        check_cmd.args(["-p", p]);
    }
    if !check_cmd.status()?.success() {
        // Code no longer compiles after edit — penalize.
        return Ok(0.0);
    }

    // Re-run the judgement pass on fresh artifacts.
    let crate_artifact_dir = artifact_dir.join(crate_name);

    // Diagnose empty artifact dir before attempting judgement.
    let artifact_entries: Vec<_> = std::fs::read_dir(&artifact_dir)
        .map(|rd| rd.filter_map(|e| e.ok()).map(|e| e.file_name().to_string_lossy().to_string()).collect())
        .unwrap_or_default();
    if artifact_entries.is_empty() {
        bail!(
            "canon-rustc-v3 wrote no artifacts to {} (is it built with rustc-driver feature?)",
            artifact_dir.display()
        );
    }
    if !crate_artifact_dir.exists() {
        bail!(
            "artifact subdir for crate '{}' not found; got: {:?}",
            crate_name, artifact_entries
        );
    }

    let judgement_out = tmp.path().join("judgement_out");
    std::fs::create_dir_all(&judgement_out)?;

    let jout = Command::new(judgement_bin)
        .arg("--artifacts-dir")
        .arg(&crate_artifact_dir)
        .arg("--output")
        .arg(&judgement_out)
        .arg("--passes")
        .arg("all")
        .stdout(Stdio::null())
        .output()?;

    if !jout.status.success() {
        let stderr = String::from_utf8_lossy(&jout.stderr);
        bail!("judgement failed for crate {crate_name}: {}", stderr.trim());
    }

    let new_judgement = judgement_out.join("judgement.jsonl");
    if !new_judgement.exists() {
        bail!("judgement produced no output at {}", new_judgement.display());
    }

    // Compute reward via `judgement delta`.
    let output = Command::new(judgement_bin)
        .arg("delta")
        .arg("--before")
        .arg(baseline_judgement)
        .arg("--after")
        .arg(&new_judgement)
        .output()?;

    if !output.status.success() {
        bail!("judgement delta failed");
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    parse_reward(&stdout)
}

/// Parse `reward: +0.012345` from the `judgement delta` stdout.
/// Maps reward ∈ (-∞, +∞) → score ∈ [0, 1] with 0.5 = no change.
fn parse_reward(stdout: &str) -> Result<f32> {
    for line in stdout.lines() {
        if let Some(rest) = line.strip_prefix("reward:") {
            let val: f64 = rest.trim().parse()?;
            return Ok(((val.clamp(-1.0, 1.0) + 1.0) / 2.0) as f32);
        }
    }
    bail!("no 'reward:' line in judgement delta output:\n{stdout}")
}

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
        VerifyMode::Check => verify::cargo_check_pkg(tmp.path(), pkg.as_deref()),
        VerifyMode::Test => verify::cargo_test_pkg(tmp.path(), pkg.as_deref()),
        VerifyMode::Composite => verify::composite_pkg(tmp.path(), pkg.as_deref()),
    };
    drop(tmp); // keep alive until scoring done
    Ok(score)
}

/// Apply `ops` to a directory via the structural-editor JSONL subprocess.
pub fn apply_ops(dir: &Path, ops: &[Op], editor_bin: &Path) -> Result<()> {
    let batch = Batch {
        root: dir,
        label: None,
        ops,
    };
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
        bail!(
            "structural-editor: {}",
            response["error"].as_str().unwrap_or("unknown")
        );
    }
    Ok(())
}

// ── Sandbox preparation ───────────────────────────────────────────────────────

/// Returns (tempdir, crate_path_inside_tempdir, package_name).
fn prepare_sandbox(crate_root: &Path) -> Result<(TempDir, PathBuf, Option<String>)> {
    // Place sandboxes under <workspace>/.sandboxes/ — same filesystem as
    // compiled artifacts, avoids /tmp space pressure, and critically NOT
    // under <workspace>/target/ (canon-rustc-v3 skips capture for crates
    // whose CARGO_MANIFEST_DIR lives inside target/).
    let sandbox_base = find_workspace_root(crate_root)
        .map(|ws| ws.join(".sandboxes"))
        .unwrap_or_else(std::env::temp_dir);
    std::fs::create_dir_all(&sandbox_base)?;
    let tmp = tempfile::Builder::new()
        .prefix("canon-search-")
        .tempdir_in(&sandbox_base)?;
    let pkg = package_name(crate_root);

    match find_workspace_root(crate_root) {
        Some(ws_root) => {
            // Relative path of crate inside the workspace (e.g. "search").
            let rel = crate_root
                .strip_prefix(&ws_root)
                .unwrap_or(Path::new("crate"));
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
            let ws_toml = std::fs::read_to_string(ws_root.join("Cargo.toml")).unwrap_or_default();
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
        if line == "[package]" {
            in_package = true;
            continue;
        }
        if line.starts_with('[') {
            in_package = false;
        }
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
        let Some(path_idx) = line.find("path") else {
            continue;
        };
        let after_path = &line[path_idx + "path".len()..];
        let Some(eq_idx) = after_path.find('=') else {
            continue;
        };
        let after_eq = after_path[eq_idx + 1..].trim_start();
        let Some(rest) = after_eq.strip_prefix('"') else {
            continue;
        };
        let Some(end_idx) = rest.find('"') else {
            continue;
        };
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
        if name == "target" || name.starts_with('.') {
            continue;
        }
        let rel = entry.path().strip_prefix(src)?;
        let dst_path = dst.join(rel);
        if entry.file_type().is_dir() {
            std::fs::create_dir_all(&dst_path)?;
        } else if entry.file_type().is_file() {
            if let Some(p) = dst_path.parent() {
                std::fs::create_dir_all(p)?;
            }
            std::fs::copy(entry.path(), &dst_path)?;
        }
        // Skip sockets, pipes, broken symlinks, etc.
    }
    Ok(())
}
