use std::path::Path;
use std::process::Command;

// ── Package-aware variants ────────────────────────────────────────────────────
// When `pkg` is Some, adds `-p <pkg>` so cargo operates on a workspace member
// rather than the crate at `--manifest-path`.

pub fn cargo_check_pkg(root: &Path, pkg: Option<&str>) -> f32 {
    let mut cmd = Command::new("cargo");
    cmd.args(["check", "--manifest-path"])
        .arg(root.join("Cargo.toml"))
        .arg("--quiet");
    if let Some(p) = pkg {
        cmd.args(["-p", p]);
    }
    match cmd.output() {
        Ok(out) if out.status.success() => 1.0,
        Ok(out) => {
            // Print first 400 chars of stderr so failures are diagnosable.
            let err = String::from_utf8_lossy(&out.stderr);
            eprintln!(
                "[verify] cargo check failed: {}",
                &err[..err.len().min(400)]
            );
            0.0
        }
        Err(e) => {
            eprintln!("[verify] cargo spawn error: {e}");
            0.0
        }
    }
}

pub fn cargo_test_pkg(root: &Path, pkg: Option<&str>) -> f32 {
    let mut cmd = Command::new("cargo");
    cmd.args(["test", "--manifest-path"])
        .arg(root.join("Cargo.toml"));
    if let Some(p) = pkg {
        cmd.args(["-p", p]);
    }
    match cmd.output() {
        Ok(out) if out.status.success() => 1.0,
        Ok(out) => parse_test_ratio(&String::from_utf8_lossy(&out.stdout)),
        Err(_) => 0.0,
    }
}

pub fn composite_pkg(root: &Path, pkg: Option<&str>) -> f32 {
    let check = cargo_check_pkg(root, pkg);
    if check < 1.0 {
        return 0.0;
    }
    cargo_test_pkg(root, pkg)
}

// ── Legacy single-crate variants (kept for direct use) ───────────────────────

pub fn cargo_check(root: &Path) -> f32 {
    cargo_check_pkg(root, None)
}
pub fn cargo_test(root: &Path) -> f32 {
    cargo_test_pkg(root, None)
}
pub fn composite(root: &Path) -> f32 {
    composite_pkg(root, None)
}

fn parse_test_ratio(output: &str) -> f32 {
    for line in output.lines() {
        if line.contains("test result:") {
            let passed = extract_count(line, "passed").unwrap_or(0);
            let failed = extract_count(line, "failed").unwrap_or(0);
            let total = passed + failed;
            if total > 0 {
                return passed as f32 / total as f32;
            }
        }
    }
    0.0
}

fn extract_count(line: &str, keyword: &str) -> Option<u32> {
    let pos = line.find(keyword)?;
    line[..pos].trim().rsplit_once(' ')?.1.parse().ok()
}
