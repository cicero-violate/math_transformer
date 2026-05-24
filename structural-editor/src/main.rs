use anyhow::Result;
use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use structural_editor::op::OpBatch;

/// JSONL stdin → JSONL stdout structural editor.
///
/// Input line:
///   { "root": "/abs/path/to/project", "label": "optional", "ops": [...] }
///
/// Output line:
///   { "ok": true/false, "label": "...", "deltas": [...], "error": null }
fn main() -> Result<()> {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = io::BufWriter::new(stdout.lock());

    for line in stdin.lock().lines() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with("//") {
            continue;
        }

        let response = match parse_and_run(trimmed) {
            Ok(r) => r,
            Err(e) => serde_json::json!({ "ok": false, "error": e.to_string() }),
        };

        serde_json::to_writer(&mut out, &response)?;
        out.write_all(b"\n")?;
        out.flush()?;
    }

    Ok(())
}

fn parse_and_run(line: &str) -> Result<serde_json::Value> {
    #[derive(serde::Deserialize)]
    struct Request {
        root: PathBuf,
        #[serde(flatten)]
        batch: OpBatch,
    }

    let req: Request = serde_json::from_str(line)?;
    let result = structural_editor::executor::apply(&req.batch, &req.root);
    Ok(serde_json::to_value(result)?)
}
