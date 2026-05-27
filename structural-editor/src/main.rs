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
    run(stdin.lock(), &mut out)
}

fn run<R, W>(input: R, out: &mut W) -> Result<()>
where
    R: BufRead,
    W: Write,
{
    for line in input.lines() {
        process_line(&line?, out)?;
    }
    Ok(())
}

fn process_line<W: Write>(line: &str, out: &mut W) -> Result<()> {
    let trimmed = line.trim();
    if should_skip(trimmed) {
        return Ok(());
    }

    let response = response_for(trimmed);
    write_response(out, &response)
}

fn should_skip(line: &str) -> bool {
    line.is_empty() || line.starts_with("//")
}

fn response_for(line: &str) -> serde_json::Value {
    match parse_and_run(line) {
        Ok(response) => response,
        Err(error) => error_response(error),
    }
}

fn write_response<W: Write>(out: &mut W, response: &serde_json::Value) -> Result<()> {
    serde_json::to_writer(&mut *out, response)?;
    out.write_all(b"\n")?;
    out.flush()?;
    Ok(())
}

fn error_response(error: impl ToString) -> serde_json::Value {
    serde_json::json!({ "ok": false, "error": error.to_string() })
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
