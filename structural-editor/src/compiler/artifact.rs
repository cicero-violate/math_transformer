use anyhow::Result;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::op::{Receipt, Rollback};

pub fn apply_receipt(
    op: &Receipt,
    root: &Path,
    buffers: &mut HashMap<PathBuf, Option<Vec<u8>>>,
) -> Result<()> {
    let abs = root.join(&op.receipt_path);
    let content = format!(
        "{{\"kind\":\"receipt\",\"summary\":{:?},\"rollback_required\":{}}}\n",
        op.summary, op.rollback_required
    );
    buffers.insert(abs, Some(content.into_bytes()));
    Ok(())
}

pub fn apply_rollback(
    op: &Rollback,
    root: &Path,
    buffers: &mut HashMap<PathBuf, Option<Vec<u8>>>,
) -> Result<()> {
    let abs = root.join(&op.rollback_path);
    let content = format!("{{\"kind\":\"rollback\",\"manifest\":{:?}}}\n", op.manifest);
    buffers.insert(abs, Some(content.into_bytes()));
    Ok(())
}
