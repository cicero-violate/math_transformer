use anyhow::{bail, Result};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::op::{Verify, VerifyPredicate};

use super::text::{buffered_exists, read_buffered_text};

pub fn apply(op: &Verify, root: &Path, buffers: &HashMap<PathBuf, Option<Vec<u8>>>) -> Result<()> {
    let ok = check(&op.predicate, root, buffers);
    if !ok {
        let msg = op.message.as_deref().unwrap_or("verify predicate failed");
        bail!("{msg}: {:?}", op.predicate);
    }
    Ok(())
}

fn check(pred: &VerifyPredicate, root: &Path, buffers: &HashMap<PathBuf, Option<Vec<u8>>>) -> bool {
    match pred {
        VerifyPredicate::FileExists { path } => {
            let abs = root.join(path);
            buffered_exists(&abs, buffers)
        }
        VerifyPredicate::FileAbsent { path } => {
            let abs = root.join(path);
            !buffered_exists(&abs, buffers)
        }
        VerifyPredicate::ContainsText { path, text } => {
            let abs = root.join(path);
            read_buffered_text(&abs, buffers)
                .map(|buffered| buffered.contains(text.as_str()))
                .unwrap_or(false)
        }
        VerifyPredicate::TextAbsent { path, text } => {
            let abs = root.join(path);
            read_buffered_text(&abs, buffers)
                .map(|buffered| !buffered.contains(text.as_str()))
                .unwrap_or(true)
        }
        VerifyPredicate::SymbolExists { path, selector } => {
            let abs = root.join(path);
            read_buffered_text(&abs, buffers)
                .map(|buffered| buffered.contains(selector.split("::").last().unwrap_or(selector)))
                .unwrap_or(false)
        }
    }
}
